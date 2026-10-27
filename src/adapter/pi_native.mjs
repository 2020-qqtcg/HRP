// Run Pi's actual coding-agent session in the benchmark container. This adapter
// observes the native loop; it does not implement a replacement agent loop.
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";

const config = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
if (!["run", "sample"].includes(config.operation ?? "run")) throw new Error("operation must be run or sample");
const root = path.resolve(config.harness_root);
const output = path.resolve(config.output_dir);
fs.mkdirSync(path.join(output, "states"), { recursive: true });
const secret = /^(authorization|api_?key|access_?token|refresh_?token|password|client_?secret)$/i;
function serializable(value) {
	return JSON.parse(JSON.stringify(value, (key, item) => {
		if (secret.test(key)) return "[REDACTED]";
		if (typeof item === "function" || item instanceof AbortSignal) return undefined;
		if (item instanceof Error) return { name: item.name, message: item.message };
		if (item instanceof Map) return Object.fromEntries(item);
		if (item instanceof Set) return [...item];
		return item;
	}));
}
function canonical(value) {
	if (Array.isArray(value)) return value.map(canonical);
	if (value && typeof value === "object") {
		return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
	}
	return value;
}
function digest(value) {
	return createHash("sha256").update(JSON.stringify(canonical(value))).digest("hex");
}
function writeJson(file, value) {
	fs.writeFileSync(file, JSON.stringify(serializable(value), null, 2) + "\n");
}
function event(type, value = {}) {
	fs.appendFileSync(path.join(output, "events.jsonl"), JSON.stringify(serializable({
		type, timestamp: new Date().toISOString(), ...value,
	})) + "\n");
}
function treeFingerprint() {
	const hash = createHash("sha256");
	function visit(dir) {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
			const file = path.join(dir, entry.name);
			if (entry.isDirectory()) visit(file);
			else if (entry.isFile()) hash.update(path.relative(root, file)).update("\0").update(fs.readFileSync(file));
		}
	}
	for (const name of ["coding-agent", "agent", "ai", "tui", "telemetry", "protocol", "client"]) {
		visit(path.join(root, "packages", name, "src"));
		hash.update(fs.readFileSync(path.join(root, "packages", name, "package.json")));
	}
	for (const file of ["package-lock.json", "tsconfig.json", "tsconfig.base.json"]) hash.update(fs.readFileSync(path.join(root, file)));
	return hash.digest("hex");
}

// Use Pi's checked-in TS sources and workspace resolution consistently. Some
// snapshots ship only partial dist trees; silently mixing those is unsafe.
const { register } = await import(pathToFileURL(path.join(root, "node_modules/tsx/dist/esm/api/index.mjs")).href);
register({ tsconfig: path.join(root, "tsconfig.json") });
const {
	createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager,
} = await import(pathToFileURL(path.join(root, "packages/coding-agent/src/index.ts")).href);
const version = JSON.parse(fs.readFileSync(path.join(root, "packages/coding-agent/package.json"))).version;
const sourceFingerprint = treeFingerprint();
const adapterFingerprint = createHash("sha256").update(fs.readFileSync(new URL(import.meta.url))).digest("hex");
const started = Date.now();
// Null is unbounded; JSON serialization writes Infinity back as null in logs.
const budget = Object.fromEntries(Object.entries(config.budget).map(([key, value]) => [key, value ?? Infinity]));
const counters = { top_level_steps: 0, logical_model_calls: 0, model_calls: 0, output_tokens: 0, input_tokens: 0, cache_read_tokens: 0, cache_write_tokens: 0, usage_missing_calls: 0 };
let seed = Number(config.seed ?? 42) >>> 0;
let stopReason = null;
let summaryPhase = null;
let summaryRetry = false;
let session;
const timeAbort = new AbortController();
const timer = Number.isFinite(budget.wall_time_sec) ? setTimeout(() => {
	stopReason = "wall_time_sec";
	event("budget_exhausted", { reason: stopReason, counters });
	timeAbort.abort();
	void session?.abort();
}, budget.wall_time_sec * 1000) : undefined;
timer?.unref();
const restoreScope = {
	next_decision: true,
	full_controller_continuation: false,
	environment: "owned_by_benchmark_gateway",
	limitations: [
		"Saved native request can be replayed without tools or prefix regeneration.",
		"Runtime promises, retry timers, abort controllers, and live processes are not restorable.",
	],
};
const settings = SettingsManager.inMemory();
const agentDir = path.join(output, "agent-config");
fs.mkdirSync(agentDir, { recursive: true });
const modelRuntime = await ModelRuntime.create({
	modelsPath: null, authPath: path.join(agentDir, "auth.json"), refreshOnCreate: false,
});
const isOpenRouter = /openrouter/i.test(config.model.api_base) || config.model.backend === "openrouter";
modelRuntime.registerProvider("hrp-native", {
	api: "openai-completions", baseUrl: config.model.api_base, apiKey: config.model.api_key ?? "EMPTY",
	models: [{
		id: config.model.id, name: config.model.id, reasoning: true, input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: config.model.context_window, maxTokens: config.model.max_tokens,
		compat: { supportsDeveloperRole: false, supportsReasoningEffort: isOpenRouter, maxTokensField: "max_tokens" },
	}],
});
const model = modelRuntime.getModel("hrp-native", config.model.id);
const environment = Object.fromEntries(Object.entries(process.env).map(([key, value]) => [
	key, /KEY|TOKEN|PASSWORD|SECRET|AUTH|CREDENTIAL/i.test(key) ? "[REDACTED]" : value,
]));

function controllerSnapshot() {
	if (!session) return null;
	return {
		turn_index: session._turnIndex,
		retry_attempt: session._retryAttempt,
		overflow_recovery_attempted: session._overflowRecoveryAttempted,
		compaction_active: Boolean(session._compactionAbortController || session._autoCompactionAbortController),
		branch_summary_active: Boolean(session._branchSummaryAbortController),
		pending_steering: session._steeringMessages,
		pending_follow_up: session._followUpMessages,
		pending_next_turn: session._pendingNextTurnMessages,
		pending_bash_messages: session._pendingBashMessages,
		active_tools: session.getActiveToolNames(), tool_execution: session.agent.toolExecution,
		session_id: session.sessionId, session_leaf_id: session.sessionManager.getLeafId(),
		session_entries: session.sessionManager.getEntries(),
		agent_state: session.agent.state,
	};
}
function exhausted(reason) {
	stopReason ??= reason;
	event("budget_exhausted", { reason, counters });
	timeAbort.abort();
	session?.agent.abort();
	throw new Error(`HRP budget exhausted: ${reason}`);
}
function resultKind(message) {
	if (message.stopReason === "error" || message.stopReason === "aborted" || message.stopReason === "length") {
		return message.stopReason;
	}
	if (message.content.some((block) => block.type === "toolCall")) return "tool_calls";
	return message.content.some((block) => block.type === "text" && block.text.trim()) ? "final_text" : "empty_response";
}

const manifest = {
	schema_version: "pi-native-v1", harness: "pi", interface: "native", version,
	source_fingerprint: sourceFingerprint, source_runtime: "tsx_workspace_sources",
	node_version: process.version, tsx_version: JSON.parse(fs.readFileSync(path.join(root, "node_modules/tsx/package.json"))).version,
	tools: ["read", "bash", "edit", "write"], native_session: "createAgentSession",
	subagents: { enabled: false, reason: "Not part of Pi's default coding-agent configuration." },
	permissions: "native tools execute in benchmark task container under its process user",
	network: "controlled by benchmark container and inference gateway",
	resources: "native task-local context and skill discovery; isolated empty global config; optional extension autoload disabled",
	extension_autoload: false,
	compaction: settings.getCompactionSettings(), retry: settings.getRetrySettings(),
	provider_retry: settings.getProviderRetrySettings(), tool_execution: "parallel",
	model: serializable(config.model), budget, seed: config.seed ?? 42,
	thinking_level: "medium",
	adapter_fingerprint: adapterFingerprint,
	thinking_token_budget: null, restore_scope: restoreScope,
	step_definition: "top-level decision calls excluding application retries; summaries and every HTTP attempt counted separately",
	operation: config.operation ?? "run",
};
writeJson(path.join(output, "manifest.json"), manifest);

let previousStateId = null;
async function observedStream(original, requestModel, context, options = {}, sourceState = null) {
	const kind = sourceState?.call_kind ?? (summaryPhase
		? `${summaryPhase}${summaryRetry ? "_retry" : ""}`
		: session?._retryAttempt > 0 || session?._overflowRecoveryAttempted ? "decision_retry" : "decision");
	if (timeAbort.signal.aborted) exhausted(stopReason ?? "wall_time_sec");
	if (counters.model_calls >= budget.max_model_calls) exhausted("max_model_calls");
	if (counters.output_tokens >= budget.max_output_tokens) exhausted("max_output_tokens");
	if (!sourceState && kind === "decision") {
		if (counters.top_level_steps >= budget.max_steps) exhausted("max_steps");
		counters.top_level_steps++;
	}
	const callIndex = ++counters.logical_model_calls;
	seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
	const requestSeed = seed & 0x7fffffff;
	const retry = settings.getProviderRetrySettings();
	const nativeOptions = {
		...options,
		maxTokens: Math.min(options.maxTokens ?? requestModel.maxTokens, config.model.max_tokens, budget.max_output_tokens - counters.output_tokens),
		temperature: config.model.temperature ?? options.temperature,
		maxRetries: options.maxRetries ?? retry.maxRetries,
		maxRetryDelayMs: options.maxRetryDelayMs ?? retry.maxRetryDelayMs,
		timeoutMs: options.timeoutMs ?? retry.timeoutMs ?? settings.getHttpIdleTimeoutMs(),
	};
	const frozen = serializable({
		schema_version: "pi-native-state-v1", harness: "pi", interface: "native", version,
		source_fingerprint: sourceFingerprint, adapter_fingerprint: adapterFingerprint, native_model: requestModel,
		native_context: context, native_options: nativeOptions,
		controller: sourceState?.controller ?? controllerSnapshot(),
		cwd: config.cwd, environment, call_kind: kind, logical_call_index: callIndex,
		trace: { agent_id: session?.sessionId ?? "root", agent_role: "root", parent_agent_id: null,
			previous_state_id: previousStateId },
		seed: requestSeed, remaining_budget: {
			steps: budget.max_steps - counters.top_level_steps,
			model_calls: budget.max_model_calls - counters.model_calls,
			output_tokens: budget.max_output_tokens - counters.output_tokens,
			wall_time_sec: Math.max(0, budget.wall_time_sec - (Date.now() - started) / 1000),
		}, restore_scope: restoreScope,
	});
	let stateId = null;
	let callAttempt = 0;
	const originalPayload = options.onPayload;
	const nativeFetch = options.fetch ?? globalThis.fetch;
	const stream = await original(requestModel, context, {
		...nativeOptions,
		apiKey: config.model.api_key ?? "EMPTY",
		signal: options.signal ? AbortSignal.any([options.signal, timeAbort.signal]) : timeAbort.signal,
			onPayload: async (payload, currentModel) => {
			const transformed = await originalPayload?.(payload, currentModel);
			// Replay the recorded final payload as well as the native context. This
			// preserves any native provider transformations without rerunning hooks.
			const finalPayload = {
				...(sourceState ? structuredClone(sourceState.provider_payload) : transformed ?? payload),
				model: requestModel.id, seed: requestSeed,
			};
			if (sourceState) {
				const limitKey = "max_completion_tokens" in finalPayload ? "max_completion_tokens" : "max_tokens";
				finalPayload[limitKey] = Math.min(finalPayload[limitKey] ?? nativeOptions.maxTokens, nativeOptions.maxTokens);
				if (nativeOptions.temperature !== undefined) finalPayload.temperature = nativeOptions.temperature;
			}
			if (isOpenRouter) {
				finalPayload.reasoning = { effort: nativeOptions.reasoning ?? "medium", exclude: true };
				delete finalPayload.reasoning_effort;
			}
			if ("thinking_token_budget" in finalPayload) throw new Error("Unexpected thinking_token_budget in Pi native request");
			frozen.provider_payload = serializable(finalPayload);
			stateId = sourceState?.state_id ?? digest(frozen);
			frozen.state_id = stateId;
			previousStateId = stateId;
			if (!sourceState) writeJson(path.join(output, "states", `${stateId}.json`), frozen);
			event("model_call_start", { state_id: stateId, call_kind: kind, logical_call_index: callIndex, seed: requestSeed });
			return finalPayload;
		},
		fetch: async (input, init) => {
			if (!stateId) throw new Error("Pi provider bypassed onPayload; refusing an unrecorded request");
			if (counters.model_calls >= budget.max_model_calls) exhausted("max_model_calls");
			callAttempt++;
			counters.model_calls++;
			const headers = new Headers(input instanceof Request ? input.headers : undefined);
			for (const [key, value] of new Headers(init?.headers)) headers.set(key, value);
			headers.set("X-HRP-State-ID", stateId);
			headers.set("X-HRP-Call-Kind", callAttempt > 1 ? `${kind}_http_retry` : kind);
			const metadata = { state_id: stateId, call_kind: kind, logical_call_index: callIndex, attempt: callAttempt, model_call_index: counters.model_calls };
			event("http_attempt_start", metadata);
			try {
				const response = await nativeFetch(input, { ...init, headers });
				event("http_attempt_response", { ...metadata, status: response.status });
				return response;
			} catch (error) {
				event("http_attempt_error", { ...metadata, error });
				throw error;
			}
		},
	});
	stream.result().then((message) => {
		counters.input_tokens += (message.usage?.input ?? 0) + (message.usage?.cacheRead ?? 0) + (message.usage?.cacheWrite ?? 0);
		counters.cache_read_tokens += message.usage?.cacheRead ?? 0;
		counters.cache_write_tokens += message.usage?.cacheWrite ?? 0;
		counters.output_tokens += message.usage?.output ?? 0;
		const missingUsage = callAttempt > 0 && ["error", "aborted"].includes(message.stopReason) && !message.usage?.totalTokens;
		if (missingUsage) counters.usage_missing_calls++;
		event("model_call_end", { state_id: stateId, call_kind: kind, logical_call_index: callIndex, outcome: resultKind(message), usage_status: missingUsage ? "missing" : "recorded", message });
	}, (error) => event("model_call_error", { state_id: stateId, call_kind: kind, logical_call_index: callIndex, error }));
	return stream;
}

try {
	let message;
	if (config.operation === "sample") {
		const frozen = JSON.parse(fs.readFileSync(config.state_path, "utf8"));
		if (frozen.harness !== "pi" || frozen.interface !== "native" || !frozen.restore_scope?.next_decision) {
			throw new Error("State is not a replayable Pi native decision");
		}
		if (frozen.source_fingerprint !== sourceFingerprint) throw new Error("Pi source fingerprint differs from frozen state");
		if (frozen.adapter_fingerprint !== adapterFingerprint) throw new Error("Pi adapter fingerprint differs from frozen state");
		const { state_id: expectedStateId, ...stateContent } = frozen;
		if (digest(stateContent) !== expectedStateId) throw new Error("Frozen Pi state checksum mismatch");
		const stream = await observedStream(
			modelRuntime.streamSimple.bind(modelRuntime), model,
			structuredClone(frozen.native_context), structuredClone(frozen.native_options), frozen,
		);
		message = await stream.result();
	} else {
		const resourceLoader = new DefaultResourceLoader({ cwd: config.cwd, agentDir, settingsManager: settings, noExtensions: true });
		await resourceLoader.reload();
		({ session } = await createAgentSession({
			cwd: config.cwd, agentDir, modelRuntime, model, thinkingLevel: "medium", resourceLoader,
			settingsManager: settings,
			sessionManager: SessionManager.create(config.cwd, path.join(output, "sessions")),
		}));
		manifest.system_prompt = session.systemPrompt;
		manifest.active_tools = session.getActiveToolNames();
		manifest.project_context_files = resourceLoader.getAgentsFiles();
		manifest.loaded_extensions = resourceLoader.getExtensions().extensions.map((entry) => entry.path);
		manifest.loaded_skills = resourceLoader.getSkills().skills.map((entry) => ({ name: entry.name, file_path: entry.filePath }));
		writeJson(path.join(output, "manifest.json"), manifest);
		session.subscribe((nativeEvent) => {
			if (nativeEvent.type === "compaction_start") summaryPhase = "compaction";
			if (nativeEvent.type === "compaction_end") { summaryPhase = null; summaryRetry = false; }
			if (nativeEvent.type === "summarization_retry_attempt_start") summaryRetry = true;
			if (nativeEvent.type === "summarization_retry_finished") summaryRetry = false;
			// Finalized messages/results retain complete text, reasoning and calls.
			if (!["message_update", "tool_execution_update"].includes(nativeEvent.type)) event("native_event", { event: nativeEvent });
		});
		const original = session.agent.streamFunction;
		session.agent.streamFunction = (requestModel, context, options) => observedStream(original, requestModel, context, options);
		const originalStop = session.agent.shouldStopAfterTurn;
		session.agent.shouldStopAfterTurn = async (turn, signal) => {
			if (await originalStop?.(turn, signal)) return true;
			const hasCalls = turn.message.content.some((block) => block.type === "toolCall");
			if (hasCalls && counters.top_level_steps >= budget.max_steps) {
				stopReason = "max_steps";
				event("budget_exhausted", { reason: stopReason, counters });
				return true;
			}
			return false;
		};
		await session.prompt(config.instruction, { expandPromptTemplates: false });
		message = [...session.messages].reverse().find((entry) => entry.role === "assistant");
	}
	writeJson(path.join(output, "result.json"), {
		harness: "pi", interface: "native", operation: config.operation ?? "run",
		status: stopReason ? "budget_exhausted" : message?.stopReason === "error" ? "infrastructure_error" : "completed",
		stop_reason: stopReason ?? (message ? resultKind(message) : "no_assistant_message"),
		message, counters, elapsed_sec: (Date.now() - started) / 1000,
		source_state_id: config.operation === "sample" ? JSON.parse(fs.readFileSync(config.state_path, "utf8")).state_id : undefined,
		restore_scope: restoreScope,
	});
} catch (error) {
	event("adapter_error", { error, counters });
	writeJson(path.join(output, "result.json"), {
		harness: "pi", interface: "native", status: stopReason ? "budget_exhausted" : "infrastructure_error",
		stop_reason: stopReason, error, counters, elapsed_sec: (Date.now() - started) / 1000,
	});
	if (!stopReason) process.exitCode = 1;
} finally {
	clearTimeout(timer);
	session?.dispose();
}
