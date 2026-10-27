#!/usr/bin/env node
// Container-side DSH runner. This composes the official unattended coding
// plugins; the framework owns the loop, tools, delegation, and compaction.
// Usage: node deepseek_native.mjs config.json  (or JSON on stdin with "-").
import fs from "node:fs";
import path from "node:path";
import { AsyncLocalStorage } from "node:async_hooks";
import { createHash, randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";

const config = JSON.parse(fs.readFileSync(process.argv[2] && process.argv[2] !== "-" ? process.argv[2] : 0, "utf8"));
if (!["run", "sample"].includes(config.operation)) throw new Error("operation must be run or sample");
for (const field of ["cwd", "output_dir", "harness_root"]) {
	if (typeof config[field] !== "string") throw new Error(`${field} is required`);
	config[field] = path.resolve(config[field]);
}
if (!config.model?.id || !config.model.api_base || !config.model.api_key) throw new Error("model id/api_base/api_key are required");
if (config.operation === "sample" && !config.state_path) throw new Error("sample requires state_path");
if (config.state_path) config.state_path = path.resolve(config.state_path);
if (config.operation === "run" && typeof config.instruction !== "string") throw new Error("run requires instruction");
const output = config.output_dir;
fs.mkdirSync(output, { recursive: true });
for (const filename of ["events.jsonl", "result.json", "manifest.json"]) {
	if (fs.existsSync(path.join(output, filename))) throw new Error(`refusing to reuse output: ${output}`);
}
fs.mkdirSync(path.join(output, "states"));
fs.mkdirSync(path.join(output, "calls"));
process.chdir(config.cwd);
process.env.HRP_NATIVE_API_KEY = config.model.api_key;
const route = "hrp-native";
const started = Date.now();
const budget = {
	max_steps: null,
	max_model_calls: null,
	max_output_tokens: null,
	wall_time_sec: null,
	...config.budget,
};
for (const [key, value] of Object.entries(budget)) {
	if (value === null) { budget[key] = Infinity; continue; }
	if (!Number.isFinite(value) || value <= 0 || (key !== "wall_time_sec" && !Number.isInteger(value))) throw new Error(`invalid budget ${key}`);
}
const counts = { top_level_steps: 0, model_calls: 0, http_attempts: 0, input_tokens: 0, output_tokens: 0, usage_missing_calls: 0, unreported_output_token_cap: 0, reserved_output_tokens: 0, by_kind: {} };
const callContext = new AsyncLocalStorage();
let sequence = 0;
let stoppedBy = null;
let rootAgent;
let timer;
let activeContext;
const errors = [];
const replayDegradations = [];
const decisions = [];
const jsonCopy = value => JSON.parse(JSON.stringify(value));
const digest = value => createHash("sha256").update(typeof value === "string" ? value : JSON.stringify(value)).digest("hex");
const writeJson = (filename, value) => fs.writeFileSync(path.join(output, filename), JSON.stringify(value, null, 2) + "\n");
function record(type, data = {}) {
	fs.appendFileSync(path.join(output, "events.jsonl"), JSON.stringify({ seq: sequence++, timestamp: new Date().toISOString(), type, ...data }) + "\n");
}
function stop(reason) {
	stoppedBy ??= reason;
	record("budget_stop", { reason });
	rootAgent?.cancel({ kind: "hook", reason });
}

// The provider constructs its own request. Add the local sampling seed and
// headers connecting transport attempts to the exact boundary, including children.
const originalFetch = globalThis.fetch;
globalThis.fetch = async function observedFetch(input, init) {
	const current = callContext.getStore();
	if (!current) return originalFetch(input, init);
	const headers = new Headers(input instanceof Request ? input.headers : undefined);
	new Headers(init?.headers).forEach((value, key) => headers.set(key, value));
	headers.set("X-HRP-State-ID", current.state_id);
	headers.set("X-HRP-Call-Kind", current.kind);
	counts.http_attempts++;
	let payload;
	let body = init?.body;
	if (typeof body === "string") {
		try { payload = JSON.parse(body); } catch { /* Non-JSON provider transport is recorded without decoding. */ }
	}
	if (payload && config.model.backend !== "openrouter") {
		payload.seed = current.seed;
		body = JSON.stringify(payload);
	}
	record("http_request", { state_id: current.state_id, call_kind: current.kind, attempt: counts.http_attempts, ...(payload === undefined ? {} : { payload }) });
	return originalFetch(input, { ...init, ...(body === undefined ? {} : { body }), headers });
};

const packageDirs = {
	cordis: "vendor/cordis",
	llm: "packages/llm/llm",
	provider: "packages/llm/llm-pi-ai",
	spine: "packages/examples/agent-spine-demo",
	session: "packages/core/session",
	tools: "packages/core/tools",
	subprocess: "packages/subprocess/subprocess-local",
	bash: "packages/shell/bash-local",
	fs: "packages/fs/fs-local",
	fsPolicy: "packages/fs/fs-observation-policy",
	fsTools: "packages/fs/tool-fs",
	subagent: "packages/subagent/subagent",
	spawn: "packages/subagent/subagent-spawn-in-process",
	delegate: "packages/subagent/tool-subagent",
	todo: "packages/todo/tool-todo",
	meter: "packages/llm/token-meter",
	compaction: "packages/compaction/compaction-basic",
	persistence: "packages/session/session-persistence-jsonl",
	checkpoint: "packages/session/session-checkpoint-policy",
};
const packages = Object.fromEntries(await Promise.all(Object.entries(packageDirs).map(async ([name, relative]) => [
	name, await import(pathToFileURL(path.join(config.harness_root, relative, "lib/index.js")).href),
])));
const { Context, Logger } = packages.cordis;
const { SessionId } = packages.session;
const { LlmError, createUserMessage } = packages.llm;
const safeModel = { ...config.model };
delete safeModel.api_key;
const providerConfig = {
	api: "openai-completions",
	apiKeyEnv: "HRP_NATIVE_API_KEY",
	baseURL: config.model.api_base.replace(/\/$/, ""),
	displayName: "HRP benchmark endpoint",
	models: [{
		id: config.model.id,
		contextWindow: config.model.context_window ?? 65536,
		maxTokens: config.model.max_tokens ?? 32768,
		...(config.model.backend === "openrouter" ? { reasoningEfforts: { off: null, [config.model.reasoning_effort ?? "medium"]: config.model.reasoning_effort ?? "medium" } } : {}),
	}],
	compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens" },
	timeoutMs: Math.min(3600, budget.wall_time_sec) * 1000,
	retryPolicy: { mode: "normal", maxRetries: 10 },
};
const manifest = {
	schema: "hrp-native-dsh-v1",
	harness: "deepseek",
	interface: "native",
	profile: "official-jsonrpc-foreground-coding",
	profile_source: "examples/jsonrpc-agent/cordis.yml",
	version: JSON.parse(fs.readFileSync(path.join(config.harness_root, "package.json"), "utf8")).version,
	operation: config.operation,
	model: safeModel,
	provider: providerConfig,
	budget,
	seed: config.seed ?? 42,
	seed_policy: "Local provider transport seed is (seed + call_index) modulo 2^31; OpenRouter receives no added seed. Exact wire payloads are recorded.",
	tools: ["bash", "read", "write", "edit", "subagent", "todo_write"],
	plugins: packageDirs,
	composition: {
		persona: "You are a coding agent.",
		workspace_context: false,
		skills: false,
		bash: { cwd: config.cwd, timeoutMs: 60000, enableRunInBackground: false },
		subagent: { provider: "spawn", backgroundMode: "one-shot", enableRunInBackground: false, maxDepth: 3 },
		todo: { allowParallelInProgress: true },
		compaction: { thresholdRatio: 0.8, retainRatio: 0.16, maxTokens: 8192, compactionRetries: 1 },
		filesystem: "fs-local + fs-observation-policy in the task container",
		permission: "No additional permission plugin, matching the selected unattended profile; the task container is the execution boundary.",
		network: "Inherited task-container network; model endpoint supplied by benchmark controller.",
	},
	profile_changes: [
		"Programmatic composition of the official JSON-RPC coding plugins; CLI JSON replaces the SDK transport only.",
		"Use the official llm-pi-ai adapter for the configured OpenAI-compatible endpoint instead of llm-deepseek.",
		"Add observation and explicit budget plugins; keep native tool schemas, feedback, loop, retries, and termination.",
		"Persist uncompressed JSONL for auditability.",
		"Provider retry maxRetries=10 matches the existing analysis configuration.",
		"Attach local sampling seeds and observation headers at HTTP dispatch; OpenRouter receives no added seed.",
	],
	disabled_in_selected_official_profile: ["background bash", "background delegation", "job tools", "skills", "workspace context", "interactive approval UI", "ask-user", "plan-mode", "web tools", "fork delegation", "workflow", "read_image (no attachment service)"],
	restore_contract: {
		next_decision: "Replay captured GenerateOptions directly through the same native LLM adapter; no prefix, compaction, or tools are rerun.",
		full_controller_restore: false,
		environment: config.environment_snapshot ?? { available: false, reason: "Container/process snapshots belong to the benchmark controller; this runner records their supplied identity only." },
		cross_model_private_replay: "Preserve native history; record provider replay degradation and mark cross-model private state as unverified.",
	},
	budget_semantics: "Top-level steps use native step/start; retries, summaries, and child requests consume the shared model-call budget. Output budget reserves each admitted request's complete output cap, without changing its native cap. Missing usage is recorded as unknown and charged at its reserved output cap for admission; token totals contain observed usage only.",
	thinking_token_budget: null,
};
manifest.config_fingerprint = digest(manifest);
writeJson("manifest.json", manifest);

function environment() {
	const keys = ["PATH", "LANG", "LC_ALL", "TERM", "NO_COLOR", "PAGER", "GIT_PAGER", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "TZ"];
	return Object.fromEntries(keys.filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
}
function outcome(chunks, error, schemas) {
	const finish = chunks.findLast(chunk => chunk.type === "finish");
	const content = chunks.filter(chunk => chunk.type === "block-end").map(chunk => chunk.block);
	const calls = content.filter(block => block.type === "tool-call");
	const validationErrors = calls.flatMap(block => {
		let args;
		try { args = JSON.parse(block.arguments); } catch { return [{ call_id: block.id, error: "invalid_json" }]; }
		const schema = schemas?.find(definition => definition.name === block.name);
		if (!schema) return [{ call_id: block.id, error: "unknown_tool" }];
		return packages.tools.validateJsonSchemaValue(schema.parameters, args).map(error => ({ call_id: block.id, error }));
	});
	const invalid = validationErrors.length > 0;
	const visible = content.filter(block => block.type === "text").map(block => block.text).join("").trim();
	return {
		content, finish_reason: finish?.reason ?? null, validation_errors: validationErrors,
		classification: finish?.reason?.failure?.code === "EMPTY_RESPONSE" ? "empty_reply"
			: error || finish?.reason?.kind === "error" ? "provider_error"
			: finish?.reason?.kind === "max-tokens" ? "length_truncated"
				: invalid ? "invalid_call" : calls.length ? "tool_calls" : visible ? "normal_end" : "empty_reply",
		flags: { empty_public_reply: !calls.length && !visible, invalid_arguments: invalid, length_truncated: finish?.reason?.kind === "max-tokens" },
	};
}

function observe(ctx) {
	const stepAttempts = new Map();
	const previousByAgent = new Map();
	const parentStateByAgent = new Map();
	ctx.logger.exporter({
		levels: { default: 3 },
		export(message) {
			const text = Logger.format({}, message);
			const state_id = callContext.getStore()?.state_id ?? null;
			record("native_log", { level: message.type, logger: message.name, message: text, state_id });
			if (text.includes("unusable replay state")) {
				const detail = { state_id, reason: text };
				replayDegradations.push(detail);
				record("replay_degradation", detail);
			}
		},
	});
	ctx.on("session/event", (session, event) => {
		record("session_event", { session_id: String(session.id), event });
		if (session === rootAgent?.session && event.type === "step/start") counts.top_level_steps++;
	}, { global: true });
	ctx.on("llm/stream", (request, next) => (async function* () {
		const session = request.sessionId === undefined ? undefined : ctx.sessions?.get(request.sessionId);
		const agent = ctx.agents?.currentInitiator();
		const latestStep = session?.events.findLast(event => event.type === "step/start");
		const stepKey = session && latestStep ? `${session.id}:${latestStep.seq}` : null;
		const attempt = request.purpose || stepKey === null ? 0 : (stepAttempts.get(stepKey) ?? 0);
		if (!request.purpose && stepKey !== null) stepAttempts.set(stepKey, attempt + 1);
		const kind = request.purpose ?? (attempt > 0 ? "retry" : session?.header.parentSession ? "subagent" : "decision");
		const cap = request.maxTokens ?? config.model.max_tokens ?? 32768;
		const reason = stoppedBy ?? (counts.model_calls >= budget.max_model_calls ? "max_model_calls"
			: counts.output_tokens + counts.unreported_output_token_cap + counts.reserved_output_tokens + cap > budget.max_output_tokens ? "max_output_tokens"
				: null);
		if (reason) {
			stop(reason);
			throw new LlmError(`HRP budget exhausted: ${reason}`, "HRP_BUDGET_EXHAUSTED");
		}
		const index = counts.model_calls++;
		counts.reserved_output_tokens += cap;
		counts.by_kind[kind] = (counts.by_kind[kind] ?? 0) + 1;
		const { signal: _signal, ...serializableRequest } = request;
		const frozen = jsonCopy(serializableRequest);
		const state_id = `${String(index).padStart(6, "0")}-${digest(frozen).slice(0, 16)}`;
		const agentId = String(session?.id ?? agent?.id ?? "root");
		const parentAgentId = session?.header.parentSession ? String(session.header.parentSession) : null;
		if (parentAgentId && !parentStateByAgent.has(agentId)) parentStateByAgent.set(agentId, previousByAgent.get(parentAgentId) ?? null);
		const trace = { agent_id: agentId, parent_agent_id: parentAgentId,
			agent_role: parentAgentId ? "subagent" : "root", parent_state_id: parentStateByAgent.get(agentId) ?? null,
			previous_state_id: previousByAgent.get(agentId) ?? null, native_step_key: stepKey };
		previousByAgent.set(agentId, state_id);
		const transportSeed = ((Number(config.seed ?? 42) + index) >>> 0) & 0x7fffffff;
		const state = {
			schema: "hrp-native-dsh-state-v1", state_id, config_fingerprint: manifest.config_fingerprint,
			call_index: index, call_kind: kind, request: frozen, request_hash: digest(frozen),
			trace,
			attempt_within_native_step: attempt, transport: { seed: config.model.backend === "openrouter" ? null : transportSeed },
			cwd: session?.header.cwd ?? config.cwd, relevant_environment: environment(),
			environment_snapshot: config.environment_snapshot ?? { available: false },
			session: session ? { header: session.header, events: session.events, request_header: session.requestHeader() } : null,
			controller: agent ? {
				agent_id: String(agent.id), options: agent.options, status: agent.status,
				inbox: { next_step: agent.inbox.nextStep, next_turn: agent.inbox.nextTurn },
				last_step: session?.events.findLast(event => event.type === "step/start")?.data ?? null,
				full_controller_restore: false,
			} : null,
			remaining_budget: { top_level_steps: budget.max_steps - counts.top_level_steps, model_calls: budget.max_model_calls - index, output_tokens: budget.max_output_tokens - counts.output_tokens - counts.unreported_output_token_cap, wall_time_sec: budget.wall_time_sec - (Date.now() - started) / 1000 },
			next_decision_replay: true,
		};
		writeJson(`states/${state_id}.json`, state);
		record("model_call_start", { state_id, call_kind: kind, call_index: index, state_path: `states/${state_id}.json` });
		const chunks = [];
		let failure;
		const scope = { state_id, kind, seed: transportSeed };
		const iterator = callContext.run(scope, () => next()[Symbol.asyncIterator]());
		try {
			while (true) {
				const item = await callContext.run(scope, () => iterator.next());
				if (item.done) break;
				chunks.push(jsonCopy(item.value));
				yield item.value;
			}
		} catch (error) {
			failure = { name: error.name, message: error.message, code: error.code ?? null };
			throw error;
		} finally {
			await callContext.run(scope, () => iterator.return?.());
			counts.reserved_output_tokens -= cap;
			const usage = chunks.findLast(chunk => chunk.type === "usage")?.usage ?? {};
			if (usage.outputTokens === undefined) {
				counts.usage_missing_calls++;
				counts.unreported_output_token_cap += cap;
			}
			counts.input_tokens += usage.inputTokens ?? 0;
			counts.output_tokens += usage.outputTokens ?? 0;
			const decision = { state_id, call_kind: kind, call_index: index, ...outcome(chunks, failure, request.tools), usage, error: failure ?? null };
			decisions.push(decision);
			writeJson(`calls/${state_id}.json`, { ...decision, chunks });
			record("model_call_end", { ...decision, response_path: `calls/${state_id}.json` });
		}
	})(), { global: true, prepend: true });
}

async function waitFor(ctx, service) {
	const until = Date.now() + 15000;
	while (ctx.get(service) === undefined) {
		if (Date.now() >= until) throw new Error(`native plugin did not start: ${service}`);
		await new Promise(resolve => setTimeout(resolve, 5));
	}
}
async function mount(ctx, name, settings = {}) {
	const plugin = packages[name];
	await ctx.plugin(plugin.default ?? plugin, settings);
}

let result;
try {
	const ctx = new Context();
	activeContext = ctx;
	if (config.operation === "sample") {
		await mount(ctx, "llm");
		await mount(ctx, "session");
		observe(ctx);
		await mount(ctx, "provider", { providers: { [route]: providerConfig } });
		const rawState = fs.readFileSync(path.resolve(config.state_path), "utf8");
		const state = JSON.parse(rawState);
		if (state.schema !== "hrp-native-dsh-state-v1" || digest(state.request) !== state.request_hash) throw new Error("invalid or altered DSH state");
		const request = jsonCopy(state.request);
		const crossModel = request.model !== config.model.id;
		const privateHistory = request.messages.some(message => message.source?.replayState !== undefined || message.content?.some(block => block.type === "reasoning"));
		if (crossModel && privateHistory) {
			replayDegradations.push({ state_id: state.state_id, reason: "cross_model_private_replay_unverified" });
			record("replay_comparability", { state_id: state.state_id, cross_model_private_replay: "unverified" });
		}
		request.provider = route;
		request.model = config.model.id;
		request.temperature = config.model.temperature ?? request.temperature;
		if (config.model.backend === "openrouter") {
			request.reasoningEffort = config.model.reasoning_effort ?? "medium";
			delete request.samplingSeed;
		} else {
			request.samplingSeed = config.seed ?? 42;
			delete request.reasoningEffort;
		}
		delete request.sessionId;
		if (Number.isFinite(budget.wall_time_sec)) request.signal = AbortSignal.timeout(Math.round(budget.wall_time_sec * 1000));
		for await (const _chunk of ctx.llm.stream(request)) { /* Observation records every native chunk. */ }
		result = { status: "completed", source_state_id: state.state_id, source_state_hash: digest(rawState), tools_executed: 0, decision: decisions.at(-1), source_unchanged: digest(fs.readFileSync(path.resolve(config.state_path), "utf8")) === digest(rawState) };
	} else {
		await mount(ctx, "spine", {
			persona: "You are a coding agent.", workspaceContext: false, skills: { enabled: false },
			toolBash: { enableRunInBackground: false }, toolJobs: false,
		});
		await waitFor(ctx, "agentLoop");
		observe(ctx);
		await mount(ctx, "provider", { providers: { [route]: providerConfig } });
		await mount(ctx, "subprocess");
		await mount(ctx, "bash", { cwd: config.cwd, timeoutMs: 60000 });
		await mount(ctx, "fs", { cwd: config.cwd });
		await mount(ctx, "fsPolicy");
		await mount(ctx, "fsTools");
		await mount(ctx, "subagent");
		await mount(ctx, "spawn", { providerName: "spawn" });
		await mount(ctx, "delegate", { provider: "spawn", toolName: "subagent", enableRunInBackground: false });
		await mount(ctx, "todo", { allowParallelInProgress: true });
		await mount(ctx, "meter");
		await mount(ctx, "compaction", manifest.composition.compaction);
		await mount(ctx, "persistence", { root: path.join(output, "sessions"), compression: "none" });
		await mount(ctx, "checkpoint");
		await waitFor(ctx, "compaction");
		await waitFor(ctx, "sessionPersistence");
		ctx.on("agent/request", async (_payload, next) => ({
			...await next(), temperature: config.model.temperature,
			...(config.model.backend === "openrouter" ? { reasoningEffort: config.model.reasoning_effort ?? "medium" } : {}),
		}), { global: true });
		ctx.on("agent/pre-step", async ({ agent }, next) => {
			if (agent === rootAgent && counts.top_level_steps >= budget.max_steps) {
				stoppedBy ??= "max_steps";
				record("budget_stop", { reason: stoppedBy });
				return { kind: "reject" };
			}
			return next();
		}, { global: true, prepend: true });
		const handle = await ctx.agents.create({
			sessionId: SessionId(`hrp-native-${randomUUID()}`), meta: { cwd: config.cwd },
			agentOptions: { provider: route, model: config.model.id, maxTokens: config.model.max_tokens ?? 32768 },
		});
		rootAgent = handle.agent;
		if (Number.isFinite(budget.wall_time_sec)) timer = setTimeout(() => stop("wall_time_sec"), Math.max(1, Math.round(budget.wall_time_sec * 1000 - (Date.now() - started))));
		rootAgent.followup(createUserMessage({ content: [{ type: "text", text: config.instruction }], source: { kind: "user" } }));
		await rootAgent.whenIdle();
		const ending = rootAgent.session.events.findLast(event => event.type === "turn/end")?.data.reason ?? null;
		result = { status: stoppedBy ? "budget_exhausted" : ending?.kind === "error" ? "error" : "completed", native_stop_reason: ending, session_id: String(rootAgent.id), decision: decisions.at(-1) ?? null };
	}
} catch (error) {
	errors.push({ name: error.name, message: error.message, stack: error.stack });
	record("runner_error", errors.at(-1));
	result = { status: stoppedBy ? "budget_exhausted" : "error", error: errors.at(-1) };
} finally {
	clearTimeout(timer);
	try { await activeContext?.fiber.dispose(); } catch (error) { errors.push({ name: error.name, message: error.message }); }
	globalThis.fetch = originalFetch;
	if (errors.length && result.status === "completed") result.status = "error";
	result = { ...result, schema: manifest.schema, harness: "deepseek", interface: "native", operation: config.operation, config_fingerprint: manifest.config_fingerprint, counts, stopped_by: stoppedBy, wall_time_sec: (Date.now() - started) / 1000, replay_degradations: replayDegradations, errors, states: decisions.map(decision => ({ state_id: decision.state_id, call_kind: decision.call_kind, classification: decision.classification })) };
	writeJson("result.json", result);
	process.stdout.write(JSON.stringify({ result: path.join(output, "result.json"), status: result.status }) + "\n");
	if (result.status === "error") process.exitCode = 1;
}
