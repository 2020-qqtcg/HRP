# Semantic action mappings

This audit records semantic actions and comparison limits for the paper experiments.

## MiniWoB H0–H5 finite action menus (renumbered 2026-09-15)

Current IDs are H0 direct, H1 program verify, H2 automatic gate, H3 two-click
batch, H4 frozen-LM reviewer, and H5 revised system instructions. Current H4
and H5 preserve the behavior of recorded H5 and H7 respectively. The former
H4 plan executor is retired. Sealed sources, configurations and checkpoint
paths retain their original IDs. Their `harness_numbering.json` records the
current-to-recorded mapping; never infer old semantics from a new numeric ID.

Native semantic requests are `click:<public ID>` and checkbox `submit`.
H0 commits directly. H1/H4 additionally expose `verify:<native request>` for
submission and the five tile clicks, storing advisory feedback without
executing or gating the target. H1 uses the public program; H4 uses an
independent frozen greedy LM and may disagree. A native action clears the
feedback. H2 retains the native menu and gates commit actions automatically,
consuming a decision on rejection and disabling review after its first
rejection. H1/H4 verification is therefore not equivalent to H2 execution.

H3 adds all ordered two-click batches, including repeats, with one actor
decision and no intermediate observation. Native termination stops a batch
immediately. The same primitive click has equivalent DOM effects, while the
full batch differs from two separate actor decisions in budget and feedback.
Identical successors at individual states do not establish interface-wide
equivalence. There is no plan-queue action in the current registry.

H5 changes only the system instructions. H0/H5 preserve semantic actions,
arguments, ordering, user observations, controller states, transition/reward
kernels, costs and budgets. Both the actor and frozen reference read the H5
prompt. Thus equivalent execution does not imply equal reference priors,
regularized optima or ceilings. Compare complete messages as well as kernels
when checking that renaming preserves the experiment.

The complete parameterized menu uses reversible A–Z/a–z tokens and a joint
softmax. Cross-evaluation re-decodes each checkpoint in the deployment menu.
No shared action-column alignment or fixed-state bank is assumed across
non-equivalent interfaces. Each interface has its own frozen reference and DP
optimum. Fixed-bank centered-energy MSE is an interface-local diagnostic;
occupancy-weighted KL mismatch is calculated independently of the return gap.
All groups retain gamma 0.95, tau 0.1, ten actor decisions and zero explicit
costs. Actor decisions, native clicks and reviewer calls are recorded
separately. Equal update or decision counts do not imply equal compute.

New preparations use protocol `h0_h5_v2`; old configs and probability caches
must be replayed with their sealed source. See
[the current harness specification](../../environment/hrp_miniwob/harnesses/miniwob_harnesses.py).

## Native TB2 / SWE-bench Lite coding profiles (2026-09-18)

The new `native` condition is distinct from the historical `shared_bash`
adapters. Pi 0.84.1 uses actual `createAgentSession` and its four coding tools;
DeepAgents 0.7.5 uses the default graph plus official TodoList middleware and a
container-local LocalShellBackend; DSH 0.1.1-rc.2 composes the official foreground
JSON-RPC coding plugins. Profile choices and actual schemas are stored per run.
The files and shell of every native agent operate inside the scorer's task
container. Pi optional extensions are disabled explicitly; DSH's chosen profile
explicitly omits background jobs, skills, workspace context and interactive tools.

Native read/search/edit tools have different range units, truncation, guards,
atomic-edit behavior and feedback text. Matching path names or final file bytes
does not merge them. Complete arguments, visible text, call order and state/file
version remain in local invocation identities. Those identities are not verified
cross-tool semantic classes. General scripts, network and asynchronous behavior
remain unresolved unless deterministic contracts and isolated effect/feedback
checks establish the claimed state-local equivalence. The seven-intent and
operation/object views are explicitly lossy sequences, not equivalence proofs.

Pi/DSH public sessions cannot restore arbitrary in-flight controller state;
DeepAgents graph checkpoints do not serialize live asyncio tasks or subprocess
memory. The supported measurement scope is frozen next-model-request replay
through original provider decoders, without regenerating prefixes or executing
tools. All actual HTTP attempts, including summary/delegation/retry calls, have
their own records and costs. Container filesystem snapshots do not restore
process/network state; concurrent subagent effects and cross-model private
reasoning replay remain explicit limitations. Unavailable states are retained,
and sparse local pairs never receive a fabricated zero error/full-space score.

The retained shared-Bash code still owns its older compression/retry/stop
conventions. New evaluations route it through the common wire cost ledger;
historical top-level step counts are not retrospectively labeled equal compute.
See [the native experiment configuration](../benchmark/config/native_experiment.json).

## TB2.1 / Terminal-Bench Lite and two added harnesses (2026-09-19)

The active datasets are pinned TB2.1 (89 tasks) and Terminal-Bench Lite (39 tasks).
SWE-bench adapters and downloaded upstream images are retired; historical results
and sources remain archived. All five native harnesses use the exact task image
IDs in the preparation inventory and Harbor's official verifier.

Terminus-2 uses Harbor 0.22.0's original loop, JSON parser, tmux state/feedback,
completion confirmation, summaries and summary subcalls. Its ordered keystrokes,
durations and completion flag are native decisions, not equivalent Bash strings:
the active shell or foreground program, prior unfinished input and wait duration
can change behavior. Native parser repairs remain visible. Requests are captured
before the original LLM call, including summary calls; transport retries are
separate ledger entries. Frozen-request sampling does not recover running tmux
or complete controller state, so it provides no full-state semantic guarantee.

mini-SWE-agent 2.4.6 uses official mini.yaml, DefaultAgent, LitellmModel and
LocalEnvironment inside the task container. It retains subshell execution,
multi-call order, output formatting, format-error feedback, retries and the
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT termination convention. The selected
unattended profile omits the CLI confirmation UI, disables the dollar budget,
uses the task's default wall-time budget, and retains unknown monetary cost as unknown
in the wire ledger. DefaultAgent has no delegation or automatic summarizer.
Its Bash schema and termination convention are not silently merged with the
historical shared-Bash contract. New harnesses currently support native only.

All available frozen boundaries stay in HRP output. Unrestorable process state,
unknown tools and unsupported fine-grained action pairs remain explicit;
teacher-relative diagnostics do not imply calibrated accuracy or ranking.

The active TB2.1/Lite protocol imposes a 50 top-level-step cap but no cumulative
model-call or cumulative output-token caps. Task-specific agent and verifier
timeouts are resolved from the pinned task.toml, without a multiplier. Agent
setup has Harbor's separate 360-second deadline. Native termination, request
and tool timeouts remain in effect; total calls and tokens are still recorded.
The common cap is passed through as Terminus max_turns and mini step_limit, and
is checked at Pi, DeepSeek and DeepAgents native decision boundaries. A null cap
remains supported only for explicit uncapped comparisons. Historical runs keep
their recorded settings.
Frozen HRP samples retain their separate per-sample safety budgets. These new
wall-time comparisons are not relabeled as equal-token or equal-step experiments.

## Five-harness frozen-request diagnostics v2 (2026-09-20)

Native HRP now defaults to 40 shared sampler processes. After the two initial
role-specific validation samples pass, admissions interleave in proportion to
target/teacher sample counts (`proportional_interleaved_v1` in the manifest).
This changes scheduling, not native tools or decision decoding. Each sample
keeps its own state, output directory and discovery/estimation assignment;
historical configurations and caches remain separate.

`native-layered-pairs-v2` counts every valid native invocation, including Bash,
Terminus input and delegation, without requiring a filesystem snapshot. This
does not turn an unverified execution into a verified semantic action. Exact
JSON object-key normalization is permitted; arguments, batch order, patches,
terminal waits and completion flags remain distinct. Raw responses are retained.
Complete public decisions include accompanying text; the separate invocation
projection omits accompanying explanation and is explicitly lossy. Summary
and return text cannot be omitted because they constitute those decisions.

The current pipeline emits complete-decision, invocation, intent,
operation/object, control and evidence-gated semantic diagnostics. General
scripts and network/asynchronous operations may have countable invocations
while their execution equivalence remains unverified. No automatic effect
validator or general shell equivalence solver is supplied. Admitting verified
equivalence requires the existing same-state deterministic-contract evidence
check; an identical output or patch alone remains insufficient.

Fresh traces record native predecessors and parent decisions. DeepAgents uses
agent-scoped tool-call IDs plus callback ancestry; DSH uses the selected
foreground profile's session parent and initiating decision. Terminus records
summary maintenance and pending completion. HTTP attempts retain logical-call
IDs and overlapping admitted requests. Unknown historical parent decisions stay
unknown. Child text returns and summaries are not root task termination.
Global arrival order is not a causal sequence and receives no gamma discount.

Frozen branches use independent decoder processes, leave source states intact,
and verify the original context/tools on every generated request. Optional
filesystem snapshots restrict effect validation, not request-only sampling.
Full tmux/process/concurrent-controller restoration and cross-model private
reasoning interpretation remain unproven. Unavailable states, sparse pairs and
invalid outputs retain explicit status and mass. Versioned caches include the
frozen request, native state, wire record and configuration fingerprints.

Supported-pair means are labeled by representation and measured subset, with
coverage and missing-state counts. No complete HRP energy, success prediction
or deterministic ranking is emitted. Historical shared-Bash results retain
their original semantics and cost convention. See
[the native experiment entrypoint](../benchmark/native_experiment.py) for commands and schema.
