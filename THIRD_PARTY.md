# Third-party code

Bundled dependencies retain their original copyright notices and licenses.
HRP adapters and local harness changes are included in this source release.

| Directory | Upstream | License |
| --- | --- | --- |
| `agents/deepagents/` | [Deep Agents](https://github.com/langchain-ai/deepagents), library 0.7.5 | [MIT](agents/deepagents/LICENSE) |
| `agents/pi/` | [Pi](https://github.com/earendil-works/pi), packages 0.84.1 | [MIT](agents/pi/LICENSE) |
| `agents/deepseek-harness/` | [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), 0.1.1-rc.2 | [MIT](agents/deepseek-harness/LICENSE) |
| `agents/mini-swe-agent/` | [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent), 2.4.6 | [MIT](agents/mini-swe-agent/LICENSE.md) |
| `environment/vendor/verl/` | [verl](https://github.com/verl-project/verl), 0.9.0 | [Apache-2.0](environment/vendor/verl/LICENSE), [notice](environment/vendor/verl/Notice.txt) |
| `environment/vendor/miniwob/` | [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus) | [MIT](environment/vendor/miniwob/LICENSE) |

Pinned MiniWoB asset hashes are in `environment/hrp_miniwob/upstream.json` and
`environment/hrp_miniwob/v2/upstream.json`. Training dependency revisions are
recorded in `environment/upstream.json`. Benchmark repositories are downloaded
at the revisions in `src/benchmark/config/native_experiment.json`.

Additional notices shipped inside third-party directories remain applicable.

The shared MiniWoB harness loader was developed from the
[Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) experimental loop.
The bundled verl copy contains its runtime package and build metadata.

Pi's missing `NodeExecutionEnv` source is restored from its published 0.84.1
source map. Its Cloudflare provider has an explicit API type for source builds.
