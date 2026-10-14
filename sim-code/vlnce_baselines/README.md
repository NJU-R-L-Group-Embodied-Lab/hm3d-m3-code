# vlnce_baselines

LaViRA agent code used by the HM3D-M3 simulation benchmark.

This package is derived from
[CA-Nav](https://github.com/Chenkehan21/CA-Nav-code), which itself builds on
[VLN-CE](https://github.com/jacobkrantz/VLN-CE), and has been modified for
multi-agent HM3D-M3 ObjectNav. The single-agent LaViRA base code was
contributed by a co-author.

Layout:

- `ZS_Evaluator_mp.py`: `VLMReasoningAgent` (language-action / vision-action
  LLM reasoning) and the original single-agent zero-shot evaluator.
- `multi/`: multi-agent runner (`ma_runner`), shared mapping/planning,
  decision helpers and the step visualizer.
- `map/`, `models/`: semantic mapping, FMM planner and waypoint policies.
- `utils/`: prompts, LLM API client, depth/geometry and visualization helpers.
- `config/`: experiment configs (yacs). Each one points to a Habitat task
  config in `habitat_extensions/config/` via `BASE_TASK_CONFIG_PATH`.

LLM access is configured only through environment variables: `LLM_API_KEY`,
`LLM_BASE_URL` (default `https://dashscope.aliyuncs.com/compatible-mode/v1`)
and `LLM_MODEL` (default `qwen3.5-27b`). `DASHSCOPE_API_KEY`,
`DASHSCOPE_BASE_URL` and `LAVIRA_API_KEY` are accepted as fallbacks.

`import vlnce_baselines` is lightweight; submodules load on first use.
Run commands from `sim-code/`, e.g. `python -m vlnce_baselines.multi.ma_runner --help`.
