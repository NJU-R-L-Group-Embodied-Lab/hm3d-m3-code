# habitat_extensions

Habitat extensions used by the HM3D-M3 simulation code.

Derived from the `habitat_extensions` package of
[VLN-CE](https://github.com/jacobkrantz/VLN-CE) and patched for
**Habitat v0.1.7** (habitat-lab / habitat-sim 0.1.7). Changes for HM3D-M3:

- `multi_agent/`: the `Sim-MA-v1` simulator (K agents in one scene) and the
  multi-agent task config (`multi_agent/__init__.py`).
- `task.py`: the `ObjectNav-v1` dataset loader for HM3D-M3 episodes (the VLN-CE
  `VLN-CE-v1` / `RxR-VLN-CE-v1` loaders are kept for compatibility).
- `sensors.py`, `measures.py`, `maps.py`, `habitat_simulator.py`, `pose_utils.py`:
  sensors, metrics and top-down map helpers used by the runners.
- `config/`: Habitat task configs. The experiment configs in
  `vlnce_baselines/config/*.yaml` point here via `BASE_TASK_CONFIG_PATH`;
  the paper runs use `multi_objectnav_v2.yaml`.

Importing `habitat_extensions` registers these components with the Habitat
registry.
