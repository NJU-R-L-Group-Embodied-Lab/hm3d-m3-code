"""
Multi-agent extension for lavira (P1 of the multi-agent migration).

Importing this package has two side effects:
  1. Adds NUM_AGENTS / AGENT_<i> defaults to habitat_extensions' task config so
     YACS will accept multi-agent YAML files without unknown-key errors.
  2. Registers `Sim-MA-v1`, a multi-agent variant of lavira's `Sim-v1`.

Both habitat-lab v0.1.7 and v0.2.x use the uppercase YACS API for SIMULATOR.*,
so the patch mirrors Co-NavGPT2's habitat fork without further version branching.
"""

from habitat.config.default import Config as CN

from habitat_extensions.config.default import _C as _TASK_C


# Official val uses K<=4; the multi-agent task config is declared for up to 9 agents.
MAX_SIM_AGENTS = 9


def _ensure_multi_agent_defaults(max_agents: int = MAX_SIM_AGENTS) -> None:
    """Pre-declare NUM_AGENTS and AGENT_1..AGENT_{n-1} on the shared task config.

    Calling this is idempotent. We always declare up to `max_agents` slots so
    the YAML can pick `NUM_AGENTS` in [1, max_agents] without further edits.
    """
    was_frozen = _TASK_C.is_frozen()
    if was_frozen:
        _TASK_C.defrost()

    if "NUM_AGENTS" not in _TASK_C.SIMULATOR:
        _TASK_C.SIMULATOR.NUM_AGENTS = 1

    # AGENT_0 already exists (defined by habitat-lab itself). Mirror its schema
    # for AGENT_1..N-1 so YACS knows the key set.
    template = _TASK_C.SIMULATOR.AGENT_0.clone()
    for i in range(1, max_agents):
        name = f"AGENT_{i}"
        if name not in _TASK_C.SIMULATOR:
            _TASK_C.SIMULATOR[name] = template.clone()

    if was_frozen:
        _TASK_C.freeze()


_ensure_multi_agent_defaults()

# Importing multi_sim triggers the @registry.register_simulator side effect.
from habitat_extensions.multi_agent import multi_sim  # noqa: E402,F401
