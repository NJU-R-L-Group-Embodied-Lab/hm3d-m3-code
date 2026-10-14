"""
Multi-agent habitat simulator wrapper.

Subclasses lavira's `Sim-v1` (which already extends habitat-lab's `HabitatSim`
with `step_without_obs`). The only multi-agent additions live in three methods:

  * `create_sim_config`: builds N agent_configs instead of 1.
  * `_update_agents_state`: applies IS_SET_START_STATE per agent.
  * `reset` / `step`: return a list of per-agent observation dicts.

Ported from Co-NavGPT2's `multi-robot-setting/habitat/sims/habitat_simulator/`
patch, with habitat-lab v0.1.7-compatible naming (uppercase YACS keys).
"""

from typing import Any, List, Optional, Union

import numpy as np

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import Observations

from habitat.sims.habitat_simulator.actions import HabitatSimActions

from habitat_extensions.habitat_simulator import Simulator as SingleAgentSimulator


def _is_stop_action(action: Any) -> bool:
    """habitat-sim 0.1.7 has no `stop` move_fn."""
    if action is None:
        return False
    if isinstance(action, (list, tuple, np.ndarray)) and not isinstance(
        action, (str, bytes)
    ):
        if len(action) != 1:
            return False
        action = action[0]
    if isinstance(action, str):
        return action.strip().lower() == "stop"
    try:
        return int(action) == int(HabitatSimActions.STOP)
    except (TypeError, ValueError):
        return False


@registry.register_simulator(name="Sim-MA-v1")
class MultiAgentSimulator(SingleAgentSimulator):
    """Multi-agent variant of `Sim-v1`.

    Expects `SIMULATOR.NUM_AGENTS` and `SIMULATOR.AGENTS = ["AGENT_0", ...]`
    in the habitat task config. Per-agent sensor specs are taken from the
    `AGENT_0` block (all agents share an identical sensor stack for now).
    """

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _num_agents(self) -> int:
        return int(getattr(self.habitat_config, "NUM_AGENTS", 1))

    def _agent_names(self) -> List[str]:
        names = list(getattr(self.habitat_config, "AGENTS", ["AGENT_0"]))
        n = self._num_agents()
        if len(names) < n:
            names = [f"AGENT_{i}" for i in range(n)]
        return names[:n]

    # ------------------------------------------------------------------ #
    # Sim configuration
    # ------------------------------------------------------------------ #
    def create_sim_config(self, _sensor_suite) -> habitat_sim.Configuration:
        # Defer to the single-agent path to build sim_cfg + a single
        # agent_config (with sensors fully populated), then replicate.
        cfg = super().create_sim_config(_sensor_suite)

        n = self._num_agents()
        if n <= 1:
            return cfg

        base_agent_cfg = cfg.agents[0]
        cfg.agents = [base_agent_cfg for _ in range(n)]
        return cfg

    # ------------------------------------------------------------------ #
    # Per-agent start-state application
    # ------------------------------------------------------------------ #
    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, name in enumerate(self._agent_names()):
            agent_cfg = getattr(self.habitat_config, name)
            if getattr(agent_cfg, "IS_SET_START_STATE", False):
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True
        return is_updated

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #
    def _per_agent_obs(self) -> List[Observations]:
        out = []
        for i in range(self._num_agents()):
            sim_obs_i = self.get_sensor_observations(agent_ids=i)
            out.append(self._sensor_suite.get_observations(sim_obs_i))
        return out

    def reset(self) -> List[Observations]:
        n = self._num_agents()
        if n <= 1:
            return super().reset()

        # habitat_sim's reset() returns the default agent's obs; we drop it
        # and re-fetch per-agent observations explicitly.
        _ = habitat_sim.Simulator.reset(self)
        self._update_agents_state()

        obs = self._per_agent_obs()
        self._prev_sim_obs = obs
        return obs

    def step(
        self,
        action: Union[int, str, List, np.ndarray],
        agent_id: Optional[int] = None,
    ) -> List[Observations]:
        n = self._num_agents()
        if n <= 1:
            act = action
            if isinstance(action, (list, tuple, np.ndarray)) and not isinstance(
                action, (str, bytes)
            ):
                act = action[0]
            # Same hold as K>1: do not send STOP into habitat-sim 0.1.7.
            if _is_stop_action(act):
                raw = self.get_sensor_observations()
                obs = self._sensor_suite.get_observations(raw)
                self._prev_sim_obs = raw
                return obs
            return super().step(action)

        # action can be a list/array of length N, or a single action paired
        # with `agent_id` to step a single agent (the others noop).
        # habitat-sim 0.1.7 has no `stop` move_fn; treat STOP as a hold.
        if agent_id is None:
            assert hasattr(action, "__len__") and len(action) == n, (
                f"Multi-agent step expects {n} actions, got {action!r}"
            )
            action_dict = {
                i: action[i]
                for i in range(n)
                if int(action[i]) != int(HabitatSimActions.STOP)
            }
            if action_dict:
                habitat_sim.Simulator.step(self, action_dict)
        elif int(action) != int(HabitatSimActions.STOP):
            habitat_sim.Simulator.step(self, {agent_id: action})

        obs = self._per_agent_obs()
        self._prev_sim_obs = obs
        return obs

    # ------------------------------------------------------------------ #
    # Misc accessors (kept for parity with Co-NavGPT2 callers)
    # ------------------------------------------------------------------ #
    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self.get_agent(agent_id).get_state()
