"""
Async multi-agent ObjectNav evaluator that reuses lavira's VLMReasoningAgent
per agent, a shared Semantic_Mapping for FMM, and an inter-agent announcement
buffer ("broadcast") fed into LA prompts via the existing `feedback` parameter.

Loop (per sim step):

    for i in 0..N-1:
        s = states[i]
        if s.stopped: continue
        if s.needs_decision:
            # 1) collect panorama via 12-step in-place rotation; OTHER agents
            #    keep executing their queued FMM actions during these 12 steps
            #    (this is the async point from Q9=A).
            pano = collect_panorama_async(sim, agent_i=i)
            # 2) double-check stop
            if agents[i].double_check_stop(pano, ...):
                EPISODE_STOP; return
            # 3) LA pick direction (with broadcast of other agents' last directions)
            decision = agents[i].navigate_or_backtrack(feedback=ANN, ...)
            # 4) VA pick bbox on the chosen direction's RGB → world target
            bbox = agents[i].query_llm(...)
            target_xy_map = bbox_to_world(bbox, pano)
            s.set_target(target_xy_map)
            announcements[i] = (direction, step)
            s.needs_decision = False
            continue  # don't double-act this step
        actions[i] = next_fmm_action(s, shared_map)
        if reached(s): s.needs_decision = True
    sim.step(actions)
    update_shared_map(obs)

The complete lavira algorithm runs per-agent (LA + VA + optional TODO +
optional backtrack). The only multi-agent coupling lives at three points:
shared map, shared TODO, and the announcement string injected into LA
feedback.

P0 of this rewrite: leave TODO + backtrack off for the first sanity run,
then layer them back once async + map sharing is stable.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import habitat_sim
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import habitat_extensions  # noqa: F401
import habitat_extensions.multi_agent  # noqa: F401
from habitat_extensions.config.default import get_extended_config

from vlnce_baselines.multi import ma_decision
from vlnce_baselines.multi.ma_visualizer import MAVisualizer
from vlnce_baselines.multi.mp3d_obs import wrap_sim_visual_obs
from vlnce_baselines.multi.ma_mapping_planner import (
    SharedMappingPlanner,
    WAYPOINT_REACH_M,
    world_xz_reached,
)


def set_num_agents(cfg, n: int) -> None:
    """Set Habitat / mapper agent count for this episode (K=1..9)."""
    n = int(n)
    names = [f"AGENT_{i}" for i in range(n)]
    cfg.defrost()
    cfg.SIMULATOR.NUM_AGENTS = n
    if hasattr(cfg.SIMULATOR, "AGENTS"):
        cfg.SIMULATOR.AGENTS = names
    if hasattr(cfg, "TASK_CONFIG") and hasattr(cfg.TASK_CONFIG, "SIMULATOR"):
        cfg.TASK_CONFIG.defrost()
        cfg.TASK_CONFIG.SIMULATOR.NUM_AGENTS = n
        if hasattr(cfg.TASK_CONFIG.SIMULATOR, "AGENTS"):
            cfg.TASK_CONFIG.SIMULATOR.AGENTS = names
        cfg.TASK_CONFIG.freeze()
    cfg.freeze()


def _episode_num_agents(ep, default: int = 2) -> int:
    n = getattr(ep, "num_agents", None)
    if n:
        return int(n)
    agents = getattr(ep, "agents", None)
    if agents:
        return len(agents)
    return int(default)


def consume_va_bbox_overlay(states, forward_rgbs):
    """Use a one-shot VA box only on this frame; later RGB tiles stay clean."""
    rgbs = list(forward_rgbs)
    boxes = [None] * len(states)
    for i, state in enumerate(states):
        bbox = getattr(state, "va_bbox_flash", None)
        if bbox is None:
            continue
        flash_rgb = getattr(state, "va_bbox_flash_rgb", None)
        if flash_rgb is not None and i < len(rgbs):
            rgbs[i] = flash_rgb
        boxes[i] = bbox
        state.va_bbox_flash = None
        state.va_bbox_flash_rgb = None
    return rgbs, boxes


# ---------------------------------------------------------------------------- #
# Per-agent state
# ---------------------------------------------------------------------------- #
@dataclass
class PerAgentState:
    agent_id: int
    needs_decision: bool = True
    stopped: bool = False
    success: bool = False
    target_xy_world: Optional[np.ndarray] = None
    last_direction: str = ""
    last_direction_step: int = -10**9
    waypoint_history: List[Dict] = field(default_factory=list)
    path_length: float = 0.0
    start_geo: float = float("inf")
    last_position: Optional[np.ndarray] = None
    consec_stuck: int = 0
    # geodesic distance to the current sub-goal at the previous step, used for
    # progress-based stuck detection (reset whenever a new sub-goal is picked)
    last_tg_d: float = float("inf")
    last_map_d: float = float("inf")
    consec_turns: int = 0
    nav_steps_since_decision: int = 0
    geom_replan_only: bool = False
    going_to_stop: bool = False
    pending_claim_goal_id: Optional[str] = None
    stop_feedback: str = ""
    pending_pano: Optional[List[Dict]] = None
    last_pano: Optional[List[Dict]] = None
    last_pano_step: int = -1
    # rolling RGB-frame history (every HISTORY_INTERVAL sim steps), used by
    # VLMReasoningAgent.navigate_or_backtrack / double_check_stop via the
    # history_images parameter when use_continuous_history=True
    history_images: List[Dict] = field(default_factory=list)
    # full world-position trail (for ma_visualizer path overlay)
    position_trail: List[np.ndarray] = field(default_factory=list)
    # most recent VA bbox (kept for text / bbox_step dumps)
    last_va_bbox: Optional[Dict] = None
    # One-shot overlay: draw only on the next combined frame, then clear.
    va_bbox_flash: Optional[Dict] = None
    va_bbox_flash_rgb: Optional[np.ndarray] = None
    # most recent target cell in this env's mapping full_map: (row, col)
    target_cell_rc: Optional[Tuple[int, int]] = None
    # most recent low-level action issued this step (for info panel)
    last_action: Optional[int] = None
    # most recent LA reasoning string (for info panel)
    last_la_reasoning: Optional[str] = None
    # Latest LA progress_analysis for async board+PA broadcast (uncut).
    last_progress_analysis: str = ""
    last_progress_step: int = -1
    # True after arriving at the current X; hold until the next LA/VA target.
    holding_waypoint: bool = False


# ---------------------------------------------------------------------------- #
# Helpers: yaw + panorama
# ---------------------------------------------------------------------------- #
def yaw_from_quat(rotation) -> float:
    if hasattr(rotation, "w"):
        w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    else:
        w, x, y, z = rotation
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(siny_cosp, cosy_cosp))


def collect_panorama_async(
    sim,
    agent_i: int,
    n_agents: int,
    other_action_fn,
    initial_obs: List[Dict],
    mapping_planner=None,
    on_sim_step=None,
    step_ref: Optional[List[int]] = None,
) -> List[Dict]:
    """Spin agent_i in place for 12 sim steps; other agents continue executing
    `other_action_fn(j)` actions. Returns 4 panorama frames (every 3rd of 12,
    matching lavira's get_panorama).

    If `mapping_planner` is provided, also feeds each per-step obs into
    lavira's Semantic_Mapping so its pose-delta tracking remains tick-accurate
    (otherwise we accumulate ~24 panorama steps of drift between mp.step calls
    and the agent_view rotation/translation overflows the local-map window)."""
    panorama_frames = []
    last_obs = initial_obs
    for turn_step in range(1, 13):
        actions = [HabitatSimActions.STOP] * n_agents
        actions[agent_i] = HabitatSimActions.TURN_LEFT
        for j in range(n_agents):
            if j == agent_i:
                continue
            a = other_action_fn(j)
            actions[j] = a if a is not None else HabitatSimActions.STOP
        last_obs = sim.step(actions)
        if step_ref is not None:
            step_ref[0] += 1
        if mapping_planner is not None:
            try:
                mapping_planner.step(sim, list(last_obs))
            except Exception:
                pass
        if on_sim_step is not None:
            try:
                on_sim_step(last_obs, actions)
            except Exception:
                pass
        panorama_frames.append({
            "rgb": last_obs[agent_i]["rgb"].copy(),
            "depth": last_obs[agent_i]["depth"].copy(),
            "angle": (turn_step * 30) % 360,
            "step": turn_step,
        })
    panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]
    return panorama_frames[::3], last_obs


def collect_panorama_sync_all(
    sim,
    n_agents: int,
    initial_obs: List[Dict],
    mapping_planner=None,
    on_sim_step=None,
    step_ref: Optional[List[int]] = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """All agents turn left together for 12×30° (360°), then 4-dir frames each.

    Matches single-agent LaViRA's opening spin: nobody walks until every agent has a
    panorama. Later mid-episode decisions still use ``collect_panorama_async``.
    """
    per_agent: List[List[Dict]] = [[] for _ in range(n_agents)]
    last_obs = initial_obs
    for turn_step in range(1, 13):
        actions = [HabitatSimActions.TURN_LEFT] * n_agents
        last_obs = sim.step(actions)
        if step_ref is not None:
            step_ref[0] += 1
        if mapping_planner is not None:
            try:
                mapping_planner.step(sim, list(last_obs))
            except Exception:
                pass
        if on_sim_step is not None:
            try:
                on_sim_step(last_obs, actions)
            except Exception:
                pass
        for i in range(n_agents):
            per_agent[i].append({
                "rgb": last_obs[i]["rgb"].copy(),
                "depth": last_obs[i]["depth"].copy(),
                "angle": (turn_step * 30) % 360,
                "step": turn_step,
            })
    out = []
    for frames in per_agent:
        frames = [frames[-1]] + frames[:-1]
        out.append(frames[::3])
    return out, last_obs


def collect_panorama_sync_subset(
    sim,
    spin_ids: Sequence[int],
    n_agents: int,
    other_action_fn,
    initial_obs: List[Dict],
    mapping_planner=None,
    on_sim_step=None,
    step_ref: Optional[List[int]] = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """One 12-step Habitat tick: listed agents TURN_LEFT together, others act.

    Habitat has a single sim, so panoramas cannot be collected on threads.
    Agents that need a new pano this wave can still share the same 12 steps.
    """
    ordered = [int(i) for i in spin_ids]
    if not ordered:
        return [], initial_obs
    spin_set = set(ordered)
    per_agent: Dict[int, List[Dict]] = {i: [] for i in ordered}
    last_obs = initial_obs
    for turn_step in range(1, 13):
        actions = [HabitatSimActions.STOP] * n_agents
        for i in ordered:
            actions[i] = HabitatSimActions.TURN_LEFT
        for j in range(n_agents):
            if j in spin_set:
                continue
            a = other_action_fn(j)
            actions[j] = a if a is not None else HabitatSimActions.STOP
        last_obs = sim.step(actions)
        if step_ref is not None:
            step_ref[0] += 1
        if mapping_planner is not None:
            try:
                mapping_planner.step(sim, list(last_obs))
            except Exception:
                pass
        if on_sim_step is not None:
            try:
                on_sim_step(last_obs, actions)
            except Exception:
                pass
        for i in ordered:
            per_agent[i].append({
                "rgb": last_obs[i]["rgb"].copy(),
                "depth": last_obs[i]["depth"].copy(),
                "angle": (turn_step * 30) % 360,
                "step": turn_step,
            })
    out = []
    for i in ordered:
        frames = per_agent[i]
        frames = [frames[-1]] + frames[:-1]
        out.append(frames[::3])
    return out, last_obs


# ---------------------------------------------------------------------------- #
# Main runner
# ---------------------------------------------------------------------------- #
class MultiAgentRunner:
    """Single-process async multi-agent ObjectNav evaluator.

    For the first sanity run we use a trivial sub-policy (no VLMReasoningAgent
    plumbing yet) — each agent picks a *random* navmesh point as its target
    and FMM-walks to it via habitat's built-in pathfinder. This validates the
    async + sim integration end-to-end before we wire in LA/VA/TODO.

    P1 of this module flips `use_vlm=True` to invoke VLMReasoningAgent at each
    waypoint. P2 layers TODO + broadcast + backtrack.
    """

    def __init__(
        self,
        cfg,
        max_episode_steps: int = 500,
        success_distance: float = 1.0,
        min_spawn_distance: float = 3.0,
        save_dir: Optional[str] = None,
        log_every: int = 25,
        use_vlm: bool = False,
        la_thinking_budget: int = 128,
        la_enable_thinking: bool = True,
        announcement_window_steps: int = 30,
        use_continuous_history: bool = True,
        history_interval: int = 2,
        visualize: bool = False,
        viz_every: int = 1,
        vlm_parallel_workers: int = 0,
    ):
        self.cfg = cfg
        self.n_agents = int(cfg.SIMULATOR.NUM_AGENTS)
        self.max_steps = max_episode_steps
        self.success_distance = success_distance
        self.min_spawn_distance = min_spawn_distance
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_every = log_every
        self.use_vlm = use_vlm
        self.la_thinking_budget = la_thinking_budget
        self.la_enable_thinking = bool(la_enable_thinking)
        # 0 = one worker per ready agent this wave.
        self.vlm_parallel_workers = int(vlm_parallel_workers)
        self.announcement_window_steps = announcement_window_steps
        self.use_continuous_history = use_continuous_history
        self.history_interval = history_interval
        self.visualize = visualize
        self.viz_every = viz_every
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Sensor params for bbox→world projection.
        self._width = int(cfg.SIMULATOR.RGB_SENSOR.WIDTH)
        self._height = int(cfg.SIMULATOR.RGB_SENSOR.HEIGHT)
        self._hfov = float(cfg.SIMULATOR.RGB_SENSOR.HFOV)
        self._dmin = float(cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH)

        # Lavira-native Semantic_Mapping + FMM (always on when MAP cfg present)
        self.mp: Optional[SharedMappingPlanner] = None
        if hasattr(cfg, "MAP"):
            # `cfg` here is the lavira top-level config; SIMULATOR lives under
            # TASK_CONFIG but the planner expects it at `cfg.SIMULATOR`. The
            # main() wrapper builds `runner_cfg` that exposes both.
            try:
                self.mp = SharedMappingPlanner(cfg, num_agents=self.n_agents)
                print(f"[ma-runner] SharedMappingPlanner ready (N={self.n_agents})")
            except Exception as e:
                print(f"[ma-runner][warn] could not construct SharedMappingPlanner: {e!r}")
                self.mp = None

        self._viz: Optional[MAVisualizer] = None
        if self.visualize and self.save_dir:
            self._viz = MAVisualizer(
                save_dir=str(self.save_dir),
                hfov_deg=self._hfov,
                depth_h=self._height,
                depth_w=self._width,
                depth_min_m=self._dmin,
                depth_max_m=float(cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH),
                sensor_height_m=float(cfg.SIMULATOR.DEPTH_SENSOR.POSITION[1]),
            )
        self._dmax = float(cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH)

        # Per-agent VLM (created lazily on first use to avoid the cost when
        # use_vlm=False).
        self._vlms: List = []

    # ------------------------------------------------------------------ #
    # VLM construction (lazy)
    # ------------------------------------------------------------------ #
    def _ensure_vlms(self) -> None:
        if not self.use_vlm:
            return
        from vlnce_baselines.ZS_Evaluator_mp import VLMReasoningAgent
        from vlnce_baselines.utils.visualization import LaViRAVisualizer
        from vlnce_baselines.utils.api import env_llm_api_key, env_llm_base_url, env_llm_model
        # Grow to current K. Extra agents from a previous larger K are reused.
        while len(self._vlms) < self.n_agents:
            viz = LaViRAVisualizer(mapping_module=None, visualize=False,
                                   save_dir=str(self.save_dir or "/tmp"))
            agent = VLMReasoningAgent(
                visualizer=viz,
                task_type="OBJNAV",
                use_todo_list=False,         # P1: TODO off
                use_working_memory=False,
                backtrack_second_chance=False,
                use_backtrack=True,
                no_todo_prompt=True,
                la_model_override=env_llm_model(),
                la_base_url_override=env_llm_base_url(),
                la_api_key_override=env_llm_api_key(),
                la_enable_thinking=self.la_enable_thinking,
                la_thinking_budget=self.la_thinking_budget,
            )
            self._vlms.append(agent)
        if self._vlms:
            print(
                f"[ma-runner] VLMReasoningAgent ready n={len(self._vlms)} "
                f"(need {self.n_agents}) thinking="
                f"{'on' if self.la_enable_thinking else 'off'} "
                f"budget={self.la_thinking_budget}"
            )

    def _ensure_num_agents(self, n: int) -> bool:
        """Match Habitat + shared map to this episode's K. True if sim must rebuild."""
        n = int(n)
        cur = int(self.cfg.SIMULATOR.NUM_AGENTS)
        map_n = int(getattr(self.mp, "n", -1)) if self.mp is not None else n
        if cur == n and self.n_agents == n and map_n == n:
            return False
        set_num_agents(self.cfg, n)
        self.n_agents = n
        if hasattr(self.cfg, "MAP"):
            try:
                self.mp = SharedMappingPlanner(self.cfg, num_agents=n)
                print(f"[ma-runner] SharedMappingPlanner rebuilt (N={n})")
            except Exception as e:
                print(f"[ma-runner][warn] could not rebuild SharedMappingPlanner: {e!r}")
                self.mp = None
        print(f"[ma-runner] num_agents {cur} -> {n}")
        return True

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(self, episodes, seed: int = 0) -> List[Dict]:
        results = []
        sim = None
        current_scene = None
        for idx, ep in enumerate(episodes):
            print(f"\n[ma-runner] === episode {ep.episode_id} ({idx+1}/{len(episodes)}) "
                  f"goal={ep.object_category} ===")
            want_n = _episode_num_agents(ep, default=self.n_agents)
            rebuilt_k = self._ensure_num_agents(want_n)
            if sim is None or current_scene != ep.scene_id or rebuilt_k:
                if sim is not None:
                    sim.close()
                sim = self._build_sim(ep.scene_id)
                current_scene = ep.scene_id
            try:
                r = self._run_episode(sim, ep, seed=seed + idx)
            except Exception as e:
                import traceback
                print(f"[ma-runner][ERR] {e!r}")
                traceback.print_exc()
                continue
            results.append(r)
            print(f"[ma-runner] ep {ep.episode_id} steps={r['steps']} "
                  f"joint_SR={r['joint_success']} team_SPL={r['team_spl']:.3f} "
                  f"per_agent_SR={[s['success'] for s in r['per_agent']]}")
            if self.save_dir:
                import json
                with open(self.save_dir / f"episode_{r['episode_id']}.json", "w") as f:
                    json.dump(r, f, indent=2, default=float)
        if sim is not None:
            sim.close()
        if self.save_dir and results:
            self._dump_summary(results)
        return results

    def evaluate_dynamic(
        self, claim_fn, seed: int = 0, on_done=None, on_fail=None
    ) -> List[Dict]:
        """Pop episodes from ``claim_fn(prefer_scene=...)`` until it returns None."""
        results = []
        sim = None
        current_scene = None
        idx = 0
        while True:
            ep = claim_fn(prefer_scene=current_scene)
            if ep is None:
                print(f"[ma-runner] dynamic queue empty after {idx} episodes")
                break
            print(
                f"\n[ma-runner] === episode {ep.episode_id} "
                f"(dynamic #{idx + 1}) goal={ep.object_category} ==="
            )
            want_n = _episode_num_agents(ep, default=self.n_agents)
            rebuilt_k = self._ensure_num_agents(want_n)
            if sim is None or current_scene != ep.scene_id or rebuilt_k:
                if sim is not None:
                    sim.close()
                sim = self._build_sim(ep.scene_id)
                current_scene = ep.scene_id
            try:
                r = self._run_episode(sim, ep, seed=seed + idx)
            except Exception as e:
                import traceback
                print(f"[ma-runner][ERR] {e!r}")
                traceback.print_exc()
                if on_fail is not None:
                    try:
                        on_fail(str(ep.episode_id))
                    except Exception as cb_exc:  # noqa: BLE001
                        print(f"[ma-runner][warn] on_fail: {cb_exc!r}")
                idx += 1
                continue
            results.append(r)
            print(
                f"[ma-runner] ep {ep.episode_id} steps={r['steps']} "
                f"joint_SR={r['joint_success']} team_SPL={r['team_spl']:.3f} "
                f"per_agent_SR={[s['success'] for s in r['per_agent']]}"
            )
            if self.save_dir:
                import json
                with open(self.save_dir / f"episode_{r['episode_id']}.json", "w") as f:
                    json.dump(r, f, indent=2, default=float)
            if on_done is not None:
                try:
                    on_done(str(r["episode_id"]))
                except Exception as cb_exc:  # noqa: BLE001
                    print(f"[ma-runner][warn] on_done: {cb_exc!r}")
            idx += 1
        if sim is not None:
            sim.close()
        if self.save_dir and results:
            self._dump_summary(results)
        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_sim(self, scene_id: str):
        sim_cfg = self.cfg.SIMULATOR.clone()
        sim_cfg.defrost()
        sim_cfg.SCENE = scene_id
        sid = str(scene_id).replace("\\", "/")
        if "/mp3d/" in sid or "/train/" in sid or "/qual_scenes/" in sid:
            sim_cfg.SCENE_DATASET = ""
        elif "SCENE_DATASET" not in sim_cfg:
            sim_cfg.SCENE_DATASET = (
                "data/scene_datasets/hm3d-1.0/hm3d_v0.2/"
                "hm3d_annotated_basis.scene_dataset_config.json"
            )
        sim_cfg.freeze()
        return wrap_sim_visual_obs(make_sim(id_sim=sim_cfg.TYPE, config=sim_cfg), sid)

    def _spawn(self, sim, ep) -> List[np.ndarray]:
        n = self.n_agents
        starts: List[np.ndarray] = [np.asarray(ep.start_position, dtype=np.float32)]
        tries = 0
        while len(starts) < n and tries < 500:
            tries += 1
            p = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            if not sim.pathfinder.is_navigable(p):
                continue
            if any(np.linalg.norm(p - q) < self.min_spawn_distance for q in starts):
                continue
            if abs(p[1] - starts[0][1]) > 0.5:
                continue
            starts.append(p)
        if len(starts) < n:
            raise RuntimeError(f"could not sample {n} starts for ep {ep.episode_id}")
        return starts

    def _min_geodesic(self, sim, start: np.ndarray, goals: List[np.ndarray]) -> float:
        best = float("inf")
        for g in goals:
            d = sim.geodesic_distance(start, g)
            if d is None or not np.isfinite(d):
                continue
            if d < best:
                best = float(d)
        return best

    def _next_fmm_action(self, sim, agent_id: int, target_world: np.ndarray) -> int:
        """Habitat-native shortest-path next action (used as a stand-in until
        the lavira FMM/Semantic_Mapping integration lands)."""
        state = sim.get_agent_state(agent_id)
        pos = np.asarray(state.position, dtype=np.float32)
        if sim.geodesic_distance(pos, target_world) <= 0.3:
            return -1  # reached
        path = habitat_sim.ShortestPath()
        path.requested_start = pos
        path.requested_end = np.asarray(target_world, dtype=np.float32)
        if not sim.pathfinder.find_path(path) or len(path.points) < 2:
            return HabitatSimActions.MOVE_FORWARD
        nxt = np.asarray(path.points[1], dtype=np.float32)
        delta = nxt - pos
        delta[1] = 0.0
        if np.linalg.norm(delta) < 1e-6:
            return HabitatSimActions.MOVE_FORWARD
        yaw = yaw_from_quat(state.rotation)
        agent_fwd = np.array([math.sin(yaw), -math.cos(yaw)], dtype=np.float32)
        target_dir = np.array([delta[0], delta[2]], dtype=np.float32)
        target_dir /= max(np.linalg.norm(target_dir), 1e-6)
        dot = float(np.clip(np.dot(agent_fwd, target_dir), -1.0, 1.0))
        cross = float(agent_fwd[0] * target_dir[1] - agent_fwd[1] * target_dir[0])
        ang = math.degrees(math.atan2(cross, dot))
        turn = float(sim.habitat_config.TURN_ANGLE)
        if ang > turn:
            return HabitatSimActions.TURN_LEFT
        if ang < -turn:
            return HabitatSimActions.TURN_RIGHT
        return HabitatSimActions.MOVE_FORWARD

    def _pick_random_navigable_target(
        self,
        sim,
        rng,
        around: np.ndarray,
        min_dist_m: float = 3.0,
        avoid=None,
        agent_id: int = 0,
    ) -> np.ndarray:
        """Random same-floor nav point far enough that FMM will actually walk."""
        around = np.asarray(around, dtype=np.float32)
        avoid_pts = [
            np.asarray(a, dtype=np.float32)
            for a in (avoid or [])
            if a is not None
        ]
        best = None
        best_d = -1.0
        for _ in range(80):
            p = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            if p.size < 3 or not np.all(np.isfinite(p)):
                continue
            if abs(float(p[1]) - float(around[1])) > 0.5:
                continue
            d = float(np.linalg.norm(p - around))
            if d > best_d:
                best, best_d = p, d
            if d < float(min_dist_m):
                continue
            if any(float(np.linalg.norm(p - a)) < 2.0 for a in avoid_pts):
                continue
            return p
        if best is not None and best_d >= 1.5:
            return best
        return self._forward_probe_target(sim, around, agent_id=agent_id)

    def _forward_probe_target(
        self, sim, around: np.ndarray, dist_m: float = 3.0, agent_id: int = 0
    ) -> np.ndarray:
        """Last-resort target: snap a point ahead of the agent's heading."""
        around = np.asarray(around, dtype=np.float32)
        yaw = 0.0
        try:
            yaw = yaw_from_quat(sim.get_agent_state(int(agent_id)).rotation)
        except Exception:
            yaw = 0.0
        probe = around.copy()
        probe[0] += float(dist_m) * math.sin(yaw)
        probe[2] -= float(dist_m) * math.cos(yaw)
        try:
            snapped = np.asarray(sim.pathfinder.snap_point(probe), dtype=np.float32)
            if snapped.size >= 3 and np.all(np.isfinite(snapped)):
                return snapped
        except Exception:
            pass
        return around

    # ------------------------------------------------------------------ #
    # Per-episode rollout
    # ------------------------------------------------------------------ #
    def _prepare_la_va_job(
        self,
        sim,
        agent_id: int,
        state: "PerAgentState",
        panorama: List[Dict],
        instruction: str,
        announcement_text: str,
        step: int,
        episode_id: str,
        active_goal_id: Optional[str] = None,
        stop_max_depth_m: Optional[float] = None,
        team_topdown_map=None,
        team_topdown_legend: Optional[str] = None,
        target_reference_images: Optional[List[Dict]] = None,
    ):
        """Append waypoint + snapshot pose. Returns (HTTP job kwargs, pose snap)."""
        self._ensure_vlms()
        cur_pos = np.asarray(sim.get_agent_state(agent_id).position, dtype=np.float32)
        from vlnce_baselines.multi.ma_decision import habitat_full_pose
        full_pose = habitat_full_pose(sim, agent_id)
        try:
            from PIL import Image
            init_img = Image.fromarray(panorama[0]["rgb"].astype(np.uint8))
        except Exception:
            init_img = None
        wp = {
            "step": step,
            "init_image": init_img,
            "panorama_frames": panorama,
            "world_coords": (float(cur_pos[0]), float(cur_pos[2])),
            "full_pose": full_pose,
            "description": (
                (state.last_va_bbox or {}).get("target")
                or state.last_direction
                or "decision waypoint"
            ),
        }
        state.waypoint_history.append(wp)
        mapping_pose = None
        yaw_0_rad = 0.0
        if self.mp is not None and self.mp.mapping is not None:
            st = self.mp.mapping.state[agent_id, :3]
            mapping_pose = np.asarray(
                [float(st[0]), float(st[1]), float(st[2])],
                dtype=np.float32,
            )
            yaw_0_rad = self.mp.initial_yaw_rad(agent_id)
        refs = target_reference_images
        if refs is None:
            refs = getattr(self, "_target_reference_images", None)
        if refs is not None:
            refs = list(refs)
        job = {
            "vlm_agent": self._vlms[agent_id],
            "instruction": instruction,
            "visited_targets": state.waypoint_history,
            "panorama_frames": panorama,
            "announcement_text": announcement_text,
            "episode_id": episode_id,
            "current_step": step,
            "history_images": (
                state.history_images if self.use_continuous_history else None
            ),
            "target_reference_images": refs,
            "team_topdown_map": team_topdown_map,
            "team_topdown_legend": team_topdown_legend,
            "width": self._width,
            "height": self._height,
            "agent_id": agent_id,
        }
        snap = {
            "full_pose": full_pose,
            "mapping_pose": mapping_pose,
            "yaw_0_rad": yaw_0_rad,
            "agent_habitat": cur_pos.copy(),
            "active_goal_id": active_goal_id,
            "stop_max_depth_m": (
                float(stop_max_depth_m)
                if stop_max_depth_m is not None
                else float(self.success_distance) + 1.0
            ),
            "step": step,
            "episode_id": episode_id,
        }
        return job, snap

    def _finish_la_va(self, sim, agent_id: int, raw, snap: Dict, panorama: List[Dict]):
        """Emit stop events + project bbox on the Habitat thread."""
        active_goal_id = snap.get("active_goal_id")
        if getattr(raw, "stop_proposed", False):
            reported = str(getattr(raw, "reported_claim_goal_id", "") or "").strip()
            if not active_goal_id and reported:
                active_goal_id = reported
            sink = getattr(self, "_stop_event_sink", None)
            if sink is not None:
                event = {
                    "event_type": "la_stop_proposed",
                    "agent_id": int(agent_id),
                    "step": int(snap.get("step") or 0),
                    "goal_id": str(active_goal_id or ""),
                    "attempt_position": [
                        float(x) for x in np.asarray(snap["agent_habitat"]).ravel()[:3]
                    ],
                }
                try:
                    sink(event)
                except Exception as exc:
                    print(f"[ma-decide][warn] stop event sink failed: {exc!r}")
        return ma_decision.project_la_va_to_decision(
            raw=raw,
            sim=sim,
            agent_id=agent_id,
            panorama_frames=panorama,
            width=self._width,
            height=self._height,
            hfov_deg=self._hfov,
            depth_min=self._dmin,
            depth_max=self._dmax,
            mapping_pose_override=snap.get("mapping_pose"),
            initial_habitat_yaw_rad=float(snap.get("yaw_0_rad") or 0.0),
            agent_habitat_override=snap.get("agent_habitat"),
            active_goal_id=active_goal_id,
        )

    def _run_la_va_wave(self, jobs: List[Dict]):
        return ma_decision.run_la_va_parallel(
            jobs,
            max_workers=getattr(self, "vlm_parallel_workers", 0),
        )

    def _run_api_wave(self, jobs: List[Dict]):
        return ma_decision.run_agent_api_parallel(
            jobs,
            max_workers=getattr(self, "vlm_parallel_workers", 0),
        )

    def _vlm_pick_target(self, sim, agent_id: int, state: "PerAgentState",
                          panorama: List[Dict], instruction: str,
                          announcement_text: str, step: int,
                          episode_id: str, active_goal_id: Optional[str] = None,
                          stop_max_depth_m: Optional[float] = None,
                          team_topdown_map=None,
                          team_topdown_legend: Optional[str] = None):
        """Sequential LA+VA wrapper (single agent). Wave path uses prepare/finish."""
        job, snap = self._prepare_la_va_job(
            sim, agent_id, state, panorama, instruction, announcement_text,
            step, episode_id, active_goal_id=active_goal_id,
            stop_max_depth_m=stop_max_depth_m,
            team_topdown_map=team_topdown_map,
            team_topdown_legend=team_topdown_legend,
        )
        raw = ma_decision.run_la_va(**job)
        return self._finish_la_va(sim, agent_id, raw, snap, panorama), snap["full_pose"]

    def _apply_backtrack(self, sim, agent_id: int, state: "PerAgentState", waypoint_id: int) -> bool:
        """Navigate back to a previous waypoint and drop later history.

        The current decision already appended a waypoint before LA ran, so
        valid ids are ``0 .. len(history)-2``. On success the agent walks to
        that pose; arriving will set ``needs_decision`` again.
        """
        hist = list(state.waypoint_history or [])
        if len(hist) < 2:
            return False
        try:
            wp_id = int(waypoint_id)
        except (TypeError, ValueError):
            return False
        wp_id = max(0, min(wp_id, len(hist) - 2))
        wp = hist[wp_id]
        coords = wp.get("world_coords")
        if not coords or len(coords) < 2:
            return False
        cur = np.asarray(sim.get_agent_state(agent_id).position, dtype=np.float32)
        target_3d = np.array([float(coords[0]), float(cur[1]), float(coords[1])], np.float32)
        snapped = np.asarray(sim.pathfinder.snap_point(target_3d), dtype=np.float32)
        if not np.all(np.isfinite(snapped)):
            snapped = target_3d
        state.waypoint_history = hist[: wp_id + 1]
        cut_step = int(wp.get("step", 0) or 0)
        state.history_images = [
            item for item in state.history_images
            if int(item.get("step", -1)) <= cut_step
        ]
        state.target_xy_world = snapped
        state.holding_waypoint = False
        state.last_direction = "backtrack"
        state.last_va_bbox = None
        state.va_bbox_flash = None
        state.va_bbox_flash_rgb = None
        print(
            f"[ma-runner]   agent_{agent_id} backtrack to waypoint {wp_id} "
            f"at ({float(coords[0]):.2f},{float(coords[1]):.2f})"
        )
        return True

    def _append_history_if_due(self, states, last_obs, step: int) -> None:
        """Enqueue each agent's current forward RGB every `history_interval` env steps."""
        if not self.use_continuous_history:
            return
        if int(step) <= 0 or int(step) % int(self.history_interval) != 0:
            return
        try:
            from PIL import Image
        except ImportError:
            return
        for i, state in enumerate(states):
            if state.stopped:
                continue
            if state.history_images and int(state.history_images[-1].get("step", -1)) == int(step):
                continue
            rgb = last_obs[i].get("rgb") if last_obs is not None else None
            if rgb is None:
                continue
            if getattr(rgb, "dtype", None) is not None and rgb.dtype != np.uint8:
                rgb_to_save = (rgb * 255).astype(np.uint8)
            else:
                rgb_to_save = rgb
            state.history_images.append({
                "step": int(step),
                "image": Image.fromarray(rgb_to_save),
            })

    def _per_sim_step_hook(self, *, sim, ep, states, step_ref, n):
        """Build a closure that renders one frame at the current sim step.
        Called from inside the panorama loop AND the action phase so the
        visualization captures every env step. `step_ref` is a [step] list
        used to read the live step counter from the outer loop."""

        def _hook(last_obs, _actions=None):
            if self._viz is None:
                return
            step = int(step_ref[0])
            # Update visualizer trail + ingest depth for the incremental occ map
            # used by the fallback renderer.
            try:
                from vlnce_baselines.multi.ma_decision import yaw_from_quat as _yaw
                from vlnce_baselines.multi.ma_visualizer import snapshot_world_xyz
                positions = [snapshot_world_xyz(sim.get_agent_state(i).position)
                             for i in range(n)]
                yaws = [_yaw(sim.get_agent_state(i).rotation) for i in range(n)]
                for i in range(n):
                    if states[i].stopped:
                        continue
                    states[i].position_trail.append(positions[i].copy())
                from vlnce_baselines.multi.ma_visualizer import AGENT_COLORS_BGR
                other_agents_overlay = [
                    (positions[i], yaws[i], AGENT_COLORS_BGR[i % len(AGENT_COLORS_BGR)])
                    for i in range(1, n) if not states[i].stopped
                ]
                paths_overlay = [
                    (states[i].position_trail, AGENT_COLORS_BGR[i % len(AGENT_COLORS_BGR)])
                    for i in range(n) if states[i].position_trail
                ]
                goals_overlay = [np.asarray(g.position, dtype=np.float32)
                                  for g in getattr(ep, "goals", [])]
                # Build a SINGLE merged habitat-frame map showing both agents'
                # obstacle/explored union, trails, targets, and the goal —
                # rotates each agent's mapping into habitat axes using its
                # initial yaw, so per-agent yaw-rotated maps overlay coherently.
                lavira_maps_per_agent = None
                lavira_map = None
                if self.mp is not None:
                    try:
                        lavira_map = self.mp.render_merged_habitat_map(
                            sim,
                            agent_paths_world=[states[i].position_trail for i in range(n)],
                            agent_targets_world=[states[i].target_xy_world for i in range(n)],
                            goals_world=goals_overlay,
                        )
                    except Exception:
                        lavira_map = None
                forward_rgbs = [last_obs[i].get("rgb") for i in range(n)]
                goal_name = getattr(self, "_viz_goal_name", None) or getattr(ep, "object_category", "")
                self._viz.render_step(
                    step=step,
                    agent_positions=positions,
                    agent_yaws=yaws,
                    forward_rgbs=forward_rgbs,
                    goal_name=goal_name,
                    episode_id=str(ep.episode_id),
                    instruction=f"Find: {goal_name}",
                    agent_paths=[states[i].position_trail for i in range(n)],
                    waypoints_per_agent=None,
                    goals_world=goals_overlay,
                    per_agent_target_world=[states[i].target_xy_world for i in range(n)],
                    action_per_agent=[states[i].last_action for i in range(n)],
                    la_direction_per_agent=[
                        states[i].last_direction if states[i].last_direction else None
                        for i in range(n)
                    ],
                    la_reasoning_per_agent=[states[i].last_la_reasoning for i in range(n)],
                    todo_list=None,
                    lavira_map_bgr=lavira_map,
                    per_agent_lavira_maps=lavira_maps_per_agent,
                    va_bbox_per_agent=[states[i].last_va_bbox for i in range(n)],
                )
            except Exception as e:
                print(f"[ma-runner][viz-warn] hook failed at step {step}: {e!r}")

        return _hook

    def _run_episode(self, sim, ep, seed: int) -> Dict:
        rng = np.random.default_rng(seed)
        n = self.n_agents
        if self.use_vlm:
            self._ensure_vlms()
            # Reset per-episode state on each VLM agent (waypoint history, etc.)
            for vlm in self._vlms:
                if hasattr(vlm, "last_object"):
                    vlm.last_object = ""
                if hasattr(vlm, "todo_list"):
                    vlm.todo_list = None
                if hasattr(vlm, "reasoning_todo"):
                    vlm.reasoning_todo = ""
                if hasattr(vlm, "stop_feedback"):
                    vlm.stop_feedback = ""
        # Co-NavGPT2-style init: ALL agents share `ep.start_position`. Agent 0
        # uses `ep.start_rotation` if present; agents 1+ use random yaw quats
        # so they immediately diverge in exploration direction.
        ep_pos = np.asarray(ep.start_position, dtype=np.float32)
        starts = [ep_pos.copy() for _ in range(n)]
        rng_init = np.random.default_rng(seed)
        rotations: List[List[float]] = []
        for i in range(n):
            if i == 0 and hasattr(ep, "start_rotation") and ep.start_rotation is not None:
                rotations.append([float(x) for x in ep.start_rotation])
            else:
                alpha = float(rng_init.uniform(0.0, 2.0 * math.pi))
                # quaternion [x, y, z, w] for yaw α around +y axis
                rotations.append([0.0, math.sin(alpha / 2.0), 0.0, math.cos(alpha / 2.0)])

        sim.reset()
        for i in range(n):
            sim.set_agent_state(
                position=[float(x) for x in starts[i]],
                rotation=[float(x) for x in rotations[i]],
                agent_id=i,
            )
        obs = sim._per_agent_obs()
        # Occupancy canvas covers the whole pathfinder house, not a 24 m crop.
        if self._viz is not None:
            from vlnce_baselines.multi.ma_visualizer import occupancy_frame_from_sim
            origin, size_m = occupancy_frame_from_sim(sim)
            self._viz.begin_episode(origin_xz=origin, size_m=size_m)

        # Per-episode planner reset — MUST run AFTER `set_agent_state` so
        # `mp.states[i].world_origin_xz` and `.initial_habitat_yaw_rad`
        # capture the NEW episode's spawn pose, not the previous episode's
        # leftover state.
        if self.mp is not None:
            self.mp.begin_episode(sim)

        goal_positions = [np.asarray(g.position, dtype=np.float32) for g in ep.goals]
        states = [PerAgentState(agent_id=i, last_position=starts[i].copy()) for i in range(n)]
        for i in range(n):
            states[i].start_geo = self._min_geodesic(sim, starts[i], goal_positions)

        finite = [s.start_geo for s in states if np.isfinite(s.start_geo)]
        if not finite:
            print(f"[ma-runner][SKIP] ep {ep.episode_id}: no reachable goal")
            return self._build_result(ep, states, steps=0, elapsed=0.0)
        team_start_geo = min(finite)
        print(f"[ma-runner] starts/geo = {[(i, round(s.start_geo, 2)) for i, s in enumerate(states)]}")

        # Announcement buffer: each entry = "agent_j direction_at_step"
        announcements: List[Optional[Dict]] = [None] * n

        t0 = time.time()
        step_ref = [0]
        episode_stopped = False
        last_obs = obs

        # Panorama-time render closure: same panel as action-phase, called
        # every sim tick (including pano rotation ticks). The original
        # pano-time regression (bb19 4/4 → bb21 0/4) was driven by the
        # yaw-offset coord bug + strict-traversable FMM degeneration; with
        # both fixed, rendering during pano is safe.
        viz_hook = (
            self._per_sim_step_hook(sim=sim, ep=ep, states=states, step_ref=step_ref, n=n)
            if self._viz is not None else None
        )

        def pano_tick_hook(last_obs_tick, actions=None):
            self._append_history_if_due(states, last_obs_tick, int(step_ref[0]))
            if viz_hook is not None:
                viz_hook(last_obs_tick, actions)

        opening_panos, last_obs = collect_panorama_sync_all(
            sim,
            n_agents=n,
            initial_obs=last_obs,
            mapping_planner=self.mp,
            on_sim_step=pano_tick_hook,
            step_ref=step_ref,
        )
        for i, pano in enumerate(opening_panos):
            states[i].pending_pano = pano
            states[i].last_pano = pano
            states[i].last_pano_step = int(step_ref[0])
        print(f"[ma-runner] opening 360° sync spin done at step {step_ref[0]}")

        while step_ref[0] < self.max_steps and not episode_stopped:
            step = step_ref[0]
            # Decide phase: joint pano for everyone who needs one, then LA+VA.
            pending_vlm = []

            def _apply_runner_vlm_decision(item, decision) -> None:
                nonlocal episode_stopped
                i = item["agent_id"]
                pano = item["pano"]
                cur = item["cur"]
                step_i = item["step"]
                from vlnce_baselines.multi.ma_decision import (
                    Backtrack,
                    EpisodeStop,
                    NewTarget,
                )
                if isinstance(decision, EpisodeStop):
                    states[i].stopped = True
                    states[i].success = True
                    print(
                        f"[ma-runner]   agent_{i} reported {decision.reason} "
                        f"at step {step_i}"
                    )
                    episode_stopped = True
                elif isinstance(decision, NewTarget):
                    wx, wz = decision.world_xz_m
                    target_3d = np.array([wx, float(cur[1]), wz], dtype=np.float32)
                    snapped = sim.pathfinder.snap_point(target_3d)
                    snapped = np.asarray(snapped, dtype=np.float32)
                    if not np.all(np.isfinite(snapped)):
                        tgt = self._pick_random_navigable_target(sim, rng, cur)
                        states[i].target_xy_world = tgt
                    else:
                        states[i].target_xy_world = snapped
                    prev_desc = ""
                    if announcements[i] is not None:
                        prev_desc = announcements[i].get("description", "") or ""
                    announcements[i] = {
                        "direction": decision.direction,
                        "step": step_i,
                        "description": decision.description,
                        "progress_analysis": decision.progress_analysis,
                        "prev_description": prev_desc,
                    }
                    states[i].last_direction = decision.direction
                    states[i].last_direction_step = step_i
                    states[i].last_va_bbox = decision.bbox
                    if decision.bbox is not None:
                        idx = max(0, min(int(decision.bbox_frame_idx or 0), len(pano) - 1))
                        states[i].va_bbox_flash = decision.bbox
                        states[i].va_bbox_flash_rgb = pano[idx].get("rgb")
                    states[i].target_cell_rc = decision.mapping_cell_rc
                    states[i].going_to_stop = bool(decision.stop_approach)
                    states[i].pending_claim_goal_id = decision.claim_goal_id
                    if self._viz is not None and decision.bbox is not None:
                        try:
                            idx = max(0, min(decision.bbox_frame_idx, len(pano) - 1))
                            fwd_for_bbox = [last_obs[j].get("rgb") for j in range(n)]
                            fwd_for_bbox[i] = pano[idx]["rgb"]
                            bbox_for_save = [states[j].last_va_bbox for j in range(n)]
                            bbox_for_save[i] = decision.bbox
                            self._viz.save_bbox_frame(
                                step=step_i,
                                episode_id=str(ep.episode_id),
                                forward_rgbs=fwd_for_bbox,
                                va_bbox_per_agent=bbox_for_save,
                            )
                        except Exception as e:
                            print(f"[ma-runner][viz-warn] bbox save failed: {e!r}")
                elif isinstance(decision, Backtrack):
                    if self._apply_backtrack(sim, i, states[i], decision.waypoint_id):
                        announcements[i] = {
                            "direction": "backtrack",
                            "description": f"backtrack to waypoint {decision.waypoint_id}",
                            "step": step_i,
                        }
                    else:
                        tgt = self._pick_random_navigable_target(sim, rng, cur)
                        states[i].target_xy_world = tgt
                        announcements[i] = {"direction": "fallback", "step": step_i}
                else:
                    tgt = self._pick_random_navigable_target(sim, rng, cur)
                    states[i].target_xy_world = tgt
                    announcements[i] = {"direction": "fallback", "step": step_i}
                states[i].needs_decision = False
                states[i].holding_waypoint = False
                states[i].last_tg_d = float("inf")
                states[i].last_map_d = float("inf")

            spin_ids = [
                i for i in range(n)
                if (not states[i].stopped)
                and states[i].needs_decision
                and getattr(states[i], "pending_pano", None) is None
            ]
            if spin_ids:
                spin_panos, last_obs = collect_panorama_sync_subset(
                    sim,
                    spin_ids,
                    n,
                    other_action_fn=lambda j: self._next_action_for(sim, states[j]),
                    initial_obs=last_obs,
                    mapping_planner=self.mp,
                    on_sim_step=pano_tick_hook,
                    step_ref=step_ref,
                )
                for i, pano in zip(spin_ids, spin_panos):
                    states[i].pending_pano = pano
                    states[i].last_pano = pano
                    states[i].last_pano_step = int(step_ref[0])
                if len(spin_ids) > 1:
                    print(
                        f"[ma-vlm] joint pano n={len(spin_ids)} at step {step_ref[0]}"
                    )

            for i in range(n):
                if states[i].stopped or not states[i].needs_decision:
                    continue
                if getattr(states[i], "pending_pano", None) is not None:
                    pano = states[i].pending_pano
                    states[i].pending_pano = None
                else:
                    # Async panorama collect (12 sim steps). Other agents keep
                    # moving. step_ref is incremented per tick by the helper so
                    # filenames combined_step{N}.jpg form a continuous sequence.
                    pano, last_obs = collect_panorama_async(
                        sim,
                        agent_i=i,
                        n_agents=n,
                        other_action_fn=lambda j: self._next_action_for(sim, states[j]),
                        initial_obs=last_obs,
                        mapping_planner=self.mp,
                        on_sim_step=pano_tick_hook,
                        step_ref=step_ref,
                    )
                step = step_ref[0]

                cur = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                if self.use_vlm:
                    # Build inter-agent announcement text from `announcements[1-i]` ...[N].
                    # Each line tells THIS agent "what teammate j has been doing
                    # and what it plans to do next" (richer than just direction).
                    other_announces = []
                    for j in range(n):
                        if j == i or announcements[j] is None:
                            continue
                        ann = announcements[j]
                        if step - ann["step"] > self.announcement_window_steps:
                            continue
                        chunks = [f"agent_{j} ({step - ann['step']} steps ago):"]
                        prev = ann.get("prev_description")
                        if prev:
                            chunks.append(f"just navigated toward '{prev}',")
                        nxt_dir = ann.get("direction", "forward")
                        nxt_desc = ann.get("description") or ""
                        if nxt_desc:
                            chunks.append(f"now heading {nxt_dir} toward '{nxt_desc}'")
                        else:
                            chunks.append(f"now heading {nxt_dir}")
                        prog = ann.get("progress_analysis") or ""
                        if prog:
                            # Cap to ~120 chars to keep LA prompt compact.
                            chunks.append(f"(status: {prog[:120]})")
                        other_announces.append(" ".join(chunks))
                    ann_text = " | ".join(other_announces)
                    instruction = f"Find a {ep.object_category}."
                    job, snap = self._prepare_la_va_job(
                        sim, i, states[i], pano,
                        instruction=instruction,
                        announcement_text=ann_text,
                        step=step,
                        episode_id=str(ep.episode_id),
                    )
                    pending_vlm.append({
                        "agent_id": i,
                        "pano": pano,
                        "cur": cur,
                        "step": step,
                        "job": job,
                        "snap": snap,
                    })
                    continue
                tgt = self._pick_random_navigable_target(sim, rng, cur)
                states[i].target_xy_world = tgt
                announcements[i] = {"direction": "random", "step": step}
                states[i].needs_decision = False
                states[i].holding_waypoint = False
                states[i].last_tg_d = float("inf")
                states[i].last_map_d = float("inf")
                if step >= self.max_steps:
                    break

            if pending_vlm:
                raws = self._run_la_va_wave([item["job"] for item in pending_vlm])
                for item, raw in zip(pending_vlm, raws):
                    try:
                        decision = self._finish_la_va(
                            sim, item["agent_id"], raw, item["snap"], item["pano"],
                        )
                    except Exception as e:
                        print(f"[ma-runner][LA-warn] agent_{item['agent_id']}: {e!r}")
                        decision = None
                    _apply_runner_vlm_decision(item, decision)
                    if episode_stopped:
                        break
            step = step_ref[0]

            if step >= self.max_steps:
                break

            # Action phase. Default STOP so a skipped agent does not walk.
            actions: List[int] = [HabitatSimActions.STOP] * n
            for i in range(n):
                if states[i].stopped:
                    continue
                a = self._next_action_for(sim, states[i])
                actions[i] = a

            # Keep STOP as a hold. Remapping it used to walk/turn past the waypoint.
            step_actions = list(actions)
            for i in range(n):
                states[i].last_action = step_actions[i]
            last_obs = sim.step(step_actions)
            step_ref[0] += 1
            step = step_ref[0]

            # Update each agent's world-position trail (for viz overlay).
            for i in range(n):
                if states[i].stopped:
                    continue
                from vlnce_baselines.multi.ma_visualizer import snapshot_world_xyz
                states[i].position_trail.append(
                    snapshot_world_xyz(sim.get_agent_state(i).position)
                )

            # Feed all agents' obs into lavira's Semantic_Mapping (one update tick per sim step).
            if self.mp is not None:
                try:
                    self.mp.step(sim, list(last_obs))
                except Exception as e:
                    import traceback
                    print(f"[ma-runner][mp-warn] mapping step failed at {step}: {e!r}")
                    traceback.print_exc()

            # Feed each non-stopped agent's depth + pose into the shared
            # incremental occupancy map (lavira-style, built as we walk).
            if self._viz is not None:
                from vlnce_baselines.multi.ma_decision import yaw_from_quat as _yaw
                for i in range(n):
                    if states[i].stopped:
                        continue
                    try:
                        pos = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                        yaw = _yaw(sim.get_agent_state(i).rotation)
                        d = last_obs[i].get("depth")
                        if d is None:
                            continue
                        self._viz.ingest(depth_norm=d, agent_pos=pos, agent_yaw_rad=yaw)
                    except Exception as e:
                        print(f"[ma-runner][viz-warn] ingest failed agent_{i} step {step}: {e!r}")

            # Render the combined panel every viz_every sim steps.
            if self._viz is not None and (step % max(1, self.viz_every) == 0):
                from vlnce_baselines.multi.ma_decision import yaw_from_quat as _yaw
                try:
                    positions = [np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                                 for i in range(n)]
                    yaws = [_yaw(sim.get_agent_state(i).rotation) for i in range(n)]
                    forward_rgbs = [last_obs[i].get("rgb") for i in range(n)]
                    forward_rgbs, va_boxes = consume_va_bbox_overlay(states, forward_rgbs)
                    # waypoints overlay: each agent's recorded world_coords trail
                    waypoints_per_agent = []
                    for i in range(n):
                        wps = []
                        for wp in states[i].waypoint_history:
                            wc = wp.get("world_coords")
                            if wc and len(wc) >= 2:
                                wps.append(np.array([wc[0], 0.0, wc[1]], dtype=np.float32))
                        waypoints_per_agent.append(wps)
                    # Single merged habitat-frame map: union of both agents'
                    # explored + obstacle, both agents' arrows / trails /
                    # targets, plus the episode goal.
                    lavira_maps_per_agent = None
                    lavira_map = None
                    if self.mp is not None:
                        try:
                            lavira_map = self.mp.render_merged_habitat_map(
                                sim,
                                agent_paths_world=[states[i].position_trail for i in range(n)],
                                agent_targets_world=[states[i].target_xy_world for i in range(n)],
                                goals_world=[np.asarray(g.position, dtype=np.float32)
                                             for g in getattr(ep, "goals", [])],
                            )
                        except Exception:
                            lavira_map = None
                    self._viz.render_step(
                        step=step,
                        agent_positions=positions,
                        agent_yaws=yaws,
                        forward_rgbs=forward_rgbs,
                        goal_name=getattr(ep, "object_category", ""),
                        episode_id=str(ep.episode_id),
                        instruction=f"Find a {getattr(ep, 'object_category', '')}.",
                        agent_paths=[states[i].position_trail for i in range(n)],
                        waypoints_per_agent=waypoints_per_agent,
                        goals_world=[np.asarray(g.position, dtype=np.float32)
                                     for g in getattr(ep, "goals", [])],
                        per_agent_target_world=[states[i].target_xy_world for i in range(n)],
                        action_per_agent=[states[i].last_action for i in range(n)],
                        la_direction_per_agent=[
                            (states[i].last_direction if states[i].last_direction else None)
                            for i in range(n)
                        ],
                        la_reasoning_per_agent=[states[i].last_la_reasoning for i in range(n)],
                        todo_list=None,  # P1: TODO off; pass shared TODO once enabled
                        lavira_map_bgr=lavira_map,
                        per_agent_lavira_maps=lavira_maps_per_agent,
                        va_bbox_per_agent=va_boxes,
                    )
                except Exception as e:
                    print(f"[ma-runner][viz-warn] render_step failed at {step}: {e!r}")

            self._append_history_if_due(states, last_obs, step)

            # Post-step bookkeeping per agent
            for i in range(n):
                if states[i].stopped:
                    continue
                s = states[i]
                cur = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                disp = float(np.linalg.norm(cur - s.last_position))
                s.path_length += disp
                s.last_position = cur
                # Oracle stop on any goal
                d = self._min_geodesic(sim, cur, goal_positions)
                if d <= self.success_distance:
                    s.stopped = True
                    s.success = True
                    print(f"[ma-runner]   agent_{i} reached goal at step {step}")
                    episode_stopped = True
                    break
                # Top-down map Euclidean to the current VA waypoint (LaViRA
                # 0.75 m). Do not use Habitat navmesh geodesic here.
                made_progress = False
                if (
                    self.mp is not None
                    and s.target_cell_rc is not None
                ):
                    map_d = self.mp.map_euclidean_m(i, s.target_cell_rc)
                    if map_d is not None:
                        if map_d < s.last_map_d - 0.05:
                            made_progress = True
                        s.last_map_d = map_d
                        if self.mp.waypoint_reached_on_map(i, s.target_cell_rc):
                            s.needs_decision = True
                # Progress-based stuck detection: count a tick when the agent
                # barely moved, OR when it issued MOVE_FORWARD yet got no closer
                # to the sub-goal (grinding into a wall). Lateral sliding no
                # longer resets the counter — pure-displacement detection used
                # to miss wall-grinding because sliding registers >5cm motion.
                if disp < 0.05:
                    s.consec_stuck += 1
                elif s.last_action == HabitatSimActions.MOVE_FORWARD and not made_progress:
                    s.consec_stuck += 1
                else:
                    s.consec_stuck = 0
                # ~4 wasted FWD attempts (~1m into a wall) → ask LA/VA for a new
                # target instead of grinding. Snappy without being trigger-happy.
                if s.consec_stuck >= 4:
                    s.needs_decision = True
                    s.consec_stuck = 0

            if step % self.log_every == 0:
                dists = [round(self._min_geodesic(sim, np.asarray(sim.get_agent_state(i).position),
                                                  goal_positions), 2) for i in range(n)]
                print(f"[ma-runner]   step {step:3d} dists={dists}")

        elapsed = time.time() - t0
        result = self._build_result(ep, states, steps=step, elapsed=elapsed,
                                    team_start_geo=team_start_geo)
        return result

    def _mark_waypoint_hold(self, state: PerAgentState, reason: str) -> int:
        """Hold on the current X and wait for the next LA/VA cycle."""
        if not state.holding_waypoint:
            print(f"[ma-nav] agent_{state.agent_id} hold at waypoint ({reason})")
        state.holding_waypoint = True
        state.needs_decision = True
        state.geom_replan_only = False
        return HabitatSimActions.STOP

    def _waypoint_visually_reached(self, sim, state: PerAgentState) -> bool:
        """Arrival matches the drawn X (world XZ) and/or the mapping cell."""
        if (
            state.target_cell_rc is not None
            and self.mp is not None
            and self.mp.waypoint_reached_on_map(state.agent_id, state.target_cell_rc)
        ):
            return True
        if state.target_xy_world is None:
            return False
        pos = sim.get_agent_state(state.agent_id).position
        return world_xz_reached(pos, state.target_xy_world, WAYPOINT_REACH_M)

    def _next_action_for(self, sim, state: PerAgentState) -> int:
        """Per-agent next action when we're in the action phase.

        If `SharedMappingPlanner` is available (lavira Semantic_Mapping + FMM),
        delegate to it; otherwise fall back to the habitat-pathfinder
        short-path action."""
        if state.stopped:
            return HabitatSimActions.STOP
        if state.target_xy_world is None:
            # No waypoint: stand still until LA/VA issues a new X.
            # Walking here used to rush past the old X during teammate pano.
            state.needs_decision = True
            state.geom_replan_only = False
            return HabitatSimActions.STOP
        if self.mp is not None:
            tgt = state.target_xy_world
            if self._waypoint_visually_reached(sim, state):
                if (
                    state.target_cell_rc is not None
                    and self.mp.waypoint_reached_on_map(
                        state.agent_id, state.target_cell_rc
                    )
                ):
                    map_d = self.mp.map_euclidean_m(
                        state.agent_id, state.target_cell_rc
                    )
                    reason = f"map_d={map_d:.2f}m < {WAYPOINT_REACH_M:.2f}"
                else:
                    pos = sim.get_agent_state(state.agent_id).position
                    world_d = math.hypot(
                        float(pos[0]) - float(tgt[0]),
                        float(pos[2]) - float(tgt[2]),
                    )
                    reason = f"world_xz={world_d:.2f}m < {WAYPOINT_REACH_M:.2f}"
                return self._mark_waypoint_hold(state, reason)
            a = self.mp.next_action_for(
                state.agent_id,
                (float(tgt[0]), float(tgt[2])),
                sim,
                target_cell_override=state.target_cell_rc,
            )
            if a == HabitatSimActions.STOP:
                return self._mark_waypoint_hold(state, "FMM STOP")
            return a
        if self._waypoint_visually_reached(sim, state):
            return self._mark_waypoint_hold(state, "world_xz")
        a = self._next_fmm_action(sim, state.agent_id, state.target_xy_world)
        if a == -1:
            return self._mark_waypoint_hold(state, "pathfinder done")
        return a

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _build_result(self, ep, states, steps: int, elapsed: float,
                      team_start_geo: float = float("inf")) -> Dict:
        n = len(states)
        per_agent = []
        joint = 0
        for s in states:
            if s.success and s.path_length > 0 and np.isfinite(s.start_geo):
                spl = float(s.start_geo / max(s.path_length, s.start_geo))
            else:
                spl = 0.0
            per_agent.append({
                "agent_id": s.agent_id,
                "success": int(s.success),
                "spl": spl,
                "path_length": s.path_length,
                "start_geo": s.start_geo,
            })
            joint = max(joint, int(s.success))
        if joint and np.isfinite(team_start_geo):
            succ_paths = [s.path_length for s in states if s.success]
            min_path = min(succ_paths) if succ_paths else float("inf")
            team_spl = float(team_start_geo / max(min_path, team_start_geo))
        else:
            team_spl = 0.0
        return {
            "episode_id": str(ep.episode_id),
            "scene_id": ep.scene_id,
            "object_category": getattr(ep, "object_category", ""),
            "steps": steps,
            "elapsed_s": elapsed,
            "per_agent": per_agent,
            "joint_success": joint,
            "team_spl": team_spl,
            "team_start_geo": team_start_geo,
        }

    def _dump_summary(self, results):
        import json
        n_eps = len(results)
        if not n_eps:
            return
        joint_SR = float(np.mean([r["joint_success"] for r in results]))
        team_SPL = float(np.mean([r["team_spl"] for r in results]))
        n_agents = len(results[0]["per_agent"])
        per_agent_SR = [float(np.mean([r["per_agent"][i]["success"] for r in results])) for i in range(n_agents)]
        per_agent_SPL = [float(np.mean([r["per_agent"][i]["spl"] for r in results])) for i in range(n_agents)]
        summary = {
            "n_episodes": n_eps,
            "joint_SR": joint_SR,
            "team_SPL": team_SPL,
            "per_agent_SR": per_agent_SR,
            "per_agent_SPL": per_agent_SPL,
            "avg_steps": float(np.mean([r["steps"] for r in results])),
            "avg_elapsed_s": float(np.mean([r["elapsed_s"] for r in results])),
        }
        with open(self.save_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n[ma-runner] === SUMMARY ===")
        print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vlnce_baselines/config/multi_objectnav_v2.yaml")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default="logs/multi_agent_eval/ma_runner_p0")
    parser.add_argument("--use-vlm", action="store_true",
                        help="Drive each agent with VLMReasoningAgent (LA + VA via qwen3.5-27b).")
    parser.add_argument("--la-thinking-budget", type=int, default=128)
    parser.add_argument("--no-continuous-history", action="store_true",
                        help="Disable rolling RGB-frame history (default: enabled).")
    parser.add_argument("--history-interval", type=int, default=2,
                        help="Sample 1 RGB frame every N sim steps into history.")
    parser.add_argument("--visualize", action="store_true",
                        help="Write Co-NavGPT2-style multi-agent combined PNG per step.")
    parser.add_argument("--viz-every", type=int, default=1,
                        help="Render the multi-agent panel every N action-phase sim "
                             "steps. Panorama rotation steps (12 per waypoint per "
                             "agent) are never rendered.")
    parser.add_argument("--debug-nav", type=int, default=0,
                        help="Print FMM action-selection diagnostics for the first N "
                             "next_action_for calls of each episode (per-agent: habitat "
                             "yaw, mapping heading, agent_cell, tgt_cell, stg_cell, "
                             "target_angle, rel, chosen action).")
    parser.add_argument("--vlm-parallel-workers", type=int, default=0,
                        help="Max concurrent LA+VA HTTP calls (0 = one per ready agent).")
    parser.add_argument("--no-vlm-parallel", action="store_true",
                        help="Force sequential VLM requests.")
    args = parser.parse_args()

    # Load lavira's top-level config (contains MAP block + EVAL knobs) when
    # `--config` points to a vlnce_baselines/config/*.yaml; otherwise fall back
    # to bare habitat task config for backwards compat.
    if "vlnce_baselines/config" in args.config or args.config.endswith("multi_objectnav_v2.yaml") and args.config.startswith("vlnce_baselines"):
        from vlnce_baselines.config.default import get_config as _lavira_get_config
        full_cfg = _lavira_get_config(args.config)
        task_cfg = full_cfg.TASK_CONFIG
        # Expose SIMULATOR / DATASET at top level for the runner / planner
        full_cfg.defrost()
        full_cfg.SIMULATOR = task_cfg.SIMULATOR
        full_cfg.DATASET = task_cfg.DATASET
        full_cfg.DATASET.EPISODES_ALLOWED = []
        ds_cfg = task_cfg.DATASET.clone()
        ds_cfg.defrost()
        if hasattr(ds_cfg, "EPISODES_ALLOWED"):
            ds_cfg.EPISODES_ALLOWED = None
        ds_cfg.freeze()
        full_cfg.freeze()
        cfg = full_cfg
    else:
        cfg = get_extended_config(args.config)
        cfg.defrost()
        cfg.DATASET.EPISODES_ALLOWED = []
        ds_cfg = cfg.DATASET.clone()
        ds_cfg.defrost()
        if hasattr(ds_cfg, "EPISODES_ALLOWED"):
            ds_cfg.EPISODES_ALLOWED = None
        ds_cfg.freeze()
        cfg.freeze()
    ds = make_dataset(id_dataset=ds_cfg.TYPE, config=ds_cfg)
    eps = ds.episodes[: args.num_episodes]
    print(f"[ma-runner] loaded {len(eps)} episodes")

    runner = MultiAgentRunner(
        cfg=cfg,
        max_episode_steps=args.max_episode_steps,
        save_dir=args.save_dir,
        use_vlm=args.use_vlm,
        la_thinking_budget=args.la_thinking_budget,
        use_continuous_history=not args.no_continuous_history,
        history_interval=args.history_interval,
        visualize=args.visualize,
        viz_every=args.viz_every,
        vlm_parallel_workers=(
            1 if args.no_vlm_parallel else int(args.vlm_parallel_workers)
        ),
    )
    if args.debug_nav > 0 and runner.mp is not None:
        runner.mp.debug_nav_budget = int(args.debug_nav)
        print(f"[ma-runner] FMM nav debug ON: first {args.debug_nav} next_action_for calls per episode")
    runner.evaluate(eps, seed=args.seed)


if __name__ == "__main__":
    main()
