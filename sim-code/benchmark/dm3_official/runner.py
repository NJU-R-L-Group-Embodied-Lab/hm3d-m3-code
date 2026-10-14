"""DM3-Nav OFFICIAL algorithm core, ported to the LaViRA Habitat stack (plan v2).

Drives the official agent stack (fetched at pinned commits into
`third_party/home_robot/` by `third_party/fetch.sh`, MIT — see
`third_party/README.md`):
`SimulationMultiAgentGoatAgent` + GOAT semantic map (log-odds) + instance
memory + Detic/SuperGlue/CLIP matching + MapMerger + FMM frontier planner,
on our Habitat 0.1.7 multi-agent sim, with HM3D-M3 harness rules on top:

* per-agent starts from canonical episode `agents[]` (official code spawns all
  agents at the same pose; HM3D-M3 has distributed starts — poses are fed to
  the agents' GPS/compass in a SHARED episode frame so the official sim
  assumption of identity map registration stays valid);
* task selection, navigation and detection stay official (Detic +
  CLIP/SuperGlue instance matching + FMM frontier planner). A goal counts as found through the shared detector FOUND gate (`od_found`),
  the same gate as the other adapted baselines; official stops are logged.

Fidelity table: `benchmark/dm3_official/README.md`. Official references are
given as `file:line` comments against DM3_Nav@5037cf3.

    python -m benchmark.dm3_official.runner \
        --episodes data/datasets/ma_objectnav_ovon/val_unseen/ma_stratified_100_seed42.json \
        --iin-images-dir data/datasets/ma_objectnav_ovon/val_unseen/iin_images \
        --save-dir logs/dm3_official --limit 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import List, Optional

import numpy as np

# --- official third-party code (must precede home_robot imports) -----------
# Assembled by `bash third_party/fetch.sh` (run from sim-code/); see
# third_party/README.md. DM3_HOME_ROBOT_DIR overrides the location.
_SIM_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = Path(os.environ.get("DM3_HOME_ROBOT_DIR", str(_SIM_ROOT / "third_party" / "home_robot")))
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
# detectron2 runnable tree (compiled ops + restored data/) on the data mount
_D2 = _SIM_ROOT / "data" / "datasets" / "dm3_official" / "detectron2"
if _D2.exists() and str(_D2) not in sys.path:
    sys.path.insert(0, str(_D2.resolve()))

os.environ.setdefault(
    "DM3_SUPERGLUE_WEIGHTS_DIR",
    str((_SIM_ROOT / "data" / "datasets" / "dm3_official" / "superglue_weights").resolve()),
)
os.environ.setdefault(
    "DM3_DETIC_CHECKPOINT",
    str((_SIM_ROOT / "data" / "datasets" / "dm3_official" / "detic.pth").resolve()),
)

from vlnce_baselines.multi.ma_decision import yaw_from_quat
from vlnce_baselines.multi.ma_runner import (HabitatSimActions, MultiAgentRunner,
                                             PerAgentState)
from .. import coord_shell, metrics
from ..od_found import ODFoundGate, PROTOCOL_NAME, attach_od_metrics, claim_from_obs

# official DM3-Nav / home-robot imports
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.agent.goat_agent.multiagent_goat_agent import SimulationMultiAgentGoatAgent

# DiscreteNavigationAction -> habitat 0.1.7 sim action
_ACTION_MAP = {
    DiscreteNavigationAction.STOP: HabitatSimActions.STOP,
    DiscreteNavigationAction.MOVE_FORWARD: HabitatSimActions.MOVE_FORWARD,
    DiscreteNavigationAction.TURN_LEFT: HabitatSimActions.TURN_LEFT,
    DiscreteNavigationAction.TURN_RIGHT: HabitatSimActions.TURN_RIGHT,
}

MIN_DEPTH_REPLACEMENT_VALUE = 10000   # home_robot/utils/constants.py


# --- official DM3-Nav config ------------------------------------------------
# Official DM3-Nav config, reconstructed as a plain namespace.
#
# Values copied 1:1 from the official DM3_Nav@5037cf3 files:
#   - projects/habitat_goat/configs/agent/hm3d_eval_new.yaml  (AGENT.*, SUPERGLUE.*)
#   - habitat-lab config benchmark/nav/goat/multiagent_goat_hm3d_rgbd_with_semantic.yaml
#     (simulator: turn_angle 30, sensors 640x480 hfov79 @ 0.88 m, depth 0.0-5.0
#     [min_depth forced to 0.0 by multiagent_eval_episode.read_configs],
#     habitat.environment.max_episode_steps=2500)
#
# Only what the official agent stack (sim-code/third_party/home_robot) actually reads is provided.
def build_dm3_config(*, dump_location: str, exp_name: str, num_agents: int = 2,
                     max_steps: int = 500, no_gpu: int = 0,
                     visualization_level: int = 0) -> NS:
    agent = NS(
        max_steps=max_steps,                 # AGENT.max_steps=500
        panorama_start=1,                    # 360-degree spin at episode start
        exploration_strategy="frontier",
        frontier_metric="distance",
        exploration_type="raycast",
        radius=0.05,
        cat_match_threshold=0.4,
        search_found_goal_freq=10,
        COMMUNICATION=NS(radius=5, cooldown=10, location_valid=3),
        SEMANTIC_MAP=NS(
            semantic_categories="dataset",
            map_size_cm=4800,
            map_resolution=5,
            vision_range=100,
            global_downscaling=2,
            du_scale=1,
            cat_pred_threshold=5.0,
            exp_pred_threshold=1.0,
            map_pred_threshold=1.0,
            been_close_to_radius=200,
            target_blacklisting_radius=100,
            min_obs_height_cm=20,
            goal_filtering=True,
        ),
        PLANNER=NS(
            collision_threshold=0.10,
            min_obs_dilation_selem_radius=2,
            obs_dilation_selem_radius=3,
            map_downsample_factor=1,
            map_update_frequency=1,
            step_size=5,
            discrete_actions=True,
            stop_distance=100,
        ),
        SUPERGLUE=NS(
            max_keypoints=1024,
            keypoint_threshold=0.005,
            nms_radius=4,
            superglue_model="indoor",
            sinkhorn_iterations=20,
            match_threshold=0.2,
            score_function="confidence_sum",
            score_thresh_image=24.5,
            score_thresh_lang=0.24,
            match_projection_threshold=0.2,
            goto_past_pose=True,
        ),
    )
    depth_sensor = NS(
        width=640, height=480, hfov=79,
        min_depth=0.0, max_depth=5.0, position=[0, 0.88, 0],
    )
    habitat = NS(
        task=NS(type="MultiAgentGoat-v1"),
        simulator=NS(
            turn_angle=30,
            agents=NS(agent0=NS(sim_sensors=NS(depth_sensor=depth_sensor))),
        ),
        environment=NS(max_episode_steps=2500),
    )
    return NS(
        NO_GPU=no_gpu,
        DUMP_LOCATION=dump_location,
        EXP_NAME=exp_name,
        REAL_WORLD=False,
        NUM_AGENTS=num_agents,
        SEQ=0,
        DATASET="goat",
        VISUALIZATION_LEVEL=visualization_level,
        GROUND_TRUTH_SEMANTICS=0,
        USE_YOLO=0,
        ENVIRONMENT=NS(max_subtasks_per_episode=10, turn_angle=30),
        AGENT=agent,
        habitat=habitat,
    )
MAX_DEPTH_REPLACEMENT_VALUE = 10001


def resolve_iin_path(path: Optional[str], images_dir: Optional[str] = None) -> Optional[str]:
    """Resolve a manifest image path against cwd or `--iin-images-dir`."""
    if not path:
        return None
    candidates = [path]
    if images_dir:
        candidates.append(os.path.join(images_dir, os.path.basename(path)))
        if not os.path.isabs(path):
            candidates.append(os.path.join(images_dir, path))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


class DM3OfficialAgent(SimulationMultiAgentGoatAgent):
    """Official agent; records the last action sent to the simulator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_action_out = None

    def _search_for_goal(self):  # noqa: C901 - mirrors official control flow
        import torch as _torch

        with _torch.no_grad():
            for i, task in enumerate(self.tasks):
                if self.tasks_done[i] or self.tasks_failed[i]:
                    continue
                if self.inst_goals[i] is None or self.total_timesteps % self.search_found_goal_freq == 0:
                    inst_goal_id, best_score = self.matching.search_for_goal(
                        task,
                        self.match_memory,
                        self.semantic_map.global_pose,
                        score_thresh=0 if self.navigate_to_best else None,
                    )
                    if inst_goal_id is not None:
                        # official keeps `or True` (multiagent_goat_agent.py:131)
                        if self.inst_goals[i] is None or self.inst_goals[i]["score"] < best_score or True:
                            self.log.info(
                                f"Found (better) instance goal for task {i} with score {best_score}"
                            )
                            self.inst_goals[i] = {"id": inst_goal_id, "score": best_score}
                else:
                    self.log.debug(
                        f"Instance {self.inst_goals[i]} already found for goal {i}"
                    )
            self.match_memory = False


class DM3OfficialRunner(MultiAgentRunner):
    """HM3D-M3 runner around the official DM3 agent stack."""

    iin_lookup: dict = {}
    iin_images_dir: str = "data/datasets/ma_objectnav_ovon/val_unseen/iin_images"
    _vocab: Optional[List[str]] = None
    _dm3_config = None

    # ------------------------------------------------------------------ #
    # Official vocabulary: sorted unique dataset categories
    # (habitat_goat_env.py:72-84 fetch_vocabulary over dataset.all_categories)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_vocabulary(episodes) -> List[str]:
        cats = set()
        for ep in episodes:
            for g in ep.goals_set:
                cats.add("_".join(g["object_category"].split(" ")))
        return sorted(cats)

    # ------------------------------------------------------------------ #
    # Observations adapter: lavira sim -> home_robot Observations
    # (habitat_goat_env.py:_preprocess_obs + fork EpisodicGPS/CompassSensor,
    #  but in a SHARED episode frame so identity map-merge stays valid)
    # ------------------------------------------------------------------ #
    def _frame_from_episode(self, ep):
        import quaternion  # numpy-quaternion

        starts = [np.asarray(a["start_position"], dtype=np.float64) for a in ep.agents]
        origin = np.mean(starts, axis=0)
        # canonical start_rotation is xyzw; np.quaternion takes (w, x, y, z)
        r = [float(x) for x in ep.agents[0]["start_rotation"]]
        q_frame = np.quaternion(r[3], r[0], r[1], r[2])
        return origin, q_frame

    def _pose_to_gps_compass(self, position, rotation_xyzw, origin, q_frame):
        """fork GPSSensor/CompassSensor formulas, shared frame.

        nav.py: EpisodicGPSSensor.get_observation / EpisodicCompassSensor +
        habitat_abstract_env.py:_preprocess_xy ([x, -y]).
        """
        import quaternion
        from habitat_sim.utils.common import quat_rotate_vector

        q_agent = np.quaternion(float(rotation_xyzw[3]), float(rotation_xyzw[0]),
                                float(rotation_xyzw[1]), float(rotation_xyzw[2]))
        rel = quat_rotate_vector(q_frame.inverse(), np.asarray(position, dtype=np.float64) - origin)
        gps_h = np.array([-rel[2], rel[0]], dtype=np.float32)          # fork gps 2D
        gps = np.array([gps_h[0], -gps_h[1]])                           # _preprocess_xy
        hv = quat_rotate_vector(q_agent.inverse() * q_frame, np.array([0, 0, -1]))
        # fork: cartesian_to_polar(-hv[2], hv[0])[1] == atan2(hv[0], -hv[2])
        compass = np.array([math.atan2(hv[0], -hv[2])], dtype=np.float32)
        return gps, compass

    def _depth_to_home_robot(self, depth_norm: np.ndarray) -> np.ndarray:
        """lavira habitat-lab 0.1.7 depth obs are normalized [0,1] over
        [self._dmin, self._dmax] (0.1/5.0 m). DM3's habitat_goat_env
        ._preprocess_depth (min=0/max=5) then applies: metric, with
        normalized==0 -> MIN sentinel, normalized>=1 -> MAX sentinel."""
        d = np.asarray(depth_norm, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        d_m = self._dmin + d * (self._dmax - self._dmin)   # metric
        out = np.clip(d_m, 0.0, 5.0)
        out[d <= 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        out[d >= 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return out

    def _build_tasks(self, ep) -> List[dict]:
        """task_observations["tasks"] — multigoal sensor equivalent
        (goat_task.py:MultiGoalSensor.get_observation); order = goal_index."""
        vocab = self._vocab
        tasks = []
        for g in sorted(ep.goals_set, key=lambda x: x["goal_index"]):
            cat = "_".join(g["object_category"].split(" "))
            semantic_id = vocab.index(cat) + 1       # GoatCategories: 1-indexed
            mod = g["modality"]
            if mod == "IIN":
                path = resolve_iin_path(
                    self.iin_lookup.get((str(ep.episode_id), g["goal_index"])),
                    self.iin_images_dir,
                )
                img = None
                if path and os.path.exists(path):
                    from PIL import Image
                    img = np.asarray(Image.open(path).convert("RGB"))
                elif path:
                    print(f"[dm3-official][warn] missing IIN image ep={ep.episode_id} "
                          f"goal={g['goal_index']}: {path}")
                tasks.append({"category": g["object_category"], "semantic_id": semantic_id,
                              "type": "imagenav", "image": img, "description": None})
            elif mod == "TN":
                tasks.append({"category": g["object_category"], "semantic_id": semantic_id,
                              "type": "languagenav", "image": None,
                              "description": g.get("tn_text")})
            else:
                tasks.append({"category": g["object_category"], "semantic_id": semantic_id,
                              "type": "objectnav", "image": None, "description": None})
        return tasks

    def _make_obs(self, sim, i, tasks, origin, q_frame):
        from home_robot.core.interfaces import Observations

        agent_state = sim.get_agent_state(i)
        obs_i = self._last_obs[i]
        gps, compass = self._pose_to_gps_compass(
            agent_state.position,
            [agent_state.rotation.x, agent_state.rotation.y,
             agent_state.rotation.z, agent_state.rotation.w],
            origin, q_frame,
        )
        return Observations(
            rgb=np.asarray(obs_i["rgb"])[..., :3].astype(np.uint8),
            depth=self._depth_to_home_robot(obs_i["depth"]),
            compass=compass,
            gps=gps,
            task_observations={"tasks": tasks, "top_down_map": None},
            camera_pose=None,
            third_person_image=None,
        )

    # ------------------------------------------------------------------ #
    # Episode loop — port of multiagent_eval_episode.py __main__
    # ------------------------------------------------------------------ #
    def _run_episode(self, sim, ep, seed: int = 0):
        n = self.n_agents
        goals = sorted(ep.goals_set, key=lambda g: g["goal_index"])
        M = len(goals)
        self._cur_ep_id = str(ep.episode_id)
        t0 = time.time()
        official_claims: List[dict] = []

        cfg_n = int(getattr(self._dm3_config, "NUM_AGENTS", -1)) if self._dm3_config is not None else -1
        if self._dm3_config is None or cfg_n != n:
            self._dm3_config = build_dm3_config(
                dump_location=str(self.save_dir / "dm3_dump") if self.save_dir else "datadump",
                exp_name="m3",
                num_agents=n,
                max_steps=self.max_steps,
            )
            print(f"[dm3-official] rebuilt official config NUM_AGENTS={n}", flush=True)
        # official main() initializes the home_robot root logger before agents
        # (multiagent_eval_episode.py:111); agent loggers derive from it.
        from home_robot.utils.logger import get_logger as _hr_get_logger
        _hr_run_dir = os.path.join(
            str(self.save_dir) if self.save_dir else "logs", "dm3_dump", "runlogs")
        os.makedirs(_hr_run_dir, exist_ok=True)
        _hr_get_logger(run_dir=_hr_run_dir)

        # per-agent starts (HM3D-M3 contract; official spawn patched away)
        starts = [np.asarray(a["start_position"], np.float32) for a in ep.agents]
        sim.reset()
        for i in range(n):
            rot = ep.agents[i].get("start_rotation") or [0, 0, 0, 1]
            sim.set_agent_state(position=[float(x) for x in starts[i]],
                                rotation=[float(x) for x in rot], agent_id=i)
        self._last_obs = sim._per_agent_obs()
        if self.mp is not None:
            self.mp.begin_episode(sim)

        origin, q_frame = self._frame_from_episode(ep)
        tasks = self._build_tasks(ep)

        agents: List[DM3OfficialAgent] = []
        for i in range(n):
            a = DM3OfficialAgent(self._dm3_config, self._vocab, agent_id=i)
            agents.append(a)

        states = [PerAgentState(agent_id=i, last_position=starts[i].copy()) for i in range(n)]
        found = [False] * M
        found_by = [-1] * M
        idle = [False] * n
        step_ref = [0]
        log_traj = getattr(self, "log_traj", False)
        from .. import traj_log
        gxz = traj_log.goal_xz(goals)
        traj, found_events = traj_log.init_traj(starts, n) if log_traj else (None, None)
        ch = coord_shell.CoordChannel(congestion_radius=1.0)
        od_gate = ODFoundGate.from_runner(self, n, goals, ep.episode_id)

        for a in agents:
            a.reset(str(ep.scene_id), str(ep.episode_id))

        def positions():
            return [np.asarray(sim.get_agent_state(i).position, np.float32) for i in range(n)]

        viz_hook = (
            self._per_sim_step_hook(sim=sim, ep=ep, states=states, step_ref=step_ref, n=n)
            if self._viz is not None else None
        )

        def viz_tick(last_obs, actions=None):
            if self._viz is None:
                return
            from vlnce_baselines.multi.ma_decision import yaw_from_quat as _yaw
            self._viz_goal_name = "  ".join(
                f"{goals[j]['object_category']}[{goals[j]['modality']}]"
                f"{'+' if found[j] else '-'}" for j in range(M)
            )
            if self.mp is not None and last_obs is not None:
                try:
                    self.mp.step(sim, last_obs)
                except Exception as exc:
                    print(f"[dm3-official][viz-warn] mp.step failed: {exc!r}")
            for i in range(n):
                if states[i].stopped:
                    continue
                try:
                    pos = np.asarray(sim.get_agent_state(i).position, np.float32)
                    yaw = _yaw(sim.get_agent_state(i).rotation)
                    depth = last_obs[i].get("depth") if last_obs is not None else None
                    if depth is not None:
                        self._viz.ingest(depth_norm=depth, agent_pos=pos, agent_yaw_rad=yaw)
                except Exception as exc:
                    print(f"[dm3-official][viz-warn] ingest failed agent_{i}: {exc!r}")
            if viz_hook is not None:
                viz_hook(last_obs, actions)

        if self._viz is not None:
            self._viz.begin_episode()
            viz_tick(self._last_obs)

        def poll_od():
            if od_gate is None or self._last_obs is None:
                return
            for i in range(n):
                if states[i].stopped or i >= len(self._last_obs):
                    continue
                newly = claim_from_obs(
                    od_gate, sim, i, self._last_obs[i], step_ref[0], found, found_by,
                )
                for j in newly:
                    agent = agents[i]
                    try:
                        if agent.tasks_done is not None and j < len(agent.tasks_done):
                            agent.tasks_done[j] = True
                        if getattr(agent, "active_task", None) == j:
                            agent.active_task = None
                            agent.reset_for_next_task()
                    except Exception as exc:
                        print(f"[dm3-official][od-found-warn] {exc!r}")
                    if log_traj:
                        found_events.append([
                            step_ref[0], i, j, goals[j]["object_category"],
                            goals[j]["modality"], gxz[j][0], gxz[j][1],
                        ])
                    print(
                        f"[dm3-official][od-found] agent_{i} found goal{j} "
                        f"({goals[j]['object_category']}/{goals[j]['modality']}) "
                        f"@ step {step_ref[0]}"
                    )

        # ---- main loop (multiagent_eval_episode.py:154-214) ---------------
        while step_ref[0] < self.max_steps and not all(found):
            poll_od()
            if all(found):
                break

            # official loop updates EVERY living agent (eval_episode.py:172-185).
            for i, agent in enumerate(agents):
                if states[i].stopped:
                    continue
                agent.update_state(self._make_obs(sim, i, tasks, origin, q_frame))
                agent.update_maps()

            actions, stucks = [], []
            for i, agent in enumerate(agents):
                if states[i].stopped:
                    actions.append({"action": DiscreteNavigationAction.STOP,
                                    "action_args": {"agent_id": i, "task_idx": None}})
                    stucks.append(False)
                    continue
                others = [x for x in agents if x.agent_id != agent.agent_id
                          and not states[x.agent_id].stopped]
                agent.simulate_receive_map(others)     # comm radius 5m + cooldown 10
                action, _info, stuck = agent.act()
                actions.append(action)
                stucks.append(stuck)
                self._note_official_stops(
                    sim, agent, i, found, found_by, official_claims,
                    goals, gxz, log_traj, found_events, step_ref[0])

            if all(stucks):                            # eval_episode loop:187-192
                actions = []
                for i, agent in enumerate(agents):
                    action = agent._process_action((None, DiscreteNavigationAction.STOP))
                    agent.handle_stop(action)
                    actions.append(action)
                    self._note_official_stops(
                        sim, agent, i, found, found_by, official_claims,
                        goals, gxz, log_traj, found_events, step_ref[0])

            any_moving = not all(a["action"] == DiscreteNavigationAction.STOP for a in actions)
            for i, a in enumerate(actions):
                if a["action_args"]["task_idx"] is None and a["action"] == DiscreteNavigationAction.STOP:
                    if any_moving:
                        a["action"] = DiscreteNavigationAction.TURN_RIGHT
                    else:
                        idle[i] = True

            if all(idle[i] or states[i].stopped for i in range(n)):
                break

            step_actions = []
            for i, a in enumerate(actions):
                act = a["action"]
                if act == DiscreteNavigationAction.STOP:
                    act = HabitatSimActions.TURN_RIGHT      # sim has no STOP; agent is idle
                    idle[i] = idle[i] or a["action_args"]["task_idx"] is None
                else:
                    act = _ACTION_MAP[act]
                agents[i].last_action_out = act
                states[i].last_action = act
                step_actions.append(act)
            self._last_obs = sim.step(step_actions)
            step_ref[0] += 1
            ch.observe_positions([p.tolist() for p in positions()])
            viz_tick(self._last_obs)

            for i in range(n):
                if states[i].stopped:
                    continue
                s = states[i]
                cur = np.asarray(sim.get_agent_state(i).position, np.float32)
                s.path_length += float(np.linalg.norm(cur - s.last_position))
                s.last_position = cur
                if log_traj:
                    traj[i].append([float(cur[0]), float(cur[2])])

        # ---- scoring -------------------------------------------------------
        scored_found_by = found_by
        scored_found = [o >= 0 for o in scored_found_by]
        per_agent = [{"path_length": round(states[i].path_length, 3),
                      "goals_found": [j for j in range(M) if scored_found_by[j] == i]}
                     for i in range(n)]
        if log_traj and self.save_dir:
            traj_log.dump_traj(self.save_dir, ep, n, starts, goals, found_by,
                               found_events, step_ref[0], traj, gxz)
        m = metrics.team_metrics(ep._raw, {"per_agent": per_agent, **ch.summary()})
        m["protocol"] = PROTOCOL_NAME
        m["policy_claim_raw"] = int(sum(scored_found))
        m["official_match_stops"] = len(official_claims)
        print(f"[dm3-official] ep {ep.episode_id} found={sum(scored_found)}/{M} "
              f"progress={m['progress']} steps={step_ref[0]} "
              f"elapsed={time.time()-t0:.0f}s")
        pa = [{"agent_id": i, "success": int(len(per_agent[i]["goals_found"]) > 0),
               "spl": 0.0, "path_length": per_agent[i]["path_length"],
               "goals_found": per_agent[i]["goals_found"]} for i in range(n)]
        result = {"episode_id": str(ep.episode_id), "scene_id": ep.scene_id,
                  "object_category": ep.object_category, "steps": step_ref[0],
                  "elapsed_s": time.time() - t0, "per_agent": pa,
                  "joint_success": int(all(scored_found)), "team_spl": m["mspl"],
                  "progress": m["progress"], "metrics": m}
        attach_od_metrics("dm3-official", od_gate, m, result)
        result.update({
            "protocol": PROTOCOL_NAME,
            "protocol_snapshot": {
                "policy_proposal": "ovon_od_owl_clip_superglue",
                "four_view_double_check": False,
                "gt_viewpoints": False,
                "evaluator_gt_found_gate": False,
                "official_match_stops_logged": True,
            },
            "official_claims": official_claims,
        })
        return result

    def _note_official_stops(self, sim, agent, agent_id, found, found_by,
                             official_claims, goals, gxz, log_traj, found_events, step):
        """Snapshot the first official handle_stop for each goal (agent-side done)."""
        if not agent.tasks_done:
            return
        pos = np.asarray(sim.get_agent_state(agent_id).position, np.float32)
        for j, done in enumerate(agent.tasks_done):
            if not done:
                continue
            if any(int(c.get("goal_index", -1)) == int(j) and int(c.get("agent_id", -1)) == int(agent_id)
                   for c in official_claims):
                continue
            official_claims.append({
                "agent_id": int(agent_id),
                "goal_index": int(j),
                "step": int(step),
                "claim_position": [float(x) for x in pos],
            })
            if log_traj:
                found_events.append([
                    int(step), int(agent_id), int(j), goals[j]["object_category"],
                    goals[j]["modality"], gxz[j][0], gxz[j][1],
                ])

    @staticmethod
    def _min_reach_geo(sim, position, reach_pts) -> float:
        best = float("inf")
        for point in reach_pts:
            d = sim.geodesic_distance(position, point)
            if d is not None and np.isfinite(d):
                best = min(best, float(d))
        return best

    def _score_official_gt_viewpoint(self, sim, official_claims, n_goals, reach):
        """Official MultiAgentGOATSuccess: geodesic(current, GT view_points) < 1.

        HM3D-M3 analog of GOAT view_points is episode reach_points. Compare is
        strict `<` to match habitat/tasks/nav/nav.py:1858.
        """
        owners = [-1] * n_goals
        for rec in official_claims:
            j = int(rec["goal_index"])
            pos = np.asarray(rec["claim_position"], np.float32)
            dist = self._min_reach_geo(sim, pos, reach[j])
            rec["evaluator_distance_m"] = (
                round(float(dist), 4) if np.isfinite(dist) else None)
            rec["evaluator_valid"] = bool(np.isfinite(dist) and dist < 1.0)
            if rec["evaluator_valid"] and owners[j] < 0:
                owners[j] = int(rec["agent_id"])
        return owners

    # ------------------------------------------------------------------ #
    def _dump_summary(self, results):
        agg = metrics.aggregate([r["metrics"] for r in results if "metrics" in r])
        print("\n[dm3-official] === AGGREGATE ===")
        print(json.dumps(agg, indent=2))
        if self.save_dir:
            with open(self.save_dir / "aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)
            rows = []
            for r in results:
                if "metrics" not in r:
                    continue
                row = dict(r["metrics"])
                row.setdefault("episode_id", str(r.get("episode_id", "")))
                rows.append(row)
            with open(self.save_dir / "metrics_list.json", "w") as f:
                json.dump(rows, f)


class _NoMapCfg:
    """Proxy hiding cfg.MAP so MultiAgentRunner skips lavira's
    SharedMappingPlanner (GroundingDINO+SAM) — the port uses official maps."""

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, name):
        if name == "MAP":
            raise AttributeError(name)
        return getattr(self._cfg, name)


def main(argv=None):
    from ..ma_search import EpisodeAdapter, _load_cfg, _set_num_agents
    p = argparse.ArgumentParser(description="DM3-Nav official-core ported runner")
    p.add_argument("--episodes", required=True)
    p.add_argument("--config", default="vlnce_baselines/config/multi_objectnav_v2.yaml")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--save-dir", default="logs/dm3_official")
    p.add_argument("--iin-images-dir",
                   default="data/datasets/ma_objectnav_ovon/val_unseen/iin_images")
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--episode-ids", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--visualize", action="store_true",
                   help="Write per-step combined RGB+map frames (existing MAVisualizer).")
    p.add_argument("--log-traj", action="store_true",
                   help="Dump per-agent trajectories as traj_*.json.")
    args = p.parse_args(argv)

    data = json.load(open(args.episodes))
    eps = data["episodes"]
    if args.episode_ids:
        want = set(args.episode_ids.split(","))
        eps = [e for e in eps if str(e["episode_id"]) in want]
    elif args.limit:
        eps = eps[: args.limit]
    elif args.start or args.count:
        eps = eps[args.start:(args.start + args.count) if args.count else None]
    adapters = [EpisodeAdapter(d, args.scenes_dir) for d in eps]
    adapters.sort(key=lambda e: e.scene_id)

    raw_cfg = _load_cfg(args.config)
    _set_num_agents(raw_cfg, 2)
    # Keep cfg.MAP when visualizing so SharedMappingPlanner.render_merged_habitat_map
    # is available. Official actions still come from the official planner, not FMM.
    cfg = raw_cfg if args.visualize else _NoMapCfg(raw_cfg)
    runner = DM3OfficialRunner(cfg=cfg, max_episode_steps=args.max_episode_steps,
                               save_dir=args.save_dir, use_vlm=False,
                               use_continuous_history=False, visualize=args.visualize)
    runner.log_traj = args.log_traj
    runner.iin_images_dir = args.iin_images_dir
    runner._vocab = DM3OfficialRunner.build_vocabulary(adapters)
    manifest = os.path.join(args.iin_images_dir, "manifest.json")
    if os.path.exists(manifest):
        runner.iin_lookup = {
            (str(it["episode_id"]), it["goal_index"]): resolve_iin_path(
                it["image"], args.iin_images_dir)
            for it in json.load(open(manifest)).get("items", [])
        }
    print(f"[dm3-official] {len(adapters)} episodes, vocab={len(runner._vocab)} cats, "
          f"IIN imgs={len(runner.iin_lookup)}, od_found=True")
    runner.evaluate(adapters, seed=args.seed)


if __name__ == "__main__":
    main()
