"""Co-NavGPT2 reproduction (high-fidelity) — centralized GPT-4o frontier allocation.

Faithful to ybgdgh/Co-NavGPT2:
* **Shared semantic RGB top-view map** built by point-cloud fusion + z-buffer
  (`ConavSemanticMap` below, numpy reimpl of Global_Map_Proc).
* **Centralized VLM allocation** every num_local_steps=25: per-frontier candidate
  image = [semantic top-view with that frontier highlighted + R0/R1 + id | first-person
  frontier-direction view] (Co-NavGPT2 "full" prompt variant), sent to the unified
  **Bailian/DashScope qwen3.5-27b** VLM; response assigns each robot a frontier.
* Goal-found confirmed by perception (`dm3_perception`: detect / SIFT-IIN / CLIP-TN)
  — extends Co-NavGPT2's single-category YOLO to our multimodal benchmark.

Substitution under the container's network limits: detector = GroundingDINO+SAM /
MaskRCNN (local) in place of YOLO-World+mobile_sam (GitHub-blocked); both open-vocab.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np

from vlnce_baselines.multi.ma_runner import HabitatSimActions, MultiAgentRunner, PerAgentState
from . import metrics
from .od_found import ODFoundGate, attach_od_metrics, claim_from_obs, sensor_approach_target
from .frontier import detect_frontiers

import re

# VLM backend: one OpenAI-compatible endpoint for the whole benchmark
# (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL; the paper used qwen3.5-27b on
# Aliyun Bailian / DashScope). The "gpt4o" alias is kept for CLI back-compat but
# intentionally points at the same backend.
from benchmark.api_auth import get_llm_api_key, get_llm_base_url, get_llm_model


# --------------------------------------------------------------------------
# Shared semantic top-view map
#
# Co-NavGPT2 reproduction — shared semantic (RGB) top-view map.
#
# Faithful reimplementation (numpy, no open3d) of Co-NavGPT2's `build_full_scene_pcd`
# + `Global_Map_Proc.Map_Extraction`: back-project each agent's depth+RGB into 3-D
# world points (coloured by the camera RGB), accumulate across agents/steps, and
# project top-down with a z-buffer into:
#   * `top_view`  — RGB photometric top-down (highest point's colour wins) — the
#     "semantic" colour map the VLM reasons over,
#   * `obstacle` / `explored` — occupancy from the height band.
#
# Frame: absolute Habitat world (shared across agents, centred on agent-0's start),
# which is equivalent in content to Co-NavGPT2's init-relative Open3D frame but
# cleaner for a shared multi-agent map.
# --------------------------------------------------------------------------
def _quat_to_R(q) -> np.ndarray:
    if hasattr(q, "w"):
        w, x, y, z = q.w, q.x, q.y, q.z
    else:
        x, y, z, w = q  # [x,y,z,w]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class ConavSemanticMap:
    def __init__(self, hfov_deg: float, width: int, height: int, max_depth: float,
                 origin_xz: Tuple[float, float], map_size_cm: int = 2400,
                 resolution: int = 5, height_band_m: float = 0.5, stride: int = 4):
        self.W, self.H = int(width), int(height)
        self.fx = (self.W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx, self.cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        self.max_depth = float(max_depth)
        self.res_m = resolution / 100.0
        self.size = map_size_cm // resolution
        self.center = self.size // 2
        self.cx0, self.cz0 = float(origin_xz[0]), float(origin_xz[1])
        self.band = height_band_m
        self.stride = stride
        self.obstacle = np.zeros((self.size, self.size), np.uint8)
        self.explored = np.zeros((self.size, self.size), np.uint8)
        self.top_view = np.zeros((self.size, self.size, 3), np.uint8)
        self.zbuf = np.full((self.size, self.size), -np.inf)

    def world_to_cell(self, wx, wz):
        return (int((wx - self.cx0) / self.res_m + self.center),
                int((wz - self.cz0) / self.res_m + self.center))

    def cell_to_world(self, col, row):
        return (self.cx0 + (col - self.center) * self.res_m,
                self.cz0 + (row - self.center) * self.res_m)

    def update(self, rgb: np.ndarray, depth: np.ndarray, position, rotation_quat):
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        if float(np.nanmax(d)) <= 1.0 + 1e-6:        # normalized [0,1] -> metres
            d = d * self.max_depth
        s = self.stride
        d = d[::s, ::s]
        rgb_s = np.asarray(rgb)[::s, ::s, :3]
        hh, ww = d.shape
        us = (np.arange(0, self.W, s)[:ww]).astype(np.float32)
        vs = (np.arange(0, self.H, s)[:hh]).astype(np.float32)
        uu, vv = np.meshgrid(us, vs)
        valid = (d > 0.1) & (d < self.max_depth)
        dv = d[valid]
        if dv.size == 0:
            return
        # habitat camera frame: +x right, +y up, -z forward
        xc = (uu[valid] - self.cx) / self.fx * dv
        yc = -(vv[valid] - self.cy) / self.fy * dv
        zc = -dv
        pts_cam = np.stack([xc, yc, zc], axis=-1)                  # (N,3)
        R = _quat_to_R(rotation_quat)
        pos = np.asarray(position, dtype=np.float64)
        pts_w = pts_cam @ R.T + pos                               # camera->world
        cols = rgb_s[valid]
        cam_y = float(pos[1])

        wx, wy, wz = pts_w[:, 0], pts_w[:, 1], pts_w[:, 2]
        col = ((wx - self.cx0) / self.res_m + self.center).astype(int)
        row = ((wz - self.cz0) / self.res_m + self.center).astype(int)
        inb = (col >= 0) & (col < self.size) & (row >= 0) & (row < self.size)
        col, row, wy, cols = col[inb], row[inb], wy[inb], cols[inb]
        # explored = any point at/below ceiling; obstacle = points in height band
        self.explored[row, col] = 1
        obs = (wy >= cam_y - self.band) & (wy <= cam_y + self.band)
        self.obstacle[row[obs], col[obs]] = 1
        # RGB z-buffer (highest point colour wins)
        for r, c, y, cc in zip(row, col, wy, cols):
            if y > self.zbuf[r, c]:
                self.zbuf[r, c] = y
                self.top_view[r, c] = cc

    def maps(self):
        return self.obstacle, self.explored, self.top_view

_BAILIAN = (get_llm_base_url(), get_llm_api_key(), get_llm_model())
VLM_BACKENDS = {"qwen": _BAILIAN, "gpt4o": _BAILIAN}


def _parse_assign_json(txt):
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None
    return None

SYS_PROMPT = (
    "You are a multi-robot navigation agent with a vision-language model. Assign "
    "frontiers (exploration areas) to each robot to find the target object(s) as "
    "efficiently as possible. For each candidate frontier you are given a combined "
    "image: LEFT = the global semantic top-view map (robots are black 'R0','R1'; the "
    "candidate frontier is a thick red region with its red id number top-left), RIGHT "
    "= a first-person view facing that frontier. Decide, per robot, the best frontier "
    "balancing travel distance and the semantic relevance between the targets and the "
    'frontier view. Respond ONLY as JSON like {"robot_0":"frontier_1","robot_1":"frontier_0"} '
    "(ids may repeat if fewer frontiers than robots)."
)


def _goal_public_id(goal: dict, local_idx: int) -> str:
    return f"goal_{goal.get('goal_index', local_idx)}"


def _goal_label(goal: dict, local_idx: int) -> str:
    label = f"{_goal_public_id(goal, local_idx)} [{goal.get('modality', 'ON')}]: {goal['object_category']}"
    if goal.get("modality") == "TN" and goal.get("tn_text"):
        label += f" ({goal['tn_text']})"
    elif goal.get("modality") == "IIN":
        label += " (reference target image provided separately when available)"
    return label


class CoNavRunner(MultiAgentRunner):
    clean_found_state: bool = True
    realloc_every: int = 25
    approach_radius: float = 4.0
    iin_inlier_thresh: int = 18
    tn_clip_thresh: float = 0.24
    _percep = None
    _vlm = None
    vlm_url, vlm_key, vlm_model = VLM_BACKENDS["qwen"]   # default: Bailian qwen (debug)
    iin_lookup: dict = {}

    def _get_percep(self):
        if self._percep is None:
            from .dm3_perception import DM3Perception
            self._percep = DM3Perception()
        return self._percep

    def _get_vlm(self):
        if self._vlm is None:
            from openai import OpenAI
            self._vlm = OpenAI(api_key=self.vlm_key, base_url=self.vlm_url, timeout=120)
        return self._vlm

    def _geo(self, sim, a, b):
        d = sim.geodesic_distance(a, b)
        return float(d) if (d is not None and np.isfinite(d)) else float("inf")

    def _geo_to_reach(self, sim, pos, reach):
        return min((self._geo(sim, pos, p) for p in reach), default=float("inf"))

    # ----- candidate render (full variant: semantic map | first-person) -----
    def _render_candidate(self, smap, fr_cells, agent_cells, fi, fp_rgb):
        import cv2
        from PIL import Image
        _, _, top = smap.maps()
        img = top.copy()
        img[(top == 0).all(axis=-1)] = (60, 60, 60)             # unknown = dark grey
        cx, cy = fr_cells[fi][0], fr_cells[fi][1]
        cv2.circle(img, (int(cx), int(cy)), 7, (255, 0, 0), -1)
        for ai, (acx, acy) in enumerate(agent_cells):
            cv2.putText(img, f"R{ai}", (int(acx), int(acy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, str(fi), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        img = np.flipud(img)
        left = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(np.asarray(fp_rgb)[:, :, :3], (480, 480)) if fp_rgb is not None \
            else np.full((480, 480, 3), 200, np.uint8)
        combined = np.hstack((left, right.astype(np.uint8)))
        buf = io.BytesIO()
        Image.fromarray(combined).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _central_assign(self, smap, fr_cells, fr_world, agent_cells, agent_pos, last_obs, targets):
        n_fr = len(fr_cells)
        if n_fr == 0:
            return None
        imgs = []
        for fi in range(n_fr):
            # first-person view facing the frontier = forward RGB of the nearest agent
            ai = int(np.argmin([np.linalg.norm(np.asarray(agent_pos[a])[[0, 2]] - np.asarray(fr_world[fi]))
                                for a in range(self.n_agents)]))
            imgs.append(self._render_candidate(smap, fr_cells, agent_cells, fi, last_obs[ai].get("rgb")))
        content = [{"type": "text", "text": f"{self.n_agents} robots need to find: {', '.join(targets)}"}]
        for b in imgs:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
        msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": content}]
        try:
            r = self._get_vlm().chat.completions.create(
                model=self.vlm_model, messages=msgs, temperature=0.1, max_tokens=300)
            j = _parse_assign_json(r.choices[0].message.content)
            if not j:
                raise ValueError(f"unparseable VLM content: {r.choices[0].message.content!r}")
            return [min(max(int(str(j.get(f"robot_{i}", "frontier_0")).split("_")[-1]), 0), n_fr - 1)
                    for i in range(self.n_agents)]
        except Exception as e:  # noqa: BLE001
            print(f"[conav][vlm-warn] {e!r}")
            return [i % n_fr for i in range(self.n_agents)]

    def _run_episode(self, sim, ep, seed: int = 0):
        self._get_percep()
        n = self.n_agents
        rng = np.random.default_rng(seed)
        goals = ep.goals_set
        M = len(goals)
        self._cur_ep_id = ep.episode_id
        reach = [[np.asarray(p, np.float32) for p in g["reach_points"]] for g in goals]
        budget = ep.budget_geo or float("inf")

        starts = [np.asarray(a["start_position"], np.float32) for a in ep.agents]
        sim.reset()
        for i in range(n):
            rot = ep.agents[i].get("start_rotation") or [0, math.sin(rng.uniform(0, math.pi)), 0, math.cos(rng.uniform(0, math.pi))]
            sim.set_agent_state(position=[float(x) for x in starts[i]], rotation=[float(x) for x in rot], agent_id=i)
        last_obs = sim._per_agent_obs()
        if self.mp is not None:
            self.mp.begin_episode(sim)         # proven occupancy map drives nav frontiers

        smap = ConavSemanticMap(self._hfov, self._width, self._height, self._dmax,
                                origin_xz=(float(starts[0][0]), float(starts[0][2])))

        def map_update():
            if self.mp is not None:            # proven map (nav/frontier source)
                try:
                    self.mp.step(sim, list(last_obs))
                except Exception as e:  # noqa: BLE001
                    print(f"[conav][mp-warn] {e!r}")
            for i in range(n):                 # semantic RGB map (VLM image only)
                st = sim.get_agent_state(i)
                ss = st.sensor_states.get("rgb", st) if hasattr(st, "sensor_states") else st
                smap.update(last_obs[i].get("rgb"), last_obs[i].get("depth"), ss.position, ss.rotation)

        map_update()
        states = [PerAgentState(agent_id=i, last_position=starts[i].copy()) for i in range(n)]
        found = [False] * M
        found_by = [-1] * M
        locked = [None] * n
        blacklist = [set() for _ in range(n)]
        target_list = [_goal_label(goals[j], j) for j in range(M)]
        step_ref = [0]
        t0 = time.time()
        log_traj = getattr(self, "log_traj", False)
        from . import traj_log
        gxz = traj_log.goal_xz(goals)
        traj, found_events = traj_log.init_traj(starts, n) if log_traj else (None, None)
        od_gate = ODFoundGate.from_runner(self, n, goals, ep.episode_id)

        def claim():
            for i in range(n):
                if states[i].stopped:
                    continue
                for j in claim_from_obs(
                    od_gate, sim, i, last_obs[i], step_ref[0], found, found_by, locked,
                ):
                    if log_traj:
                        found_events.append([step_ref[0], i, j, goals[j]["object_category"],
                                             goals[j]["modality"], gxz[j][0], gxz[j][1]])
                    print(
                        f"[conav][od-found] agent_{i} found goal{j} "
                        f"({goals[j]['object_category']}/{goals[j]['modality']}) "
                        f"@ step {step_ref[0]}"
                    )

        claim()
        while step_ref[0] < self.max_steps and not all(found):
            cur_all = [np.asarray(sim.get_agent_state(i).position, np.float32) for i in range(n)]
            if not getattr(self, "clean_found_state", True):
                for i in range(n):
                    if states[i].stopped:
                        continue
                    if locked[i] is not None and (found[locked[i]] or locked[i] in blacklist[i]):
                        locked[i] = None
                    if locked[i] is None:
                        near = [(self._geo_to_reach(sim, cur_all[i], reach[j]), j) for j in range(M)
                                if not found[j] and j not in blacklist[i]]
                        near = [(d, j) for d, j in near if d <= self.approach_radius]
                        if near:
                            locked[i] = min(near)[1]
                    if locked[i] is not None:
                        states[i].target_xy_world = min(
                            reach[locked[i]], key=lambda p: self._geo(sim, cur_all[i], p))

            # centralized VLM frontier (re)assignment for non-locked agents.
            # Frontiers + world coords come from the PROVEN occupancy map (self.mp,
            # same as DM3/ours) so navigation works; the semantic RGB map is only
            # for the VLM image (frontiers drawn at their world positions).
            if step_ref[0] % self.realloc_every == 0 and self.mp is not None:
                layers = self.mp.merged_layers()
                fr = detect_frontiers(layers["explored"], layers["obstacle"], min_cluster=12) if layers else []
                if fr:
                    fr_world = [layers["cell_to_world"](cx, cy) for cx, cy, _ in fr]   # proven nav coords
                    fr_cells_img = [smap.world_to_cell(wx, wz) for (wx, wz) in fr_world]  # for drawing
                    agent_cells = [smap.world_to_cell(float(c[0]), float(c[2])) for c in cur_all]
                    asg = self._central_assign(
                        smap, fr_cells_img, fr_world, agent_cells, cur_all, last_obs, target_list)
                    for i in range(n):
                        if locked[i] is None and asg:
                            wx, wz = fr_world[asg[i]]
                            tgt = np.array([wx, float(cur_all[i][1]), wz], np.float32)
                            sp = np.asarray(sim.pathfinder.snap_point(tgt), np.float32)
                            states[i].target_xy_world = sp if np.all(np.isfinite(sp)) else tgt
            if od_gate is not None:
                for i in range(n):
                    if states[i].stopped:
                        continue
                    tgt = sensor_approach_target(od_gate, i, found, float(cur_all[i][1]))
                    if tgt is not None:
                        states[i].target_xy_world = tgt
            for i in range(n):
                if states[i].target_xy_world is None and not states[i].stopped:
                    states[i].target_xy_world = self._pick_random_navigable_target(sim, rng, cur_all[i])

            actions = [HabitatSimActions.TURN_RIGHT] * n
            for i in range(n):
                if not states[i].stopped:
                    actions[i] = self._next_action_for(sim, states[i])
            step_actions = [HabitatSimActions.TURN_RIGHT if a == HabitatSimActions.STOP else a for a in actions]
            last_obs = sim.step(step_actions)
            step_ref[0] += 1
            map_update()

            for i in range(n):
                if states[i].stopped:
                    continue
                s = states[i]
                cur = np.asarray(sim.get_agent_state(i).position, np.float32)
                s.path_length += float(np.linalg.norm(cur - s.last_position))
                s.last_position = cur
                if log_traj:
                    traj[i].append([float(cur[0]), float(cur[2])])
                if s.path_length >= budget:
                    s.stopped = True
                    continue
                tg = s.target_xy_world
                if tg is not None:
                    d = self._geo(sim, cur, tg)
                    if not np.isfinite(d) or d <= self.success_distance + 0.1:
                        s.target_xy_world = None
                    elif d < s.last_tg_d - 0.05:
                        s.last_tg_d = d
                        s.consec_stuck = 0
                    else:
                        s.consec_stuck += 1
                        if s.consec_stuck >= 20:
                            s.consec_stuck = 0
                            if locked[i] is not None:
                                blacklist[i].add(locked[i]); locked[i] = None
                            s.target_xy_world = None
            claim()
            if all(s.stopped for s in states):
                break

        scored_found_by = found_by
        scored_found = [owner >= 0 for owner in scored_found_by]
        per_agent = [{"path_length": round(states[i].path_length, 3),
                      "goals_found": [j for j in range(M) if scored_found_by[j] == i]} for i in range(n)]
        if log_traj and self.save_dir:
            traj_log.dump_traj(self.save_dir, ep, n, starts, goals, found_by, found_events, step_ref[0], traj, gxz)
        m = metrics.team_metrics(ep._raw, {"per_agent": per_agent})
        print(f"[conav] ep {ep.episode_id} found={sum(scored_found)}/{M} progress={m['progress']} steps={step_ref[0]}")
        pa = [{"agent_id": i, "success": int(len(per_agent[i]["goals_found"]) > 0), "spl": 0.0,
               "path_length": per_agent[i]["path_length"], "goals_found": per_agent[i]["goals_found"]} for i in range(n)]
        result = {"episode_id": str(ep.episode_id), "scene_id": ep.scene_id,
                "object_category": ep.object_category, "steps": step_ref[0],
                "elapsed_s": time.time() - t0, "per_agent": pa,
                "joint_success": int(all(scored_found)), "team_spl": m["mspl"],
                "progress": m["progress"], "metrics": m}
        attach_od_metrics("conav", od_gate, m, result)
        return result

    def _dump_summary(self, results):
        mets = [r["metrics"] for r in results if "metrics" in r]
        agg = metrics.aggregate(mets)
        print("\n[conav] === AGGREGATE ===")
        print(json.dumps(agg, indent=2))
        if self.save_dir:
            with open(self.save_dir / "aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)
            with open(self.save_dir / "metrics_list.json", "w") as f:
                json.dump(mets, f)   # per-episode metrics, for shard merging


def main(argv=None):
    from .ma_search import EpisodeAdapter, _load_cfg
    p = argparse.ArgumentParser(description="Co-NavGPT2 reproduction (centralized GPT-4o frontier)")
    p.add_argument("--episodes", required=True)
    p.add_argument("--config", default="vlnce_baselines/config/multi_objectnav_v2.yaml")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--save-dir", default="logs/conav_run")
    p.add_argument("--iin-images-dir", default="data/datasets/ma_objectnav/val/iin_images")
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vlm", default="qwen", choices=["qwen", "gpt4o"],
                   help="VLM backend (both aliases map to Bailian/DashScope qwen3.5-27b; API unified)")
    p.add_argument("--log-traj", action="store_true")
    p.add_argument("--legacy-gt-approach-lock", action="store_true",
                   help="enable the old GT reach_points approach-lock (not valid for clean comparisons)")
    p.add_argument("--episode-ids", default="")
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
    runner = CoNavRunner(cfg=_load_cfg(args.config), max_episode_steps=args.max_episode_steps,
                         save_dir=args.save_dir, use_vlm=False,
                         use_continuous_history=False, visualize=False)
    runner.vlm_url, runner.vlm_key, runner.vlm_model = VLM_BACKENDS[args.vlm]
    runner.log_traj = args.log_traj
    runner.clean_found_state = not args.legacy_gt_approach_lock
    manifest = os.path.join(args.iin_images_dir, "manifest.json")
    if os.path.exists(manifest):
        runner.iin_lookup = {(str(it["episode_id"]), it["goal_index"]): it["image"]
                             for it in json.load(open(manifest)).get("items", [])}
    print(f"[conav] {len(adapters)} episodes (VLM={args.vlm}:{runner.vlm_model}), IIN imgs={len(runner.iin_lookup)}")
    runner.evaluate(adapters, seed=args.seed)


if __name__ == "__main__":
    main()
