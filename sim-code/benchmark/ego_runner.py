"""EGO-Swarm reproduction — pure swarm exploration + detection (no LLM).

Faithful-in-spirit adaptation of ZJU FAST-Lab **ego-planner-swarm**
(https://github.com/ZJU-FAST-Lab/ego-planner-swarm). The original is a UAV
*local trajectory optimiser* whose defining feature is **decentralized swarm
dispersion** — each drone plans independently and a mutual *repulsion* term
pushes drones apart so the team naturally spreads to cover space, with **no
central allocation and no semantic intent communication** (only relative
positions). Its exploration counterpart in spirit is "spread out and cover".

We transplant that core to our Habitat ObjectNav benchmark as the most naive
geometric-swarm baseline:

* **Map**: a shared occupancy map (`SharedMappingPlanner.merged_layers`) — same
  shared map as our baseline / Co-NavGPT2 (the ONLY thing dropped vs DM3 is the
  semantic *intent broadcast*, isolating "repulsion-dispersion vs intent").
* **Coordination (the ego-swarm core)**: each agent independently picks an
  exploration frontier maximising a coverage score that **repels from teammates'
  current positions** — frontiers far from other agents score higher, so the
  swarm disperses. NO central planner, NO intent/yield negotiation, NO comm gate.
* **Perception**: pure detection — Mask R-CNN + SIFT(IIN) + CLIP(TN) via
  `benchmark.dm3_perception` (identical to DM3 ⇒ the perception variable is
  held fixed; only the *exploration paradigm* differs). No LLM/VLM anywhere.

Contrast in one line: our baseline = random-point explore; DM3 = frontier +
intent broadcast; ego-swarm = frontier + **position repulsion dispersion**.

    python -m benchmark.ego_runner --episodes .../ma_stratified_100_seed42.json \
        --limit 1 --max-episode-steps 500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import List, Optional

import numpy as np

from vlnce_baselines.multi.ma_runner import (HabitatSimActions, MultiAgentRunner,
                                             PerAgentState)
from . import coord_shell, metrics
from .frontier import detect_frontiers
from .od_found import ODFoundGate, attach_od_metrics, claim_from_obs, sensor_approach_target


class EgoSwarmRunner(MultiAgentRunner):
    clean_found_state: bool = True
    approach_radius: float = 4.0
    realloc_every: int = 25
    iin_inlier_thresh: int = 18
    tn_clip_thresh: float = 0.24
    repulse_lambda: float = 0.6          # weight of the swarm-dispersion repulsion term
    _percep = None
    iin_lookup: dict = {}

    def _get_percep(self):
        if self._percep is None:
            from .dm3_perception import DM3Perception
            self._percep = DM3Perception()
        return self._percep

    def _geo(self, sim, a, b):
        d = sim.geodesic_distance(a, b)
        return float(d) if (d is not None and np.isfinite(d)) else float("inf")

    def _geo_to_reach(self, sim, pos, reach):
        return min((self._geo(sim, pos, p) for p in reach), default=float("inf"))

    # ----- EGO-Swarm core: coverage-MAXIMIZING frontier pick (FUEL/RACER-style) -----
    def _pick_frontier(self, sim, i, positions, fblack=None):
        """Decentralized, coverage-maximizing frontier choice on the SHARED map.

        FUEL/RACER-style information-gain exploration: each frontier's *score* is
        its **information gain** (cluster size = expected newly-revealed area)
        discounted by travel cost ``exp(-d/D0)`` (so the agent maximises area
        covered per unit path), PLUS a swarm **dispersion** term repelling from
        teammates so the team spreads. Recently-failed (unreachable / stuck)
        frontiers in ``fblack`` are suppressed so the agent stops re-locking dead
        targets. No central allocation, no intent broadcast.
        """
        if self.mp is None:
            return None
        layers = self.mp.merged_layers()
        if layers is None:
            return None
        fr = detect_frontiers(layers["explored"], layers["obstacle"], min_cluster=8)
        if not fr:
            return None
        cur = np.asarray(positions[i], dtype=np.float32)
        others = [np.asarray(positions[j], np.float32) for j in range(len(positions)) if j != i]
        max_sz = max(s for _, _, s in fr) or 1
        D0 = 5.0                                                # distance-discount scale (m)
        best, best_score = None, -1e9
        for (cx, cy, sz) in fr:
            wx, wz = layers["cell_to_world"](cx, cy)
            tgt = np.array([wx, float(cur[1]), wz], np.float32)
            d = self._geo(sim, cur, tgt)
            if not np.isfinite(d):
                continue                                        # unreachable -> skip entirely
            info_gain = (sz / max_sz)                           # expected new coverage (FUEL)
            reach = math.exp(-d / D0)                           # discount by travel cost
            if others:                                          # swarm dispersion (repulsion)
                rep = min(float(np.linalg.norm(np.array([wx, wz]) - np.array([o[0], o[2]])))
                          for o in others)
            else:
                rep = 0.0
            score = info_gain * reach + self.repulse_lambda * math.tanh(rep / 4.0)
            if fblack:                                          # suppress recently-failed frontiers
                if min((np.hypot(wx - bx, wz - bz) for (bx, bz) in fblack), default=9e9) < 1.5:
                    score -= 5.0
            if score > best_score:
                best_score, best = score, tgt
        return best

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
            sim.set_agent_state(position=[float(x) for x in starts[i]],
                                rotation=[float(x) for x in rot], agent_id=i)
        last_obs = sim._per_agent_obs()
        if self.mp is not None:
            self.mp.begin_episode(sim)

        states = [PerAgentState(agent_id=i, last_position=starts[i].copy()) for i in range(n)]
        found = [False] * M
        found_by = [-1] * M
        locked = [None] * n
        blacklist = [set() for _ in range(n)]          # goals an agent has given up on
        frontier_black = [[] for _ in range(n)]        # world (x,z) of recently-failed frontiers
        step_ref = [0]
        t0 = time.time()
        log_traj = getattr(self, "log_traj", False)
        traj = [[ [float(starts[i][0]), float(starts[i][2])] ] for i in range(n)] if log_traj else None
        found_events = []          # (step, agent, goal_idx, category, modality, wx, wz)
        goal_xz = [[float(g["reach_points"][0][0]), float(g["reach_points"][0][2])] for g in goals]
        ch = coord_shell.CoordChannel(congestion_radius=1.0)   # coordination-cost log
        od_gate = ODFoundGate.from_runner(self, n, goals, ep.episode_id)

        def positions():
            return [np.asarray(sim.get_agent_state(i).position, np.float32) for i in range(n)]

        def claim():
            for i in range(n):
                if states[i].stopped:
                    continue
                for j in claim_from_obs(
                    od_gate, sim, i, last_obs[i], step_ref[0], found, found_by, locked,
                ):
                    if log_traj:
                        found_events.append([step_ref[0], i, j, goals[j]["object_category"],
                                             goals[j]["modality"], goal_xz[j][0], goal_xz[j][1]])
                    print(
                        f"[ego][od-found] agent_{i} found goal{j} "
                        f"({goals[j]['object_category']}/{goals[j]['modality']}) "
                        f"@ step {step_ref[0]}"
                    )

        claim()
        while step_ref[0] < self.max_steps and not all(found):
            pos = positions()
            for i in range(n):
                if states[i].stopped:
                    continue
                cur = pos[i]
                handled = False
                if not getattr(self, "clean_found_state", True):
                    if locked[i] is not None and (found[locked[i]] or locked[i] in blacklist[i]):
                        locked[i] = None
                    if locked[i] is None:
                        near = [(self._geo_to_reach(sim, cur, reach[j]), j) for j in range(M)
                                if not found[j] and j not in blacklist[i]]
                        near = [(d, j) for d, j in near if d <= self.approach_radius]
                        if near:
                            locked[i] = min(near)[1]
                    if locked[i] is not None:
                        j = locked[i]
                        states[i].target_xy_world = min(reach[j], key=lambda p: self._geo(sim, cur, p))
                        handled = True
                if not handled and od_gate is not None:
                    tgt = sensor_approach_target(od_gate, i, found, float(cur[1]))
                    if tgt is not None:
                        states[i].target_xy_world = tgt
                        handled = True
                if (not handled and (states[i].needs_decision or states[i].target_xy_world is None
                                     or step_ref[0] % self.realloc_every == 0)):
                    # coverage-maximizing re-plan: event-driven (arrival / stuck) AND
                    # periodic (adapt to freshly-revealed map). No 25-step lockout, so a
                    # stuck agent re-plans immediately instead of oscillating for 25 steps.
                    # decentralized EGO re-plan reads all teammates' positions off the
                    # shared map -> a position broadcast each agent consumes (comm cost).
                    ch.message(i, payload_bytes=12 * n)
                    tgt = self._pick_frontier(sim, i, pos, frontier_black[i])
                    states[i].target_xy_world = tgt if tgt is not None else self._pick_random_navigable_target(sim, rng, cur)
                states[i].needs_decision = False

            actions = [HabitatSimActions.TURN_RIGHT] * n
            for i in range(n):
                if not states[i].stopped:
                    actions[i] = self._next_action_for(sim, states[i])
            step_actions = [HabitatSimActions.TURN_RIGHT if a == HabitatSimActions.STOP else a for a in actions]
            last_obs = sim.step(step_actions)
            step_ref[0] += 1
            ch.observe_positions([p.tolist() for p in positions()])   # congestion proxy
            if self.mp is not None:
                try:
                    self.mp.step(sim, list(last_obs))
                except Exception as e:  # noqa: BLE001
                    print(f"[ego][mp-warn] {e!r}")

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
                        s.needs_decision = True
                    elif d < s.last_tg_d - 0.05:
                        s.last_tg_d = d
                        s.consec_stuck = 0
                    else:
                        s.consec_stuck += 1
                        if s.consec_stuck >= 12:           # faster recovery (was 20)
                            s.needs_decision = True
                            s.consec_stuck = 0
                            if locked[i] is not None:
                                blacklist[i].add(locked[i])
                                locked[i] = None
                            elif tg is not None:           # blacklist the dead frontier
                                fb = frontier_black[i]
                                fb.append((float(tg[0]), float(tg[2])))
                                if len(fb) > 12:
                                    fb.pop(0)
            claim()
            if all(s.stopped for s in states):
                break

        scored_found_by = found_by
        scored_found = [owner >= 0 for owner in scored_found_by]
        per_agent = [{"path_length": round(states[i].path_length, 3),
                      "goals_found": [j for j in range(M) if scored_found_by[j] == i]} for i in range(n)]
        if log_traj and self.save_dir:
            td = {"episode_id": str(ep.episode_id), "scene_id": ep.scene_id, "n_agents": n,
                  "starts_xz": [[float(s[0]), float(s[2])] for s in starts],
                  "goals": [{"idx": j, "category": goals[j]["object_category"],
                             "modality": goals[j]["modality"], "xz": goal_xz[j],
                             "reach_xz": [[float(p[0]), float(p[2])] for p in goals[j]["reach_points"]],
                             "found_by": found_by[j]} for j in range(M)],
                  "found_events": found_events, "steps": step_ref[0], "traj_xz": traj}
            with open(self.save_dir / f"traj_{ep.episode_id}.json", "w") as fh:
                json.dump(td, fh)
            print(f"[ego][traj] dumped traj_{ep.episode_id}.json ({step_ref[0]} steps)")
        m = metrics.team_metrics(ep._raw, {"per_agent": per_agent, **ch.summary()})
        print(f"[ego] ep {ep.episode_id} found={sum(scored_found)}/{M} progress={m['progress']} "
              f"steps={step_ref[0]} comm={m['comm_volume']} congestion={m['congestion']}")
        pa = [{"agent_id": i, "success": int(len(per_agent[i]["goals_found"]) > 0), "spl": 0.0,
               "path_length": per_agent[i]["path_length"], "goals_found": per_agent[i]["goals_found"]} for i in range(n)]
        result = {"episode_id": str(ep.episode_id), "scene_id": ep.scene_id,
                "object_category": ep.object_category, "steps": step_ref[0],
                "elapsed_s": time.time() - t0, "per_agent": pa,
                "joint_success": int(all(scored_found)), "team_spl": m["mspl"],
                "progress": m["progress"], "metrics": m}
        attach_od_metrics("ego", od_gate, m, result)
        return result

    def _dump_summary(self, results):
        mets = [r["metrics"] for r in results if "metrics" in r]
        agg = metrics.aggregate(mets)
        print("\n[ego] === AGGREGATE ===")
        print(json.dumps(agg, indent=2))
        if self.save_dir:
            with open(self.save_dir / "aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)
            with open(self.save_dir / "metrics_list.json", "w") as f:
                json.dump(mets, f)   # per-episode metrics, for shard merging


def main(argv=None):
    from .ma_search import EpisodeAdapter, _load_cfg
    p = argparse.ArgumentParser(description="EGO-Swarm reproduction runner (no-LLM, dispersion)")
    p.add_argument("--episodes", required=True)
    p.add_argument("--config", default="vlnce_baselines/config/multi_objectnav_v2.yaml")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--save-dir", default="logs/ego_run")
    p.add_argument("--iin-images-dir", default="data/datasets/ma_objectnav/val/iin_images")
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--repulse-lambda", type=float, default=0.6)
    p.add_argument("--legacy-gt-approach-lock", action="store_true",
                   help="enable the old GT reach_points approach-lock (not valid for clean comparisons)")
    p.add_argument("--log-traj", action="store_true", help="dump per-step trajectory JSON for visualization")
    p.add_argument("--episode-ids", default="", help="comma-separated episode_ids to run (overrides start/count)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
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
    runner = EgoSwarmRunner(cfg=_load_cfg(args.config), max_episode_steps=args.max_episode_steps,
                            save_dir=args.save_dir, use_vlm=False,
                            use_continuous_history=False, visualize=False)
    runner.clean_found_state = not args.legacy_gt_approach_lock
    runner.repulse_lambda = args.repulse_lambda
    runner.log_traj = args.log_traj
    manifest = os.path.join(args.iin_images_dir, "manifest.json")
    if os.path.exists(manifest):
        runner.iin_lookup = {(str(it["episode_id"]), it["goal_index"]): it["image"]
                             for it in json.load(open(manifest)).get("items", [])}
    print(f"[ego] {len(adapters)} episodes, repulse_lambda={args.repulse_lambda}, IIN imgs={len(runner.iin_lookup)}")
    runner.evaluate(adapters, seed=args.seed)


if __name__ == "__main__":
    main()
