"""VLFM baseline under the HM3D-M3 coordination shell.

VLFM (Yokoyama et al., 2024) HabitatITMPolicyV2: BLIP-2 ITC cosine
(``Seems like there is a {object} ahead.``) splatted into a value map;
frontiers ranked by nearby value. Initial 11 left turns match official
Habitat initialization. Goals are declared through the shared detector
FOUND gate inherited from :class:`EgoSwarmRunner`.

    python -m benchmark.zeroshot_runner --method vlfm --episodes <set>.json \
        --scenes-dir data/scene_datasets/hm3d-1.0 --save-dir logs/vlfm --limit 40
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from vlnce_baselines.multi.ma_runner import HabitatSimActions

from .ego_runner import EgoSwarmRunner
from .frontier import detect_frontiers
from .vlfm_value import BLIP2ValueMap


def _forward_yaw(rotation) -> float:
    """World-frame heading (atan2(dz,dx)) of an agent from its quaternion.

    Habitat agents look down -Z; rotate (0,0,-1) by the quaternion and read XZ.
    ``rotation`` may be a quaternion object (w,x,y,z) or a list [x,y,z,w].
    """
    try:
        w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    except AttributeError:
        x, y, z, w = rotation
    fx = -2.0 * (x * z + w * y)
    fz = -(1.0 - 2.0 * (x * x + y * y))
    return math.atan2(fz, fx)


def _obs_rgb_depth(obs) -> tuple:
    rgb = obs.get("rgb") if obs else None
    depth = obs.get("depth") if obs else None
    if rgb is not None:
        rgb = np.asarray(rgb)[:, :, :3]
    return rgb, depth


class ZeroShotRunner(EgoSwarmRunner):
    method: str = "vlfm"
    det_box_thresh: float = 0.32
    value_radius: float = 1.5
    vlfm_backend: str = "blip2"     # {"blip2","blip2-itm"} — paper default is blip2 (ITC)
    realloc_every: int = 1          # official VLFM re-evaluates every step

    # Found uses EgoSwarmRunner OD gate (OWL / crop CLIP / SuperGlue). No GT geodesic.
    # Do not special-case GroundingDINO category hits — same ruler as the EGO-Swarm / Co-NavGPT2 / DM3-Nav rows.

    def _spin_steps(self) -> int:
        # VLFM HabitatMixin._initialize: 11 left turns (comment says 12 / 360°).
        return 11

    def _run_episode(self, sim, ep, seed: int = 0):
        n = self.n_agents
        cats = sorted({g["object_category"] for g in ep.goals_set})
        device = getattr(self, "device", "cuda")
        self._spin_left = [self._spin_steps() for _ in range(n)]
        backend = str(self.vlfm_backend or "blip2")
        kw = {"match_head": "itm"} if backend == "blip2-itm" else {"match_head": "itc"}
        self._vmaps = [BLIP2ValueMap(device=device, **kw) for _ in range(n)]
        for vm in self._vmaps:
            vm.set_targets(cats)
        return super()._run_episode(sim, ep, seed)

    def _update_vlfm(self, sim, i) -> None:
        vm = self._vmaps[i]
        try:
            obs = sim._per_agent_obs()
            rgb, _depth = _obs_rgb_depth(obs[i])
            st = sim.get_agent_state(i)
            val = vm.score_image(rgb) if rgb is not None else 0.0
            vm.update(val, (float(st.position[0]), float(st.position[2])),
                      _forward_yaw(st.rotation))
        except Exception as e:  # noqa: BLE001
            print(f"[vlfm][warn] value update failed: {e!r}")

    def _next_action_for(self, sim, state):
        i = state.agent_id
        left = getattr(self, "_spin_left", None)
        if left is not None and i < len(left) and left[i] > 0:
            if getattr(self, "_vmaps", None):
                self._update_vlfm(sim, i)
            left[i] -= 1
            return HabitatSimActions.TURN_LEFT
        return super()._next_action_for(sim, state)

    # -- frontier / exploit policy ------------------------------------------
    def _pick_frontier(self, sim, i, positions, fblack=None):
        if self.mp is None:
            return None
        layers = self.mp.merged_layers()
        if layers is None:
            return None
        fr = detect_frontiers(layers["explored"], layers["obstacle"], min_cluster=8)
        if not fr:
            return None
        cur = np.asarray(positions[i], dtype=np.float32)

        # VLFM HabitatITMPolicyV2: update value map, rank frontiers by value
        self._update_vlfm(sim, i)
        vm = self._vmaps[i]

        D0 = 5.0
        best, best_score, observed_any = None, -1e9, False
        for (cx, cy, sz) in fr:
            wx, wz = layers["cell_to_world"](cx, cy)
            tgt = np.array([wx, float(cur[1]), wz], np.float32)
            d = self._geo(sim, cur, tgt)
            if not np.isfinite(d):
                continue
            v = vm.value_at(wx, wz, radius=self.value_radius)
            if v >= 0:
                observed_any = True
                score = v + 0.3 * math.exp(-d / D0)
            else:
                score = 0.05 * math.exp(-d / D0)
            if fblack and min((np.hypot(wx - bx, wz - bz) for (bx, bz) in fblack),
                              default=9e9) < 1.5:
                score -= 5.0
            if score > best_score:
                best_score, best = score, tgt
        if not observed_any:
            return super()._pick_frontier(sim, i, positions, fblack)
        return best


def _pin_cpu_threads() -> None:
    """Match the LaViRA GPU-eval launches: one BLAS/OpenMP thread.

    Habitat Magnum otherwise opens a dozen cores per process; 16 workers
    then oversubscribe the machine and each episode stretches to ~100 min.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


def main(argv=None):
    _pin_cpu_threads()
    from .ma_search import EpisodeAdapter, _load_cfg
    p = argparse.ArgumentParser(description="VLFM baseline under the MA shell")
    p.add_argument("--method", default="vlfm", choices=["vlfm"])
    p.add_argument("--episodes", required=True)
    p.add_argument("--config", default="vlnce_baselines/config/multi_objectnav_v2.yaml")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--save-dir", default="logs/zeroshot_run")
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--det-box-thresh", type=float, default=0.32)
    p.add_argument("--iin-images-dir", default="data/datasets/ma_objectnav/val/iin_images")
    p.add_argument("--vlfm-backend", default="blip2",
                   choices=["blip2", "blip2-itm"],
                   help="VLFM value head. blip2=official ITC; blip2-itm=ITM P(match).")
    p.add_argument("--episode-ids", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    data = json.load(open(args.episodes))
    eps = data["episodes"]
    if args.episode_ids:
        want = set(args.episode_ids.split(","))
        eps = [e for e in data["episodes"] if str(e["episode_id"]) in want]
    if args.start:
        eps = eps[args.start:]
    if args.count:
        eps = eps[:args.count]
    if args.limit:
        eps = eps[:args.limit]
    adapters = [EpisodeAdapter(e, args.scenes_dir) for e in eps]
    adapters.sort(key=lambda e: e.scene_id)

    runner = ZeroShotRunner(cfg=_load_cfg(args.config), max_episode_steps=args.max_episode_steps,
                            save_dir=args.save_dir, use_vlm=False,
                            use_continuous_history=False,
                            visualize=False)
    runner.method = args.method
    runner.det_box_thresh = args.det_box_thresh
    runner.vlfm_backend = args.vlfm_backend
    runner.realloc_every = 1
    manifest = os.path.join(args.iin_images_dir, "manifest.json")
    if os.path.exists(manifest):
        runner.iin_lookup = {(str(it["episode_id"]), it["goal_index"]): it["image"]
                             for it in json.load(open(manifest)).get("items", [])}
    print(f"[zeroshot:{args.method}] {len(adapters)} episodes, backend="
          f"{args.vlfm_backend} "
          f"IIN imgs={len(getattr(runner, 'iin_lookup', {}) or {})}")
    runner.evaluate(adapters, seed=args.seed)


if __name__ == "__main__":
    main()
