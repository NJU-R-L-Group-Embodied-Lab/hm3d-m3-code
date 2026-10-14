"""Rebake TESM v2 HM3D navmeshes so Habitat-sim 0.1.7 PathFinder can load them.

The zhuhu00 train habitat tarball ships mixed TESM versions. 0.1.7 loads v1
and rejects v2. This rewrites v2 files in place (original kept as *.tesm2).
"""
from __future__ import annotations

import argparse
import os
import struct
from typing import List, Optional, Sequence


def tesm_version(path: str) -> Optional[int]:
    with open(path, "rb") as f:
        hdr = f.read(8)
    if len(hdr) < 8 or hdr[:4] != b"TESM":
        return None
    return int(struct.unpack("<I", hdr[4:8])[0])


def scene_keys_from_content(content_dir: str, scenes_dir: str) -> List[str]:
    from .goal_sources import iter_scene_files

    by_hash = {}
    for split in ("hm3d_v0.2/train", "hm3d/train"):
        root = os.path.join(scenes_dir, split)
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if "-" in name and os.path.isdir(os.path.join(root, name)):
                by_hash[name.split("-", 1)[-1]] = name
    keys = []
    for f in iter_scene_files(content_dir):
        h = os.path.basename(f).replace(".json.gz", "")
        key = by_hash.get(h)
        if key is None:
            print("[rebake] no train folder for", h)
            continue
        keys.append(key)
    return keys


def _navmesh_path(scenes_dir: str, scene_key: str) -> Optional[str]:
    for split in ("hm3d_v0.2/train", "hm3d/train"):
        folder = os.path.join(scenes_dir, split, scene_key)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.endswith(".basis.navmesh") and not name.endswith(".tesm2"):
                return os.path.join(folder, name)
    return None


def _glb_path(navmesh: str) -> str:
    return navmesh[: -len(".navmesh")] + ".glb"


def rebake_one(navmesh: str, *, radius: float, height: float, max_climb: float, max_slope: float) -> bool:
    import habitat_sim

    glb = _glb_path(navmesh)
    if not os.path.isfile(glb):
        print("[rebake] no glb", glb)
        return False
    backup = navmesh + ".tesm2"
    if tesm_version(navmesh) == 2 and not os.path.isfile(backup):
        os.replace(navmesh, backup)
        src_nav = backup
    else:
        src_nav = navmesh

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.abspath(glb)
    sim_cfg.create_renderer = False
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = False
    sim_cfg.requires_textures = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    backend = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(backend)
    try:
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius = float(radius)
        ns.agent_height = float(height)
        ns.agent_max_climb = float(max_climb)
        ns.agent_max_slope = float(max_slope)
        ok = bool(sim.recompute_navmesh(sim.pathfinder, ns))
        if not ok or not sim.pathfinder.is_loaded:
            print("[rebake] recompute failed", navmesh)
            return False
        tmp = navmesh + ".tmp"
        sim.pathfinder.save_nav_mesh(tmp)
        os.replace(tmp, navmesh)
        area = float(sim.pathfinder.navigable_area)
    finally:
        sim.close()

    ver = tesm_version(navmesh)
    if ver != 1:
        print("[rebake] wrote non-v1", navmesh, "ver", ver)
        return False
    print("[rebake] ok", os.path.basename(os.path.dirname(navmesh)), "area=%.1f" % area, "kept", os.path.basename(src_nav) if src_nav != navmesh else "inplace")
    return True


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--content-dir", default="data/datasets/hm3d_ovon/train/content")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--only-tesm2", action="store_true", default=True)
    p.add_argument("--radius", type=float, default=0.1)
    p.add_argument("--height", type=float, default=0.88)
    p.add_argument("--max-climb", type=float, default=0.20)
    p.add_argument("--max-slope", type=float, default=45.0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    keys = scene_keys_from_content(args.content_dir, args.scenes_dir)
    print("[rebake] ovon scenes", len(keys))
    n_ok = n_skip = n_fail = 0
    for i, key in enumerate(keys):
        if args.limit and i >= args.limit:
            break
        nav = _navmesh_path(args.scenes_dir, key)
        if not nav:
            print("[rebake] no navmesh", key)
            n_fail += 1
            continue
        ver = tesm_version(nav)
        if args.only_tesm2 and ver == 1:
            n_skip += 1
            continue
        try:
            if rebake_one(
                nav,
                radius=args.radius,
                height=args.height,
                max_climb=args.max_climb,
                max_slope=args.max_slope,
            ):
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:  # noqa: BLE001
            print("[rebake] fail", key, type(exc).__name__, exc)
            n_fail += 1
    print("[rebake] done ok=%d skip_v1=%d fail=%d" % (n_ok, n_skip, n_fail))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
