#!/usr/bin/env python3
"""Merge ``benchmark.gen_tn --sidecar`` JSONL files into episode JSON(s) in place.

    python scripts/gen/merge_tn_sidecars.py --episodes EP.json [--episodes MINI.json] \
        --sidecar-dir logs/fill_iin_tn_xxx [--replace]

Each sidecar line is ``{"episode_id", "goal_index", "tn_text"}``; the text is
written into the matching TN goal (only into empty ``tn_text`` unless --replace).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_updates(sidecar_dir: str):
    updates = {}
    n_lines = 0
    for path in sorted(Path(sidecar_dir).glob("tn_shard*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("tn_text"):
                    continue
                updates[(rec["episode_id"], int(rec["goal_index"]))] = rec["tn_text"]
                n_lines += 1
    return updates, n_lines


def apply_updates(ep_path: str, updates, replace: bool) -> str:
    path = Path(ep_path)
    with open(path) as f:
        data = json.load(f)
    n_apply = n_tn = n_filled = 0
    for ep in data["episodes"]:
        for g in ep["goals_set"]:
            if g.get("modality") != "TN":
                continue
            n_tn += 1
            key = (ep["episode_id"], int(g["goal_index"]))
            if key in updates and (replace or not g.get("tn_text")):
                g["tn_text"] = updates[key]
                n_apply += 1
            if g.get("tn_text"):
                n_filled += 1
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return f"{path}: applied={n_apply} tn_filled={n_filled}/{n_tn}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", action="append", required=True,
                   help="episode JSON to update in place (repeatable)")
    p.add_argument("--sidecar-dir", required=True)
    p.add_argument("--replace", action="store_true", help="overwrite existing tn_text")
    args = p.parse_args(argv)
    updates, n_lines = load_updates(args.sidecar_dir)
    print(f"sidecar_lines={n_lines} unique={len(updates)}")
    for ep in args.episodes:
        print(apply_updates(ep, updates, args.replace))


if __name__ == "__main__":
    main()
