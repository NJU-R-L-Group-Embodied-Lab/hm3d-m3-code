"""P2: generate TN (text-goal) descriptions for TN-modality goals via VLM.

Our episodes are composed from HM3D *ObjectNav* goals, which carry only a
category — no text. To materialise the TN modality we render the goal instance's
viewpoint and ask a vision LLM (qwen3.5-27b on the Aliyun/DashScope
OpenAI-compatible endpoint — the same VA model the evaluator uses) for ONE
instance-distinguishing sentence, then write it into ``goal['tn_text']``.

Default ``--sample-mode goat`` follows GOAT-Bench LanguageNav viewpoint
sampling (Stretch 512x512, r in {1.0,1.5,2.0} m every 10 deg, keep max
frame coverage, plus GT nearby-object boxes). The caption model is
``qwen3.5-27b`` instead of BLIP-2 + ChatGPT-3.5. ``--sample-mode viewpoint``
keeps the old stored ``iin_viewpoint`` render.

    python -m benchmark.gen_tn \
        --episodes data/datasets/ma_objectnav/val/ma_stratified_100_seed42.json \
        --limit 3                       # smoke; omit --limit for all TN goals

Idempotent: goals that already have ``tn_text`` are skipped unless ``--overwrite``.
Requires the Simulator (GPU/EGL) for rendering and network for the API.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

from . import config as _cfg
from .render_iin import (
    _make_sim,
    _render_at,
    _render_stretch512_goal,
    _scene_dataset_config,
    _scene_glb,
    pick_languagenav_view,
)

from benchmark.api_auth import get_llm_api_key, get_llm_base_url, get_llm_model

DEFAULT_BASE_URL = get_llm_base_url()

PROMPT = (
    "This is a view of a {cat} in an indoor home. Write ONE concise English "
    "sentence (max 25 words) describing THIS specific {cat} so a robot could tell "
    "it apart from other {cat}s in the house. Mention its colour, material/shape, "
    "and any nearby objects or where it is. Start with 'The {cat}'. Do not mention "
    "that this is an image or a render. Output only the sentence."
)

NOT_VISIBLE = re.compile(
    r"\b(not visible|cannot see|can't see|i don't see|unable to see|"
    r"is missing|no \w+ (is )?visible|only .+ (are|is) (present|shown|visible))\b",
    re.I,
)
FIND_PROMPT = (
    "Write ONE concise English sentence (max 25 words) starting with 'The {cat}' "
    "that helps a robot find THIS specific {cat}. Use colour, nearby objects, and "
    "room layout. Do not say the object is missing, not visible, or that this is "
    "an image. Output only the sentence."
)


GOAT_PROMPT = (
    "Generate an informative and natural instruction to a robot agent. "
    "Target category: {cat}. "
    "You are given one RGB view of the target and optional region semantics "
    "(2D boxes as xmin,ymin,xmax,ymax; (0,0) is top-left). "
    "Describe the target's appearance and its location relative to other objects. "
    "Rules: do not use absolute numbers; do not show bounding-box coordinates; "
    "use relative directions only. "
    "Prefix the sentence with 'Instruction: Find the' or 'Instruction: Go to'. "
    "Output only the instruction.\n"
    "Region semantics: {region}"
)


def _ascii_tn(txt: str) -> str:
    return (
        (txt or "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def tn_text_unusable(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    if NOT_VISIBLE.search(t):
        return True
    low = t.lower()
    if "this is an image" in low or "as an ai" in low:
        return True
    return False


def _img_to_data_url(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img)[:, :, :3]).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _region_text(nearby: Optional[list], category: str) -> str:
    if not nearby:
        return f"target={category}; nearby=[]"
    bits = []
    for rec in nearby[:12]:
        bits.append(
            "{cat} cov={cov:.3f} depth={d:.2f} bbox={bbox} target={tgt}".format(
                cat=rec.get("category") or "unknown",
                cov=float(rec.get("coverage") or 0.0),
                d=float(rec.get("mean_depth_m") or 0.0),
                bbox=rec.get("bbox"),
                tgt=bool(rec.get("is_target")),
            )
        )
    return "target=%s; %s" % (category, " | ".join(bits))


def _describe(client, model: str, img: np.ndarray, category: str,
              max_retries: int = 4, sample_mode: str = "viewpoint",
              nearby=None) -> Optional[str]:
    data_url = _img_to_data_url(img)
    if sample_mode == "goat":
        text = GOAT_PROMPT.format(cat=category, region=_region_text(nearby, category))
    else:
        text = PROMPT.format(cat=category)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, max_tokens=80, temperature=0.3)
            txt = _ascii_tn((r.choices[0].message.content or "").strip().strip('"').replace("\n", " "))
            if txt and not tn_text_unusable(txt):
                return txt
            if txt and tn_text_unusable(txt) and attempt + 1 < max_retries:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FIND_PROMPT.format(cat=category)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }]
                continue
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries}] {str(e)[:90]} (sleep {wait}s)")
            time.sleep(wait)
    return None


def _sidecar_key(eid: str, goal_index) -> str:
    return f"{eid}#{int(goal_index)}"


def _load_sidecar_keys(path: str) -> set:
    done = set()
    if not path or not os.path.isfile(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("tn_text"):
                done.add(_sidecar_key(rec["episode_id"], rec["goal_index"]))
    return done


def generate(args) -> dict:
    from openai import OpenAI

    with open(args.episodes) as f:
        data = json.load(f)
    episodes = data["episodes"]
    sidecar_done = _load_sidecar_keys(args.sidecar)
    only_keys = None
    if getattr(args, "only_goals", None):
        raw = json.loads(open(args.only_goals).read())
        items = raw.get("keys") or raw.get("items") or raw
        only_keys = set()
        for it in items:
            if isinstance(it, dict):
                only_keys.add((str(it["episode_id"]), int(it["goal_index"])))
            else:
                only_keys.add((str(it[0]), int(it[1])))

    # collect TN goals (optionally only those missing text), grouped by scene
    by_scene = defaultdict(list)
    n_total = n_skip = 0
    for ep in episodes:
        for g in ep["goals_set"]:
            if g["modality"] != "TN":
                continue
            if only_keys is not None and (ep["episode_id"], int(g["goal_index"])) not in only_keys:
                continue
            if args.sample_mode == "viewpoint" and not g.get("iin_viewpoint"):
                continue
            if args.sample_mode == "goat" and g.get("position") is None:
                continue
            n_total += 1
            key = _sidecar_key(ep["episode_id"], g["goal_index"])
            if not args.overwrite and (g.get("tn_text") or key in sidecar_done):
                n_skip += 1
                continue
            by_scene[ep["scene_id"]].append((ep["episode_id"], g))

    if args.n_shards > 1:
        scenes = sorted(by_scene)
        keep = {s for i, s in enumerate(scenes) if i % args.n_shards == args.shard_index}
        by_scene = {s: v for s, v in by_scene.items() if s in keep}

    todo = sum(len(v) for v in by_scene.values())
    if args.limit:
        # keep only the first --limit goals overall (across scenes)
        kept, scenes = 0, {}
        for sid, items in by_scene.items():
            take = items[: max(0, args.limit - kept)]
            if take:
                scenes[sid] = take
                kept += len(take)
            if kept >= args.limit:
                break
        by_scene = scenes
        todo = kept

    print(
        f"[gen_tn] TN goals={n_total} already_done={n_skip} "
        f"shard={args.shard_index}/{args.n_shards} to_generate={todo} "
        f"api_workers={args.api_workers} sample_mode={args.sample_mode}",
        flush=True,
    )
    client = OpenAI(api_key=args.api_key or get_llm_api_key(), base_url=args.base_url, timeout=120)
    sidecar_lock = threading.Lock()
    if args.sidecar:
        os.makedirs(os.path.dirname(os.path.abspath(args.sidecar)) or ".", exist_ok=True)

    done = fail = 0
    done_lock = threading.Lock()

    def _write_sidecar(eid, g, txt):
        if not args.sidecar:
            return
        rec = {
            "episode_id": eid,
            "goal_index": g["goal_index"],
            "object_category": g["object_category"],
            "tn_text": txt,
        }
        with sidecar_lock:
            with open(args.sidecar, "a") as sf:
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _on_text(eid, g, txt):
        nonlocal done, fail
        if txt:
            g["tn_text"] = txt
            _write_sidecar(eid, g, txt)
            with done_lock:
                done += 1
                n = done
            print(f"  [{n}/{todo}] {eid} g{g['goal_index']} {g['object_category']}: {txt}", flush=True)
        else:
            with done_lock:
                fail += 1
            print(f"  [FAIL] {eid} g{g['goal_index']} {g['object_category']}", flush=True)

    for scene_id, items in by_scene.items():
        glb = _scene_glb(args.scenes_dir, scene_id)
        sds = _scene_dataset_config(args.scenes_dir, glb)
        if args.sample_mode == "goat":
            cam = dict(_cfg.TN_CAMERA_GOAT)
            cam["height_m"] = 0.0
            cam["with_depth"] = True
            sim = _make_sim(
                glb, sds, cam,
                args.gpu_device_id, with_semantic=True,
            )
        else:
            sim = _make_sim(
                glb, sds,
                items[0][1]["iin_viewpoint"]["camera"], args.gpu_device_id,
            )
        try:
            rendered = []
            for eid, g in items:
                if args.sample_mode == "goat":
                    img, extra = pick_languagenav_view(sim, dict(_cfg.TN_CAMERA_GOAT), g)
                    g["_tn_sample"] = {
                        "frame_coverage": extra.get("frame_coverage"),
                        "languagenav": extra.get("languagenav"),
                        "n_nearby": len(extra.get("nearby_objects") or []),
                    }
                    backup = None
                    if only_keys is not None and g.get("position") is not None:
                        cam512 = dict(_cfg.IIN_CAMERA_STRETCH_512)
                        cam512["with_depth"] = True

                        class _RescueArgs:
                            standoff_m = 1.8
                            rescue = True

                        backup, _ex = _render_stretch512_goal(sim, cam512, g, _RescueArgs())
                    rendered.append((eid, g, img, extra.get("nearby_objects"), backup))
                else:
                    vp = g["iin_viewpoint"]
                    img = _render_at(sim, vp["position"], vp["rotation"])
                    rendered.append((eid, g, img, None, None))
        finally:
            sim.close()

        def _caption(eid, g, img, nearby, backup):
            txt = _describe(
                client, args.model, img, g["object_category"],
                sample_mode=args.sample_mode, nearby=nearby,
            )
            if (not txt or tn_text_unusable(txt)) and backup is not None:
                txt = _describe(
                    client, args.model, backup, g["object_category"],
                    sample_mode="viewpoint", nearby=None,
                )
            return txt

        if args.api_workers <= 1:
            for eid, g, img, nearby, backup in rendered:
                _on_text(eid, g, _caption(eid, g, img, nearby, backup))
            continue

        if args.save_renders_dir:
            import imageio

            os.makedirs(args.save_renders_dir, exist_ok=True)
            for eid, g, img, _nearby, _backup in rendered:
                fname = "%s__g%s_%s.png" % (eid, g["goal_index"], g["object_category"])
                imageio.imwrite(os.path.join(args.save_renders_dir, fname), img)

        with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
            futs = {
                pool.submit(_caption, eid, g, img, nearby, backup): (eid, g)
                for eid, g, img, nearby, backup in rendered
            }
            for fut in as_completed(futs):
                eid, g = futs[fut]
                try:
                    txt = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] {eid} g{g['goal_index']} {e}", flush=True)
                    txt = None
                _on_text(eid, g, txt)

    if not args.sidecar:
        out = args.out or args.episodes
        with open(out, "w") as f:
            json.dump(data, f)
        print(f"[gen_tn] generated={done} failed={fail} -> {out}", flush=True)
        return {"generated": done, "failed": fail, "out": out}

    print(f"[gen_tn] generated={done} failed={fail} sidecar={args.sidecar}", flush=True)
    return {"generated": done, "failed": fail, "sidecar": args.sidecar}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate TN goal descriptions via qwen VLM")
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", default="", help="default: overwrite --episodes in place")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--model", default=get_llm_model(), help="default: $LLM_MODEL or qwen3.5-27b")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default="", help="default: $LLM_API_KEY (fallback $DASHSCOPE_API_KEY)")
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all TN goals")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing tn_text")
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--api-workers", type=int, default=1, help="in-scene VLM API threads")
    p.add_argument("--sidecar", default="", help="append JSONL updates; do not rewrite --episodes")
    p.add_argument(
        "--sample-mode",
        default="goat",
        choices=["goat", "viewpoint"],
        help="goat=GOAT LanguageNav grid + qwen; viewpoint=stored iin_viewpoint",
    )
    p.add_argument("--save-renders-dir", default="", help="optional PNG dump of TN sample views")
    p.add_argument(
        "--only-goals",
        default="",
        help="JSON with keys=[[episode_id, goal_index], ...] to generate a subset",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    generate(build_argparser().parse_args(argv))


if __name__ == "__main__":
    main()
