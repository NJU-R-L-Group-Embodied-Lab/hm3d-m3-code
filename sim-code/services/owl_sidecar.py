#!/usr/bin/env python3
"""OWLv2 + SuperGlue + CLIP detection sidecar for HM3D-M3.

Serves the detector FOUND gate (``ovon_od_detector_stop``, benchmark/od_found.py
and benchmark/ovon_od.py) of the four adapted baselines over HTTP:

  GET  /health        model / device / batching info
  POST /detect        OWLv2 open-vocabulary detection of one text query
  POST /detect_batch  several /detect items in one request
  POST /match_image   SuperPoint + SuperGlue match score against a goal image (IIN)
  POST /score_lang    CLIP image-text cosine similarity (TN)

Images arrive either as file-backed raw RGB slots in OWL_SHM_DIR (written by
benchmark/owl_shm.py; the client and the sidecar must see the same directory)
or as legacy JPEG+base64. Resize / normalize runs on the GPU with torchvision,
and concurrent /detect calls are batched so one process can serve several
Habitat workers.

Environment (all optional; command-line flags take precedence):
  OWL_MODEL            HF id or local dir   (google/owlv2-base-patch16-ensemble)
  CLIP_MODEL           HF id or local dir   (openai/clip-vit-base-patch32)
  OWL_DEVICE           torch device         (cuda)
  OWL_BIND_HOST, OWL_PORT  bind address     (127.0.0.1 / 18210)
  OWL_SHM_DIR          shared RGB slot directory
  OWL_MAX_BATCH, OWL_BATCH_WAIT_MS          batching knobs (8 / 8 ms)
  OWL_SUPERGLUE_DIR    checkout of SuperGluePretrainedNetwork with the HM3D-M3
                       weights-dir patch (default: sim-code/third_party/src/superglue,
                       created by third_party/fetch.sh)
  DM3_SUPERGLUE_WEIGHTS_DIR  directory with superpoint_v1.pth and
                       superglue_indoor.pth
                       (default: sim-code/data/datasets/dm3_official/superglue_weights)
  HF_HOME, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE  standard Hugging Face cache knobs
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import queue
import struct
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image


MAGIC = b"OWL1"
HEADER_FMT = "<4sIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
OWL_SIZE = 960


def _slot_path(shm_dir: str, name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
    return os.path.join(shm_dir, safe + ".rgb")


def _read_shm(shm_dir: str, name: str, expect_seq=None):
    path = _slot_path(shm_dir, name)
    if not os.path.isfile(path):
        return None
    data = np.memmap(path, dtype=np.uint8, mode="r")
    if data.size < HEADER_SIZE + 4:
        return None
    magic, h, w, c, seq, nbytes = struct.unpack(HEADER_FMT, bytes(data[:HEADER_SIZE]))
    if magic != MAGIC or h <= 0 or w <= 0 or c not in (1, 3, 4):
        return None
    end = HEADER_SIZE + int(nbytes)
    if data.size < end + 4:
        return None
    seq2 = struct.unpack("<I", bytes(data[end:end + 4]))[0]
    if seq != seq2:
        return None
    if expect_seq is not None and int(expect_seq) != int(seq):
        return None
    arr = np.array(data[HEADER_SIZE:end], dtype=np.uint8).reshape(int(h), int(w), int(c))
    return arr[..., :3]


def _load_owl(model_id: str, device: str):
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def _load_clip(model_id: str, device: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


_SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _superglue_root() -> str:
    """Locate a SuperGluePretrainedNetwork checkout (non-commercial license, user-supplied)."""
    env = os.environ.get("OWL_SUPERGLUE_DIR", "").strip()
    candidates = [env] if env else []
    candidates += [
        os.path.join(_SIM_ROOT, "third_party", "src", "superglue"),
        os.path.join(_SIM_ROOT, "third_party", "home_robot", "home_robot", "agent",
                     "imagenav_agent", "SuperGluePretrainedNetwork"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "models", "matching.py")):
            return cand
    raise FileNotFoundError(
        "SuperGluePretrainedNetwork not found (looked in: %s). Run "
        "`SUPERGLUE_DIR=... bash third_party/fetch.sh`, set OWL_SUPERGLUE_DIR, or start "
        "the sidecar with --skip-superglue / --no-match." % ", ".join(c for c in candidates if c)
    )


def _load_superglue(device: str):
    import torch

    sg_root = _superglue_root()
    if sg_root not in sys.path:
        sys.path.insert(0, sg_root)
    os.environ.setdefault(
        "DM3_SUPERGLUE_WEIGHTS_DIR",
        os.path.join(_SIM_ROOT, "data", "datasets", "dm3_official", "superglue_weights"),
    )
    from models.matching import Matching

    matcher = (
        Matching(
            {
                "superpoint": {
                    "nms_radius": 4,
                    "keypoint_threshold": 0.005,
                    "max_keypoints": 1024,
                },
                "superglue": {
                    "weights": "indoor",
                    "sinkhorn_iterations": 20,
                    "match_threshold": 0.2,
                },
            }
        )
        .eval()
        .to(device)
    )
    return matcher


def _b64_to_pil(raw_b64: str) -> Image.Image:
    raw = base64.b64decode(raw_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _b64_to_rgb(raw_b64: str) -> np.ndarray:
    return np.asarray(_b64_to_pil(raw_b64), dtype=np.uint8)


def _payload_rgb(payload: dict, prefix: str = "") -> np.ndarray:
    shm_name = payload.get(prefix + "shm_name") or (payload.get("shm_name") if not prefix else None)
    shm_dir = os.environ.get("OWL_SHM_DIR", "") or payload.get(prefix + "shm_dir") or payload.get("shm_dir") or ""
    seq = payload.get(prefix + "seq")
    if shm_name and shm_dir:
        rgb = _read_shm(str(shm_dir), str(shm_name), expect_seq=seq)
        if rgb is not None:
            return rgb
    key = "goal_image_b64" if prefix.startswith("goal") else "image_b64"
    if prefix == "goal_" and "goal_image_b64" in payload:
        key = "goal_image_b64"
    if payload.get(key):
        return _b64_to_rgb(payload[key])
    raise ValueError("no image in payload prefix=%r" % prefix)


def _pil_to_gray_tensor(image: Image.Image, device: str):
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = (
        0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    ) / 255.0
    return torch.from_numpy(gray)[None, None].to(device)


def _gpu_pixel_values(rgbs, device):
    """OWLv2 preprocess on GPU: /255, pad-to-square, blur+resize 960, CLIP norm."""
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.v2 import functional as tvF

    tensors = []
    for rgb in rgbs:
        img = np.ascontiguousarray(np.asarray(rgb)[..., :3])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        t = torch.from_numpy(img).to(device, non_blocking=True)
        t = t.permute(2, 0, 1).float().mul_(1.0 / 255.0)
        tensors.append(t)
    shapes = {(int(t.shape[-2]), int(t.shape[-1])) for t in tensors}
    if len(shapes) == 1:
        x = torch.stack(tensors, 0)
        _, _, h, w = x.shape
        side = max(h, w)
        if h != side or w != side:
            x = F.pad(x, (0, side - w, 0, side - h))
        factor = float(side) / float(OWL_SIZE)
        if factor > 1.0:
            sigma = max((factor - 1.0) / 2.0, 0.0)
            if sigma > 0:
                k = int(2 * math.ceil(3 * sigma) + 1)
                if k % 2 == 0:
                    k += 1
                x = tvF.gaussian_blur(x, [k, k], [sigma, sigma])
        x = F.interpolate(x, size=(OWL_SIZE, OWL_SIZE), mode="bilinear", align_corners=False)
        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        return x.sub_(mean).div_(std)
    out = []
    for t in tensors:
        _, h, w = t.shape
        side = max(h, w)
        if h != side or w != side:
            t = F.pad(t, (0, side - w, 0, side - h))
        factor = float(side) / float(OWL_SIZE)
        if factor > 1.0:
            sigma = max((factor - 1.0) / 2.0, 0.0)
            if sigma > 0:
                k = int(2 * math.ceil(3 * sigma) + 1)
                if k % 2 == 0:
                    k += 1
                t = tvF.gaussian_blur(t.unsqueeze(0), [k, k], [sigma, sigma]).squeeze(0)
        t = F.interpolate(t.unsqueeze(0), size=(OWL_SIZE, OWL_SIZE), mode="bilinear", align_corners=False)
        for c, (m, s) in enumerate(zip(CLIP_MEAN, CLIP_STD)):
            t[0, c].sub_(m).div_(s)
        out.append(t)
    return torch.cat(out, 0)


class DetectBatcher:
    def __init__(self, infer_fn, max_batch: int, wait_s: float):
        self.infer_fn = infer_fn
        self.max_batch = max(1, int(max_batch))
        self.wait_s = max(0.0, float(wait_s))
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, item):
        fut = queue.Queue(maxsize=1)
        self.q.put((item, fut))
        ok, value = fut.get()
        if not ok:
            raise value
        return value

    def _loop(self):
        while True:
            item, fut = self.q.get()
            batch = [(item, fut)]
            deadline = time.time() + self.wait_s
            while len(batch) < self.max_batch:
                timeout = deadline - time.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=timeout))
                except queue.Empty:
                    break
            items = [x for x, _ in batch]
            try:
                results = self.infer_fn(items)
                if len(results) != len(batch):
                    raise RuntimeError("batch size mismatch")
                for (_, f), r in zip(batch, results):
                    f.put((True, r))
            except Exception as exc:
                for _, f in batch:
                    f.put((False, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("OWL_BIND_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OWL_PORT", "18210")))
    parser.add_argument(
        "--model",
        default=os.environ.get("OWL_MODEL", "google/owlv2-base-patch16-ensemble"),
    )
    parser.add_argument(
        "--clip-model",
        default=os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32"),
    )
    parser.add_argument("--device", default=os.environ.get("OWL_DEVICE", "cuda"))
    parser.add_argument(
        "--shm-dir",
        default=os.environ.get("OWL_SHM_DIR", ""),
        help="default dir for shm_name if payload omits shm_dir",
    )
    parser.add_argument("--max-batch", type=int, default=int(os.environ.get("OWL_MAX_BATCH", "8")))
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=float(os.environ.get("OWL_BATCH_WAIT_MS", "8")),
    )
    parser.add_argument(
        "--no-match",
        action="store_true",
        help="skip SuperGlue + CLIP load (ON-only)",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="skip CLIP (IIN does not score language)",
    )
    parser.add_argument(
        "--skip-superglue",
        action="store_true",
        help="skip SuperGlue (TN does not match goal images)",
    )
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    if args.shm_dir:
        os.environ["OWL_SHM_DIR"] = args.shm_dir
        os.makedirs(args.shm_dir, exist_ok=True)

    import torch

    owl_processor, owl_model = _load_owl(args.model, args.device)
    clip_processor = clip_model = matcher = None
    if not args.no_match:
        if not args.skip_clip:
            clip_processor, clip_model = _load_clip(args.clip_model, args.device)
        if not args.skip_superglue:
            matcher = _load_superglue(args.device)

    app = Flask(__name__)
    infer_lock = threading.Lock()
    text_cache = {}
    post = getattr(owl_processor, "post_process_grounded_object_detection", None)
    if post is None:
        post = owl_processor.post_process_object_detection

    def _text_inputs(texts):
        key = tuple(t[0] if t else "" for t in texts)
        cached = text_cache.get(key)
        if cached is not None:
            return {k: v.clone() for k, v in cached.items()}
        # Owlv2Processor cats per-image encodings without padding token
        # length. Mixed queries ("chair" vs "radiator") then crash the batcher.
        flat = [t[0] if t else "" for t in texts]
        tok = owl_processor.tokenizer(flat, return_tensors="pt", padding=True)
        packed = {k: v.to(args.device) for k, v in tok.items() if k != "pixel_values"}
        if len(text_cache) < 256:
            text_cache[key] = {k: v.detach() for k, v in packed.items()}
        return packed

    def _infer_detect_batch(items):
        rgbs = [it["rgb"] for it in items]
        texts = [[str(it["text"])] for it in items]
        thresholds = [float(it["threshold"]) for it in items]
        sizes = [(int(r.shape[0]), int(r.shape[1])) for r in rgbs]
        with infer_lock:
            pix = _gpu_pixel_values(rgbs, args.device)
            inputs = _text_inputs(texts)
            inputs["pixel_values"] = pix
            with torch.no_grad():
                outputs = owl_model(**inputs)
            target_sizes = torch.tensor(sizes, device=args.device)
            raw = post(outputs=outputs, threshold=0.0, target_sizes=target_sizes)
        out = []
        for i, rec in enumerate(raw):
            thr = thresholds[i]
            boxes = rec["boxes"].detach().cpu()
            scores = rec["scores"].detach().cpu()
            keep = scores >= thr
            out.append(
                {
                    "boxes": boxes[keep].tolist(),
                    "scores": scores[keep].tolist(),
                }
            )
        return out

    batcher = DetectBatcher(
        _infer_detect_batch,
        max_batch=args.max_batch,
        wait_s=args.batch_wait_ms / 1000.0,
    )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "ok": True,
                "model": args.model,
                "clip": None if (args.no_match or args.skip_clip) else args.clip_model,
                "superglue": matcher is not None,
                "device": args.device,
                "threaded": True,
                "shm_dir": args.shm_dir or os.environ.get("OWL_SHM_DIR", ""),
                "gpu_preprocess": True,
                "max_batch": int(args.max_batch),
                "batch_wait_ms": float(args.batch_wait_ms),
            }
        )

    @app.route("/detect", methods=["POST"])
    def detect():
        payload = request.get_json(force=True)
        text = str(payload.get("text", "")).strip()
        threshold = float(payload.get("threshold", 0.2))
        if not text:
            return jsonify({"boxes": [], "scores": []})
        rgb = _payload_rgb(payload)
        result = batcher.submit({"rgb": rgb, "text": text, "threshold": threshold})
        return jsonify(result)

    @app.route("/detect_batch", methods=["POST"])
    def detect_batch():
        payload = request.get_json(force=True)
        items = payload.get("items") or []
        parsed = []
        for it in items:
            text = str(it.get("text", "")).strip()
            if not text:
                parsed.append({"boxes": [], "scores": []})
                continue
            parsed.append(
                {
                    "rgb": _payload_rgb(it),
                    "text": text,
                    "threshold": float(it.get("threshold", 0.2)),
                }
            )
        need = [p for p in parsed if "rgb" in p]
        if need:
            recs = _infer_detect_batch(need)
            ri = 0
            out = []
            for p in parsed:
                if "rgb" in p:
                    out.append(recs[ri])
                    ri += 1
                else:
                    out.append(p)
        else:
            out = parsed
        return jsonify({"results": out})

    @app.route("/match_image", methods=["POST"])
    def match_image():
        if matcher is None:
            return jsonify({"error": "superglue not loaded"}), 400
        payload = request.get_json(force=True)
        frame = _payload_rgb(payload)
        try:
            goal = _payload_rgb(payload, prefix="goal_")
        except ValueError:
            goal = _b64_to_rgb(payload["goal_image_b64"])
        img0 = _pil_to_gray_tensor(Image.fromarray(goal), args.device)
        img1 = _pil_to_gray_tensor(Image.fromarray(frame), args.device)
        with infer_lock:
            with torch.no_grad():
                pred = matcher({"image0": img0, "image1": img1})
            matches = pred["matches0"][0].detach().cpu().numpy()
            conf = pred["matching_scores0"][0].detach().cpu().numpy()
        keep = matches != -1
        score = float(conf[keep].sum()) if keep.any() else 0.0
        return jsonify({"score": score, "n_matches": int(keep.sum())})

    @app.route("/score_lang", methods=["POST"])
    def score_lang():
        if clip_model is None:
            return jsonify({"error": "clip not loaded"}), 400
        payload = request.get_json(force=True)
        text = str(payload.get("text", "")).strip()
        image = Image.fromarray(_payload_rgb(payload))
        if not text:
            return jsonify({"score": 0.0})
        with infer_lock:
            inputs = clip_processor(
                text=[text], images=image, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = clip_model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            score = float((ie * te).sum().item())
        return jsonify({"score": score})

    print(
        "OWL sidecar ready",
        args.model,
        "clip" if clip_model is not None else "no-clip",
        "sg" if matcher is not None else "no-sg",
        args.device,
        "gpu-pre",
        "batch=%d/%sms" % (args.max_batch, args.batch_wait_ms),
        "shm=%s" % (args.shm_dir or os.environ.get("OWL_SHM_DIR", "")),
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True, processes=1)


if __name__ == "__main__":
    main()
