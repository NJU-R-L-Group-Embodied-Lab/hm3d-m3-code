"""Pure image encoding, response parsing and LA/VA message assembly."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .config import (
    PANO_MAX_DIM,
    VA_MAX_DIM,
)
from .prompts_real import (
    CURRENT_VIEWS_SECTION_LABEL,
    HISTORY_SECTION_LABEL,
    INITIAL_VIEWS_SECTION_LABEL,
    history_text,
    origin_views_text,
)


def encode_image(image: Any, max_dim: int = PANO_MAX_DIM) -> str:
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        # ROS / OpenCV is often BGR
        if image.shape[2] == 3:
            image = image[:, :, ::-1]
        image = Image.fromarray(image)
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def image_part(image: Any, max_dim: int = PANO_MAX_DIM) -> Dict:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + encode_image(image, max_dim)},
    }


def parse_json_object(text: str) -> Dict:
    if not text:
        return {}
    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        text = re.sub(r"^json", "", text, flags=re.I).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"raw": text}
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob = blob.replace("True", "true").replace("False", "false")
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return {"raw": text}


def build_la_content(prompt, origin_views, history, frames, reference_parts):
    content: List[Dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": INITIAL_VIEWS_SECTION_LABEL})
    content.append({"type": "text", "text": origin_views_text(origin_views or [])})
    for rec in origin_views or []:
        name = str(rec.get("name") or "?")
        if rec.get("rgb") is None:
            content.append({"type": "text", "text": f"Initial {name}: NO IMAGE"})
            continue
        pose = rec.get("pose")
        pose_txt = ""
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            pose_txt = " @ (%.2f, %.2f)" % (float(pose[0]), float(pose[1]))
        content.append({"type": "text", "text": f"Initial {name}{pose_txt}"})
        content.append(image_part(rec["rgb"]))
    content.append({"type": "text", "text": HISTORY_SECTION_LABEL})
    content.append({"type": "text", "text": history_text(history)})
    for rec in history:
        kind = str(rec.get("kind") or "nav")
        if rec.get("rgb") is None:
            content.append(
                {
                    "type": "text",
                    "text": f"Frame {rec.get('waypoint_id', '?')} front ({kind}): NO IMAGE",
                }
            )
            continue
        pose = rec.get("pose")
        pose_txt = ""
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            pose_txt = " @ (%.2f, %.2f)" % (float(pose[0]), float(pose[1]))
        content.append(
            {
                "type": "text",
                "text": f"Frame {rec.get('waypoint_id', '?')} front{pose_txt} ({kind})",
            }
        )
        content.append(image_part(rec["rgb"]))
    content.append({"type": "text", "text": CURRENT_VIEWS_SECTION_LABEL})
    for fr in frames:
        if fr.get("rgb") is None:
            content.append({"type": "text", "text": f"{fr['name']}: NO IMAGE"})
            continue
        content.append({"type": "text", "text": f"View {fr['name']}"})
        content.append(image_part(fr["rgb"]))
    content.extend(reference_parts)
    return content


def build_va_content(prompt, view, reference_parts):
    content = [
        {"type": "text", "text": prompt},
        image_part(view["rgb"], VA_MAX_DIM),
    ]
    if view.get("depth") is not None:
        import numpy as np

        arr = np.asarray(view["depth"], dtype=np.float32)
        if arr.max() > 20:
            arr = arr / 1000.0
        arr = np.clip(arr / 5.0, 0, 1)
        gray = (arr * 255).astype("uint8")
        content.append({"type": "text", "text": "aligned depth (dark=near)"})
        content.append(image_part(gray, VA_MAX_DIM))
    content.extend(reference_parts)
    return content
