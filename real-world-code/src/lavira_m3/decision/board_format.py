"""Goal identifiers and human/model-readable board text."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import ID_TO_ROBOT, ROBOT_IDS

NONE_PA = "none"


def goal_public_id(goal: dict, idx: int) -> str:
    return f"goal_{goal.get('goal_index', idx)}"


def parse_goal_index(goal_id: str, goals: List[dict]) -> Optional[int]:
    text = str(goal_id or "").strip()
    if not text or not goals:
        return None
    if text.lower().startswith("goal_"):
        tail = text.split("_", 1)[1]
        try:
            idx = int(tail)
        except ValueError:
            return None
        if 0 <= idx < len(goals):
            return idx
        for i, g in enumerate(goals):
            if int(g.get("goal_index", i)) == idx:
                return i
        return None
    try:
        idx = int(text)
    except ValueError:
        return None
    if 0 <= idx < len(goals):
        return idx
    return None


def _goal_line(goal: dict, idx: int, finder: int) -> str:
    gid = goal_public_id(goal, idx)
    modality = goal.get("modality", "ON")
    line = f"- {gid} [{modality}]: {goal.get('object_category')}"
    if modality == "TN" and goal.get("tn_text"):
        line += f"; description: {goal['tn_text']}"
    elif modality == "IIN":
        if finder >= 0:
            line += "; reference image already verified"
        elif goal.get("reference_image"):
            line += "; reference image attached in prompt"
        else:
            line += "; reference image unavailable"
        scene = " ".join(str(goal.get("scene_text") or "").split())
        if scene:
            line += f"; scene: {scene}"
    if finder >= 0:
        who = ID_TO_ROBOT.get(finder, f"agent_{finder}")
        line += f" — found by agent_{finder} ({who})"
    else:
        line += " — unresolved"
    return line


def build_board_text(goals: List[dict], found_by: List[int]) -> str:
    if not goals:
        return (
            "Targets (team object board; V2-verified FOUND only):\n- (empty board, not initialized)"
        )
    lines = [
        "Targets (team object board; V2-verified FOUND only):",
        "Already-found / claimed goals are done. Do not search, navigate toward, or claim them. Only unresolved goals remain.",
    ]
    for i, goal in enumerate(goals):
        finder = found_by[i] if i < len(found_by) else -1
        lines.append(_goal_line(goal, i, int(finder)))
    return "\n".join(lines)


def _pa_line(robot: str, rec: Dict[str, Any], current_step: int) -> str:
    agent_id = ROBOT_IDS.get(robot, -1)
    text = str((rec or {}).get("text") or "").strip()
    last_step = int((rec or {}).get("step", -1))
    label = f"agent_{agent_id}" if agent_id >= 0 else robot
    if not text or last_step < 0:
        return f"- {label}: {NONE_PA}"
    age = max(0, int(current_step) - last_step)
    return f"- {label} ({age} steps ago): {text}"


def build_teammate_progress(progress: Dict[str, Dict], robot: str, current_step: int) -> str:
    lines = [
        "Teammate claimed progress_analysis (for reasoning / action / STOP only; "
        "do not copy into progress_analysis):",
    ]
    others = [name for name in progress if name != robot]
    if not others:
        lines.append(f"- {NONE_PA}")
        return "\n".join(lines)
    for name in sorted(others):
        lines.append(_pa_line(name, progress[name], current_step))
    return "\n".join(lines)
