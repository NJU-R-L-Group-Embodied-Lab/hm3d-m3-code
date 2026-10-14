"""English prompts for configured camera directions."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from lavira_m3.profiles import AGENTS, profile

from .config import FOUND_MAX_DEPTH_M, claim_max_m_for, la_action_dirs

DOOR_RULE = (
    "Never choose a glass door. Treat a glass door as an impassable wall. "
    "Every closed door is not enterable and cannot be opened. "
    "Only an already-open non-glass doorway or open passage is a valid entrance or exit."
)

PEER_ROBOT_RULE = (
    "Other robots are never a search target, waypoint, or boxed object. "
    "Do not choose, follow, approach, box, or claim "
    "any configured teammate. "
    "If one of these robots is visible, prefer another direction. "
    "Prefer not to follow, approach, or cluster with a teammate."
)

HALL_SEARCH_RULE = (
    "Board objects are more often visible in the open hall "
    "(cubicle / workstation aisles and the kitchenette / lounge edge) "
    "than behind side-room or glass-office doors. "
    "A doorway into another room is usually a poor landmark."
)

PEER_SPREAD_RULE = PEER_ROBOT_RULE

SHARED_CONSTRAINTS = "\n".join(
    (
        f"- {HALL_SEARCH_RULE}",
        f"- {DOOR_RULE}",
        f"- {PEER_ROBOT_RULE}",
        "- Search only unresolved targets. Do not search, navigate toward, or claim a goal already marked found on the board.",
    )
)


def _std_prompt(role: str, context: str, instructions: str, constraints: str, output: str) -> str:
    return "\n".join(
        (
            "# Role",
            role.strip(),
            "",
            "# Context",
            context.strip(),
            "",
            "# Instructions",
            instructions.strip(),
            "",
            "# Constraints",
            constraints.strip(),
            "",
            "# Output",
            output.strip(),
        )
    )


BOARD_PROTOCOL = (
    "You are {who}. Search only unresolved targets on the object board. "
    "Goals already marked found / claimed on the board are finished."
)

INITIAL_VIEWS_SECTION_LABEL = (
    "\n# Initial Panorama\n"
    "=== INITIAL PANORAMA ===\n"
    "Start RGB of this session. Use with Navigation History for progress_analysis only. "
    "Not the current panorama. Views are labelled with their configured directions."
)

HISTORY_SECTION_LABEL = (
    "\n# Navigation History\n"
    "=== NAVIGATION HISTORY ===\n"
    "Body-front RGB every 5 s while moving, in time order. "
    "progress_analysis uses Initial Panorama plus this sequence only. If empty, say so."
)

CURRENT_VIEWS_SECTION_LABEL = (
    "\n# Current Panorama\n"
    "=== CURRENT PANORAMA ===\n"
    "Do not copy these frames into progress_analysis."
)

IMAGE_GOAL_SECTION_LABEL = (
    "\n# Image Goals\n"
    "=== IMAGE GOALS ===\n"
    "IIN photos. For each photo, describe the PLACE (room type, furniture, nearby objects) "
    "in image_goal_scenes. Use the same photo as identity. Do not invent map coordinates."
)


def image_goal_scenes_block(goals: Optional[Sequence[Dict]] = None) -> str:
    lines = []
    for i, goal in enumerate(goals or []):
        text = " ".join(str(goal.get("scene_text") or "").split())
        if not text:
            continue
        gid = f"goal_{goal.get('goal_index', i)}"
        lines.append(f"- {gid}: {text}")
    if not lines:
        return ""
    return "Image-goal scenes:\n" + "\n".join(lines)


def normalize_image_goal_scenes(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        text = " ".join(str(value or "").split())
        if not text:
            continue
        token = str(key or "").strip().lower()
        if token.startswith("goal_"):
            gid = token
        else:
            try:
                gid = "goal_%d" % int(token)
            except (TypeError, ValueError):
                continue
        out[gid] = text
    return out


def _meters_txt(claim_max_m: float) -> str:
    m = float(claim_max_m)
    if abs(m - round(m)) < 1e-6:
        return str(int(round(m)))
    return f"{m:.1f}"


def _bullet(label: str, value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if "\n" in text:
        return f"{label}:\n{text}\n"
    return f"- {label}: {text}\n"


def dcs_reject_feedback(claim_max_m: float = FOUND_MAX_DEPTH_M) -> str:
    del claim_max_m
    return (
        "System: previous claim failed the double-check (not visible, or not that target). "
        "You may look again if it is still visible, "
        "or keep searching other unresolved goals. Do not re-claim a board-found goal."
    )


DCS_REJECT_FEEDBACK = dcs_reject_feedback(FOUND_MAX_DEPTH_M)

AFTER_FOUND_SWITCH_FEEDBACK = (
    "System: last claim passed the double-check and is now found on the board. "
    "Do not navigate toward that object again. "
    "Do not claim a different goal_id for the object you just found."
)

NAV_BLOCKED_FEEDBACK = (
    "System: last VA map point was not executable (on or near an obstacle). "
    "Pick free space. Prefer not repeating that doorway, wall, glass door, closed door, or robot."
)

SHORT_HOP_VA_IOU_MAX = 0.3


def short_hop_la_feedback(
    prev_la: Optional[Dict[str, Any]] = None, prev_va: Optional[Dict[str, Any]] = None
) -> str:
    """Inject last LA/VA and forbid reselecting the same object after a short hop."""
    prev_la = prev_la or {}
    prev_va = prev_va or {}
    action = str(prev_la.get("action") or "").strip() or "unknown"
    claim = str(prev_la.get("claim_goal_id") or "").strip() or "none"
    direction = str(prev_va.get("direction") or prev_la.get("direction") or "").strip() or "unknown"
    target = str(prev_va.get("target") or "").strip() or "unknown"
    bbox = prev_va.get("bbox_2d")
    bbox_txt = json.dumps(bbox) if isinstance(bbox, (list, tuple)) else "none"
    return (
        "System: last move was shorter than 1.5 m; reuse the previous panorama.\n"
        "Previous LA: action=%s claim_goal_id=%s direction=%s\n"
        "Previous VA: target=%s bbox_2d=%s\n"
        "If last cycle's claim_goal_id is still unresolved and that object is still visible, "
        "prefer that same claim_goal_id rather than walking away or remapping it. "
        "If that previous claim is already found, prefer leaving it. "
        "If last cycle was only a search landmark and not a board goal, "
        "prefer a different legal direction or a different object."
        % (action, claim, direction, target, bbox_txt)
    )


def short_hop_va_feedback(prev_va: Optional[Dict[str, Any]] = None, direction: str = "") -> str:
    prev_va = prev_va or {}
    prev_dir = str(prev_va.get("direction") or "").strip()
    target = str(prev_va.get("target") or "").strip() or "unknown"
    bbox = prev_va.get("bbox_2d")
    bbox_txt = json.dumps(bbox) if isinstance(bbox, (list, tuple)) else "none"
    same = bool(prev_dir) and str(direction) == prev_dir
    extra = (
        " You are looking in the SAME direction as last cycle. "
        "Box a different object. Do not reuse the previous bbox "
        "(IoU with the previous box must stay under 0.3) and do not repeat the previous target name."
        if same
        else " This is a new direction after a short hop; do not box the previous object."
    )
    return "System: short-hop replan. Previous VA target=%s bbox_2d=%s direction=%s.%s" % (
        target,
        bbox_txt,
        prev_dir or "unknown",
        extra,
    )


def action_list(dirs: Sequence[str], extra_actions: Optional[Sequence[str]] = None) -> str:
    lines = [f'- "navigate to {d}"' for d in dirs]
    for action in extra_actions or []:
        text = str(action or "").strip()
        if text:
            lines.append(f'- "{text}"')
    return "\n".join(lines)


AFTER_BACKTRACK_FEEDBACK = (
    "System: you just returned to an earlier reached point. "
    "Prefer not immediately going back along the same path you just left."
)

BACKTRACK_STUCK_FEEDBACK = (
    "System: recent moves made little progress, or forward views look blocked. "
    "A listed backtrack is available if you want to leave this pocket. "
    "You do not have to choose it."
)

SCB_FAIL_HIDE_CYCLES = 2


def scb_walk_failed_feedback(idx: int, state: str = "") -> str:
    why = str(state or "failed").strip() or "failed"
    return (
        "System: last backtrack to %s did not walk (%s). "
        "Do not immediately repeat that same backtrack id. "
        "Prefer a different legal direction or another listed id." % (int(idx), why)
    )


def backtrack_prompt_block(items: Sequence[tuple], max_m: float) -> str:
    """items: (id, (x, y), dist_m). Soft option only; never required."""
    if not items:
        return ""
    ids = ", ".join(str(int(i)) for i, _, _ in items)
    lines = [
        f"Reached points within {float(max_m):.0f} m (id from 0; nav-success poses only):",
        f"Available IDs: {ids}",
    ]
    for i, xy, dist in items:
        lines.append(
            '- "backtrack to %s"  (%.2f, %.2f)  %.1f m'
            % (int(i), float(xy[0]), float(xy[1]), float(dist))
        )
    lines.append(
        "If the current pocket looks closed or already searched, you may backtrack to one of these IDs. "
        "You do not have to. Only IDs with an archived arrival RGB panorama are listed."
    )
    return "\n".join(lines)


def la_backtrack_replan_prompt(
    who: str,
    dirs: Sequence[str],
    instruction: str = "",
    previous_action: str = "",
    failed_path_note: str = "",
    hardware_dirs: Optional[Sequence[str]] = None,
) -> str:
    """Second-chance LA at an archived waypoint. Direction only; no backtrack, no claim."""
    cam_dirs = hardware_dirs if hardware_dirs is not None else dirs
    action_dirs = la_action_dirs(dirs)
    task = (instruction or "").strip() or "Search only unresolved targets on the object board."
    prev = (previous_action or "unknown").strip() or "unknown"
    failed = (failed_path_note or "").strip() or "None"
    context = (
        f"- Task: {task}\n"
        f"- Robot: {who}\n"
        f"- Cameras: {', '.join(str(x) for x in cam_dirs)}\n"
        f"- Previous Action from this waypoint: navigate to {prev}\n"
        f"- Previous trajectory after this waypoint: {failed}\n"
        f"Legal actions:\n{action_list(action_dirs)}"
    )
    instructions = (
        "You have a second chance to choose a direction from this earlier waypoint. "
        "The 4/7-directional images below are the archived views from that waypoint, not the live cameras. "
        "Review the previous trajectory from here, then pick ONE legal direction. "
        "The previous choice can be reconsidered. "
        "Do not choose backtrack. Do not set claim_goal_id; a later double-check uses live front views only."
    )
    constraints = "\n".join(
        (
            SHARED_CONSTRAINTS,
            "- After T0, rear-facing cameras are not legal actions. "
            "Do not choose behind, left_back, or right_back.",
            "- Prefer not immediately repeating the Previous Action if that path failed.",
            "- If you believe the target is already at this waypoint, set stop true. "
            "FOUND still requires a later live-front double-check after the robot arrives.",
        )
    )
    output = (
        "JSON only:\n"
        "{\n"
        '    "reasoning": "<why this new direction>",\n'
        '    "action": "navigate to <direction>",\n'
        '    "stop": false\n'
        "}"
    )
    return _std_prompt(
        f"You are {who}. You have backtracked to a previous waypoint for a second-chance action.",
        context,
        instructions,
        constraints,
        output,
    )


def la_prompt(
    who: str,
    dirs: Sequence[str],
    board: str,
    teammate_pa: str,
    extra_feedback: str = "",
    hardware_dirs: Optional[Sequence[str]] = None,
    blocked_note: str = "",
    current_step: int = 0,
    instruction: str = "",
    claim_max_m: Optional[float] = None,
    image_goal_scenes: str = "",
    t0_sector: str = "",
    teammate_live: str = "",
    extra_actions: Optional[Sequence[str]] = None,
    backtrack_note: str = "",
) -> str:
    cam_dirs = hardware_dirs if hardware_dirs is not None else dirs
    if "left_back" in cam_dirs:
        cameras = (
            "7 directions and NO true BACK (180°) image: "
            "front, left_front, left, left_back, right_back, right, right_front."
        )
    elif "behind" in cam_dirs:
        cameras = "four fixed cameras: front, left, behind, right."
    else:
        cameras = ", ".join(str(x) for x in cam_dirs)
    action_dirs = la_action_dirs(dirs)
    note = blocked_note.strip()
    if note and note.lower().startswith("near-range mask"):
        mask = note
    elif note:
        mask = f"Near-range mask: {note}"
    else:
        mask = ""
    near = _meters_txt(claim_max_m if claim_max_m is not None else claim_max_m_for(who))
    task = (instruction or "").strip() or "Search only unresolved targets on the object board."
    context = (
        f"- Task: {task}\n"
        f"- Step: {int(current_step)}\n"
        f"- Robot: {who}\n"
        f"- Cameras: {cameras}\n"
        f"{_bullet('Mask', mask)}"
        f"{_bullet('Object board', board or 'board unavailable this cycle')}"
        f"{_bullet('Image-goal scenes', image_goal_scenes)}"
        f"{_bullet('T0 sector', t0_sector)}"
        f"{_bullet('Teammates', teammate_live)}"
        f"{_bullet('Teammate progress_analysis', teammate_pa or 'none')}"
        f"{_bullet('Feedback', extra_feedback)}"
        f"{_bullet('Backtrack', backtrack_note)}"
        f"Legal actions:\n{action_list(action_dirs, extra_actions)}"
    )
    instructions = (
        "1. Write progress_analysis from YOUR Initial Panorama plus Navigation History only. "
        "If the motion history is empty, say so and use the Initial Panorama. "
        "Do not caption current panorama frames. Do not mention teammate claims.\n"
        "2. Choose ONE legal action. T0 sector and teammate poses are context. "
        "Prefer not walking toward a teammate pose or waypoint. "
        "A nearby clear unresolved object may be worth a closer look; a tiny far blob usually is not."
    )
    extras = [str(a).strip() for a in (extra_actions or []) if str(a).strip()]
    if extras:
        instructions += (
            " If backtrack IDs are listed, they are an optional way to leave a closed or already-searched pocket; "
            "do not treat them as required."
        )
    instructions += (
        f"\n3. claim_goal_id may be set when one unresolved goal is clearly visible now "
        f"and looks about {near} m or nearer, or already large in a current view, "
        "and a teammate does not already appear to be on that object. "
        "If it is only a tiny far blob, leaving claim empty is fine."
    )
    constraints = "\n".join(
        (
            SHARED_CONSTRAINTS,
            "- Prefer not re-exploring places you already walked. Use Initial Panorama and Navigation History as memory.",
            "- After T0, rear-facing cameras are not legal actions. "
            "Do not choose behind, left_back, or right_back. Rear images are observation only.",
            "- reasoning, action, and claim_goal_id may use the board, T0 sector, teammate poses, "
            "teammate progress_analysis (already-searched places only), references, "
            "Initial Panorama, Navigation History, and current views.",
            "- A later double-check uses only your current views and must output FOUND before the claim is reported.",
            "- After a failed claim, keep searching remaining unresolved targets. "
            "After a passed claim, prefer leaving that object rather than remapping it.",
            "- The object board and teammate progress_analysis are not claim evidence.",
            "- If Image-goal scenes are given, they may hint which place an object belongs to. "
            "Confirm the object with the attached reference image. Do not invent map coordinates.",
        )
    )
    action_schema = (
        "navigate to <direction> or backtrack to <id>" if extras else "navigate to <direction>"
    )
    output = (
        "JSON only:\n"
        "{\n"
        '    "progress_analysis": "<what YOU observed and did from Initial Panorama and motion history only>",\n'
        '    "reasoning": "<why this action>",\n'
        f'    "action": "{action_schema}",\n'
        '    "claim_goal_id": "<goal_id or empty>"\n'
        "}"
    )
    return _std_prompt(
        f"You are {who}, a language-action planner. Search only unresolved targets on the object board.",
        context,
        instructions,
        constraints,
        output,
    )


def va_prompt(
    instruction: str,
    direction: str,
    progress: str,
    current_step: int = 0,
    width: int = 640,
    height: int = 480,
    image_goal_scenes: str = "",
    extra_feedback: str = "",
    t0_sector: str = "",
    teammate_live: str = "",
) -> str:
    context = (
        f"- Task: {instruction}\n"
        f"- Step: {int(current_step)}\n"
        f"- Direction: {direction}\n"
        f"- Image size: {int(width)}x{int(height)} pixels\n"
        "- Bounding box coordinates: 0..999 (not raw pixels)\n"
        f"- Progress Info: {progress}\n"
        f"{_bullet('Image-goal scenes', image_goal_scenes)}"
        f"{_bullet('T0 sector', t0_sector)}"
        f"{_bullet('Teammates', teammate_live)}"
        f"{_bullet('Feedback', extra_feedback)}"
    )
    instructions = (
        "1. Box ONE useful next landmark in the current view.\n"
        "2. A clearly visible unresolved board object can be a useful box; "
        "a teammate, a found goal, a glass door, or a closed door usually is not. "
        "A side-room doorway is usually a poor landmark. "
        "Do not box a glass door, a closed door, or another robot.\n"
        "3. Provide the bounding box of the target."
    )
    constraints = "\n".join(
        (
            SHARED_CONSTRAINTS,
            "- If you box a board goal, box the object itself, even if it is already close. "
            "Do not box floor or a doorway instead of that object.",
            "- If an Image-goal scene is given, it may hint where that object belongs. "
            "Do not box empty floor instead of a visible object.",
            "- Do not box or aim at a goal already marked found on the board.",
            "- Never output a zero box such as [0,0,0,0].",
        )
    )
    output = (
        "JSON only:\n"
        "{\n"
        '    "reasoning": "<brief, under 128 tokens>",\n'
        '    "bbox_2d": [x1, y1, x2, y2],\n'
        '    "target": "<short description>"\n'
        "}"
    )
    return _std_prompt(
        "You are a visual-action planner. Box one landmark in the current camera view.",
        context,
        instructions,
        constraints,
        output,
    )


def dcs_target_context(claim_goal_id: str, goal: Optional[Dict[str, Any]] = None) -> str:
    """ON name, TN description, or IIN reference image. Id alone is not enough."""
    goal = goal if isinstance(goal, dict) else {}
    modality = str(goal.get("modality") or "ON").strip().upper() or "ON"
    name = " ".join(str(goal.get("object_category") or "").split())
    description = " ".join(str(goal.get("tn_text") or "").split())
    has_ref = bool(goal.get("reference_image") or goal.get("reference_image_b64"))
    lines = [f"- Proposed claim: {claim_goal_id}"]
    if modality == "TN":
        lines.append("- Modality: TN (text)")
        lines.append(f"- Description: {description or name or 'unavailable'}")
        evidence = "the current panoramic views only"
    elif modality == "IIN":
        lines.append("- Modality: IIN (image)")
        if name:
            lines.append(f"- Name: {name}")
        lines.append(
            "- Reference: the attached reference image"
            if has_ref
            else "- Reference: image unavailable"
        )
        evidence = (
            "the current panoramic views and the attached reference image"
            if has_ref
            else "the current panoramic views only"
        )
    else:
        lines.append("- Modality: ON (name)")
        lines.append(f"- Name: {name or 'unavailable'}")
        evidence = "the current panoramic views only"
    lines.append(f"- Evidence: {evidence}.")
    return "\n".join(lines)


def stop_check_prompt(
    instruction: str,
    claim_goal_id: str,
    claim_max_m: float = FOUND_MAX_DEPTH_M,
    goal: Optional[Dict[str, Any]] = None,
) -> str:
    del instruction, claim_max_m
    context = dcs_target_context(claim_goal_id, goal)
    instructions = (
        "FOUND if that claimed object appears in the current panorama. CONTINUE if it does not."
    )
    constraints = (
        "- Identify the claimed object from the name, description, or reference image above.\n"
        "- Decide only from the current panoramic views whether that object appears now.\n"
        "- The claim is reported only if you output FOUND."
    )
    output = (
        "JSON only:\n"
        "{\n"
        '  "reasoning": "<brief, under 40 tokens>",\n'
        '  "decision": "FOUND" or "CONTINUE"\n'
        "}"
    )
    return _std_prompt(
        "You are the double-check. The claim is reported only if you output FOUND.",
        context,
        instructions,
        constraints,
        output,
    )


def parse_direction(action: str, allowed: Sequence[str]) -> str:
    text = (action or "").lower().replace("_", " ").replace("-", " ")
    if "backtrack" in text:
        return "backtrack"
    allowed = [str(d) for d in allowed]
    if not allowed:
        return "front"
    for name in sorted(allowed, key=len, reverse=True):
        token = name.replace("_", " ")
        if token in text or name in (action or "").lower():
            return name
    aliases = {
        "forward": "front",
    }
    for src, dst in aliases.items():
        if src in text and dst in allowed:
            return dst
    return allowed[0]


def parse_backtrack_id(action: str) -> int:
    text = action or ""
    digits = []
    collecting = False
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
            collecting = True
        elif collecting:
            break
    if not digits:
        return -1
    return int("".join(digits))


T0_MAP2D_CAPTION = (
    "Initial shared occupancy map: gray is unknown; black is occupied; purple is shared evidence. "
    "Colored markers identify the configured agents; short strokes show heading. "
    "+X points right and +Y points up. Use the map for initial spatial assignment."
)


def t0_view_caption(robot: str, view_name: str) -> str:
    from .config import ROBOT_IDS, VIEW_YAW

    views = profile(robot)["directions"]
    yaw = VIEW_YAW[view_name]
    return (
        f'Image: agent_{ROBOT_IDS[robot]} ({robot}), current RGB view "{view_name}". '
        f"Body-relative yaw: {yaw} degrees (positive=left). "
        f"Configured views: {', '.join(views)}."
    )


def t0_joint_la_prompt(
    robots: Sequence[str],
    board: str,
    legal_dirs_by_robot: Optional[Dict[str, Sequence[str]]] = None,
    blocked_notes: Optional[Dict[str, str]] = None,
) -> str:
    from .config import ROBOT_IDS, claim_max_m_for, la_action_dirs

    robot_blocks = []
    for name in robots:
        default_dirs = tuple(profile(name)["directions"])
        dirs = la_action_dirs(list((legal_dirs_by_robot or {}).get(name) or default_dirs))
        agent = f"agent_{ROBOT_IDS.get(name, -1)}"
        note = str((blocked_notes or {}).get(name) or "").strip()
        block = f"- {agent} ({name}) legal actions:\n{action_list(dirs)}"
        if note:
            block += "\n  " + " ".join(note.split())
        robot_blocks.append(block)
    keys = ", ".join(robots)
    near = _meters_txt(claim_max_m_for(next(iter(AGENTS))))
    context = (
        f"{board or 'Targets: board unavailable; keep searching from current views.'}\n\n"
        "Legal actions:\n" + "\n".join(robot_blocks)
    )
    instructions = (
        "Assign the FIRST action of every listed robot. After this call, each robot decides on its own.\n"
        "Prefer a different starting sector per robot. A sector is a landmark visible NOW plus a leaving bearing "
        "(example: open aisle ahead). Do not name unknown / unmapped space.\n"
        "Avoid sending two robots into the same visible cluster when another open-hall direction is available.\n"
        "Rear-facing cameras are not legal T0 actions. "
        "Do not choose behind, left_back, or right_back. "
        "Only front, left, and right may be chosen "
        "(use only the directions configured for that agent). "
        "Rear images are observation only.\n"
        "Trust the image labels above each photo. IIN photos come first; write image_goal_scenes[goal_id] "
        "in 1-3 English sentences about the PLACE. ON/TN goals have no image_goal_scenes entry. "
        "If no IIN photo is attached, output an empty object.\n"
        "Output ONLY the fields in # Output. Do not output progress_analysis, per-robot reasoning, or any other key.\n"
        "- panoramic_observation: what each robot's current RGB views show.\n"
        "- map_understanding: joint reading of map_2d, poses/headings, and RGB views.\n"
        "- team_reasoning: distinct visible starting sectors, then who gets which.\n"
        "- image_goal_scenes: place shown in each IIN photo.\n"
        "- assignments: per-robot action, region, and optional claim. "
        "region is that robot's starting sector (visible landmark + bearing). "
        "The first action may leave toward that sector."
    )
    constraints = "\n".join(
        (
            SHARED_CONSTRAINTS,
            "- The region you write is stored on the shared board and shown to later LA/VA as context, not as a locked room lease.",
            "- Prefer not assigning a first action toward a glass door, a closed door, "
            "or another robot (any configured teammate).",
            "- Do not invent map coordinates. Do not claim a robot is already at an IIN place "
            "unless that robot's current views match the photo.",
            "- Do not send any robot to search a goal already marked found on the board.",
            f"- claim_goal_id may be set if that robot's OWN current views show one "
            f"still-unresolved target about {near} m or nearer, or already large in that robot's view; "
            "if it is only a tiny far blob, leave claim empty.",
            "- Do not output a stop field.",
            "- Rear-facing cameras are not legal T0 actions. "
            "Do not choose behind, left_back, or right_back. Rear images are observation only.",
            "- action must be a legal action for THAT robot. "
            "A near-range mask may omit a camera when the center-region mean depth is under 1.5 m. "
            "Prefer a listed direction.",
            "- Include every listed robot key exactly once in panoramic_observation and assignments.",
        )
    )
    output = (
        "JSON only:\n"
        "{\n"
        '  "panoramic_observation": {\n'
        f'    "<robot>": "<what this robot\'s omnidirectional RGB views show>"\n'
        "  },\n"
        '  "map_understanding": "<joint understanding of map_2d, poses/headings, and RGB views>",\n'
        '  "team_reasoning": "<distinct visible starting sectors, then who gets which>",\n'
        '  "image_goal_scenes": {\n'
        '    "goal_0": "<1-3 sentences: place in that IIN photo>"\n'
        "  },\n"
        '  "assignments": {\n'
        f'    "<robot>": {{ ... one object per robot in [{keys}] ... }}\n'
        "  }\n"
        "}\n"
        "Each assignment object must be:\n"
        "{\n"
        '  "action": "navigate to <legal direction for THIS robot>",\n'
        '  "region": "<visible landmark + leaving bearing; no unseen rooms>",\n'
        '  "claim_goal_id": ""\n'
        "}"
    )
    return _std_prompt(
        "You are the T0 team language-action planner for a multi-robot search.",
        context,
        instructions,
        constraints,
        output,
    )


def _peer_xy(rec: Optional[Dict]) -> Optional[tuple]:
    if not isinstance(rec, dict):
        return None
    for key in ("pose", "reached_xy", "xy", "waypoint", "va_map"):
        val = rec.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                return float(val[0]), float(val[1])
            except (TypeError, ValueError):
                continue
    return None


def build_teammate_spread_note(
    robot: str,
    traces: Optional[Dict[str, Dict]] = None,
    histories: Optional[Dict[str, List[Dict]]] = None,
) -> str:
    """English LA note with last-known teammate map poses. Empty poses still keep the rule."""
    from .config import ROBOT_IDS

    lines = []
    for name in AGENTS:
        if name == robot:
            continue
        xy = _peer_xy((traces or {}).get(name))
        if xy is None:
            items = list((histories or {}).get(name) or [])
            for rec in reversed(items):
                xy = _peer_xy(rec)
                if xy is not None:
                    break
        agent = f"agent_{ROBOT_IDS.get(name, '?')}"
        rec = (traces or {}).get(name) or {}
        extra = ""
        action = str((rec.get("la") or {}).get("action") or rec.get("direction") or "").strip()
        wp = rec.get("va_map") or rec.get("waypoint")
        wp_xy = _peer_xy({"xy": wp}) if wp is not None else None
        if action:
            extra += ", last action %s" % action
        if wp_xy is not None:
            extra += ", last visual waypoint (%.2f, %.2f)" % (wp_xy[0], wp_xy[1])
        if xy is None:
            lines.append(f"- {agent} ({name}): pose unknown this cycle{extra}")
        else:
            lines.append(
                f"- {agent} ({name}): last known map pose ({xy[0]:.2f}, {xy[1]:.2f}){extra}"
            )
    return (
        "Teammate last-known map state (metres). "
        "Prefer not walking toward these poses or waypoints. "
        "Do not choose, box, or follow any configured teammate.\n" + "\n".join(lines)
    )


def build_t0_sector_block(robot: str, t0: Optional[Dict[str, Any]] = None) -> str:
    """Board-stored T0 starting sectors, shown as context to later LA/VA."""
    from .config import ROBOT_IDS

    t0 = t0 or {}
    assignments = t0.get("assignments") if isinstance(t0.get("assignments"), dict) else {}
    mine = assignments.get(robot) if isinstance(assignments.get(robot), dict) else {}
    my_region = " ".join(str((mine or {}).get("region") or "").split()) or "not assigned at T0"
    agent = f"agent_{ROBOT_IDS.get(robot, '?')}"
    lines = [
        "T0 search sector (stored on the board; a starting bearing plus a landmark "
        "visible at T0, not a complete-map room lease):",
        f"Yours ({agent} / {robot}): {my_region}",
        "Teammates:",
    ]
    for name in AGENTS:
        if name == robot:
            continue
        rec = assignments.get(name) if isinstance(assignments.get(name), dict) else {}
        region = " ".join(str((rec or {}).get("region") or "").split()) or "not assigned at T0"
        other = f"agent_{ROBOT_IDS.get(name, '?')}"
        lines.append(f"- {other} ({name}): {region}")
    reason = " ".join(str(t0.get("team_reasoning") or "").split())
    if reason:
        lines.append("T0 team_reasoning: " + reason)
    lines.append(
        "Use this as context when choosing a next direction. "
        "It is not an order to stay inside a named room."
    )
    return "\n".join(lines)


def origin_views_text(views: Iterable[Dict]) -> str:
    items = list(views or [])
    if not items:
        return "Initial Panorama: not captured yet."
    pose = None
    for rec in items:
        pose = rec.get("pose")
        if pose is not None:
            break
    pose_txt = ""
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        pose_txt = " start_pose=(%.2f, %.2f)" % (float(pose[0]), float(pose[1]))
    names = [str(rec.get("name") or "?") for rec in items]
    return "Initial Panorama (%d views%s): %s" % (len(items), pose_txt, ", ".join(names))


def history_text(history: Iterable[Dict]) -> str:
    items = list(history)
    if not items:
        return "Navigation History: empty (no motion RGB yet)."
    lines = ["Navigation History (body-front RGB every 5 s during walk or turn):"]
    for i, rec in enumerate(items):
        if rec.get("nav_ok") is True:
            status = "reached"
        elif rec.get("action"):
            status = "attempted"
        else:
            status = "during nav"
        pose = rec.get("pose")
        pose_txt = ""
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            pose_txt = f" pose=({float(pose[0]):.2f},{float(pose[1]):.2f})"
        kind = str(rec.get("kind") or "nav")
        lines.append(
            f"- Frame {rec.get('waypoint_id', i)} ({kind}, {status}): front{pose_txt} "
            f"action={rec.get('action', '')} "
            f"target={rec.get('target', '')} xy={rec.get('reached_xy') or rec.get('xy')}"
        )
    return "\n".join(lines)
