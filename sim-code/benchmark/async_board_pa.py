"""LA instruction blocks for the async LaViRA mode (Verifier V2, no team map).

Paper mode (``benchmark.ma_search --control-mode``):

* ``k2-async-keep-goals-pa`` (LaViRA full): Independent LA inputs plus
  teammate ``progress_analysis``. Own FOUND only, no object board, no teammate
  FOUND on the list (``KEEP_GOALS_PROTOCOL``).

Each ready agent runs its own LA on its own panorama. The object-board
variant (``BOARD_PROTOCOL``, ``build_async_board_instruction``: every target
marked unresolved or found by agent_k after Verifier V2) is kept for
reference; it is not one of the paper modes.

The ``*_FIELD_CONTRACT_MARKER`` strings are part of the instruction text and
are matched by ``vlnce_baselines/ZS_Evaluator_mp.py``; do not rename them.

JSON field contract (all async modes):

- ``progress_analysis``: first-person history summary only — what the agent has
  observed and done so far from its navigation-history image sequence.
  Current 4-dir views must not be captioned here.
- ``reasoning`` / action / STOP: use all inputs (target list or board,
  teammate progress_analysis when shared, references, history, current 4-dir). FOUND still needs
  own current views. The agent's own last progress_analysis is not in the
  prompt.

No team top-down map in the LA input, no lease, no exploring announcements.
This module must not import ``benchmark.ma_search`` (circular).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

NONE_PA = "none"
FIELD_CONTRACT_MARKER = "ASYNC_BOARD_PA_FIELD_CONTRACT"
IND_PA_FIELD_CONTRACT_MARKER = "ASYNC_IND_PA_FIELD_CONTRACT"
HISTORY_SECTION_LABEL = (
    "\n=== NAVIGATION HISTORY ===\n"
    "These frames are the only visual source for progress_analysis. "
    "If empty, say so in progress_analysis. "
    "reasoning / action / STOP may also use these frames together with "
    "every other input."
)
CURRENT_VIEWS_SECTION_LABEL = (
    "\n=== CURRENT 4-DIRECTIONAL VIEWS ===\n"
    "Do not copy these frames into progress_analysis. "
    "reasoning / action / STOP use these frames together with every "
    "other input."
)


def current_views_section_label(n_frames: int = 4) -> str:
    if int(n_frames) == 8:
        return (
            "\n=== CURRENT 8-VIEW 45° PANORAMA ===\n"
            "Do not copy these frames into progress_analysis. "
            "reasoning / action / STOP use these frames together with every "
            "other input. "
            "Frames: 0=0° FORWARD, 1=45°, 2=90° LEFT, 3=135°, "
            "4=180° BEHIND, 5=225°, 6=270° RIGHT, 7=315°. "
            "Choose action 'navigate to view k' for the frame you will face. "
            "Do not output focus_view."
        )
    return CURRENT_VIEWS_SECTION_LABEL


def goal_public_id(goal: dict, local_idx: int) -> str:
    return f"goal_{goal.get('goal_index', local_idx)}"


def goal_label(goal: dict, local_idx: int) -> str:
    return (
        f"{goal_public_id(goal, local_idx)} "
        f"[{goal.get('modality', 'ON')}]: {goal['object_category']}"
    )


def _goal_detail_line(
    goal: dict,
    local_idx: int,
    *,
    found: bool,
    goal_pil: Optional[Dict[int, object]],
    refs: List[dict],
) -> str:
    modality = goal.get("modality", "ON")
    line = f"- {goal_label(goal, local_idx)}"
    if modality == "TN" and goal.get("tn_text"):
        line += f"; description: {goal['tn_text']}"
    elif modality == "IIN":
        key = goal.get("goal_index", local_idx)
        goal_pil = goal_pil or {}
        attach_ref = key in goal_pil and not found
        if attach_ref:
            ref_no = len(refs) + 1
            line += f"; reference image #{ref_no}"
            refs.append({
                "goal_id": goal_public_id(goal, local_idx),
                "label": goal_label(goal, local_idx),
                "image": goal_pil[key],
            })
        elif key not in goal_pil:
            line += "; reference image unavailable"
        else:
            line += "; reference image already verified"
    return line


def build_object_board(
    goals: Sequence[dict],
    found_by: Sequence[int],
    *,
    goal_pil: Optional[Dict[int, object]] = None,
) -> Tuple[str, List[dict]]:
    """Text board: every goal, V2-FOUND only when ``found_by[j] >= 0``."""

    if len(found_by) != len(goals):
        raise ValueError(
            f"found_by length {len(found_by)} != goals length {len(goals)}"
        )
    goal_pil = goal_pil or {}
    lines: List[str] = []
    refs: List[dict] = []
    for j, goal in enumerate(goals):
        finder = int(found_by[j])
        found = finder >= 0
        line = _goal_detail_line(
            goal,
            j,
            found=found,
            goal_pil=goal_pil,
            refs=refs,
        )
        if found:
            line += f" — found by agent_{finder}"
        else:
            line += " — unresolved"
        lines.append(line)
    header = "Targets (team object board; V2-verified FOUND only):"
    return header + "\n" + "\n".join(lines), refs


def progress_age_label(current_step: int, last_step: int) -> str:
    if last_step < 0:
        return NONE_PA
    age = max(0, int(current_step) - int(last_step))
    return f"{age} steps ago"


def _pa_line(rec: Dict[str, Any], current_step: int) -> str:
    agent_id = int(rec.get("agent_id", 0))
    text = str(rec.get("text") or "").strip()
    last_step = int(rec.get("step", -1))
    if not text or last_step < 0:
        return f"- agent_{agent_id}: {NONE_PA}"
    age = progress_age_label(current_step, last_step)
    return f"- agent_{agent_id} ({age}): {text}"


def build_progress_broadcast(
    records: Sequence[Dict[str, Any]],
    current_step: int,
) -> str:
    """Full latest progress_analysis per agent. Empty / missing → ``none``. Never truncated."""

    lines = [
        "Team progress_analysis (latest full text; may be stale):",
    ]
    for rec in records:
        lines.append(_pa_line(rec, current_step))
    return "\n".join(lines)


def _record_for_agent(
    records: Sequence[Dict[str, Any]],
    agent_id: int,
) -> Dict[str, Any]:
    for rec in records:
        if int(rec.get("agent_id", -1)) == int(agent_id):
            return rec
    return {"agent_id": int(agent_id), "text": "", "step": -1}


def build_own_progress_note(
    records: Sequence[Dict[str, Any]],
    current_step: int,
    agent_id: int,
) -> str:
    rec = _record_for_agent(records, agent_id)
    return (
        "Your last progress_analysis (rewrite from YOUR navigation history "
        "image sequence: what you have observed and done so far; do not copy "
        "current 4-dir views or teammate claims):\n"
        + _pa_line(rec, current_step)
    )


def build_teammate_progress_claims(
    records: Sequence[Dict[str, Any]],
    current_step: int,
    agent_id: int,
) -> str:
    """Other agents' latest progress_analysis claims. Used by reasoning only."""

    lines = [
        "Teammate claimed progress_analysis (for reasoning_action only; "
        "do not copy into progress_analysis):",
    ]
    others = [
        rec for rec in records if int(rec.get("agent_id", -1)) != int(agent_id)
    ]
    if not others:
        lines.append(f"- {NONE_PA}")
        return "\n".join(lines)
    for rec in others:
        lines.append(_pa_line(rec, current_step))
    return "\n".join(lines)


def collect_progress_records(states: Sequence[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for j, state in enumerate(states):
        records.append({
            "agent_id": int(getattr(state, "agent_id", j)),
            "text": str(getattr(state, "last_progress_analysis", "") or ""),
            "step": int(getattr(state, "last_progress_step", -1)),
        })
    return records


BOARD_PROTOCOL = (
    "You are {who}. Continue searching unresolved targets.\n"
    f"{FIELD_CONTRACT_MARKER}\n"
    "Field sources:\n"
    "- progress_analysis: look ONLY at YOUR Navigation History image "
    "sequence. Write what you have observed and done so far from those "
    "frames. If history is empty, say so. Do not describe, caption, or "
    "quote the current 4-directional views. Do not mention teammate claims.\n"
    "- reasoning, action, and STOP: use ALL inputs (object board, "
    "teammate progress_analysis, target references, Navigation History, "
    "and current 4-directional views). "
    "Prefer unresolved targets a teammate is not covering unless your "
    "current views give stronger evidence.\n"
    "STOP protocol: stop=true proposes FOUND verification for exactly one "
    "unresolved goal_id via claim_goal_id; it does not end the episode and "
    "does not retire you. After a failed claim, keep searching. After a "
    "passed claim, the board updates; keep searching remaining unresolved "
    "targets. The episode ends only when every target is V2-verified FOUND "
    "or the team step budget is reached.\n"
    "Do not output a stair field.\n"
    "The object board and teammate progress_analysis are not STOP evidence. "
    "Confirm FOUND with your own current 4-directional views only."
)

_PA_PLACEHOLDERS = (
    '"progress_analysis": "<brief assessment of what you\'ve observed and done so far>"',
    '"progress_analysis": "<brief assessment of search progress>"',
    '"progress_analysis": "<assessment of current progress toward instruction completion>"',
    '"progress_analysis": "<assessment of search progress (e.g., target spotted, room type identified)>"',
    '"progress_analysis": "<searching/spotted/exploring>"',
    '"progress_analysis": "<assessment of search progress>"',
    '"progress_analysis": "<brief assessment of progress>"',
)
_PA_FIELD = (
    '"progress_analysis": "<from Navigation History frames ONLY; '
    'do not caption current 4-dir views>"'
)
_RA_PLACEHOLDERS = (
    '"reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>"',
    '"reasoning": "<explanation of the chosen action>"',
    '"reasoning": "<explanation of chosen action>"',
)
_RA_FIELD = (
    '"reasoning": "<use ALL inputs to choose action and STOP>"'
)
_TASK_RA_OLD = (
    "2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO\n"
    "   plus what you observe."
)
_TASK_RA_NEW = (
    "2. THEN, write `reasoning_action` and choose the next action using ALL inputs."
)
_SINGLE_AGENT_CODA = (
    "This is a single-agent LaViRA decision. "
    "Do not include map_understanding. Use only first-person views, "
    "navigation history, and the target list.\n"
)
_ASYNC_CODA = (
    "This is an async Independent LA decision (no team top-down map). "
    "Do not include map_understanding. "
    "progress_analysis: Navigation History frames only. "
    "reasoning / action / STOP: use ALL inputs.\n"
)
KEEP_GOALS_PROTOCOL = (
    "You are {who}. Continue searching unresolved targets.\n"
    f"{IND_PA_FIELD_CONTRACT_MARKER}\n"
    "Field sources:\n"
    "- progress_analysis: look ONLY at YOUR Navigation History image "
    "sequence. Write what you have observed and done so far from those "
    "frames. If history is empty, say so. Do not describe, caption, or "
    "quote the {views}. Do not mention teammate claims.\n"
    "- reasoning, action, and STOP: use ALL inputs (target list, "
    "teammate progress_analysis, target references, Navigation History, "
    "and {views}). "
    "There is no object board. Teammate FOUND is not listed.\n"
    "Do not output a stair field.\n"
    "Teammate progress_analysis is not STOP evidence. "
    "Confirm FOUND with your own {views} only."
)
_JSON_CODA_BOARD = (
    "\nJSON field contract (mandatory for this decision):\n"
    "- progress_analysis MUST use Navigation History frames only. "
    "Do not caption current 4-dir views. Do not mention teammates.\n"
    "- reasoning / action / STOP MUST use ALL inputs "
    "(board, teammate progress_analysis, references, "
    "Navigation History, and current 4-dir views).\n"
)
_JSON_CODA_KEEP = (
    "\nJSON field contract (mandatory for this decision):\n"
    "- progress_analysis MUST use Navigation History frames only. "
    "Do not caption current 4-dir views. Do not mention teammates.\n"
    "- reasoning / action / STOP MUST use ALL inputs "
    "(target list, teammate progress_analysis, references, "
    "Navigation History, and current 4-dir views).\n"
)


def apply_la_json_field_contract(
    prompt: str,
    independent_plus_pa: bool = False,
    pano_n: int = 4,
) -> str:
    """Rewrite LA JSON placeholders: progress_analysis from history only."""

    for old in _PA_PLACEHOLDERS:
        if old in prompt:
            prompt = prompt.replace(old, _PA_FIELD, 1)
            break
    for old in _RA_PLACEHOLDERS:
        if old in prompt:
            prompt = prompt.replace(old, _RA_FIELD, 1)
            break
    prompt = prompt.replace(_TASK_RA_OLD, _TASK_RA_NEW, 1)
    if _SINGLE_AGENT_CODA in prompt:
        prompt = prompt.replace(_SINGLE_AGENT_CODA, _ASYNC_CODA, 1)
    if "JSON field contract (mandatory for this decision):" not in prompt:
        if independent_plus_pa:
            prompt += _JSON_CODA_KEEP
        else:
            prompt += _JSON_CODA_BOARD
    if independent_plus_pa and int(pano_n) == 8:
        prompt = prompt.replace("current 4-dir views", "current 8-view 45° panorama")
        if "navigate to view" not in prompt:
            prompt += (
                "\n- action MUST be exactly one listed view "
                '("navigate to view 0" ... "navigate to view 7", or backtrack). '
                "The chosen view is the frame you will face and the frame "
                "used for visual targeting.\n"
                "- Do not output focus_view.\n"
            )
    return prompt


def has_la_field_contract(instruction: Optional[str]) -> bool:
    text = instruction or ""
    return (
        FIELD_CONTRACT_MARKER in text
        or IND_PA_FIELD_CONTRACT_MARKER in text
    )


def build_async_ind_pa_instruction(
    target_body: str,
    states: Sequence[Any],
    *,
    agent_id: int,
    current_step: int,
    views_phrase: str = "current 4-directional views",
) -> str:
    """Independent target list plus teammate progress_analysis claims."""

    records = collect_progress_records(states)
    claim_block = build_teammate_progress_claims(records, current_step, agent_id)
    return (
        KEEP_GOALS_PROTOCOL.format(
            who=f"agent_{int(agent_id)}",
            views=views_phrase,
        )
        + "\n"
        + str(target_body).rstrip()
        + "\n"
        + claim_block
    )


def build_async_board_instruction(
    goals: Sequence[dict],
    found_by: Sequence[int],
    states: Sequence[Any],
    *,
    agent_id: int,
    current_step: int,
    goal_pil: Optional[Dict[int, object]] = None,
    fhint: str = "",
) -> Tuple[str, List[dict]]:
    board, refs = build_object_board(goals, found_by, goal_pil=goal_pil)
    records = collect_progress_records(states)
    claim_block = build_teammate_progress_claims(records, current_step, agent_id)
    instruction = (
        BOARD_PROTOCOL.format(who=f"agent_{int(agent_id)}")
        + "\n"
        + board
        + "\n"
        + claim_block
    )
    if refs:
        instruction += (
            "\nReference images are target examples only; they are not "
            "observations or found evidence."
        )
    if fhint:
        instruction += fhint
    return instruction, refs
