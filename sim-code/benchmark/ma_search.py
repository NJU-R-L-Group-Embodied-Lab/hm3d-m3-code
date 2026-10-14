"""LaViRA multi-agent search evaluator for HM3D-M3 (unordered multi-object goals).

Runs the real search loop: K agents spawn at their distributed starts and
explore with decentralised LA target picking on a shared
``SharedMappingPlanner`` map + FMM, with Verifier-V2 FOUND claims. It needs
the Habitat simulator (GPU/EGL) and an OpenAI-compatible VLM endpoint
(``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``, see ``benchmark/api_auth.py``).

Subclasses :class:`vlnce_baselines.multi.ma_runner.MultiAgentRunner` and only
overrides the per-episode rollout to:
  * spawn agents at ``ep.agents[i].start_position`` (distributed, not co-located);
  * track an UNORDERED multi-object goal set, each goal found by *any* agent
    whose STOP claim passes verification;
  * terminate when ALL goals are found or the per-agent geodesic budget / step cap
    is hit, recording per-goal found-by + path-length;
  * emit :mod:`benchmark.metrics` (MSPL / fairness / cooperation-gain).

Paper mode (``--control-mode``; the alias is accepted as well):

  ============  ==========================  =====================================
  alias         internal mode               paper row
  ============  ==========================  =====================================
  full          k2-async-keep-goals-pa      LaViRA (full)
  ============  ==========================  =====================================

Example (run from ``sim-code/``; see ``scripts/run_method.sh`` for the exact paper flags)::

    python -m benchmark.ma_search \\
        --episodes data/datasets/ma_objectnav_ovon/val_unseen/<split>.json \\
        --control-mode full --use-vlm --clean-found-state --disable-path-budget \\
        --no-team-topdown-map --save-dir logs/lavira_full
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vlnce_baselines.multi import ma_runner as MR
from vlnce_baselines.multi.ma_runner import (
    HabitatSimActions,
    MultiAgentRunner,
    PerAgentState,
    consume_va_bbox_overlay,
)
from vlnce_baselines.multi.ma_mapping_planner import world_xz_reached
from . import metrics
from .frontier import assign_frontiers, assign_frontiers_mce, detect_frontiers
from .iin_verify import InstanceVerifier
from . import async_board_pa as ABP

from benchmark.api_auth import get_llm_api_key, get_llm_base_url, get_llm_model

# Endpoint for the verifier client. The names are kept because other
# modules may import them.
_DASH_URL = get_llm_base_url()
_DASH_KEY = get_llm_api_key()
_LLM_MODEL = get_llm_model()


def _as_obs_list(obs):
    """Keep ma_search's downstream code list-shaped in K=1 Habitat mode."""
    return obs if isinstance(obs, list) else [obs]


def _is_habitat_stop(action) -> bool:
    """True when the discrete Habitat action is STOP (no move_fn in 0.1.7)."""
    if action is None:
        return False
    if isinstance(action, str):
        return action.strip().lower() == "stop"
    name = getattr(action, "name", None)
    if isinstance(name, str) and name.strip().lower() == "stop":
        return True
    try:
        return int(action) == int(HabitatSimActions.STOP)
    except (TypeError, ValueError):
        return False


def _hold_obs_list(sim, n_agents: int):
    """Current observations without sending an action to habitat-sim."""
    if n_agents > 1 and hasattr(sim, "_per_agent_obs"):
        return _as_obs_list(sim._per_agent_obs())
    suite = getattr(sim, "_sensor_suite", None)
    if hasattr(sim, "get_sensor_observations") and suite is not None:
        raw = sim.get_sensor_observations()
        return _as_obs_list(suite.get_observations(raw))
    prev = getattr(sim, "_prev_sim_obs", None)
    if prev is not None:
        return _as_obs_list(prev)
    raise RuntimeError("K=1 STOP hold needs current observations")


def _sim_step_obs_list(sim, actions, n_agents: int):
    """Step Habitat with scalar action for K=1 and list actions for K>1.

    habitat-sim 0.1.7 has no `stop` move_fn. MultiAgentSimulator already
    holds on STOP when K>1; K=1 must do the same before `sim.step`.
    """
    if n_agents <= 1:
        action = actions[0] if isinstance(actions, (list, tuple, np.ndarray)) else actions
        if _is_habitat_stop(action):
            return _hold_obs_list(sim, n_agents)
        return _as_obs_list(sim.step(action))
    return _as_obs_list(sim.step(actions))


def _goal_public_id(goal: dict, local_idx: int) -> str:
    """Stable public id for a task target; exposes no spatial GT."""
    return f"goal_{goal.get('goal_index', local_idx)}"


def _goal_label(goal: dict, local_idx: int) -> str:
    return f"{_goal_public_id(goal, local_idx)} [{goal.get('modality', 'ON')}]: {goal['object_category']}"


def _goal_id_to_index(goals: List[dict]) -> Dict[str, int]:
    return {_goal_public_id(goal, j): j for j, goal in enumerate(goals)}


def _build_clean_target_instruction(goals: List[dict], visible_verified_found: List[bool],
                                    goal_pil: Optional[Dict[int, object]] = None,
                                    fhint: str = "") -> Tuple[str, List[dict]]:
    """Build object-level target input without evaluator-derived remaining lists.

    The visible found list is supplied by policy-visible memory/messages only.
    The function preserves the original `goals_set` order and never includes
    spatial fields such as position, rep_point, or reach_points.
    """
    goal_pil = goal_pil or {}
    lines: List[str] = []
    refs: List[dict] = []
    found_labels: List[str] = []

    for j, goal in enumerate(goals):
        modality = goal.get("modality", "ON")
        line = f"- {_goal_label(goal, j)}"
        if modality == "TN" and goal.get("tn_text"):
            line += f"; description: {goal['tn_text']}"
        elif modality == "IIN":
            key = goal.get("goal_index", j)
            if not visible_verified_found[j] and key in goal_pil:
                ref_no = len(refs) + 1
                line += f"; reference image #{ref_no}"
                refs.append({
                    "goal_id": _goal_public_id(goal, j),
                    "label": _goal_label(goal, j),
                    "image": goal_pil[key],
                })
            else:
                line += "; reference image unavailable" if key not in goal_pil else "; reference image already verified"
        lines.append(line)
        if visible_verified_found[j]:
            found_labels.append(_goal_label(goal, j))

    found_text = ", ".join(found_labels) if found_labels else "none"
    instruction = (
        "Targets:\n"
        + "\n".join(lines)
        + f"\nVisible verified found targets: {found_text}."
        + "\nContinue searching for targets not listed as visible verified found."
    )
    if refs:
        instruction += "\nReference images are target examples only; they are not observations or found evidence."
    if fhint:
        instruction += fhint
    return instruction, refs


def _policy_selected_claim_prompt(
    goals: List[dict],
    visible_verified_found: Sequence[bool],
    *,
    list_all: bool = False,
    pano_n: int = 4,
    pano_stride_deg: float = 90.0,
) -> str:
    """Tell the LA how to bind STOP to a public target id without spatial GT."""

    allowed = [
        _goal_public_id(goal, j)
        for j, goal in enumerate(goals)
        if list_all or not visible_verified_found[j]
    ]
    if not allowed:
        return ""
    kind = "target" if list_all else "unresolved target"
    return (
        f" STOP protocol: when one {kind} is visibly reached within 1 metre, "
        "set stop=true and return claim_goal_id as exactly one of: "
        + ", ".join(allowed)
        + ". "
        + _claim_doublecheck_sentence(pano_n, pano_stride_deg)
    )


def _claim_doublecheck_sentence(pano_n: int = 4, pano_stride_deg: float = 90.0) -> str:
    from benchmark.pano_views import doublecheck_phrase
    return doublecheck_phrase(pano_n, pano_stride_deg)


# Paper mode: the async Verifier-V2 stack with Independent LA inputs, teammate
# progress_analysis and no team top-down map (see main()).
PAPER_CONTROL_MODES = (
    "k2-async-keep-goals-pa",   # LaViRA (full): all goals + teammate progress_analysis
)
CONTROL_MODE_ALIASES = {
    "full": "k2-async-keep-goals-pa",
}


def resolve_control_mode(control_mode: str) -> str:
    """Map the paper alias (full) to its internal mode."""
    mode = CONTROL_MODE_ALIASES.get(str(control_mode), str(control_mode))
    if mode not in PAPER_CONTROL_MODES:
        raise ValueError(
            "unknown control mode %r; expected one of %s"
            % (control_mode, ", ".join(PAPER_CONTROL_MODES + tuple(CONTROL_MODE_ALIASES)))
        )
    return mode


def _control_mode_flags(control_mode: str, clean_found_state: bool,
                        no_frontier: bool = False) -> Dict[str, bool]:
    """Map a paper mode to its policy-visible communication switches.

    Every paper mode uses the same switches: the LA is Independent (no FOUND
    broadcast, no free-form announcements, no frontier assignment) and FOUND claims are scored by Verifier V2 with a
    policy-selected ``claim_goal_id``. ``clean_found_state`` and
    ``no_frontier`` do not change the result for these modes; the parameters
    are kept for call compatibility.
    """
    resolve_control_mode(control_mode)
    return {
        "share_found_state": False,
        "share_found_messages": False,
        "share_announcements": False,
        "share_assignment_hint": False,
        "use_frontier": False,
        "use_qwen_claim_scoring": True,
        "use_mce_primary_frontier": False,
    }


def _pano_spin_params(pano_n: int = 4, pano_stride_deg: float = 90.0):
    from benchmark.pano_views import spin_plan
    return spin_plan(pano_n, pano_stride_deg)


def _record_pano_tick(
    sim,
    last_obs,
    actions,
    agent_ids,
    turn_step,
    turn_deg,
    mapping_planner,
    on_sim_step,
    step_ref,
    per_agent,
):
    if step_ref is not None:
        step_ref[0] += 1
    if mapping_planner is not None:
        try:
            mapping_planner.step(sim, list(last_obs))
        except Exception:
            pass
    if on_sim_step is not None:
        try:
            on_sim_step(last_obs, actions)
        except Exception:
            pass
    for i in agent_ids:
        per_agent[i].append({
            "rgb": last_obs[i]["rgb"].copy(),
            "depth": last_obs[i]["depth"].copy(),
            "angle": (turn_step * turn_deg) % 360,
            "step": turn_step,
        })


def _finish_pano_frames(frames, keep_every: int):
    from benchmark.pano_views import align_spin_frames, keep_stride
    return keep_stride(align_spin_frames(frames), keep_every)


def _collect_panorama_control(
    sim,
    agent_i: int,
    n_agents: int,
    other_action_fn,
    initial_obs: List[dict],
    mapping_planner=None,
    on_sim_step=None,
    step_ref: Optional[List[int]] = None,
    pano_n: int = 4,
    pano_stride_deg: float = 90.0,
) -> Tuple[List[dict], List[dict]]:
    """Panorama collection compatible with both single-agent and MA Habitat step."""
    from benchmark.pano_views import rotate_agent_left, temporary_turn_deg

    n_turns, turn_deg, keep_every = _pano_spin_params(pano_n, pano_stride_deg)
    per_agent = {agent_i: []}
    last_obs = _as_obs_list(initial_obs)
    with temporary_turn_deg(sim, turn_deg) as patched:
        use_turn_action = bool(patched) or abs(float(turn_deg) - 30.0) < 1e-6
        for turn_step in range(1, n_turns + 1):
            actions = [HabitatSimActions.TURN_RIGHT] * n_agents
            actions[agent_i] = HabitatSimActions.TURN_LEFT
            for j in range(n_agents):
                if j == agent_i:
                    continue
                a = other_action_fn(j)
                actions[j] = a if a is not None else HabitatSimActions.MOVE_FORWARD
            if not use_turn_action:
                rotate_agent_left(sim, agent_i, turn_deg)
                actions[agent_i] = HabitatSimActions.STOP
            last_obs = _sim_step_obs_list(sim, actions, n_agents)
            _record_pano_tick(
                sim, last_obs, actions, [agent_i], turn_step, turn_deg,
                mapping_planner, on_sim_step, step_ref, per_agent,
            )
    return _finish_pano_frames(per_agent[agent_i], keep_every), last_obs


def _collect_panorama_sync_subset(
    sim,
    spin_ids: Sequence[int],
    n_agents: int,
    other_action_fn,
    initial_obs: List[dict],
    mapping_planner=None,
    on_sim_step=None,
    step_ref: Optional[List[int]] = None,
    pano_n: int = 4,
    pano_stride_deg: float = 90.0,
) -> Tuple[List[List[dict]], List[dict]]:
    """Spin listed agents together; others keep their action."""
    from benchmark.pano_views import rotate_agent_left, temporary_turn_deg

    ordered = [int(i) for i in spin_ids]
    if not ordered:
        return [], _as_obs_list(initial_obs)
    n_turns, turn_deg, keep_every = _pano_spin_params(pano_n, pano_stride_deg)
    spin_set = set(ordered)
    per_agent: Dict[int, List[dict]] = {i: [] for i in ordered}
    last_obs = _as_obs_list(initial_obs)
    with temporary_turn_deg(sim, turn_deg) as patched:
        use_turn_action = bool(patched) or abs(float(turn_deg) - 30.0) < 1e-6
        for turn_step in range(1, n_turns + 1):
            actions = [HabitatSimActions.STOP] * n_agents
            for i in ordered:
                actions[i] = HabitatSimActions.TURN_LEFT
            for j in range(n_agents):
                if j in spin_set:
                    continue
                a = other_action_fn(j)
                actions[j] = a if a is not None else HabitatSimActions.STOP
            if not use_turn_action:
                for i in ordered:
                    rotate_agent_left(sim, i, turn_deg)
                    actions[i] = HabitatSimActions.STOP
            last_obs = _sim_step_obs_list(sim, actions, n_agents)
            _record_pano_tick(
                sim, last_obs, actions, ordered, turn_step, turn_deg,
                mapping_planner, on_sim_step, step_ref, per_agent,
            )
    out = [_finish_pano_frames(per_agent[i], keep_every) for i in ordered]
    return out, last_obs


def _collect_panorama_sync_all(
    sim,
    n_agents: int,
    initial_obs: List[dict],
    mapping_planner=None,
    on_sim_step=None,
    step_ref: Optional[List[int]] = None,
    pano_n: int = 4,
    pano_stride_deg: float = 90.0,
) -> Tuple[List[List[dict]], List[dict]]:
    """All agents spin 360° together before any navigation."""
    from benchmark.pano_views import rotate_agent_left, temporary_turn_deg

    n_turns, turn_deg, keep_every = _pano_spin_params(pano_n, pano_stride_deg)
    ids = list(range(n_agents))
    per_agent: Dict[int, List[dict]] = {i: [] for i in ids}
    last_obs = _as_obs_list(initial_obs)
    with temporary_turn_deg(sim, turn_deg) as patched:
        use_turn_action = bool(patched) or abs(float(turn_deg) - 30.0) < 1e-6
        for turn_step in range(1, n_turns + 1):
            actions = [HabitatSimActions.TURN_LEFT] * n_agents
            if not use_turn_action:
                for i in ids:
                    rotate_agent_left(sim, i, turn_deg)
                    actions[i] = HabitatSimActions.STOP
            last_obs = _sim_step_obs_list(sim, actions, n_agents)
            _record_pano_tick(
                sim, last_obs, actions, ids, turn_step, turn_deg,
                mapping_planner, on_sim_step, step_ref, per_agent,
            )
    return [_finish_pano_frames(per_agent[i], keep_every) for i in ids], last_obs


# --------------------------------------------------------------------------- #
# Episode adapter: duck-type our extended JSON into what the runner reads.
# --------------------------------------------------------------------------- #
class _Goal:
    def __init__(self, position):
        self.position = position


def _resolve_scene(rel: str, scenes_dir: str) -> str:
    """Resolve a scene glb to an existing path, tolerant of dataset-version skew.

    HM3D-v2 ObjectNav stores ``hm3d_v0.2/val/<scene>/...`` while OVON stores
    ``hm3d/val//<scene>/...`` (doubled slash + ``hm3d`` vs ``hm3d_v0.2``); both map
    to the same HM3D-1.0 tree on disk. Mirrors goal_sources._navmesh_for.
    """
    import glob
    if os.path.isabs(rel) or os.path.exists(rel):
        return rel
    r = rel.replace("//", "/")
    for c in (os.path.join(scenes_dir, r),
              os.path.join(scenes_dir, r.replace("hm3d/val/", "hm3d_v0.2/val/")),
              os.path.join(scenes_dir, r.replace("hm3d/", "hm3d_v0.2/"))):
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(scenes_dir, "**", os.path.basename(r)), recursive=True)
    return hits[0] if hits else os.path.join(scenes_dir, r)


class EpisodeAdapter:
    def __init__(self, d: dict, scenes_dir: str):
        self._raw = d
        self.episode_id = d["episode_id"]
        self.scene_id = _resolve_scene(d["scene_id"], scenes_dir)
        self.agents = d["agents"]
        self.goals_set = d["goals_set"]
        self.num_agents = d["num_agents"]
        self.budget_geo = d.get("budget_geo")
        self.success_distance = d.get("success_distance", 1.0)
        # back-compat fields the parent runner/printing touches
        self.object_category = ", ".join(sorted({g["object_category"] for g in self.goals_set}))
        self.start_position = d["agents"][0]["start_position"]
        self.start_rotation = d["agents"][0].get("start_rotation")
        self.goals = [_Goal(g["position"]) for g in self.goals_set]


# --------------------------------------------------------------------------- #
# Multi-goal search runner
# --------------------------------------------------------------------------- #
class MASearchRunner(MultiAgentRunner):
    # --- 2b verification config (set by main() after construction) ---
    verify_modality: bool = False
                                      # (apples-to-apples gate vs zero-shot baselines)
    iin_lookup: dict = {}            # (episode_id, goal_index) -> goal image path
    iin_images_dir: str = ""
    verify_backend: str = "sift"
    use_frontier: bool = True        # frontier-coordinated exploration (A/B switch)
    multimodal_search: bool = False  # feed IIN goal images + TN text into LA during search
    control_mode: str = "legacy"
    share_found_state: bool = True
    share_found_messages: bool = False
    share_announcements: bool = True
    share_assignment_hint: bool = False
    use_qwen_claim_scoring: bool = False
    use_policy_selected_claim_goal: bool = False
    # Map-Complementary Exploration (MCE-Nav): complementary frontiers primary.
    use_mce_primary_frontier: bool = False
    mce_repulse_lambda: float = 0.6
    mce_realloc_every: int = 25
    qwen_claim_min_confidence: float = 0.80
    qwen_claim_max_depth_m: float = 0.90
    clean_found_state: bool = False
    # Async Independent LA + object board + full progress_analysis broadcast.
    use_async_board_pa: bool = False
    # Independent target list + teammate progress_analysis (keep-goals mode).
    use_async_ind_pa: bool = False
    _target_reference_images = None  # explicitly labeled IIN target references
    _verifier = None
    _tn_client = None

    def _get_verifier(self):
        if self._verifier is None:
            self._verifier = InstanceVerifier(backend=self.verify_backend)
        return self._verifier

    def _get_tn_client(self):
        if self._tn_client is None:
            from openai import OpenAI
            self._tn_client = OpenAI(api_key=_DASH_KEY, base_url=_DASH_URL, timeout=60)
        return self._tn_client

    def _resolve_iin_path(self, ep, goal) -> Optional[str]:
        """Prefer stamped goal['iin_image']; derived KM eids are not in the manifest."""
        img_dir = getattr(self, "iin_images_dir", "") or ""
        raw = goal.get("iin_image")
        if raw:
            cands = [str(raw)]
            if img_dir:
                cands.append(os.path.join(img_dir, os.path.basename(str(raw))))
                if not os.path.isabs(str(raw)):
                    cands.append(os.path.join(img_dir, str(raw)))
            for cand in cands:
                if cand and os.path.isfile(cand):
                    return os.path.abspath(cand)
        try:
            j = int(goal.get("goal_index"))
        except (TypeError, ValueError):
            return None
        try:
            src = int(goal.get("source_goal_index", j))
        except (TypeError, ValueError):
            src = j
        eid = str(ep.episode_id)
        keys = [(eid, src), (eid, j)]
        if "__k" in eid or "__m" in eid:
            master = eid.rsplit("__", 1)[0]
            keys.extend([(master, src), (master, j)])
        for key_eid, key_j in keys:
            path = self.iin_lookup.get((str(key_eid), int(key_j)))
            if path and os.path.isfile(path):
                return path
        return None

    def _verify_goal(self, last_obs, agent_i, ep, goal) -> bool:
        """Modality-correct instance verification at arrival.

        ON  -> category-level (geometric reach already passed) => accept.
        IIN -> SIFT/LightGlue match of current view vs the rendered goal image.
        TN  -> qwen yes/no on the current view vs the text description.
        Falls back to acceptance if the needed asset/text is missing.
        """
        if not self.verify_modality:
            return True
        modality = goal.get("modality", "ON")
        if modality == "ON":
            return True
        rgb = None
        try:
            rgb = np.asarray(last_obs[agent_i]["rgb"])[:, :, :3]
        except Exception:  # noqa: BLE001
            return True
        if modality == "IIN":
            path = self._resolve_iin_path(ep, goal)
            if not path or not os.path.exists(path):
                return True  # no goal image to compare against -> don't block
            try:
                return bool(self._get_verifier().verify(rgb, path).is_same)
            except Exception as e:  # noqa: BLE001
                print(f"[ma-search][IIN-warn] {e!r}")
                return True
        if modality == "TN":
            txt = goal.get("tn_text")
            if not txt:
                return True
            try:
                return self._tn_yes_no(rgb, goal["object_category"], txt)
            except Exception as e:  # noqa: BLE001
                print(f"[ma-search][TN-warn] {e!r}")
                return True
        return True

    def _tn_yes_no(self, rgb, category, tn_text) -> bool:
        prompt = (f"You are looking for a specific {category} described as: \"{tn_text}\". "
                  f"Does the {category} most prominent in this image match that description? "
                  f"Answer with only 'yes' or 'no'.")
        return self._image_yes_no(rgb, prompt)

    def _on_yes_no(self, rgb, category) -> bool:
        prompt = (f"Does this image contain the target object category \"{category}\"? "
                  f"Answer with only 'yes' or 'no'.")
        return self._image_yes_no(rgb, prompt)

    def _image_yes_no(self, rgb, prompt: str) -> bool:
        import base64
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=85)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        r = self._get_tn_client().chat.completions.create(
            model=_LLM_MODEL, max_tokens=5, temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}}]}])
        return (r.choices[0].message.content or "").strip().lower().startswith("y")

    def _verify_policy_found(self, last_obs, agent_i, ep, goal) -> bool:
        """Strict perception/verification gate for messages visible to agents.

        Unlike evaluator scoring, this never falls back to True. A failed
        detector, missing reference image, missing observation, or uncertain
        verifier simply means no policy-visible found message is emitted.
        """
        try:
            rgb = np.asarray(last_obs[agent_i]["rgb"])[:, :, :3]
        except Exception:  # noqa: BLE001
            return False

        modality = goal.get("modality", "ON")
        category = goal["object_category"]
        if modality == "IIN":
            path = self._resolve_iin_path(ep, goal)
            if not path or not os.path.exists(path):
                return False
            try:
                return bool(self._get_verifier().verify(rgb, path).is_same)
            except Exception as e:  # noqa: BLE001
                print(f"[ma-search][policy-IIN-warn] {e!r}")
                return False
        if modality == "TN" and goal.get("tn_text"):
            try:
                return self._tn_yes_no(rgb, category, goal["tn_text"])
            except Exception as e:  # noqa: BLE001
                print(f"[ma-search][policy-TN-warn] {e!r}")
                return False
        try:
            return self._on_yes_no(rgb, category)
        except Exception as e:  # noqa: BLE001
            print(f"[ma-search][policy-ON-warn] {e!r}")
            return False

    def _geo_to_reach(self, sim, pos, reach_pts) -> float:
        best = float("inf")
        for p in reach_pts:
            d = sim.geodesic_distance(pos, p)
            if d is not None and np.isfinite(d) and d < best:
                best = float(d)
        return best

    def _run_episode(self, sim, ep, seed: int = 0):
        if self.use_vlm:
            self._ensure_vlms()           # lazily build per-agent LA/VA (qwen3.5-27b)
        n = self.n_agents
        rng = np.random.default_rng(seed)
        goals = ep.goals_set
        M = len(goals)
        goal_id_to_idx = _goal_id_to_index(goals)
        reach = [[np.asarray(p, dtype=np.float32) for p in g["reach_points"]] for g in goals]
        goal_pos_world = [np.asarray(g["rep_point"], dtype=np.float32) for g in goals]
        # multimodal search: cache IIN goal reference images (PIL) per goal index
        goal_pil = {}
        if self.multimodal_search:
            from PIL import Image
            for g in goals:
                if g["modality"] == "IIN":
                    p = self._resolve_iin_path(ep, g)
                    if p and os.path.exists(p):
                        try:
                            goal_pil[g["goal_index"]] = Image.open(p).convert("RGB")
                        except Exception:  # noqa: BLE001
                            pass
        budget = ep.budget_geo or float("inf")

        # --- distributed spawn ---
        starts = [np.asarray(a["start_position"], dtype=np.float32) for a in ep.agents]
        sim.reset()
        for i in range(n):
            rot = ep.agents[i].get("start_rotation")
            if rot is None:
                a = float(rng.uniform(0, 2 * math.pi))
                rot = [0.0, math.sin(a / 2), 0.0, math.cos(a / 2)]
            sim.set_agent_state(position=[float(x) for x in starts[i]],
                                rotation=[float(x) for x in rot], agent_id=i)
        last_obs = sim._per_agent_obs()
        if self.mp is not None:
            self.mp.begin_episode(sim)
        if self._viz is not None:
            from vlnce_baselines.multi.ma_visualizer import occupancy_frame_from_sim
            origin, size_m = occupancy_frame_from_sim(sim)
            self._viz.begin_episode(origin_xz=origin, size_m=size_m)

        states = [PerAgentState(agent_id=i, last_position=starts[i].copy()) for i in range(n)]
        # Seed trails at spawn so the first LA top-down already shows start poses.
        from vlnce_baselines.multi.ma_visualizer import snapshot_world_xyz
        for i in range(n):
            states[i].position_trail.append(snapshot_world_xyz(starts[i]))
        self._frame_pose_log = [] if getattr(self, "log_traj", False) else None
        found = [False] * M
        found_by = [-1] * M
        local_found = [[False] * M for _ in range(n)]
        policy_found_self = [[False] * M for _ in range(n)]
        policy_found_by = [-1] * M
        closest = [float("inf")] * M       # diagnostic: closest any agent got to each goal
        announcements: List[Optional[dict]] = [None] * n

        def touch_closest():
            for i in range(n):
                if states[i].stopped:
                    continue
                cur = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                for j in range(M):
                    closest[j] = min(closest[j], self._geo_to_reach(sim, cur, reach[j]))
        policy_claim_records: List[dict] = []
        stop_attempt_records: Dict[Tuple[int, int, str], dict] = {}
        claim_counters = {
            "raw": 0,
            "accepted": 0,
            "rejected_confidence": 0,
            "rejected_distance": 0,
            "rejected_unbound": 0,
            "duplicates": 0,
            "identity_calls": 0,
            "identity_passed": 0,
            "identity_rejected": 0,
            "identity_failures": 0,
        }
        step_ref = [0]
        t0 = time.time()
        touch_closest()

        def record_stop_event(event: dict) -> None:
            agent_i = int(event.get("agent_id", -1))
            event_step = int(event.get("step", -1))
            goal_id = str(event.get("goal_id") or "")
            key = (agent_i, event_step, goal_id)
            record = stop_attempt_records.setdefault(key, {
                "attempt_id": f"agent_{agent_i}_step_{event_step}_{goal_id or 'unbound'}",
                "agent_id": agent_i,
                "step": event_step,
                "goal_id": goal_id,
                "goal_index": goal_id_to_idx.get(goal_id),
                "attempt_position": event.get("attempt_position"),
                "events": [],
            })
            if record.get("attempt_position") is None and event.get("attempt_position") is not None:
                record["attempt_position"] = event.get("attempt_position")
            event_payload = {
                key: value for key, value in event.items()
                if key not in {"agent_id", "step", "goal_id", "attempt_position"}
            }
            record["events"].append(event_payload)
            record["last_event"] = str(event.get("event_type") or "")
            if isinstance(event.get("verification"), dict):
                record["verification"] = dict(event["verification"])

        self._stop_event_sink = record_stop_event

        # --- 2b approach-lock state ---
        APPROACH_RADIUS = 4.0              # geodesic m: lock + home in on a near goal
        locked = [None] * n               # goal index each agent is homing on
        blacklist = [set() for _ in range(n)]   # goals an agent gave up on (unverified/unreachable)

        # --- frontier coordination state (decentralized + hint, geometric) ---
        REALLOC_EVERY = int(getattr(self, "mce_realloc_every", 25) or 25)
        frontiers_world = []              # [(wx, wz, size)] current frontier centroids
        frontier_assign = [None] * n      # per-agent assigned frontier world point [x,y,z]
        frontier_idx = [None] * n         # per-agent assigned frontier index (for viz)
        self._viz_frontiers = []          # reset per-episode (stale viz guard)
        self._viz_frontier_assign = [None] * n
        use_mce = bool(getattr(self, "use_mce_primary_frontier", False))

        def recompute_frontiers():
            if self.mp is None:
                return
            layers = self.mp.merged_layers()
            if layers is None:
                return
            fr = detect_frontiers(layers["explored"], layers["obstacle"], min_cluster=12)
            frontiers_world.clear()
            for cx, cy, sz in fr:
                wx, wz = layers["cell_to_world"](cx, cy)
                frontiers_world.append((float(wx), float(wz), int(sz)))
            if not fr:
                for i in range(n):
                    frontier_assign[i] = None
                    frontier_idx[i] = None
                return
            cells = [layers["world_to_cell"](float(sim.get_agent_state(i).position[0]),
                                             float(sim.get_agent_state(i).position[2]))
                     for i in range(n)]
            if use_mce:
                agent_xz = [
                    (float(sim.get_agent_state(i).position[0]),
                     float(sim.get_agent_state(i).position[2]))
                    for i in range(n)
                ]
                frontier_xz = [(float(wx), float(wz)) for wx, wz, _ in frontiers_world]
                asg = assign_frontiers_mce(
                    cells,
                    fr,
                    agent_world_xz=agent_xz,
                    frontier_world_xz=frontier_xz,
                    repulse_lambda=float(getattr(self, "mce_repulse_lambda", 0.6) or 0.6),
                )
            else:
                asg = assign_frontiers(cells, fr)              # distinct, geometric
            for i in range(n):
                fi = asg[i]
                frontier_idx[i] = fi
                if fi is None:
                    frontier_assign[i] = None
                else:
                    wx, wz, _ = frontiers_world[fi]
                    snapped = sim.pathfinder.snap_point(
                        np.array([wx, float(sim.get_agent_state(i).position[1]), wz], np.float32))
                    snapped = np.asarray(snapped, np.float32)
                    frontier_assign[i] = snapped if np.all(np.isfinite(snapped)) else None
            # stash for visualization
            self._viz_frontiers = [(f[0], f[1]) for f in frontiers_world]
            self._viz_frontier_assign = list(frontier_assign)

        def nearest_reach(cur, j):
            best, bp = float("inf"), reach[j][0]
            for p in reach[j]:
                d = sim.geodesic_distance(cur, p)
                if d is not None and np.isfinite(d) and d < best:
                    best, bp = d, p
            return bp

        def policy_visible_found(agent_i):
            if not getattr(self, "clean_found_state", False):
                return found if self.share_found_state else local_found[agent_i]
            return [
                policy_found_self[agent_i][j]
                or (self.share_found_messages and policy_found_by[j] >= 0)
                for j in range(M)
            ]

        def shared_found_feedback(agent_i):
            if not (getattr(self, "clean_found_state", False) and self.share_found_messages):
                return []
            return [
                f"agent_{policy_found_by[j]} found {_goal_label(goals[j], j)}"
                for j in range(M)
                if policy_found_by[j] >= 0 and policy_found_by[j] != agent_i
            ]

        def accept_confirmed_stop_claim(agent_i, goal_id, stop_decision, identity_result=None):
            """Record a FOUND after map-arrival + official panorama double-check."""

            claim_counters["raw"] += 1
            j = goal_id_to_idx.get(str(goal_id or ""))
            if j is None:
                claim_counters["rejected_unbound"] += 1
                return False
            already_shared = self.share_found_messages and policy_found_by[j] >= 0
            if policy_found_self[agent_i][j] or already_shared:
                claim_counters["duplicates"] += 1
                return False
            current = np.asarray(sim.get_agent_state(agent_i).position, dtype=np.float32)
            verification = getattr(stop_decision, "verification", None) or {}
            identity_dict = (
                identity_result.as_dict()
                if identity_result is not None
                else {
                    "passed": True,
                    "skipped": True,
                    "evidence": "lavira_arrival_double_check",
                }
            )
            record = {
                "agent_id": int(agent_i),
                "goal_id": str(goal_id),
                "goal_index": int(j),
                "step": int(step_ref[0]),
                "claim_position": [float(x) for x in current],
                "claim_source": "la_stop_approach_map_arrival_double_check",
                "double_check": dict(verification),
                "identity_check": identity_dict,
            }
            policy_claim_records.append(record)
            policy_found_self[agent_i][j] = True
            policy_found_by[j] = int(agent_i)
            claim_counters["accepted"] += 1
            print(
                f"[ma-search]   agent_{agent_i} confirmed-stop claim "
                f"{_goal_label(goals[j], j)} at step {step_ref[0]}"
            )
            if self._viz is not None:
                _render_viz_now("found")
            return True

        def claim_goals():
            if getattr(self, "use_qwen_claim_scoring", False):
                return
            for i in range(n):
                if states[i].stopped:
                    continue
                cur = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                for j in range(M):
                    if local_found[i][j]:
                        continue
                    dj = self._geo_to_reach(sim, cur, reach[j])
                    closest[j] = min(closest[j], dj)
                    if dj > self.success_distance:
                        continue
                    # within range: modality-correct verification gate (2b)
                    if not self._verify_goal(last_obs, i, ep, goals[j]):
                        blacklist[i].add(j)        # can't confirm -> stop homing on it
                        if locked[i] == j:
                            locked[i] = None
                        continue
                    local_found[i][j] = True
                    first_team_find = not found[j]
                    if first_team_find:
                        found[j] = True
                        found_by[j] = i
                    if getattr(self, "clean_found_state", False):
                        if (not policy_found_self[i][j]
                                and self._verify_policy_found(last_obs, i, ep, goals[j])):
                            policy_found_self[i][j] = True
                            if policy_found_by[j] < 0:
                                policy_found_by[j] = i
                            print(f"[ma-search]   agent_{i} verified-message {_goal_label(goals[j], j)} "
                                  f"at step {step_ref[0]}")
                    if locked[i] == j:
                        locked[i] = None
                    if first_team_find:
                        print(f"[ma-search]   agent_{i} found goal{j} "
                              f"({goals[j]['object_category']}/{goals[j]['modality']}) at step {step_ref[0]}")

        def _found_mask():
            return [bool(x) for x in found]

        def _policy_mask():
            return [int(x) >= 0 for x in policy_found_by]

        def _render_viz_now(event=None):
            if self._viz is None:
                return
            self._viz_event = event
            self._viz_policy_mask = _policy_mask()
            try:
                self._render_viz(
                    sim, ep, states, found, found_by, goal_pos_world, last_obs, step_ref[0],
                )
            except Exception as e:  # noqa: BLE001
                print(f"[ma-search][viz-warn] {event or 'manual'} step {step_ref[0]}: {e!r}")
            finally:
                self._viz_event = None

        def _viz_if_new_find(before_found, before_policy):
            if self._viz is None:
                return
            after_f, after_p = _found_mask(), _policy_mask()
            if after_f == before_found and after_p == before_policy:
                return
            _render_viz_now("found")

        def score_policy_claims_at_episode_end():
            """Use evaluator GT only here, after policy execution has ended."""

            if not getattr(self, "use_qwen_claim_scoring", False):
                return
            valid_claims = 0
            for record in policy_claim_records:
                j = int(record["goal_index"])
                agent_i = int(record["agent_id"])
                pos = np.asarray(record["claim_position"], dtype=np.float32)
                distance = self._geo_to_reach(sim, pos, reach[j])
                closest[j] = min(closest[j], distance)
                valid = bool(np.isfinite(distance) and distance <= self.success_distance)
                record["evaluator_distance_m"] = round(float(distance), 4) if np.isfinite(distance) else None
                record["evaluator_valid"] = valid
                if not valid:
                    continue
                valid_claims += 1
                local_found[agent_i][j] = True
                if not found[j]:
                    found[j] = True
                    found_by[j] = agent_i
            claim_counters["evaluator_valid"] = valid_claims
            claim_counters["evaluator_false"] = len(policy_claim_records) - valid_claims

        def score_stop_attempts_at_episode_end():
            """Attach evaluator GT diagnostics only after policy execution."""

            for record in stop_attempt_records.values():
                j = record.get("goal_index")
                pos = record.get("attempt_position")
                if j is None or pos is None:
                    record["evaluator_distance_m"] = None
                    record["evaluator_valid_pose"] = False
                    continue
                distance = self._geo_to_reach(
                    sim,
                    np.asarray(pos, dtype=np.float32),
                    reach[int(j)],
                )
                record["evaluator_distance_m"] = (
                    round(float(distance), 4) if np.isfinite(distance) else None
                )
                record["evaluator_valid_pose"] = bool(
                    np.isfinite(distance) and distance <= self.success_distance
                )

        def pano_hook(last_obs_h, _actions=None):
            """Render a frame on every panorama-rotation sim step too, so the
            video is continuous through the 12-step in-place spins."""
            if self._viz is None:
                return
            s = step_ref[0]
            if s % max(1, self.viz_every) != 0:
                return
            try:
                from vlnce_baselines.multi.ma_visualizer import snapshot_world_xyz
                from vlnce_baselines.multi.ma_decision import yaw_from_quat as _yaw_pano
                for i in range(n):
                    if states[i].stopped:
                        continue
                    states[i].position_trail.append(
                        snapshot_world_xyz(sim.get_agent_state(i).position)
                    )
                for i in range(n):
                    try:
                        pos = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                        yaw = _yaw_pano(sim.get_agent_state(i).rotation)
                        d = last_obs_h[i].get("depth") if last_obs_h is not None else None
                        if d is None:
                            continue
                        self._viz.ingest(depth_norm=d, agent_pos=pos, agent_yaw_rad=yaw)
                    except Exception as e:  # noqa: BLE001
                        print(f"[ma-search][viz-warn] pano ingest agent_{i}: {e!r}")
                self._render_viz(sim, ep, states, found, found_by, goal_pos_world, last_obs_h, s)
            except Exception as e:  # noqa: BLE001
                print(f"[ma-search][viz-warn] pano step {s}: {e!r}")

        if self._viz is not None:
            if self.mp is not None:
                try:
                    self.mp.step(sim, list(last_obs))
                except Exception as e:  # noqa: BLE001
                    print(f"[ma-search][viz-warn] t0 mapping step: {e!r}")
            _render_viz_now("t0")
        _before_found, _before_policy = _found_mask(), _policy_mask()
        claim_goals()  # credit goals already within range at spawn
        _viz_if_new_find(_before_found, _before_policy)
        opening_panos, last_obs = _collect_panorama_sync_all(
            sim,
            n_agents=n,
            initial_obs=last_obs,
            mapping_planner=self.mp,
            on_sim_step=(
                pano_hook if (
                    self._viz is not None
                    and not getattr(self, "skip_pano_viz", False)
                ) else None
            ),
            step_ref=step_ref,
            pano_n=int(getattr(self, "pano_n", 4) or 4),
            pano_stride_deg=float(getattr(self, "pano_stride_deg", 90.0) or 90.0),
        )
        for i, pano in enumerate(opening_panos):
            states[i].pending_pano = pano
            states[i].last_pano = pano
            states[i].last_pano_step = int(step_ref[0])
        print(f"[ma-search] opening 360° sync spin done at step {step_ref[0]}")

        def rollout_complete():
            if getattr(self, "use_qwen_claim_scoring", False):
                return all(owner >= 0 for owner in policy_found_by)
            return all(found)

        while step_ref[0] < self.max_steps and not rollout_complete():
            # ---------------- frontier (re)allocation (every REALLOC_EVERY) ----
            if self.use_frontier and step_ref[0] % REALLOC_EVERY == 0:
                try:
                    recompute_frontiers()
                except Exception as e:  # noqa: BLE001
                    print(f"[ma-search][frontier-warn] {e!r}")

            # ---------------- approach-lock phase (2b) ----------------
            # If an unfound, non-blacklisted goal is within APPROACH_RADIUS, the
            # agent commits to homing in on its nearest navigable viewpoint
            # (closing the last metre) instead of being pulled away by LA
            # exploration. This is what turns "saw it / got to ~2 m" into "found".
            if not getattr(self, "clean_found_state", False):
                for i in range(n):
                    if states[i].stopped:
                        continue
                    visible_found = policy_visible_found(i)
                    if locked[i] is not None and (visible_found[locked[i]] or locked[i] in blacklist[i]):
                        locked[i] = None
                    cur = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                    if locked[i] is None:
                        near = [(self._geo_to_reach(sim, cur, reach[j]), j)
                                for j in range(M) if not visible_found[j] and j not in blacklist[i]]
                        near = [(d, j) for d, j in near if d <= APPROACH_RADIUS]
                        if near:
                            locked[i] = min(near)[1]
                    if locked[i] is not None:
                        states[i].target_xy_world = nearest_reach(cur, locked[i])
                        states[i].needs_decision = False   # suppress LA; home in
                        states[i].holding_waypoint = False

            # ---------------- decide phase ----------------
            # Collect panos / prompts sequentially, then run LA+VA in parallel.
            pending_vlm = []

            def _apply_search_vlm_decision(item, decision):
                i = item["agent_id"]
                pano = item["pano"]
                cur = item["cur"]
                target = None
                from vlnce_baselines.multi.ma_decision import (
                    Backtrack,
                    EpisodeStop,
                    NewTarget,
                )
                if isinstance(decision, Backtrack):
                    pa = str(getattr(decision, "progress_analysis", "") or "")
                    states[i].last_progress_analysis = pa
                    states[i].last_progress_step = int(step_ref[0])
                    if self._apply_backtrack(sim, i, states[i], decision.waypoint_id):
                        announcements[i] = {
                            "direction": "backtrack",
                            "description": f"backtrack to waypoint {decision.waypoint_id}",
                            "step": step_ref[0],
                        }
                        states[i].needs_decision = False
                        states[i].last_tg_d = float("inf")
                        return
                if isinstance(decision, EpisodeStop):
                    states[i].target_xy_world = None
                    states[i].needs_decision = True
                    states[i].last_tg_d = float("inf")
                    states[i].last_map_d = float("inf")
                    return
                if isinstance(decision, NewTarget):
                    states[i].last_progress_analysis = str(
                        decision.progress_analysis or ""
                    )
                    states[i].last_progress_step = int(step_ref[0])
                    if use_mce and frontier_assign[i] is not None:
                        target = frontier_assign[i]
                        announcements[i] = {
                            "description": "a complementary unexplored frontier",
                            "step": step_ref[0],
                        }
                        states[i].last_direction = decision.direction
                    else:
                        wx, wz = decision.world_xz_m
                        snapped = sim.pathfinder.snap_point(
                            np.array([wx, float(cur[1]), wz], np.float32)
                        )
                        snapped = np.asarray(snapped, dtype=np.float32)
                        target = snapped if np.all(np.isfinite(snapped)) else None
                    desc = (decision.description or "").strip()
                    bbox_target = ""
                    if isinstance(decision.bbox, dict):
                        bbox_target = str(decision.bbox.get("target") or "").strip()
                    region_text = desc or bbox_target or "an observed target candidate"
                    if (not use_mce) and target is not None:
                        announcements[i] = {
                            "direction": decision.direction,
                            "description": region_text,
                            "step": step_ref[0],
                        }
                    states[i].last_direction = decision.direction
                    states[i].last_va_bbox = decision.bbox
                    if decision.bbox is not None:
                        idx = max(0, min(int(decision.bbox_frame_idx or 0), len(pano) - 1))
                        states[i].va_bbox_flash = decision.bbox
                        states[i].va_bbox_flash_rgb = pano[idx].get("rgb")
                        if self._viz is not None:
                            try:
                                self._render_viz(
                                    sim, ep, states, found, found_by,
                                    goal_pos_world, last_obs, step_ref[0],
                                )
                            except Exception as viz_exc:  # noqa: BLE001
                                print(f"[ma-search][viz-warn] va-bbox frame: {viz_exc!r}")
                    states[i].target_cell_rc = (
                        None if use_mce else decision.mapping_cell_rc
                    )
                    states[i].going_to_stop = bool(
                        (not use_mce) and decision.stop_approach
                    )
                    states[i].pending_claim_goal_id = (
                        decision.claim_goal_id
                        if states[i].going_to_stop else None
                    )
                if (
                    use_mce
                    and target is None
                    and frontier_assign[i] is not None
                ):
                    target = frontier_assign[i]
                    announcements[i] = {
                        "description": "a complementary unexplored frontier",
                        "step": step_ref[0],
                    }
                if target is None:
                    if frontier_assign[i] is not None:
                        target = frontier_assign[i]
                        announcements[i] = {
                            "description": "an assigned unexplored region",
                            "step": step_ref[0],
                        }
                    else:
                        avoid = [
                            states[j].target_xy_world
                            for j in range(n)
                            if j != i and states[j].target_xy_world is not None
                        ]
                        target = self._pick_random_navigable_target(
                            sim, rng, cur, min_dist_m=3.0, avoid=avoid, agent_id=i
                        )
                        announcements[i] = {
                            "description": "an unexplored area",
                            "step": step_ref[0],
                        }
                if not isinstance(decision, NewTarget):
                    states[i].going_to_stop = False
                    states[i].pending_claim_goal_id = None
                    states[i].target_cell_rc = None
                states[i].target_xy_world = target
                states[i].needs_decision = False
                states[i].holding_waypoint = False
                states[i].last_tg_d = float("inf")
                states[i].last_map_d = float("inf")
                states[i].nav_steps_since_decision = 0
                states[i].geom_replan_only = False

            spin_ids = []
            for i in range(n):
                if states[i].stopped or not states[i].needs_decision:
                    continue
                if getattr(states[i], "pending_pano", None) is not None:
                    continue
                spin_ids.append(i)
            if spin_ids:
                spin_panos, last_obs = _collect_panorama_sync_subset(
                    sim,
                    spin_ids,
                    n_agents=n,
                    other_action_fn=lambda j: self._next_action_for(sim, states[j]),
                    initial_obs=last_obs,
                    mapping_planner=self.mp,
                    on_sim_step=(
                        pano_hook if (
                            self._viz is not None
                            and not getattr(self, "skip_pano_viz", False)
                        ) else None
                    ),
                    step_ref=step_ref,
                    pano_n=int(getattr(self, "pano_n", 4) or 4),
                    pano_stride_deg=float(getattr(self, "pano_stride_deg", 90.0) or 90.0),
                )
                for i, pano in zip(spin_ids, spin_panos):
                    states[i].pending_pano = pano
                    states[i].last_pano = pano
                    states[i].last_pano_step = int(step_ref[0])
                if len(spin_ids) > 1:
                    print(
                        f"[ma-vlm] joint pano n={len(spin_ids)} at step {step_ref[0]}"
                    )

            for i in range(n):
                if states[i].stopped or not states[i].needs_decision:
                    continue
                cur = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                # geom_replan_only used to pick a random far point and skip
                # LA/VA. That is why Val 1000 V2 almost never STOP-claimed.
                # Fall through to the panorama + VLM decision instead.
                if getattr(states[i], "pending_pano", None) is not None:
                    pano = states[i].pending_pano
                    states[i].pending_pano = None
                else:
                    pano, last_obs = _collect_panorama_control(
                        sim, agent_i=i, n_agents=n,
                        other_action_fn=lambda j: self._next_action_for(sim, states[j]),
                        initial_obs=last_obs, mapping_planner=self.mp,
                        on_sim_step=(pano_hook if (self._viz is not None
                                     and not getattr(self, "skip_pano_viz", False)) else None),
                        step_ref=step_ref,
                        pano_n=int(getattr(self, "pano_n", 4) or 4),
                        pano_stride_deg=float(getattr(self, "pano_stride_deg", 90.0) or 90.0),
                    )
                    states[i].last_pano = pano
                    states[i].last_pano_step = int(step_ref[0])
                target = None
                decision = None
                visible_found = policy_visible_found(i)
                # Frontier assignment hint (non-paper option, off in the paper mode). It uses only the
                # explored/obstacle map, not evaluator goal GT.
                fhint = ""
                if self.share_assignment_hint and frontier_assign[i] is not None:
                    fd = sim.geodesic_distance(cur, frontier_assign[i])
                    if fd is not None and np.isfinite(fd):
                        fhint = (f" You are assigned to explore a distinct unexplored region described "
                                 f"by the shared map; it is about {fd:.0f} m away. Your teammate covers "
                                 f"a different region or object hypothesis, so prefer your assigned "
                                 f"region unless you directly see a target.")
                active_stop_goal_id = None
                active_stop_max_depth_m = None
                if (getattr(self, "use_qwen_claim_scoring", False)
                        and getattr(self, "use_policy_selected_claim_goal", False)):
                    # Free claim among public ids.
                    fhint += _policy_selected_claim_prompt(
                        goals,
                        visible_found,
                        pano_n=int(getattr(self, "pano_n", 4) or 4),
                        pano_stride_deg=float(
                            getattr(self, "pano_stride_deg", 90.0) or 90.0
                        ),
                    )
                if self.use_vlm:
                    self._target_reference_images = None
                    if getattr(self, "use_async_board_pa", False):
                        goal_images = goal_pil if self.multimodal_search else {}
                        instruction, refs = ABP.build_async_board_instruction(
                            goals,
                            policy_found_by,
                            states,
                            agent_id=i,
                            current_step=int(step_ref[0]),
                            goal_pil=goal_images,
                            fhint=fhint,
                        )
                        self._target_reference_images = refs or None
                    elif getattr(self, "use_async_ind_pa", False):
                        # Independent target list + teammate progress_analysis.
                        goal_images = goal_pil if self.multimodal_search else {}
                        independent_found = [policy_found_self[i][j] for j in range(M)]
                        body, refs = _build_clean_target_instruction(
                            goals, independent_found, goal_images, fhint)
                        from benchmark.pano_views import views_phrase as _views_phrase
                        instruction = ABP.build_async_ind_pa_instruction(
                            body,
                            states,
                            agent_id=i,
                            current_step=int(step_ref[0]),
                            views_phrase=_views_phrase(
                                int(getattr(self, "pano_n", 4) or 4),
                                float(getattr(self, "pano_stride_deg", 90.0) or 90.0),
                            ),
                        )
                        self._target_reference_images = refs or None
                    elif getattr(self, "clean_found_state", False):
                        goal_images = goal_pil if self.multimodal_search else {}
                        instruction, refs = _build_clean_target_instruction(
                            goals, visible_found, goal_images, fhint)
                        self._target_reference_images = refs or None
                    else:
                        remaining = sorted({goals[j]["object_category"] for j in range(M) if not visible_found[j]})
                        instruction = "Find these objects: " + ", ".join(remaining) + "." + fhint
                    # Exclusive LEASE prompt block (FOUND share + ownership, no free announcement).
                    # multimodal search: inject TN text + IIN reference images for
                    # the visible-unverified goals so the LA searches with instance cues.
                    if self.multimodal_search and not getattr(self, "clean_found_state", False):
                        tn_bits, iin_cats, refs = [], [], []
                        for j in range(M):
                            if visible_found[j]:
                                continue
                            g = goals[j]
                            if g["modality"] == "TN" and g.get("tn_text"):
                                tn_bits.append(f"{g['object_category']} (looks like: {g['tn_text']})")
                            elif g["modality"] == "IIN" and j in goal_pil:
                                iin_cats.append(g["object_category"])
                                refs.append({
                                    "goal_id": _goal_public_id(g, j),
                                    "label": _goal_label(g, j),
                                    "image": goal_pil[j],
                                })
                        if tn_bits:
                            instruction += " Text-described targets — " + "; ".join(tn_bits) + "."
                        if iin_cats:
                            instruction += (" Reference photos of these specific target instances are "
                                            f"provided (in order: {', '.join(iin_cats)}); match by appearance.")
                        self._target_reference_images = refs or None
                    if getattr(self, "use_async_board_pa", False) or getattr(
                        self, "use_async_ind_pa", False
                    ):
                        # Board+PA lists V2 FOUND; Independent+PA omits teammate
                        # FOUND / exploring announcements.
                        ann_parts = []
                    else:
                        ann_parts = shared_found_feedback(i)
                    ann_text = " | ".join(ann_parts)
                    need_dcs = bool(
                        getattr(states[i], "going_to_stop", False)
                        and getattr(self, "use_qwen_claim_scoring", False)
                        and self.use_vlm
                    )
                    claimed_goal_id = str(
                        getattr(states[i], "pending_claim_goal_id", None)
                        or active_stop_goal_id
                        or ""
                    )
                    if need_dcs:
                        states[i].going_to_stop = False
                        states[i].pending_claim_goal_id = None
                    if getattr(states[i], "stop_feedback", ""):
                        instruction = (
                            instruction + "\n" + str(states[i].stop_feedback)
                        ).strip()
                        states[i].stop_feedback = ""
                    refs = getattr(self, "_target_reference_images", None)
                    job, snap = self._prepare_la_va_job(
                        sim, i, states[i], pano, instruction=instruction,
                        announcement_text=ann_text, step=step_ref[0],
                        episode_id=str(ep.episode_id),
                        active_goal_id=active_stop_goal_id,
                        stop_max_depth_m=active_stop_max_depth_m,
                        target_reference_images=list(refs) if refs else None,
                    )
                    pending_vlm.append({
                        "agent_id": i,
                        "pano": pano,
                        "cur": np.asarray(cur, dtype=np.float32).copy(),
                        "visible_found": list(visible_found),
                        "job": {
                            **job,
                            "need_dcs": need_dcs,
                            "claim_goal_id": claimed_goal_id or None,
                            "stop_event_sink": record_stop_event,
                        },
                        "snap": snap,
                        "need_dcs": need_dcs,
                        "claimed_goal_id": claimed_goal_id,
                    })
                    continue
                # MCE fallback if VLM returned no NewTarget but frontiers exist.
                if (
                    use_mce
                    and target is None
                    and frontier_assign[i] is not None
                ):
                    target = frontier_assign[i]
                    announcements[i] = {
                        "description": "a complementary unexplored frontier",
                        "step": step_ref[0],
                    }
                if target is None:
                    # hard fallback: head to the assigned frontier (coverage),
                    # not a random point. Random only if no frontier available.
                    if frontier_assign[i] is not None:
                        target = frontier_assign[i]
                        announcements[i] = {
                            "description": "an assigned unexplored region",
                            "step": step_ref[0],
                        }
                    else:
                        avoid = [
                            states[j].target_xy_world
                            for j in range(n)
                            if j != i and states[j].target_xy_world is not None
                        ]
                        target = self._pick_random_navigable_target(
                            sim, rng, cur, min_dist_m=3.0, avoid=avoid, agent_id=i
                        )
                        announcements[i] = {
                            "description": "an unexplored area",
                            "step": step_ref[0],
                        }
                from vlnce_baselines.multi.ma_decision import NewTarget
                if not isinstance(decision, NewTarget):
                    states[i].going_to_stop = False
                    states[i].pending_claim_goal_id = None
                    states[i].target_cell_rc = None
                states[i].target_xy_world = target
                states[i].needs_decision = False
                states[i].holding_waypoint = False
                states[i].last_tg_d = float("inf")
                states[i].last_map_d = float("inf")
                states[i].nav_steps_since_decision = 0
                states[i].geom_replan_only = False
                if step_ref[0] >= self.max_steps:
                    break

            if pending_vlm:
                from types import SimpleNamespace
                api_results = self._run_api_wave([item["job"] for item in pending_vlm])
                for item, api in zip(pending_vlm, api_results):
                    if item.get("need_dcs") and api.dcs_should_stop:
                        accept_confirmed_stop_claim(
                            item["agent_id"],
                            item.get("claimed_goal_id"),
                            SimpleNamespace(verification=api.dcs_verification),
                            None,
                        )
                        record_stop_event({
                            "event_type": "found_emitted",
                            "agent_id": item["agent_id"],
                            "step": step_ref[0],
                            "goal_id": item.get("claimed_goal_id"),
                        })
                    try:
                        decision = self._finish_la_va(
                            sim, item["agent_id"], api.raw, item["snap"], item["pano"],
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"[ma-search][LA-warn] agent_{item['agent_id']}: {e!r}")
                        decision = None
                    _apply_search_vlm_decision(item, decision)
                    if step_ref[0] >= self.max_steps:
                        break

            # ---------------- action phase ----------------
            actions = [HabitatSimActions.STOP] * n
            for i in range(n):
                if states[i].stopped:
                    continue
                actions[i] = self._next_action_for(sim, states[i])
            # Keep STOP as a hold. Remapping it to FORWARD was walking past
            # LaViRA's 0.75 m map-Euclidean waypoint.
            step_actions = list(actions)
            for i in range(n):
                states[i].last_action = step_actions[i]
            last_obs = _sim_step_obs_list(sim, step_actions, n)
            step_ref[0] += 1
            if self.mp is not None:
                try:
                    self.mp.step(sim, list(last_obs))
                except Exception as e:  # noqa: BLE001
                    print(f"[ma-search][mp-warn] step {step_ref[0]}: {e!r}")

            # Record this frame's pose before viz so dashed trails include
            # every env step (previously the trail lagged the rendered frame).
            from vlnce_baselines.multi.ma_visualizer import snapshot_world_xyz
            from vlnce_baselines.multi.ma_decision import yaw_from_quat as _yaw_ingest
            for i in range(n):
                if states[i].stopped:
                    continue
                states[i].position_trail.append(
                    snapshot_world_xyz(sim.get_agent_state(i).position)
                )
            if self._viz is not None:
                for i in range(n):
                    try:
                        pos = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                        yaw = _yaw_ingest(sim.get_agent_state(i).rotation)
                        d = last_obs[i].get("depth") if last_obs is not None else None
                        if d is None:
                            continue
                        self._viz.ingest(depth_norm=d, agent_pos=pos, agent_yaw_rad=yaw)
                    except Exception as e:  # noqa: BLE001
                        print(f"[ma-search][viz-warn] ingest agent_{i} step {step_ref[0]}: {e!r}")

            # ---- visualization (multi-goal: all goals drawn on the shared map) ----
            if self._viz is not None and step_ref[0] % max(1, self.viz_every) == 0:
                self._viz_policy_mask = _policy_mask()
                try:
                    self._render_viz(sim, ep, states, found, found_by, goal_pos_world, last_obs, step_ref[0])
                except Exception as e:  # noqa: BLE001
                    print(f"[ma-search][viz-warn] step {step_ref[0]}: {e!r}")

            # ---------------- post-step bookkeeping ----------------
            for i in range(n):
                if states[i].stopped:
                    continue
                s = states[i]
                cur = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
                s.path_length += float(np.linalg.norm(cur - s.last_position))
                s.last_position = cur
                for j in range(M):
                    closest[j] = min(closest[j], self._geo_to_reach(sim, cur, reach[j]))
                s.nav_steps_since_decision = int(getattr(s, "nav_steps_since_decision", 0)) + 1
                if (
                    s.nav_steps_since_decision >= 12
                    and len(s.position_trail) >= 12
                    and not s.needs_decision
                    and not s.holding_waypoint
                ):
                    from vlnce_baselines.multi.ma_visualizer import trail_point_xz
                    uniq = set()
                    for p in s.position_trail[-12:]:
                        xz = trail_point_xz(p)
                        if xz is not None:
                            uniq.add((round(xz[0], 2), round(xz[1], 2)))
                    if len(uniq) < 2:
                        s.needs_decision = True
                        s.geom_replan_only = False
                        s.target_xy_world = None
                        s.nav_steps_since_decision = 0
                        print(f"[ma-search]   agent_{i} spin-lock → VLM replan at step {step_ref[0]}")
                if s.path_length >= budget:
                    s.stopped = True
                    continue
                map_d = None
                if self.mp is not None and s.target_cell_rc is not None:
                    map_d = self.mp.map_euclidean_m(i, s.target_cell_rc)
                if map_d is not None:
                    if self.mp.waypoint_reached_on_map(i, s.target_cell_rc):
                        s.needs_decision = True
                        s.geom_replan_only = False
                        s.holding_waypoint = True
                    elif map_d < s.last_map_d - 0.05:
                        s.last_map_d = map_d
                        s.consec_stuck = 0
                    else:
                        s.consec_stuck += 1
                        if s.consec_stuck >= 8:
                            s.needs_decision = True
                            s.geom_replan_only = False
                            s.consec_stuck = 0
                            if locked[i] is not None:   # gave up homing -> blacklist, explore
                                blacklist[i].add(locked[i])
                                locked[i] = None
                elif (
                    s.target_xy_world is not None
                    and world_xz_reached(cur, s.target_xy_world)
                ):
                    s.needs_decision = True
                    s.geom_replan_only = False
                    s.holding_waypoint = True
            _before_found, _before_policy = _found_mask(), _policy_mask()
            claim_goals()
            _viz_if_new_find(_before_found, _before_policy)
            if all(s.stopped for s in states):
                break

        score_policy_claims_at_episode_end()
        score_stop_attempts_at_episode_end()
        print(f"[ma-search] closest-approach per goal (m): "
              f"{[round(c, 2) if np.isfinite(c) else None for c in closest]} "
              f"| found={found}")
        if getattr(self, "use_qwen_claim_scoring", False):
            self._last_policy_claim_summary = {
                f"policy_claim_{key}": int(value)
                for key, value in claim_counters.items()
            }
            self._last_policy_claim_records = [dict(record) for record in policy_claim_records]
            stop_event_counts = Counter(
                event.get("event_type")
                for record in stop_attempt_records.values()
                for event in record.get("events", [])
            )
            self._last_stop_attempt_summary = {
                "stop_attempt_total": len(stop_attempt_records),
                "stop_attempt_evaluator_valid_pose": sum(
                    bool(record.get("evaluator_valid_pose"))
                    for record in stop_attempt_records.values()
                ),
                **{
                    f"stop_attempt_{name}": int(count)
                    for name, count in sorted(stop_event_counts.items())
                    if name
                },
            }
            self._last_stop_attempt_records = [
                dict(record) for record in stop_attempt_records.values()
            ]
        else:
            self._last_policy_claim_summary = None
            self._last_policy_claim_records = None
            self._last_stop_attempt_summary = None
            self._last_stop_attempt_records = None
        self._stop_event_sink = None
        if getattr(self, "log_traj", False) and self.save_dir:
            from . import traj_log
            goals = ep.goals_set
            gxz = traj_log.goal_xz(goals)
            traj = [[[float(starts[i][0]), float(starts[i][2])]]
                    + [[float(p[0]), float(p[2])] for p in states[i].position_trail] for i in range(n)]
            traj_log.dump_traj(self.save_dir, ep, n, starts, goals, found_by, [], step_ref[0], traj, gxz)
            if self._frame_pose_log is not None:
                with open(self.save_dir / f"framepose_{ep.episode_id}.json", "w") as fh:
                    json.dump(self._frame_pose_log, fh)
        return self._build_metrics_result(ep, states, found, found_by, step_ref[0], time.time() - t0)

    def _render_viz(self, sim, ep, states, found, found_by, goal_pos_world, last_obs, step):
        from vlnce_baselines.multi.ma_decision import yaw_from_quat as _yaw
        n = self.n_agents
        positions = [np.asarray(sim.get_agent_state(i).position, dtype=np.float32) for i in range(n)]
        yaws = [_yaw(sim.get_agent_state(i).rotation) for i in range(n)]
        # per-frame pose log (aligned 1:1 with the saved combined_step JPGs), so a
        # downstream viewer can drop in-place panorama-spin frames per agent.
        flog = getattr(self, "_frame_pose_log", None)
        if flog is not None:
            flog.append({"step": int(step),
                         "pos": [[float(p[0]), float(p[1]), float(p[2])] for p in positions],
                         "yaw": [float(y) for y in yaws]})
        if step <= 2 or step % 40 == 0:
            from vlnce_baselines.multi.ma_visualizer import trail_point_xz
            uniq = []
            for i in range(n):
                xs = set()
                for p in states[i].position_trail:
                    xz = trail_point_xz(p)
                    if xz is not None:
                        xs.add((round(xz[0], 2), round(xz[1], 2)))
                uniq.append(len(xs))
            print(
                "[ma-search][viz-trail] step=%d n=%s unique_xz=%s"
                % (step, [len(states[i].position_trail) for i in range(n)], uniq)
            )
        forward_rgbs = [last_obs[i].get("rgb") for i in range(n)]
        forward_rgbs, va_boxes = consume_va_bbox_overlay(states, forward_rgbs)
        found_markers = []
        found_labels_per_agent = [[] for _ in range(n)]
        n_f = 0
        for j, ok in enumerate(found):
            if not ok:
                continue
            n_f += 1
            g = goal_pos_world[j]
            label = str(ep.goals_set[j].get("object_category") or f"goal{j}")
            found_markers.append({
                "number": n_f,
                "world_xz": (float(g[0]), float(g[2] if len(g) >= 3 else g[1])),
                "label": label,
                "agent_id": int(found_by[j]) if j < len(found_by) else -1,
            })
            ai = int(found_by[j]) if j < len(found_by) else -1
            if 0 <= ai < n:
                found_labels_per_agent[ai].append(f"F{n_f} {label}")
        lavira_map = None
        if self.mp is not None:
            try:
                lavira_map = self.mp.render_merged_habitat_map(
                    sim,
                    agent_paths_world=[states[i].position_trail for i in range(n)],
                    agent_targets_world=[states[i].target_xy_world for i in range(n)],
                    goals_world=None,
                    found_markers=found_markers or None,
                    found_goal_mask=found,
                    frontiers_world=getattr(self, "_viz_frontiers", None),
                    agent_frontiers_world=getattr(self, "_viz_frontier_assign", None))
            except Exception:
                lavira_map = None
        # goal status string in the header (ASCII only — cv2.putText/Hershey
        # fonts can't render unicode ticks, they show as garbage boxes).
        gstat = "  ".join(f"{ep.goals_set[j]['object_category']}[{ep.goals_set[j]['modality']}]"
                          f"{'+' if found[j] else '-'}" for j in range(len(found)))
        self._viz.render_step(
            step=step, agent_positions=positions, agent_yaws=yaws,
            forward_rgbs=forward_rgbs, goal_name=gstat, episode_id=str(ep.episode_id),
            instruction=f"Find: {ep.object_category}  | found {sum(found)}/{len(found)}",
            agent_paths=[states[i].position_trail for i in range(n)],
            goals_world=None,
            found_goal_mask=found,
            found_markers=found_markers or None,
            found_labels_per_agent=found_labels_per_agent,
            per_agent_target_world=[states[i].target_xy_world for i in range(n)],
            action_per_agent=[states[i].last_action for i in range(n)],
            la_direction_per_agent=[states[i].last_direction or None for i in range(n)],
            va_bbox_per_agent=va_boxes,
            lavira_map_bgr=lavira_map)
        if self.save_dir:
            meta_path = Path(self.save_dir) / str(ep.episode_id) / "viz_meta.jsonl"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "step": int(step),
                "event": getattr(self, "_viz_event", None),
                "found": [bool(x) for x in found],
                "policy_found": list(getattr(self, "_viz_policy_mask", []) or []),
                "la": [states[i].last_direction or None for i in range(n)],
                "va": [],
            }
            for box in va_boxes or []:
                if not isinstance(box, dict):
                    rec["va"].append(None)
                    continue
                rec["va"].append({
                    "target": box.get("target"),
                    "action": box.get("action"),
                })
            with open(meta_path, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _build_metrics_result(self, ep, states, found, found_by, steps, elapsed):
        n = self.n_agents
        M = len(ep.goals_set)
        result = {"per_agent": [
            {"path_length": round(states[i].path_length, 3),
             "goals_found": [j for j in range(M) if found_by[j] == i]}
            for i in range(n)]}
        m = metrics.team_metrics(ep._raw, result)
        m["steps"] = steps
        m["elapsed_s"] = round(elapsed, 1)
        m["control_mode"] = getattr(self, "control_mode", "legacy")
        policy_claim_summary = getattr(self, "_last_policy_claim_summary", None)
        if policy_claim_summary:
            m.update(policy_claim_summary)
        stop_attempt_summary = getattr(self, "_last_stop_attempt_summary", None)
        if stop_attempt_summary:
            m.update(stop_attempt_summary)
        # parent evaluate() prints these keys:
        per_agent = [{"agent_id": i, "success": int(len(result["per_agent"][i]["goals_found"]) > 0),
                      "spl": 0.0, "path_length": result["per_agent"][i]["path_length"],
                      "goals_found": result["per_agent"][i]["goals_found"]} for i in range(n)]
        payload = {"episode_id": str(ep.episode_id), "scene_id": ep.scene_id,
                "object_category": ep.object_category, "steps": steps, "elapsed_s": elapsed,
                "control_mode": getattr(self, "control_mode", "legacy"),
                "per_agent": per_agent, "joint_success": int(all(found)),
                "team_spl": m["mspl"], "progress": m["progress"], "metrics": m}
        payload["policy_protocol"] = {
            "clean_found_state": bool(self.clean_found_state),
            "evaluator_gt_policy_access": False,
            "share_found_messages": bool(self.share_found_messages),
            "share_announcements": bool(self.share_announcements),
            "qwen_claim_min_confidence": float(self.qwen_claim_min_confidence),
            "qwen_claim_max_depth_m": float(self.qwen_claim_max_depth_m),
            "use_async_board_pa": bool(getattr(self, "use_async_board_pa", False)),
            "use_async_ind_pa": bool(getattr(self, "use_async_ind_pa", False)),
            "use_continuous_history": True,
            "history_interval": 2,
            "use_backtrack": True,
            "pano_n": int(getattr(self, "pano_n", 4) or 4),
            "pano_stride_deg": float(getattr(self, "pano_stride_deg", 90.0) or 90.0),
            "rgb_wh": [int(self._width), int(self._height)],
            "hfov_deg": float(self._hfov),
            "cam_height_m": float(self.cfg.SIMULATOR.DEPTH_SENSOR.POSITION[1]),
            "agent_height_m": float(
                getattr(self.cfg.SIMULATOR.AGENT_0, "HEIGHT", 0.88)
            ),
        }
        claim_records = getattr(self, "_last_policy_claim_records", None)
        if claim_records is not None:
            payload["policy_claims"] = claim_records
        stop_attempt_records = getattr(self, "_last_stop_attempt_records", None)
        if stop_attempt_records is not None:
            payload["stop_attempts"] = stop_attempt_records
        return payload

    def _dump_summary(self, results):
        mets = [r["metrics"] for r in results if "metrics" in r]
        agg = metrics.aggregate(mets)
        print("\n[ma-search] === AGGREGATE ===")
        print(json.dumps(agg, indent=2))
        if self.save_dir:
            with open(self.save_dir / "aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)
            with open(self.save_dir / "metrics_list.json", "w") as f:
                json.dump(mets, f)   # per-episode metrics, for shard merging


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_cfg(config_path: str):
    from vlnce_baselines.config.default import get_config as lavira_get_config
    full = lavira_get_config(config_path)
    task = full.TASK_CONFIG
    full.defrost()
    full.SIMULATOR = task.SIMULATOR
    full.DATASET = task.DATASET
    full.freeze()
    return full


def _set_num_agents(cfg, n: int):
    MR.set_num_agents(cfg, n)


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="LaViRA multi-agent multi-goal search evaluator (HM3D-M3)")
    p.add_argument("--episodes", required=True)
    p.add_argument("--config", default="vlnce_baselines/config/multi_objectnav_v2.yaml")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--save-dir", default="logs/ma_search_2a")
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--use-vlm", action="store_true")
    p.add_argument("--la-thinking-budget", type=int, default=1024,
                   help="LA reasoning/thinking token budget (higher = better but slower)")
    p.add_argument("--no-la-thinking", action="store_true",
                   help="Disable LA enable_thinking (budget=0 still leaves thinking on)")
    p.add_argument("--verify", action="store_true",
                   help="2b: modality verification gate (ON=reach, IIN=iin_verify, TN=qwen)")
    p.add_argument("--iin-images-dir", default="data/datasets/ma_objectnav/val/iin_images",
                   help="dir with render_iin manifest.json (for IIN verification)")
    p.add_argument("--verify-backend", default="sift", choices=["sift", "lightglue"])
    p.add_argument("--visualize", action="store_true", help="write per-step multi-goal panel PNGs")
    p.add_argument("--viz-every", type=int, default=2)
    p.add_argument("--map-size-cm", type=int, default=0,
                   help="override MAP.MAP_SIZE_CM (0 = keep config; 8000 = 80 m house canvas)")
    p.add_argument("--control-mode", required=True,
                   choices=list(PAPER_CONTROL_MODES) + list(CONTROL_MODE_ALIASES),
                   help="LaViRA paper mode: full (= k2-async-keep-goals-pa)")
    p.add_argument("--clean-found-state", action="store_true",
                   help=("do not expose evaluator GT found/unfound to policy prompts; "
                         "use object-level targets and policy-verified found messages"))
    p.add_argument("--disable-path-budget", action="store_true",
                   help="ignore episode budget_geo and rely only on max env steps")
    p.add_argument("--qwen-claim-min-confidence", type=float, default=None,
                   help="minimum qwen confidence for a policy-visible FOUND claim")
    # Multimodal goal spec (IIN images + TN text) is the benchmark's INPUT CONTRACT,
    # so it is always ON.
    p.add_argument("--multimodal-search", action="store_true",
                   help="(always ON; kept for back-compat) feed IIN images + TN text to the LA")
    p.add_argument("--log-traj", action="store_true", help="dump per-step trajectory JSON for visualization")
    p.add_argument("--skip-pano-viz", action="store_true", help="do NOT render frames during panorama in-place spins (nav steps only)")
    p.add_argument("--pano-n", type=int, default=4,
                   help="Number of panorama frames sent to the VLM (4=cardinals, 8=45°)")
    p.add_argument("--pano-stride-deg", type=float, default=90.0,
                   help="Angular spacing of VLM panorama frames")
    p.add_argument("--no-team-topdown-map", action="store_true",
                   help="(always on; kept for back-compat) the LA gets no shared top-down map")
    p.add_argument("--episode-ids", default="", help="comma-separated episode_ids to run (overrides start/count)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="episode shard start index (parallel sweeps)")
    p.add_argument("--count", type=int, default=0, help="episode shard size (0 = to end)")
    p.add_argument("--dynamic-queue", default="",
                   help="path to FileWorkQueue JSON; claim episodes instead of start/count")
    p.add_argument("--worker-id", default="",
                   help="id recorded on each dynamic-queue claim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vlm-parallel-workers", type=int, default=0,
                   help="Max concurrent LA+VA HTTP calls (0 = one per ready agent).")
    p.add_argument("--no-vlm-parallel", action="store_true",
                   help="Force sequential VLM requests.")
    args = p.parse_args(argv)
    args.control_mode = resolve_control_mode(args.control_mode)

    with open(args.episodes) as f:
        data = json.load(f)
    eps = data["episodes"]
    if args.episode_ids:
        want = set(args.episode_ids.split(","))
        eps = [e for e in eps if str(e["episode_id"]) in want]
    elif args.dynamic_queue:
        if args.limit:
            eps = eps[: args.limit]
    elif args.limit:
        eps = eps[: args.limit]
    elif args.start or args.count:        # shard for parallel workers
        eps = eps[args.start: (args.start + args.count) if args.count else None]

    if args.disable_path_budget:
        eps = [dict(e, budget_geo=None) for e in eps]

    adapters = [EpisodeAdapter(d, args.scenes_dir) for d in eps]
    if not args.dynamic_queue:
        adapters.sort(key=lambda e: e.scene_id)  # group scenes -> one sim build each
    flags = _control_mode_flags(args.control_mode, args.clean_found_state)
    print(
        f"[ma-search] {len(adapters)} episodes, use_vlm={args.use_vlm}, "
        f"control_mode={args.control_mode}, "
        f"team_topdown=False, "
        f"async_board_pa=False, "
        f"async_ind_pa={args.control_mode == 'k2-async-keep-goals-pa'}, "
        f"vlm_parallel_workers="
        f"{1 if args.no_vlm_parallel else args.vlm_parallel_workers}, "
        f"dynamic_queue={args.dynamic_queue or 'off'}, "
        f"la_thinking={'off' if args.no_la_thinking else 'on'}, "
        f"history_interval=2, use_backtrack=True, "
        f"pano_n={args.pano_n}, pano_stride_deg={args.pano_stride_deg}"
    )

    cfg = _load_cfg(args.config)
    if int(getattr(args, "map_size_cm", 0) or 0) > 0:
        cfg.defrost()
        cfg.MAP.MAP_SIZE_CM = int(args.map_size_cm)
        cfg.freeze()
        print("[ma-search] MAP_SIZE_CM=%d" % int(args.map_size_cm))
    n_agents = adapters[0].num_agents if adapters else 1
    _set_num_agents(cfg, n_agents)
    runner = MASearchRunner(cfg=cfg, max_episode_steps=args.max_episode_steps,
                            save_dir=args.save_dir, use_vlm=args.use_vlm,
                            la_thinking_budget=args.la_thinking_budget,
                            la_enable_thinking=not args.no_la_thinking,
                            use_continuous_history=True, history_interval=2,
                            visualize=args.visualize,
                            viz_every=args.viz_every,
                            vlm_parallel_workers=(
                                1 if args.no_vlm_parallel
                                else int(args.vlm_parallel_workers)
                            ))
    runner.verify_modality = args.verify
    runner.verify_backend = args.verify_backend
    runner.log_traj = args.log_traj
    runner.skip_pano_viz = args.skip_pano_viz
    runner.pano_n = int(args.pano_n)
    runner.pano_stride_deg = float(args.pano_stride_deg)
    runner.control_mode = args.control_mode
    runner.clean_found_state = args.clean_found_state
    runner.use_async_board_pa = False  # object-board variant, not a paper mode
    runner.use_async_ind_pa = args.control_mode == "k2-async-keep-goals-pa"
    runner.share_found_state = flags["share_found_state"]
    runner.share_found_messages = flags["share_found_messages"]
    runner.share_announcements = flags["share_announcements"]
    runner.share_assignment_hint = flags["share_assignment_hint"]
    runner.use_qwen_claim_scoring = flags["use_qwen_claim_scoring"]
    # The LA picks claim_goal_id among all unresolved goals.
    runner.use_policy_selected_claim_goal = args.control_mode == "k2-async-keep-goals-pa"
    runner.use_mce_primary_frontier = bool(flags.get("use_mce_primary_frontier", False))
    runner.use_frontier = flags["use_frontier"]
    if args.qwen_claim_min_confidence is not None:
        runner.qwen_claim_min_confidence = args.qwen_claim_min_confidence
    runner.multimodal_search = True   # always ON (input contract)
    if args.verify or runner.multimodal_search:   # both need the IIN goal images
        lookup = {}
        img_dir = args.iin_images_dir or ""
        if img_dir and os.path.isdir(img_dir):
            manifests = []
            main = os.path.join(img_dir, "manifest.json")
            if os.path.isfile(main):
                manifests.append(main)
            for name in sorted(os.listdir(img_dir)):
                if name.startswith("manifest.shard") and name.endswith(".json"):
                    manifests.append(os.path.join(img_dir, name))
            for manifest in manifests:
                for it in json.load(open(manifest)).get("items") or []:
                    lookup[(str(it["episode_id"]), int(it["goal_index"]))] = it["image"]
        runner.iin_lookup = lookup
        runner.iin_images_dir = args.iin_images_dir
        print(f"[ma-search] verify={args.verify} multimodal={runner.multimodal_search}; "
              f"{len(lookup)} IIN goal images loaded")
    if args.dynamic_queue:
        from .work_queue import FileWorkQueue

        queue = FileWorkQueue(args.dynamic_queue)
        worker_id = args.worker_id or os.path.basename(os.path.normpath(args.save_dir))
        by_id = {str(ep.episode_id): ep for ep in adapters}
        raw_scene = {str(ep.episode_id): str(ep._raw.get("scene_id") or ep.scene_id)
                     for ep in adapters}

        def claim_fn(prefer_scene=None):
            prefer_raw = None
            if prefer_scene:
                for eid, ep in by_id.items():
                    if ep.scene_id == prefer_scene:
                        prefer_raw = raw_scene.get(eid)
                        break
            item = queue.claim(worker_id, prefer_scene=prefer_raw)
            if item is None:
                return None
            ep = by_id.get(str(item["episode_id"]))
            if ep is None:
                print(f"[ma-search][warn] queue id {item['episode_id']} not in local catalog")
                return claim_fn(prefer_scene=prefer_scene)
            print(
                f"[ma-search] claimed {ep.episode_id} worker={worker_id} "
                f"prefer_scene={bool(prefer_raw)}"
            )
            return ep

        results = runner.evaluate_dynamic(
            claim_fn,
            seed=args.seed,
            on_done=queue.mark_done,
            on_fail=queue.release,
        )
    else:
        runner.evaluate(adapters, seed=args.seed)


if __name__ == "__main__":
    main()
