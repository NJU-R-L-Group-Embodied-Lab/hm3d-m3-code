import base64
import copy
import gzip
import io
import json
import os
import random
import time

import re
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict

import numpy as np
from PIL import Image
from fastdtw import fastdtw
from skimage.morphology import binary_closing
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
import supervision as sv

from habitat import logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

# Import Habitat visualization utilities
from habitat.utils.visualizations import maps

from .utils.prompts import *  # VLN prompts re-exported as module-level for backward compat
from .utils.prompts import get_prompts
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common import environments  # noqa: F401  (registers VLNCEZeroShotEnv)
from vlnce_baselines.common.utils import get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.utils.constant import base_classes, map_channels
from .utils.api import LaViRA_API, env_llm_api_key, env_llm_base_url, env_llm_model
from .utils.depth_utils import get_world_xz_from_pixel
from .utils.visualization import LaViRAVisualizer

import sys
# Dynamic path resolution for Docker/local compatibility
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

NAVDP_ROOT = os.path.join(PROJECT_ROOT, "navdp")
if NAVDP_ROOT not in sys.path:
    sys.path.append(NAVDP_ROOT)

# iplanner was removed from the codebase. The guards below leave downstream
# `if self.use_iplanner:` branches unreachable (default USE_IPLANNER=False).
IPlannerAgent = None
TrajViz = None

try:
    from policy_agent import NavDP_Agent
except ImportError as _e:
    logger.warning(f"NavDP import failed ({_e}); --use-navdp will fail loudly when used.")
    NavDP_Agent = None

EQADataCollector = None  # EQA data collection is not part of this release

import warnings

warnings.filterwarnings('ignore')

class VLMReasoningAgent:
    def __init__(
        self,
        visualizer: LaViRAVisualizer,
        task_type: str = "VLN",
        use_guideline=True,
        use_working_memory=True,
        allow_move_behind=True,
        debug_logging=False,
        log_dir="logs/debug_logs",
        use_todo_list=True,
        backtrack_second_chance=True,
        use_backtrack=True,
        no_todo_prompt=False,
        la_model_override="",
        va_model_override="",
        la_base_url_override="",
        la_api_key_override="",
        va_base_url_override="",
        va_api_key_override="",
        la_enable_thinking=False,
        la_thinking_budget=0,
    ):
        # Credentials/endpoint come from the environment only:
        # LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (DASHSCOPE_*/LAVIRA_API_KEY accepted as key/URL fallbacks).
        va_api_key = env_llm_api_key('VA_API_KEY')
        va_base_url = env_llm_base_url()
        va_model_name = env_llm_model()

        # la_model_name = 'gemini-2.5-pro'
        # la_model_name = 'gemini-3-flash-preview'
        # la_model_name = 'gpt-5.2'

        # la_model_name = 'Qwen/Qwen3-VL-235B-A22B-Instruct'

        la_api_key = env_llm_api_key('LA_API_KEY')
        la_base_url = env_llm_base_url()
        la_model_name = env_llm_model()
        logger.info(f"todolist:{use_todo_list}, backtrack_second_chance:{backtrack_second_chance}")
        # la_model_name = 'gemini-3-pro-preview-thinking'
        # la_model_name = 'qwen3-vl-235b-a22b-instruct'

        # la_model_name = 'gemini-2.5-pro'
        # la_model_name = 'gemini-3-pro-preview-thinking'
        # la_model_name = 'qwen3-vl-235b-a22b-instruct'

        # Apply config-driven LA / VA model and endpoint overrides. Empty string = keep default.
        if la_model_override: la_model_name = la_model_override
        if va_model_override: va_model_name = va_model_override
        if la_base_url_override: la_base_url = la_base_url_override
        if la_api_key_override:  la_api_key  = la_api_key_override
        if va_base_url_override: va_base_url = va_base_url_override
        if va_api_key_override:  va_api_key  = va_api_key_override
        logger.info(f"LA model={la_model_name} base={la_base_url}")
        logger.info(f"VA model={va_model_name} base={va_base_url}")

        self.model = LaViRA_API(
            la_api_key=la_api_key,
            la_base_url=la_base_url,
            la_model_name=la_model_name,
            va_model_name=va_model_name,
            va_api_key=va_api_key,
            va_base_url=va_base_url,
            la_enable_thinking=la_enable_thinking,
            la_thinking_budget=la_thinking_budget,
        )
        self.model.eval()
        self.visualizer = visualizer
        self.stair = False
        self.use_guideline = False
        self.use_working_memory = use_working_memory
        self.use_negative_constraints = True  # Switch to control negative constraints
        self.allow_move_behind = allow_move_behind
        self.use_todo_list = use_todo_list
        self.backtrack_second_chance = backtrack_second_chance
        self.use_backtrack = use_backtrack
        self.no_todo_prompt = no_todo_prompt
        self.task_type = task_type
        self.P = get_prompts(task_type)
        self.collector = None  # set by evaluator for EQA only
        self.guideline = ""
        self.todo_list = None
        self.reasoning_todo = ''
        self.todo_verification_feedback = ""
        self.debug_logging = debug_logging
        self.log_dir = log_dir
        self.last_object = ''
        self.last_va_log_dir = None  # cached path of most recent VA save dir

    def _ensure_va_in_step(self, episode_id, step):
        """Ensure step_<step>/ has a VA save. If VA wasn't called this step, copy
        the last VA save dir as `va_query_cached_<ts>/`."""
        if not self.debug_logging or episode_id is None or step is None:
            return
        if not self.last_va_log_dir or not os.path.isdir(self.last_va_log_dir):
            return
        try:
            import glob, shutil, time
            step_dir = os.path.join(self.log_dir, str(episode_id), f"step_{step}")
            existing = glob.glob(os.path.join(step_dir, 'va_query_*'))
            if existing:
                return
            os.makedirs(step_dir, exist_ok=True)
            dst = os.path.join(step_dir, f"va_query_cached_{time.time()}")
            shutil.copytree(self.last_va_log_dir, dst)
        except Exception as e:
            logger.warning(f'failed to mirror VA into step_{step}: {e}')

    def _ensure_last_object(self, instruction: str, initial_views: List[Dict] = None,
                             episode_id=None, step=None):
        if self.last_object:
            return
        # Extracting a "last object" from a free-text instruction is VLN-specific;
        # for ObjectNav and EQA the target/question is already structured, so skip.
        if self.task_type != "VLN":
            self.last_object = ''
            return

        prompt_text = self.P.LA_PROMPT_LAST_OBJECT_GENERATOR.format(instruction=instruction)
        logger.info("Generating Last Object...")
        logger.info(prompt_text)

        content = [{"type": "text", "text": prompt_text}]
        if initial_views:
            content.extend(initial_views)

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id),
                                    f"step_{step if step is not None else 0}",
                                    f"la_last_object_{time.time()}")

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=4096,
            temperature=0.0,
            use_la=True,
            log_path=log_path,
        )

        logger.info('Last Object:')
        logger.info("%s", output_text)
        self.last_object = output_text.strip()

    def _build_nav_prompt(self, use_backtrack_prompt, negative_constraints, action_list_items, action_desc, feedback):
        prompt_kwargs = {
            "negative_constraints": negative_constraints,
            "action_list": "\n".join(action_list_items),
            "action_desc": action_desc,
            "feedback": feedback,
        }

        if self.no_todo_prompt:
            # Pure FT: NO_TODO templates have no {todo_list} field.
            template = (
                self.P.LA_PROMPT_BACKTRACK_NO_TODO if use_backtrack_prompt
                else self.P.LA_PROMPT_NO_BACKTRACK_NO_TODO
            )
            return template.format(
                negative_constraints=prompt_kwargs["negative_constraints"],
                action_list=prompt_kwargs["action_list"],
                action_desc=prompt_kwargs["action_desc"],
                feedback=prompt_kwargs["feedback"],
            )

        if self.use_todo_list:
            prompt_kwargs["todo_list"] = self._format_todo_for_prompt()
        else:
            prompt_kwargs["todo_list"] = "(none)"
        template = self.P.LA_PROMPT_BACKTRACK if use_backtrack_prompt else self.P.LA_PROMPT_NO_BACKTRACK

        return template.format(**prompt_kwargs)

    @staticmethod
    def _prompt_without_map_understanding(prompt: str) -> str:
        """Standalone LaViRA: do not ask the LA for team-map reading."""
        prompt = re.sub(
            r'\s*"map_understanding":\s*"[^"]*",\s*\n',
            "\n",
            prompt,
            count=1,
        )
        prompt = re.sub(
            r"- When a team top-down map is present,[^\n]*\n",
            "",
            prompt,
        )
        return prompt + (
            "\nThis is a single-agent LaViRA decision. "
            "Do not include map_understanding. Use only first-person views, "
            "navigation history, and the target list.\n"
        )

    def img_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64

    def _get_initial_views(self, visited_targets):
        """Helper to extract initial views from visited_targets."""
        initial_views = []
        if visited_targets and len(visited_targets) > 0:
            panorama_images = visited_targets[0]['panorama_frames']
            from benchmark.pano_views import view_specs as _pano_view_specs
            view_definitions = _pano_view_specs(len(panorama_images), prefix="Initial")
            for view in view_definitions:
                frame_idx = int(view["index"])
                if frame_idx < len(panorama_images):
                    rgb_image = panorama_images[frame_idx]['rgb']
                    if isinstance(rgb_image, np.ndarray):
                        if rgb_image.dtype != np.uint8:
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_image)
                    else:
                        img = rgb_image
                    initial_views.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    initial_views.append({"type": "text", "text": view['label']})
        return initial_views

    def _check_blocked_directions(self, panorama_images):
        """
        Helper to check blocked directions based on depth.
        Returns a set of blocked direction names.
        """
        from benchmark.pano_views import blocked_frame_indices, view_specs

        blocked_idx = blocked_frame_indices(panorama_images)
        blocked_directions = set()
        for spec in view_specs(len(panorama_images or [])):
            if int(spec["index"]) in blocked_idx and spec["name"] in (
                "forward", "left", "behind", "right",
            ):
                blocked_directions.add(spec["name"])
        return blocked_directions

    def _blocked_frame_indices(self, panorama_images):
        from benchmark.pano_views import blocked_frame_indices

        return blocked_frame_indices(panorama_images)

    def _image_content(self, img):
        if img is None:
            return None
        if isinstance(img, np.ndarray):
            rgb = img
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)
            img = Image.fromarray(rgb)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"},
        }

    def _waypoint_label(self, waypoint_id, target=None):
        label = f"Waypoint {waypoint_id}"
        if target and target.get("is_revisit") and target.get("original_waypoint_id") is not None:
            label += f" (revisit of Waypoint {target['original_waypoint_id']})"
        return label

    def _append_history_images(self, content, items):
        for item in items:
            packed = self._image_content(item.get("image") if isinstance(item, dict) else item)
            if packed is None:
                continue
            step = item.get("step") if isinstance(item, dict) else None
            if step is not None:
                content.append({"type": "text", "text": f"history step {int(step):06d}"})
            content.append(packed)

    def _build_history_queue_content(self, history_images, visited_targets=None):
        """LaViRA history: RGB queue sliced by waypoint legs.

        Each completed decision is a waypoint. Frames sampled every N env steps
        are grouped under ``Waypoint i -> Waypoint i+1`` so the LA can name a
        waypoint id when it backtracks.
        """
        if not history_images:
            return []
        wps = list(visited_targets or [])
        content = []

        def _step_of(wp, default=0):
            try:
                return int(wp.get("step", default))
            except (TypeError, ValueError):
                return default

        if len(wps) <= 1:
            label = "Start -> Waypoint 0:" if wps else "Start -> Current Position:"
            content.append({"type": "text", "text": label})
            self._append_history_images(content, history_images)
            return content

        first_step = _step_of(wps[0])
        start_items = [
            item for item in history_images
            if int(item.get("step", -1)) <= first_step
        ]
        if start_items:
            content.append({
                "type": "text",
                "text": f"Start -> {self._waypoint_label(0, wps[0])}:",
            })
            self._append_history_images(content, start_items)

        for i in range(len(wps) - 1):
            lo = _step_of(wps[i])
            hi = _step_of(wps[i + 1], lo)
            selected = [
                item for item in history_images
                if lo < int(item.get("step", -1)) <= hi
            ]
            if not selected:
                continue
            content.append({
                "type": "text",
                "text": (
                    f"{self._waypoint_label(i, wps[i])} -> "
                    f"{self._waypoint_label(i + 1, wps[i + 1])}:"
                ),
            })
            self._append_history_images(content, selected)

        last_step = _step_of(wps[-1])
        tail = [
            item for item in history_images
            if int(item.get("step", -1)) > last_step
        ]
        if tail:
            content.append({
                "type": "text",
                "text": f"{self._waypoint_label(len(wps) - 1, wps[-1])} -> Current Position:",
            })
            self._append_history_images(content, tail)
        return content

    def navigate_or_backtrack(self, instruction, visited_targets, feedback="", episode_id=None,
                              step=None, history_images=None, target_reference_images=None,
                              team_topdown_map=None, team_topdown_legend=None):
        """
        Use the LA model on instruction + history + directional images to pick a direction or backtrack.
        Returns a language-level action: navigate to [left, right, forward, behind] / backtrack to <waypoint_id>

        Optional team_topdown_map: RGB uint8 HxWx3 shared geometric map with agent
        poses and FOUND/LEASE markers. team_topdown_legend is the matching text key.
        """

        # Images for the four directions (front, left, back, right)
        panorama_images = visited_targets[-1]['panorama_frames'] if visited_targets else []

        history_content = self._build_history_queue_content(history_images, visited_targets)
        if history_images is None:
            # Add history waypoint images and action descriptions (Legacy)
            for i, target in enumerate(visited_targets[:-1]):  # exclude the current unfinished waypoint
                if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    # history_content.append({"type": "text", "text": f"Waypoint {i}: Initial view"})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})

                # if 'turn_action' in target:
                #     history_content.append({"type": "text", "text": f"Action: {target['turn_action']}"})

                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})

                if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        # Build the current panorama (4 directions or 8 views at 45 deg)
        current_views = []
        from benchmark.pano_views import view_specs as _pano_view_specs

        # Compute blocked directions via helper
        blocked_directions = self._check_blocked_directions(panorama_images)
        num_frames = len(panorama_images)
        view_definitions = _pano_view_specs(num_frames, prefix="Current")

        for view in view_definitions:
            frame_idx = int(view["index"])
            if frame_idx < len(panorama_images):
                rgb_image = panorama_images[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append({"type": "text", "text": view['label']})
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})

        # Decide whether to offer the backtrack option
        # Policy: offer backtrack every 2-3 waypoints, or probabilistically by waypoint count
        num_waypoints = max(0, len(visited_targets) - 1)
        should_consider_backtrack = bool(self.use_backtrack)

        # Build the prompt variant
        content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\""}]
        if target_reference_images:
            content.append({
                "type": "text",
                "text": (
                    "\nTarget reference images follow. They specify target identity only; "
                    "they are NOT observations, navigation history, or evidence that a target is found."
                ),
            })
            for item in target_reference_images:
                image = item.get("image")
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype(np.uint8))
                if image is None:
                    continue
                content.append({
                    "type": "text",
                    "text": f"TARGET REFERENCE for {item.get('label') or item.get('goal_id')}",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(image)}"},
                })
        from benchmark.async_board_pa import (
            HISTORY_SECTION_LABEL,
            current_views_section_label,
            has_la_field_contract,
        )
        if has_la_field_contract(instruction):
            hist_label = HISTORY_SECTION_LABEL
            cur_label = current_views_section_label(num_frames)
        else:
            hist_label = "\nNavigation History:"
            cur_label = (
                "\nCurrent 8-view 45° panorama:"
                if num_frames == 8 else
                "\nCurrent 4-directional views:"
            )
        content.append({"type": "text", "text": hist_label})
        content.extend(history_content)
        if self.use_guideline and hasattr(self, 'guideline') and self.guideline:
            pass
            # content.append({"type": "text", "text": "\nTask-specific guidelines:"})
            # content.append({"type": "text", "text": self.guideline})
        # content.append()
        if self.use_working_memory and len(visited_targets) >= 2:
            # Working memory
            content.append({"type": "text", "text": "\nWorking Memory (previous waypoint snapshot): compare these with current views to judge progress, repeated scenes."})
            target = visited_targets[-2]
            view_definitions = _pano_view_specs(len(panorama_images), prefix="Previous")
            previous_views = []

            for view in view_definitions:
                frame_idx = int(view["index"])
                if frame_idx < len(panorama_images):
                    rgb_image = panorama_images[frame_idx]['rgb']
                    if isinstance(rgb_image, np.ndarray):
                        if rgb_image.dtype != np.uint8:
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_image)
                    else:
                        img = rgb_image
                    previous_views.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    previous_views.append({"type": "text", "text": view['label']})
                
            content.extend(previous_views)   


        content.append({"type": "text", "text": cur_label})
        content.extend(current_views)

        # Optional shared team top-down map (geometry + agents + FOUND markers).
        has_team_topdown = team_topdown_map is not None
        if has_team_topdown:
            content.append({
                "type": "text",
                "text": (
                    "\nShared team top-down map (geometric occupancy jointly built by all agents). "
                    "The legend names YOUR color — find that colored triangle. "
                    "Map orientation: up=North, down=South, left=West, right=East. "
                    "BEFORE choosing an action you MUST interpret this map: your location, "
                    "teammate location/heading, where explored (gray) vs unexplored (white) is, "
                    "FOUND goals listed in the text legend (not drawn on the map), "
                    "and the N/S/E/W direction of where you should go. "
                    "Colored X marks in an agent color are short-term navigation waypoints only "
                    "(not verified objects). "
                    "The map is NOT first-person evidence that a target is currently visible for STOP."
                ),
            })
            if team_topdown_legend:
                content.append({"type": "text", "text": str(team_topdown_legend)})
            map_img = team_topdown_map
            if isinstance(map_img, np.ndarray):
                if map_img.dtype != np.uint8:
                    map_img = np.clip(map_img, 0, 255).astype(np.uint8)
                map_img = Image.fromarray(map_img)
            content.append({"type": "text", "text": "TEAM TOP-DOWN MAP:"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(map_img)}"
                },
            })

        # Ensure reasoning state is available for the current episode.
        if self.use_todo_list and self.todo_list is None:
            initial_views = self._get_initial_views(visited_targets)
            self.generate_todo_list(instruction, initial_views, episode_id=episode_id, step=step)
        elif not self.last_object:
            initial_views = self._get_initial_views(visited_targets)
            self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)

        # Generate negative constraints based on target object
        target_objects = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
        target = None
        instruction_lower = instruction.lower()
        for obj in target_objects:
            obj_name = obj.replace('_', ' ')
            if obj_name in instruction_lower or obj in instruction_lower:
                target = obj
                break
        
        negative_constraints = ""
        if self.use_negative_constraints:
            if target:
                other_objects = [obj.replace('_', ' ') for obj in target_objects if obj != target]
                negative_constraints = f"- When looking for {target.replace('_', ' ')}, do not confuse it with other objects: {', '.join(other_objects)}."

        # Generate action list dynamically
        action_list_items = []
        action_desc_parts = []
        use_view_actions = num_frames == 8

        if use_view_actions:
            from benchmark.pano_views import navigate_action_choices

            action_list_items, action_desc = navigate_action_choices(
                num_frames, self._blocked_frame_indices(panorama_images)
            )
        else:
            # Directions we want to consider
            candidate_directions = ['forward', 'left', 'right']

            # Check if we are at the initial position (only current waypoint exists)
            is_initial_position = len(visited_targets) <= 1
            if self.allow_move_behind or is_initial_position:
                candidate_directions.append('behind')

            # Build action list based on valid directions (using blocked_directions calculated earlier)
            valid_directions = [d for d in candidate_directions if d not in blocked_directions]

            # Fallback: if all valid directions are blocked, allow behind or forward to avoid getting stuck
            if not valid_directions:
                # logger.info("All directions blocked! Force enabling behind/forward.")
                if 'behind' in candidate_directions:
                    valid_directions.append('behind')
                elif 'forward' in candidate_directions:
                    valid_directions.append('forward')

            for direction in valid_directions:
                if direction == 'forward':
                    action_list_items.append("   - navigate to forward - continue straight ahead")
                elif direction == 'left':
                    action_list_items.append("   - navigate to left - turn left and go forward")
                elif direction == 'right':
                    action_list_items.append("   - navigate to right - turn right and go forward")
                elif direction == 'behind':
                    action_list_items.append("   - navigate to behind - turn around and go forward")

                action_desc_parts.append("navigate to " + direction)

            if not action_desc_parts:
                 # Should be covered by fallback, but just in case
                 action_desc = "navigate to behind"
                 action_list_items.append("   - navigate to behind - turn around and go forward (Emergency)")
            else:
                 action_desc = "|".join(action_desc_parts)



        if should_consider_backtrack and num_waypoints > 0:
            # Collect available waypoint IDs (excluding current one which is just created)
            # visited_targets includes current one at end.
            # Available indices are 0 to len(visited_targets) - 2
            
            # Filter waypoints by distance
            MAX_BACKTRACK_DISTANCE = 6.0 # meters
            current_target = visited_targets[-1]
            current_pos = current_target.get('world_coords')
            
            available_ids = []
            if current_pos:
                for i in range(len(visited_targets) - 1):
                    target = visited_targets[i]
                    target_pos = target.get('world_coords')
                    if target_pos:
                        dist_meters = float(np.sqrt(
                            (float(current_pos[0]) - float(target_pos[0])) ** 2
                            + (float(current_pos[1]) - float(target_pos[1])) ** 2
                        ))
                        if dist_meters <= MAX_BACKTRACK_DISTANCE:
                            available_ids.append(str(i))
            else:
                # Fallback if no coords
                available_ids = [str(i) for i in range(len(visited_targets) - 1)]

            if available_ids:
                id_list_str = ", ".join(available_ids)
                action_list_items.append(f"   - backtrack to <waypoint_id> - return to a previous waypoint (Available IDs: {id_list_str})")
                action_desc += ' or "backtrack to <waypoint_id>"'
                prompt = self._build_nav_prompt(
                    use_backtrack_prompt=True,
                    negative_constraints=negative_constraints,
                    action_list_items=action_list_items,
                    action_desc=action_desc,
                    feedback=feedback,
                )
            else:
                prompt = self._build_nav_prompt(
                    use_backtrack_prompt=False,
                    negative_constraints=negative_constraints,
                    action_list_items=action_list_items,
                    action_desc=action_desc,
                    feedback=feedback,
                )
        else:
            prompt = self._build_nav_prompt(
                use_backtrack_prompt=False,
                negative_constraints=negative_constraints,
                action_list_items=action_list_items,
                action_desc=action_desc,
                feedback=feedback,
            )

        if not has_team_topdown:
            prompt = self._prompt_without_map_understanding(prompt)

        if "ASYNC_BOARD_PA_FIELD_CONTRACT" in (instruction or "") or (
            "ASYNC_IND_PA_FIELD_CONTRACT" in (instruction or "")
        ):
            from benchmark.async_board_pa import apply_la_json_field_contract

            prompt = apply_la_json_field_contract(
                prompt,
                independent_plus_pa="ASYNC_IND_PA_FIELD_CONTRACT" in (instruction or ""),
                pano_n=num_frames,
            )

        if num_frames == 8:
            prompt = (
                prompt.replace("current 4-directional views", "current 8-view 45° panorama")
                .replace("Current 4-directional views", "Current 8-view 45° panorama")
                .replace("4-directional views", "8-view 45° panorama")
                .replace("4-dir views", "8-view 45° panorama")
            )

        if "claim_goal_id" in instruction:
            claim_kind = "unresolved ID"
            prompt += (
                "\nClaim binding requirement:\n"
                "- When stop is true, the response JSON MUST include "
                f'"claim_goal_id": "goal_N" using one exact allowed {claim_kind}.\n'
                "- When stop is false, claim_goal_id may be an empty string."
            )

        # When a team top-down map is provided, force a structured map read first.
        if has_team_topdown:
            prompt += (
                "\n\nMAP UNDERSTANDING REQUIREMENT (mandatory when team top-down map is present):\n"
                "- First fill `map_understanding` BEFORE choosing the action.\n"
                "- In `map_understanding`, state: (1) where YOU are — identify the marker "
                "whose color matches YOUR color in the legend; (2) teammate position and "
                "facing by their colors; (3) explored (gray) vs unexplored (white); "
                "(4) any FOUND goals named in the legend (they are not drawn on the map); "
                "(5) the N/S/E/W direction of the place you intend to go next "
                "(up=North, down=South, left=West, right=East; e.g. toward unexplored north "
                "of me / toward my same-color X waypoint).\n"
                "- Then fill `progress_analysis`, reasoning, and `action` based on BOTH the "
                "map understanding and the first-person views.\n"
                "- Response JSON MUST include "
                '"map_understanding": "<your spatial reading of the top-down map>".'
            )

        logger.info(prompt)

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Generate the LA response
        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
             import time
             log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"la_nav_decision_{time.time()}")
             # mirror most recent VA into this step folder if VA wasn't called here
             self._ensure_va_in_step(episode_id, step)

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.0,
            use_la=True,  # LA model
            log_path=log_path,
        )

        logger.info('4o-response:')
        logger.info(f"{output_text}")
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)

        # Parse JSON response
        retry_count = 0
        max_json_retries = 5
        while not json_match and retry_count < max_json_retries:
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=8192,
                temperature=0.0,
                use_la=True,  # LA model
                log_path=log_path,
            )
            logger.info('Retried.')
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            retry_count += 1

        if retry_count >= max_json_retries and not json_match:
             logger.error("Failed to get valid JSON response after max retries")
             return {
                'action': 'NAVIGATE',
                'direction': 'forward',
                'progress_analysis': 'Failed to parse JSON',
                'reasoning': 'Max retries reached'
             }

        if json_match:
            try:
                response_data = json.loads(json_match.group())
            except:
                response_data = {}
            
            if self.use_todo_list:
                # New diff-based mechanism: apply todo_updates if present
                if 'todo_updates' in response_data and response_data['todo_updates']:
                    self._apply_todo_updates(response_data['todo_updates'])
                    logger.info(f"Applied TODO updates. Current list: {self.todo_list}")
                # Legacy fallback: support old 'updated_todo_list' field (full markdown rewrite)
                elif 'updated_todo_list' in response_data:
                    legacy = response_data['updated_todo_list']
                    if isinstance(legacy, list):
                        legacy = "\n".join(legacy)
                    if isinstance(self.todo_list, list) and isinstance(legacy, str):
                        # Old format received but we have new structured - ignore
                        logger.info("Ignoring legacy updated_todo_list field (using structured format)")
                    else:
                        self.todo_list = legacy
                        logger.info(f"Updated TODO List (legacy): {legacy}")

            action = response_data.get('action', 'navigate to forward')
            progress_analysis = response_data.get('progress_analysis', '')
            map_understanding = str(response_data.get('map_understanding', '') or '').strip()
            if map_understanding:
                # Keep map reading visible in progress analysis for logging/debug.
                progress_analysis = (
                    f"[MAP] {map_understanding}\n[PROGRESS] {progress_analysis}"
                    if progress_analysis else f"[MAP] {map_understanding}"
                )
            # Two-stage reasoning fields: reasoning_todo (for todo updates) and
            # reasoning_action (for the chosen action). Fall back to legacy 'reasoning'
            # if either is missing.
            legacy_reasoning = response_data.get('reasoning', '')
            self.reasoning_todo = response_data.get('reasoning_todo', legacy_reasoning)
            reasoning = response_data.get('reasoning_action', legacy_reasoning)
            # HM3D-M3 / same-floor ObjectNav: ignore leftover stair fields.
            self.stair = False if self.no_todo_prompt else response_data.get('stair', False)
            self.last_map_understanding = map_understanding

            if self.stair:
                logger.info(f"DEBUG: LA Model detected stairs: {self.stair}")
            stop_signal = response_data.get('stop', False)
            claim_goal_id = str(response_data.get('claim_goal_id') or '').strip()
            action = action.lower()

            if action == 'stop' or 'stop' in action.split():
                 stop_signal = True
                 if action == 'stop':
                     action = 'navigate to forward'

            # v3: TODO-based STOP verification removed entirely. LA's stop=true is
            # trusted unconditionally; the only remaining gate is the double-check
            # STOP LLM pass (visual verification that target is reached).

            if action.startswith('backtrack to'):
                waypoint_id = action.split('backtrack to ')[-1].strip()
                if waypoint_id.startswith('waypoint'):
                    waypoint_id = waypoint_id.split('waypoint')[-1].strip()
                # logger.info('Waypoint:%s', waypoint_id)
                try:
                    waypoint_id = int(waypoint_id)
                    return {
                        'action': 'BACKTRACK',
                        'waypoint': waypoint_id,
                        'progress_analysis': progress_analysis,
                        'reasoning': reasoning
                    }
                except:
                    pass  # on parse failure fall through to navigation

            # Parse direction: prefer 8-view 'view k', then cardinal / legacy focus_view
            from benchmark.pano_views import (
                nearest_cardinal_name,
                parse_focus_view,
                parse_navigate_action,
            )

            frame_idx = parse_navigate_action(action, num_frames)
            if frame_idx is None:
                frame_idx = parse_focus_view(
                    response_data.get('focus_view', response_data.get('view_index')),
                    num_frames,
                )
            if frame_idx is None:
                if 'forward' in action:
                    direction = 'forward'
                elif 'left' in action:
                    direction = 'left'
                elif 'right' in action:
                    direction = 'right'
                elif 'behind' in action:
                    direction = 'behind'
                else:
                    direction = 'forward'
                from benchmark.pano_views import direction_to_frame_idx
                frame_idx = direction_to_frame_idx(direction, num_frames)
            direction = nearest_cardinal_name(frame_idx, num_frames)

            return {
                'action': 'NAVIGATE',
                'direction': direction,
                'progress_analysis': progress_analysis,
                'reasoning': reasoning,
                'stop_signal': stop_signal,
                'claim_goal_id': claim_goal_id,
                'focus_view': frame_idx,
                'frame_idx': frame_idx,
                'la_action': action,
            }

        # Default: forward
        return {
            'action': 'NAVIGATE',
            'direction': 'forward',
            'progress_analysis': 'Unable to analyze due to parsing error',
            'reasoning': 'Fallback to forward navigation'
        }

    def generate_todo_list(self, instruction: str, initial_views: List[Dict] = None,
                           episode_id=None, step=None):
        """
        Generate initial TODO list as a structured list of dicts.
        Each item: {"content": str, "status": "pending"|"in_progress"|"completed", "result": str}
        """
        if not self.use_todo_list:
            self.todo_list = None
            self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)
            return

        # For VLN, merge TODO generation and last-object extraction into one LA call.
        # For other task types, last_object is not used (see _ensure_last_object).
        if self.task_type == "VLN" and hasattr(self.P, "LA_PROMPT_TODO_AND_LAST_OBJECT"):
            prompt_text = self.P.LA_PROMPT_TODO_AND_LAST_OBJECT.format(instruction=instruction)
            logger.info("Generating initial TODO list + last object (merged)...")
            log_purpose = "la_todo_and_last_object"
        else:
            prompt_text = self.P.LA_PROMPT_TODO_GENERATOR.format(instruction=instruction)
            logger.info("Generating initial TODO list...")
            log_purpose = "la_todo_generator"
        logger.info(prompt_text)

        content = [{"type": "text", "text": prompt_text}]
        if initial_views:
            content.extend(initial_views)

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id),
                                    f"step_{step if step is not None else 0}",
                                    f"{log_purpose}_{time.time()}")

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=4096,
            temperature=0.0,
            use_la=True,
            log_path=log_path,
        )

        logger.info('TODO+LastObject Output:' if self.task_type == "VLN" else 'TODO List Output:')
        logger.info("%s", output_text)

        # Parse JSON TODO list (also picks up last_object if present)
        self.todo_list = self._parse_todo_json(output_text, instruction)
        logger.info(f"Parsed TODO list ({len(self.todo_list)} items): {self.todo_list}")

        if self.task_type == "VLN":
            import re as _re, json as _json
            m = _re.search(r'\{[\s\S]*\}', output_text)
            if m:
                try:
                    data = _json.loads(m.group())
                    lo = data.get("last_object", "")
                    if isinstance(lo, str) and lo.strip():
                        self.last_object = lo.strip()
                        logger.info(f"Last Object (merged): {self.last_object}")
                except Exception as e:
                    logger.warning(f"Failed to parse last_object from merged output: {e}")

        # Fallback: if last_object still empty (parse failed or non-VLN), call separate API
        self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)

    def _parse_todo_json(self, text: str, instruction: str) -> List[Dict]:
        """Parse JSON TODO list from LA output. Fallback to single-item list if parse fails."""
        import json as _json
        import re as _re
        m = _re.search(r'\{[\s\S]*\}', text)
        if not m:
            logger.warning(f"Failed to parse TODO JSON, using fallback. Raw: {text[:200]}")
            return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]
        try:
            data = _json.loads(m.group())
            todos = data.get("todos", [])
            if not isinstance(todos, list) or not todos:
                return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]
            parsed = []
            for t in todos:
                if isinstance(t, dict) and "content" in t:
                    parsed.append({
                        "content": str(t["content"]),
                        "status": t.get("status", "pending") if t.get("status") in ("pending", "completed") else "pending",
                        "result": str(t.get("result", "")),
                    })
            if not parsed:
                return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]
            return parsed
        except Exception as e:
            logger.warning(f"TODO JSON parse error: {e}. Raw: {text[:200]}")
            return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]

    def _format_todo_for_prompt(self) -> str:
        """Render structured TODO as a compact text block for prompts."""
        if not self.todo_list:
            return "(empty)"
        if isinstance(self.todo_list, str):
            # Legacy format: already a string
            return self.todo_list
        lines = []
        for i, t in enumerate(self.todo_list):
            status = t.get("status", "pending")
            content = t.get("content", "")
            result = t.get("result", "")
            if status == "completed" and result:
                lines.append(f"[{i}] ({status})  {content}  => {result}")
            else:
                lines.append(f"[{i}] ({status})  {content}")
        return "\n".join(lines)

    def _apply_todo_updates(self, updates) -> None:
        """Apply incremental updates (diff) to self.todo_list. Silently skip invalid ops."""
        if not updates or not isinstance(updates, list):
            return
        if not isinstance(self.todo_list, list):
            # Legacy string TODO, skip updates
            return
        # v3: only two statuses — "pending" and "completed". No in_progress.
        VALID_STATUS = ("pending", "completed")
        for u in updates:
            if not isinstance(u, dict):
                continue
            op = u.get("op", "update")
            try:
                if op == "update":
                    idx = u.get("index")
                    if not isinstance(idx, int) or idx < 0 or idx >= len(self.todo_list):
                        continue
                    new_status = u.get("status")
                    new_result = u.get("result")
                    if new_status in VALID_STATUS:
                        # Discipline: marking "completed" without a result is rolled
                        # back to "pending" — force LA to back up the claim with an
                        # observation, or it won't count as done.
                        if new_status == "completed" and (new_result is None or not str(new_result).strip()):
                            logger.info(f"TODO idx={idx} completion lacks result; rolled back to pending")
                            new_status = "pending"
                        self.todo_list[idx]["status"] = new_status
                    if new_result is not None:
                        self.todo_list[idx]["result"] = str(new_result)
                elif op == "rewrite":
                    # Allow LA to refine a pending item's content. Blocked on completed.
                    idx = u.get("index")
                    if not isinstance(idx, int) or idx < 0 or idx >= len(self.todo_list):
                        continue
                    if self.todo_list[idx].get("status") == "completed":
                        logger.info(f"TODO idx={idx} rewrite skipped (already completed)")
                        continue
                    if "content" in u and str(u["content"]).strip():
                        self.todo_list[idx]["content"] = str(u["content"])
                    if "status" in u and u["status"] == "pending":
                        self.todo_list[idx]["status"] = "pending"  # no-op but explicit
                    if "result" in u:
                        self.todo_list[idx]["result"] = str(u["result"])
                elif op in ("add", "insert"):
                    # Unified: append by default, or insert at `index` if provided.
                    if "content" not in u:
                        continue
                    new_item = {
                        "content": str(u["content"]),
                        "status": u.get("status") if u.get("status") in VALID_STATUS else "pending",
                        "result": str(u.get("result", "")),
                    }
                    idx = u.get("index")
                    if isinstance(idx, int):
                        idx = max(0, min(idx, len(self.todo_list)))
                        self.todo_list.insert(idx, new_item)
                    else:
                        self.todo_list.append(new_item)
                elif op == "remove":
                    idx = u.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(self.todo_list):
                        self.todo_list.pop(idx)
            except Exception as e:
                logger.warning(f"apply_todo_update skipped item {u}: {e}")

    def _todo_pending_summary(self):
        """Return (num_incomplete, list_of_incomplete_contents). Handles str or list format."""
        if not self.use_todo_list or not isinstance(self.todo_list, list):
            return 0, []
        incomplete = [t for t in self.todo_list if t.get("status") != "completed"]
        return len(incomplete), [t.get("content", "") for t in incomplete]

    def query_llm(self,
                  instruction: str,
                  visited_targets: List[Dict[str, str]],
                  rgb_image: np.ndarray,
                  depth_image: np.ndarray,
                  width: int,
                  height: int,
                  current_step: int,
                  progress_analysis: str = None,
                  planned_action: str = 'NAVIGATE',
                  episode_id=None):
        try:
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                img = Image.fromarray(rgb_image)
            else:
                img = rgb_image

            if isinstance(depth_image, np.ndarray):
                if depth_image.dtype != np.uint8:
                    depth_image = (depth_image * 255).astype(np.uint8)
                if depth_image.ndim == 3 and depth_image.shape[2] == 1:
                    depth_image = np.squeeze(depth_image, axis=2)
                depth_img = Image.fromarray(depth_image)
            else:
                depth_img = depth_image

            # Build visited targets history string
            # visited_targets_str = ""
            # visited_targets
            # if len(visited_targets) > 1:
            #     visited_targets_str = f"\n\nPreviously visited targets:\n"
            #     for i, target in enumerate(visited_targets[:-1], 0):
            #         visited_targets_str += f"{i}. {target['description']} (Step {target['step']})\n"
            # else:
            #     visited_targets_str = "\n\nNo targets visited yet."

            content = [{
                'type': 'text',
                'text': f'You are a robot performing navigation task. Look at this image and the corresponding depth gray image and help the robot navigate.'
            }]

            img_base64 = self.img_to_base64(img)
            depth_base64 = self.img_to_base64(depth_img)

            content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{img_base64}"}})
            content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{depth_base64}"}})

            # Build the prompt including progress_analysis
            progress_info = ""
            if progress_analysis:
                progress_info = f"\nProgress Analysis from Navigation Decision: {progress_analysis}\n"
            
            # Re-generate negative constraints logic as it is local variable in navigate_or_backtrack
            # We need to extract it again or pass it. 
            # Since this method is called independently, we re-extract target.
            target_objects = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
            target = None
            instruction_lower = instruction.lower()
            for obj in target_objects:
                obj_name = obj.replace('_', ' ')
                if obj_name in instruction_lower or obj in instruction_lower:
                    target = obj
                    break
            
            negative_constraints = ""
            if self.use_negative_constraints:
                if target:
                    other_objects = [obj.replace('_', ' ') for obj in target_objects if obj != target]
                    negative_constraints = f"- When looking for {target.replace('_', ' ')}, do not confuse it with other objects: {', '.join(other_objects)}."

            if self.no_todo_prompt:
                # VA_PROMPT_NO_TODO has no {todo_list} field.
                prompt = self.P.VA_PROMPT_NO_TODO.format(
                    instruction=instruction,
                    current_step=current_step,
                    width=width,
                    height=height,
                    progress_info=progress_info,
                    planned_action=planned_action,
                )
            else:
                if self.use_todo_list:
                    if self.todo_list is None:
                        todo_text = "No TODO list available."
                    else:
                        todo_text = self._format_todo_for_prompt()
                else:
                    todo_text = "(none)"
                prompt = self.P.VA_PROMPT.format(
                    instruction=instruction,
                    current_step=current_step,
                    width=width,
                    height=height,
                    todo_list=todo_text,
                    progress_info=progress_info,
                    planned_action=planned_action,
                )
            logger.info(prompt)
            content.append({
                "type": "text",
                "text": prompt
            })

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            log_path = None
            if self.debug_logging and episode_id is not None:
                import time, glob, shutil
                step_dir = os.path.join(self.log_dir, str(episode_id), f"step_{current_step}")
                # remove any prior cached VA mirror in this step — a real VA call now happens
                for _stale in glob.glob(os.path.join(step_dir, 'va_query_cached_*')):
                    try:
                        shutil.rmtree(_stale)
                    except Exception:
                        pass
                log_path = os.path.join(step_dir, f"va_query_{time.time()}")
                self.last_va_log_dir = log_path

            # Use the API client to generate response
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=10240,
                temperature=0,
                log_path=log_path,
                extra_body={"enable_thinking": False}
            )
            logger.info('LLM Output:')
            logger.info("%s", output_text)

            # Try to parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            retry_count = 0
            max_json_retries = 5
            while not json_match and retry_count < max_json_retries:
                output_text = self.model.generate(
                    messages=messages,
                    max_new_tokens=4096,
                    temperature=0.0,
                    log_path=log_path,
                    extra_body={"enable_thinking": False}
                )
                logger.info('Retried LLM Output:')
                logger.info("%s", output_text)
                json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
                retry_count += 1

            if retry_count >= max_json_retries and not json_match:
                 logger.error("Failed to get valid JSON response in query_llm after max retries")
                 # Fallback will be handled by exception or check below
                 json_match = None

            if json_match:
                try:
                    response_data = json.loads(json_match.group())
                except:
                    response_data = {}

                # Extract action decision
                action_decision = response_data.get('action', planned_action).upper()

                # Qwen3.5 emits [x1, y1, x2, y2] in its native 0..999
                # coordinate system. Convert to source-image pixels exactly
                # once; malformed boxes use a pixel-space center fallback.
                bbox_2d = response_data.get('bbox_2d')
                try:
                    values = [float(v) for v in bbox_2d[:4]]
                    if len(values) != 4:
                        raise ValueError("bbox_2d must have four coordinates")
                    nx1, ny1, nx2, ny2 = values
                    if not (0.0 <= nx1 < nx2 <= 999.0 and 0.0 <= ny1 < ny2 <= 999.0):
                        raise ValueError("bbox_2d must be ordered within 0..999")
                    x1, y1 = nx1 / 999.0 * width, ny1 / 999.0 * height
                    x2, y2 = nx2 / 999.0 * width, ny2 / 999.0 * height
                except (TypeError, ValueError):
                    x1, y1 = width / 4.0, height / 4.0
                    x2, y2 = 3.0 * width / 4.0, 3.0 * height / 4.0

                # Convert to x, y, width, height format for internal use
                x = int(x1)
                y = int(y1)
                img_w, img_h = width, height
                width = int(x2 - x1)
                height = int(y2 - y1)

                bbox = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'target': response_data.get('target', 'unknown target'),
                    'action': action_decision,
                    'reasoning': response_data.get('reasoning', 'No reasoning provided'),
                    'progress': response_data.get('progress', 'Unknown progress')
                }

                # Ensure bbox is within image bounds
                bbox['x1'] = max(0, min(bbox['x1'], img_w - 1))
                bbox['y1'] = max(0, min(bbox['y1'], img_h - 1))
                bbox['x2'] = max(bbox['x1'] + 1, min(bbox['x2'], img_w))
                bbox['y2'] = max(bbox['y1'] + 1, min(bbox['y2'], img_h))

                # Update x, y, width, height based on bounded coordinates
                bbox['x'] = bbox['x1']
                bbox['y'] = bbox['y1']
                bbox['width'] = bbox['x2'] - bbox['x1']
                bbox['height'] = bbox['y2'] - bbox['y1']

                # logger.info("Parsed response:", bbox)
                # logger.info(f"BBox 2D format - x1:{bbox['x1']}, y1:{bbox['y1']}, x2:{bbox['x2']}, y2:{bbox['y2']}")
                # logger.info(f"Action: {action_decision}, Target: {bbox['target']}")
                # logger.info(f"Reasoning: {bbox['reasoning']}")
                # logger.info(f"Progress: {bbox['progress']}")

                # Record this target if it's a new navigation target
                if action_decision == 'NAVIGATE' and bbox['target'] != 'unknown target':
                    target_record = {
                        'step': current_step,
                        'description': bbox['target'],
                        'bbox': {
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'x1': bbox['x1'],
                            'y1': bbox['y1'],
                            'x2': bbox['x2'],
                            'y2': bbox['y2']
                        },
                        'reasoning': bbox['reasoning']
                        # world_coords removed: it is now set at waypoint creation
                    }

                    if visited_targets is not None:
                        visited_targets[-1].update(target_record)
                        # logger.info(f"Added target to history: {bbox['target']}")
                        # logger.info(visited_targets[-1])

                # Calculate target coordinates for visualization
                if not self.stair:
                    # coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int((bbox.get('y1', 0) + 3 * bbox.get('y2', 0)) / 4.0))
                    coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), min(int(bbox.get('y2', 0)), self.visualizer.height - 1))
                else:
                    # Stair navigation
                    # If 'up' (or True for legacy), target top (y1). If 'down', target bottom (y2).
                    if self.stair == 'down':
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y2', 0)))
                    else:
                        # Default to 'up' behavior (y1)
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y1', 0)))

                # Save RGB image with bounding box annotation
                self.visualizer._save_rgb_with_bbox(rgb_image, bbox, target_coords=coords)

                return bbox
            else:
                logger.info("Failed to parse JSON response, using default")

        except Exception as e:
            logger.info(f"Error in query_llm: {e}")

        # Fallback bbox (center of image)
        x1_fallback = width // 4
        y1_fallback = height // 4
        x2_fallback = 3 * width // 4
        y2_fallback = 3 * height // 4

        fallback_bbox = {
            'x': x1_fallback,
            'y': y1_fallback,
            'width': x2_fallback - x1_fallback,
            'height': y2_fallback - y1_fallback,
            'x1': x1_fallback,
            'y1': y1_fallback,
            'x2': x2_fallback,
            'y2': y2_fallback,
            'target': 'fallback target',
            'action': 'NAVIGATE',
            'reasoning': 'Fallback due to parsing error',
            'progress': 'Unknown due to error'
        }

        # Calculate fallback coords
        if not self.stair:
            coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), min(int(fallback_bbox.get('y2', 0)), self.visualizer.height - 1))
        else:
            if self.stair == 'down':
                coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), int(fallback_bbox.get('y2', 0)))
            else:
                coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), int(fallback_bbox.get('y1', 0)))

        # Save RGB image with fallback bounding box
        self.visualizer._save_rgb_with_bbox(rgb_image, fallback_bbox, target_coords=coords_fallback)

        return fallback_bbox

    def double_check_stop(self, instruction, panorama_frames, visited_targets, episode_id=None,
                          step=None, history_images=None, target_reference_images=None,
                          claim_goal_id=None):
        """
        Double check if the agent should really stop based on current panorama views.
        Returns: True if should stop, False otherwise.
        """
        # return True, {}
        # Construct current views
        current_views = []
        from benchmark.pano_views import view_index_legend, view_specs as _pano_view_specs
        view_definitions = _pano_view_specs(len(panorama_frames), prefix="Current")

        for view in view_definitions:
            frame_idx = int(view["index"])
            if frame_idx < len(panorama_frames):
                rgb_image = panorama_frames[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append({"type": "text", "text": view['label']})
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})

        history_content = self._build_history_queue_content(history_images, visited_targets)
        if history_images is None:
            for i, target in enumerate(visited_targets[:-1]):
                if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})
                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})
                if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        # Construct prompt
        parts = STOP_CHECK_PROMPT.split("{current_views}")
        final_target = str(claim_goal_id or self.last_object or "").strip()
        part1 = parts[0].format(instruction=instruction, target=final_target)
        rest = parts[1]
        if len(panorama_frames) == 8:
            part1 = (
                part1.replace("current 4-directional views", "current 8-view 45° panorama")
                .replace("4-directional views", "8-view 45° panorama")
            )
        
        content = [{"type": "text", "text": part1}]
        if target_reference_images:
            content.append({
                "type": "text",
                "text": (
                    "Target reference images below specify identity only. They are NOT current "
                    "observations and cannot by themselves justify STOP."
                ),
            })
            for item in target_reference_images:
                image = item.get("image")
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype(np.uint8))
                if image is None:
                    continue
                content.append({
                    "type": "text",
                    "text": f"TARGET REFERENCE for {item.get('label') or item.get('goal_id')}",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(image)}"},
                })
        n_pano = len(panorama_frames)
        legend = view_index_legend(n_pano)
        content.append({
            "type": "text",
            "text": (
                f"Only the {n_pano} CURRENT panorama views below may establish "
                f"visibility and proximity. Their view_index values are {legend}."
            ),
        })
        content.extend(current_views)
        content.append({
            "type": "text",
            "text": rest + (
            "\nIf decision is STOP, optionally return view_index and "
                                "bbox_2d=[x1,y1,x2,y2] for the target in that CURRENT view "
                                f"({legend}; bbox 0..999). "
                                "A bbox is helpful but not required to CONFIRM STOP."
            ),
        })

        messages = [{"role": "user", "content": content}]
        
        logger.info("Double-checking stop decision...")
        
        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"la_double_check_{time.time()}")

        # Call LLM
        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.0, # Low temperature for decision
            use_la=True,  # Use gemini-2.5-pro
            log_path=log_path
        )
        
        logger.info('Double-check response:')
        logger.info(f"{output_text}")
        
        # Parse response
        import json
        import re
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        
        if json_match:
            try:
                response_data = json.loads(json_match.group())
                decision = response_data.get('decision', 'CONTINUE').upper()
                return decision == 'STOP', response_data
            except:
                logger.warning("Failed to parse double-check response JSON")
                return False, {}
        
        return False, {}

    def replan_at_backtrack(self, instruction, visited_targets_up_to_backtrack, failed_path, episode_id=None, step=None, history_images=None):
        """
        Re-plan navigation from a backtrack point.
        """
        # 1. Construct History (up to backtrack point)
        history_content = []
        failed_path_content = []

        if history_images is not None:
             # Logic for continuous history images
             # Determine backtrack step (step when the backtrack waypoint was created/visited)
             backtrack_target = visited_targets_up_to_backtrack[-1]
             backtrack_step = backtrack_target['step']
             
             # Group logic similar to navigate_or_backtrack
             sorted_waypoints = sorted(visited_targets_up_to_backtrack, key=lambda x: x['step'])
             last_leg_idx = -2
             
             for item in history_images:
                 img_step = item['step']
                 img = item['image']
                 
                 # Determine if this image is part of history or failed path
                 if img_step <= backtrack_step:
                     target_list = history_content
                     
                     # Add segment header for history
                     leg_idx = -1
                     for i in range(len(sorted_waypoints) - 1):
                         if sorted_waypoints[i]['step'] <= img_step < sorted_waypoints[i+1]['step']:
                             leg_idx = i
                             break
                     if leg_idx == -1 and len(sorted_waypoints) > 0:
                         if img_step >= sorted_waypoints[-1]['step']:
                             leg_idx = len(sorted_waypoints) - 1
                             
                     if leg_idx != last_leg_idx:
                         if leg_idx >= 0:
                             start_id = leg_idx
                             if leg_idx < len(sorted_waypoints) - 1:
                                 end_id = leg_idx + 1
                                 label = f"Waypoint {start_id} -> Waypoint {end_id}: "
                             else:
                                 # For history up to backtrack, the last segment ends at the backtrack waypoint
                                 label = f"Waypoint {start_id} -> Waypoint {start_id} (Backtrack Point): "
                             target_list.append({"type": "text", "text": label})
                         last_leg_idx = leg_idx
                 else:
                     target_list = failed_path_content
                     
                     if not failed_path_content: # First item
                         target_list.append({"type": "text", "text": "Trajectory after Backtrack Point (Failed Path):"})

                 img_data = {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(img)}"
                 }}
                 target_list.append(img_data)
        else:
            for i, target in enumerate(visited_targets_up_to_backtrack):
                 if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})
                 if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})
                 if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

            # 2. Construct Failed Path Description (Images + Text)
            for i, target in enumerate(failed_path):
                idx = len(visited_targets_up_to_backtrack) + i
                if 'init_image' in target:
                    failed_path_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    failed_path_content.append({"type": "text", "text": f"Waypoint {idx}: Arrival view"})

                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    failed_path_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    failed_path_content.append({"type": "text", "text": f"Waypoint {idx}: After turn {turn_direction} view"})
                if 'description' in target:
                    failed_path_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        if not failed_path_content and not history_images:
             failed_path_content.append({"type": "text", "text": "None (Immediate failure)"})

        # 3. Construct Backtrack Point Views (Current Views) and Action List
        backtrack_point = visited_targets_up_to_backtrack[-1]
        panorama_frames = backtrack_point['panorama_frames']
        
        current_views = []
        from benchmark.pano_views import view_specs as _pano_view_specs
        num_frames = len(panorama_frames)
        view_definitions = _pano_view_specs(num_frames, prefix="Current")
        
        # Calculate blocked directions for backtrack point
        blocked_directions = self._check_blocked_directions(panorama_frames)

        action_list_items = []
        action_desc_parts = []
        if num_frames == 8:
            from benchmark.pano_views import navigate_action_choices

            action_list_items, action_desc_str = navigate_action_choices(
                num_frames, self._blocked_frame_indices(panorama_frames)
            )
        else:
            # Generate Action List
            candidate_directions = ['forward', 'left', 'right', 'behind']
            valid_directions = [d for d in candidate_directions if d not in blocked_directions]

            # Fallback if all blocked
            if not valid_directions:
                if 'behind' in candidate_directions: valid_directions.append('behind')
                elif 'forward' in candidate_directions: valid_directions.append('forward')

            for direction in valid_directions:
                if direction == 'forward':
                    action_list_items.append("   - navigate to forward - continue straight ahead")
                elif direction == 'left':
                    action_list_items.append("   - navigate to left - turn left and go forward")
                elif direction == 'right':
                    action_list_items.append("   - navigate to right - turn right and go forward")
                elif direction == 'behind':
                    action_list_items.append("   - navigate to behind - turn around and go forward")

                action_desc_parts.append("navigate to " + direction)

            action_desc_str = "|".join(action_desc_parts)
        action_list_str = "\n".join(action_list_items)

        # Build current views for prompt
        for view in view_definitions:
            frame_idx = int(view["index"])
            if frame_idx < len(panorama_frames):
                rgb_image = panorama_frames[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append({"type": "text", "text": view['label']})
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})

        # 4. Construct Prompt
        backtrack_point = visited_targets_up_to_backtrack[-1]
        previous_action = backtrack_point.get('direction_decision', 'unknown')

        # Generate negative constraints based on target object
        target_objects = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
        target = None
        instruction_lower = instruction.lower()
        for obj in target_objects:
            obj_name = obj.replace('_', ' ')
            if obj_name in instruction_lower or obj in instruction_lower:
                target = obj
                break
        
        negative_constraints = ""
        if self.use_negative_constraints:
            if target:
                other_objects = [obj.replace('_', ' ') for obj in target_objects if obj != target]
                negative_constraints = f"- When looking for {target.replace('_', ' ')}, do not confuse it with other objects: {', '.join(other_objects)}."

        prompt = self.P.LA_PROMPT_BACKTRACK_REPLAN.format(
            instruction=instruction,
            action_list=action_list_str,
            action_desc=action_desc_str,
            previous_action=previous_action,
            negative_constraints=negative_constraints
        )

        content = [{"type": "text", "text": "Navigation History:"}]
        content.extend(history_content)
        content.append({"type": "text", "text": "\nPrevious Trajectory (Path taken from here):"})
        content.extend(failed_path_content)
        content.append({"type": "text", "text": "\nCurrent 4-directional views at Backtrack Waypoint:"})
        content.extend(current_views)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        logger.info("Replanning at backtrack point...")
        
        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
             import time
             log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"la_replan_{time.time()}")

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=2048,
            temperature=0.0,
            use_la=True,
            log_path=log_path,
        )
        
        logger.info(f"Replan Output: {output_text}")
        
        # Parse JSON
        import json
        import re
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                response = json.loads(json_match.group())
                self.stair = response.get('stair', False) # Update stair status
                
                # Handle stop signal
                stop_signal = response.get('stop', False)
                action = response.get('action', 'navigate to forward').lower()
                
                if action == 'stop' or 'stop' in action.split():
                    stop_signal = True
                    if action == 'stop':
                        action = 'navigate to forward'
                
                response['stop_signal'] = stop_signal
                response['action'] = action
                
                from benchmark.pano_views import nearest_cardinal_name, parse_navigate_action
                frame_idx = parse_navigate_action(action, num_frames)
                if frame_idx is not None:
                    return nearest_cardinal_name(frame_idx, num_frames), response
                if 'left' in action: return 'left', response
                if 'right' in action: return 'right', response
                if 'behind' in action: return 'behind', response
                return 'forward', response
            except:
                pass
        
        return 'forward', {}

    def reset(self):
        # self.model.reset_stats()
        self.stair = False
        self.todo_list = None
        self.last_object = ''
        self.todo_verification_feedback = ""

    def query_llm_oracle(self, panorama_frames, instruction):
        """EQA Oracle: answer the EQA question from 4 panorama views at the destination.

        Only invoked when task_type == 'EQA'.
        """
        content = [{
            'type': 'text',
            'text': (
                'You are a robot agent. You have reached your destination.\n'
                'You have access to:\n'
                'A set of 4 Panorama Views at your current location (Destination).\n\n'
                f'Task: Answer the question: "{instruction}"'
            )
        }]

        try:
            content.append({"type": "text", "text": "--- DESTINATION PANORAMA (Current Surroundings) ---"})
            view_definitions = [
                {'angle': 0, 'label': 'View 1: Forward'}, {'angle': 90, 'label': 'View 2: Left'},
                {'angle': 180, 'label': 'View 3: Behind'}, {'angle': 270, 'label': 'View 4: Right'}
            ]
            num_frames = len(panorama_frames)
            for view in view_definitions:
                frame_idx = int((view['angle'] / 360.0) * num_frames) % num_frames if num_frames > 0 else 0
                if frame_idx < len(panorama_frames):
                    raw = panorama_frames[frame_idx]['rgb']
                    img = Image.fromarray(raw) if isinstance(raw, np.ndarray) else raw
                    content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    content.append({"type": "text", "text": view['label']})

            prompt = f"""
Task: You are a visual scene analyzer. Your ONLY job is to select the single best answer from the provided lists based ONLY on the CURRENT location views.
Question: "{instruction}"

**Allowed Vocabulary:**
- **COLORS:** ["white", "off-white", "black", "grey", "blue", "red", "green", "brown", "yellow", "slate grey", "tan", "silver", "red brown", "olive green"]
- **ROOMS:** ["kitchen", "bedroom", "spa", "lounge", "family room", "living room", "hallway", "laundry room", "gym", "closet"]

**EXTREME RULES (MUST FOLLOW):**
1. **FORCE A GUESS:** Even if the target is invisible, occluded, or ambiguous, you **MUST HALLUCINATE or GUESS** based on the scene context.
2. **STRICT SELECTION:** Your output "answer" MUST be **EXACTLY** one of the words from the "Allowed Vocabulary" lists above.

Response format (JSON):
{{
    "reasoning": "Briefly explain...",
    "answer": "<THE SELECTED WORD>"
}}
"""
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=1024,
                temperature=0.0,
                use_la=True,
            )
            logger.info(f"Oracle QA Output: {output_text}")

            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group()).get('answer', 'null')

        except Exception as e:
            logger.error(f"Error in oracle QA: {e}")

        return "null"


@baseline_registry.register_trainer(name="ZS-Evaluator-mp")
class ZeroShotVlnEvaluatorMP(BaseTrainer):
    def __init__(self, config, r2r, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self.backtrack_steps = 0  # consecutive backtrack steps
        self.r2r = r2r
        self.device = get_device(config.TORCH_GPU_ID)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.resolution = config.MAP.MAP_RESOLUTION
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)

        self.trans = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH),
                                             interpolation=Image.NEAREST)
                                         ])

        self.classes = []
        self.current_episode_id = None
        self.current_detections = None
        self.map_channels = map_channels
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)


        self.visualize = True
        self.save_dir = getattr(config, 'RGB_SAVE_DIR', './saved_rgb_images')
        # When DEBUG_LOGGING is on, also forward the debug_log_dir to the visualizer
        # so per-step topdown_map.png + rgb_obs.png land alongside the LA/VA prompts.
        _debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        _debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs') if _debug_logging else None
        self.visualizer = LaViRAVisualizer(None, self.visualize, self.save_dir, self.width, self.height,
                                           debug_log_dir=_debug_log_dir)

        self.visited_targets = []  # List of targets the agent has identified/visited
        self.history_images = [] # Initialize history images
        self.current_step = 0  # Track current step for navigation decisions

        # Distance thresholds for target management (in map units)
        # 0.75 meter threshold (75 cm / resolution)
        self.target_reached_threshold = getattr(config, 'TARGET_REACHED_THRESHOLD', 75.0 / self.resolution)
        # self.agent = VLMReasoningAgent(self.visualizer)
        
        # Statistics counters
        self.total_backtracks = 0
        self.total_waypoints = 0

        self.use_continuous_history = getattr(config.EVAL, 'USE_CONTINUOUS_HISTORY', False)
        # self.history_interval = getattr(config.EVAL, 'HISTORY_INTERVAL', 5)
        self.history_interval = 3
        
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.config.freeze()

        # Initialize agent with specified configuration
        # use_guideline=False, use_working_memory=False as requested
        # self.agent = VLMReasoningAgent(self.visualizer, use_guideline=False, use_working_memory=False)
        debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs')
        self.task_type = getattr(config, 'TASK_TYPE', 'VLN')
        self.use_todo_list = getattr(config.EVAL, 'USE_TODO_LIST', True)
        self.backtrack_second_chance = getattr(config.EVAL, 'BACKTRACK_SECOND_CHANCE', True)
        self.no_todo_prompt = getattr(config.EVAL, 'NO_TODO_PROMPT', False)
        self.agent = VLMReasoningAgent(
            self.visualizer,
            task_type=self.task_type,
            use_guideline=True,
            use_working_memory=False,
            allow_move_behind=False,
            debug_logging=debug_logging,
            log_dir=debug_log_dir,
            use_todo_list=self.use_todo_list,
            backtrack_second_chance=self.backtrack_second_chance,
            no_todo_prompt=self.no_todo_prompt,
            la_model_override=getattr(config.EVAL, 'LA_MODEL_NAME', ''),
            va_model_override=getattr(config.EVAL, 'VA_MODEL_NAME', ''),
            la_base_url_override=getattr(config.EVAL, 'LA_BASE_URL', ''),
            la_api_key_override=getattr(config.EVAL, 'LA_API_KEY', ''),
            va_base_url_override=getattr(config.EVAL, 'VA_BASE_URL', ''),
            va_api_key_override=getattr(config.EVAL, 'VA_API_KEY', ''),
            la_enable_thinking=getattr(config.EVAL, 'LA_ENABLE_THINKING', False),
            la_thinking_budget=getattr(config.EVAL, 'LA_THINKING_BUDGET', 0),
        )
        if self.task_type == "EQA" and EQADataCollector is not None:
            self.agent.collector = EQADataCollector(save_dir="finetune_data")

        # Initialize IPlanner Agent
        self.use_iplanner = getattr(config, 'USE_IPLANNER', False)  # Default to False if not specified
        # logger.info(f"DEBUG: USE_IPLANNER is set to {self.use_iplanner}")
        if self.use_iplanner:
            intrinsics = self._get_camera_intrinsics()
            intrinsics_tensor = torch.from_numpy(intrinsics).float().to(self.device)
            
            # Construct dynamic paths for iplanner model and config
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "iplanner", "plannernet.pt")
            config_path = os.path.join(project_root, "iplanner", "configs", "iplanner.yaml")
            
            logger.info(f"Loading iplanner from: {model_path} with config: {config_path}")
            
            self.iplanner_agent = IPlannerAgent(
                image_intrinsic=intrinsics_tensor,
                model_path=model_path,
                model_config_path=config_path,
                device=self.device
            )
            self.iplanner_traj_global = None
            self.iplanner_update_freq = 5
            self.last_iplanner_step = -999

            # Initialize TrajViz
            self.traj_viz = TrajViz(root_path=self.save_dir, map_name=None, intrinsic_matrix=intrinsics, img_width=self.width, img_height=self.height)

        # Initialize NavDP Agent
        self.use_navdp = getattr(config, 'USE_NAVDP', False)
        # logger.info(f"DEBUG: USE_NAVDP is set to {self.use_navdp}")
        if self.use_navdp:
            if NavDP_Agent is None:
                raise ImportError(
                    "USE_NAVDP=True but NavDP_Agent failed to import. "
                    "Likely missing `diffusers` (see navdp/requirements.txt). "
                    "Run: pip install diffusers==0.33.1"
                )
            intrinsics = self._get_camera_intrinsics()
            model_path = os.path.join(NAVDP_ROOT, "navdp-weights.ckpt")
            # logger.info(f"Loading NavDP from: {model_path}")

            self.navdp_agent = NavDP_Agent(
                image_intrinsic=intrinsics,
                navi_model=model_path,
                device=self.device
            )
            self.navdp_agent.reset(batch_size=1, threshold=0.1)

        self.use_fmm = getattr(config, 'USE_FMM', True)
        
        # Validation Logic
        if self.use_navdp and self.use_iplanner:
             raise ValueError("Configuration Error: use_navdp and use_iplanner cannot both be True.")
        
        if not self.use_fmm and not self.use_navdp and not self.use_iplanner:
            raise ValueError("Configuration Error: use_fmm, use_navdp, and use_iplanner cannot all be False.")

    def _init_envs(self) -> None:
        # logger.info("start to initialize environments")

        # Check if we have specific allowed episodes loaded from assignment file
        specific_episodes = getattr(self, 'allowed_episodes', None)
        if specific_episodes:
            # self.allowed_episodes is a set of tuples (id, scene_id)
            specific_episodes = list(specific_episodes)

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
            specific_episodes_allowed=specific_episodes,
        )
        logger.info(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()
        # logger.info("initializing environments finished!")

    def _collect_val_traj(self) -> None:
        if not self.r2r:
            role = self.config.TASK_CONFIG.DATASET.ROLES
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        
        gt_path = None
        if self.r2r:
            gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)
        else:
            gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split, role=role[0])
            
        if os.path.exists(gt_path):
            with gzip.open(gt_path) as f:
                self.gt_data = json.load(f)
        else:
            self.gt_data = None
            logger.warning(f"GT data not found at {gt_path}, skipping NDTW calculation.")

    def _calculate_metric(self, infos: List, submitted_answer: str = None, is_timeout: bool = False):
        """Dispatch metric calculation by task_type."""
        if self.task_type == "EQA":
            return self._save_eqa_metric(infos, submitted_answer=submitted_answer, is_timeout=is_timeout)
        return self._calculate_trajectory_metric(infos)

    def _save_eqa_metric(self, infos: List[Dict], submitted_answer: str = None, is_timeout: bool = False):
        if not infos:
            return
        info = infos[0]
        curr_eps = self.envs.current_episodes()
        ep_id = curr_eps[0].episode_id
        if ep_id in getattr(self, 'state_eps', {}):
            return

        question_text = curr_eps[0].question.question_text
        ground_truth_answer = curr_eps[0].question.answer_text
        predicted_answer = submitted_answer

        try:
            trajectory_list = info.get('position', {}).get('position', [])
            trajectory_json_string = json.dumps(trajectory_list)
        except Exception as e:
            logger.error(f"Failed to serialize trajectory: {e}")
            trajectory_json_string = "[]"

        metric = {
            'success': 0.0,
            'steps_taken': info.get('steps_taken', 0),
            'episode_id': ep_id,
            'question': question_text,
            'ground_truth': ground_truth_answer,
            'predicted_answer': predicted_answer if predicted_answer is not None else "TIMEOUT (No Answer)",
            'trajectory_json': trajectory_json_string,
        }

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.EVAL_CKPT_PATH_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )

        if is_timeout:
            metric['steps_taken'] = getattr(self, 'max_step', info.get('steps_taken', 0))
            logger.info(f"--- Episode {ep_id} TIMEOUT. Success: 0.0 ---")
            self.state_eps[ep_id] = metric
            try:
                with open(fname, "w") as f:
                    json.dump(self.state_eps, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save metrics to {fname}: {e}")
            return

        try:
            clean_gt = str(ground_truth_answer).strip()
            clean_pred = str(predicted_answer).strip()
            metric['success'] = 1.0 if clean_pred == clean_gt else 0.0
            logger.info(f"--- EQA JUDGMENT (Episode {ep_id}) ---")
            logger.info(f"  Ground Truth    : {ground_truth_answer!r}")
            logger.info(f"  Submitted Answer: {predicted_answer!r}")
            logger.info(f"  SUCCESS : {metric['success']}")
        except Exception as e:
            logger.error(f"EQA judgement failure: {e}")
            metric['success'] = 0.0

        self.state_eps[ep_id] = metric
        try:
            with open(fname, "w") as f:
                json.dump(self.state_eps, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics to {fname}: {e}")

    def _calculate_trajectory_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id

        metric = {}
        metric['steps_taken'] = info.get('steps_taken', 0)
        
        # Get success distance from config
        success_distance = getattr(self.config.TASK_CONFIG.TASK, 'SUCCESS_DISTANCE', 
                                 getattr(self.config.TASK_CONFIG.TASK.SUCCESS, 'SUCCESS_DISTANCE', 3.0))

        if self.gt_data is not None and str(ep_id) in self.gt_data:
            gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
            pred_path = np.array(info['position']['position'])
            distances = np.array(info['position']['distance'])
            gt_length = distances[0]
            dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
            
            metric['distance_to_goal'] = distances[-1]
            metric['success'] = 1. if distances[-1] <= success_distance else 0.
            metric['oracle_success'] = 1. if (distances <= success_distance).any() else 0.
            metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
            metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
            metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
            metric['sdtw'] = metric['ndtw'] * metric['success']
        else:
            # Use environment provided metrics if available
            metric['distance_to_goal'] = info.get('distance_to_goal', 0.0)
            metric['success'] = info.get('success', 0.0)
            metric['oracle_success'] = info.get('oracle_success', 0.0)
            metric['path_length'] = info.get('path_length', 0.0)
            metric['spl'] = info.get('spl', 0.0)
            metric['ndtw'] = info.get('ndtw', 0.0)
            metric['sdtw'] = info.get('sdtw', 0.0)
            
        self.state_eps[ep_id] = metric
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        logger.info(f'ep{ep_id}:{self.state_eps[ep_id]}')

        # Visualize trajectory
        self.draw_agent_pos_and_ref_path(success=bool(metric['success']))

        # Rename the episode directory with success status
        try:
            is_success = bool(metric['success'])
            status_suffix = "_success" if is_success else "_failed"
            
            # The current save directory for this episode
            current_dir = os.path.join(self.save_dir, str(ep_id))
            
            if os.path.exists(current_dir):
                # The new directory name
                new_dir = os.path.join(self.save_dir, f"{ep_id}{status_suffix}")
                
                if os.path.exists(new_dir):
                    import shutil
                    shutil.rmtree(new_dir)
                
                os.rename(current_dir, new_dir)
                logger.info(f"Renamed episode directory to: {new_dir}")
        except Exception as e:
            logger.info(f"Error renaming episode directory: {e}")

    def _initialize_policy(self) -> None:
        # logger.info("start to initialize policy")
        self.segment_module = GroundedSAM(self.config, self.device)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        self.visualizer.update_map(self.mapping_module)

        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()

    def _concat_obs(self, obs: Observations) -> np.ndarray:
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)  # (h, w, c)->(c, h, w)

        return state

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)  # [3, h, w]
        rgb = rgb[:, :, ::-1]  # RGB to BGR
        depth = state[:, :, 3:4]  # [1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH

        sem_seg_pred = self._get_sem_pred(rgb)  # [num_detected_classes, h, w]
        depth = self._preprocess_depth(depth, min_depth, max_depth)  # [1, h, w]

        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        ds = env_frame_width // self.map_args.FRAME_WIDTH  # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8)))  # resize
            depth = depth[ds // 2::ds, ds // 2::ds]  # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)  # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)  # (4+num_detected_classes, h, w)

        return state

    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        cls2 = self.classes.copy()
        if self.agent.stair:
            cls2 = ["stairs"]
        elif self.current_step <= 12:
            cls2.append('stairs')
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=cls2)
        if self.visualize:
            pass
            # save_path = os.path.join(self.save_dir, str(self.current_episode_id), f'step{self.current_step}_mask.png')
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cv2.imwrite(save_path, annotated_images)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        # logger.info("current step detected classes (before filtering): ", labels)

        class_names = self._process_labels(labels)
        masks = self._process_masks(masks, class_names)

        return masks.transpose(1, 2, 0)

    def _process_labels(self, labels: List[str]) -> List:
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)

        return class_names

    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        if masks.shape[0] > 0:  # Check if there are any masks
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx)  # dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)

            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]

            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))

        return final_masks

    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        # logger.info('max:',depth.max())
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99  # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 1.0  # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0

        return depth

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)

        return state  # state.shape=(c,h,w)

    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                                 [(0, max_channels - state.shape[0]),
                                  (0, 0),
                                  (0, 0)],
                                 mode='constant')
                          for state in n_states], axis=0)

        return torch.from_numpy(batch).to(self.device)

    def _random_policy(self):
        action = np.random.choice([
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ])

        return {"action": action}

    def _process_classes(self, base_class: List, target_class: List) -> List:
        for item in target_class:
            if item in base_class:
                base_class.remove(item)
        base_class.extend(target_class)

        return base_class


    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        # logger.info(f'{navigable_index}, {not_navigable_index}')
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)

        obstacles = one_step_full_map[0, ...].astype(bool)
        explored_area = one_step_full_map[1, ...].astype(bool)

        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=disk(kernel_size))

        return floor

    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int = 3) -> tuple:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)

        obstacles = full_map[0, ...].astype(bool)
        explored_area = full_map[1, ...].astype(bool)

        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)

        selem = disk(3)
        obstacles_closed = binary_closing(obstacles, footprint=selem)
        objects_closed = binary_closing(objects, footprint=selem)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=selem)

        untraversable = np.logical_or(objects_closed, obstacles_closed)
        # ensure stairs override untraversable
        untraversable[navigable_closed == 1] = 0
        # untraversable[stairs_mask == 1] = 0
        untraversable = remove_small_objects(untraversable, min_size=64)
        untraversable = binary_closing(untraversable, footprint=disk(3))
        traversable = np.logical_not(untraversable)

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=selem)
        traversable = np.logical_or(floor, traversable)

        explored_area = binary_closing(explored_area, footprint=selem)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversable, floor, frontiers.astype(np.uint8)

    def _maps_initialization(self):
        obs = self.envs.reset()  # type(obs): list
        
        # Handle instruction extraction for different tasks (VLN / ObjectNav / EQA)
        current_episode = self.envs.current_episodes()[0]
        if self.task_type == "EQA" and hasattr(current_episode, 'question'):
            self.instruction = current_episode.question.question_text
        elif 'instruction' in obs[0]:
            self.instruction = obs[0]['instruction']['text']
        elif hasattr(current_episode, 'object_category'):
            self.instruction = f"Find the {current_episode.object_category}"
        elif 'objectgoal' in obs[0]:
            self.instruction = f"Find the object with ID {obs[0]['objectgoal'][0]}"
        else:
            self.instruction = "Explore the environment"
            
        self.destination = "goal"
        self.classes = self.base_classes.copy()
        self.current_episode_id = self.envs.current_episodes()[0].episode_id

        self.visualizer._save_rgb_frame(obs[0], 0, None, self.current_episode_id)

        # Initialize trajectory tracking
        self.pos_list = []
        # Get reference path if available
        self.ref_path = []
        if self.gt_data is not None and str(self.current_episode_id) in self.gt_data:
            self.ref_path = self.gt_data[str(self.current_episode_id)]['locations']

        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses, self.current_step)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)

        return obs, full_pose

    def _look_around(self):
        full_pose, obs, dones, infos = None, None, None, None
        for step in range(0, 12):
            self._action = HabitatSimActions.TURN_LEFT
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                actions.append({"action": HabitatSimActions.TURN_LEFT})
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            self.current_step = step
            if dones[0]:
                return full_pose, obs, dones, infos

            # Save RGB frame during look around phase
            self.visualizer._save_rgb_frame(obs[0], step, self.visited_targets, self.current_episode_id)

            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses, self.current_step)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

        return full_pose, obs, dones, infos


    def reset(self) -> None:
        self.classes = []
        self.current_detections = None
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)

        # Reset target tracking
        self.visited_targets = []
        self.history_images = []
        self.current_step = 0
        self.backtrack_steps = 0
        self.last_submitted_answer = None  # EQA: set when agent submits final answer
        self._eqa_done_requested = False  # EQA: set True when Oracle-QA fires (STOP alone doesn't end EQA)

        self.policy.reset()
        self.mapping_module.reset()
        self.agent.reset()
        # self.agent.model.reset_stats() # Do not reset stats between episodes!

    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix for depth projection"""
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        width = self.width
        height = self.height
        vfov = 2 * np.arctan(height / width * np.tan(hfov / 2))

        fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
        fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
        cx = width / 2
        cy = height / 2

        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
        return intrinsics

    def get_panorama(self, obs: Observations, step: int):
        """
        Collect a panorama
        by turning a full circle (12 turns).
        """

        panorama_frames = []

        for turn_step in range(1, 12 + 1):
            # Turn
            turn_action = [{"action": HabitatSimActions.TURN_LEFT}]  # turn left 30 deg
            turn_outputs = self.envs.step(turn_action)
            turn_obs, _, turn_dones, turn_infos = [list(x) for x in zip(*turn_outputs)]

            if turn_dones[0]:
                # logger.info("Episode ended during panorama collection")
                # Return signal that episode is done - caller should handle this
                return {'turn_direction': 'episode_done', 'episode_finished': True}

            # Collect RGB and depth for the current view
            panorama_frames.append({
                'rgb': turn_obs[0]['rgb'].copy(),
                'depth': turn_obs[0]['depth'].copy(),
                'angle': turn_step * 30 % 360,  # 30 deg per turn
                'step': turn_step
            })

            # Update the map (not saved; keeps state consistent)
            batch_obs = self._batch_obs(turn_obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in turn_obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses, self.current_step)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step + turn_step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
        panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]
        # logger.info([x['angle'] for x in panorama_frames])
        # logger.info(f"Collected {len(panorama_frames)} panorama frames")

        return panorama_frames[::3]

    def draw_agent_pos_and_ref_path(self, success: bool = False):
        """
        Visualize agent path, reference path, and target object/viewpoints using Matplotlib.
        """
        if not getattr(self, "visualize", False):
            return None

        pos_list = getattr(self, "pos_list", None) or []
        ref_path = getattr(self, "ref_path", None) or []
        
        # Get target object and viewpoints
        current_episode = self.envs.current_episodes()[0]
        goals = getattr(current_episode, 'goals', [])
        
        # Extract data for plotting
        # Habitat coordinates: Y is up. We visualize X-Z plane (Top-down).
        agent_x = [p[0] for p in pos_list]
        agent_z = [p[2] for p in pos_list]
        
        ref_x = [p[0] for p in ref_path]
        ref_z = [p[2] for p in ref_path]
        
        object_centers = []
        closest_viewpoints = []
        all_viewpoints = []
        radius_list = []
        
        for goal in goals:
            if hasattr(goal, 'position'):
                obj_pos = goal.position
                object_centers.append(obj_pos)
                # Check for radius
                if hasattr(goal, 'radius') and goal.radius is not None:
                     radius_list.append((obj_pos, goal.radius))
            
                # Find closest viewpoint for this goal
                if hasattr(goal, 'view_points') and len(goal.view_points) > 0:
                    best_vp = None
                    min_dist = float('inf')
                    
                    for vp in goal.view_points:
                        if hasattr(vp, 'agent_state') and hasattr(vp.agent_state, 'position'):
                            vp_pos = vp.agent_state.position
                            all_viewpoints.append(vp_pos)
                            # Calculate Euclidean distance in 3D
                            dist = np.linalg.norm(np.array(vp_pos) - np.array(obj_pos))
                            if dist < min_dist:
                                min_dist = dist
                                best_vp = vp_pos
                    
                    if best_vp is not None:
                        closest_viewpoints.append(best_vp)

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Plot Object Centers
        for obj in object_centers:
            plt.scatter(obj[0], obj[2], c='red', s=200, marker='*', label='Object Center')
            
        # Plot Radius zones
        for center, rad in radius_list:
             circle = plt.Circle((center[0], center[2]), rad, color='green', fill=True, alpha=0.2)
             plt.gca().add_patch(circle)

        # Plot ALL ViewPoints (but circle only closest)
        if all_viewpoints:
            vp_x_all = [vp[0] for vp in all_viewpoints]
            vp_z_all = [vp[2] for vp in all_viewpoints]
            plt.scatter(vp_x_all, vp_z_all, c='green', s=30, alpha=0.6, label='ViewPoints')

        # Draw Success Zones (1.0m) for closest viewpoints only
        if closest_viewpoints:
            vp_x_closest = [vp[0] for vp in closest_viewpoints]
            vp_z_closest = [vp[2] for vp in closest_viewpoints]
            
            for vx, vz in zip(vp_x_closest, vp_z_closest):
                circle = plt.Circle((vx, vz), 1.0, color='green', fill=False, linestyle='--', alpha=0.5)
                plt.gca().add_patch(circle)
        
        # Plot Reference Path
        if ref_x:
            plt.plot(ref_x, ref_z, c='green', linewidth=2, label='Reference Path', linestyle='-')
            # Start and End of Reference Path
            plt.scatter(ref_x[0], ref_z[0], c='orange', s=100, marker='o', label='Ref Start')
            plt.scatter(ref_x[-1], ref_z[-1], c='blue', s=100, marker='X', label='Ref Goal')

        # Plot Agent Trajectory
        if agent_x:
            plt.plot(agent_x, agent_z, c='red', linewidth=2, label='Agent Path', linestyle='-')
            if agent_x:
                plt.scatter(agent_x[0], agent_z[0], c='blue', s=100, marker='^', label='Agent Start')
                plt.scatter(agent_x[-1], agent_z[-1], c='red', s=80, marker='v', label='Agent End')

        plt.title(f'Episode {getattr(self, "current_episode_id", "unknown")} Trajectory')
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters) - Top Down View')
        
        # Avoid duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        episode_id = str(getattr(self, "current_episode_id", "unknown"))
        # out_dir = os.path.join(self.save_dir, episode_id)
        # Modified to save to traj_viz directory
        out_dir = os.path.join(self.save_dir, "traj_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"traj_viz_success_{success}_{episode_id}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved trajectory visualization to: {out_path}")

        return out_path

    def draw_agent_pos_and_ref_path_OLD(self, success: bool = False):
        """
        Visualize agent path, reference path, and target object/viewpoints.
        """
        if not getattr(self, "visualize", False):
            return None

        pos_list = getattr(self, "pos_list", None) or []
        ref_path = getattr(self, "ref_path", None) or []
        
        # Get target object and viewpoints
        current_episode = self.envs.current_episodes()[0]
        goals = getattr(current_episode, 'goals', [])
        
        # Collect all points to determine canvas bounds
        all_points = []
        if len(pos_list) > 0: all_points.extend(pos_list)
        if len(ref_path) > 0: all_points.extend(ref_path)
        
        object_centers = []
        viewpoints = []
        
        for goal in goals:
            if hasattr(goal, 'position'):
                object_centers.append(goal.position)
                all_points.append(goal.position)
            if hasattr(goal, 'view_points'):
                for vp in goal.view_points:
                    if hasattr(vp, 'agent_state') and hasattr(vp.agent_state, 'position'):
                        viewpoints.append(vp.agent_state.position)
                        all_points.append(vp.agent_state.position)
        
        if not all_points:
            return None

        # Colors (BGR)
        COLOR_AGENT = (0, 0, 255)  # red
        COLOR_REF = (0, 255, 0)    # green
        COLOR_START = (255, 128, 0)
        COLOR_END = (0, 128, 255)
        COLOR_OBJ = (0, 0, 255) # Red for object (star)
        COLOR_VP = (0, 255, 0) # Green for viewpoints

        def _to_array(points: List) -> np.ndarray:
            arr = np.asarray(points, dtype=np.float32)
            if arr.size == 0:
                return np.zeros((0, 0), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        # Decide axis once so agent/ref/waypoints stay aligned.
        all_arr = _to_array(all_points)
        axis = [0, 1]
        if all_arr.shape[1] >= 3:
            std_y = float(np.std(all_arr[:, 1]))
            std_z = float(np.std(all_arr[:, 2]))
            if std_z > std_y * 2.0 or std_z > 1e-6: # Prefer X-Z plane for Habitat
                axis = [0, 2]

        def _points_to_xy(points: List) -> np.ndarray:
            arr = _to_array(points)
            if arr.size == 0 or arr.shape[1] < 2:
                return np.zeros((0, 2), dtype=np.float32)
            if axis == [0, 2] and arr.shape[1] >= 3:
                return arr[:, [0, 2]]
            return arr[:, :2]

        agent_xy = _points_to_xy(pos_list)
        ref_xy = _points_to_xy(ref_path)
        obj_xy = _points_to_xy(object_centers)
        vp_xy = _points_to_xy(viewpoints)

        # Recalculate all_xy based on 2D projection
        all_xy_list = [x for x in [agent_xy, ref_xy, obj_xy, vp_xy] if x.shape[0] > 0]
        if not all_xy_list:
             return None
             
        all_xy = np.concatenate(all_xy_list, axis=0)
        min_xy = np.min(all_xy, axis=0)
        max_xy = np.max(all_xy, axis=0)

        canvas = 1024
        margin = 60
        span = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 1e-6)
        scale = (canvas - 2 * margin) / span

        def _project(xy: np.ndarray) -> np.ndarray:
            if xy.shape[0] == 0:
                return np.zeros((0, 2), dtype=np.int32)
            xs = (xy[:, 0] - min_xy[0]) * scale + margin
            ys = (max_xy[1] - xy[:, 1]) * scale + margin  # y up -> flip for image coordinates
            pts = np.stack([xs, ys], axis=1)
            pts = np.clip(pts, 0, canvas - 1)
            return pts.astype(np.int32)

        agent_pts = _project(agent_xy)
        ref_pts = _project(ref_xy)
        obj_pts = _project(obj_xy)
        vp_pts = _project(vp_xy)

        img = np.ones((canvas, canvas, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (0, 0), (canvas - 1, canvas - 1), (220, 220, 220), 2)

        # Draw ViewPoints (Navigable)
        if vp_pts.shape[0] > 0:
            for pt in vp_pts:
                 cv2.circle(img, tuple(pt), 3, COLOR_VP, -1)
            # Draw success zones? 1.0m radius.
            radius_px = int(1.0 * scale)
            if radius_px > 0:
                 for pt in vp_pts:
                      cv2.circle(img, tuple(pt), radius_px, COLOR_VP, 1)

        # Draw Object Centers
        if obj_pts.shape[0] > 0:
            for pt in obj_pts:
                # Diamond
                cv2.drawMarker(img, tuple(pt), COLOR_OBJ, markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                cv2.putText(img, "Target", (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OBJ, 1)

        if ref_pts.shape[0] >= 2:
            cv2.polylines(img, [ref_pts.reshape(-1, 1, 2)], False, COLOR_REF, 2)
        if agent_pts.shape[0] >= 2:
            cv2.polylines(img, [agent_pts.reshape(-1, 1, 2)], False, COLOR_AGENT, 2)

        # Start/End markers
        if ref_pts.shape[0] > 0:
            cv2.circle(img, tuple(ref_pts[0]), 5, COLOR_START, -1)
            cv2.circle(img, tuple(ref_pts[-1]), 5, COLOR_END, -1)
        if agent_pts.shape[0] > 0:
            cv2.circle(img, tuple(agent_pts[0]), 5, COLOR_START, -1)
            cv2.circle(img, tuple(agent_pts[-1]), 7, COLOR_END, -1)

        cv2.putText(img, 'Ref path', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_REF, 2, cv2.LINE_AA)
        cv2.putText(img, 'Agent traj', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_AGENT, 2, cv2.LINE_AA)
        cv2.putText(img, 'Targets & Viewpoints', (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_OBJ, 2, cv2.LINE_AA)

        episode_id = str(getattr(self, "current_episode_id", "unknown"))
        out_dir = os.path.join(self.save_dir, episode_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"traj_viz_success_{success}.png")
        cv2.imwrite(out_path, img)
        logger.info(f"Saved trajectory visualization to: {out_path}")

        return img

    def rollout(self):
        """
        Execute a whole episode using bounding box target navigation
        """
        if self.use_navdp:
            self.navdp_agent.reset_env(0)

        obs, full_pose = self._maps_initialization()

        # Check allowed episodes
        if getattr(self, 'allowed_episodes', None) is not None:
             current_ep = self.envs.current_episodes()[0]
             
             # Robust matching for (id, scene_id) to handle path differences
             is_allowed = False
             for allowed_id, allowed_scene_id in self.allowed_episodes:
                 if str(current_ep.episode_id) == allowed_id:
                     # Check Scene match (basename or suffix)
                     # allowed_scene_id is usually relative, current_ep.scene_id is absolute
                     if current_ep.scene_id.endswith(allowed_scene_id) or \
                        os.path.basename(current_ep.scene_id) == os.path.basename(allowed_scene_id):
                         is_allowed = True
                         break
             
             if not is_allowed:
                 ep_info = f"{current_ep.episode_id} ({os.path.basename(current_ep.scene_id)})"
                 logger.info(f"Skipping episode {ep_info} (Not in assignment)")
                 return



        dones = [False] * self.config.NUM_ENVIRONMENTS
        infos = [{}] * self.config.NUM_ENVIRONMENTS


        # --- Init ---
        self._action = None
        action_list = []
        going_to_stop = False
        panorama_got = False
        navigate_or_not = False
        collided = 0
        search_destination = False
        current_pose = full_pose[0] if full_pose is not None else None
        
        self.latest_la_output = {}

        target_map_x, target_map_y = None, None
        is_backtracking = False  # Flag to indicate if current target is a backtrack target
        backtracking_only = False  # True when we first return to an old waypoint before re-deciding.
        current_la_action = 'NAVIGATE'  # Track LA model's decision ('NAVIGATE' or 'STOP')
        waypoint = None  # init waypoint
        
        # New: Track consecutive stop failures
        self.stop_feedback = ""
        self.consecutive_stop_failures = 0

        # Steps since the current target was set
        max_steps_to_target = 15  # re-plan after this many steps
        target_set_step = None  # step when the target was set

        # Initial map state
        full_map = self.mapping_module.get_full_map()
        step = 0
        action_step = 0
        last_step = action_step
        while step < self.max_step:
            pos = self.envs.call_at(0,'_env')._sim.get_agent_state().position
            self.pos_list.append(pos)
            if self._action != 1:
                last_step = action_step
            # =================================================================
            # 1. Perceive/analyze (ANALYZE STATE for step N)
            #    Process the result of the previous action
            # =================================================================
            if dones[0]:
                self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None))
                return
            self.visualizer.instruction = self.instruction
            self.visualizer.destination = self.destination
            self.visualizer._action = self._action

            logger.info(f"\nepisode:{self.current_episode_id}, step:{step}")

            gray_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
            # logger.info(f"instr: {self.instruction}")
            # logger.info(f"Targets visited: {len(self.visited_targets)}")

            # Update pose and position
            last_pose = current_pose
            current_pose = full_pose[0]
            self.current_step = step
            self.visualizer.sync(step, self.current_episode_id)

            position = current_pose[:2] * 100 / self.resolution
            agent_map_x, agent_map_y = int(position[0]), int(position[1])
            # logger.info("full pose: ", current_pose)

            # Save visualization frame
            # Prepare iplanner trajectory data
            iplanner_traj_list = None
            if getattr(self, 'iplanner_traj_global', None) is not None:
                iplanner_traj_list = [(pt[0], pt[1]) for pt in self.iplanner_traj_global]

            self.visualizer._save_rgb_frame(obs[0], step, self.visited_targets, self.current_episode_id, (target_map_x, target_map_y), todo_list=self.agent._format_todo_for_prompt() if self.agent.use_todo_list else None, la_output=self.latest_la_output, iplanner_traj=iplanner_traj_list)

            # Save continuous history images
            if self.use_continuous_history and step % self.history_interval == 0:
                rgb_to_save = obs[0]['rgb'].copy()
                if isinstance(rgb_to_save, np.ndarray):
                    if rgb_to_save.dtype != np.uint8:
                        rgb_to_save = (rgb_to_save * 255).astype(np.uint8)
                    img_save = Image.fromarray(rgb_to_save)
                else:
                    img_save = rgb_to_save
                
                self.history_images.append({'step': step, 'image': img_save})

            # =================================================================
            # 2. Plan/decide (PLAN/DECIDE for step N)
            #    Four-step navigation logic:
            #    Step 1: no target -> spin to collect a panorama
            #    Step 2: after spinning, query the LLM for a target
            #    Step 3: with a target, navigate to it  
            #    Step 4: on timeout or arrival, reset and go back to Step 1
            # =================================================================

            # High-level decisions only when action_list is empty
            if not action_list:
                # Reset target on timeout
                if target_map_x is not None and target_map_y is not None and target_set_step is not None:
                    steps_since_target_set = step - target_set_step
                    if steps_since_target_set >= max_steps_to_target:
                        # logger.info(f"Target timeout: {steps_since_target_set} steps since target set, exceeding {max_steps_to_target} limit.")
                        if backtracking_only:
                            logger.info("Backtrack-only return timed out, resetting navigation state.")

                        if len(self.visited_targets) > 0:
                            self.visited_targets.pop()
                        
                        # Step 4: reset state and restart
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        backtracking_only = False
                        # logger.info("Reset navigation state due to timeout.")

                # Arrival check (distance)
                if target_map_x is not None and target_map_y is not None:
                    distance_to_target = np.sqrt((target_map_x - agent_map_x) ** 2 + (target_map_y - agent_map_y) ** 2)
                    
                    # Determine threshold based on navigation type
                    # For backtracking, use same threshold as normal nav (or specific if needed)
                    # User requested: Normal 0.75m, Backtrack 0.75m. So just use self.target_reached_threshold (which is 0.75m)
                    current_threshold = self.target_reached_threshold
                    
                    # logger.info(f"Agent: ({agent_map_x}, {agent_map_y}), Target: ({target_map_x}, {target_map_y})")
                    # logger.info(f"Distance to target: {distance_to_target:.2f} (threshold: {current_threshold})")
                    
                    if distance_to_target < current_threshold:
                        # logger.info(f"Target reached! Distance: {distance_to_target:.2f}")
                        if backtracking_only:
                            logger.info("Reached backtrack waypoint. Returning to normal decision flow.")

                        # Record arrival image
                        if len(self.visited_targets) > 0:
                            dist_calc = lambda target: np.sqrt(
                                (target['world_coords'][0] - self.visited_targets[-1]['world_coords'][0]) ** 2 + (
                                            target['world_coords'][1] - self.visited_targets[-1]['world_coords'][
                                        1]) ** 2) if 'world_coords' in target else float('inf')
                            for target in self.visited_targets[:-1]:
                                if dist_calc(target) < self.target_reached_threshold:
                                    # logger.info('Removed duplicate waypoint due to proximity. Distance: %f', dist_calc(target))
                                    self.visited_targets.pop()
                                    break

                        # Step 4: reset state for the next target
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        is_backtracking = False # Reset flag
                        backtracking_only = False
                        current_la_action = 'NAVIGATE' # Reset action
                        # logger.info("Reset navigation state - target reached.")

                # Step 1: no target and no spin yet -> spin for a panorama and pick a direction
                if target_map_x is None and not panorama_got and going_to_stop:
                    # Double-check stop logic
                    # logger.info("Performing Double-Check Stop...")
                    
                    # Acquire panorama for check                    
                    panorama_frames = self.get_panorama(obs[0], step)
                    step += 12 # get_panorama takes 12 steps
                    
                    if 'episode_finished' in panorama_frames:
                        break
                        
                    should_stop, stop_response = self.agent.double_check_stop(self.instruction, panorama_frames, self.visited_targets, episode_id=self.current_episode_id, step=step, history_images=self.history_images)
                    self.latest_la_output = stop_response
                    
                    if should_stop:
                        # logger.info('Final stop confirmed by Double-Check.')
                        action_list.append(0)  # STOP action
                        self.consecutive_stop_failures = 0 # Success
                        if self.task_type == "EQA":
                            # EQA: STOP doesn't terminate the episode in habitat
                            # (ANSWER is a separate action). Oracle-QA once, mark
                            # done, let the post-step block exit the rollout.
                            try:
                                self.last_submitted_answer = self.agent.query_llm_oracle(panorama_frames, self.instruction)
                            except Exception as e:
                                logger.error(f"EQA oracle failed: {e}")
                                self.last_submitted_answer = None
                            self._eqa_done_requested = True
                    else:
                        # logger.info('Double-Check rejected stop. Continuing navigation.')
                        going_to_stop = False
                        navigate_or_not = False
                        panorama_got = False # Ensure we re-enter Step 1 logic properly
                        current_la_action = 'NAVIGATE' # Ensure we default back to navigate
                        
                        # Handle rejection feedback
                        self.consecutive_stop_failures += 1
                        self.stop_feedback = f"System: Your previous decision to STOP was rejected by the double-check mechanism because the target was not clearly visible or not close enough (Failure count: {self.consecutive_stop_failures}). Please navigate closer to the target or verify your position."
                        
                        if self.consecutive_stop_failures >= 3:
                            logger.info(f"Forcing STOP after {self.consecutive_stop_failures} consecutive double-check rejections.")
                            action_list.append(0) # STOP action
                            # Reset flags to prevent loop
                            self.stop_feedback = ""
                            self.consecutive_stop_failures = 0
                            if self.task_type == "EQA":
                                try:
                                    self.last_submitted_answer = self.agent.query_llm_oracle(panorama_frames, self.instruction)
                                except Exception as e:
                                    logger.error(f"EQA oracle failed: {e}")
                                    self.last_submitted_answer = None
                                self._eqa_done_requested = True
                        
                elif target_map_x is None and not navigate_or_not:
                    # logger.info("Step 1: Getting panorama and deciding navigation direction...")
                    # Record current view as init_image
                    current_rgb = obs[0]['rgb'].copy()

                    # Collect panorama
                    panorama_frames = self.get_panorama(obs[0], step)
                    step += 12 # get_panorama takes 12 steps
                    
                    if 'episode_finished' in panorama_frames:
                        break

                    # New waypoint record with the current position
                    waypoint_id = len(self.visited_targets)
                    self.visited_targets.append({
                        'step': step,
                        'init_image': Image.fromarray(current_rgb) if isinstance(current_rgb,
                                                                                 np.ndarray) else current_rgb,
                        'panorama_frames': panorama_frames,
                        'world_coords': (agent_map_x, agent_map_y),  # current position
                        'full_pose': current_pose # full pose (x, y, heading)
                    })
                    self.total_waypoints += 1

                    # Save the waypoint panorama RGB
                    self.visualizer._save_waypoint_panorama_rgb(panorama_frames, waypoint_id, step)

                    # Combine stop feedback and todo verification feedback
                    combined_feedback = self.stop_feedback
                    todo_feedback = getattr(self.agent, 'todo_verification_feedback', "")
                    if todo_feedback:
                        combined_feedback = (combined_feedback + "\n" + todo_feedback).strip()
                        # Consume the feedback (don't repeat it forever)
                        self.agent.todo_verification_feedback = ""

                    # Call navigate_or_backtrack
                    decision = self.agent.navigate_or_backtrack(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        feedback=combined_feedback,
                        episode_id=self.current_episode_id,
                        step=step,
                        history_images=self.history_images if self.use_continuous_history else None
                    )
                    self.latest_la_output = decision
                    # logger.info(f"Navigation decision: {decision}")

                    # If LA decides to NAVIGATE (and not STOP), clear the feedback
                    if decision.get('action') == 'NAVIGATE' or decision.get('action') == 'BACKTRACK':
                        self.stop_feedback = ""
                        # If we had stop failures but now LA decided to move, we can reset the counter
                        # self.consecutive_stop_failures = 0 # Optional: reset or keep counting? Reset seems fair.
                        if decision.get('action') == 'NAVIGATE':
                             # Only reset if it's a new navigation decision, backtrack might still be "stuck" logic
                             pass

                    if decision.get('action') == 'STOP':
                        current_la_action = 'STOP'

                    if decision.get('action', 'NAVIGATE') == 'BACKTRACK':
                        self.total_backtracks += 1
                        target_waypoint_id = decision.get('waypoint', 0)
                        if isinstance(target_waypoint_id, int) and target_waypoint_id < len(self.visited_targets) - 1:
                            visited_targets_up_to_backtrack = self.visited_targets[:target_waypoint_id + 1]
                            failed_path = self.visited_targets[target_waypoint_id + 1:]
                            backtrack_point = visited_targets_up_to_backtrack[-1]

                            if self.agent.backtrack_second_chance:
                                logger.info(f"Initiating Backtrack Re-planning to Waypoint {target_waypoint_id}")

                                # 2. Replan
                                new_direction, replan_response = self.agent.replan_at_backtrack(
                                    self.instruction,
                                    visited_targets_up_to_backtrack,
                                    failed_path,
                                    episode_id=self.current_episode_id,
                                    step=step,
                                    history_images=self.history_images if self.use_continuous_history else None
                                )
                                self.latest_la_output = replan_response
                                logger.info(f"Re-planned direction: {new_direction}")

                                # 3. Get corresponding image from backtrack point
                                panorama_frames = backtrack_point['panorama_frames']
                                direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                                target_angle = direction_map.get(new_direction, 0)
                                frame_idx = target_angle // 90

                                if frame_idx < len(panorama_frames):
                                    # 4. Get BBox from VA model
                                    dir_rgb = panorama_frames[frame_idx]['rgb']
                                    dir_depth = panorama_frames[frame_idx]['depth']

                                    bbox = self.agent.query_llm(
                                        instruction=self.instruction,
                                        visited_targets=visited_targets_up_to_backtrack,
                                        rgb_image=dir_rgb,
                                        depth_image=dir_depth,
                                        width=self.width,
                                        height=self.height,
                                        current_step=self.current_step,
                                        progress_analysis="Backtracking and Replanning",
                                        episode_id=self.current_episode_id
                                    )

                                    bbox_save_path = os.path.join(self.save_dir, str(self.current_episode_id), f'bbox_step{step}.png')
                                    os.makedirs(os.path.dirname(bbox_save_path), exist_ok=True)

                                    d_img = dir_depth
                                    if len(d_img.shape) == 3:
                                        d_img = d_img[:, :, 0]

                                    dummy_depth = np.expand_dims(d_img, axis=2) # (H,W,1)
                                    processed_depth = self._preprocess_depth(dummy_depth, 0.1, 5.0) / 100.0

                                    if not self.agent.stair:
                                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), min(int(bbox.get('y2', 0)), self.height - 1))
                                    else:
                                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y1', 0)))

                                    base_pose = backtrack_point.get('full_pose')
                                    if base_pose is None:
                                        target_map_x, target_map_y = backtrack_point['world_coords']
                                    else:
                                        frame_angle = panorama_frames[frame_idx]['angle']

                                        if isinstance(base_pose, torch.Tensor):
                                            view_pose = base_pose.clone().cpu().numpy()
                                        else:
                                            view_pose = base_pose.copy()

                                        if view_pose.ndim == 2:
                                            view_pose = view_pose[0]

                                        view_pose[2] += frame_angle
                                        view_pose[2] = view_pose[2] % 360

                                        target = get_world_xz_from_pixel(
                                            pixel_coords=coords,
                                            depth_image=processed_depth,
                                            full_pose=view_pose,
                                            camera_intrinsics=self._get_camera_intrinsics(),
                                        )

                                        new_target_x = int(target[0] * 100.0 / self.resolution)
                                        new_target_y = int(target[1] * 100.0 / self.resolution)
                                        new_target_x = max(0, min(new_target_x, self.map_shape[0] - 1))
                                        new_target_y = max(0, min(new_target_y, self.map_shape[1] - 1))

                                        target_map_x, target_map_y = new_target_x, new_target_y

                                revisit_waypoint = {
                                    'step': step,
                                    'init_image': backtrack_point['init_image'],
                                    'panorama_frames': backtrack_point['panorama_frames'],
                                    'world_coords': backtrack_point['world_coords'],
                                    'full_pose': backtrack_point['full_pose'],
                                    'is_revisit': True,
                                    'original_waypoint_id': target_waypoint_id,
                                    'backtrack_from_waypoint_id': len(self.visited_targets) - 1
                                }

                                self.visited_targets.append(revisit_waypoint)
                                self.total_waypoints += 1

                                self.visited_targets[-1]['direction_decision'] = new_direction
                                self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb, np.ndarray) else dir_rgb
                                self.visited_targets[-1]['bbox'] = bbox
                                self.visited_targets[-1]['progress_analysis'] = "Backtracking Re-plan"

                                is_backtracking = False # Treat as new target
                                backtracking_only = False
                            else:
                                logger.info(f"Initiating Backtrack-only return to Waypoint {target_waypoint_id}")
                                if backtrack_point.get('world_coords') is not None:
                                    target_map_x, target_map_y = backtrack_point['world_coords']
                                else:
                                    decision['action'] = 'NAVIGATE'
                                    panorama_got = True
                                    continue

                                revisit_waypoint = {
                                    'step': step,
                                    'init_image': backtrack_point['init_image'],
                                    'panorama_frames': backtrack_point['panorama_frames'],
                                    'world_coords': backtrack_point['world_coords'],
                                    'full_pose': backtrack_point['full_pose'],
                                    'is_revisit': True,
                                    'original_waypoint_id': target_waypoint_id,
                                    'backtrack_from_waypoint_id': len(self.visited_targets) - 1,
                                    'progress_analysis': "Backtracking Only"
                                }

                                self.visited_targets.append(revisit_waypoint)
                                self.total_waypoints += 1
                                is_backtracking = True
                                backtracking_only = True

                            target_set_step = step

                        else:
                            # logger.info("Invalid waypoint ID for backtrack, continuing with navigation")
                            decision['action'] = 'NAVIGATE'
                        panorama_got = True
                    if decision.get('action', 'NAVIGATE') == 'NAVIGATE':
                        navigate_or_not = True
                        direction = decision.get('direction', 'forward')

                        if decision.get('stop_signal', False):
                            current_la_action = 'STOP'
                        else:
                            current_la_action = 'NAVIGATE'

                        progress_analysis = decision.get('progress_analysis', '')
                        reasoning = decision.get('reasoning', '')

                        # Record decision
                        self.visited_targets[-1].update({
                            'progress_analysis': progress_analysis,
                            'reasoning': reasoning,
                            'direction_decision': direction
                        })

                        # Image for the chosen direction
                        direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                        target_angle = direction_map.get(direction, 0)
                        frame_idx = target_angle // 90

                        if frame_idx < len(panorama_frames):
                            dir_rgb = panorama_frames[frame_idx]['rgb']
                            self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb,
                                             np.ndarray) else dir_rgb
                            self.visited_targets[-1]['turn_action'] = f"turn {direction}"

                        # Turn actions for the chosen direction
                        if direction == 'left':
                            action_list.extend([2] * 3)  # TURN_LEFT 90 deg (3 * 30 deg)
                        elif direction == 'right':
                            action_list.extend([3] * 3)  # TURN_RIGHT 90 deg
                        elif direction == 'behind':
                            action_list.extend([2] * 6)  # TURN_LEFT 180 deg (6 * 30 deg)
                        # forward needs no turn

                        panorama_got = True
                        # logger.info(f"Step 1 completed: Direction decision = {direction}, added turn actions")

                # Step 2: spin done and no pending actions -> query the LLM for a target
                elif target_map_x is None and panorama_got and not action_list:
                    # logger.info("Step 2: Querying LLM for specific target...")

                    # Call query_llm on the current observation with progress_analysis
                    progress_analysis = self.visited_targets[-1].get('progress_analysis', '')

                    bbox = self.agent.query_llm(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        rgb_image=obs[0]['rgb'],
                        depth_image=obs[0]['depth'],
                        width=self.width,
                        height=self.height,
                        current_step=self.current_step,
                        progress_analysis=progress_analysis,
                        planned_action=current_la_action,
                        episode_id=self.current_episode_id
                    )
                    # logger.info(f"LLM response: {bbox}")

                    # Update waypoint info
                    self.visited_targets[-1].update({
                        'description': bbox.get('target', 'unknown target'),
                        'bbox': bbox,
                        'llm_reasoning': bbox.get('reasoning', ''),
                        'llm_progress': bbox.get('progress', '')
                    })

                    # If the current action is STOP, we are approaching the target.
                    # Once we reach this target (in Step 4), we should then trigger the double check.
                    if current_la_action == 'STOP':
                        going_to_stop = True

                    # Handle the target bbox returned by the LLM
                    depth_image = self._preprocess_depth(obs[0]['depth'], 0.1, 5.0) / 100.0
                    if not self.agent.stair:
                        # coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y2', 0)))
                        # coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int((bbox.get('y1', 0) + bbox.get('y2', 0)) / 2.0))
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), min(int(bbox.get('y2', 0)), self.height - 1))
                    else:
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y1', 0)))

                    # Pixel -> map coordinates
                    while True:
                        target = get_world_xz_from_pixel(
                            pixel_coords=coords,
                            depth_image=depth_image,
                            full_pose=current_pose,
                            camera_intrinsics=self._get_camera_intrinsics(),
                        )
                        new_target_x = int(target[0] * 100.0 / self.resolution)
                        new_target_y = int(target[1] * 100.0 / self.resolution)
                        new_target_x = max(0, min(new_target_x, self.map_shape[0] - 1))
                        new_target_y = max(0, min(new_target_y, self.map_shape[1] - 1))

                        if self.traversable[new_target_y, new_target_x] == 1 or depth_image.max() < 0.1:
                            target_map_x, target_map_y = new_target_x, new_target_y
                            target_set_step = step
                            # logger.info(f"Target set at map coordinates: ({target_map_x}, {target_map_y}) at step {step}")

                            waypoint = np.array([target_map_y, target_map_x])

                            # Controller Selection Logic (Step 2)
                            use_special_controller = False
                            selected_controller = None # 'iplanner', 'navdp'

                            # Determine if we should use a special controller (NavDP/iPlanner)
                            # Logic:
                            # 1. if use_fmm is False -> MUST use special controller (NavDP or iPlanner, whichever is True)
                            # 2. if use_fmm is True -> use special controller ONLY IF (stair is True AND one of them is True)
                            
                            has_special_controller = self.use_navdp or self.use_iplanner
                            
                            if not self.use_fmm:
                                # FMM disabled, must use special controller
                                if self.use_iplanner: selected_controller = 'iplanner'
                                elif self.use_navdp: selected_controller = 'navdp'
                                use_special_controller = True
                            else:
                                # FMM enabled
                                if has_special_controller and self.agent.stair:
                                     # Hybrid mode: Use special controller for stairs
                                     if self.use_iplanner: selected_controller = 'iplanner'
                                     elif self.use_navdp: selected_controller = 'navdp'
                                     use_special_controller = True
                                else:
                                     # Use FMM (Normal case or stairs but no special controller enabled)
                                     use_special_controller = False

                            if use_special_controller:
                                traj_np = None  # Initialize to avoid undefined variable error
                                if selected_controller == 'iplanner':
                                    logger.info(f"DEBUG: Step 2 - Using iPlanner (stair={self.agent.stair})")
                                    # Calculate local goal from target_map_x, target_map_y
                                    # Convert map indices to world meters
                                    tx = target_map_x * self.resolution / 100.0
                                    ty = target_map_y * self.resolution / 100.0
                                
                                    ax = current_pose[0] # x in meters
                                    ay = current_pose[1] # y in meters (z in world)
                                    ah = np.deg2rad(current_pose[2]) # heading in radians (converted from degrees)
                                
                                    # Relative vector in world frame
                                    dx = tx - ax
                                    dy = ty - ay
                                    
                                    # Rotate to robot frame (X-forward, Y-left, Z-up)
                                    x_body = dx * np.cos(ah) + dy * np.sin(ah)   # Forward distance
                                    y_body = -dx * np.sin(ah) + dy * np.cos(ah)  # Left distance
                                    z_body = 0.0                                 # Up distance
                                    
                                    local_goal = torch.tensor([[x_body, y_body, z_body]]).to(self.device)
                                    
                                    # Call iplanner (Force update)
                                    min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                                    max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                                    depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                                    
                                    dep_tensor = torch.from_numpy(depth_meters).to(self.device).unsqueeze(0) # (1, H, W, 1)
                                    keypoints, traj, fear = self.iplanner_agent.step_pointgoal(dep_tensor, local_goal)
                                    traj_np = None  # Fix: initialize to avoid undefined variable
                                    
                                    # Store Global Trajectory
                                    traj_np = traj[0].cpu().numpy()
                                    global_traj = []
                                    for pt in traj_np:
                                        xb, yb, zb = pt
                                        w_dx = xb * np.cos(ah) - yb * np.sin(ah)
                                        w_dy = xb * np.sin(ah) + yb * np.cos(ah)
                                        wx = ax + w_dx
                                        wy = ay + w_dy
                                        global_traj.append([wx, wy, zb])
                                    self.iplanner_traj_global = np.array(global_traj)
                                    self.last_iplanner_step = step

                                    if traj_np is not None:  # Check if traj_np is defined
                                        # Visualization (Project traj_np to depth)
                                        if self.visualize:
                                            try:
                                                # Use traj_np (Local Frame)
                                                viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                                if viz_depth is not None:
                                                    if len(viz_depth.shape) == 2:
                                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                                    elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                                    
                                                    intrinsics = self._get_camera_intrinsics()
                                                    fx, fy = intrinsics[0,0], intrinsics[1,1]
                                                    cx, cy = intrinsics[0,2], intrinsics[1,2]
                                                    h, w = viz_depth.shape[:2]
                                                    
                                                    valid_points = []
                                                    for pt in traj_np:
                                                        x, y, z = pt # x=forward, y=left, z=up
                                                        
                                                        xc = -y
                                                        yc = -z 
                                                        zc = x
                                                        
                                                        if x > 0.1:
                                                            u = fx * (-y) / x + cx
                                                            u = int(u)
                                                            v = int(fy * (self.config.MAP.AGENT_HEIGHT - z) / x + cy)
                                                            
                                                            if 0 <= u < w and 0 <= v < h:
                                                                valid_points.append((u, v))

                                                    # Draw lines
                                                    for i in range(len(valid_points)-1):
                                                        cv2.line(viz_depth, valid_points[i], valid_points[i+1], (0, 0, 255), 2) # Red
                                                    for pt in valid_points:
                                                        cv2.circle(viz_depth, pt, 3, (0, 255, 0), -1) # Green
                                                        
                                                    # Save
                                                    episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                                    os.makedirs(episode_dir, exist_ok=True)
                                                    cv2.imwrite(os.path.join(episode_dir, f"iplanner_traj_step{step}.png"), viz_depth)
                                            except Exception as e:
                                                logger.info(f"Error visualizing iplanner: {e}")

                                        # Direct Control Action
                                        # Find target point
                                        lookahead_dist = 1.0
                                        target_pt = None
                                        for pt in traj_np:
                                            dist = np.linalg.norm(pt[:2]) # x, y
                                            if dist > lookahead_dist:
                                                target_pt = pt
                                                break
                                        if target_pt is None and len(traj_np) > 0:
                                            target_pt = traj_np[-1]
                                        
                                        action_id = HabitatSimActions.MOVE_FORWARD
                                        if target_pt is not None:
                                            x, y = target_pt[:2]
                                            angle = np.arctan2(y, x) # y is left, x is forward. +angle = left.
                                            # Threshold 15 deg = 0.26 rad
                                            if angle > 0.26:
                                                action_id = HabitatSimActions.TURN_LEFT
                                            elif angle < -0.26:
                                                action_id = HabitatSimActions.TURN_RIGHT
                                            else:
                                                action_id = HabitatSimActions.MOVE_FORWARD
                                        
                                        action_list.append(action_id)
                                        break # Skip FMM
                                elif selected_controller == 'navdp':
                                    # logger.info(f"DEBUG: Step 2 - Using NavDP (stair={self.agent.stair})")
                                    tx = target_map_x * self.resolution / 100.0
                                    ty = target_map_y * self.resolution / 100.0
                                    ax = current_pose[0]
                                    ay = current_pose[1]
                                    ah = np.deg2rad(current_pose[2])
                                    
                                    dx = tx - ax
                                    dy = ty - ay
                                    
                                    x_body = dx * np.cos(ah) + dy * np.sin(ah)
                                    y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                                    z_body = 0.0
                                    
                                    goals = np.array([[x_body, y_body, z_body]])
                                    rgb_img = obs[0]['rgb']
                                    min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                                    max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                                    depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                                    depth_img = depth_meters
                                    
                                    if depth_img.ndim == 2:
                                        depth_img = depth_img[:, :, np.newaxis]
                                    images = rgb_img[np.newaxis, ...]
                                    depths = depth_img[np.newaxis, ...]
                                    
                                    next_point, traj, val, mask = self.navdp_agent.step_pointgoal(goals, images, depths)
                                    
                                    # Visualization
                                    local_traj = next_point[0]
                                    global_traj = []
                                    for pt in local_traj:
                                        lb_x, lb_y = pt[:2]
                                        wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                                        wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                                        global_traj.append([wx, wy])
                                    self.iplanner_traj_global = np.array(global_traj)

                                    x, y = local_traj[0][:2]
                                    angle = np.arctan2(y, x)
                                    if angle > 0.26:
                                        action_id = HabitatSimActions.TURN_LEFT
                                    elif angle < -0.26:
                                        action_id = HabitatSimActions.TURN_RIGHT
                                    else:
                                        action_id = HabitatSimActions.MOVE_FORWARD
                                    
                                    action_list.append(action_id)
                                    break


                            # Generate the first navigation action immediately
                            if not use_special_controller and self.use_fmm:
                                # logger.info(f"DEBUG: Step 2 - Generating FMM action. use_iplanner={self.use_iplanner}, use_navdp={self.use_navdp}, stair={self.agent.stair}")
                                # Add Visualization for FMM
                                if self.visualize:
                                    try:
                                        viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                        if viz_depth is not None:
                                            if len(viz_depth.shape) == 2:
                                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                            elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                            
                                            intrinsics = self._get_camera_intrinsics()
                                            fx, fy = intrinsics[0,0], intrinsics[1,1]
                                            cx, cy = intrinsics[0,2], intrinsics[1,2]
                                            h, w = viz_depth.shape[:2]

                                            # Draw Goal
                                            ax = current_pose[0]
                                            ay = current_pose[1]
                                            ah = np.deg2rad(current_pose[2])
                                            
                                            tx_viz = target_map_x * self.resolution / 100.0
                                            ty_viz = target_map_y * self.resolution / 100.0
                                            dx_viz = tx_viz - ax
                                            dy_viz = ty_viz - ay
                                            gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                                            gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                                            gz = 0.0
                                            
                                            if gx > 0.1:
                                                gu = fx * (-gy) / gx + cx
                                                gv = fy * (self.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                                gu, gv = int(gu), int(gv)
                                                if 0 <= gu < w and 0 <= gv < h:
                                                    # Draw Yellow Star for Goal
                                                    cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                                    cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                                            episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                            os.makedirs(episode_dir, exist_ok=True)
                                            cv2.imwrite(os.path.join(episode_dir, f"iplanner_traj_step{step}.png"), viz_depth)
                                    except Exception as e:
                                        logger.info(f"Error in FMM viz: {e}")

                                navigation_action = self.policy._get_action(
                                    current_pose, waypoint, full_map[0], self.traversable,
                                    self.collision_map, step, self.current_episode_id,
                                    self.detected_classes, search_destination
                                )
                                action_list.append(navigation_action)
                                # logger.info(f"Added initial navigation action: {navigation_action}")
                            elif not use_special_controller and not self.use_fmm:
                                logger.warning("Step 2 - FMM is disabled and no other planner took over! This should not happen due to init validation.")
                            break
                        depth_image = depth_image - 0.1

                    panorama_got = False  # reset for next time
                    # logger.info("Step 2 completed: Target acquired from LLM")

                # Step 3: target set and no pending actions -> keep navigating
                elif target_map_x is not None and target_map_y is not None and not action_list:
                    # logger.info(f"Step 3: Continuing navigation to target ({target_map_x}, {target_map_y})")
                    waypoint = np.array([target_map_y, target_map_x])
                    
                    # Controller Selection Logic (Step 3)
                    use_special_controller = False
                    selected_controller = None 
                    
                    has_special_controller = self.use_navdp or self.use_iplanner
                    
                    if not self.use_fmm:
                        if self.use_iplanner: selected_controller = 'iplanner'
                        elif self.use_navdp: selected_controller = 'navdp'
                        use_special_controller = True
                    else:
                        if has_special_controller and self.agent.stair:
                             if self.use_iplanner: selected_controller = 'iplanner'
                             elif self.use_navdp: selected_controller = 'navdp'
                             use_special_controller = True
                        else:
                             use_special_controller = False

                    if selected_controller == 'iplanner':
                        traj_local_np = None
                        
                        ax = current_pose[0]
                        ay = current_pose[1]
                        ah = np.deg2rad(current_pose[2])
                        
                        if step - self.last_iplanner_step >= self.iplanner_update_freq or self.iplanner_traj_global is None:
                            # Replan
                            tx = target_map_x * self.resolution / 100.0
                            ty = target_map_y * self.resolution / 100.0
                            
                            dx = tx - ax
                            dy = ty - ay
                            
                            x_body = dx * np.cos(ah) + dy * np.sin(ah)
                            y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                            z_body = 0.0
                            
                            local_goal = torch.tensor([[x_body, y_body, z_body]]).to(self.device)
                            
                            min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                            max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                            depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                            dep_tensor = torch.from_numpy(depth_meters).to(self.device).unsqueeze(0)
                            
                            keypoints, traj, fear = self.iplanner_agent.step_pointgoal(dep_tensor, local_goal)
                            
                            # Update Global
                            traj_local_np = traj[0].cpu().numpy()
                            global_traj = []
                            for pt in traj_local_np:
                                xb, yb, zb = pt
                                w_dx = xb * np.cos(ah) - yb * np.sin(ah)
                                w_dy = xb * np.sin(ah) + yb * np.cos(ah)
                                wx = ax + w_dx
                                wy = ay + w_dy
                                global_traj.append([wx, wy, zb])
                            self.iplanner_traj_global = np.array(global_traj)
                            self.last_iplanner_step = step
                            logger.info(f"DEBUG: iPlanner generated trajectory with {len(global_traj)} points in Step 3")
                        else:
                            # Use stored global traj -> local
                            if self.iplanner_traj_global is not None and len(self.iplanner_traj_global) > 0:
                                local_traj = []
                                for gpt in self.iplanner_traj_global:
                                    gx, gy, gz = gpt
                                    dx = gx - ax
                                    dy = gy - ay
                                    # Inverse rotate
                                    xb = dx * np.cos(ah) + dy * np.sin(ah)
                                    yb = -dx * np.sin(ah) + dy * np.cos(ah)
                                    local_traj.append([xb, yb, gz])
                                traj_local_np = np.array(local_traj)
                            else:
                                traj_local_np = np.array([])

                        # Unified Visualization on VLM Map (Update current frame)
                        if self.visualize and self.iplanner_traj_global is not None:
                            try:
                                traj_world_list = [(pt[0], pt[1]) for pt in self.iplanner_traj_global]
                                self.visualizer._save_rgb_frame(
                                    obs[0], 
                                    step, 
                                    self.visited_targets, 
                                    self.current_episode_id, 
                                    (target_map_x, target_map_y), 
                                    todo_list=self.agent._format_todo_for_prompt() if self.agent.use_todo_list else None,
                                    iplanner_traj=traj_world_list
                                )
                            except Exception as e:
                                logger.info(f"Error updating VLM map viz in Step 3: {e}")

                        # Visualization (Project traj_local_np to depth)
                        if self.visualize:
                            try:
                                viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                if viz_depth is not None:
                                    if len(viz_depth.shape) == 2:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    
                                    intrinsics = self._get_camera_intrinsics()
                                    fx, fy = intrinsics[0,0], intrinsics[1,1]
                                    cx, cy = intrinsics[0,2], intrinsics[1,2]
                                    h, w = viz_depth.shape[:2]
                                    
                                    valid_points = []
                                    if traj_local_np is not None:
                                        for pt in traj_local_np:
                                            x, y, z = pt
                                            if x > 0.1:
                                                u = fx * (-y) / x + cx
                                                v = fy * (self.config.MAP.AGENT_HEIGHT - z) / x + cy
                                                u = int(u)
                                                v = int(v)
                                                if 0 <= u < w and 0 <= v < h:
                                                    valid_points.append((u, v))
                                    
                                    for i in range(len(valid_points)-1):
                                        cv2.line(viz_depth, valid_points[i], valid_points[i+1], (0, 0, 255), 2)
                                    for pt in valid_points:
                                        cv2.circle(viz_depth, pt, 3, (0, 255, 0), -1)
                                        
                                    # Draw Goal
                                    # Recalculate local goal for visualization
                                    tx_viz = target_map_x * self.resolution / 100.0
                                    ty_viz = target_map_y * self.resolution / 100.0
                                    dx_viz = tx_viz - ax
                                    dy_viz = ty_viz - ay
                                    gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                                    gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                                    gz = 0.0
                                    
                                    if gx > 0.1:
                                        gu = fx * (-gy) / gx + cx
                                        gv = fy * (self.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                        gu, gv = int(gu), int(gv)
                                        if 0 <= gu < w and 0 <= gv < h:
                                            # Draw Yellow Star for Goal
                                            cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                            cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                                    episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                    os.makedirs(episode_dir, exist_ok=True)
                                    cv2.imwrite(os.path.join(episode_dir, f"iplanner_traj_step{step}.png"), viz_depth)
                            except Exception as e:
                                logger.info(f"Error in iplanner viz: {e}")

                        # Action
                        lookahead_dist = 0.1
                        target_pt = None
                        if traj_local_np is not None:
                            for pt in traj_local_np:
                                dist = np.linalg.norm(pt[:2])
                                if dist > lookahead_dist:
                                    target_pt = pt
                                    break
                            if target_pt is None and len(traj_local_np) > 0:
                                target_pt = traj_local_np[-1]
                        
                        action_id = HabitatSimActions.MOVE_FORWARD
                        if target_pt is not None:
                            x, y = target_pt[:2]
                            angle = np.arctan2(y, x)
                            if angle > 0.26:
                                action_id = HabitatSimActions.TURN_LEFT
                            elif angle < -0.26:
                                action_id = HabitatSimActions.TURN_RIGHT
                            else:
                                action_id = HabitatSimActions.MOVE_FORWARD
                        
                        action_list.append(action_id)
                    elif selected_controller == 'navdp':
                        tx = target_map_x * self.resolution / 100.0
                        ty = target_map_y * self.resolution / 100.0
                        ax = current_pose[0]
                        ay = current_pose[1]
                        ah = np.deg2rad(current_pose[2])
                        
                        dx = tx - ax
                        dy = ty - ay
                        x_body = dx * np.cos(ah) + dy * np.sin(ah)
                        y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                        z_body = 0.0
                        
                        goals = np.array([[x_body, y_body, z_body]])
                        rgb_img = obs[0]['rgb']
                        min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                        max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                        depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                        depth_img = depth_meters
                        # NavDP Frequency Control
                        # Initialize variables if not present
                        if not hasattr(self, 'last_navdp_step'):
                            self.last_navdp_step = -999
                        if not hasattr(self, 'navdp_local_traj'):
                            self.navdp_local_traj = None

                        # Execute NavDP planning every 5 steps or if no trajectory exists
                        if step - self.last_navdp_step >= 5 or self.navdp_local_traj is None or len(self.navdp_local_traj) == 0:
                            # logger.info(f"DEBUG: Executing NavDP planning at step {step}")
                            
                            if depth_img.ndim == 2:
                                depth_img = depth_img[:, :, np.newaxis]
                            images = rgb_img[np.newaxis, ...]
                            depths = depth_img[np.newaxis, ...]
                            
                            next_point, traj, val, mask = self.navdp_agent.step_pointgoal(goals, images, depths)
                            
                            # Store the new trajectory and update timestamp
                            self.navdp_local_traj = next_point[0] # (N, 3)
                            self.last_navdp_step = step
                            
                            # Update Global Visualization (Optional, but good for debugging)
                            local_traj = self.navdp_local_traj
                            global_traj = []
                            for pt in local_traj:
                                lb_x, lb_y = pt[:2]
                                wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                                wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                                global_traj.append([wx, wy])
                            self.iplanner_traj_global = np.array(global_traj)
                        
                        # Let's use the GLOBAL trajectory method for stability across steps.
                        if self.iplanner_traj_global is not None and len(self.iplanner_traj_global) > 0:
                            # Find a lookahead point on the global trajectory relative to CURRENT pose
                            target_pt = None
                            lookahead_dist = 0.5
                            
                            # Convert global trajectory back to CURRENT local frame
                            current_local_traj = []
                            for gpt in self.iplanner_traj_global:
                                gx, gy = gpt
                                dx = gx - ax
                                dy = gy - ay
                                # Inverse rotate to get local x, y
                                lx = dx * np.cos(ah) + dy * np.sin(ah)
                                ly = -dx * np.sin(ah) + dy * np.cos(ah)
                                current_local_traj.append([lx, ly])
                            
                            # Now find the target point in current local frame
                            for pt in current_local_traj:
                                dist = np.linalg.norm(pt[:2])
                                if dist > lookahead_dist:
                                    target_pt = pt
                                    break
                            
                            if target_pt is None and len(current_local_traj) > 0:
                                target_pt = current_local_traj[-1]
                        else:
                            # Fallback
                            target_pt = None

                        if target_pt is not None:
                            x, y = target_pt[:2]
                            angle = np.arctan2(y, x)
                            if angle > 0.26:
                                action_id = HabitatSimActions.TURN_LEFT
                            elif angle < -0.26:
                                action_id = HabitatSimActions.TURN_RIGHT
                            else:
                                action_id = HabitatSimActions.MOVE_FORWARD
                        else:
                            action_id = HabitatSimActions.MOVE_FORWARD
                        
                        action_list.append(action_id)
                    elif not use_special_controller and self.use_fmm:
                        logger.info(f"DEBUG: Step 3 - Using FMM. use_iplanner={self.use_iplanner}, use_navdp={self.use_navdp}, stair={self.agent.stair}")
                        # Clear special controller trajectories to avoid stale visualization
                        self.iplanner_traj_global = None
                        self.navdp_local_traj = None
                        
                        # Add Visualization for FMM
                        if self.visualize:
                            try:
                                viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                if viz_depth is not None:
                                    if len(viz_depth.shape) == 2:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    
                                    intrinsics = self._get_camera_intrinsics()
                                    fx, fy = intrinsics[0,0], intrinsics[1,1]
                                    cx, cy = intrinsics[0,2], intrinsics[1,2]
                                    h, w = viz_depth.shape[:2]

                                    # Draw Goal
                                    ax = current_pose[0]
                                    ay = current_pose[1]
                                    ah = np.deg2rad(current_pose[2])
                                    
                                    tx_viz = target_map_x * self.resolution / 100.0
                                    ty_viz = target_map_y * self.resolution / 100.0
                                    dx_viz = tx_viz - ax
                                    dy_viz = ty_viz - ay
                                    gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                                    gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                                    gz = 0.0
                                    
                                    if gx > 0.1:
                                        gu = fx * (-gy) / gx + cx
                                        gv = fy * (self.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                        gu, gv = int(gu), int(gv)
                                        if 0 <= gu < w and 0 <= gv < h:
                                            # Draw Yellow Star for Goal
                                            cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                            cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                                    episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                    os.makedirs(episode_dir, exist_ok=True)
                                    cv2.imwrite(os.path.join(episode_dir, f"iplanner_traj_step{step}.png"), viz_depth)
                            except Exception as e:
                                logger.info(f"Error in FMM viz: {e}")

                        navigation_action = self.policy._get_action(
                            current_pose, waypoint, full_map[0], self.traversable,
                            self.collision_map, step, self.current_episode_id,
                            self.detected_classes, search_destination
                        )
                        action_list.append(navigation_action)
                    else:
                        logger.warning("Step 3 - FMM is disabled and no other planner took over!")
                    # logger.info(f"Added navigation action: {navigation_action}")


            # Act only when action_list is non-empty
            if action_list:
                # =================================================================
                # 3. Act (ACT for step N)
                #    Execute the chosen action
                # =================================================================
                self._action = action_list[0]
                action_list.pop(0)
                actions = [{"action": self._action}]

                # logger.info(f'Action actually performed: {self._action}')

                outputs = self.envs.step(actions)
                step += 1

                # =================================================================
                # 4. Update (UPDATE for step N+1)
                #    Update map and pose with the post-action state
                # =================================================================
                obs, _, dones, infos = [list(x) for x in zip(*outputs)]
                # logger.info('Sensor pose', obs[0]['sensor_pose'])

                if not dones[0]:
                    batch_obs = self._batch_obs(obs)
                    poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
                    self.mapping_module(batch_obs, poses, self.current_step)
                    full_map, full_pose, one_step_full_map = \
                        self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
                    self.mapping_module.one_step_full_map.fill_(0.)
                    self.mapping_module.one_step_local_map.fill_(0.)

                    self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
                    self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

                    last_pose = current_pose
                    current_pose = full_pose[0]
                    if last_pose is not None and current_pose is not None:
                        displacement = calculate_displacement(last_pose, current_pose, self.resolution)
                        if displacement < 0.2 * 100 / self.resolution:
                            collided += 1
                        else:
                            collided = 0
                            replan = False
                        if collided >= 30:
                            fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                                 f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                            with open(fname, "a") as f:
                                f.writelines(
                                    f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")

                    current_action = self._action
                    if last_pose is not None and current_action is not None and current_action == 1:
                        collision_map = collision_check_fmm(last_pose, current_pose, self.resolution,
                                                            self.mapping_module.map_shape)
                        self.collision_map = np.logical_or(self.collision_map, collision_map)
                    self.traversable[self.collision_map == 1] = 0
                else:
                    self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None))
                    return
                # For EQA, STOP action does not set dones[0]=True; exit here
                # after the Oracle QA branch has set the done-requested flag.
                if getattr(self, '_eqa_done_requested', False):
                    self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None))
                    return
            else:
                pass
                # Empty action_list: skip acting, continue the loop
                # logger.info("No action to execute this step, continuing to next iteration")
            action_step += 1
        self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None), is_timeout=True)

    def eval(self):
        # Load detailed assignments if available. File is namespaced by exp_name
        # (derived from the unique EVAL_CKPT_PATH_DIR) to allow parallel runs.
        exp_name = os.path.basename(self.config.EVAL_CKPT_PATH_DIR.rstrip('/'))
        assignment_file = f"data/logs/running_log/worker_{self.local_rank}_assignments_{exp_name}.json"
        if not os.path.exists(assignment_file):
            # Backward compat: fall back to the un-namespaced filename
            assignment_file = f"data/logs/running_log/worker_{self.local_rank}_assignments.json"
        if os.path.exists(assignment_file):
            with open(assignment_file, 'r') as f:
                assignments = json.load(f)
                # Set of (id, scene_id) tuples for fast lookup
                self.allowed_episodes = set((str(item['id']), item['scene_id']) for item in assignments)
            logger.info(f"Loaded {len(self.allowed_episodes)} specific episodes to process from {assignment_file}")

        # Sampling logic
        if self.config.EVAL.EPISODE_COUNT > -1:
            pass # Sampling is handled in run_mp.py, here we just use the assigned episodes


        # self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        self.agent.reset()

        # Reset model usage stats
        self.agent.model.reset_stats()

        eps_to_eval = sum(self.envs.number_of_episodes)

        self.state_eps = {}
        t1 = time.time()
        
        logger.info(f"Worker {self.local_rank}: Starting eval loop with {eps_to_eval} episodes")
        
        for i in tqdm(range(eps_to_eval), desc=f"Worker {self.local_rank}"):
            self.rollout()
            self.reset()

        self.envs.close()

        # Print final statistics
        logger.info("=== FINAL MODEL USAGE STATISTICS ===")
        final_stats = self.agent.model.print_usage_stats()

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)

        # Calculate and print average metrics
        # avg_metrics = defaultdict(float)
        # num_episodes = len(self.state_eps)
        # if num_episodes > 0:
        #     for metrics in self.state_eps.values():
        #         for k, v in metrics.items():
        #             if isinstance(v, (int, float)):
        #                 avg_metrics[k] += v

        #     for k in avg_metrics:
        #         avg_metrics[k] /= num_episodes

        #     logger.info("Average Metrics:")
        #     logger.info(json.dumps(dict(avg_metrics), indent=2))
        # else:
        #     logger.info("No episodes evaluated.")

        # Save model usage stats to separate file
        stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                   f"model_usage_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(stats_fname, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Model usage statistics saved to: {stats_fname}")

        # Save navigation statistics (backtracks, waypoints)
        nav_stats = {
            'total_backtracks': self.total_backtracks,
            'total_waypoints': self.total_waypoints
        }
        nav_stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                       f"nav_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(nav_stats_fname, "w") as f:
            json.dump(nav_stats, f, indent=2)
        logger.info(f"Navigation statistics saved to: {nav_stats_fname}")

        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        logger.info("test time: %d", t2 - t1)

    def eval_dynamic(self, ep_queue):
        """Dynamic work-stealing eval: pop episodes from a shared mp.Queue.

        Each worker reuses its trainer/policy/model across episodes but
        rebuilds the habitat env per episode (so EPISODES_ALLOWED can be
        re-targeted to a single id). The shared queue lets fast workers
        grab the next pending episode instead of being stuck with a fixed
        slice.
        """
        import queue as _queue
        # Heavy initialisation done once per worker
        self._initialize_policy()
        self._collect_val_traj()
        self.agent.model.reset_stats()
        self.state_eps = {}
        t1 = time.time()
        ep_count = 0
        logger.info(f"Worker {self.local_rank}: dynamic queue eval starting")

        while True:
            try:
                ep_info = ep_queue.get(timeout=2)
            except _queue.Empty:
                logger.info(f"Worker {self.local_rank}: queue empty, exiting after {ep_count} eps")
                break
            ep_id = str(ep_info['id'])
            scene_id = ep_info['scene_id']

            # Re-target the dataset filter to a single ep id and rebuild env.
            self.config.defrost()
            self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [ep_id]
            self.config.freeze()
            self.allowed_episodes = {(ep_id, scene_id)}
            if hasattr(self, 'envs') and self.envs is not None:
                try:
                    self.envs.close()
                except Exception as e:
                    logger.warning(f"envs.close() failed: {e}")
            self._init_envs()
            self.agent.reset()

            try:
                self.rollout()
            except Exception as e:
                logger.error(f"Worker {self.local_rank} ep {ep_id} rollout failed: {e}")
            try:
                self.reset()
            except Exception as e:
                logger.error(f"Worker {self.local_rank} ep {ep_id} reset failed: {e}")
            ep_count += 1

        # Final cleanup + stats persistence (mirrors eval())
        if hasattr(self, 'envs') and self.envs is not None:
            try:
                self.envs.close()
            except Exception:
                pass

        final_stats = self.agent.model.print_usage_stats()
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        stem = f"{split}_r{self.local_rank}_w{self.world_size}"
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"stats_ep_ckpt_{stem}.json"), "w") as f:
            json.dump(self.state_eps, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"model_usage_stats_{stem}.json"), "w") as f:
            json.dump(final_stats, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"nav_stats_{stem}.json"), "w") as f:
            json.dump({'total_backtracks': self.total_backtracks, 'total_waypoints': self.total_waypoints}, f, indent=2)
        logger.info(f"Worker {self.local_rank}: ran {ep_count} eps in {time.time() - t1:.0f}s")


def merge_model_usage_stats(stats_dir, split="val_unseen"):
    """
    Merge model usage statistics from multiple processes.
    Args:
        stats_dir: directory with the statistics files
        split: dataset split name
    """
    import glob
    import json

    # Find all statistics files
    pattern = os.path.join(stats_dir, f"model_usage_stats_{split}_r*_w*.json")
    stat_files = glob.glob(pattern)

    if not stat_files:
        print(f"No model usage stat files found in {stats_dir} with pattern {pattern}")
        return

    # Initialize merged statistics
    merged_stats = {
        'la': {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'va': {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'total_calls': 0,
        'total_tokens': 0,
        'num_processes': 0,
        'process_stats': []
    }

    # Merge statistics from all files
    for stat_file in stat_files:
        try:
            with open(stat_file, 'r') as f:
                stats = json.load(f)

            merged_stats['la']['calls'] += stats['la']['calls']
            merged_stats['la']['input_tokens'] += stats['la']['input_tokens']
            merged_stats['la']['output_tokens'] += stats['la']['output_tokens']
            merged_stats['la']['total_tokens'] += stats['la']['total_tokens']

            merged_stats['va']['calls'] += stats['va']['calls']
            merged_stats['va']['input_tokens'] += stats['va']['input_tokens']
            merged_stats['va']['output_tokens'] += stats['va']['output_tokens']
            merged_stats['va']['total_tokens'] += stats['va']['total_tokens']

            merged_stats['total_calls'] += stats['total_calls']
            merged_stats['total_tokens'] += stats['total_tokens']
            merged_stats['num_processes'] += 1

            # Per-process statistics
            process_info = {
                'file': os.path.basename(stat_file),
                'stats': stats
            }
            merged_stats['process_stats'].append(process_info)

            # print(f"Loaded stats from: {stat_file}")

        except Exception as e:
            print(f"Error loading {stat_file}: {e}")

    # Save merged statistics
    merged_file = os.path.join(stats_dir, f"merged_model_usage_stats_{split}.json")
    with open(merged_file, 'w') as f:
        json.dump(merged_stats, f, indent=2)

    # Print summary
    print("=== MERGED MODEL USAGE STATISTICS ===")
    print(f"Number of processes: {merged_stats['num_processes']}")
    print(f"Language Action Model:")
    print(f"  - Total calls: {merged_stats['la']['calls']:,}")
    print(f"  - Total input tokens: {merged_stats['la']['input_tokens']:,}")
    print(f"  - Total output tokens: {merged_stats['la']['output_tokens']:,}")
    print(f"  - Total tokens: {merged_stats['la']['total_tokens']:,}")
    print(f"Vision Action Model:")
    print(f"  - Total calls: {merged_stats['va']['calls']:,}")
    print(f"  - Total input tokens: {merged_stats['va']['input_tokens']:,}")
    print(f"  - Total output tokens: {merged_stats['va']['output_tokens']:,}")
    print(f"  - Total tokens: {merged_stats['va']['total_tokens']:,}")
    print(f"OVERALL TOTAL:")
    print(f"  - Total calls: {merged_stats['total_calls']:,}")
    print(f"  - Total tokens: {merged_stats['total_tokens']:,}")
    print(f"Merged statistics saved to: {merged_file}")
    print("=====================================")

    return merged_stats
