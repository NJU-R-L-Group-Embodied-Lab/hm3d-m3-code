"""
Shared mapping + planner module for the multi-agent runner.

Wraps lavira's `Semantic_Mapping` (built incrementally from each agent's
RGB+depth+pose) and `FMMPlanner` (geodesic short-term goal in map coords)
so the multi-agent loop can:

  * `begin_episode(sim, agent_starts)` — initialize map & per-agent poses.
  * `step(sim, obs_list)`              — feed all agents' obs into the map,
                                          advance one map update tick.
  * `next_action_for(agent_id, target_world_xy)` — return the next discrete
                                          habitat action that walks `agent_id`
                                          toward `target_world_xy` along the
                                          FMM geodesic.
  * `render_vlm_map(...)`              — call Semantic_Mapping's existing
                                          `create_vlm_map_from_state` (lavira's
                                          VLM Navigation Map) and overlay
                                          per-agent arrows/paths in colors.

NUM_ENVIRONMENTS = N (one mapping batch slot per agent). Each agent's slot
starts centered at that agent's spawn point in world coords. We do *not*
attempt to merge maps across slots; for visualization we render slot 0 (the
episode-anchored agent) and overlay other-agent markers using world→slot-0
coordinate conversion.

Semantic detection is **disabled** in this port (no GroundingDINO call) —
`num_detected_classes = 0`. We get obstacle + explored layers only, which is
sufficient for FMM. Semantic categories on the map will be empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.fmm_planner import FMMPlanner

# LaViRA waypoint arrival: 75 cm on the occupancy map (5 cm/cell → 15 cells).
WAYPOINT_REACH_M = 0.75


def world_xz_reached(
    agent_xyz,
    target_xyz,
    threshold_m: float = WAYPOINT_REACH_M,
) -> bool:
    """True when Habitat XZ is inside the same 0.75 m disk as the drawn X."""

    if agent_xyz is None or target_xyz is None:
        return False
    try:
        dx = float(agent_xyz[0]) - float(target_xyz[0])
        dz = float(agent_xyz[2]) - float(target_xyz[2])
    except (TypeError, IndexError, ValueError):
        return False
    return math.hypot(dx, dz) < float(threshold_m)


def map_cells_euclidean_m(
    agent_rc: Tuple[int, int],
    target_rc: Tuple[int, int],
    resolution_cm: float,
) -> float:
    """Straight-line distance on the top-down map. No Habitat navmesh."""

    dr = float(target_rc[0] - agent_rc[0])
    dc = float(target_rc[1] - agent_rc[1])
    return math.hypot(dr, dc) * float(resolution_cm) / 100.0

# ---- monkey-patch lavira's rotation utils ---- #
# lavira's `get_r_matrix` uses `np.eye(3)` (float64); when matmul'd against
# a float32 depth point-cloud inside `Semantic_Mapping.forward`, pytorch
# raises "expected mat1 and mat2 to have the same dtype". We wrap the
# original so its output is always float32.
import vlnce_baselines.utils.rotation_utils as _rot_utils
_orig_get_r_matrix = _rot_utils.get_r_matrix


def _get_r_matrix_f32(*a, **k):
    return _orig_get_r_matrix(*a, **k).astype(np.float32)


_rot_utils.get_r_matrix = _get_r_matrix_f32
# also rebind the imported reference inside depth_utils (uses `ru.get_r_matrix`)
import vlnce_baselines.utils.depth_utils as _du
_du.ru.get_r_matrix = _get_r_matrix_f32


# --------------------------------------------------------------------------- #
# Per-agent map slot
# --------------------------------------------------------------------------- #
@dataclass
class AgentMapState:
    agent_id: int
    last_sim_loc: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x_map, y_map, o)
    last_world_pos: Optional[np.ndarray] = None  # habitat (x,y,z)
    world_origin_xz: Optional[Tuple[float, float]] = None  # the world (x,z) corresponding to env's map center
    # Initial habitat yaw (rad) at episode start. The mapping module's
    # internal frame is anchored to "agent's initial forward direction" —
    # mapping heading 0 means facing this initial yaw. To convert a delta
    # between habitat world (Wx, Wz) and mapping (mx, my) when yaw_0 ≠ 0,
    # we must rotate by yaw_0; the lavira flip `mx = −Wz, my = −Wx` only
    # holds for yaw_0 = 0. Stored once in `begin_episode`.
    initial_habitat_yaw_rad: float = 0.0


def _agent_sim_location(sim, agent_id: int) -> Tuple[float, float, float]:
    """Same recipe as `habitat_extensions.pose_utils.get_sim_location` but
    parametrized by agent_id so it works in multi-agent sims (the original
    hard-codes `get_agent_state(0)`)."""
    import quaternion
    state = sim.get_agent_state(agent_id)
    x = -float(state.position[2])
    y = -float(state.position[0])
    axis = quaternion.as_euler_angles(state.rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(state.rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, float(o)


def _rel_pose_change(curr: Tuple[float, float, float],
                      prev: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Verbatim port of `habitat_extensions.pose_utils.get_rel_pose_change`.
    All three coords are mapping-frame meters/radians."""
    x1, y1, o1 = prev
    x2, y2, o2 = curr
    theta = math.atan2(y2 - y1, x2 - x1) - o1
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dx = dist * math.cos(theta)
    dy = dist * math.sin(theta)
    do = o2 - o1
    return dx, dy, do


# --------------------------------------------------------------------------- #
# Main planner / mapper
# --------------------------------------------------------------------------- #
class SharedMappingPlanner:
    def __init__(self, cfg, num_agents: int, device: str = "cuda"):
        self.cfg = cfg
        self.n = num_agents
        self.device = device if torch.cuda.is_available() else "cpu"

        # Build a SimpleNamespace that matches Semantic_Mapping's expected `args`.
        # MAP_* keys come from cfg.MAP; sensor params from SIMULATOR.
        sim_cfg = cfg.SIMULATOR if hasattr(cfg, "SIMULATOR") else cfg.TASK_CONFIG.SIMULATOR
        rgb = sim_cfg.RGB_SENSOR
        depth = sim_cfg.DEPTH_SENSOR
        map_cfg = cfg.MAP
        agent_height_m = float(depth.POSITION[1])  # camera height ~= 0.88m

        self._cam_h, self._cam_w = int(rgb.HEIGHT), int(rgb.WIDTH)
        cfg_w, cfg_h = int(map_cfg.FRAME_WIDTH), int(map_cfg.FRAME_HEIGHT)
        # Stretch is 360x640 (9:16). Default FRAME 160x120 is 4:3; anamorphic
        # resize fills rooms as obstacles. Keep integer stride so camera_matrix
        # matches the pixels (360x640 / 4 → 90x160).
        if self._cam_w * cfg_h != self._cam_h * cfg_w:
            ds = 1
            for cand in (4, 2, 8, 5, 3):
                if self._cam_w % cand == 0 and self._cam_h % cand == 0:
                    fw, fh = self._cam_w // cand, self._cam_h // cand
                    if 80 <= min(fw, fh) <= 200:
                        ds = cand
                        cfg_w, cfg_h = fw, fh
                        break
        else:
            ds = max(1, self._cam_w // max(1, cfg_w))
        args = SimpleNamespace(
            HFOV=float(rgb.HFOV),
            MIN_Z=float(map_cfg.MIN_Z),
            DEVICE=self.device,
            DU_SCALE=int(map_cfg.DU_SCALE),
            VISUALIZE=False,
            FRAME_WIDTH=int(cfg_w),
            FRAME_HEIGHT=int(cfg_h),
            VISION_RANGE=int(map_cfg.VISION_RANGE),
            MAP_RESOLUTION=int(map_cfg.MAP_RESOLUTION),
            PRINT_IMAGES=False,
            NUM_ENVIRONMENTS=num_agents,
            GLOBAL_DOWNSCALING=int(map_cfg.GLOBAL_DOWNSCALING),
            CAT_PRED_THRESHOLD=float(map_cfg.CAT_PRED_THRESHOLD),
            EXP_PRED_THRESHOLD=float(map_cfg.EXP_PRED_THRESHOLD),
            MAP_PRED_THRESHOLD=float(map_cfg.MAP_PRED_THRESHOLD),
            MAP_SIZE_CM=int(map_cfg.MAP_SIZE_CM),
            CENTER_RESET_STEPS=int(map_cfg.CENTER_RESET_STEPS),
            MAX_SEM_CATEGORIES=int(map_cfg.MAX_SEM_CATEGORIES),
            AGENT_HEIGHT=agent_height_m,
        )
        self.mapping = Semantic_Mapping(args).to(self.device).eval()
        self.resolution = int(map_cfg.MAP_RESOLUTION)         # cm/cell
        # Robot safety radius (m) used to dilate KNOWN obstacles before FMM so
        # paths keep a margin from walls instead of hugging / clipping them.
        self.SAFETY_RADIUS_M = 0.18
        self.map_size_cm = int(map_cfg.MAP_SIZE_CM)
        self.map_size_cells = self.map_size_cm // self.resolution
        self._dmin = float(depth.MIN_DEPTH)
        self._dmax = float(depth.MAX_DEPTH)
        self._frame_h, self._frame_w = int(cfg_h), int(cfg_w)
        self._ds = int(ds)

        self.states: List[AgentMapState] = [AgentMapState(agent_id=i) for i in range(num_agents)]
        self._full_map_np: Optional[np.ndarray] = None  # (N, C, H, W) numpy
        self._traversable: Optional[np.ndarray] = None  # union (H, W) bool
        self._step_count = 0
        # `debug_nav_remaining` is decremented each `next_action_for` call;
        # while >0 the planner logs per-agent FMM bookkeeping (habitat yaw,
        # mapping heading, target cell, stg cell, chosen action). Reset per
        # episode via `begin_episode`.
        self.debug_nav_budget: int = 0
        self._debug_nav_remaining: int = 0
        # +1 last TURN_LEFT, -1 last TURN_RIGHT. Stops FMM STG jitter
        # flipping left/right in place.
        self._last_turn_sign: List[int] = [0] * num_agents
        self._consec_turns: List[int] = [0] * num_agents

    # ---------------------- preprocessing ---------------------- #
    def _preprocess_one(self, obs: Dict) -> np.ndarray:
        """RGB(0..255) + Depth(cm scaled like lavira) → (4, frame_h, frame_w)."""
        rgb = obs["rgb"].astype(np.uint8)        # (H, W, 3)
        rgb = rgb[:, :, ::-1]                    # → BGR (lavira preprocesses BGR)
        depth_norm = obs["depth"].astype(np.float32)
        if depth_norm.ndim == 3:
            depth_norm = depth_norm[..., 0]
        # Mirror lavira's _preprocess_depth: invalid → 1.0, then scale to cm.
        depth = depth_norm.copy()
        depth[depth == 0.0] = 1.0
        depth[depth > 0.99] = 0.0
        mask = depth == 0
        depth[mask] = 1.0
        depth = self._dmin * 100.0 + depth * (self._dmax - self._dmin) * 100.0  # → cm

        if self._ds != 1:
            rgb = rgb[self._ds // 2 :: self._ds, self._ds // 2 :: self._ds]
            depth = depth[self._ds // 2 :: self._ds, self._ds // 2 :: self._ds]
        rgb = rgb[: self._frame_h, : self._frame_w]
        depth = depth[: self._frame_h, : self._frame_w]
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        depth = depth[None, ...].astype(np.float32)
        # mapping.forward requires at least 1 semantic channel (else the
        # AvgPool2d on obs[:, 4:] sees a 0-channel tensor and crashes).
        sem_dummy = np.zeros((1, depth.shape[1], depth.shape[2]), dtype=np.float32)
        return np.concatenate([rgb, depth, sem_dummy], axis=0)   # (5, h, w)

    # ---------------------- episode lifecycle ---------------------- #
    def begin_episode(self, sim) -> None:
        # Reset all internal map tensors.
        self.mapping.init_map_and_pose(num_detected_classes=0)
        self.mapping.reset()
        self._step_count = 0
        self._debug_nav_remaining = int(self.debug_nav_budget)
        self._last_turn_sign = [0] * self.n
        self._consec_turns = [0] * self.n
        # Per-env visited-trajectory canvases. lavira's Semantic_Mapping uses a
        # single shared `self.visited_vis` array — when we render env i then
        # env j, env j's line write leaks into env i's slice next time. We keep
        # one canvas per env and swap into the mapping module before each
        # render to keep each agent's trail isolated.
        self._vv_per_env: List[np.ndarray] = [
            np.zeros_like(self.mapping.visited_vis) for _ in range(self.n)
        ]
        for i in range(self.n):
            state = sim.get_agent_state(i)
            pos = np.asarray(state.position, dtype=np.float32)
            self.states[i].last_world_pos = pos
            init_sim_loc = _agent_sim_location(sim, i)
            self.states[i].last_sim_loc = init_sim_loc
            self.states[i].world_origin_xz = (pos[0], pos[2])
            # `_agent_sim_location[2]` IS the habitat yaw expressed via the
            # same lavira ZYZ-Euler heuristic that drives mapping integration,
            # so we snapshot it directly (no separate quat-to-yaw needed).
            self.states[i].initial_habitat_yaw_rad = float(init_sim_loc[2])

    # ---------------------- per-step update ---------------------- #
    def step(self, sim, obs_list: List[Dict]) -> None:
        # 1) per-agent pose delta (mapping coords, lavira convention)
        poses = np.zeros((self.n, 3), dtype=np.float32)
        for i in range(self.n):
            curr = _agent_sim_location(sim, i)
            dx, dy, do = _rel_pose_change(curr, self.states[i].last_sim_loc)
            poses[i] = [dx, dy, do]
            self.states[i].last_sim_loc = curr
            self.states[i].last_world_pos = np.asarray(
                sim.get_agent_state(i).position, dtype=np.float32
            )

        # 2) batch obs (force float32 throughout — mismatched dtypes silently
        # bubble into mat1/mat2 mismatch inside grid_sample.)
        states_np = np.stack([self._preprocess_one(obs) for obs in obs_list], axis=0)
        if self._step_count == 0:
            print(f"[ma-planner][dbg] step 0: states_np shape={states_np.shape} "
                  f"depth_min={states_np[:, 3].min():.1f} depth_max={states_np[:, 3].max():.1f} "
                  f"poses={poses.tolist()}")
        batch_obs = torch.from_numpy(states_np.astype(np.float32)).to(self.device)
        pose_obs = torch.from_numpy(poses.astype(np.float32)).to(self.device)

        # 3) forward + update
        try:
            with torch.no_grad():
                self.mapping(batch_obs, pose_obs, step=self._step_count)
                full_map, _, _ = self.mapping.update_map(
                    self._step_count, detected_classes=None, current_episode_id=0
                )
            self.mapping.one_step_full_map.fill_(0.0)
            self.mapping.one_step_local_map.fill_(0.0)
            self._full_map_np = full_map  # (N, C, H, W)
            self._compute_traversable()
        except Exception as e:
            import traceback
            print(f"[ma-planner][warn] mapping update failed: {e!r}")
            traceback.print_exc()
        self._step_count += 1

    def _compute_traversable(self) -> None:
        """Union obstacle/explored across N envs into a single traversable mask."""
        if self._full_map_np is None:
            return
        obs_union = np.zeros(self.mapping.map_shape, dtype=bool)
        exp_union = np.zeros(self.mapping.map_shape, dtype=bool)
        for i in range(self.n):
            obs_union |= (np.rint(self._full_map_np[i, 0]) == 1)
            exp_union |= (np.rint(self._full_map_np[i, 1]) == 1)
        # Traversable = explored AND not obstacle
        self._traversable = (exp_union & (~obs_union)).astype(np.uint8)

    def comm_gated_layers(self, ego: int, agent_positions, r_comm: float = 5.0) -> Optional[dict]:
        """DM3-style decentralized map: (obstacle, explored) in EGO agent's frame,
        merging only agents within ``r_comm`` metres of ego (point-to-point comm),
        instead of the global union. Returns masks + cell<->world transforms.

        ``agent_positions`` = list of world (x,y,z) per agent.
        """
        if self._full_map_np is None or self.states[ego].world_origin_xz is None:
            return None
        ego_pos = np.asarray(agent_positions[ego], dtype=np.float32)
        in_comm = []
        for j in range(self.n):
            if self.states[j].world_origin_xz is None:
                continue
            d = float(np.linalg.norm(np.asarray(agent_positions[j], dtype=np.float32) - ego_pos))
            if j == ego or d <= r_comm:
                in_comm.append(j)
        return self._merge_into_frame(ego, in_comm)

    def merged_layers(self) -> Optional[dict]:
        """Merged (obstacle, explored) masks in agent_0's frame + cell<->world
        transforms — same rotation+origin-translation as
        `render_merged_habitat_map`, so frontier cells convert to world coords
        consistently with what is drawn. Returns None until the map exists.
        """
        return self._merge_into_frame(0, list(range(self.n)))

    def _merge_into_frame(self, ego: int, agents: list) -> Optional[dict]:
        """Union the given agents' ego-maps into `ego`'s frame (rotation + origin
        translation), returning masks + cell<->world transforms."""
        if self._full_map_np is None or self.states[ego].world_origin_xz is None:
            return None
        canvas = int(self.map_size_cells)
        map_center = canvas // 2
        res_m = float(self.resolution) / 100.0
        yaw0 = float(self.states[ego].initial_habitat_yaw_rad)
        cx, cz = self.states[ego].world_origin_xz
        s0, c0 = math.sin(yaw0), math.cos(yaw0)

        obstacle = np.zeros((canvas, canvas), dtype=np.uint8)
        explored = np.zeros((canvas, canvas), dtype=np.uint8)
        center_pt = (float(map_center), float(map_center))
        for i in agents:
            oi = self.states[i].world_origin_xz
            if oi is None:
                continue
            obs_i = (np.rint(self._full_map_np[i, 0]) == 1).astype(np.uint8)
            exp_i = (np.rint(self._full_map_np[i, 1]) == 1).astype(np.uint8)
            if i == ego:
                obstacle |= obs_i
                explored |= exp_i
                continue
            yaw_i = float(self.states[i].initial_habitat_yaw_rad)
            M = cv2.getRotationMatrix2D(center_pt, math.degrees(yaw0 - yaw_i), 1.0)
            dwx, dwz = oi[0] - cx, oi[1] - cz
            M[0, 2] += (-s0 * dwx - c0 * dwz) / res_m
            M[1, 2] += (-c0 * dwx + s0 * dwz) / res_m
            obstacle |= cv2.warpAffine(obs_i, M, (canvas, canvas), flags=cv2.INTER_NEAREST, borderValue=0)
            explored |= cv2.warpAffine(exp_i, M, (canvas, canvas), flags=cv2.INTER_NEAREST, borderValue=0)

        def cell_to_world(col, row):
            d_mx, d_my = (col - map_center) * res_m, (row - map_center) * res_m
            return (cx + (-s0 * d_mx - c0 * d_my), cz + (-c0 * d_mx + s0 * d_my))

        def world_to_cell(wx, wz):
            dwx, dwz = wx - cx, wz - cz
            return (int((-s0 * dwx - c0 * dwz) / res_m + map_center),
                    int((-c0 * dwx + s0 * dwz) / res_m + map_center))

        def world_to_cell_f(wx, wz):
            dwx, dwz = float(wx) - cx, float(wz) - cz
            return ((-s0 * dwx - c0 * dwz) / res_m + map_center,
                    (-c0 * dwx + s0 * dwz) / res_m + map_center)

        return {"obstacle": obstacle, "explored": explored,
                "cell_to_world": cell_to_world, "world_to_cell": world_to_cell,
                "world_to_cell_f": world_to_cell_f,
                "yaw0": yaw0, "origin_xz": (cx, cz),
                "map_center": map_center, "canvas": canvas,
                "cell_m": res_m}

    def _merged_layers_OLD(self) -> Optional[dict]:
        """(superseded by _merge_into_frame) Returns None until the map exists."""
        if self._full_map_np is None or self.states[0].world_origin_xz is None:
            return None
        canvas = int(self.map_size_cells)
        map_center = canvas // 2
        res_m = float(self.resolution) / 100.0
        yaw0 = float(self.states[0].initial_habitat_yaw_rad)
        cx, cz = self.states[0].world_origin_xz
        s0, c0 = math.sin(yaw0), math.cos(yaw0)

        obstacle = (np.rint(self._full_map_np[0, 0]) == 1).astype(np.uint8)
        explored = (np.rint(self._full_map_np[0, 1]) == 1).astype(np.uint8)
        center_pt = (float(map_center), float(map_center))
        for i in range(1, self.n):
            yaw_i = float(self.states[i].initial_habitat_yaw_rad)
            M = cv2.getRotationMatrix2D(center_pt, math.degrees(yaw0 - yaw_i), 1.0)
            oi = self.states[i].world_origin_xz
            if oi is not None:
                dwx, dwz = oi[0] - cx, oi[1] - cz
                M[0, 2] += (-s0 * dwx - c0 * dwz) / res_m
                M[1, 2] += (-c0 * dwx + s0 * dwz) / res_m
            obs_i = (np.rint(self._full_map_np[i, 0]) == 1).astype(np.uint8)
            exp_i = (np.rint(self._full_map_np[i, 1]) == 1).astype(np.uint8)
            obstacle |= cv2.warpAffine(obs_i, M, (canvas, canvas), flags=cv2.INTER_NEAREST, borderValue=0)
            explored |= cv2.warpAffine(exp_i, M, (canvas, canvas), flags=cv2.INTER_NEAREST, borderValue=0)

        def cell_to_world(col, row):
            d_mx, d_my = (col - map_center) * res_m, (row - map_center) * res_m
            return (cx + (-s0 * d_mx - c0 * d_my), cz + (-c0 * d_mx + s0 * d_my))

        def world_to_cell(wx, wz):
            dwx, dwz = wx - cx, wz - cz
            return (int((-s0 * dwx - c0 * dwz) / res_m + map_center),
                    int((-c0 * dwx + s0 * dwz) / res_m + map_center))

        return {"obstacle": obstacle, "explored": explored,
                "cell_to_world": cell_to_world, "world_to_cell": world_to_cell,
                "map_center": map_center, "canvas": canvas}

    # ---------------------- coordinate helpers ---------------------- #
    def initial_yaw_rad(self, agent_id: int) -> float:
        """Habitat yaw at episode start for `agent_id` (rad). Used by
        `ma_decision` to bridge mapping → habitat coords correctly when
        the agent starts at non-zero habitat orientation."""
        return float(self.states[agent_id].initial_habitat_yaw_rad)

    def _world_to_env_map_cell(self, agent_id: int, world_xz: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Convert habitat world (x, z) → env_<agent_id>'s full-map cell.

        Lavira's `Semantic_Mapping` integrates relative pose deltas; the
        module's internal frame is anchored to the agent's INITIAL forward
        direction at episode start (mapping heading 0 ↔ habitat yaw_0). For
        a generic yaw_0 ≠ 0 the (−Wz, −Wx) flip is only the yaw_0 = 0 case;
        the full habitat→mapping rotation is the inverse of the rotation
        that takes mapping axes to world axes. We anchor on `mapping.state`
        (drift-aware) plus the per-agent yaw_0 snapshot.
        """
        if self.mapping is None or self.states[agent_id].last_world_pos is None:
            return None
        state = self.mapping.state[agent_id]
        agent_x_map = float(state[0])
        agent_y_map = float(state[1])
        agent_pos = self.states[agent_id].last_world_pos
        dx_habitat = float(world_xz[0]) - float(agent_pos[0])
        dz_habitat = float(world_xz[1]) - float(agent_pos[2])
        # Habitat → mapping frame delta (rotate by −yaw_0; mx, my are
        # the projections onto the agent's initial forward / right axes
        # expressed in (Wx, Wz)).
        yaw0 = float(self.states[agent_id].initial_habitat_yaw_rad)
        s0, c0 = math.sin(yaw0), math.cos(yaw0)
        dx_map = -dx_habitat * s0 - dz_habitat * c0
        dy_map = -dx_habitat * c0 + dz_habitat * s0
        target_x_map = agent_x_map + dx_map
        target_y_map = agent_y_map + dy_map
        row = int(target_y_map * 100.0 / self.resolution)
        col = int(target_x_map * 100.0 / self.resolution)
        if 0 <= row < self.map_size_cells and 0 <= col < self.map_size_cells:
            return row, col
        return None

    def agent_map_cell(self, agent_id: int) -> Optional[Tuple[int, int]]:
        """Agent cell on this env's occupancy map (row, col). Mapping pose only."""

        if self.mapping is None:
            return None
        x_map = float(self.mapping.state[agent_id, 0])
        y_map = float(self.mapping.state[agent_id, 1])
        row = int(y_map * 100.0 / self.resolution)
        col = int(x_map * 100.0 / self.resolution)
        if 0 <= row < self.map_size_cells and 0 <= col < self.map_size_cells:
            return row, col
        return None

    def map_euclidean_m(
        self,
        agent_id: int,
        target_cell_rc: Optional[Tuple[int, int]],
    ) -> Optional[float]:
        """Top-down map Euclidean metres between the agent and a target cell."""

        agent_rc = self.agent_map_cell(agent_id)
        if agent_rc is None or target_cell_rc is None:
            return None
        return map_cells_euclidean_m(agent_rc, target_cell_rc, float(self.resolution))

    def waypoint_reached_on_map(
        self,
        agent_id: int,
        target_cell_rc: Optional[Tuple[int, int]],
        threshold_m: float = WAYPOINT_REACH_M,
    ) -> bool:
        """True when map Euclidean is under single-agent LaViRA's 0.75 m arrival radius."""

        dist_m = self.map_euclidean_m(agent_id, target_cell_rc)
        return dist_m is not None and dist_m < float(threshold_m)

    # ---------------------- planner ---------------------- #
    @staticmethod
    def _angle_and_direction(a: np.ndarray, b: np.ndarray, turn_angle: float):
        """Verbatim port of `vlnce_baselines.utils.map_utils.angle_and_direction`."""
        unit_a = a / (np.linalg.norm(a) + 1e-5)
        unit_b = b / (np.linalg.norm(b) + 1e-5)
        cross_product = float(np.cross(unit_a, unit_b))
        dot_product = float(np.dot(unit_a, unit_b))
        angle = math.acos(max(-1.0, min(1.0, dot_product)))
        angle_deg = math.degrees(angle)
        if cross_product > 0 and angle_deg >= (turn_angle / 2 + 0.01):
            return angle_deg, 3   # TURN_RIGHT
        if cross_product < 0 and angle_deg >= (turn_angle / 2):
            return angle_deg, 2   # TURN_LEFT
        if cross_product == 0 and angle_deg == 180:
            return angle_deg, 3
        return angle_deg, 1   # MOVE_FORWARD

    @staticmethod
    def _stg_heading_action(agent_cell, stg_cell, heading_deg: float, turn_angle: float):
        """Same ndarray-frame heading test as single-agent LaViRA Policy.

        Mapping heading is negated and rotated 90° before comparing to the
        short-term-goal vector in (row, col) cells. The previous atan2(raw
        heading) form could stay past the turn threshold forever.
        """
        from vlnce_baselines.utils.map_utils import angle_to_vector

        position = np.array(
            [float(agent_cell[0]), float(agent_cell[1])], dtype=np.float64
        )
        waypoint = np.array([float(stg_cell[0]), float(stg_cell[1])], dtype=np.float64)
        waypoint_vector = waypoint - position
        if float(np.linalg.norm(waypoint_vector)) < 1e-6:
            return HabitatSimActions.STOP, 0.0
        heading = -float(heading_deg)
        rotation_matrix = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
        heading_vector = np.dot(rotation_matrix, angle_to_vector(heading))
        angle_deg, direction = SharedMappingPlanner._angle_and_direction(
            heading_vector, waypoint_vector, float(turn_angle)
        )
        if direction == 2:
            return HabitatSimActions.TURN_LEFT, float(angle_deg)
        if direction == 3:
            return HabitatSimActions.TURN_RIGHT, float(angle_deg)
        return HabitatSimActions.MOVE_FORWARD, float(angle_deg)

    def next_action_for(self, agent_id: int, target_world_xz: Tuple[float, float],
                         sim, target_cell_override: Optional[Tuple[int, int]] = None) -> int:
        """Compute next discrete habitat action for `agent_id` toward
        `target_world_xz` using FMM on the (per-agent) traversable map."""
        if self._full_map_np is None or self._traversable is None:
            return HabitatSimActions.MOVE_FORWARD

        if target_cell_override is not None:
            tgt_cell = target_cell_override
        else:
            tgt_cell = self._world_to_env_map_cell(agent_id, target_world_xz)
        if tgt_cell is None:
            self._last_turn_sign[agent_id] = 0
            return HabitatSimActions.MOVE_FORWARD
        if not (0 <= tgt_cell[0] < self.map_size_cells and 0 <= tgt_cell[1] < self.map_size_cells):
            self._last_turn_sign[agent_id] = 0
            return HabitatSimActions.MOVE_FORWARD

        # Use the per-agent env's local view for FMM, since coordinate frames
        # are env-local.
        try:
            full_map = self._full_map_np[agent_id]
            obstacles = (np.rint(full_map[0]) == 1).astype(np.uint8)

            # Agent's full-map cell from mapping state (drift-aware).
            x_map = float(self.mapping.state[agent_id, 0])
            y_map = float(self.mapping.state[agent_id, 1])
            heading_deg = float(self.mapping.state[agent_id, 2])
            col = int(x_map * 100.0 / self.resolution)
            row = int(y_map * 100.0 / self.resolution)
            agent_cell = (row, col)

            # Optimistic-but-safe traversable mask. Unknown cells stay
            # traversable (the strict explored∧¬obstacle mask degenerates FMM:
            # any target outside the small explored bubble becomes unreachable,
            # the distance field flattens, and `get_short_term_goal`'s argmin
            # tie-breaks into the nearest wall → stuck). But KNOWN obstacles are
            # dilated by the robot radius so FMM paths keep a margin instead of
            # hugging walls / clipping corners (the "pushing forward into a wall" failure).
            radius_cells = max(1, int(round(self.SAFETY_RADIUS_M * 100.0 / self.resolution)))
            if obstacles.any():
                k = 2 * radius_cells + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                obstacles_dil = cv2.dilate(obstacles, kernel, iterations=1)
            else:
                obstacles_dil = obstacles
            traversable = (~obstacles_dil.astype(bool)).astype(np.uint8)

            # Dilation can swallow the agent's own cell (it may sit within a
            # radius of a mapped wall) or the target cell — carve small free
            # disks around both so FMM always has a valid source and goal.
            def _carve(cell):
                r0 = max(0, cell[0] - radius_cells)
                r1 = min(self.map_size_cells, cell[0] + radius_cells + 1)
                c0 = max(0, cell[1] - radius_cells)
                c1 = min(self.map_size_cells, cell[1] + radius_cells + 1)
                traversable[r0:r1, c0:c1] = 1
            _carve(agent_cell)
            _carve(tgt_cell)

            if traversable.sum() < 4:
                self._last_turn_sign[agent_id] = 0
                return HabitatSimActions.MOVE_FORWARD
            planner = FMMPlanner(self.cfg, traversible=traversable, scale=1, step_size=5)
            planner.set_goal(np.array(tgt_cell))

            stg_x, stg_y, stop = planner.get_short_term_goal(
                np.array(agent_cell, dtype=np.float32), np.array(tgt_cell)
            )
            if stop:
                self._last_turn_sign[agent_id] = 0
                self._consec_turns[agent_id] = 0
                return HabitatSimActions.STOP
            turn = float(sim.habitat_config.TURN_ANGLE)
            chosen, rel = self._stg_heading_action(
                agent_cell, (stg_x, stg_y), heading_deg, turn
            )
            if chosen == HabitatSimActions.MOVE_FORWARD:
                self._last_turn_sign[agent_id] = 0
                self._consec_turns[agent_id] = 0
            else:
                desired = 1 if chosen == HabitatSimActions.TURN_LEFT else -1
                last = int(self._last_turn_sign[agent_id])
                # STG flicker across the agent flips the sign every step.
                if last != 0 and desired != last and abs(rel) < 90.0:
                    chosen = HabitatSimActions.MOVE_FORWARD
                    self._last_turn_sign[agent_id] = 0
                    self._consec_turns[agent_id] = 0
                else:
                    self._last_turn_sign[agent_id] = desired
                    self._consec_turns[agent_id] = int(self._consec_turns[agent_id]) + 1
                    n_turns = int(self._consec_turns[agent_id])
                    # 16 in-place turns → ask the runner to replan a farther target.
                    if n_turns >= 16:
                        self._last_turn_sign[agent_id] = 0
                        self._consec_turns[agent_id] = 0
                        return HabitatSimActions.STOP
                    # 8 turns = 240° with no forward: take one step to break
                    # the spin — unless already inside the arrival disk.
                    if n_turns >= 8:
                        dist_m = math.hypot(
                            float(agent_cell[0] - tgt_cell[0]),
                            float(agent_cell[1] - tgt_cell[1]),
                        ) * float(self.resolution) / 100.0
                        if dist_m < WAYPOINT_REACH_M:
                            self._last_turn_sign[agent_id] = 0
                            self._consec_turns[agent_id] = 0
                            return HabitatSimActions.STOP
                        chosen = HabitatSimActions.MOVE_FORWARD
                        self._last_turn_sign[agent_id] = 0
                        self._consec_turns[agent_id] = 0

            if self._debug_nav_remaining > 0:
                # Per-agent FMM diagnostic. Compare mapping heading against
                # the live habitat yaw so we can spot frame-of-reference drift.
                import quaternion as _quat  # noqa: WPS433
                rot = sim.get_agent_state(agent_id).rotation
                if hasattr(rot, "w"):
                    w, x_q, y_q, z_q = rot.w, rot.x, rot.y, rot.z
                else:
                    w, x_q, y_q, z_q = rot
                yaw_hab_deg = math.degrees(math.atan2(
                    2.0 * (w * y_q + z_q * x_q),
                    1.0 - 2.0 * (x_q * x_q + y_q * y_q),
                ))
                act_name = {
                    HabitatSimActions.TURN_LEFT: "LEFT",
                    HabitatSimActions.TURN_RIGHT: "RIGHT",
                    HabitatSimActions.MOVE_FORWARD: "FWD",
                    HabitatSimActions.STOP: "STOP",
                }.get(chosen, str(chosen))
                print(
                    f"[nav-dbg] step={self._step_count:4d} a={agent_id} "
                    f"hab_yaw={yaw_hab_deg:+6.1f} map_hdg={heading_deg:+6.1f} "
                    f"agent_rc={agent_cell} tgt_rc={tuple(tgt_cell)} stg_rc=({stg_x},{stg_y}) "
                    f"rel={rel:+6.1f} → {act_name}"
                )
                self._debug_nav_remaining -= 1
            return chosen
        except Exception as e:
            print(f"[ma-planner][warn] FMM step failed agent_{agent_id}: {e!r}")
            self._last_turn_sign[agent_id] = 0
            self._consec_turns[agent_id] = 0
            return HabitatSimActions.MOVE_FORWARD

    # ---------------------- rendering ---------------------- #
    # ------------------------------------------------------------------ #
    # Full-map renderer (lavira palette, but on the full 480×480 grid so
    # off-window agents are also visible).
    # ------------------------------------------------------------------ #
    def render_merged_habitat_map(
        self,
        sim,
        *,
        agent_paths_world: Optional[List[List[np.ndarray]]] = None,
        agent_targets_world: Optional[List[Optional[np.ndarray]]] = None,
        goals_world: Optional[List[np.ndarray]] = None,
        frontiers_world: Optional[List[Tuple[float, float]]] = None,
        agent_frontiers_world: Optional[List[Optional[np.ndarray]]] = None,
        found_markers: Optional[List[Dict]] = None,
        lease_markers: Optional[List[Dict]] = None,
        found_goal_mask: Optional[List[bool]] = None,
        output_size: Tuple[int, int] = (960, 960),
    ) -> Optional[np.ndarray]:
        """Merged map rendered on the **full 480×480 mapping canvas** (no
        local-window crop), using lavira's color palette and arrow shape.

        Lavira's `create_vlm_map_from_state` crops to `local_map` whose
        window `lmb = [gx1, gx2, gy1, gy2]` is recentered every
        `CENTER_RESET_STEPS` (=25) ticks — that recenter causes the
        rendered position of agent_1 and the drawn trails to JUMP on the
        canvas at every recenter point. Working on `full_map` keeps the
        view anchored to agent_0's mapping origin for the whole episode,
        eliminating the jumps.

        Layout:
            - Background: lavira white. Explored = light gray, obstacle = black.
            - Agent_0: lavira-red filled triangle + center circle (lavira style).
            - Agent_i>0: lavira-blue filled triangle + center circle (same style).
            - Dashed trails per agent (same identity colors as the RGB borders).
            - Per-agent current target: owner-colored TILTED_CROSS (X).
            - FOUND markers: green F# disks + category at verified goal poses.
            - LEASE markers: orange numbered disks L# at lease poses.
        """
        if self.mapping is None or self._full_map_np is None:
            return None
        try:
            # lavira palette (BGR).
            from vlnce_baselines.multi.ma_visualizer import (
                agent_bgr,
                crop_map_ink,
                draw_dashed_polyline,
                draw_nesw_compass,
                polyline_in_bounds_segments,
                snapshot_world_xyz,
            )
            COLOR_BG = (255, 255, 255)
            COLOR_EXPLORED = (220, 220, 220)
            COLOR_OBSTACLE = (0, 0, 0)

            canvas_size = int(self.map_size_cells)
            res_m = float(self.resolution) / 100.0         # 0.05
            map_center = canvas_size // 2
            yaw0 = float(self.states[0].initial_habitat_yaw_rad)
            origin0 = self.states[0].world_origin_xz
            if origin0 is None:
                return None
            center_wx, center_wz = origin0
            sin0, cos0 = math.sin(yaw0), math.cos(yaw0)

            # --- 1. Union of obstacle/explored in agent_0's full-map frame.
            obstacle = (np.rint(self._full_map_np[0, 0]) == 1).astype(np.uint8)
            explored = (np.rint(self._full_map_np[0, 1]) == 1).astype(np.uint8)
            center_pt = (float(map_center), float(map_center))
            for i in range(1, self.n):
                yaw_i = float(self.states[i].initial_habitat_yaw_rad)
                d_yaw_deg = math.degrees(yaw0 - yaw_i)
                M = cv2.getRotationMatrix2D(center_pt, d_yaw_deg, 1.0)
                # Distributed starts: agent_i's ego-map is centered at ITS OWN
                # spawn origin, so a rotation-only merge leaves its content at
                # agent_0's center. Translate so agent_i's origin lands at its
                # true location in agent_0's frame. Exactly zero when origins
                # coincide (original shared-start case) -> backward-compatible.
                oi = self.states[i].world_origin_xz
                if oi is not None:
                    dwx_i = oi[0] - center_wx
                    dwz_i = oi[1] - center_wz
                    M[0, 2] += (-sin0 * dwx_i - cos0 * dwz_i) / res_m
                    M[1, 2] += (-cos0 * dwx_i + sin0 * dwz_i) / res_m
                obs_i = (np.rint(self._full_map_np[i, 0]) == 1).astype(np.uint8)
                exp_i = (np.rint(self._full_map_np[i, 1]) == 1).astype(np.uint8)
                obs_rot = cv2.warpAffine(
                    obs_i, M, (canvas_size, canvas_size),
                    flags=cv2.INTER_NEAREST, borderValue=0,
                )
                exp_rot = cv2.warpAffine(
                    exp_i, M, (canvas_size, canvas_size),
                    flags=cv2.INTER_NEAREST, borderValue=0,
                )
                obstacle |= obs_rot
                explored |= exp_rot

            canvas = np.full((canvas_size, canvas_size, 3), COLOR_BG, dtype=np.uint8)
            canvas[explored == 1] = COLOR_EXPLORED
            canvas[obstacle == 1] = COLOR_OBSTACLE
            # Upscale occupancy first so dashed trails / markers stay continuous
            # (1px overlays on 480 + INTER_NEAREST used to look gappy).
            if canvas.shape[0] != output_size[1] or canvas.shape[1] != output_size[0]:
                canvas = cv2.resize(canvas, output_size, interpolation=cv2.INTER_NEAREST)

            # --- 2. Coord helpers (habitat → full_map cell → output px).
            scale_x = output_size[0] / float(canvas_size)
            scale_y = output_size[1] / float(canvas_size)

            def _w2px(wxz, clamp: bool = False):
                dwx = float(wxz[0]) - center_wx
                dwz = float(wxz[1]) - center_wz
                d_mx = -sin0 * dwx - cos0 * dwz
                d_my = -cos0 * dwx + sin0 * dwz
                cc = d_mx / res_m + map_center
                cr = d_my / res_m + map_center
                if clamp:
                    cc = min(max(cc, 0.0), canvas_size - 1e-3)
                    cr = min(max(cr, 0.0), canvas_size - 1e-3)
                elif not (0 <= cr < canvas_size and 0 <= cc < canvas_size):
                    return None
                return (int(round(cc * scale_x)), int(round(cr * scale_y)))

            # --- 3. Historical trails — every recorded sample, dashed.
            # Append live pose so the trail reaches the current agent marker.
            def _extend_with_live(ai, path):
                pts = [snapshot_world_xyz(p) for p in (path or [])]
                try:
                    pos = snapshot_world_xyz(sim.get_agent_state(ai).position)
                    if not pts:
                        pts.append(pos)
                    else:
                        last = pts[-1]
                        if math.hypot(float(pos[0]) - float(last[0]),
                                      float(pos[2]) - float(last[2])) > 1e-4:
                            pts.append(pos)
                except Exception:
                    pass
                return pts

            dash_px = max(6.0, 6.0 * scale_x)
            gap_px = max(4.0, 4.0 * scale_x)
            trail_th = max(2, int(round(2 * scale_x)))
            if agent_paths_world:
                for ai, path in enumerate(agent_paths_world):
                    live_path = _extend_with_live(ai, path)
                    if len(live_path) < 2:
                        continue
                    color = agent_bgr(ai)
                    for seg in polyline_in_bounds_segments(live_path, _w2px):
                        # Thin solid underlay so the walked polyline is visible
                        # even when dashes skip; dashes follow the same samples.
                        if len(seg) >= 2:
                            cv2.polylines(
                                canvas,
                                [np.array(seg, dtype=np.int32)],
                                False,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                        draw_dashed_polyline(
                            canvas, seg, color,
                            thickness=trail_th, dash_px=dash_px, gap_px=gap_px,
                        )

            # --- 4. Per-agent current target — colored X (red for agent_0,
            # blue for agent_1) so it's clear which X belongs to whom.
            marker_sz = max(15, int(round(15 * scale_x)))
            if agent_targets_world:
                for ai, tgt in enumerate(agent_targets_world):
                    if tgt is None:
                        continue
                    pt = _w2px((float(tgt[0]), float(tgt[2])))
                    if pt is not None:
                        color = agent_bgr(ai)
                        cv2.drawMarker(canvas, pt, color,
                                       cv2.MARKER_TILTED_CROSS, marker_sz, 3)

            # --- 4b. Frontiers (small magenta dots) + each agent's ASSIGNED
            # frontier highlighted in that agent's color (ring + connector line),
            # so the division of labour is visible.
            COLOR_FRONTIER = (255, 0, 255)     # magenta
            if frontiers_world:
                for fw in frontiers_world:
                    pt = _w2px((float(fw[0]), float(fw[1])))
                    if pt is not None:
                        cv2.circle(canvas, pt, max(3, int(round(3 * scale_x))),
                                   COLOR_FRONTIER, -1)
            if agent_frontiers_world:
                for ai, fw in enumerate(agent_frontiers_world):
                    if fw is None:
                        continue
                    pt = _w2px((float(fw[0]), float(fw[2])))
                    if pt is None:
                        continue
                    color = agent_bgr(ai)
                    cv2.circle(canvas, pt, max(8, int(round(8 * scale_x))), color, 2)
                    ap = None
                    try:
                        apos = np.asarray(sim.get_agent_state(ai).position, dtype=np.float32)
                        ap = _w2px((float(apos[0]), float(apos[2])))
                    except Exception:
                        ap = None
                    if ap is not None:
                        cv2.line(canvas, ap, pt, color, 1)        # agent -> its frontier

            # --- 5. Live per-agent arrows (lavira triangle style).
            def _draw_lavira_arrow(img, center, map_hdg_rad, color):
                arrow_length = max(15, int(round(15 * scale_x)))
                ax, ay = center
                a = map_hdg_rad
                p_tip = (int(ax + arrow_length * math.cos(a)),
                         int(ay + arrow_length * math.sin(a)))
                p_left = (int(ax + (arrow_length / 2) * math.cos(a + math.pi * 5 / 6)),
                          int(ay + (arrow_length / 2) * math.sin(a + math.pi * 5 / 6)))
                p_right = (int(ax + (arrow_length / 2) * math.cos(a - math.pi * 5 / 6)),
                           int(ay + (arrow_length / 2) * math.sin(a - math.pi * 5 / 6)))
                cv2.fillPoly(img, [np.array([[p_tip, p_left, p_right]], np.int32)], color)
                cv2.circle(img, (ax, ay), max(8, int(round(8 * scale_x))), color, -1)

            for i in range(self.n):
                try:
                    state_i = sim.get_agent_state(i)
                except Exception:
                    continue
                pos = np.asarray(state_i.position, dtype=np.float32)
                rot = state_i.rotation
                if hasattr(rot, "w"):
                    w, xq, yq, zq = rot.w, rot.x, rot.y, rot.z
                else:
                    w, xq, yq, zq = rot
                yaw_live = math.atan2(2.0 * (w * yq + zq * xq),
                                       1.0 - 2.0 * (xq * xq + yq * yq))
                map_hdg = yaw_live - yaw0
                pt = _w2px((float(pos[0]), float(pos[2])), clamp=True)
                if pt is None:
                    continue
                color = agent_bgr(i)
                _draw_lavira_arrow(canvas, pt, map_hdg, color)
                # Small numeric agent id near arrow for multi-agent LA maps.
                cv2.putText(
                    canvas,
                    str(i),
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            COLOR_LEASE = (0, 165, 255)
            found_r = max(12, int(round(12 * scale_x)))
            pending_found = []
            if found_markers:
                for mi, marker in enumerate(found_markers):
                    wxz = marker.get("world_xz") or marker.get("xz")
                    if wxz is None:
                        continue
                    pt = _w2px((float(wxz[0]), float(wxz[1])))
                    if pt is None:
                        continue
                    pending_found.append((pt, marker, mi))
            if lease_markers:
                for mi, marker in enumerate(lease_markers):
                    wxz = marker.get("world_xz") or marker.get("xz")
                    if wxz is None:
                        continue
                    pt = _w2px((float(wxz[0]), float(wxz[1])))
                    if pt is None:
                        continue
                    num = int(marker.get("number", mi + 1))
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(canvas, (x, y), found_r, COLOR_LEASE, -1)
                    cv2.circle(canvas, (x, y), found_r, (0, 0, 0), 2)
                    text = f"L{num}"
                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                    )
                    cv2.putText(
                        canvas, text, (x - tw // 2, y + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA,
                    )

            # Mapping ndarray +y is image-down. Lavira flips the canvas so a
            # Habitat TURN_LEFT stays left on the map (same as RGB). Without
            # this, the visual Y axis is inverted and RGB headings look reversed.
            canvas = np.ascontiguousarray(np.flipud(canvas))
            if pending_found:
                from vlnce_baselines.multi.ma_visualizer import draw_found_marker
                h_flip = int(canvas.shape[0])
                for (x, y), marker, mi in pending_found:
                    draw_found_marker(
                        canvas,
                        (int(x), int(h_flip - 1 - y)),
                        int(marker.get("number", mi + 1)),
                        color=(0, 180, 0),
                        radius=max(12, int(round(14 * scale_x))),
                        label=marker.get("label"),
                    )
            # Crop before the compass: otherwise the N/E/S/W overlay in the
            # corner keeps an 80 m white canvas and the house shrinks to a stamp.
            canvas = crop_map_ink(canvas, pad_px=24)
            canvas = np.ascontiguousarray(canvas)
            draw_nesw_compass(canvas)
            return canvas
        except Exception as e:
            import traceback
            print(f"[ma-planner][warn] merged map render failed: {e!r}")
            traceback.print_exc()
            return None

    def render_full_map(self,
                        primary_agent: int = 0,
                        output_size: Tuple[int, int] = (1024, 1024),
                        auto_crop: bool = True,
                        crop_pad_cells: int = 24,
                        overlay_other_agents: Optional[List[Tuple[np.ndarray, float, Tuple[int, int, int]]]] = None,
                        overlay_paths: Optional[List[Tuple[List[np.ndarray], Tuple[int, int, int]]]] = None,
                        overlay_goals: Optional[List[np.ndarray]] = None,
                        ) -> Optional[np.ndarray]:
        """Render `self.mapping.full_map[primary_agent]` directly (no local
        window crop). Returns a BGR image, palette identical to lavira's
        `create_vlm_map_from_state` (white/gray/black + agent arrow)."""
        if self.mapping is None or self._full_map_np is None:
            return None
        try:
            COLOR_BG = (255, 255, 255)
            COLOR_EXPLORED = (220, 220, 220)
            COLOR_OBSTACLE = (0, 0, 0)

            # Merge all envs' obstacle/explored into env `primary_agent`'s
            # map frame so the visualization shows EVERYONE's contribution.
            full_map_primary = self._full_map_np[primary_agent]
            H, W = full_map_primary.shape[1:]
            obstacle = (np.rint(full_map_primary[0]) == 1).copy()
            explored = (np.rint(full_map_primary[1]) == 1).copy()
            primary_origin_xz = self.states[primary_agent].world_origin_xz or (0.0, 0.0)
            res_cm = float(self.resolution)
            for i in range(self.n):
                if i == primary_agent or self._full_map_np is None:
                    continue
                origin_i_xz = self.states[i].world_origin_xz
                if origin_i_xz is None:
                    continue
                # Shift env_i's map by Δ so its mapping-frame aligns to primary's.
                # In mapping coords: row index → y_map*20; col index → x_map*20.
                # row(env_0) - row(env_i) = (-W_x_0 - (-W_x_i)) * 20 = (W_x_i - W_x_0) * 20 [×100/res]
                drow = int((origin_i_xz[0] - primary_origin_xz[0]) * 100.0 / res_cm)
                dcol = int((origin_i_xz[1] - primary_origin_xz[1]) * 100.0 / res_cm)
                obs_i = np.rint(self._full_map_np[i, 0]) == 1
                exp_i = np.rint(self._full_map_np[i, 1]) == 1
                # Shift via numpy roll-and-mask. Source pixel at (r, c) in env_i
                # maps to (r+drow, c+dcol) in primary frame. Equivalently, we
                # take env_i's array and paste it into primary's at offset.
                # Use a destination-region copy with clipping.
                src_r0 = max(0, -drow)
                src_r1 = min(H, H - drow)
                src_c0 = max(0, -dcol)
                src_c1 = min(W, W - dcol)
                dst_r0 = max(0, drow)
                dst_r1 = dst_r0 + (src_r1 - src_r0)
                dst_c0 = max(0, dcol)
                dst_c1 = dst_c0 + (src_c1 - src_c0)
                if src_r1 > src_r0 and src_c1 > src_c0:
                    obstacle[dst_r0:dst_r1, dst_c0:dst_c1] |= obs_i[src_r0:src_r1, src_c0:src_c1]
                    explored[dst_r0:dst_r1, dst_c0:dst_c1] |= exp_i[src_r0:src_r1, src_c0:src_c1]
            vis = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
            vis[explored] = COLOR_EXPLORED
            vis[obstacle] = COLOR_OBSTACLE

            # Draw primary agent arrow from mapping.state[primary_agent].
            start_x, start_y, start_o = float(self.mapping.state[primary_agent, 0]), \
                                          float(self.mapping.state[primary_agent, 1]), \
                                          float(self.mapping.state[primary_agent, 2])
            full_r = int(start_y * 100.0 / self.resolution)
            full_c = int(start_x * 100.0 / self.resolution)
            if 0 <= full_r < H and 0 <= full_c < W:
                # lavira's arrow uses `angle_corrected = -start_o (deg)` then radians.
                ang = math.radians(-start_o)
                arrow_length = 15
                p_center = (full_c, full_r)
                p_tip = (int(p_center[0] + arrow_length * math.cos(ang)),
                         int(p_center[1] - arrow_length * math.sin(ang)))
                p_left = (int(p_center[0] + (arrow_length / 2) * math.cos(ang + math.pi * 5 / 6)),
                          int(p_center[1] - (arrow_length / 2) * math.sin(ang + math.pi * 5 / 6)))
                p_right = (int(p_center[0] + (arrow_length / 2) * math.cos(ang - math.pi * 5 / 6)),
                           int(p_center[1] - (arrow_length / 2) * math.sin(ang - math.pi * 5 / 6)))
                cv2.fillPoly(vis, [np.array([[p_tip, p_left, p_right]], np.int32)], (0, 0, 255))
                cv2.circle(vis, p_center, 8, (0, 0, 255), -1)

            # Overlay world-positioned markers (goals / other agents / paths).
            init_xz = self.states[primary_agent].world_origin_xz or (0.0, 0.0)
            center_m = self.map_size_cm / 100.0 / 2.0
            init_x_map = -init_xz[1]
            init_y_map = -init_xz[0]

            def _w2full(world_xz):
                x_map = center_m + ((-world_xz[1]) - init_x_map)
                y_map = center_m + ((-world_xz[0]) - init_y_map)
                full_row = int(y_map * 100.0 / self.resolution)
                full_col = int(x_map * 100.0 / self.resolution)
                if 0 <= full_row < H and 0 <= full_col < W:
                    return (full_col, full_row)
                return None

            if overlay_goals:
                # Deduplicate goal markers that fall on (nearly) the same cell.
                seen_cells = set()
                for g in overlay_goals:
                    pt = _w2full((float(g[0]), float(g[2])))
                    if pt is None:
                        continue
                    key = (pt[0] // 6, pt[1] // 6)
                    if key in seen_cells:
                        continue
                    seen_cells.add(key)
                    cv2.drawMarker(vis, pt, (0, 255, 0),
                                   markerType=cv2.MARKER_TILTED_CROSS,
                                   markerSize=14, thickness=2)
            if overlay_paths:
                for path, color in overlay_paths:
                    if not path or len(path) < 2:
                        continue
                    pts = []
                    for p in path:
                        pt = _w2full((float(p[0]), float(p[2])))
                        if pt:
                            pts.append(pt)
                    if len(pts) >= 2:
                        cv2.polylines(vis, [np.array(pts, np.int32)], False,
                                      color, thickness=2, lineType=cv2.LINE_AA)
            if overlay_other_agents:
                for pos, yaw_rad, color in overlay_other_agents:
                    pt = _w2full((float(pos[0]), float(pos[2])))
                    if pt:
                        ax, ay = pt
                        # Use same mapping-frame arrow convention as primary agent.
                        # Other agent's "start_o" needs to be computed via
                        # _agent_sim_location for that agent (the orientation
                        # in mapping frame, in degrees).
                        # We don't have it here directly; approximate via yaw.
                        # For consistency with lavira's arrow rendering (which
                        # uses negate-degrees), pass radians directly with the
                        # same sign convention used for primary agent.
                        # In mapping coords, lavira's start_o is in degrees;
                        # arrow uses ang = -start_o (rad). We pass yaw_rad as
                        # the mapping-frame orientation (rad) so:
                        ang = -yaw_rad   # mirror primary's sign convention
                        L = 14
                        p_tip = (int(ax + L * math.cos(ang)),
                                 int(ay - L * math.sin(ang)))
                        p_left = (int(ax + (L / 2) * math.cos(ang + math.pi * 5 / 6)),
                                  int(ay - (L / 2) * math.sin(ang + math.pi * 5 / 6)))
                        p_right = (int(ax + (L / 2) * math.cos(ang - math.pi * 5 / 6)),
                                   int(ay - (L / 2) * math.sin(ang - math.pi * 5 / 6)))
                        cv2.fillPoly(vis, [np.array([[p_tip, p_left, p_right]], np.int32)], color)
                        cv2.circle(vis, (ax, ay), 6, color, -1)

            # lavira applies flipud before resizing — match that so the agent
            # orientation matches what the model would see.
            vis = np.flipud(vis)

            # Auto-crop to non-white bbox.
            if auto_crop:
                not_white = np.any(vis != 255, axis=-1)
                if not_white.any():
                    rows = np.where(not_white.any(axis=1))[0]
                    cols = np.where(not_white.any(axis=0))[0]
                    r0 = max(0, int(rows.min()) - crop_pad_cells)
                    r1 = min(vis.shape[0], int(rows.max()) + crop_pad_cells + 1)
                    c0 = max(0, int(cols.min()) - crop_pad_cells)
                    c1 = min(vis.shape[1], int(cols.max()) + crop_pad_cells + 1)
                    vis = vis[r0:r1, c0:c1]

            if vis.shape[0] != output_size[0] or vis.shape[1] != output_size[1]:
                vis = cv2.resize(vis, output_size, interpolation=cv2.INTER_NEAREST)
            return vis
        except Exception as e:
            print(f"[ma-planner][warn] render_full_map failed: {e!r}")
            return None

    def render_vlm_map(self,
                       primary_agent: int = 0,
                       visited_targets: Optional[List[Dict]] = None,
                       goal_xy_map: Optional[Tuple[float, float]] = None,
                       output_size: Tuple[int, int] = (1024, 1024),
                       auto_crop: bool = True,
                       crop_pad_cells: int = 20,
                       overlay_other_agents: Optional[List[Tuple[np.ndarray, float, Tuple[int, int, int]]]] = None,
                       overlay_paths: Optional[List[Tuple[List[np.ndarray], Tuple[int, int, int]]]] = None,
                       overlay_goals: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """Use lavira's `create_vlm_map_from_state` on env `primary_agent`.

        Lavira's renderer paints the full 480×480 (resized to output_size)
        with COLOR_BG=white for unexplored cells; that makes the image look
        empty in early steps. When `auto_crop=True` we crop to the bounding
        box of the explored / obstacle pixels (padded by `crop_pad_cells`).
        """
        try:
            goal_tensor = None
            if goal_xy_map is not None:
                goal_tensor = torch.tensor([goal_xy_map[0], goal_xy_map[1]], dtype=torch.float32)
            # Swap to env_primary's own visited_vis so the line lavira draws
            # into mapping.visited_vis stays scoped to this env.
            saved_vv = self.mapping.visited_vis
            if getattr(self, "_vv_per_env", None) and 0 <= primary_agent < len(self._vv_per_env):
                self.mapping.visited_vis = self._vv_per_env[primary_agent]
            try:
                img = self.mapping.create_vlm_map_from_state(
                    current_episode_id=0,
                    id=primary_agent,
                    goal=goal_tensor,
                    output_size=output_size,
                    visited_targets=visited_targets,
                    display_last=True,
                )
            finally:
                # Persist what was just drawn into this env's canvas, then
                # restore the shared array.
                if getattr(self, "_vv_per_env", None) and 0 <= primary_agent < len(self._vv_per_env):
                    self._vv_per_env[primary_agent] = self.mapping.visited_vis
                self.mapping.visited_vis = saved_vv
            if img is None:
                return None

            # Overlay other agents / paths / goals before cropping.
            # create_vlm_map_from_state outputs a `output_size`-sized image of
            # env `primary_agent`'s local map; coordinates are scaled by
            # (output_size / local_map_size). We map world → env_primary's
            # local-map cell via _world_to_env_map_cell (which uses full-map
            # gx1/gy1 derived from state[primary]).
            try:
                # Resolve local-frame extent so we can rescale world→local-cell
                # to the output_size image.
                local_w = int(self.mapping.local_w)
                local_h = int(self.mapping.local_h)
                if local_w > 0 and local_h > 0:
                    sx = output_size[1] / float(local_w)
                    sy = output_size[0] / float(local_h)
                else:
                    sx = sy = 1.0
                primary_origin = self.states[primary_agent].world_origin_xz
                gx1 = int(self.mapping.state[primary_agent, 3])
                gy1 = int(self.mapping.state[primary_agent, 5])

                # Lavira's mapping doesn't use absolute world coords; it
                # centers env_primary's full_pose at (MAP_SIZE_CM/2/100, ...)
                # = (12, 12) meters at episode start. To project a habitat-world
                # (x, z) into mapping coords we add the delta from env's start.
                init_xz = primary_origin if primary_origin is not None else (0.0, 0.0)
                center_m = self.map_size_cm / 100.0 / 2.0   # = 12.0 for default cfg
                init_x_map = -init_xz[1]  # = -world_z_init
                init_y_map = -init_xz[0]  # = -world_x_init

                def _world_to_outpx(world_xz):
                    # Mapping convention: x_map = -world_z, y_map = -world_x.
                    # Lavira indexes the local map as [row=y_map*20-gx1, col=x_map*20-gy1].
                    x_map = center_m + ((-world_xz[1]) - init_x_map)
                    y_map = center_m + ((-world_xz[0]) - init_y_map)
                    full_row = int(y_map * 100.0 / self.resolution)  # uses y_map!
                    full_col = int(x_map * 100.0 / self.resolution)  # uses x_map!
                    local_r = full_row - gx1
                    local_c = full_col - gy1
                    if 0 <= local_r < local_h and 0 <= local_c < local_w:
                        # Lavira applies np.flipud to vis_map before resizing.
                        out_x = int(local_c * sx)
                        out_y = int((local_h - 1 - local_r) * sy)
                        return out_x, out_y
                    return None

                # Draw goals (green X)
                if overlay_goals:
                    for g in overlay_goals:
                        pt = _world_to_outpx((float(g[0]), float(g[2])))
                        if pt:
                            cv2.drawMarker(img, pt, (0, 255, 0),
                                           markerType=cv2.MARKER_TILTED_CROSS,
                                           markerSize=18, thickness=3)
                # Draw other-agent paths
                if overlay_paths:
                    for path, color in overlay_paths:
                        if not path or len(path) < 2:
                            continue
                        pts = []
                        for p in path:
                            pt = _world_to_outpx((float(p[0]), float(p[2])))
                            if pt:
                                pts.append(pt)
                        if len(pts) >= 2:
                            cv2.polylines(img, [np.array(pts, np.int32)],
                                          False, color, thickness=3, lineType=cv2.LINE_AA)
                # Draw other-agent arrows
                if overlay_other_agents:
                    for pos, yaw_rad, color in overlay_other_agents:
                        pt = _world_to_outpx((float(pos[0]), float(pos[2])))
                        if pt:
                            ax, ay = pt
                            angle = math.pi / 2 + yaw_rad
                            length = 18
                            p_tip = (int(ax + length * math.cos(angle)),
                                     int(ay - length * math.sin(angle)))
                            p_left = (int(ax + (length / 2) * math.cos(angle + math.pi * 5 / 6)),
                                      int(ay - (length / 2) * math.sin(angle + math.pi * 5 / 6)))
                            p_right = (int(ax + (length / 2) * math.cos(angle - math.pi * 5 / 6)),
                                       int(ay - (length / 2) * math.sin(angle - math.pi * 5 / 6)))
                            cv2.fillPoly(img, [np.array([[p_tip, p_left, p_right]], np.int32)], color)
                            cv2.circle(img, (ax, ay), 6, color, -1)
            except Exception as e:
                print(f"[ma-planner][warn] overlay failed: {e!r}")
            if not auto_crop:
                return img
            # Detect non-white region (white = unexplored bg in lavira palette).
            # COLOR_BG = (255, 255, 255). Anything else means content.
            not_white = np.any(img != 255, axis=-1)
            if not not_white.any():
                return img
            rows = np.where(not_white.any(axis=1))[0]
            cols = np.where(not_white.any(axis=0))[0]
            r0 = max(0, int(rows.min()) - crop_pad_cells)
            r1 = min(img.shape[0], int(rows.max()) + crop_pad_cells + 1)
            c0 = max(0, int(cols.min()) - crop_pad_cells)
            c1 = min(img.shape[1], int(cols.max()) + crop_pad_cells + 1)
            return img[r0:r1, c0:c1]
        except Exception as e:
            print(f"[ma-planner][warn] render_vlm_map failed: {e!r}")
            return None


# --------------------------------------------------------------------------- #
# Helper (yaw extraction)
# --------------------------------------------------------------------------- #
def _yaw_from_quat(rotation) -> float:
    if hasattr(rotation, "w"):
        w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    else:
        w, x, y, z = rotation
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(siny_cosp, cosy_cosp))
