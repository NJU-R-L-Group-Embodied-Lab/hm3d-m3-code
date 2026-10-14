# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for processing depth images.
"""
from argparse import Namespace

import itertools
import numpy as np
import torch

import vlnce_baselines.utils.rotation_utils as ru
import torch

def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.) / 2.  # x camera
    zc = (height - 1.) / 2.  # z camera
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for _ in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * \
        Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * \
        Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis],
                          Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ


def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ


def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    for XYZ_cm in XYZ_cms:
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)

        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0,
                            Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        counts = np.reshape(count, [map_size, map_size, n_z_bins])

    counts = counts.reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return counts


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """
    Transform from pixel axis to camera axis.
    Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    # Y_t.shape = (batchsize, H, W)
    # grid_x.shape = grid_z.shape = (W, H)
    grid_x, grid_z = torch.meshgrid(torch.arange(Y_t.shape[-1]),  # grid_x range: [0, H - 1]
                                    torch.arange(Y_t.shape[-2] - 1, -1, -1))  # grid_z range: [W - 1, 0]
    grid_x = grid_x.transpose(1, 0).to(device)  # (H, W)
    grid_z = grid_z.transpose(1, 0).to(device)  # (H, W)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())  # (batchsize, H, W)
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())  # (batchsize, H, W)

    X_t = (grid_x[:, ::scale, ::scale] - camera_matrix.xc) * \
          Y_t[:, ::scale, ::scale] / camera_matrix.f
    Z_t = (grid_z[:, ::scale, ::scale] - camera_matrix.zc) * \
          Y_t[:, ::scale, ::scale] / camera_matrix.f

    XYZ = torch.stack(
        (X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.size()))

    return XYZ


def transform_camera_view_t(
        XYZ, sensor_height, camera_elevation_degree, device):
    """
    Transform from camera axis to world axis, height and elevation first
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = torch.matmul(XYZ.reshape(-1, 3), torch.from_numpy(R).transpose(1, 0).to(device)).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height

    return XYZ


def transform_pose_t(XYZ, current_pose, device):
    """
    Transform from camera axis to world axis, x,y,heading second
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = torch.matmul(XYZ.reshape(-1, 3), torch.from_numpy(R).transpose(1, 0).to(device)).reshape(XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]

    return XYZ


def splat_feat_nd(init_grid, feat, coords):
    """
    Args:
        init_grid: B X nF X W X H X D X ..
        feat: B X nF X nPt
        coords: B X nDims X nPt in [-1, 1]
    Returns:
        grid: B X nF X W X H X D X ..
    """
    wts_dim = []  # store weights
    pos_dim = []
    grid_dims = init_grid.shape[2:]  # [vr, vr, max_height - min_height]

    B = init_grid.shape[0]
    F = init_grid.shape[1]  # num_categories + 1

    n_dims = len(grid_dims)  # n_dims=3

    grid_flat = init_grid.view(B, F, -1)  # [bs, 17, 100*100*80]

    '''
    coords: XYZ_cm_std => [bs, 3, x*y]
    XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
    XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] - vision_range // 2.) / vision_range * 2.
    XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
    XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.) / (max_h - min_h) * 2.
    since XYZ_cm_std were normalized, so pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2 recovers XYZ_cm_std to unnormalized
    '''
    for d in range(n_dims):
        pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2  # [bs, 1, 19200=120*160]; pos has negative values
        pos_d = []
        wts_d = []

        for ix in [0, 1]:
            # when ix=0, round down; when ix=1, round up
            pos_ix = torch.floor(pos) + ix  # [bs, 19200]
            safe_ix = (pos_ix > 0) & (pos_ix < grid_dims[d])
            safe_ix = safe_ix.type(pos.dtype)

            # when round down: e.g. 1.75 -> 1.0; weight = 1 - abs(1.75 - 1.0) = 0.25;
            # when round up: e.g. 1.75 -> 1.0 + 1 = 2.0; weight = 1 - abs(1.75 - 2.0) = 0.75
            wts_ix = 1 - torch.abs(pos - pos_ix)

            wts_ix = wts_ix * safe_ix  # positions outside range have weight=0
            pos_ix = pos_ix * safe_ix  # positions outside range are all set to 0

            pos_d.append(pos_ix)
            wts_d.append(wts_ix)

        # len(pos_d)=2, len(pos_dim=3) pos_dim = [[[...],[...]],[],[]], pos_dim[0][0].shape=[bs, 1, 19200]
        # pos_dim[0][0]: x_floor; pos_dim[0][1]: x_ceil
        # pos_dim[1][0]: y_floor; pos_dim[1][1]: y_ceil
        # pos_dim[2][0]: z_floor; pos_dim[2][1]: z_ceil
        # actually pos_dim saves point clouds in vision range, which are truely usefull for subsequent steps
        pos_dim.append(pos_d)
        wts_dim.append(wts_d)

    l_ix = [[0, 1] for d in range(n_dims)]
    for ix_d in itertools.product(*l_ix):
        '''
        each dimension(x,y,z) should consider floor and ceiling
        ix_d:
        (0,0,0) => x floor, y floor, z floor
        (0,0,1) => x floor, y floor, z ceiling
        (0,1,0) => x floor, y ceiling, z floor
        (0,1,1) => x floor, y ceiling, z ceiling
        (1,0,0) => x ceiling, y floor, z floor
        (1,0,1) => x ceiling, y floor, z ceil
        (1,1,0) => x ceiling, y ceiling, z floor
        (1,1,1) => x ceiling, y ceiling, z ceiling
        '''
        wts = torch.ones_like(wts_dim[0][0])  # [bs, 1, 19200]
        index = torch.zeros_like(wts_dim[0][0])  # [bs, 1, 19200]
        for d in range(n_dims):
            # pos_dim[0]=[floor(x), floor(x) + 1]; len(pos_dim[0])=2
            # ix_d of first iteration: (0,0,0); ix_d[0]=0
            # pos_dim[0][0].shape=[bs, 1, 19200]
            index = index * grid_dims[d] + pos_dim[d][ix_d[d]]  # [bs, 1, 19200]
            wts = wts * wts_dim[d][ix_d[d]]  # [bs, 1, 19200]

        index = index.long()  # [bs, 1, 19200]

        # grid_flat.shape = [bs, 17, 100*100*80]
        # index.expand(-1, F, -1).shape=[1, 17, 19200] => repeat 17 times
        # scatter_add_(dim, index, src)
        # index = [233960(index=0), 241960, ..., 233960(index=320)]
        # src = feat * wts; feat.shape=[bs, 17, 19200]
        # grid_flat[233960] = src[0] + src[320]
        grid_flat.scatter_add_(2, index.expand(-1, F, -1), feat * wts)  # [bs, 17, 100*100*80]
        grid_flat = torch.round(grid_flat)

    return grid_flat.view(init_grid.shape)  # [bs, 17, 100, 100, 80]


def get_world_xz_from_pixel(
        pixel_coords: tuple = None,
        bbox: dict = None,
        depth_image: np.ndarray = None,
        full_pose: np.ndarray = None,
        camera_intrinsics: np.ndarray = None,
) -> np.ndarray:
    """
    Project an image target to 2D world coordinates (x, z). Two modes:
    1. Point mode: pixel_coords gives a single pixel.
    2. BBox mode: pick the pixel whose depth is closest to the median depth inside bbox.

    Args:
        pixel_coords (tuple, optional): Target pixel (u, v), i.e. (column, row).
        bbox (dict, optional): Bounding box dict with x1, y1, x2, y2.
        depth_image (np.ndarray): Depth image (H, W) in meters.
        full_pose (np.ndarray): Agent global 2D pose [x, z, heading_rad].
                                x and z in meters, heading in radians.
                                heading=0 faces the +X axis.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsics matrix K.

    Returns:
        np.ndarray: 2D world coordinates [x, z], or None if depth is invalid.
    """

    # Select mode from the given arguments
    if bbox is not None:
        # BBox mode: find the pixel whose depth is closest to the median over the box
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        # Clamp bbox to image bounds
        h, w = depth_image.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        # Extract depth values inside the bounding box
        depth_roi = depth_image[y1:y2, x1:x2]

        # Filter invalid depth values
        valid_mask = (depth_roi > 0) & np.isfinite(depth_roi) & (depth_roi < 1000.0)
        valid_depths = depth_roi[valid_mask]

        if len(valid_depths) == 0:
            logger.info(f"Warning: no valid depth inside the bounding box")
            return np.array([12.0, 12.0])

        # Median depth
        median_depth = np.median(valid_depths)

        # Pixel whose depth is closest to the median
        depth_diff = np.abs(depth_roi - median_depth)
        depth_diff[~valid_mask] = np.inf  # exclude invalid pixels

        # ROI-relative coordinates of the minimum-difference pixel
        roi_y, roi_x = np.unravel_index(np.argmin(depth_diff), depth_diff.shape)

        # Convert to absolute image coordinates
        u = x1 + roi_x
        v = y1 + roi_y
        depth = depth_roi[roi_y, roi_x]

    else:
        # Point mode: use pixel_coords
        if pixel_coords is None:
            raise ValueError("Either pixel_coords or bbox must be provided")

        u, v = pixel_coords

        # logger.info('full_pose=', full_pose)

        # Sample depth from a window around the target pixel for robustness
        def get_robust_depth(depth_image, u, v, window_size=5):
            """
            Robust depth from a window_size x window_size window around pixel (u, v).

            Args:
                depth_image: depth image
                u, v: target pixel coordinates
                window_size: window size (odd)

            Returns:
                robust_depth: robust depth, or None if unavailable
            """
            h, w = depth_image.shape
            half_window = window_size // 2

            # Window bounds, clipped to the image
            u_min = max(0, u - half_window)
            u_max = min(w, u + half_window + 1)
            v_min = max(0, v - half_window)
            v_max = min(h, v + half_window + 1)

            # Depth values in the window
            depth_window = depth_image[v_min:v_max, u_min:u_max]

            # Filter invalid depth values (<=0, inf, nan)
            valid_depths = depth_window[
                (depth_window > 0) &
                np.isfinite(depth_window) &
                (depth_window < 1000.0)  # drop overly large depth
                ]

            if len(valid_depths) == 0:
                return None

            # Median as a robust estimate (less noise-sensitive than the mean)
            robust_depth = np.median(valid_depths)

            return robust_depth

        depth = None
        for window_size in [3, 5, 7, 9]:
            depth = get_robust_depth(depth_image, u, v, window_size)
            if depth is not None:
                break

        if depth is None or depth <= 0:
            # logger.info(f"Warning: no valid depth at pixel ({u}, {v}) or its neighborhood.")
            return np.array([12.0, 12.0])

        # logger.info(f"Point mode - pixel: ({u},{v}), depth: {depth:.3f}m")

    # --- 2. Pixel -> camera frame (back-projection) ---
    # Camera frame: X right, Y down, Z forward
    K_inv = np.linalg.inv(camera_intrinsics)
    camera_coords = depth * (K_inv @ np.array([u, v, 1]))

    # aligned_coords is the point in the agent's local frame (X-right, Y-up, Z-forward)
    aligned_coords = np.array([
        camera_coords[0],
        -camera_coords[1],
        camera_coords[2]
    ])
    local_x, local_y, local_z = aligned_coords

    # --- 3. Rotation via heading vectors---
    agent_x_world, agent_z_world, heading_rad = full_pose
    heading_rad = np.deg2rad(heading_rad)

    # 3.1. Agent direction vectors in the world frame
    # heading is counter-clockwise from +X (standard convention)
    # Forward vector (agent forward direction)
    forward_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    # Right vector (agent right direction)
    right_vec = np.array([np.sin(heading_rad), -np.cos(heading_rad)])

    # 3.2. Build world coordinates from the vectors
    # world point = agent position + offset along agent Z + offset along agent X
    world_z = agent_z_world + local_z * forward_vec[1] + local_x * right_vec[1]
    world_x = agent_x_world + local_z * forward_vec[0] + local_x * right_vec[0]
    # --- 4. Return ---
    return np.array([world_x, world_z])
