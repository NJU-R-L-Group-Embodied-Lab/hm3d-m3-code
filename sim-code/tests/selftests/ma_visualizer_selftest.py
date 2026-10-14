"""Offline occupancy-canvas checks. No Habitat."""
from __future__ import annotations

import numpy as np

from vlnce_baselines.multi.ma_visualizer import (
    IncrementalOccupancy,
    OccGridConfig,
    crop_map_ink,
    draw_found_marker,
    occupancy_frame_from_bounds,
)


def main() -> None:
    origin, size_m = occupancy_frame_from_bounds(
        (-20.0, -1.0, -5.0), (23.0, 8.0, 32.0), pad_m=6.0
    )
    assert abs(origin[0] - 1.5) < 1e-6
    assert abs(origin[1] - 13.5) < 1e-6
    assert size_m >= 55.0

    cfg = OccGridConfig(size_m=80.0, resolution_m=0.05)
    occ = IncrementalOccupancy(
        cfg, hfov_deg=79.0, depth_h=8, depth_w=8, sensor_height_m=0.88,
        origin_xz=origin,
    )
    row, col = occ.world_to_cell(origin[0], origin[1])
    mid = occ.cells // 2
    assert abs(row - mid) <= 1
    assert abs(col - mid) <= 1
    # House rim must stay inside the canvas (this is the 24 m crop failure).
    row_e, col_e = occ.world_to_cell(23.0, 32.0)
    assert occ.in_bounds(row_e, col_e)

    canvas = np.full((800, 800, 3), 255, dtype=np.uint8)
    canvas[200:500, 250:620] = (220, 220, 220)
    canvas[300:320, 300:500] = (0, 0, 0)
    cropped = crop_map_ink(canvas, pad_px=8)
    assert cropped.shape[0] < 800 and cropped.shape[1] < 800
    assert cropped.shape[0] == cropped.shape[1]

    mark = np.full((80, 80, 3), 255, dtype=np.uint8)
    draw_found_marker(mark, (40, 40), 1, label="sofa")
    assert int(mark[40, 40, 1]) > 100

    print("ma_visualizer_selftest: ok", size_m, occ.cells, cropped.shape)


if __name__ == "__main__":
    main()
