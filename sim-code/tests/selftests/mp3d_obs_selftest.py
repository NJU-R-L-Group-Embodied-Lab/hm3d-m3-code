"""Offline flip check. No Habitat."""
from __future__ import annotations

import numpy as np

from vlnce_baselines.multi.mp3d_obs import (
    flip_visual_obs,
    scene_needs_rgb_flip,
)


def main() -> None:
    assert not scene_needs_rgb_flip("data/qual_scenes/mp3d/vyrNrziPKCB/vyrNrziPKCB.glb")
    assert not scene_needs_rgb_flip("data/qual_scenes/00758-HfMobPm86Xn/x.basis.glb")
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[0] = 10
    rgb[-1] = 200
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)
    out = flip_visual_obs({"rgb": rgb, "depth": depth, "collided": False})
    assert int(out["rgb"][0, 0, 0]) == 200
    assert int(out["rgb"][-1, 0, 0]) == 10
    assert float(out["depth"][0, 0]) == float(depth[-1, 0])
    again = flip_visual_obs(out)
    assert np.array_equal(again["rgb"], rgb)
    batched = flip_visual_obs([{"rgb": rgb}, {"rgb": rgb}])
    assert int(batched[0]["rgb"][0, 0, 0]) == 200
    print("mp3d_obs_selftest: ok")


if __name__ == "__main__":
    main()
