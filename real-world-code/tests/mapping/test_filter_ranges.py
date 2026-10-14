import math

from lavira_m3.mapping.bodies import FILTER_RANGES, inside_filter

for name, modes in FILTER_RANGES.items():
    for mode, (rear, front, half) in modes.items():
        for yaw in (0, math.pi / 2, 0.7):
            for z in (-100, -1, 0, 2, 100, None):
                for x, y, expected in (
                    (0, 0, True),
                    (front - 0.001, 0, True),
                    (-rear + 0.001, 0, True),
                    (0, half - 0.001, True),
                    (front + 0.001, 0, False),
                    (-rear - 0.001, 0, False),
                    (0, half + 0.001, False),
                ):
                    px = 2 + math.cos(yaw) * x - math.sin(yaw) * y
                    py = 3 + math.sin(yaw) * x + math.cos(yaw) * y
                    assert (
                        inside_filter(px, py, name, 2, 3, yaw, z, own=mode == "own") == expected
                    ), (name, mode, yaw, z, x, y)
print("PASS: 756 height/orientation/boundary cases")
