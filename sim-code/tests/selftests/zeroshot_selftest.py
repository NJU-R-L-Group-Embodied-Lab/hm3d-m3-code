"""Habitat-free checks that VLFM follows the original paper."""
from __future__ import annotations

from pathlib import Path


from benchmark.vlfm_value import BLIP2ValueMap, ConeValueMap, official_prompt

ROOT = Path(__file__).resolve().parents[2] / "benchmark"


def test_vlfm_default_is_blip2() -> None:
    assert official_prompt("chair") == "Seems like there is a chair ahead."
    src = (ROOT / "zeroshot_runner.py").read_text()
    assert "BLIP2ValueMap" in src
    assert 'vlfm_backend: str = "blip2"' in src
    assert "CLIP-ITM" not in src
    assert "blip2-itm" in src
    blip = (ROOT / "vlfm_value.py").read_text()
    assert 'match_head: str = "itc"' in blip
    assert "use_image_text_matching_head=use_itm" in blip
    val = (ROOT / "vlfm_value.py").read_text()
    assert 'match_head: str = "itc"' in val


def test_cone_splat_and_query() -> None:
    vm = ConeValueMap(cell_size=0.5, cone_range=3.0, cone_halfangle_deg=39.0)
    vm.set_targets = lambda *_a, **_k: None  # type: ignore[method-assign]
    vm.update(0.8, (0.0, 0.0), yaw_rad=0.0)
    assert vm.observed_cells > 0
    assert vm.value_at(1.0, 0.0, radius=1.5) > 0.0
    assert vm.value_at(20.0, 20.0, radius=1.5) == -1.0


def test_value_map_classes() -> None:
    assert issubclass(BLIP2ValueMap, ConeValueMap)


def main() -> None:
    test_vlfm_default_is_blip2()
    test_cone_splat_and_query()
    test_value_map_classes()
    print("zeroshot_selftest: PASS")


if __name__ == "__main__":
    main()
