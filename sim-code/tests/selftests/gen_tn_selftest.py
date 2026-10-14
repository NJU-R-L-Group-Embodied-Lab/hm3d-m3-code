"""Offline checks for TN caption usability filters."""
from benchmark.gen_tn import _ascii_tn, tn_text_unusable


def main() -> None:
    assert tn_text_unusable("")
    assert tn_text_unusable("The stool is not visible in this image; only a bed is present.")
    assert tn_text_unusable("The tv is not visible; only a wooden dresser appears.")
    assert not tn_text_unusable("The stool is wooden and stands beside the floral bed.")
    assert _ascii_tn("children\u2019s") == "children's"
    assert _ascii_tn("\u201cADVENTURE\u201d") == '"ADVENTURE"'
    print("[gen_tn_selftest] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
