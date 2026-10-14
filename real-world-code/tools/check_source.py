#!/usr/bin/env python3
"""Audit distributable source for secrets, machine paths and generated artifacts.

Identity terms (user, host and affiliation names) are deliberately not listed in this
file. They come from the maintainers' repository-level leak gate
``../../tools/check_source.py`` when it is present (it is not part of the release), and/or from an untracked
plain-text file named by ``$LEAK_TERMS_FILE`` or ``tools/identity_terms.local.txt``
(one term per line). When neither source is available the identity check is skipped
and a note is printed.
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_GATE = ROOT.parent / "tools" / "check_source.py"
LOCAL_TERMS = Path(__file__).resolve().parent / "identity_terms.local.txt"
IGNORED = {
    "log",
    ".runtime",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".git",
}
PATTERNS = {
    "credential": re.compile(r"sk-[A-Za-z0-9._-]{20,}"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "machine-path": re.compile(r"/(?:home|Users)/[A-Za-z][^\s\"']*"),
    "private-endpoint": re.compile(
        r"(?:https?://|[\"'])(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)"
    ),
    "root-path": re.compile(r"(?<![\w.])/(?:root|nvme\d*|mnt)/"),
}


def load_identity_terms():
    """Return a matcher with a ``find(text)`` method, or None when no term source exists."""
    local = [os.environ.get("LEAK_TERMS_FILE"), str(LOCAL_TERMS)]
    if ROOT_GATE.is_file():
        spec = importlib.util.spec_from_file_location("_hm3d_m3_root_gate", ROOT_GATE)
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        matcher = gate.load_terms(gate.term_files(local))
        return matcher if matcher else None
    terms = []
    for name in local:
        if name and Path(name).is_file():
            terms += [
                t.strip().lower()
                for t in Path(name).read_text().splitlines()
                if t.strip() and not t.lstrip().startswith("#")
            ]
    if not terms:
        return None

    class _Plain:
        def __init__(self, words):
            self.rx = re.compile("|".join(r"\b%s\b" % re.escape(w) for w in words), re.I)

        def find(self, text):
            return [m.group(0) for m in self.rx.finditer(text)]

    return _Plain(terms)


def main():
    failures = []
    count = 0
    identity = load_identity_terms()
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(p in IGNORED or p.endswith(".egg-info") for p in rel.parts) or not path.is_file():
            continue
        if path.name.endswith(".local.txt"):
            continue  # untracked local configuration (never committed)
        count += 1
        if path.is_symlink() or path.suffix in (".so", ".o", ".a", ".bag", ".db3", ".pcd"):
            failures.append((str(rel), 0, "generated artifact or symlink"))
            continue
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text()
        except UnicodeError:
            # Documentation screenshots are allowed; other binaries remain rejected.
            if rel.parent == Path("assets") and rel.suffix == ".png":
                data = path.read_bytes()
                if data.startswith(b"\x89PNG\r\n\x1a\n") and data.endswith(b"IEND\xaeB`\x82"):
                    continue
            failures.append((str(rel), 0, "unexpected binary"))
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    failures.append((str(rel), number, label))
            if identity is not None and identity.find(line):
                failures.append((str(rel), number, "internal-identity"))
        if rel.parts[0] == "src" and (
            "unitree_sdk2py" in text or "from map2d." in text or "import ultra" in text
        ):
            failures.append((str(rel), 0, "legacy runtime import"))
    for path, line, label in failures:
        print(f"{path}:{line}: {label}")  # Never print matched credentials or terms.
    if identity is None:
        print("note: identity-term check skipped (no root gate or local term file)")
    print(f"Scanned {count} source files; {len(failures)} findings")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
