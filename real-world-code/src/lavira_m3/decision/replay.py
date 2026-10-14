"""Exact-request model replay; mismatches never fall back to a paid endpoint."""

import hashlib
import json
from pathlib import Path


def request_key(role, messages):
    raw = json.dumps([role, messages], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class ReplayRouter:
    def __init__(self, path):
        records = json.loads(Path(path).read_text())
        self.records = {r["request_sha256"]: r["response"] for r in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate replay request")

    def _generate(self, role, messages):
        key = request_key(role, messages)
        if key not in self.records:
            raise ValueError("no replay response for " + role + " request " + key)
        return self.records[key]

    def generate_la(self, messages, max_tokens=2048):
        return self._generate("la", messages)

    def generate_va(self, messages, max_tokens=1024):
        return self._generate("va", messages)

    def note_record(self, kind, step=None):
        pass

    def bind_call_logger(self, logger):
        pass
