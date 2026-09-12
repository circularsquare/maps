"""
Colour lookup for the japanrail poster, matching the interactive map exactly.

The tables come from build/colors.json, which bake.py lifts out of
riders/japanriders/index.html, so the poster cannot drift from the web map.
The resolution order is a port of that page's colorLineFor().
"""

from __future__ import annotations

import json
from pathlib import Path

BUILD = Path(__file__).parent / "build"

_C = None


def load():
    global _C
    if _C is None:
        p = BUILD / "colors.json"
        if not p.exists():
            raise SystemExit("build/colors.json missing — run bake.py first")
        _C = json.loads(p.read_text(encoding="utf-8"))
        _C["_by_op"] = {}
        for key, col in _C["lines"].items():
            op, ln = key.split("||", 1)
            _C["_by_op"].setdefault(op, {})[ln] = col
    return _C


def operator_color(op):
    c = load()
    jr = c["jr"].get(op)
    return jr["color"] if jr else c["other"]


def line_color(op, line):
    """Official line colour, falling back to the operator's own colour.

    The throughput dataset bundles some private lines into one feature under a
    ・-joined name ('京都線・千里線・嵐山線'), while other sources carry the
    component names on their own. Match any component, and allow the bundled
    form's habit of eliding 線 on non-final components (大阪 -> 大阪線).
    """
    c = load()
    m = c["_by_op"].get(op)
    if m:
        if line in m:
            return m[line]
        for k, col in m.items():
            if "・" not in k:
                continue
            for part in k.split("・"):
                if part == line or part + "線" == line:
                    return col
    return operator_color(op)


def operator_en(op):
    jr = load()["jr"].get(op)
    return jr["en"] if jr else op
