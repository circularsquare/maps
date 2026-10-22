"""Helpers for the small-territory builds (je, im, gg, ck, fk, gi, nu, tk; session eca0185b-terr).

written(node): origin_mix.mix() answers in DRAWN ids (it reads countries' regrouped counts), but
a census-mapping module and its tree.d fragment must carry WRITTEN ids (taxonomy/regroup.txt moves
them afterwards; `origin_mix.py --fragment` expects written ones). This undoes the move: it
returns the written id whose drawn id is `node`, or `node` itself when nothing moves to it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "taxonomy"))
_INV = None


def _inverse():
    global _INV
    if _INV is None:
        from regroup import move
        inv = {}
        files = [ROOT / "taxonomy" / "tree.txt", *sorted((ROOT / "taxonomy" / "tree.d").glob("*.txt"))]
        for f in files:
            for ln in f.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "|" in ln:
                    w = ln.split("|")[0].strip()
                    d = move(w)
                    if d != w:
                        inv.setdefault(d, w)
        _INV = inv
    return _INV


def written(node):
    return _inverse().get(node, node)


def mix_written(iso, dest):
    """origin_mix.mix(iso, dest) with written node ids, shares merged where two drawn ids share one."""
    from origin_mix import mix
    out = {}
    for n, s in mix(iso, dest).items():
        w = written(n)
        out[w] = out.get(w, 0.0) + s
    return out
