"""A trial build_model run that records each stage of every register line's sections.

    python harness.py <cc> <out dir>

Writes <out>/lines.json etc. as build_model --out does, plus <out>/stages.json:
reader output, RINF raw pieces (rinf.GROUPS), before and after drop_unridden_sections, with
the route share of each junction-ended section. Wrapper registers get rinf.split_pieces
exposed as if they had `split_pieces = rinf.split_pieces` (if rinf has one)."""
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
if os.environ.get("HARNESS_DEV"):
    sys.path.insert(0, os.environ["HARNESS_DEV"])     # dev copies of rinf.py / pieces.py
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rebuild import REGISTER  # noqa: E402

cc, out = sys.argv[1], Path(sys.argv[2])
if not out.is_absolute():
    out = Path(__file__).resolve().parent / out      # never inside the project
out.mkdir(parents=True, exist_ok=True)
reg = REGISTER.get(cc, f"rinf:data/raw/rinf/{cc}")
import build_model as bm  # noqa: E402
import rinf  # noqa: E402
print(f"harness: rinf from {rinf.__file__}", flush=True)

mod = importlib.import_module(reg.split(":")[0])
if mod is not rinf and hasattr(rinf, "split_pieces") and not hasattr(mod, "split_pieces") \
        and os.environ.get("HARNESS_HOOK", "1") == "1":
    mod.split_pieces = rinf.split_pieces
    mod.LINE_PIECES = rinf.LINE_PIECES
REC = {}
orig_build = mod.build


def secs_of(lines):
    return {l["id"]: {"name": l["name"], "ref": l.get("ref", ""),
                      "sections": [list(s[:3]) for s in l["sections"]]}
            for l in lines if l.get("src", "osm") != "osm" and not l.get("service")}


def build(path, log):
    r = orig_build(path, log)
    REC["reader"] = secs_of(r[0])
    REC["groups"] = dict(rinf.GROUPS)
    REC["reader_junction"] = sorted(k for k, v in r[1].items() if v.get("junction"))
    return r


mod.build = build
orig_drop = bm.drop_unridden_sections


def drop(lines, stations, geoms, route_share, log):
    REC["predrop"] = secs_of(lines)
    junction = {sid for sid, s in stations.items() if s.get("junction")}
    REC["junction"] = sorted(junction)
    REC["share"] = {f"{lid}|{k}": round(v, 3) for (lid, k), v in route_share.items()
                    if lid in REC["predrop"]}
    REC["served"] = sorted(f"{l['id']}|{k}" for l in lines for k in (l.get("served_sections") or ()))
    r = orig_drop(lines, stations, geoms, route_share, log)
    REC["postdrop"] = secs_of([l for l in lines if l["id"] not in r])
    return r


bm.drop_unridden_sections = drop
sys.argv = ["build_model.py", "--region", cc, "--register", reg, "--out", str(out)]
try:
    bm.main()
finally:
    with open(out / "stages.json", "w", encoding="utf-8") as f:
        json.dump(REC, f, ensure_ascii=False)
