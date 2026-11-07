"""T-2690: an easy, medium or hard label for every place a new game can start in (notes/T-2690.md).

    python pipeline/start_difficulty.py            measure (resumes), then write the labels
    python pipeline/start_difficulty.py --tiers    only the labels, from the measured lines
    python pipeline/start_difficulty.py --fresh    measure every pack again
    python pipeline/start_difficulty.py --jobs 2   measure with two batches at once

A start's score is the weekday trip legs with both ends inside a circle of RADIUS_KM round its
centre (`centre`), by every purpose but attractions and with no rail, per km² of the circle (each
pack the circle reaches counts its own trips, the sim's `start_difficulty` bin; trips between two
packs are not counted), times its country's fare scale over the US's to FARE_POW: fares follow
income, construction does not, and Anita asked for a weak weight (2026-11-06). The cut points split
the well-known starts (`notable`) by EASY_SHARE and HARD_SHARE, and every start takes the label of
its score. Writes `app/src/game/start-difficulty.json` (the easy and medium ids; any other start is
hard) and the full table to `../artifacts/trainworld/t2690/scores.json`. Measuring runs one pack at
a time per job, 18-24 minutes on one core (`--jobs N` runs N batches at once).
"""
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import simbin

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "app/src/game/registry.json"
PACKS = ROOT / "data/packs"
OUT = ROOT / "app/src/game/start-difficulty.json"
WORK = ROOT.parent / "artifacts/trainworld/t2690"
RADII_KM = [5.0, 10.0, 15.0]
RADIUS_KM = 10.0
# the suggested starts (ui/NewGame.tsx START_CITIES)
QUICK = ["nyc", "lon", "par", "tyo", "tor"]
# a zone (kind "zone") counts as well known with this many people in the circle
NOTABLE_ZONE_POP = 1_500_000
CENTRE_REACH_KM = 40
BATCH = 40
INCOME = ROOT / "app/src/game/countryIncome.json"
FARE_POW = 0.2
# shares of the well-known starts: thirds widened to 80% more medium, 20% of a third taken from
# easy and 60% from hard (Anita, 2026-11-07, T-2748)
EASY_SHARE = (1 - 0.2) / 3
HARD_SHARE = (1 - 0.6) / 3


def fare_weight(country, gdp):
    """(fare scale over the US's)^FARE_POW, the scale as game/income.ts `incomeScale` has it: 11.6 x
    (GDP per head over the US's)^0.5, in tenths; India's GDP for a country without one."""
    g = gdp.get(country) or gdp["IN"]
    return (min(19.9, max(0.1, round(11.6 * (g / gdp["US"]) ** 0.5 * 10) / 10)) / 11.6) ** FARE_POW


def _fold(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).casefold().strip()


def centre(c):
    """The start's centre: for a metro (kind "city"), the OSM city or town in its own place list
    named as it is (the part before any bracket or comma: "Rotterdam [The Hague]", "Washington, DC"),
    the nearest such within CENTRE_REACH_KM of the registry centre (OSM also has a metro label node:
    Sydney's is 38 km west of the city); else, and for every zone, the registry centre. Some
    registry centres are far from their city's (Atlanta 23 km, Oklahoma City 31; notes/T-2690.md)."""
    lon, lat = c["center"]
    f = PACKS / f"{c['id']}.places.json"
    if c.get("kind") == "zone" or not f.exists():
        return lon, lat, False
    d = json.loads(f.read_text(encoding="utf-8"))
    want = _fold(re.split(r"[\[,(]", c["name"])[0])
    best = None
    for name, k, plon, plat, pop, *rest in d["places"]:
        if d["kinds"][k] not in ("city", "town") or want not in {_fold(n) for n in [name, *(rest[0] if rest else [])]}:
            continue
        km = math.hypot((plon - lon) * 111.32 * math.cos(math.radians(lat)), (plat - lat) * 111.32)
        if km <= CENTRE_REACH_KM and (best is None or km < best[0]):
            best = (km, plon, plat)
    return (best[1], best[2], True) if best else (lon, lat, False)


def starts(reg):
    out = []
    for c in reg["cities"]:
        lon, lat, _ = centre(c)
        out.append({"id": c["id"], "lon": round(lon, 5), "lat": round(lat, 5)})
    return out


def lines_of(spec):
    return WORK / f"lines-{hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]}.jsonl"


def measure(reg, spec, fresh, jobs=1):
    WORK.mkdir(parents=True, exist_ok=True)
    spec_path, lines_path = WORK / "starts.json", lines_of(spec)
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    if fresh and lines_path.exists():
        lines_path.unlink()
    done = set()
    if lines_path.exists():
        for line in lines_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["pack"])
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut off by a stopped run
    todo = [c["id"] for c in reg["cities"] if c["id"] not in done and (PACKS / f"{c['id']}.json").exists()]
    print(f"{len(done)} packs measured, {len(todo)} to go; lines in {lines_path}", flush=True)
    if not todo:
        return lines_path
    exe = simbin.exe(ROOT, "start_difficulty", jobs="1")
    t0 = time.time()
    lock, done_n = threading.Lock(), [0]
    with open(lines_path, "a", encoding="utf-8") as out:
        def batch(part):
            r = subprocess.run([str(exe), str(spec_path), *[f"{p}.json" for p in part]], cwd=PACKS,
                               capture_output=True, text=True, encoding="utf-8")
            with lock:
                for line in r.stdout.splitlines():
                    if line.startswith("{"):
                        out.write(line + "\n")
                out.flush()
                if r.returncode:
                    print(f"batch from {part[0]} exited {r.returncode}: {r.stderr[-400:]}", flush=True)
                done_n[0] += len(part)
                n, el = done_n[0], time.time() - t0
                print(f"{n}/{len(todo)} packs, {el / 60:.1f} min, about {el / n * (len(todo) - n) / 60:.0f} min left", flush=True)
        with ThreadPoolExecutor(max(1, jobs)) as pool:
            list(pool.map(batch, [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]))
    return lines_path


def scores(reg, lines_path):
    """Per start, per radius: the six sums over every pack (commute legs, walked, outing legs,
    walked, people, jobs)."""
    sums = {c["id"]: [[0.0] * 6 for _ in RADII_KM] for c in reg["cities"]}
    errors = []
    for line in lines_path.read_text(encoding="utf-8").splitlines():
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in v:
            errors.append((v["pack"], v["error"]))
            continue
        for sid, per in v["starts"].items():
            if sid in sums:
                for r, vals in enumerate(per):
                    for k, x in enumerate(vals):
                        sums[sid][r][k] += x
    if errors:
        print(f"{len(errors)} packs failed: {errors[:10]}", flush=True)
    return sums


def notable(reg, sums, r):
    """The well-known starts: the suggested ones, every metro built as a city, and zones with
    NOTABLE_ZONE_POP people in the circle."""
    return [c["id"] for c in reg["cities"]
            if c["id"] in QUICK or c.get("kind") != "zone" or sums[c["id"]][r][4] >= NOTABLE_ZONE_POP]


def tiers(reg, sums, centres):
    r = RADII_KM.index(RADIUS_KM)
    area = math.pi * RADIUS_KM ** 2
    gdp = json.loads(INCOME.read_text(encoding="utf-8"))["usd"]
    country = {c["id"]: c.get("country") for c in reg["cities"]}
    score = {i: (v[r][0] + v[r][2]) / area * fare_weight(country[i], gdp) for i, v in sums.items()}
    known = sorted(notable(reg, sums, r), key=lambda i: -score[i])
    n = len(known)
    easy_min = score[known[round(n * EASY_SHARE) - 1]]
    medium_min = score[known[n - round(n * HARD_SHARE) - 1]]
    tier = {i: "easy" if s >= easy_min else "medium" if s >= medium_min else "hard" for i, s in score.items()}
    out = {
        "task": "T-2690",
        "rule": f"weekday trip legs a km² with both ends within {RADIUS_KM:g} km of the start's centre, no rail, "
                f"times (fare scale over the US's)^{FARE_POW:g}; "
                f"easy from {easy_min:,.0f}, medium from {medium_min:,.0f} ({EASY_SHARE:.0%} easy, {HARD_SHARE:.0%} hard "
                f"of {n} well-known starts)",
        "easy": sorted(i for i, t in tier.items() if t == "easy"),
        "medium": sorted(i for i, t in tier.items() if t == "medium"),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    table = {c["id"]: {"name": c["name"], "country": c.get("country"), "kind": c.get("kind", "city"),
                       "notable": c["id"] in set(known), "tier": tier[c["id"]], "score": round(score[c["id"]]),
                       "centre": centres[c["id"]],
                       "registry_centre": c["center"],
                       "by_radius": {f"{RADII_KM[k]:g}": sums[c["id"]][k] for k in range(len(RADII_KM))}}
             for c in reg["cities"]}
    (WORK / "scores.json").write_text(json.dumps({"radius_km": RADIUS_KM, "easy_min": easy_min, "medium_min": medium_min,
                                                   "starts": table}, ensure_ascii=False, indent=1), encoding="utf-8")
    n = {t: sum(1 for x in tier.values() if x == t) for t in ("easy", "medium", "hard")}
    print(f"{len(known)} well-known starts; cuts {easy_min:,.0f} and {medium_min:,.0f} weighted legs a km²; "
          f"all {len(tier)}: {n}; wrote {OUT} and {WORK / 'scores.json'}", flush=True)


if __name__ == "__main__":
    reg = json.loads(REG.read_text(encoding="utf-8"))
    spec = {"radii_km": RADII_KM, "starts": starts(reg)}
    lines = lines_of(spec)
    args = sys.argv[1:]
    if "--tiers" not in args:
        jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 1
        lines = measure(reg, spec, "--fresh" in args, jobs)
    tiers(reg, scores(reg, lines), {s["id"]: [s["lon"], s["lat"]] for s in spec["starts"]})
