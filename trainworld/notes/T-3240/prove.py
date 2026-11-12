"""T-3240 proof in scratch: new market packs (normalisers with the pull, px) for a sample of zones, written to a
scratch root with the pulled registry rule, every other pack the composition needs copied with its excess,
then the sim's market_pack --only over the scratch root. Nothing under data/packs or app/ is written.
    python prove.py packs ID ...      the sample's market packs into the scratch root
    python prove.py rows              the scratch rows (runs market_pack)
"""
import copy, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
os.environ["OMP_NUM_THREADS"] = "3"
import market_zones as M
import shared_lock, simbin
M.PULL = json.loads((M.Path(M.__file__).parent / "market_pull.json").read_text(encoding="utf-8"))
M.PULL["on"] = True  # the proof runs it as the rebuild will

S = Path(__file__).parent / "root"
(S / "data/packs").mkdir(parents=True, exist_ok=True)
(S / "app/src/game").mkdir(parents=True, exist_ok=True)
reg = shared_lock.read_json(M.REG)
reg2 = copy.deepcopy(reg)
M.assign_places(reg2)
(S / "app/src/game/registry.json").write_text(json.dumps(reg2, indent=1, ensure_ascii=False), encoding="utf-8")
rows = json.loads((M.ROOT / "app/src/game/market-rows.json").read_text(encoding="utf-8"))
PX = M.PULL["places"]

if sys.argv[1] == "packs":
    ids = sys.argv[2:]
    t0 = time.time()
    F = M.Field()
    print(f"field with the pull in {time.time() - t0:.0f} s", flush=True)
    M.groups = M.Groups(reg, set(ids))
    air = M.airports_by_zone(reg, set(ids))
    for cid in ids:
        t = time.time()
        out = M.build(cid, reg2, F, air)
        M.write(S / f"data/packs/{cid}.market.json", out)
        (S / f"data/packs/{cid}.new").write_text("")
        old = json.loads((M.PACKS / f"{cid}.market.json").read_text(encoding="utf-8"))
        rg = np.array(out["zone_ng"]) / np.array(old["zone_ng"]); rt = np.array(out["zone_nt"]) / np.array(old["zone_nt"])
        print(f"{cid}: {len(out['zones'])} zones in {time.time() - t:.0f} s, px {out['px']}, ground normalisers x{rg.min():.3f}-{rg.max():.3f}, tail x{rt.min():.3f}-{rt.max():.3f}", flush=True)
else:
    sample = sorted(p.name[:-12] for p in (S / "data/packs").glob("*.market.json") if (S / f"data/packs/{p.name[:-12]}.new").exists())
    ids = list(rows)
    lon, lat = np.array([rows[i]["lon"] for i in ids]), np.array([rows[i]["lat"] for i in ids])
    compose = set()
    for z in sample:
        d = M.W.km(rows[z]["lon"], rows[z]["lat"], lon, lat)
        compose |= {ids[k] for k in np.nonzero(d < 276)[0]}
        pk = json.loads((S / f"data/packs/{z}.market.json").read_text(encoding="utf-8"))
        compose |= {c["id"] for c in reg["cities"] if c.get("place") in set(pk.get("commute_place", [])) | set(pk.get("everyday_place", []))}
    for z in sorted(compose - set(sample)):
        pk = json.loads((M.PACKS / f"{z}.market.json").read_text(encoding="utf-8"))
        pk["px"] = PX.get(z, [0.0])[0]
        (S / f"data/packs/{z}.market.json").write_text(json.dumps(pk, separators=(",", ":")), encoding="utf-8")
    allrows = {z: ({**r, "px": PX[z][0]} if PX.get(z, [0])[0] > 0 else r) for z, r in rows.items()}
    (S / "app/src/game/market-rows.json").write_text(json.dumps(allrows, ensure_ascii=False), encoding="utf-8")
    only = S / "only.json"
    only.write_text(json.dumps({"rows": sample, "compose": sorted(compose | set(sample))}))
    exe = simbin.exe(M.ROOT, "market_pack")
    print(f"rows of {len(sample)} from {len(compose | set(sample))} packs", flush=True)
    subprocess.run([str(exe), "--root", str(S), "--out", str(S / "rows.json"), "--only", str(only)], check=True)
