"""What would a code change do to each country? Build the working tree into a temporary folder
and compare with what is shipped in dist/data. Writes nothing in the project.

    python tools/ab.py be nl            # trial-build both, compare, print what moved
    python tools/ab.py --all            # every built country (3 at once, longest first)
    python tools/ab.py -j 1 cz          # one at a time

Use it BEFORE landing a shared change, on the countries the change can touch: dist/data must
be the build of the code as it was, so run it while the change is in the working tree but
before rebuilding. A change gated on one country (`if region == "hr"`, a COUNTRY key nobody
else sets) needs no run on the others. After landing, the real rebuild is tools/rebuild.py
plus tools/compare_lines.py.

Compared per country: every line (src, name, km to 0.01, sections as station pairs, closed),
station ids gone or new, and whether foot.json and ways.json are byte-identical. Trial builds
go to <temp>/noritetsu_ab/<cc>/ (build_model's --out; the last build's station ids are still
read from dist/data, so aliases carry as in a real build). Logs beside them.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild import MINUTES, REGISTER, ROOT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
DIST = ROOT / "dist" / "data"
TMP = Path(tempfile.gettempdir()) / "noritetsu_ab"


def load(d, name):
    with open(d / name, encoding="utf-8") as f:
        return json.load(f)


def trial(cc):
    out = TMP / cc
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    reg = REGISTER.get(cc, f"rinf:data/raw/rinf/{cc}")
    t0 = time.time()
    with open(TMP / f"{cc}.log", "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "build_model.py", "--region", cc, "--register", reg,
                            "--out", str(out)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def compare(cc):
    a, b = DIST / cc, TMP / cc
    la = {l["id"]: l for l in load(a, "lines.json")["lines"]}
    lb = {l["id"]: l for l in load(b, "lines.json")["lines"]}
    rows = []
    for k in sorted(set(la) | set(lb)):
        x, y = la.get(k), lb.get(k)
        if x is None or y is None:
            z = x or y
            rows.append(f"{'gone' if y is None else 'new '} {z.get('src')} {z['name']} "
                        f"{z['km']:.2f} km")
            continue
        d = []
        if round(x["km"], 2) != round(y["km"], 2):
            d.append(f"km {x['km']:.2f}->{y['km']:.2f}")
        if sorted(tuple(s[:2]) for s in x["sections"]) != \
                sorted(tuple(s[:2]) for s in y["sections"]):
            d.append(f"sections {len(x['sections'])}->{len(y['sections'])}")
        for f in ("name", "name_en", "src", "service", "closed"):
            if x.get(f) != y.get(f):
                d.append(f)
        if d:
            rows.append(f"chg  {x.get('src')} {x['name']}: {', '.join(d)}")
    sa, sb = load(a, "stations.json")["stations"], load(b, "stations.json")["stations"]
    same = {n: (a / n).exists() and (b / n).exists()
            and (a / n).read_bytes() == (b / n).read_bytes() for n in ("foot.json", "ways.json")}
    reg = sum(1 for r in rows if " osm " not in f" {r} ")
    head = (f"{cc}: {len(rows)} lines differ ({reg} not OSM), stations {len(sa)}->{len(sb)} "
            f"({len(set(sa) - set(sb))} gone, {len(set(sb) - set(sa))} new); "
            + ", ".join(f"{n} {'same' if s else 'DIFFERS'}" for n, s in same.items()))
    return head, rows


def one(cc):
    code, secs = trial(cc)
    if code:
        return f"{cc}: TRIAL BUILD FAILED (exit {code}, see {TMP / (cc + '.log')})", []
    head, rows = compare(cc)
    return f"{head}  [{secs:.0f} s]", rows


def main():
    args = sys.argv[1:]
    jobs = 3
    if "-j" in args:
        jobs = max(1, int(args[args.index("-j") + 1]))
        del args[args.index("-j"):args.index("-j") + 2]
    if "--all" in args:
        regions = sorted(p.name for p in DIST.iterdir() if (p / "lines.json").exists())
    else:
        regions = [a for a in args if not a.startswith("-")]
    if not regions:
        sys.exit(__doc__)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "2")
    TMP.mkdir(parents=True, exist_ok=True)
    regions.sort(key=lambda cc: -MINUTES.get(cc, 1))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for head, rows in pool.map(one, regions):
            print(head, flush=True)
            for r in rows[:25]:
                print("    " + r)
            if len(rows) > 25:
                print(f"    ... {len(rows) - 25} more")
    print(f"trial builds and logs in {TMP}")


if __name__ == "__main__":
    main()
