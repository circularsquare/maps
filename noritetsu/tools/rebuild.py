"""Rebuild the model and tiles of several countries, after a shared-file change.

    python tools/rebuild.py be nl at          # build_model then build_tiles, per country
    python tools/rebuild.py --model-only cz   # skip the tiles
    python tools/rebuild.py -j 1 jp           # one country at a time (default: 3 at once)

Each step's output goes to data/logs/rebuild_<cc>_<step>.txt; one line per step is printed.
Compare register lines before and after with tools/compare_lines.py (save first, then diff).
A country's register argument is in REGISTER below; any other code is taken to be a RINF
country (rinf:data/raw/rinf/<cc>).

PARALLEL. A build uses one or two cores (numpy capped by OMP_NUM_THREADS, which is set to 2
here if unset), so three countries at once stays within the ~6 cores Anita lets builds take.
Countries are started longest first so the slow ones (cn, ru, fr, jp, pl) do not trail at the
end. Each country's model and tiles still run in order. Memory: ru peaks near 3 GB, cn and fr
near 2 GB, so three at once is fine on this machine.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "logs"
REGISTER = {
    "jp": "n02:data/raw/N02-24_GML.zip",
    "ch": "schienennetz:data/raw/schienennetz_2056_de.gdb.zip",
    "fr": "fr_register:data/raw/fr",
    "kr": "kr_register:data/raw/kr",
    "tw": "tw_register:data/raw/tw",
    "cn": "cn_register:data/raw/cn",
    "hk": "hk_register:data/raw/hk",
    "sg": "sg_register:data/raw/sg",
}
# Rough build_model + build_tiles minutes on 2026-10-01, for ordering only.
MINUTES = {"cn": 11, "ru": 9, "fr": 6, "jp": 4, "pl": 4, "de": 8, "it": 5, "es": 4,
           "ch": 2, "at": 2, "cz": 2, "be": 1, "nl": 1}


def run_country(cc, model_only):
    reg = REGISTER.get(cc, f"rinf:data/raw/rinf/{cc}")
    steps = [["build_model.py", "--region", cc, "--register", reg]]
    if not model_only:
        steps.append(["build_tiles.py", "--region", cc])
    out = []
    for step in steps:
        t0 = time.time()
        log = LOGS / f"rebuild_{cc}_{step[0][:-3]}.txt"
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run([sys.executable] + step, cwd=ROOT, stdout=f,
                               stderr=subprocess.STDOUT)
        line = f"{cc} {step[0]}: exit {r.returncode}, {time.time() - t0:.0f} s  ({log.name})"
        print(line, flush=True)
        out.append(line)
        if r.returncode:
            break                    # tiles read the model: do not tile a failed build
    return out


def main():
    args = sys.argv[1:]
    model_only = "--model-only" in args
    jobs = 3
    if "-j" in args:
        jobs = max(1, int(args[args.index("-j") + 1]))
        del args[args.index("-j"):args.index("-j") + 2]
    regions = [a for a in args if not a.startswith("--")]
    if not regions:
        sys.exit(__doc__)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "2")
    LOGS.mkdir(parents=True, exist_ok=True)
    regions.sort(key=lambda cc: -MINUTES.get(cc, 1))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda cc: run_country(cc, model_only), regions))
    failed = [r for rs in results for r in rs if " exit 0," not in r]
    print(f"done: {len(regions)} countries in {(time.time() - t0) / 60:.1f} min, "
          f"{len(failed)} failed steps" + ("".join(f"\n  {f}" for f in failed)))


if __name__ == "__main__":
    main()
