"""Rebuild the model and tiles of several countries in turn, after a shared-file change.

    python tools/rebuild.py be nl at          # build_model then build_tiles, per country
    python tools/rebuild.py --model-only cz   # skip the tiles

Each step's output goes to data/logs/rebuild_<cc>_<step>.txt; one line per step is printed.
Compare register lines before and after with tools/compare_lines.py (save first, then diff).
A country's register argument is in REGISTER below; any other code is taken to be a RINF
country (rinf:data/raw/rinf/<cc>).
"""
import subprocess
import sys
import time
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


def main():
    args = sys.argv[1:]
    model_only = "--model-only" in args
    regions = [a for a in args if not a.startswith("--")]
    if not regions:
        sys.exit(__doc__)
    LOGS.mkdir(parents=True, exist_ok=True)
    for cc in regions:
        reg = REGISTER.get(cc, f"rinf:data/raw/rinf/{cc}")
        steps = [["build_model.py", "--region", cc, "--register", reg]]
        if not model_only:
            steps.append(["build_tiles.py", "--region", cc])
        for step in steps:
            t0 = time.time()
            log = LOGS / f"rebuild_{cc}_{step[0][:-3]}.txt"
            with open(log, "w", encoding="utf-8") as f:
                r = subprocess.run([sys.executable] + step, cwd=ROOT, stdout=f,
                                   stderr=subprocess.STDOUT)
            print(f"{cc} {step[0]}: exit {r.returncode}, {time.time() - t0:.0f} s  ({log.name})",
                  flush=True)


if __name__ == "__main__":
    main()
