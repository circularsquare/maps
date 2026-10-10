"""Rebuild the model and tiles of several countries, after a shared-file change.

    python tools/rebuild.py be nl at          # build_model then build_tiles, per country
    python tools/rebuild.py --model-only cz   # skip the tiles
    python tools/rebuild.py --tiles-only cz   # only the tiles, from the model as it is
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
    "us": "us_register:data/raw/us/narn_passenger.geojson",
    "ca": "ca_register:data/raw/ca/narn_passenger.geojson",
    "au": "au_register:data/raw/au/ga_rail_lines.geojson",
    "in": "in_register:data/raw/in",
    "no": "no_register:data/raw/no",
    "gb": "gb_register:data/raw/gb",
    "mx": "mx_register:data/raw/mx",
    "th": "th_register:data/raw/th",
    "my": "my_register:data/raw/my",
    "id": "id_register:data/raw/id",
    "tr": "tr_register:data/raw/tr",
    "nz": "nz_register:data/raw/nz",
    "vn": "vn_register:data/raw/vn",
    "za": "za_register:data/raw/rinf/za",
    "br": "br_register:data/raw/br",
    "ar": "ar_register:data/raw/ar",
    "cl": "cl_register:data/raw/cl",
    "ir": "ir_register:data/raw/ir",
    # Central Asia: the tariff guide's sheets through rinf.py (casia_register.py)
    "kz": "casia_register:data/raw/rinf/kz", "uz": "casia_register:data/raw/rinf/uz",
    "kg": "casia_register:data/raw/rinf/kg", "tj": "casia_register:data/raw/rinf/tj",
    "tm": "casia_register:data/raw/rinf/tm",
    # The Caucasus: Russia's tariff guide sheets through rinf.py (`caucasus_register.py --clip
    # <cc>` after every extract of the three).
    "ge": "caucasus_register:data/raw/rinf/ge", "am": "caucasus_register:data/raw/rinf/am",
    "az": "caucasus_register:data/raw/rinf/az",
    # Abkhazia: cut from Georgia's extract (`caucasus_register.py --clip xa` after Georgia's).
    "xa": "caucasus_register:data/raw/rinf/xa",
    # North Africa: one reader writes rinf.py's inputs (nafrica_register.py). After an
    # extract (with --station-areas): `--clip <cc>` then `--fill <cc>`.
    "ma": "nafrica_register:data/raw/rinf/ma", "dz": "nafrica_register:data/raw/rinf/dz",
    "tn": "nafrica_register:data/raw/rinf/tn", "eg": "nafrica_register:data/raw/rinf/eg",
    # The Balkans: one reader writes rinf.py's inputs (balkans_register.py; `--clip ba` after
    # every Bosnia extract).
    "rs": "balkans_register:data/raw/rinf/rs", "ba": "balkans_register:data/raw/rinf/ba",
    "me": "balkans_register:data/raw/rinf/me", "mk": "balkans_register:data/raw/rinf/mk",
    "al": "balkans_register:data/raw/rinf/al", "xk": "balkans_register:data/raw/rinf/xk",
    # The Middle East: hand lists through rinf.py (mideast_register.py, nafrica's code).
    # After an extract: `mideast_register.py --clip <cc>`; then `--join iq` for Iraq and
    # `--fill jo` for Jordan. Qatar is all metro and tram: no register (None).
    "sa": "mideast_register:data/raw/rinf/sa", "ae": "mideast_register:data/raw/rinf/ae",
    "iq": "mideast_register:data/raw/rinf/iq", "jo": "mideast_register:data/raw/rinf/jo",
    "qa": None,
    "pk": "pk_register:data/raw/pk",
    # Sri Lanka and Bangladesh: hand lists through rinf.py (lk_register.py is the engine for
    # both); after an extract `lk_register.py --clip` / `bd_register.py --clip`, then `--fill`
    # (handoff_notes/bd_lk_build.md).
    "lk": "lk_register:data/raw/rinf/lk", "bd": "bd_register:data/raw/rinf/bd",
    # West and Central Africa: hand lists through rinf.py (wafrica_register.py; after an
    # extract `--clip <cc>`, then `--fill`; handoff_notes/wafrica_build.md).
    **{cc: f"wafrica_register:data/raw/rinf/{cc}"
       for cc in ("ng", "cm", "ao", "ga", "cg", "sn", "gh", "bf", "cd")},
    # East and Southern Africa: the same pattern (eafrica_register.py; --clip <cc>, --fill;
    # handoff_notes/eafrica_build.md). Mauritius is all metro: no register (None).
    **{cc: f"eafrica_register:data/raw/rinf/{cc}"
       for cc in ("ke", "et", "dj", "mz", "zm", "zw", "tz", "mg", "mw", "ug")},
    "mu": None,
    # Latin America: the ways of hand-listed OSM routes (latam_register.py on ar_register's
    # code; `--clip <cc>` after every extract; handoff_notes/latam_build.md). Santo Domingo
    # and Puerto Rico are metro only: no register (None).
    **{cc: f"latam_register:data/raw/{cc}"
       for cc in ("cr", "pa", "cu", "pe", "bo", "ec", "uy", "ve", "co")},
    "do": None, "pr": None,
    # The rest of Asia: lk_register's engine (asia_register.py; --clip <cc>, --join, --fill;
    # handoff_notes/asia_build.md).
    **{cc: f"asia_register:data/raw/rinf/{cc}" for cc in ("kh", "la", "ph", "mm", "mn", "np")},
    # Israel: hand-listed infrastructure lines on OSM track, stops from the MOT feed's rail
    # subset (il_register.py on my_register's Track; `--clip` after an extract;
    # handoff_notes/il_build.md).
    "il": "il_register:data/raw/il",
    # North Korea: OSM named track + line relations (kp_register.py; `python kp_register.py
    # --clip` after an extract; kp_sources.md).
    "kp": "kp_register:data/raw/kp",
}
# Rough build_model + build_tiles minutes on 2026-10-01, for ordering only.
MINUTES = {"us": 12, "au": 11, "cn": 11, "ru": 9, "de": 8, "fr": 6, "it": 5, "jp": 4, "pl": 4,
           "es": 4, "in": 6, "gb": 6, "ca": 3, "se": 2, "no": 2, "ie": 1, "mx": 1, "th": 1, "my": 1, "id": 1, "ua": 2, "tr": 1, "nz": 1, "vn": 1, "za": 1, "br": 1, "ar": 2, "cl": 1, "ir": 1, "ch": 2, "at": 2, "cz": 2, "be": 1, "nl": 1, "pk": 1, "lk": 1, "bd": 1}


def run_country(cc, model_only, tiles_only=False):
    reg = REGISTER.get(cc, f"rinf:data/raw/rinf/{cc}")
    steps = [] if tiles_only else [["build_model.py", "--region", cc]
                                   + (["--register", reg] if reg else [])]
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
    tiles_only = "--tiles-only" in args
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
        results = list(pool.map(lambda cc: run_country(cc, model_only, tiles_only), regions))
    failed = [r for rs in results for r in rs if " exit 0," not in r]
    print(f"done: {len(regions)} countries in {(time.time() - t0) / 60:.1f} min, "
          f"{len(failed)} failed steps" + ("".join(f"\n  {f}" for f in failed)))


if __name__ == "__main__":
    main()
