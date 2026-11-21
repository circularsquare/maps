"""T-3887: the mode frontier, on a made-up city of even density. No game is run: the game's own Rust (track model,
demand kernel, trip explorer) runs natively in sim/src/bin/frontier.rs on that city, one straight line of one mode at
a time, and the fares come from the game's own TypeScript (tools/frontier_fares.ts). Method and findings:
notes/T-3887.md.

    python trainworld/tools/frontier.py                 build, run every mode x spacing x service, draw both charts
    python trainworld/tools/frontier.py run [--only bus:400,tram:800] [--services A]
    python trainworld/tools/frontier.py charts [result.json]

`run` writes result.json beside the charts (artifacts/trainworld/T-3887/); `charts` draws share.png and profit.png
from it. Modes, spacings, trip lengths, services and the city are the tables below.
"""
import argparse, json, os, subprocess, sys

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the maps root of this tree
TW = os.path.join(TREE, "trainworld")
SIM = os.path.join(TW, "sim")
# the maps root that holds the shared artifacts (a worktree is <maps>/.tw-trees/<name>)
MAPS = os.path.dirname(os.path.dirname(TREE)) if os.path.basename(os.path.dirname(TREE)) == ".tw-trees" else TREE
ART = os.path.join(MAPS, "artifacts", "trainworld", "T-3887")
SCRATCH = os.environ.get("TW_FRONTIER_SCRATCH", os.path.join(os.environ.get("TEMP", "."), "tw-frontier"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The made-up city: one density everywhere. 10,000 residents a km² (New York City's average, a third of Paris's);
# jobs, commuters and everyday attractors per resident as Chicago's pack has them (US ratios). 120 by 60 km, so the
# 60 km line has 30 km of city past each end and to each side: an edge closer than that crowds trips onto the line
# (80 by 30 km gave it 14-23% more riders than 80 by 60, and that 4-6% more than 120 by 60).
CITY = dict(width_km=120, height_km=60, residents_km2=10000, jobs_km2=4938, commuters_per_resident=0.4173,
            errand_jobs_per_resident=0.0741, leisure_jobs_per_resident=0.0395, country="US",
            everyday_rate={"errands": 0.577, "social": 0.424, "medical": 0.044}, origin=[-87.6313, 41.8987])
LINE_KM = 60
ROUNDS = 3  # crowding rounds, as the demand client (demandClient.ts ROUNDS)
AMORTIZE_DAYS = 60  # building cost spread over 60 game days, as the first run (the loan's term)

# mode: level, track and train flags, cars, platform m. Each mode runs at every spacing; a bus at 400 m is its own
# placing (stops on the 400 m street grid), wider its player spacing.
MODES = {
    "bus":   dict(bus=True, level=0),
    "tram":  dict(tram=True, level=0, cars=2, platform=40),
    "metro": dict(level=-2, cars=6, platform=120),
    "rail":  dict(level=1, cars=8, platform=160),
    "hsr":   dict(level=1, hsr=True, fast=True, cars=8, platform=200),
}
NAMES = {"bus": "Bus", "tram": "Tram", "metro": "Metro", "rail": "Rail", "hsr": "High speed"}
SPACINGS = [400, 800, 1500, 3000, 6000, 12000]  # m
TRIP_KM = [2, 5, 15, 40]
# headways s at high, medium and low demand: the bus's new-line default, twice as often, four times
SERVICES = {"A": [480, 900, 1800], "B": [240, 480, 900], "C": [120, 240, 480]}
# both ends of a trip the same distance to one side of the line: on it, and one and two rows of cells off (cells are
# 324 m, centred on the line), weighted by the strip of people each row stands for, so the shares are of people
# living within 810 m of the line; its start at 12 points spread over one stop spacing
TRIPS = dict(lengths_km=TRIP_KM, phases=12, offsets_m=[0, 324, 648], offset_weights=[0.5, 1, 1])


def fares():
    """The game's fares for a new game in the city's country, by its own functions (tools/frontier_fares.ts)."""
    os.makedirs(SCRATCH, exist_ok=True)
    out = os.path.join(SCRATCH, "fares.mjs")
    if os.path.exists(out):
        os.remove(out)
    cli = os.path.join(TW, "app", "node_modules", "rolldown", "bin", "cli.mjs")
    subprocess.run(["node", cli, os.path.join(TW, "tools", "frontier_fares.ts"), "--platform", "node", "--format", "esm", "-o", out],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    r = subprocess.run(["node", out, CITY["country"]], check=True, capture_output=True, text=True, creationflags=NO_WINDOW)
    return json.loads(r.stdout.strip().splitlines()[-1])


def cmd_run(a):
    os.makedirs(ART, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    runs = []
    for mode, m in MODES.items():
        for sp in SPACINGS:
            if only and f"{mode}:{sp}" not in only:
                continue
            for s in a.services.split(","):
                runs.append(dict(mode=mode, spacing_m=sp, service=s, headways=SERVICES[s], **m))
    spec = dict(city=CITY, fares=fares(), line_km=LINE_KM, rounds=ROUNDS, trips=TRIPS, runs=runs)
    path = os.path.join(SCRATCH, "spec.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    exe = os.path.join(SIM, "target", "agent", "frontier.exe" if os.name == "nt" else "frontier")
    if not a.no_build:
        subprocess.run([sys.executable, os.path.join(TW, "tools", "agent_cargo.py"), "cargo", "build", "--profile", "agent",
                        "--features", "demand", "--bin", "frontier"], check=True, creationflags=NO_WINDOW)
    log = os.path.splitext(a.out)[0] + ".log"
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run([exe, "--spec", path, "--out", a.out], check=True, creationflags=NO_WINDOW, stderr=f)
    print(f"-> {a.out} (log {log})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--only", default=None, help="mode:spacing,... (e.g. bus:400,tram:800)")
    r.add_argument("--services", default=",".join(SERVICES))
    r.add_argument("--out", default=os.path.join(ART, "result.json"))
    r.add_argument("--no-build", action="store_true")
    c = sub.add_parser("charts")
    c.add_argument("result", nargs="?", default=os.path.join(ART, "result.json"))
    a = ap.parse_args()
    if a.cmd in (None, "run"):
        if a.cmd is None:
            a = argparse.Namespace(cmd=None, only=None, services=",".join(SERVICES), out=os.path.join(ART, "result.json"), no_build=False)
        cmd_run(a)
    if a.cmd in (None, "charts"):
        from frontier_chart import charts
        charts(getattr(a, "result", None) or a.out, ART)


if __name__ == "__main__":
    main()
