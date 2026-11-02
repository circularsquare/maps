r"""A sim copy for a long run: HEAD's trainworld/sim in <folder>\simroot\sim, with junctions
<folder>\simroot\data and \app to this tree's, so a run's build and gate read a crate nobody is
editing (docs/world-run.md "Running the pipeline"; docs/adding-a-country.md 3).

    setup_sim.py <folder> [make|check|refresh|remove]

      make     (default) make the copy; refuses if <folder>\simroot\sim exists
      check    verify the junctions point at this tree's data and app
      refresh  replace sim/ with HEAD's again, keeping its target/ (the build cache)
      remove   take the junctions off with rmdir (never touching what they point at), then the copy

<folder> is yours (your scratchpad); nothing else is written outside it. make, check and refresh
print the two variables every run .cmd sets:

    TW_SIM_DIR=<folder>\simroot\sim
    TW_CARGO_TARGET_DIR=<folder>\simroot\sim\target

The maps root is found from this file's place (tools/country/ under trainworld/).
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = os.path.dirname(TW)
JUNCTIONS = ("data", "app")


def paths(folder):
    root = os.path.join(os.path.abspath(folder), "simroot")
    return root, os.path.join(root, "sim")


def extract(sim):
    tar = subprocess.run(["git", "-C", MAPS, "archive", "HEAD", "trainworld/sim"],
                         check=True, capture_output=True).stdout
    n = 0
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            rel = m.name[len("trainworld/sim"):].lstrip("/")
            if not rel or not m.isfile():
                continue
            out = os.path.join(sim, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(t.extractfile(m).read())
            n += 1
    return n


def check(root, sim):
    ok = True
    for name in JUNCTIONS:
        p = os.path.join(root, name)
        real = os.path.realpath(p)
        want = os.path.realpath(os.path.join(TW, name))
        good = os.path.isdir(p) and os.path.normcase(real) == os.path.normcase(want)
        probe = os.path.join(p, "src", "game", "registry.json") if name == "app" else os.path.join(p, "packs")
        good = good and os.path.exists(probe)
        print(f"{name}: {p} -> {real} {'OK' if good else 'BAD'}")
        ok = ok and good
    has = os.path.exists(os.path.join(sim, "Cargo.toml"))
    print("sim Cargo.toml:", has)
    return ok and has


def say_env(sim):
    print(f"TW_SIM_DIR={sim}")
    print(f"TW_CARGO_TARGET_DIR={os.path.join(sim, 'target')}")


def make(root, sim):
    if os.path.exists(sim):
        sys.exit(f"{sim} exists (use refresh, or remove first)")
    os.makedirs(root, exist_ok=True)
    n = extract(sim)
    print(f"{n} files of HEAD's sim/ -> {sim}")
    for name in JUNCTIONS:
        subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(root, name), os.path.join(TW, name)],
                       check=True, stdout=subprocess.DEVNULL)


def refresh(sim):
    if not os.path.isdir(sim):
        sys.exit(f"{sim} does not exist (use make)")
    for name in os.listdir(sim):
        if name == "target":
            continue
        p = os.path.join(sim, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    print(f"{extract(sim)} files of HEAD's sim/ -> {sim}")


def remove(root):
    # rmdir takes a junction off without following it; rmtree on the folder first could follow it
    for name in JUNCTIONS:
        p = os.path.join(root, name)
        if os.path.lexists(p):
            os.rmdir(p)
            print("junction removed:", p)
    if os.path.isdir(root):
        shutil.rmtree(root)
        print("removed", root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="your own folder; the copy goes in <folder>\\simroot")
    ap.add_argument("action", nargs="?", default="make", choices=["make", "check", "refresh", "remove"])
    a = ap.parse_args()
    root, sim = paths(a.folder)
    if a.action == "remove":
        remove(root)
        return
    if a.action == "make":
        make(root, sim)
    elif a.action == "refresh":
        refresh(sim)
    ok = check(root, sim)
    say_env(sim)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
