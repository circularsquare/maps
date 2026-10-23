"""Make the T-897 sim copy: HEAD trainworld/sim, with
data and app junctions beside it. Run once; `check` verifies the junctions."""
import io, os, shutil, subprocess, sys, tarfile

MAPS = r"C:\Users\anita\projects\maps"
TW = os.path.join(MAPS, "trainworld")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "simroot")
SIM = os.path.join(ROOT, "sim")


def check():
    ok = True
    for name in ("data", "app"):
        p = os.path.join(ROOT, name)
        real = os.path.realpath(p)
        want = os.path.realpath(os.path.join(TW, name))
        good = os.path.isdir(p) and os.path.normcase(real) == os.path.normcase(want)
        reg = os.path.join(p, "src", "game", "registry.json") if name == "app" else os.path.join(p, "packs")
        good = good and os.path.exists(reg)
        print(f"{name}: {p} -> {real} {'OK' if good else 'BAD'}")
        ok = ok and good
    print("sim Cargo.toml:", os.path.exists(os.path.join(SIM, "Cargo.toml")))
    return ok


def refresh():
    """HEAD's sim/ again, keeping the build's target folder."""
    for name in os.listdir(SIM):
        if name == "target":
            continue
        p = os.path.join(SIM, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    extract()


def extract():
    tar = subprocess.run(["git", "-C", MAPS, "archive", "HEAD", "trainworld/sim"],
                         check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            rel = m.name[len("trainworld/sim"):].lstrip("/")
            if not rel or not m.isfile():
                continue
            out = os.path.join(SIM, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(t.extractfile(m).read())


def make():
    if os.path.exists(SIM):
        sys.exit(f"{SIM} exists")
    os.makedirs(ROOT, exist_ok=True)
    tar = subprocess.run(["git", "-C", MAPS, "archive", "HEAD", "trainworld/sim"],
                         check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            rel = m.name[len("trainworld/sim"):].lstrip("/")
            if not rel or not m.isfile():
                continue
            out = os.path.join(SIM, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(t.extractfile(m).read())
    for name in ("data", "app"):
        subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(ROOT, name), os.path.join(TW, name)],
                       check=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        sys.exit(0 if check() else 1)
    if sys.argv[1:] == ["refresh"]:
        refresh()
        sys.exit(0 if check() else 1)
    make()
    check()
