"""The sim crate's binaries, run or built through cargo for the pipeline (T-1114).

`root` is the caller's tree root, taken at call time so a `world_run.py --root` step builds into
that tree. TW_SIM_DIR names a copy of sim/ to build from while the tree's own is mid-edit, with
TW_CARGO_TARGET_DIR a target of its own so the shared binaries are not overwritten.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def env(root, jobs="2"):
    e = dict(os.environ)
    e["PATH"] = os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + e.get("PATH", "")
    e.setdefault("CARGO_BUILD_JOBS", jobs)
    # its own target dir, so the pipeline never waits on (or blocks) the app's wasm builds; under
    # the root, not WORK, so a scratch build that points WORK elsewhere still shares the compiled crate
    e["CARGO_TARGET_DIR"] = os.environ.get("TW_CARGO_TARGET_DIR", os.path.join(root, "data", "work", "target"))
    return e


def _cargo(e):
    cargo = shutil.which("cargo", path=e["PATH"])  # Windows does not search env's PATH itself
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    return cargo


def _manifest(root):
    return os.path.join(os.environ.get("TW_SIM_DIR", os.path.join(root, "sim")), "Cargo.toml")


def run(root, name, args, **kw):
    """`cargo run` the binary `name` with `args`; `kw` goes to subprocess.run (check is always on)."""
    e = env(root)
    return subprocess.run([_cargo(e), "run", "--quiet", "--release", "--features", "demand", "--bin", name,
                           "--manifest-path", _manifest(root), "--", *args], env=e, check=True, **kw)


def exe(root, name, jobs="3"):
    """The binary `name`'s path, built first if its code changed."""
    e = env(root, jobs)
    subprocess.run([_cargo(e), "build", "--quiet", "--release", "--features", "demand", "--bin", name,
                    "--manifest-path", _manifest(root)], env=e, check=True)
    return Path(e["CARGO_TARGET_DIR"]) / "release" / (name + (".exe" if os.name == "nt" else ""))
