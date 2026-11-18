"""Agents' fast Rust builds (T-3664): the `agent` cargo profile (sim/Cargo.toml: no LTO, incremental), with the
cores claimed in the ledger for the run and released after.

    python trainworld/tools/agent_cargo.py test    both suites: cargo test --profile agent, then with --features demand --lib
    python trainworld/tools/agent_cargo.py wasm    the tree's two WASM modules (npm run wasm:agent)
    python trainworld/tools/agent_cargo.py cargo <args>    any other cargo command, run from trainworld/sim

Run from anywhere; it works on the tree this file is in. Never ships: landing, 8800 and deploy use release/wasm.
"""

import os
import subprocess
import sys

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the tree's root
MAPS_CORES = None
for up in (TREE, os.path.dirname(os.path.dirname(TREE))):  # a worktree is <maps>/.tw-trees/<name>
    if os.path.exists(os.path.join(up, "cores.py")):
        MAPS_CORES = os.path.join(up, "cores.py")
        break
SIM = os.path.join(TREE, "trainworld", "sim")
APP = os.path.join(TREE, "trainworld", "app")
WANT = 3


def claim(name):
    r = subprocess.run([sys.executable, MAPS_CORES, "claim", str(WANT), name, "--pid", str(os.getpid()), "Rust build"],
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["1"])[-1]
    try:
        return max(1, min(WANT, int(last.split()[0])))
    except ValueError:
        return 1


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("test", "wasm", "cargo"):
        sys.exit(__doc__)
    name = "rust-" + os.path.basename(TREE)
    jobs = claim(name)
    env = dict(os.environ, CARGO_BUILD_JOBS=str(jobs))
    cargo_bin = os.path.join(os.environ.get("USERPROFILE", ""), ".cargo", "bin")
    env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    sh = os.name == "nt"
    rc = 0
    try:
        if sys.argv[1] == "test":
            for extra in ([], ["--features", "demand", "--lib"]):
                rc = subprocess.call(["cargo", "test", "--profile", "agent"] + extra, cwd=SIM, env=env, shell=sh)
                if rc:
                    break
        elif sys.argv[1] == "wasm":
            rc = subprocess.call(["npm", "run", "wasm:agent", "--prefix", APP], env=env, shell=sh)
        else:
            rc = subprocess.call(["cargo"] + sys.argv[2:], cwd=SIM, env=env, shell=sh)
    finally:
        subprocess.run([sys.executable, MAPS_CORES, "release", name], capture_output=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
