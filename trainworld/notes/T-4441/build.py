import os, subprocess, sys
TREE = r"C:\Users\anita\projects\maps\.tw-trees\trainparts"
OUT = sys.argv[1]
r = subprocess.run([sys.executable, os.path.join(TREE, "cores.py"), "claim", "3", "rust-trainparts", "--pid", str(os.getpid()), "--min", "2", "wasm build for T-4441"], stdout=subprocess.PIPE, text=True)
jobs = (r.stdout.strip().splitlines() or ["1"])[-1].split()[0]
env = dict(os.environ, CARGO_BUILD_JOBS=jobs, BINARYEN_CORES=jobs)
env["PATH"] = os.path.join(os.environ["USERPROFILE"], ".cargo", "bin") + os.pathsep + env["PATH"]
try:
    rc = subprocess.call(["wasm-pack", "build", "../sim", "--target", "web", "--profile", "agent", "--out-dir", OUT, "--out-name", "sim"], cwd=os.path.join(TREE, "trainworld", "app"), env=env, shell=True)
finally:
    subprocess.run([sys.executable, os.path.join(TREE, "cores.py"), "release", "rust-trainparts"], capture_output=True)
sys.exit(rc)
