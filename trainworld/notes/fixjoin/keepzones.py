"""T-1091 for the packs that join nothing: back up, pack_gravity --keep-zones in place, read_pack.
keepzones.py <threads> ids..."""
import os, shutil, subprocess, sys, time
TW = r"C:\Users\anita\projects\maps\trainworld"
PY = r"C:\Users\anita\projects\maps\venv\Scripts\python.exe"
EXE = os.path.join(TW, r"data\work\target\release\pack_gravity.exe")
threads, ids = sys.argv[1], sys.argv[2:]
bak = os.path.join(TW, r"data\work\pre-fixjoin\unjoined\data\packs")
os.makedirs(bak, exist_ok=True)
env = dict(os.environ, RAYON_NUM_THREADS=threads, OMP_NUM_THREADS=threads)
bad = []
for i in ids:
    for ext in (".json", ".bin"):
        src, dst = os.path.join(TW, "data", "packs", i + ext), os.path.join(bak, i + ext)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    t0 = time.time()
    p = os.path.join(TW, "data", "packs", i + ".json")
    r = subprocess.run([EXE, p, p, "--keep-zones"], env=env, capture_output=True, text=True)
    t1 = time.time()
    q = subprocess.run([PY, "-I", os.path.join(TW, "pipeline", "read_pack.py"), i], env=env, capture_output=True, text=True, cwd=TW)
    tail = (q.stdout + q.stderr).strip().splitlines()[-3:]
    ok = r.returncode == 0 and q.returncode == 0
    print(f"{i}: pack_gravity {'ok' if r.returncode == 0 else 'FAILED'} {t1 - t0:.0f} s; read_pack {'ok' if q.returncode == 0 else 'FAILED'}: {' | '.join(tail)}", flush=True)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
    if not ok:
        bad.append(i)
print("failed:", bad or "none")
