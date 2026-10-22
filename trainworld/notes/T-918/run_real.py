"""T-918: the real networks (calib, native) for each curve; skips outputs that exist."""
import os
import subprocess
import sys
import time

SC = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/8a7f07b9-e376-4102-ab13-dfd98a17993c/scratchpad/softcurve'
RB = os.environ.get('T918_RB') == '1'
EXE = f'{SC}/target_rb/release/calib.exe' if RB else f'{SC}/target/release/calib.exe'
TW = 'C:/Users/anita/projects/maps/trainworld'
OUT = f'{SC}/real_rb' if RB else f'{SC}/real'
VARS = {
    'today': None,
    'today1': 'curve=1,cv_a=1',
    'nob': 'rail_floor=0',
    'L07': 'curve=1,cv_a=0.7,rail_floor=0',
    'mix10': 'curve=2,cv_a=1.0942,cv_w=0.1,rail_floor=0',
    'mix20': 'curve=2,cv_a=1.2088,cv_w=0.2,rail_floor=0',
    'mix30': 'curve=2,cv_a=1.3507,cv_w=0.3,rail_floor=0',
    'atan': 'curve=3,cv_a=0.8875,rail_floor=0',
    'ratio3.5': 'curve=4,cv_g=3.476,rail_floor=0',
    'ratio5.5': 'curve=4,cv_g=5.485,rail_floor=0',
    'ratio3': 'curve=4,cv_g=3,rail_floor=0',
    'mix15': 'curve=2,cv_a=1.1486,cv_w=0.15,rail_floor=0',
}
os.makedirs(OUT, exist_ok=True)
cities = sys.argv[1].split(',') if len(sys.argv) > 1 else ['nyc', 'lon', 'par', 'tyo']
names = sys.argv[2].split(',') if len(sys.argv) > 2 else list(VARS)
env = dict(os.environ, TW_ROOT=TW)
for c in cities:
    for v in names:
        out = f'{OUT}/{c}_{v}.json'
        if os.path.exists(out):
            continue
        cmd = [EXE, '--net', f'{TW}/data/work/realnet/{c}_real.json', '--out', out]
        if VARS[v]:
            cmd += ['--set', VARS[v]]
        t = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(c, v, f'{time.time() - t:.0f}s', r.returncode, r.stderr.strip().splitlines()[-3:] if r.stderr else '', flush=True)
