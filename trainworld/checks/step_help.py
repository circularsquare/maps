"""T-2398: `world_run.py step <name> --help` (and -h) prints help and never runs the step.

    ..\\venv\\Scripts\\python.exe checks/step_help.py      (from trainworld; a few seconds)

1. step_main with --help or -h never reaches run_main (a stand-in that fails if called).
2. Every pipeline/ and checks/ script with a __main__ block, run for real as `step <name> --help`:
   exit 0, says it was not run, and the git status of the tree is the same before and after.
"""
import subprocess
import sys
from pathlib import Path

TW = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
fails = 0


def report(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def status():
    return subprocess.run(["git", "-C", str(TW), "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout


def stand_in():
    sys.path.insert(0, str(TW / "pipeline"))
    import world_run
    for flag in ("--help", "-h"):
        def boom(*a):
            raise AssertionError("step ran")
        world_run.run_main = boom
        try:
            world_run.step_main(["seams", flag])
            report(f"step_main {flag} does not run", True)
        except BaseException as e:
            report(f"step_main {flag} does not run", False, repr(e))


def every_step():
    names = sorted(p.stem for d in ("pipeline", "checks") for p in (TW / d).glob("*.py")
                   if "__main__" in p.read_text(encoding="utf-8", errors="replace"))
    before = status()
    bad = []
    for n in names:
        for flag in ("--help", "-h"):
            r = subprocess.run([str(PY), "-I", str(TW / "pipeline/world_run.py"), "step", n, flag],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            if r.returncode != 0 or "was not run" not in r.stdout:
                bad.append(f"{n} {flag} (exit {r.returncode})")
    report(f"{len(names)} steps print help on --help and -h", not bad, ", ".join(bad[:10]))
    report("tree unchanged", status() == before)


stand_in()
every_step()
sys.exit(1 if fails else 0)
