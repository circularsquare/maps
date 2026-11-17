"""Rebuild 8800 at a commit: runs checks/snapbuild.cmd (or snapbuild-frozen.cmd with a freeze) from the maps root,
then writes the new build id to notes/board-live.json (tasks.py set --build).

Usage: venv\Scripts\python.exe trainworld/tools/ops/snap8800.py <commit> [<freeze>] [--log <file>]
"""
import subprocess
import sys
from pathlib import Path

MAPS = Path(__file__).resolve().parents[3]


def main(argv):
    log = None
    if "--log" in argv:
        i = argv.index("--log")
        log = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if not argv or len(argv) > 2:
        print(__doc__)
        return 2
    script = "snapbuild.cmd" if len(argv) == 1 else "snapbuild-frozen.cmd"
    cmd = ["cmd", "/c", str(Path("trainworld") / "checks" / script), *argv]
    out = open(log, "w", encoding="utf-8", errors="replace") if log else None
    try:
        # CREATE_NO_WINDOW: started from a console-less parent, cmd would otherwise open a window on her screen
        rc = subprocess.run(cmd, cwd=str(MAPS), stdout=out, stderr=subprocess.STDOUT if out else None,
                            creationflags=0x08000000).returncode
    finally:
        if out:
            out.close()
    if log:
        print(Path(log).read_text(encoding="utf-8", errors="replace")[-300:])
    if rc == 0:
        # the board's Builds row shows what 8800 runs
        subprocess.run([sys.executable, str(MAPS / "trainworld" / "tools" / "tasks.py"), "set", "--build", f"8800={argv[0][:8]}"],
                       cwd=str(MAPS))
    print(f"exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
