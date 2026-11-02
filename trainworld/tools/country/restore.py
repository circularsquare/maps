"""Put back what `backup.py` saved, undoing a join (or a run's registrations).

    restore.py <task> <stage> [--go] [--tw <trainworld folder>]

    restore.py T1990 prejoin          list what would be restored
    restore.py T1990 prejoin --go     restore it

Copies every file of data/work/pre-<task>/<stage>/ back to trainworld/ (each through a .part file and
an atomic replace), and removes the joint files (data/work/*.joint_commutes.json, .joint_report.json,
.joint_dump.json) written after the backup that it does not hold. Without --go it only lists.
Run it only while no other run or join writes data/packs or data/terrain, which are the files it
overwrites. --tw points at another tree of the same layout (a scratch tree).
"""
import argparse
import glob
import os
import shutil
import sys

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("stage")
    ap.add_argument("--go", action="store_true", help="do it (without, list only)")
    ap.add_argument("--tw", default=TW, help="trainworld folder (default: this tree's)")
    a = ap.parse_args()
    tw = os.path.abspath(a.tw)
    src = os.path.join(tw, "data", "work", f"pre-{a.task}", a.stage)
    marker = os.path.join(src, "app", "src", "game", "registry.json")
    if not os.path.exists(marker):
        sys.exit(f"{src} holds no backup (no app/src/game/registry.json in it)")
    t0 = os.path.getmtime(marker)
    n = 0
    for root, _, files in os.walk(src):
        for f in files:
            p = os.path.join(root, f)
            n += 1
            if a.go:
                dst = os.path.join(tw, os.path.relpath(p, src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(p, dst + ".part")
                os.replace(dst + ".part", dst)
    print(n, "files", "restored" if a.go else "to restore")
    held = {os.path.basename(p) for p in glob.glob(os.path.join(src, "data", "work", "*.joint_*.json"))}
    for pat in ("*.joint_commutes.json", "*.joint_report.json", "*.joint_dump.json"):
        for p in glob.glob(os.path.join(tw, "data", "work", pat)):
            if os.path.basename(p) not in held and os.path.getmtime(p) > t0 - 60:
                print("remove" if a.go else "would remove", os.path.basename(p))
                if a.go:
                    os.remove(p)


if __name__ == "__main__":
    main()
