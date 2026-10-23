"""Undo a T-897 join from its backup: restore.py <wave> [--go]
Copies every file of data/work/pre-T897/<wave>/ back to trainworld/, and removes the joint files
(data/work/*.joint_commutes.json, .joint_report.json, .joint_dump.json) written after the backup that
the backup does not hold. Without --go it only lists what it would do."""
import glob, os, shutil, sys
TW = r"C:\Users\anita\projects\maps\trainworld"
wave, go = sys.argv[1], "--go" in sys.argv
src = os.path.join(TW, "data", "work", "pre-T897", wave)
t0 = os.path.getmtime(os.path.join(src, "app", "src", "game", "registry.json"))
n = 0
for root, _, files in os.walk(src):
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, src)
        n += 1
        if go:
            dst = os.path.join(TW, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst + ".part")
            os.replace(dst + ".part", dst)
print(n, "files", "restored" if go else "to restore")
held = {os.path.basename(p) for p in glob.glob(os.path.join(src, "data", "work", "*.joint_*.json"))}
for pat in ("*.joint_commutes.json", "*.joint_report.json", "*.joint_dump.json"):
    for p in glob.glob(os.path.join(TW, "data", "work", pat)):
        if os.path.basename(p) not in held and os.path.getmtime(p) > t0 - 60:
            print("remove" if go else "would remove", os.path.basename(p))
            if go:
                os.remove(p)
