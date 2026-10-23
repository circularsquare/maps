"""T-1065: the code bundle a wave's machine downloads (launch.py --file code=<out>). docs/aws.md.

    bundle.py <out.tgz> <wave plan.txt>

trainworld's pipeline/, sim/ (no target), checks/, tools/aws/ and app/src/game/*.json as they are in
the working tree (the plan and registry as runs left them; code edits not yet committed included),
git's tracked and untracked-but-not-ignored files only, plus the wave plan as wave/plan.txt.
Refuses a file that looks like it holds an AWS key. Makes no AWS call.
"""
import io
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TW = HERE.parents[1]
MAPS = TW.parent
DIRS = ["trainworld/pipeline", "trainworld/sim", "trainworld/checks", "trainworld/tools/aws", "trainworld/app/src/game"]
SKIP = re.compile(r"(^|/)(target|__pycache__|node_modules|pkg)(/|$)")
KEY = re.compile(rb"AKIA[0-9A-Z]{16}|aws_secret_access_key|AWS_SECRET_ACCESS_KEY\s*=")


def files():
    out = set()
    for flags in (["--cached"], ["--others", "--exclude-standard"]):
        p = subprocess.run(["git", "-C", str(MAPS), "ls-files", "-z", *flags, "--", *DIRS], capture_output=True, check=True)
        out |= {f for f in p.stdout.decode("utf-8").split("\0") if f}
    keep = []
    for f in sorted(out):
        if SKIP.search(f) or not (MAPS / f).is_file():
            continue
        if f.startswith("trainworld/app/src/game/") and (not f.endswith(".json") or "/" in f[len("trainworld/app/src/game/"):]):
            continue
        keep.append(f)
    return keep


def main(out, plan):
    fs = files()
    total = 0
    with tarfile.open(out + ".part", "w:gz") as t:
        for f in fs:
            data = (MAPS / f).read_bytes()
            if KEY.search(data) and not f.endswith(("guard.py", "bundle.py")):
                raise SystemExit(f"REFUSED: {f} looks like it holds an AWS key; nothing bundled")
            ti = tarfile.TarInfo(f)
            ti.size, ti.mtime, ti.mode = len(data), int((MAPS / f).stat().st_mtime), 0o755 if f.endswith(".sh") else 0o644
            t.addfile(ti, io.BytesIO(data))
            total += len(data)
        data = Path(plan).read_bytes().replace(b"\r\n", b"\n")
        ti = tarfile.TarInfo("wave/plan.txt")
        ti.size, ti.mtime = len(data), int(time.time())
        t.addfile(ti, io.BytesIO(data))
    os.replace(out + ".part", out)
    print(f"{out}: {len(fs)} files, {total / 1e6:.1f} MB raw, {os.path.getsize(out) / 1e6:.1f} MB gzipped; plan {plan}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
