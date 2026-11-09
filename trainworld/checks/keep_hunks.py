"""Stage only some hunks of one file's working-tree diff.

usage: python keep_hunks.py <repo> <path> <marker> [<marker> ...]
Keeps a hunk if any of its added lines contains a marker; applies the result to the index
(git apply --cached --recount). Prints each hunk's first added line and whether it was kept.
"""
import subprocess
import sys

repo, path, *markers = sys.argv[1:]
diff = subprocess.run(["git", "-C", repo, "diff", "-U0", "--", path], capture_output=True, check=True).stdout.decode("utf-8")
lines = diff.splitlines(keepends=True)
head, hunks, cur = [], [], None
for line in lines:
    if line.startswith("@@"):
        cur = [line]
        hunks.append(cur)
    elif cur is None:
        head.append(line)
    else:
        cur.append(line)
kept = []
for h in hunks:
    added = [l for l in h[1:] if l.startswith("+")]
    keep = any(m in l for l in added for m in markers)
    print(("KEEP " if keep else "skip ") + (added[0][:110].rstrip() if added else h[0].rstrip()))
    if keep:
        kept.append(h)
if not kept:
    sys.exit("nothing kept")
patch = "".join(head) + "".join("".join(h) for h in kept)
subprocess.run(["git", "-C", repo, "apply", "--cached", "--recount", "--unidiff-zero", "-"], input=patch.encode("utf-8"), check=True)
print(f"staged {len(kept)} of {len(hunks)} hunks of {path}")
