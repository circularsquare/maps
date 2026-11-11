"""Find subagent transcripts whose first user message (the brief) names one of the given task ids,
and print weighted tokens, wall time and model for each. Usage: python find_agents.py T-1303 T-1305 ..."""
import glob
import json
import os
import re
import sys

sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\checks")
import token_usage as tu  # noqa: E402

ROOT = os.path.expanduser(r"~\.claude\projects\c--Users-anita-projects-maps")
ids = sys.argv[1:]
pat = re.compile(r"\b(" + "|".join(re.escape(i) for i in ids) + r")\b")


def brief(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message")
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, list):
                    c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                return str(c)[:3000]
    return ""


rows = []
for p in glob.glob(os.path.join(ROOT, "*", "subagents", "agent-*.jsonl")):
    b = brief(p)
    head = b[:int(os.environ.get("HEADN", "600"))]
    m = pat.findall(head)
    if not m:
        continue
    cs = tu.calls(p)
    if not cs:
        continue
    w = sum(tu.weighted(c[2]) for c in cs)
    t0, t1 = cs[0][0], cs[-1][0]
    models = sorted({c[1] for c in cs})
    rows.append((t0, sorted(set(m)), w, t1, models, os.path.basename(p), head[:160].replace("\n", " ")))
rows.sort()
for t0, m, w, t1, models, name, head in rows:
    print(f"{t0[:16]} -> {t1[11:16]}  {','.join(m):22s} {w/1e6:6.2f}M  {','.join(models)}  {name}\n    {head}")
