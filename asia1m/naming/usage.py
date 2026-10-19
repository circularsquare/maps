"""Log what a naming agent cost -> usage.csv (one row per batch run; a rerun replaces its row).

    python asia1m/naming/usage.py <batch> <round> <agent transcript .jsonl>

The transcript is ~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl. Tokens are
summed once per API request: fresh input, cache writes, cache reads (cheap, roughly a tenth of
fresh input) and output. Tool calls are counted by name so web searches and fetches show.
"""
import os, sys, csv, json
from collections import Counter
from datetime import datetime
HERE = os.path.dirname(os.path.abspath(__file__))
batch, rnd, path = sys.argv[1], sys.argv[2], sys.argv[3]

per_req = {}; tools = Counter(); models = {}; times = []
for line in open(path, encoding="utf-8"):
    try: d = json.loads(line)
    except ValueError: continue
    if d.get("timestamp"): times.append(d["timestamp"])
    m = d.get("message")
    if d.get("type") != "assistant" or not isinstance(m, dict): continue
    for c in m.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "tool_use":
            tools[c.get("name")] += 1
    rid = d.get("requestId") or m.get("id")
    if not m.get("usage"): continue
    # a request's blocks repeat its usage; early blocks carry partial output counts, so keep the max
    u = m["usage"]; cur = per_req.setdefault(rid, Counter())
    for k, f in [("input", "input_tokens"), ("cache_write", "cache_creation_input_tokens"),
                 ("cache_read", "cache_read_input_tokens"), ("output", "output_tokens")]:
        cur[k] = max(cur[k], u.get(f, 0) or 0)
    models[rid] = m.get("model")
tok = Counter()
for c in per_req.values(): tok.update(c)
seen = per_req
models = Counter(models.values())

ts = sorted(datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times)
facts = json.load(open(os.path.join(HERE, "batches", batch, "facts.json"), encoding="utf-8"))
row = dict(batch=batch, round=rnd, model=models.most_common(1)[0][0] if models else "",
           regions=len(facts["regions"]), requests=len(seen),
           input=tok["input"], cache_write=tok["cache_write"], cache_read=tok["cache_read"], output=tok["output"],
           total=sum(tok.values()), per_region=round(sum(tok.values()) / max(len(facts["regions"]), 1)),
           websearch=tools["WebSearch"], webfetch=tools["WebFetch"], bash=tools["Bash"], read=tools["Read"],
           # working time: gaps over 10 min are waits between rounds, not work
           minutes=round(sum(min(g, 600) if g < 600 else 0 for g in
                             ((b - a).total_seconds() for a, b in zip(ts, ts[1:]))) / 60, 1) if ts else "",
           date=ts[0].date().isoformat() if ts else "")
out = os.path.join(HERE, "usage.csv")
rows = list(csv.DictReader(open(out, encoding="utf-8"))) if os.path.exists(out) else []
rows = [r for r in rows if not (r["batch"] == batch and r["round"] == rnd)] + [row]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerows(rows)
print(", ".join(f"{k}={v}" for k, v in row.items()))
