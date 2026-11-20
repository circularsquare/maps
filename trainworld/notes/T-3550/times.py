"""times.py <tag> [<tag2>]: per-edit ms from sweep logs; with two tags, side by side and the ratio."""
import os, re, sys, statistics

S = os.path.dirname(os.path.abspath(__file__))
STEPS = ["count", "count_undo", "count2", "count2_undo"]


def read(tag):
    out = {}
    d = os.path.join(S, tag)
    for f in os.listdir(d):
        if not f.endswith(".log"):
            continue
        key = f[:-4]
        r = {}
        name = ""
        for line in open(os.path.join(d, f), encoding="utf-8", errors="replace"):
            m = re.match(r"^(\w+)\s+([\d.]+) ms  r=", line)
            if m:
                r[m.group(1)] = float(m.group(2))
            m = re.match(r"^\d+ lines; line \d+ (\".*?\")", line)
            if m:
                name = m.group(1)
        out[key] = (name, r)
    return out


def summary(label, vals):
    vals = sorted(vals)
    if not vals:
        return f"{label}: none"
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    return f"{label}: n={len(vals)} median {q(0.5):.0f} p90 {q(0.9):.0f} max {vals[-1]:.0f} ms; over 500 ms: {sum(v > 500 for v in vals)}"


a = read(sys.argv[1])
b = read(sys.argv[2]) if len(sys.argv) > 2 else None
keys = sorted(a, key=lambda k: (k.split("-")[0], int(k.split("-")[1])))
slow = [k for k in keys if k.startswith("lon") and a[k][1].get("count", 0) > 100]
for k in keys:
    name, r = a[k]
    row = f"{k:<8} {name[:22]:<24}" + " ".join(f"{r.get(s, float('nan')):8.0f}" for s in STEPS)
    if b and k in b:
        rb = b[k][1]
        row += "  |" + " ".join(f"{rb.get(s, float('nan')):8.0f}" for s in STEPS)
    if k.startswith("nyc") or k in slow or len(sys.argv) > 3:
        print(row)
print(f"London lines whose +1 medium took over 100 ms: {len(slow)}")
for s in STEPS:
    print(summary(f"{sys.argv[1]} {s} (those lines)", [a[k][1][s] for k in slow if s in a[k][1]]))
    if b:
        print(summary(f"{sys.argv[2]} {s} (those lines)", [b[k][1][s] for k in slow if k in b and s in b[k][1]]))
print(summary(f"{sys.argv[1]} every London step", [v for k in keys if k.startswith("lon") for v in a[k][1].values()]))
if b:
    print(summary(f"{sys.argv[2]} every London step", [v for k in keys if k.startswith("lon") and k in b for v in b[k][1].values()]))
