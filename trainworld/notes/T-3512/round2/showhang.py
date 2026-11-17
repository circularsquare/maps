"""Show a hang.mjs run's steps, its wait logs (page lines only, W duplicates dropped) and the hang dump: showhang.py name [regex]"""
import json, re, sys
OUT = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/out/"
d = json.load(open(OUT + sys.argv[1] + ".json", encoding="utf-8"))
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else re.compile(r"T3512|T1733|T1609|T1223|stopp|error|panic|disagree|whole|stale")
print("load", d.get("load"), "error", (d.get("error") or "")[:100], "errorsN", d.get("errorsN"), d.get("errors")[:3])
def show(log):
    seen = set()
    for t, x in log:
        if x.startswith("W "):
            continue
        if pat.search(x):
            print("    ", t, x[:230])
for s in d["steps"]:
    if s["op"] == "wait":
        print("WAIT rep", s.get("rep"), "settle", s["settle"], "rail", s["fig"]["rail"], "v", s["fig"]["version"])
        show(s["log"])
    else:
        print("OP", s.get("rep"), s["op"], s.get("res", "")[:100])
h = d.get("hang")
if h:
    print("HANG at", h["at"])
    print(json.dumps(h["dump"], indent=None)[:3000])
    show(h["log"])
