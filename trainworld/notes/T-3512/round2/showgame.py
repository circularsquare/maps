"""The game-side records of a hang.mjs run: each wait's and the hang's GAME, and the snapshots: showgame.py name"""
import json, sys
OUT = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/out/"
d = json.load(open(OUT + sys.argv[1] + ".json", encoding="utf-8"))
print("startedAt", d.get("startedAt"))
for s in d["steps"]:
    if s["op"] == "wait":
        print("WAIT", s["settle"], "v", s["fig"]["version"], json.dumps(s.get("game"))[:900])
    else:
        print("OP", s["op"], "at", s.get("at"), s.get("res", "")[:80])
h = d.get("hang")
if h:
    print("HANG game", json.dumps(h.get("game"))[:1500])
    print("view", h["dump"].get("view"), "client.version", h["dump"]["client"].get("version"), "waiting", h["dump"]["client"].get("waiting"))
