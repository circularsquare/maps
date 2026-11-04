"""Add shown names (and aliases) to app/src/game/zone-names.json, one entry a line as the file keeps them.
setnames.py names.json   where names.json is {id: {"name": ..., "aliases": [...]}}; existing entries are
merged (aliases joined), never dropped."""
import json, os, sys
P = r"C:\Users\anita\projects\maps\trainworld\app\src\game\zone-names.json"
cur = json.load(open(P, encoding="utf-8"))
add = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in add.items():
    e = cur.setdefault(k, {})
    if v.get("name"):
        e["name"] = v["name"]
    al = e.get("aliases", []) + [a for a in v.get("aliases", []) if a not in e.get("aliases", []) and a != e.get("name")]
    if al:
        e["aliases"] = al
lines = ",\n".join(f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}" for k, v in cur.items())
tmp = P + ".part"
open(tmp, "w", encoding="utf-8", newline="\n").write("{\n" + lines + "\n}\n")
os.replace(tmp, P)
print(len(add), "entries set;", len(cur), "in all")
