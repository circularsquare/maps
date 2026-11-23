"""The solve units the touched registered ids join, with member counts and countries (read only).
usage: groups.py a,b,c"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import world_run  # noqa: E402

reg = {c["id"]: c for c in json.load(open(os.path.join(TW, "app", "src", "game", "registry.json"), encoding="utf-8"))["cities"]}
for u in world_run.groups_of(sys.argv[1].split(",")):
    s = json.dumps(u, default=str)
    print(s[:300])
