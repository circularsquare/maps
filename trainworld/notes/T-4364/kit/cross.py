"""Commutes crossing zone lines, now and with zones merged (read-only; the joint solve's figures in the packs).
usage: cross.py a,b,c d,e ...   each argument a group of registered ids merged into one zone
Per member: within (gravity.trips), out and in (gravity.joint), its crossing share (out + in) / (within + out + in).
Per group: the commutes between its members (the market packs' commute lists, home pack to place) become within,
so the merged zone's crossing share is worked out exactly for a merge (a split or unit move needs a new solve)."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
PACKS = os.path.join(TW, "data", "packs")
reg = {c["id"]: c for c in json.load(open(os.path.join(TW, "app", "src", "game", "registry.json"), encoding="utf-8"))["cities"]}


def head(cid):
    g = json.load(open(os.path.join(PACKS, f"{cid}.json"), encoding="utf-8")).get("gravity") or {}
    j = g.get("joint") or {}
    return g.get("trips", 0.0), j.get("commuters_out", 0.0), j.get("commuters_in", 0.0)


def flows(cid, places):
    m = json.load(open(os.path.join(PACKS, f"{cid}.market.json"), encoding="utf-8"))
    own = m["place"]
    return sum(t for p, t in zip(m["commute_place"], m["commute_trips"]) if p in places and p != own)


for arg in sys.argv[1:]:
    ids = arg.split(",")
    places = {reg[i]["place"] for i in ids}
    W = O = I = 0.0
    for i in ids:
        w, o, n = head(i)
        W, O, I = W + w, O + o, I + n
        print(f"  {i} {reg[i]['name']}: within {w:,.0f}, out {o:,.0f}, in {n:,.0f}, crossing {(o + n) / max(1, w + o + n):.3f}")
    between = sum(flows(i, places) for i in ids)  # each pair counted once, at its home end
    w2, o2, i2 = W + between, O - between, I - between
    print(f"{arg}: crossing {(O + I) / max(1, W + O + I):.3f} summed over members -> {(o2 + i2) / max(1, w2 + o2 + i2):.3f} merged "
          f"({between:,.0f} commutes a day between members become within)")
