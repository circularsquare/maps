"""T-3240 proof: the scratch rows (new packs, pulled rule) against the shipped rows, both ways a weekday.
Near pairs from the rows' near lists (every 6 km zone pair added up); far ones by the rule at the centres."""
import json, math, sys
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
S = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\pull\root"
TW = r"C:\Users\anita\projects\maps\trainworld"
old = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
new = json.load(open(S + r"\rows.json", encoding="utf-8"))["rows"]
allnew = json.load(open(S + r"\app\src\game\market-rows.json", encoding="utf-8")); allnew.update(new)
rule0 = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["market"]["rule"]
rule1 = json.load(open(S + r"\app\src\game\registry.json", encoding="utf-8"))["market"]["rule"]
OPEN = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE NO IS CH LI GB".split())


def km(p, q):
    a, b = math.radians(p["lat"]), math.radians(q["lat"])
    h = math.sin((b - a) / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(math.radians(q["lon"] - p["lon"]) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(h, 1)))


def directed(R, p, q, d):
    on = lambda x, w: min(1, max(0, (x - w[0]) / (w[1] - w[0])))
    gr = on(d, R["onset_km"]) * (1 - on(d, R["tail_onset_km"])) / max(d, R["d0_km"]) ** R["A"]
    tl = on(d, R["tail_onset_km"]) / max(d, R["D1_km"]) ** R["G"]
    ip, iq = min(p["inc"], 1), min(q["inc"], 1)
    rel = min(iq / ip, 1) ** R["eps_t"]
    gen = R["border"] * ip ** R["border_eps"]
    b = 1 if p["country"] == q["country"] else (max(0.5, gen) if p["country"] in OPEN and q["country"] in OPEN else gen)
    w = 1
    px = q.get("px", 0)
    if px > 0 and R.get("pull_gamma") is not None and p["country"] not in R["pull_off"]:
        g = R["pull_gamma"] * R["pull_country_gamma"].get(q["country"], 1) if p["country"] == q["country"] else R["pull_gamma_x"]
        w = 1 + math.expm1(g * px) * on(d, R["pull_phase_km"])
    return b * w * (p["g"] * gr * q["a"] + p["t"] * tl * q["ta"] * rel)


def both(rows, R, x, y):
    tot = 0
    for yy in y.split("+"):
        p, q = rows[x], rows[yy]
        d = km(p, q)
        if d < R["near_km"]:
            t = next((v for j, v in p.get("near", []) if j == q["place"]), None)
            if t is None:
                t = next((v for j, v in q.get("near", []) if j == p["place"]), 0)
            tot += 2 * t
        else:
            tot += directed(R, p, q, d) + directed(R, q, p, d)
    return tot


for x, y, tg in [("war", "kra", "25,000"), ("war", "tro+gda+stz", "25,000"), ("war", "lod", ""), ("kra", "kat", ""), ("ber", "mun", "22,000"),
                 ("ham", "mun", "10,000"), ("ber", "ham", ""), ("ber", "war", ""), ("jnj", "nag", "6,000"), ("jnj", "yzo", ""), ("phl", "nyc", ""), ("phl", "wdc", ""), ("phl", "bal", "")]:
    print(f"| {x} - {y} | {both(old, rule0, x, y):,.0f} | {both(allnew, rule1, x, y):,.0f} | {tg} |")
for z in ["war", "kra", "osr", "olz", "kic", "ber", "jnj", "phl"]:
    o, n = old[z], new[z]
    near_o = sum(v for j, v in o["near"] if j != o["place"]); near_n = sum(v for j, v in n["near"] if j != n["place"])
    own_o = next((v for j, v in o["near"] if j == o["place"]), 0); own_n = next((v for j, v in n["near"] if j == n["place"]), 0)
    print(f"  {z}: departures {o['dep']:,.0f} -> {n['dep']:,.0f}; own {own_o:,.0f} -> {own_n:,.0f}; near places {near_o:,.0f} -> {near_n:,.0f}; g {n['g']/o['g']:.3f}x t {n['t']/o['t']:.3f}x; px {n.get('px', 0)}")
