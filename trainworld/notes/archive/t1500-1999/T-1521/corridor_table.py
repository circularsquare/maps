"""T-1521 round 3: the corridor runs as a table, rail / air / car of the longer trips, then rail of rail +
air, per variant, against the published figures; and the error in log-odds over the ten pairs."""
import json, math, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "trainworld", "T-1521", "corridor")
PUB = {  # rail of rail + air %, (rail, air, car) of all where published
    ("sha", "bei"): (86, None), ("bei", "gua"): (25, None), ("tyo", "osa"): (85, (64, 11, 25)), ("tyo", "fuk"): (11, None),
    ("par", "lyo"): (90, None), ("par", "mar"): (77, None), ("mad", "bar"): (78, (64, 18, 18)), ("lon", "cit"): (45, None),
    ("nyc", "wdc"): (79, (20, 5, 64)), ("nyc", "bos"): (53, (5, 4, 87)),
}
NAMES = {"sha": "Shanghai", "bei": "Beijing", "gua": "Guangzhou", "tyo": "Tokyo", "osa": "Osaka", "fuk": "Fukuoka", "ngo": "Nagoya",
         "par": "Paris", "lyo": "Lyon", "mar": "Marseille", "mad": "Madrid", "bar": "Barcelona", "lon": "London", "cit": "Edinburgh",
         "ber": "Berlin", "ham": "Hamburg", "nyc": "New York", "wdc": "Washington", "bos": "Boston", "phl": "Philadelphia"}
VARS = ["before", "built", "plr", "upkeep"]
res = {}
for v in VARS:
    for case in ["china", "jpn", "eu", "nec"]:
        p = os.path.join(OUT, f"{case}_{v}.json")
        if not os.path.exists(p):
            continue
        for q in json.load(open(p))["pairs"]:
            if not q.get("missing"):
                res[(tuple(q["pair"]), v)] = q

def cell(q):
    r, a, c = q["shares"]
    ra = r / (r + a) if r + a > 0 else 0
    return f"{100*r:.0f}/{100*a:.0f}/{100*c:.0f}, {100*ra:.0f}%"

lo = lambda p: math.log(min(max(p, .005), .995) / (1 - min(max(p, .005), .995)))
pairs = list(PUB) + sorted({k for k, _ in res} - set(PUB))
print("| Pair | km | trips | " + " | ".join(VARS) + " | Published |")
print("|---|---|---|" + "---|" * (len(VARS) + 1))
for k in pairs:
    qs = [res.get((k, v)) for v in VARS]
    if not any(qs):
        continue
    q0 = next(q for q in qs if q)
    pub = PUB.get(k)
    ptxt = "" if not pub else (f"{pub[0]}%" + (f" ({'/'.join(map(str, pub[1]))})" if pub[1] else ""))
    print(f"| {NAMES[k[0]]} - {NAMES[k[1]]} | {q0['km']:.0f} | {q0['trips']:,.0f} | " + " | ".join(cell(q) if q else "" for q in qs) + f" | {ptxt} |")
for v in VARS:
    e, n = 0, 0
    for k, (pr, _) in PUB.items():
        q = res.get((k, v))
        if q:
            r, a, _ = q["shares"]
            e += abs(lo(r / (r + a) if r + a else 0) - lo(pr / 100)); n += 1
    car = {f"{NAMES[k[0]]}-{NAMES[k[1]]}": round(100 * res[(k, v)]["shares"][2]) for k in PUB if (k, v) in res and 280 < res[(k, v)]["km"] < 700}
    print(f"{v}: error {e / max(n, 1):.2f} over {n} pairs; car of all on 300-650 km pairs: {car}")
