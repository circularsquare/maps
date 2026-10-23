import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "dry2_rows.json")))["rows"]
zones = json.load(open(os.path.join(HERE, "dry2_zones_500000.json")))["zones"]
F = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY", "72": "PR"}
W = {"pilot": "GA FL", "1 south": "TX OK AR LA MS AL TN SC NC", "2 midwest": "OH IN IL MI WI MN IA MO KS NE SD ND KY",
     "3 west": "CA OR WA NV AZ UT CO NM ID MT WY AK HI", "4 northeast": "NY PA NJ DE MD DC VA WV CT RI MA VT NH ME PR"}
wof = {s: w for w, ss in W.items() for s in ss.split()}
c = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
for r in rows:
    w = wof[F[r["lead_state"]]]
    c[w][0] += 1
    c[w][2] += r["people"] / 1e6
for z in zones:
    w = wof[F[z["state"]]]
    c[w][1] += 1
    c[w][3] += z["people"] / 1e6
for w in W:
    m, n, pm, pz = c[w]
    print(f"{w}: {m} metros ({pm:.1f}M), {n} zones ({pz:.1f}M), {m + n} rows, {pm + pz:.1f}M")
cross = [r["name"] + " " + "+".join(F[s] for s in r["states"]) for r in rows if len(r["states"]) > 1]
print("cross-state metros:", cross)
