"""T-456: OSM shop and food counts per country from Geofabrik's taginfo, per 1,000 residents
(World Bank SP.POP.TOTL, latest year)."""
import os, sys, json, time, requests
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\Users\anita\projects\maps\trainworld\data\raw\t456\taginfo.json"
UA = {"User-Agent": "trainworld-research/0.1 (open data analysis)"}
REG = {"NG": "africa:nigeria", "GH": "africa:ghana", "KE": "africa:kenya", "ET": "africa:ethiopia", "ZA": "africa:south-africa",
       "EG": "africa:egypt", "IN": "asia:india", "CN": "asia:china", "ID": "asia:indonesia", "PK": "asia:pakistan",
       "BD": "asia:bangladesh", "PH": "asia:philippines", "VN": "asia:vietnam", "TH": "asia:thailand", "KR": "asia:south-korea",
       "JP": "asia:japan", "TW": "asia:taiwan", "GB": "europe:great-britain", "DE": "europe:germany", "FR": "europe:france",
       "IT": "europe:italy", "ES": "europe:spain", "PL": "europe:poland", "NL": "europe:netherlands", "RU": "russia",
       "TR": "europe:turkey", "BR": "south-america:brazil", "MX": "north-america:mexico", "CA": "north-america:canada",
       "AR": "south-america:argentina", "CO": "south-america:colombia", "AU": "australia-oceania:australia"}
FOOD = ["restaurant", "cafe", "fast_food", "bar", "pub"]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for iso, reg in REG.items():
    if iso in res:
        continue
    base = f"https://taginfo.geofabrik.de/{reg}/api/4"
    try:
        r = requests.get(f"{base}/key/stats", params={"key": "shop"}, headers=UA, timeout=60).json()
        shop = next(x["count"] for x in r["data"] if x["type"] == "all")
        food = 0
        for v in FOOD:
            r = requests.get(f"{base}/tag/stats", params={"key": "amenity", "value": v}, headers=UA, timeout=60).json()
            food += next(x["count"] for x in r["data"] if x["type"] == "all")
        r = requests.get(f"{base}/key/stats", params={"key": "building"}, headers=UA, timeout=60).json()
        bld = next(x["count"] for x in r["data"] if x["type"] == "all")
        p = requests.get(f"https://api.worldbank.org/v2/country/{iso}/indicator/SP.POP.TOTL",
                         params={"format": "json", "mrnev": 1}, headers=UA, timeout=60).json()[1][0]["value"]
        res[iso] = {"region": reg, "shop": shop, "food": food, "building": bld, "pop": p}
        json.dump(res, open(OUT, "w"), indent=1)
    except Exception as e:
        print(iso, e)
    time.sleep(1)
print("| country | Geofabrik region | residents | OSM shops /1k | OSM food /1k | buildings per shop |")
print("|---|---|---|---|---|---|")
for iso, v in sorted(res.items(), key=lambda kv: -kv[1]["shop"] / kv[1]["pop"]):
    print(f"| {iso} | {v['region']} | {v['pop'] / 1e6:.0f}M | {1000 * v['shop'] / v['pop']:.2f} | {1000 * v['food'] / v['pop']:.2f} | {v['building'] / max(v['shop'], 1):.0f} |")
