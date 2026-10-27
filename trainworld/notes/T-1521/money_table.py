"""T-1521: add to app/src/game/countryIncome.json, per ISO 3166 alpha-2,
- `usd`: GDP per head at market rates, current US$ (World Bank NY.GDP.PCAP.CD, latest year), the
  income the fare rule reads; a country with none takes its PPP figure (`gdp`) times its subregion's
  median ratio of the two, Taiwan the IMF's figure;
- `car`: what a car costs to run a km, US$: gasoline at the pump (GlobalPetrolPrices, octane 95)
  times 8.5 litres a 100 km, plus the motorway toll a km where tolls go by distance on most of the
  network (table below). A country with no pump price takes its subregion's median.

    python trainworld/notes/T-1521/money_table.py <wb_gdp_usd.json> <gasoline.csv> [trainworld folder]

wb_gdp_usd.json is the API's answer to
https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&mrnev=1&per_page=5000
"""
import csv, json, statistics, sys
from pathlib import Path

ROOT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[2]
LITRES_PER_KM = 0.085  # IEA GFEI: new cars 7.1 L/100 km rated (2019), about 20% more on the road (EC 2024)
# US$ a km for a car, at the World Bank's 2025 average rates; sources in notes/T-1521.md
TOLL = {
    "JP": 21 / 149.66,   # NEXCO 24.6 yen/km + 150, tax, long-distance discounts: about 21 yen/km over 500 km
    "CN": 0.5 / 7.19,    # expressways 0.40-0.60 yuan/km for a small car
    "KR": 44 / 1422.4,   # 44 won/km plus 900 won
    "TW": 1.0 / 31.0,    # ETC 1.2 NT$/km, 0.9 past 200 km
    "FR": 0.104 / 0.88,  # autoroutes, class 1, about 0.104 euro/km
    "IT": 0.075 / 0.88,  # ASPI class A 0.072-0.085 euro/km
    "PT": 0.08 / 0.88,   # A1 Lisbon - Porto 25.05 euro
    "GR": 0.065 / 0.88,  # Egnatia 0.05, Olympia Odos 0.08 euro/km
    "HR": 0.078 / 0.88,  # 7.5-8 euro a 100 km
    "RS": 4.9 / 103.0,   # Belgrade - Nis 1,180 dinars
    "MK": 0.035 / 0.88,  # Skopje - Gevgelija
    "TR": 0.056,         # Ankara - Nigde 925 lira
    "IN": 1.8 / 87.16,   # NHAI base 1.47 rupees/km, expressways about 2.2
    "MX": 1.4 / 19.0,    # Capufe 1.1-1.8 pesos/km
    "BR": 0.28 / 5.6,    # Sao Paulo concessions 0.2-0.35 reais/km
}
# GlobalPetrolPrices' names that differ from the world field's
ALIAS = {"USA": "United States of America", "UK": "United Kingdom", "Czech Republic": "Czechia", "South Korea": "Republic of Korea",
         "Russia": "Russian Federation", "Taiwan": "Taiwan", "Hong Kong": "Hong Kong", "Macao": "Macao", "Vietnam": "Viet Nam",
         "Laos": "Lao People's Democratic Republic", "Iran": "Iran (Islamic Republic of)", "Syria": "Syrian Arab Republic",
         "Bolivia": "Bolivia (Plurinational State of)", "Venezuela": "Venezuela (Bolivarian Republic of)", "Tanzania": "United Republic of Tanzania",
         "Moldova": "Republic of Moldova", "North Korea": "Democratic People's Republic of Korea", "Ivory Coast": "Côte d'Ivoire",
         "DR Congo": "Democratic Republic of the Congo", "Congo": "Congo", "Turkey": "Türkiye", "Netherlands": "Netherlands (Kingdom of the)",
         "Brunei": "Brunei Darussalam", "Cape Verde": "Cabo Verde", "Swaziland": "Eswatini", "Micronesia": "Micronesia (Federated States of)"}

# names the world field spells differently, straight to ISO alpha-2
ISO2 = {"Trinidad & Tobago": "TT", "DR Congo": "CD", "UAE": "AE", "Curacao": "CW", "Dominican Republic": "DO", "Burma": "MM",
        "Cayman Islands": "KY", "Bosnia and Herzegovina": "BA", "Central African Republic": "CF", "Wallis and Futuna": "WF", "Mayotte": "YT"}

wb = json.load(open(sys.argv[1], encoding="utf-8"))[1]
countries = json.load(open(ROOT / "data/work/world_field.countries.json", encoding="utf-8"))["countries"]
p = ROOT / "app/src/game/countryIncome.json"
table = json.load(open(p, encoding="utf-8"))
iso2_of = {c["iso3"]: c["iso2"] for c in countries if c.get("iso2") and c.get("iso3")}
iso2_of["XKX"] = "XK"  # the World Bank's code for Kosovo
sub = {c["iso2"]: c["subregion"] for c in countries if c.get("iso2")}
norm = lambda s: s.lower().replace(" ", "").replace(".", "").replace("'", "").replace("’", "")
by_name = {norm(c["name"]): c["iso2"] for c in countries if c.get("iso2")}
med = lambda vs: statistics.median(vs)

# GDP per head at market rates
own = {}
for r in wb:
    k = iso2_of.get(r["countryiso3code"])
    if k and r["value"]:
        own[k] = r["value"]
own["TW"] = 34400.  # IMF World Economic Outlook, April 2025 (the World Bank lists no Taiwan)
ratio = {}
for k, v in own.items():
    if k in table["gdp"]:
        ratio.setdefault(sub.get(k), []).append(v / table["gdp"][k])
allr = med([x for v in ratio.values() for x in v])
usd, filled = {}, []
for k, g in table["gdp"].items():
    if k in own:
        usd[k] = round(own[k])
    else:
        usd[k] = round(g * med(ratio.get(sub.get(k), [allr])))
        filled.append(k)

# car running cost
fuel, unmatched = {}, []
for row in csv.DictReader(open(sys.argv[2], encoding="utf-8")):
    name = row["country"].strip()
    k = ISO2.get(name) or by_name.get(norm(ALIAS.get(name, name))) or by_name.get(norm(name))
    if k:
        fuel[k] = float(row["usd_per_litre"])
    else:
        unmatched.append(name)
fuel.setdefault("XK", 1.66)  # Kosovo's price cap, 1.47-1.49 euro/l, August 2026
fsub = {}
for k, v in fuel.items():
    fsub.setdefault(sub.get(k), []).append(v)
fall = med(list(fuel.values()))
car, cfill = {}, []
for k in table["gdp"]:
    f = fuel.get(k)
    if f is None:
        f = med(fsub.get(sub.get(k), [fall]))
        cfill.append(k)
    car[k] = round(f * LITRES_PER_KM + TOLL.get(k, 0.), 4)

table["usd_note"] = "GDP per head at market rates, current US$, World Bank NY.GDP.PCAP.CD latest (Taiwan IMF WEO 2025); gaps: PPP x subregion median ratio (T-1521)"
table["usd"] = dict(sorted(usd.items()))
table["car_note"] = "US$ a car-km: GlobalPetrolPrices gasoline 2026-10-19 x 8.5 L/100 km, plus distance tolls (notes/T-1521.md); gaps: subregion median pump price"
table["car"] = dict(sorted(car.items()))
p.write_text(json.dumps(table, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
print(f"usd: {len(usd)} ({len(filled)} filled: {' '.join(sorted(filled))})")
print(f"car: {len(car)} ({len(cfill)} filled: {' '.join(sorted(cfill))})")
print("unmatched pump-price names:", unmatched)
print({k: (usd[k], car[k]) for k in ("US", "JP", "CN", "KR", "FR", "DE", "GB", "ES", "IN", "NG", "TW", "HK", "XK")})
