"""T-1521: real rail fares against the game's starting curve, per country, at PPP and market rates."""
import json, math, sys
TW = r"C:\Users\anita\projects\maps\trainworld"
gdp = json.load(open(TW + r"\app\src\game\countryIncome.json", encoding="utf-8"))["gdp"]

def scale(c):
    return min(9.9, max(0.1, round((gdp[c] / gdp["IN"]) ** 0.5 * 10) / 10))

R = 6371.0088
def gc(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

# World Bank 2025: PPP (PA.NUS.PPP) and market (PA.NUS.FCRF), LCU per $
FX = {"US": (1, 1), "JP": (97.08, 149.66), "CN": (3.398, 7.19), "KR": (816.33, 1422.44), "FR": (0.6776, 0.88),
      "DE": (0.71, 0.88), "GB": (0.6771, 0.76), "ES": (0.571, 0.88), "IN": (20.089, 87.16), "NG": (195.02, 1518.38)}

P = {  # station coordinates
    "nyc": (40.7527, -73.9772), "nhv": (41.2977, -72.9264), "wdc": (38.8973, -77.0063),
    "tyo": (35.6812, 139.7671), "odw": (35.2564, 139.1551), "osa": (34.7334, 135.5002), "hkt": (33.5897, 130.4207),
    "bjs": (39.8650, 116.3790), "tsn": (39.1360, 117.2100), "shq": (31.1940, 121.3200),
    "seo": (37.5547, 126.9707), "dje": (36.3324, 127.4343), "pus": (35.1151, 129.0422),
    "pau": (48.8423, 2.3659), "orl": (47.9078, 1.9045), "pgl": (48.8443, 2.3744), "lys": (45.7606, 4.8594), "mrs": (43.3027, 5.3806),
    "fra": (50.1071, 8.6638), "man": (49.4794, 8.4699), "ber": (52.5250, 13.3690), "ham": (53.5530, 10.0060), "muc": (48.1400, 11.5580),
    "lst": (51.5178, -0.0823), "cbg": (52.1943, 0.1375), "kgx": (51.5320, -0.1233), "edb": (55.9521, -3.1890),
    "atc": (40.4065, -3.6903), "tol": (39.8625, -4.0118), "bcn": (41.3793, 2.1400), "svq": (37.3917, -5.9750),
    "bct": (18.9690, 72.8205), "csmt": (18.9398, 72.8355), "pun": (18.5286, 73.8740), "ndl": (28.6430, 77.2193),
    "ebm": (6.4880, 3.3830), "iba": (7.5250, 3.9100), "wri": (5.5167, 5.7500), "itk": (7.6167, 6.3000),
}
def gk(a, b):
    return 1.05 * gc(P[a], P[b])

# country, band, trip, game km, fare local, source
ROWS = [
    ("US", "short", "NYC subway single", 10, 3.00, "MTA OMNY fare from 4 Jan 2026 (ABC7 NY)"),
    ("US", "medium", "Grand Central - New Haven off-peak", gk("nyc", "nhv"), 20.25, "CTDOT/MTA schedule, Jul 2026 (New Haven Independent)"),
    ("US", "long", "New York - Washington, Regional", gk("nyc", "wdc"), 106.0, "Amtrak FY25 Regional $0.471 a passenger-mile x 225 mi (Eno Center)"),
    ("JP", "short", "Tokyo Metro 7-11 km", 9, 210, "Tokyo Metro fare table"),
    ("JP", "medium", "JR Tokyo - Odawara local", gk("tyo", "odw"), 1600, "JR East fare from March 2026"),
    ("JP", "long", "Tokyo - Shin-Osaka Nozomi reserved", gk("tyo", "osa"), 14720, "JR Central fare (Kyodo, 2026)"),
    ("JP", "long", "Tokyo - Hakata Nozomi reserved", gk("tyo", "hkt"), 23810, "Ekitan"),
    ("CN", "short", "Beijing subway 6-12 km", 9, 4, "Beijing Municipal Government fare rule"),
    ("CN", "medium", "Beijing South - Tianjin, 2nd class", gk("bjs", "tsn"), 54.5, "chinahighlights / chinafortravelers"),
    ("CN", "long", "Beijing South - Shanghai Hongqiao, 2nd class", gk("bjs", "shq"), 600, "G trains ¥553-662 before May 2026 (Yicai); ceiling ¥795 since"),
    ("KR", "short", "Seoul metro basic (to 10 km)", 9, 1550, "Korea Times, June 2025 fare"),
    ("KR", "medium", "Seoul - Daejeon KTX standard", gk("seo", "dje"), 23700, "Korail base fare (koreacharts); about ₩21,400 after Sept 2026"),
    ("KR", "long", "Seoul - Busan KTX standard", gk("seo", "pus"), 59800, "Korail base fare (Kyunghyang); about ₩54,400 after Sept 2026"),
    ("FR", "short", "Paris metro/RER single", 8, 2.55, "Ile-de-France Mobilites 2026"),
    ("FR", "medium", "Paris - Orleans TER full fare", gk("pau", "orl"), 24, "TER Centre-Val de Loire"),
    ("FR", "long", "Paris - Lyon TGV INOUI", gk("pgl", "lys"), 52, "INOUI average 11.1 c/km excl. VAT 2023 (ART via FNAUT) +10% VAT x 427 km"),
    ("FR", "long", "Paris - Marseille TGV INOUI", gk("pgl", "mrs"), 92, "same rate x 750 km"),
    ("DE", "short", "Berlin AB single", 8, 4.00, "BVG 2026 (berlin.de)"),
    ("DE", "medium", "Frankfurt - Mannheim regional", gk("fra", "man"), 18, "cheapest listed (kombo.co); weak"),
    ("DE", "long", "Berlin - Hamburg ICE", gk("ber", "ham"), 43, "DB Fernverkehr 2025: 14.2 c a pkm net, +7% VAT, x 285 km"),
    ("DE", "long", "Berlin - Munich ICE", gk("ber", "muc"), 89, "same rate x 585 km"),
    ("GB", "short", "Tube zones 1-2 PAYG", 8, 3.35, "TfL Mar 2026: £3.60 peak, £3.10 off-peak (MD3464)"),
    ("GB", "medium", "London - Cambridge off-peak single", gk("lst", "cbg"), 24.20, "National Rail fare via aggregator"),
    ("GB", "long", "London - Edinburgh LNER", gk("kgx", "edb"), 70, "typical advance; LNER Advance from £41.40, Anytime £204.80"),
    ("ES", "short", "Madrid metro zone A single", 6, 1.75, "€1.50-2.00 (redtransporte 2026)"),
    ("ES", "medium", "Madrid - Toledo Avant", gk("atc", "tol"), 11, "cheapest listed (Omio)"),
    ("ES", "long", "Madrid - Sevilla high-speed", gk("atc", "svq"), 49, "CNMC average, Q2 2025"),
    ("ES", "long", "Madrid - Barcelona high-speed", gk("atc", "bcn"), 63, "CNMC average, Q2 2025"),
    ("IN", "short", "Mumbai suburban 2nd class 5-10 km", 8, 10, "band fares ₹5-20 (weak)"),
    ("IN", "short", "Delhi Metro 5-12 km", 9, 32, "DMRC from Aug 2025 (ETV Bharat)"),
    ("IN", "medium", "Mumbai - Pune Deccan Queen 2nd sitting", gk("csmt", "pun"), 110, "punekarnews, Aug 2025 (chair car ₹390)"),
    ("IN", "long", "Mumbai - New Delhi sleeper", gk("bct", "ndl"), 625, "Mail/Express sleeper ₹575-675 (trainspread)"),
    ("IN", "long", "Mumbai - New Delhi Rajdhani 3AC", gk("bct", "ndl"), 1780, "ixigo / trainstracking ₹1,750-1,810"),
    ("NG", "short", "Lagos Blue Line, whole line", 13, 750, "mapa-metro (weak)"),
    ("NG", "medium", "Lagos - Ibadan SGR standard", gk("ebm", "iba"), 6500, "trainstracking (unverified; ₦3,600 in 2023)"),
    ("NG", "long", "Warri - Itakpe regular", gk("wri", "itk"), 5000, "NRC via Nairametrics, Feb 2025"),
]

def curve(nodes, km):
    i = 1
    while i < len(nodes) - 1 and nodes[i][0] <= km: i += 1
    (k0, f0), (k1, f1) = nodes[i - 1], nodes[i]
    return max(0, f0 + (f1 - f0) * (km - k0) / (k1 - k0))

OLD = [(0, 0.5357), (20, 0.5357 + 0.03571 * 20), (100, 0.5357 + 0.03571 * 100)]

def table(nodes, label):
    print(f"\n== {label} ==")
    print(f"{'country':3} {'band':6} {'trip':42} {'km':>5} {'real PPP $':>10} {'real mkt $':>10} {'game $':>7} {'PPP/game':>8} {'mkt/game':>8}")
    rat = {}
    for c, band, trip, km, f, src in ROWS:
        ppp, mkt = FX[c]
        g = scale(c) * curve(nodes, km)
        rp, rm = f / ppp, f / mkt
        rat.setdefault(band, []).append((c, rp / g, rm / g))
        print(f"{c:3} {band:6} {trip:42} {km:5.0f} {rp:10.2f} {rm:10.2f} {g:7.2f} {rp / g:8.2f} {rm / g:8.2f}")
    for band, v in rat.items():
        gp = math.exp(sum(math.log(x[1]) for x in v) / len(v)); gm = math.exp(sum(math.log(x[2]) for x in v) / len(v))
        sp = (sum(math.log(x[1] / gp) ** 2 for x in v) / len(v)) ** .5; sm = (sum(math.log(x[2] / gm) ** 2 for x in v) / len(v)) ** .5
        print(f"  {band:6}: geo mean real/game at PPP {gp:.2f} (log spread {sp:.2f}); at market {gm:.2f} (spread {sm:.2f})")

print("scales:", {c: scale(c) for c in FX})
table(OLD, "old starting curve")
if len(sys.argv) > 1:
    nodes = json.loads(sys.argv[1])
    table(nodes, f"new {nodes}")
