"""Merge the regional coverage CSVs into coverage.csv and print population by tier.

Tiers (for a one-dot-per-person first-language map):
  A  census language data at district level or finer; single answer, or an
     indigenous-only question whose remainder is one language (Mexico: Spanish)
  B  real language data, but coarse (province/national), multi-response, or last
     asked before ~2005
  C  no usable language question, but ~95%+ speak one language (a judgement call)
  D  ethnicity only, published finely enough to crosswalk
  E  nothing usable from the census; surveys, old censuses or atlases only

Population is World Bank SP.POP.TOTL 2024 (wb_pop.json, fetched 2026-10-03),
Taiwan added by hand. The UK rows are split by hand.
"""
import csv, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "wb_pop.json")

TIERS = {
    "A": """IND PAK NPL ETH ZAF ZMB CAF NAM BWA MUS SYC USA PRI CAN MEX GTM NIC CRI COL
            VEN PER BOL CHL ARG BRA HKG AUS VUT PYF GUM SGP GBR_EW GBR_SC GBR_NI ROU SVK
            HUN POL CZE HRV BGR SRB BIH MKD MNE ALB XKX EST FIN LTU MLT CHE LUX LIE GEO
            MDA UKR FRO AZE ARM KGZ CUW ABW VIR""",
    "B": """RUS BLR CYP TKM UZB TJK MOZ AGO THA IDN KHM SLB TWN MAC BRN TLS PLW SEN MLI
            GIN SLE BFA CIV GNB STP MAR ESH ECU PRY NZL NCL BLZ SLV DEU IRL SUR AUT SVN
            LVA SXM BES""",
    "C": """JPN KOR PRK HTI CUB DOM JAM BHS BRB ATG DMA GRD KNA VCT BMU TCA VGB AIA CYM
            MSR GLP MTQ RWA BDI LSO SWZ SOM MDG COM TON WSM TUV MHL KIR MDV PRT ISL EGY
            TUN JOR PSE LBN YEM AND MCO SMR VAT CPV GRL""",
    "D": """CHN VNM PHL MMR MYS LAO MNG BGD LKA KEN UGA MWI GHA LBR TGO BEN GMB HND PAN
            URY TTO GUY FJI NRU FSM KAZ LCA""",
    "E": """NGA COD FRA ESP ITA BEL NLD TUR IRN AFG DZA SDN IRQ SYR CMR TZA ZWE NER GNQ
            GAB COG ERI DJI SSD SAU ARE QAT KWT BHR OMN LBY ISR MRT TCD DNK NOR SWE BTN
            PNG GRC GUF""",
}
TIER_OF = {iso: t for t, s in TIERS.items() for iso in s.split()}
UK = {"GBR_EW": 61.0, "GBR_SC": 5.5, "GBR_NI": 1.9}

wb = json.load(open(WB, encoding="utf-8"))[1]
pop = {r["countryiso3code"]: (r["value"] or 0) / 1e6 for r in wb}
pop["TWN"] = 23.4
pop.update(UK)
world = pop["WLD"] + pop["TWN"]

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "*.csv"))):
    if os.path.basename(f) == "coverage.csv":
        continue
    with open(f, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            r["region"] = os.path.basename(f)[:-4]
            r["pop_m"] = round(pop.get(r["iso3"], 0.0), 2)
            r["tier"] = TIER_OF.get(r["iso3"], "?")
            rows.append(r)

missing = [r["iso3"] for r in rows if r["tier"] == "?"]
assert not missing, missing
extra = set(TIER_OF) - {r["iso3"] for r in rows}
assert not extra, extra

rows.sort(key=lambda r: (r["tier"], -r["pop_m"]))
fields = ["tier", "iso3", "country", "region", "pop_m"] + [
    k for k in rows[0] if k not in ("tier", "iso3", "country", "region", "pop_m")]
with open(os.path.join(HERE, "coverage.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

total = sum(r["pop_m"] for r in rows)
print(f"{len(rows)} rows, {total:.0f}M of {world:.0f}M world")
for t in "ABCDE":
    tr = [r for r in rows if r["tier"] == t]
    p = sum(r["pop_m"] for r in tr)
    big = ", ".join(f"{r['country']} {r['pop_m']:.0f}" for r in tr[:8])
    print(f"{t}: {len(tr):3d} countries {p:7.0f}M {100 * p / world:5.1f}%  {big}")
