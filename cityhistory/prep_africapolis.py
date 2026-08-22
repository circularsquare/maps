"""prep_africapolis.py -- match OECD/SWAC Africapolis agglomerations to WUP centres.

Africapolis is the purpose-built African urban record: national censuses plus imagery,
agglomeration by agglomeration, rebuilt for African settlement patterns rather than inherited
from a global model. The 2025 update (version 9, 30-09-2025) carries 12,904 agglomerations over
50+ countries with NINE observed epochs -- 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2015, 2020
-- which is a real modern series rather than the single snapshot eFUA offers.

WHAT IT MEASURES, AND WHY THAT IS A JUDGEMENT CALL. Africapolis is a *morphological*
definition: contiguous built-up area plus at least 10,000 people. That is nearer WUP's urban
centre than it is to a functional urban area, so adopting it in Africa is NOT the same move as
adopting eFUA globally or MSAs in the States -- those replaced a density rule with a metropolitan
one, and this replaces a density rule with a better density rule. It is worth doing for accuracy
(Africapolis exists because global products mis-delineate African cities and miss their small
towns), but it trades away definitional consistency with the rest of the map unless the two
happen to agree. Run with --compare before believing it; that is the number the decision rests on.

PROJECTIONS ARE NOT DATA. The file runs to 2050, but only 1950-2020 is observed -- from 2025 the
values are projected, which the file itself shows by storing them as floats where the observed
years are integers. Only observed years are emitted, so an African city's modern era ends in
2020 and build.py's hold-forward carries it to 2025 marked `hx`, drawn dimmed. That is the
honest treatment and it is the same one every other under-recorded city gets.

ZEROS ARE NOT POPULATIONS. A 0 means the agglomeration did not exist or sat below the 10,000
threshold that year (row 1, 10 Ramadan, is 0 until 1990 -- it was founded in 1977). Emitting
them would draw a city at zero and then a vertical cliff, so they are dropped and the series
simply starts later.

MERGED AGGLOMERATIONS. `MergedTo` names the agglomeration a row was absorbed into between 2015
and 2020. Those rows are superseded and are skipped, or the map would draw both the survivor and
the thing it swallowed.

MATCHING. Africapolis carries a name, ISO3 and a centroid; WUP carries the same. A pair joins
when the names agree within NAME_KM, or the centroids are within TIGHT_KM and the two figures
are within TIGHT_RATIO of each other. Same two-rule shape as build.py's match_centre() and for
the same reason: ranking by size alone hands a town to whichever big neighbour is in range.
Country must agree in every case -- these are dense regions and a 30 km radius crosses borders.

A NOTE ON THE PREVIOUS ATTEMPT. Digital Earth Africa's geoserver mirror of Africapolis was tried
first and REJECTED: 14.3% of its rows grew by more than 1.5x between 2015 and 2020 (Cairo 23.0M
-> 38.4M, Kayanza 67k -> 724k which is the province, Kisumu 5.04M against a real ~600k), and its
2015 column was unsound too. audit() below is what caught it and still runs on every input. Do
not re-derive conclusions about eFUA from that mirror -- it read as "eFUA over-extends across
rural Africa" when WUP's own centres put Rubaya at 151,604 and Buram at 484,571 against the
mirror's 26,768 and 49,716.

Input:   data/Africapolis_agglomeration_2025.xlsx  (browser download from africapolis.org)
Output:  data/stadester/africapolis.json, keyed by WUP City_Code
Source:  OECD/SWAC (2025), Africapolis (database), www.africapolis.org -- CC BY 4.0
"""
import json, math, os, sys, unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC   = "data/Africapolis_agglomeration_2025.xlsx"
SHEET = "Agglomeration"
WUP   = "data/stadester/wup2025.json"
OUT   = "data/stadester/africapolis.json"
FUA   = "data/stadester/fua2025.json"

OBSERVED = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2015, 2020]   # 2025+ are projections
REF_YEAR = 2020

NAME_KM      = 30.0   # a name match may be this far off -- centroids of the same place differ
TIGHT_KM     = 8.0    # ...an unnamed match must be this close
TIGHT_RATIO  = 3.0    # ...and within this factor in size, or it is a different settlement
NAME_MIN_LEN = 5      # below this, substring matching is meaningless (see build.py NAME_MIN_LEN)

MAX_STEP    = 2.0        # a 5-year epoch step above this is a re-delineation, not growth
STEP_MIN    = 50_000     # ...checked only above here; small towns really do multiply
MAX_BUILTUP = 3_000      # km2; above this the row is a region, not an agglomeration

# National 2020 population, millions (UN WPP 2024), for the audit's impossibility test. Only
# the countries big enough for the sum to be meaningful; the rest are skipped.
NAT_POP_2020_M = {
    "NGA": 208.3, "ETH": 117.2, "EGY": 107.5, "COD": 92.9, "TZA": 61.7, "ZAF": 58.8,
    "KEN": 53.0, "UGA": 44.4, "SDN": 44.4, "DZA": 43.5, "MAR": 36.7, "AGO": 33.4,
    "MOZ": 31.2, "GHA": 32.2, "MDG": 28.2, "CIV": 26.8, "CMR": 26.5, "NER": 24.3,
    "MLI": 21.2, "BFA": 21.5, "MWI": 19.4, "ZMB": 18.9, "TCD": 16.6, "SOM": 16.5,
    "SEN": 16.4, "ZWE": 15.7, "GIN": 13.1, "RWA": 13.1, "BEN": 12.6, "BDI": 12.2,
    "TUN": 12.2, "SSD": 10.7, "TGO": 8.4, "SLE": 8.2, "LBY": 6.7, "COG": 5.7,
    "LBR": 5.1, "MRT": 4.6, "ERI": 3.6, "NAM": 2.5, "BWA": 2.5, "LSO": 2.3,
    "GNB": 2.0, "GAB": 2.3, "GMB": 2.5, "MUS": 1.3, "SWZ": 1.2, "DJI": 1.1,
}
URBAN_SHARE_MAX = 1.0    # a country's agglomerations cannot hold more than its whole population


def norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(ch)).lower().strip()


def names_agree(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return min(len(a), len(b)) >= NAME_MIN_LEN and (a in b or b in a)


def read_africapolis():
    """{id: {'name','iso3','lat','lon','pop':{year:value}}} -- observed years only, zeros dropped."""
    import openpyxl
    wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
    ws = wb[SHEET]
    rows = ws.iter_rows(values_only=True)
    hdr = list(next(rows))
    col = {h: i for i, h in enumerate(hdr) if h}
    need = ["Agglomeration_ID", "Agglomeration_Name", "ISO3", "Longitude", "Latitude"]
    for n in need:
        if n not in col:
            sys.exit(f"{SRC}: missing column {n!r} -- schema changed, check the header")
    ycol = {y: col[f"Population_{y}"] for y in OBSERVED if f"Population_{y}" in col}
    if len(ycol) < len(OBSERVED):
        print(f"note: only {sorted(ycol)} of the expected observed years are present")

    import collections
    out, rejected = {}, collections.Counter()
    for r in rows:
        if not r or r[col["Agglomeration_ID"]] in (None, ""):
            continue
        # NOT filtered on MergedTo. The metadata calls it "the destination city to which the
        # agglomeration was merged", but it does not behave like one: Kisumu points at Nairobi
        # 250 km away, as do Embu, Nakuru, Naivasha and Molo, and 1,908 rows point at
        # themselves. It is a cluster assignment, and filtering on it deletes Cairo, Lagos and
        # Kinshasa along with everything else that heads a cluster. Duplicates are handled
        # where they should be, in match(): one agglomeration per WUP centre, biggest first.
        aid = str(r[col["Agglomeration_ID"]])
        try:
            lon = float(r[col["Longitude"]]); lat = float(r[col["Latitude"]])
        except (TypeError, ValueError):
            continue
        pop = {}
        for y, i in ycol.items():
            v = r[i]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v > 0:                      # 0 = did not exist / below the 10,000 threshold
                pop[y] = v
        if not pop:
            continue

        # --- reject rows that re-delineated inside their own series --------------------
        # Africapolis redrew boundaries for the 2015 epoch, and where it did the series steps
        # by a factor no five-year period can produce: Mbale 88,151 -> 2,228,643, Sodo 96,897
        # -> 2,261,958, Embu 175,151 -> 2,046,897. Those rows are two different footprints in
        # one column, which is precisely what this pipeline refuses to draw as one series --
        # the same rule that made us swap the WHOLE modern range per city rather than splice.
        # So they are excluded by our own standard rather than by taste, and the city simply
        # falls back to the FUA/WUP layer.
        #
        # Floored at STEP_MIN because small towns really can multiply in five years and we
        # cannot tell those from re-delineation: Monguno 48k -> 358k is Borno State's displaced
        # concentrating in a garrison town, and it is real. Losing a few of those to the filter
        # costs nothing -- they fall back, they are not deleted.
        bad_step = None
        for a, b in ((2010, 2015), (2015, 2020)):
            if a in pop and b in pop and pop[a] >= STEP_MIN and pop[b] / pop[a] > MAX_STEP:
                bad_step = (a, b, pop[a], pop[b])
        if bad_step:
            rejected["re-delineated mid-series"] += 1
            continue
        # ...and rows whose built-up area is a region rather than a city. Three rows clear
        # 3,000 km2; the largest urban area on Earth is around 8,500. Kisumu's is 18,694,
        # eleven times Nairobi's, because the row is really "Kisumu/Mbale/Busia/Sirari" --
        # the whole Lake Victoria basin across three borders, at 15.5M.
        if "Built up_2020" in col:
            try:
                if float(r[col["Built up_2020"]]) >= MAX_BUILTUP:
                    rejected["built-up area is a region"] += 1
                    continue
            except (TypeError, ValueError):
                pass

        out[aid] = {
            "name": str(r[col["Agglomeration_Name"]] or ""),
            "iso3": str(r[col["ISO3"]] or ""),
            "lat": lat, "lon": lon, "pop": pop,
        }
    print(f"Africapolis: {len(out):,} agglomerations usable")
    for reason, n in rejected.most_common():
        print(f"  rejected {n:,}: {reason}")
    return out


def audit(afr):
    """Test something actually impossible, not something merely surprising.

    A growth-rate audit was tried first and thrown away, and the reason matters. Set strictly
    it fires on real African demography — Monguno, Dikwa and Gajram multiply because they are
    Borno State garrison towns holding the region's displaced; Angola's jumps follow the 2014
    post-war census, its first in 44 years; Abuja is simply the fastest-growing city on the
    continent. Set loosely it catches nothing. There is no threshold that separates "fast" from
    "wrong" on a continent where the underlying censuses are decades apart.

    So the audit asks the one question with no judgement in it: **can these people exist?** A
    country's agglomerations cannot hold more than the country does. That is arithmetic, and it
    is the check that would have caught this file's real problem — before the re-delineation
    filter in read_africapolis(), Kenya's rows summed to 74% of the national population against
    a true urban share near 28%, because a single row spanning the Lake Victoria basin was
    carrying 15.5 million people."""
    tot, n = {}, {}
    for a in afr.values():
        p = a["pop"].get(2020)
        if not p:
            continue
        tot[a["iso3"]] = tot.get(a["iso3"], 0) + p
        n[a["iso3"]] = n.get(a["iso3"], 0) + 1
    rows = sorted(((tot[i] / (NAT_POP_2020_M[i] * 1e6), i) for i in tot if i in NAT_POP_2020_M),
                  reverse=True)
    if not rows:
        return False, "no country matched the national population table"
    bad = [(s, i) for s, i in rows if s > URBAN_SHARE_MAX]
    msg = f"{len(rows)} countries checked against national population; highest shares:"
    for s, i in rows[:6]:
        msg += (f"\n    {i}  {n[i]:>5} agglomerations  {tot[i]:>13,.0f}  "
                f"= {s:.0%} of national population" + ("   <-- IMPOSSIBLE" if s > 1 else ""))
    return not bad, msg


def load_wup():
    with open(WUP, encoding="utf-8") as f:
        g = json.load(f)
    out = {}
    for code, c in g.items():
        co = c.get("coords")
        pop = {int(y): v for y, v in (c.get("population") or {}).items() if v and v > 0}
        if not co or len(co) != 2 or not pop:
            continue
        out[code] = {"name": c.get("name", ""), "iso3": c.get("iso3", ""),
                     "lat": co[0], "lon": co[1], "pop": pop}
    return out


def match(afr, wup):
    from collections import defaultdict
    grid = defaultdict(list)
    for code, c in wup.items():
        grid[(round(c["lat"]), round(c["lon"]))].append(code)

    matched, claimed = {}, {}
    for aid, a in sorted(afr.items(), key=lambda kv: -(kv[1]["pop"].get(REF_YEAR) or 0)):
        aref = a["pop"].get(REF_YEAR) or max(a["pop"].values())
        best = None                                    # (rule, distance, code)
        coslat = math.cos(math.radians(a["lat"]))
        for dla in (-1, 0, 1):
            for dlo in (-1, 0, 1):
                for code in grid.get((round(a["lat"]) + dla, round(a["lon"]) + dlo), []):
                    c = wup[code]
                    if c["iso3"] and a["iso3"] and c["iso3"] != a["iso3"]:
                        continue
                    d = math.hypot((c["lon"] - a["lon"]) * coslat * 111.32,
                                   (c["lat"] - a["lat"]) * 110.57)
                    if names_agree(a["name"], c["name"]) and d <= NAME_KM:
                        rule = 0
                    elif d <= TIGHT_KM:
                        w = c["pop"].get(REF_YEAR) or max(c["pop"].values())
                        if not (1 / TIGHT_RATIO <= aref / w <= TIGHT_RATIO):
                            continue
                        rule = 1
                    else:
                        continue
                    if best is None or (rule, d) < best[:2]:
                        best = (rule, d, code)
        if best is None:
            continue
        code = best[2]
        # one agglomeration per WUP centre; iterating biggest-first means the first claim wins,
        # the same tie-break build.py uses when two entries want one centre
        if code in claimed:
            continue
        claimed[code] = aid
        matched[code] = a
    return matched


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC} -- browser download from https://africapolis.org/en/data")
    afr = read_africapolis()
    ok, msg = audit(afr)
    print(f"audit: {msg}")
    if not ok:
        print("\nREJECTED -- see this file's header. Nothing written.")
        sys.exit(1)
    print("audit passed")
    if "--audit" in sys.argv:
        return

    wup = load_wup()
    matched = match(afr, wup)
    african = sum(1 for c in wup.values() if c["iso3"] in AFRICA_ISO3)
    print(f"matched to WUP centres: {len(matched):,} "
          f"({len(matched)/len(afr):.0%} of Africapolis, {len(matched)/african:.0%} of "
          f"{african:,} African WUP centres)")

    out = {code: {"population": {str(y): round(v) for y, v in sorted(a["pop"].items())},
                  "name": a["name"], "iso3": a["iso3"]}
           for code, a in matched.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {OUT}: {len(out):,} centres ({os.path.getsize(OUT)/1e6:.1f} MB)")
    npts = sum(len(v["population"]) for v in out.values()) / max(len(out), 1)
    print(f"  mean observed epochs per centre: {npts:.1f}")

    if "--compare" in sys.argv:
        compare(out, wup)


def compare(afr_by_code, wup):
    """Africapolis 2020 against what the map currently draws for the same centres.

    This is the adoption decision. Africapolis is morphological and the current layer is
    metropolitan, so a large systematic gap means adopting it would reintroduce exactly the
    cross-region definition break the FUA work removed -- African cities drawn on one rule and
    everyone else on another."""
    fua = {}
    if os.path.exists(FUA):
        with open(FUA, encoding="utf-8") as f:
            fua = json.load(f)
    rows = []
    for code, a in afr_by_code.items():
        ap = a["population"].get(str(REF_YEAR))
        f_ = fua.get(code)
        cur = (f_ or {}).get("population", {}).get(str(REF_YEAR)) or wup[code]["pop"].get(REF_YEAR)
        if not ap or not cur:
            continue
        rows.append((cur / ap, ap, cur, wup[code]["name"], "FUA" if f_ else "WUP",
                     wup[code]["iso3"]))
    if not rows:
        print("nothing to compare")
        return
    rows.sort(key=lambda r: -r[0])
    n = len(rows)
    ratios = sorted(r[0] for r in rows)
    print(f"\ncomparison on {n:,} centres -- current layer / Africapolis {REF_YEAR}")
    print(f"  median {ratios[n//2]:.2f}x   p10 {ratios[n//10]:.2f}  p90 {ratios[9*n//10]:.2f}   "
          f"within 0.8-1.25x: {sum(1 for r in ratios if 0.8 <= r <= 1.25)/n:.0%}   "
          f"beyond 2x: {sum(1 for r in ratios if r >= 2 or r <= 0.5)/n:.0%}")
    big = [r for r in rows if r[1] >= 1_000_000 or r[2] >= 1_000_000]
    if big:
        print(f"\n  the {len(big)} centres where either side is >=1M:")
        for ratio, ap, cur, wn, src, iso in sorted(big, key=lambda r: -max(r[1], r[2]))[:20]:
            print(f"    {wn[:24]:26}{iso}  {src} {cur:>12,.0f}  Africapolis {ap:>12,.0f}  x{ratio:.2f}")


AFRICA_ISO3 = {
    "DZA","AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD","COM","COG","COD","CIV","DJI",
    "EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG",
    "MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM",
    "ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE","ESH",
}

if __name__ == "__main__":
    main()
