"""
Greenland: the Survey of Living Conditions in the Arctic's Christian share, on the Greenland-born.

    python sources/gl.py --fetch     PxWeb tables + the SLiCA results tables, then build
    python sources/gl.py             rebuild from data/raw/gl/

    -> data/normalized/gl.csv              (municipality x node, Greenland-born people)
    -> data/raw/gl/bexstd_2026_born.csv    (locality x place of birth, 1 January 2026)

THE SOURCE. SLiCA, the Survey of Living Conditions in the Arctic, Greenland component: face to face
interviews by Statistics Greenland, December 2003 to August 2006, with a random sample of people
born in Greenland aged 15 and over drawn from the population register, 1,197 interviews, 83%
participation (Kruse et al., Int J Circumpolar Health 2008/2012, "Design and methods in a survey
of living conditions in the Arctic"). Its question "Do you consider yourself to be a Christian?"
is self-identification. SLiCA Results Tables (ISER, University of Alaska Anchorage, March 2007),
Cultural Continuity Table 162 (by country): Greenland 98% yes, 2% no, estimated total 35,969;
Table 163 (by region): Sydgronland 99, Midgronland 97, Diskobugten 99, Nordgronland >99,
Ostgronland 99. The PDF is data/raw/gl/slica_results_tables_2007.pdf (Wayback 20180612024209 of
iseralaska.org/static/living_conditions/images/SLICA_Results_Tables.pdf; the live URL is 404).

ONE NATIONAL MIX. The five SLiCA regions are the pre-2009 ones and do not nest in today's
municipalities (Ilulissat was Disko Bay and is now Avannaata; Sermersooq holds both Nuuk and East
Greenland), and they span 97 to 99%, a difference SLiCA's sample cannot separate from noise. So
every municipality takes the national 98%, as Cuba, Eritrea and Comoros take one mix (Anita's
rulings, 2026-10-03), and note_public says so.

THE CHRISTIANS ARE SPLIT BY THE CHURCH ROLL (spec 3.1: a roll may split a self-identified
category, never add to it). Statistics Greenland's BEXKIRK, 1 January 2026, born in Greenland:
47,850 members of the Church of Denmark (the Church of Greenland is its Greenlandic diocese) and
37 also in a free congregation, of 49,685 Greenland-born residents, 96.38%. That is less than the
98% who call themselves Christian, so it fits inside it: 96.38% -> christianity.lutheran, the
other 1.62% -> christianity (Christian, church not established), 2% -> unknown (SLiCA's "no",
which is not Christian and says nothing more: no religion, Inuit belief and other religions are
all in it).

THE UNIVERSE IS THE GREENLAND-BORN. SLiCA sampled only them. The 7,030 residents born outside
Greenland on 1 January 2026 (12.4%, mostly from Denmark, and the Filipino and Thai workers) are not
drawn, and `gap` says so. Children are drawn at the adults' mix, as everywhere on this map.

POPULATION. BEXSTD, Greenland-born by locality, 1 January 2026; summed to the five municipalities
by the locality code's first three digits and checked against BEXST8G's municipality totals.
"""
import io
import json
import os
import sys
import urllib.request

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "gl")
OUT = os.path.join(ROOT, "data", "normalized", "gl.csv")
API = "https://bank.stat.gl/api/v1/en/Greenland/BE/BE01/"
YEAR = "2026"

SLICA_URL = ("https://web.archive.org/web/20180612024209id_/https://iseralaska.org/static/"
             "living_conditions/images/SLICA_Results_Tables.pdf")
SLICA_PDF = os.path.join(RAW, "slica_results_tables_2007.pdf")

# SLiCA Table 162, Greenland. Integers as printed.
SLICA_CHRISTIAN = 0.98
SLICA_REGIONS = {"Sydgrønland": 99, "Midgrønland": 97, "Diskobugten": 99,
                 "Nordgrønland": 99.5, "Østgrønland": 99}           # >99 read as 99.5
SLICA_REGION_N = {"Sydgrønland": 5_037, "Midgrønland": 15_594, "Diskobugten": 8_453,
                  "Nordgrønland": 4_733, "Østgrønland": 2_153}

# Locality code prefix -> municipality (BEXST8G's own codes), and the unit it is drawn in.
MUNI = {"955": ("0955", "Kujalleq", "GL-KU"), "956": ("0956", "Sermersooq", "GL-SM"),
        "957": ("0957", "Qeqqata", "GL-QE"), "959": ("0959", "Qeqertalik", "GL-QT"),
        "960": ("0960", "Avannaata", "GL-AV"),
        # Outside the municipalities: Pituffik and the rest. Drawn in the unit whose ground the
        # locality is on, which sources/gl_geo.py checks; the unknown remainder goes to the
        # national park.
        "961": ("0961", "Outside municipalities", None)}
OUTSIDE_UNIT = {"9612070PIT": "GL-AV", "9612099ZZZ": "GL-UO"}

SOURCE_ID = "gl_slica_2003_2006"


def post(table, query):
    body = json.dumps({"query": query, "response": {"format": "csv"}}).encode()
    req = urllib.request.Request(API + table, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8-sig")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    sel = lambda code, vals: {"code": code, "selection": {"filter": "item", "values": vals}}
    meta = json.load(urllib.request.urlopen(API + "BE0120/BEXSTD.px", timeout=120))
    locs = next(v for v in meta["variables"] if v["code"] == "locality")
    txt = post("BE0120/BEXSTD.px", [sel("place of birth", ["T", "N", "S"]),
                                    sel("locality", locs["values"]), sel("time", [YEAR])])
    open(os.path.join(RAW, "bexstd_2026_born.csv"), "w", encoding="utf-8").write(txt)
    txt = post("BE0125/BEXST8G.px", [sel("place of birth", ["TO", "GR"]),
                                     sel("district", ["TOT", "0955", "0956", "0957", "0959",
                                                      "0960", "0961"]),
                                     sel("time", [YEAR])])
    open(os.path.join(RAW, "bexst8g_2026.csv"), "w", encoding="utf-8").write(txt)
    txt = post("BE0120/BEXKIRK.px", [sel("cprcode", ["T", "F", "M", "S", "A", "U"]),
                                     sel("pob", ["T", "N"])])
    open(os.path.join(RAW, "bexkirk.csv"), "w", encoding="utf-8").write(txt)
    if not os.path.exists(SLICA_PDF):
        with urllib.request.urlopen(SLICA_URL, timeout=300) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            raise SystemExit("the SLiCA tables did not come back as a PDF")
        open(SLICA_PDF, "wb").write(data)


def check_slica():
    """The two tables, read off the PDF, so the hand-typed constants cannot drift from it."""
    import fitz
    d = fitz.open(SLICA_PDF)
    t = "\n".join(p.get_text() for p in d)
    i = t.find("Cultural Continuity Table 162: Consider Self to be Christian by Country \n")
    j = t.find("Cultural Continuity Table 163: Consider Self to be Christian by Region \n")
    if i < 0 or j < 0:
        raise SystemExit("SLiCA Tables 162/163 not found in the PDF")
    t162 = t[i:i + 400].split()
    if "35,969" not in t162 or "98%" not in t162:
        raise SystemExit(f"SLiCA Table 162 no longer reads Greenland 98% of 35,969: {t162[:40]}")
    t163 = t[j:j + 900]
    for reg, n in SLICA_REGION_N.items():
        if f"{n:,}" not in t163:
            raise SystemExit(f"SLiCA Table 163: {reg}'s estimated total {n:,} not found")
    w = sum(SLICA_REGIONS[r] * SLICA_REGION_N[r] for r in SLICA_REGIONS) / sum(SLICA_REGION_N.values())
    print(f"  SLiCA Greenland: 98% Christian (Table 162); regions 97-99%, "
          f"population-weighted {w:.1f}% (Table 163)")


def main():
    if "--fetch" in sys.argv or not os.path.exists(os.path.join(RAW, "bexstd_2026_born.csv")):
        fetch()
    check_slica()

    # ---- the roll, for the split inside the Christians ----
    k = pd.read_csv(os.path.join(RAW, "bexkirk.csv"))
    k.columns = ["status", "pob"] + list(k.columns[2:])
    born = k[k["pob"] == "Born in Greenland"].set_index("status")[YEAR]
    members = int(born["Member of Church of Denmark"] +
                  born["Member of a Congregation (and Church of Denmark)"])
    base = int(born["Total"])
    roll = members / base
    print(f"  church roll, born in Greenland, 1 January {YEAR}: {members:,} of {base:,} "
          f"= {roll:.2%} members of the Church of Denmark")
    if roll >= SLICA_CHRISTIAN:
        raise SystemExit("the roll now exceeds the self-identified Christian share; the split "
                         "no longer fits inside it (spec 3.1) and has to be rethought")
    shares = {"Church of Greenland (Lutheran)": roll,
              "Christian, other or church not established": SLICA_CHRISTIAN - roll,
              "Not Christian": 1 - SLICA_CHRISTIAN}

    # ---- population: Greenland-born by locality, to municipality ----
    s = pd.read_csv(os.path.join(RAW, "bexstd_2026_born.csv"))
    s.columns = ["pob", "locality", "n"]
    meta = json.load(open(os.path.join(RAW, "bexstd_meta.json"), encoding="utf-8"))
    locs = next(v for v in meta["variables"] if v["code"] == "locality")
    code = dict(zip(locs["valueTexts"], locs["values"]))
    s["code"] = s["locality"].map(code)
    if s["code"].isna().any():
        raise SystemExit(f"localities with no code: {sorted(s.loc[s['code'].isna(), 'locality'])}")
    gb = s[(s["pob"] == "Greenland") & (s["code"] != "0000000GRL")].copy()
    total_gb = int(s.loc[(s["pob"] == "Greenland") & (s["code"] == "0000000GRL"), "n"].iloc[0])
    if int(gb["n"].sum()) != total_gb:
        raise SystemExit(f"localities sum to {int(gb['n'].sum()):,}, total {total_gb:,}")
    gb["muni"] = gb["code"].str[:3].map(lambda p: MUNI[p][0])
    gb["unit"] = [OUTSIDE_UNIT.get(c) or MUNI[c[:3]][2] for c in gb["code"]]
    gb.to_csv(os.path.join(RAW, "gl_localities_born_2026.csv"), index=False, encoding="utf-8")

    m = pd.read_csv(os.path.join(RAW, "bexst8g_2026.csv"))
    m.columns = ["pob", "district", "n"]
    m8 = m[m["pob"] == "Greenland"].set_index("district")["n"]
    names = {v[0]: v[1] for v in MUNI.values()}
    for mc, n in gb.groupby("muni")["n"].sum().items():
        want = int(m8[[d for d in m8.index if d.startswith(names[mc].split()[0])
                       or names[mc] in d][0]])
        if int(n) != want:
            raise SystemExit(f"{names[mc]}: localities {int(n):,}, BEXST8G {want:,}")
    allres = int(s.loc[(s["pob"] == "Total") & (s["code"] == "0000000GRL"), "n"].iloc[0])
    print(f"  1 January {YEAR}: {allres:,} residents, {total_gb:,} born in Greenland "
          f"({total_gb / allres:.2%}), {allres - total_gb:,} born outside, not drawn")

    # ---- the rows ----
    pop = gb.groupby("unit")["n"].sum()
    rows = []
    for unit, p in pop.items():
        left = int(p)
        cats = list(shares.items())
        for i, (cat, sh) in enumerate(cats):
            n = left if i == len(cats) - 1 else int(round(p * sh))
            left -= n
            rows.append(dict(geo_id=unit, geo_level="municipality", geo_name=unit,
                             source_category=cat, count=n, basis="self_id",
                             year="2003-2006", source_id=SOURCE_ID,
                             note=("SLiCA Greenland 2003-2006, 98% consider themselves Christian "
                                   "(national); Christians split by the 2026 church roll of the "
                                   f"Greenland-born ({roll:.2%}); applied to the {int(p):,} "
                                   f"Greenland-born residents of 1 January {YEAR}")))
    out = pd.DataFrame(rows)
    if int(out["count"].sum()) != total_gb:
        raise SystemExit("rows do not add to the Greenland-born total")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} rows, {int(out['count'].sum()):,} people, "
          f"{out['geo_id'].nunique()} units)")
    for cat, sh in shares.items():
        print(f"    {sh:7.2%}  {cat}")
    print(pop.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
