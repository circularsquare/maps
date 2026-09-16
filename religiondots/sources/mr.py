"""Mauritania: nobody is asked their religion. Nationals are drawn on Islam, and foreign residents by
nationality, both on the 2023 census.

Reads data/raw/mr/Theme-16-Population-etrangere-vivant-en-Mauritanie.pdf, data/geo/mr/mr_lookup.csv,
data/raw/mr/undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx and
data/raw/estimates/pew.zip; writes data/normalized/mr.csv (Mauritanian nationals) and
data/normalized/mr_foreign.csv (foreign residents). `sources/mr.md` is the record in prose;
`ask/RULINGS.md` 2026-09-15 and 2026-09-16 the rulings ("draw it", on a compiler's figure, with
foreigners by wilaya; no rings).

## NOBODY IS ASKED

The 2013 census form has no religion item (`sources.md` §11aq). The 2023 form is not published, but
none of ANSADE's sixteen thematic reports on the 2023 census has a table on religion, and Thème 16
(§16.2.2) lists what was asked of everyone (nationality at Q07; marital status, education, health,
occupation) and of foreigners 15 and over (four migration questions). The Arab Barometer's 3,200
Mauritanian rows (waves VII, VIII) all leave `Q1012` empty; Afrobarometer does not put the question
in Mauritania; the DHS 2019-21 has no item.

## TWO POPULATIONS, ONE CENSUS

Thème 1's wilaya populations (`sources/mr_geo.py`) are everyone counted; Thème 16's Tableau 16.6
counts the foreign residents in each wilaya; nationals are the difference. Tableau 16.1 prints
4,801,600 Mauritanians and 125,933 foreigners, which is one more than the census total; the wilaya
rows give 4,801,598.

## NATIONALS ARE ALL DRAWN ON ISLAM, NOT ON PEW'S RESIDUAL

Pew Research Center's 2020 estimate covers everyone living in Mauritania, foreigners included:
99.185% Muslim, 10,754 Christians, 4,899 unaffiliated, 102 Jews, 21,717 other religions. Drawing
those shares on the country AND the foreigner layer beside them counts foreign non-Muslims twice.
The two clean constructions are (1) nationals take Pew's non-Muslims less what the foreigner layer
already holds, or (2) nationals are all on Islam and every non-Muslim dot is a foreign resident.
This file draws (2): the CIA World Factbook's entry is `Muslim (official) 100%`; no source says
anything about nationals who are not Muslim; and (1)'s residual would be mostly Pew's 21,717 `other
religions`, a cell nothing explains. `christian_witness` prints the foreigner layer's Christians
against Pew's, as a share of the population, inside a band written before the first run.

## FOREIGN RESIDENTS

  * **per wilaya**: Tableau 16.6, 125,933.
  * **nationality, national only**: Tableau 16.2's eight groups; Tableau A.5 (urban and rural) is
    read as a check on the transcription.
  * **the refugees are placed first.** Tableau 16.5 counts 46,800 refugees and asylum seekers. Thème
    16 (p.15) puts Hodh Chargui's 59,555 foreigners down to the refugees there, and Thème 15 (p.39)
    to the Mbera camp's Malian refugees. So 46,800 Malians go to Hodh Chargui, and the other 79,133
    foreigners take the national mix without them in every wilaya, Hodh Chargui's remaining 12,755
    included. Without this, 47% of the country's Europeans would be drawn in Hodh Chargui.
  * **the refugees are from northern Mali**, so they do not take Pew's national Mali row, which put
    2,746 non-Muslims on them (21.6% of every non-Muslim drawn in the country; review of
    2026-09-15). They take Mali's own 2022 census answers (`data/normalized/ml.csv`) pooled over
    each région as it was before the 2023 reform, weighted by UNHCR's 2018 map of where Mbera's
    refugees came from (`ORIGIN_2018`). The other 36,881 Malians keep Pew's Mali row.
  * **inside the census's `other` groups**: UN DESA's *International Migrant Stock 2024* names
    Mauritania's migrants by origin. Where its named countries reach `COVER_BAR` of the census's
    group, they weight it (other African countries, other Arab countries); where they do not, the
    group takes Pew's regional rows (Europe; the rest of the world as Asia-Pacific, Latin America and
    North America summed). DESA's 32,947 from Western Sahara are more than the census's two `other`
    groups together, so they are not in the census's foreign count under either, and are left out.
  * **each nationality**: Pew 2020 through `taxonomy/origin_religion.py`, Muslim branches folded to
    `islam` as `sources/ma.py` does, Pew's unplaced `other religions` on `other.mr`.

Usage:
    python sources/mr.py --fetch    Thème 16 (1.1 MB) and the DESA workbook (6.0 MB) if missing
    python sources/mr.py            rebuild data/normalized/mr.csv and mr_foreign.csv
"""

import io
import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from afrobarometer import round_within_rows
from mr_geo import WILAYAS, isnum, num
import origin_religion as origin
from ml2022 import MAP as ML_MAP
from ml_geo import PARENT as ML_PARENT

RAW = os.path.join(ROOT, "data", "raw", "mr")
THEME16 = os.path.join(RAW, "Theme-16-Population-etrangere-vivant-en-Mauritanie.pdf")
THEME16_URL = ("https://admin.ansade.mr/wp-content/uploads/2026/01/"
               "Theme-16-Population-etrangere-vivant-en-Mauritanie.pdf")
DESA = os.path.join(RAW, "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
DESA_URL = ("https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/"
            "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
LOOKUP = os.path.join(ROOT, "data", "geo", "mr", "mr_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "mr.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "mr_foreign.csv")

OTHER_NODE = "other.mr"
FOREIGNERS = 125_933
MAURITANIANS_PRINTED = 4_801_600
REFUGEES = 46_800
REFUGEE_UNIT = "MR01"                 # Hodh Chargui, the Mbera camps
REFUGEE_ISO = "ML"
ML_CSV = os.path.join(ROOT, "data", "normalized", "ml.csv")

# Where the refugees came from. UNHCR, "Areas of Origin of Malian refugees living in Mbera camp as
# of 17 Sep 2018" (map mli_refugee_origins, data.unhcr.org/en/documents/details/79788, opened
# 2026-09-15), shades Mali's nine régions before the 2023 reform in four classes of "% of total
# population living in M'Bera", each labelled with its upper bound. Tombouctou is the only région
# in "<= 89.7%", Mopti the only one in "<= 6,5 %" and Ségou the only one in "<= 3,6 %"; Kidal, Gao,
# Kayes, Koulikoro, Bamako and Sikasso are "<= 0,1%". The three single-région bounds sum to 99.8%,
# so they are read as those régions' shares and the last 0.2% is split equally over the six.
ORIGIN_2018 = {"tombouctou": 0.897, "mopti": 0.065, "segou": 0.036}
ORIGIN_2018_LOW = ("kidal", "gao", "kayes", "koulikoro", "bamako", "sikasso")
# Mali's census nodes (taxonomy/ml2022.py) -> this layer's: Catholics on origin_religion's Latin
# Catholic, Mali's residual on Mauritania's.
ML_NODE_FOLD = {"christianity.catholic": origin.CATH, "other.ml": OTHER_NODE}

# Tableau 16.2's label -> the key its composition is built under
GROUPS = {"Mali": "ML", "Sénégal": "SN", "Maroc": "MA", "Algérie": "DZ",
          "Autres pays africains": "AFRICA_OTHER", "Autres pays arabes": "ARAB_OTHER",
          "Europe": "EUROPE", "Reste du Monde": "REST"}
A5_ORDER = ["ML", "SN", "MA", "DZ", "AFRICA_OTHER", "ARAB_OTHER", "EUROPE", "REST"]

# UN DESA's origin names for Mauritania -> ISO, by the census group they fall in. Trailing `*` is
# DESA's footnote mark and is stripped before matching.
DESA_GROUPS = {
    "AFRICA_OTHER": {"Guinea": "GN", "Guinea-Bissau": "GW", "Benin": "BJ", "Côte d'Ivoire": "CI",
                     "Cameroon": "CM", "Ghana": "GH", "Niger": "NE", "Togo": "TG",
                     "Democratic Republic of the Congo": "CD", "Gabon": "GA", "Chad": "TD"},
    "ARAB_OTHER": {"Syrian Arab Republic": "SY", "Saudi Arabia": "SA", "Tunisia": "TN",
                   "Egypt": "EG", "State of Palestine": "PS", "Libya": "LY", "Lebanon": "LB",
                   "Iraq": "IQ", "United Arab Emirates": "AE", "Kuwait": "KW"},
    "EUROPE": {"Spain": "ES", "Belgium": "BE", "Russian Federation": "RU", "Italy": "IT"},
    "REST": {"Brazil": "BR", "China": "CN", "United States of America": "US",
             "Republic of Korea": "KR", "Argentina": "AR", "Canada": "CA"},
}
DESA_NAMED_APART = {"Mali", "Senegal", "Morocco", "Algeria"}   # census groups of their own
DESA_LEFT_OUT = {"Western Sahara"}                             # see the docstring
COVER_BAR = 0.5
USE_DESA = {"AFRICA_OTHER": True, "ARAB_OTHER": True, "EUROPE": False, "REST": False}
REGIONAL = {"AFRICA_OTHER": ["All Sub-Saharan Africa"], "ARAB_OTHER": ["All Middle East-North Africa"],
            "EUROPE": ["All Europe"],
            "REST": ["All Asia-Pacific", "All Latin America-Caribbean", "All North America"]}

# Written before the first run (2026-09-15): the foreigner layer's Christians as a share of the
# census total, over Pew 2020's Christian share for everyone living in Mauritania.
CHRISTIAN_BAND = (0.5, 2.0)

# note_public's figures, measured 2026-09-15 and asserted; remeasured the same day once the refugees
# took northern Mali's census shares (they were 113,222 / 12,711 / 7,031 / 3,652 on Pew's Mali row)
NOTE = dict(nationals=4801598, foreigners=125933, foreign_muslim=115588, non_muslim=10345,
            christians=6136, unaffiliated=2497)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0 Safari/537.36"}
    for url, dst, magic in ((THEME16_URL, THEME16, b"%PDF"), (DESA_URL, DESA, b"PK")):
        if os.path.exists(dst) and os.path.getsize(dst) > 500_000:
            continue
        print("  GET", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=600) as r:
            data = r.read()
        if not data.startswith(magic):
            raise SystemExit(f"{url} did not return the expected file")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)


def theme16():
    import fitz

    with open(THEME16, "rb") as fh:
        if b"%%EOF" not in fh.read()[-2048:]:
            raise SystemExit(f"{THEME16} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(THEME16)
    pages = [[ln.strip() for ln in pg.get_text().splitlines() if ln.strip()] for pg in doc]

    def after(prefix):
        for ls in pages:
            for j, ln in enumerate(ls):
                if ln.startswith(prefix) and "..." not in ln and "…" not in ln:
                    return ls[j + 1:]
        raise SystemExit(f"Thème 16 has no {prefix!r}")

    t161 = after("Tableau 16. 1 :")
    mauritanians = num(t161[t161.index("Mauritaniens") + 5])
    if mauritanians != MAURITANIANS_PRINTED:
        raise SystemExit(f"Tableau 16.1's Mauritanians are {mauritanians:,}")

    t162 = after("Tableau 16. 2:")
    groups = {GROUPS[lab]: num(t162[t162.index(lab) + 5]) for lab in GROUPS}
    total = num(t162[t162.index("Total étrangers") + 5])
    if total != FOREIGNERS or sum(groups.values()) != FOREIGNERS:
        raise SystemExit(f"Tableau 16.2: groups sum to {sum(groups.values()):,}, total {total:,}")

    a5 = [num(x) for x in after("Tableau A. 5 :") if isnum(x)]
    if len(a5) != 81:
        raise SystemExit(f"Tableau A.5 has {len(a5)} numbers, expected 9 rows of 9")
    a5 = [a5[i:i + 9] for i in range(0, 81, 9)]
    urban_rural = {}
    for key, row in zip(A5_ORDER + ["Total"], a5):
        if abs(row[2] + row[5] - row[8]) > 1:
            raise SystemExit(f"Tableau A.5 {key}: urban {row[2]} + rural {row[5]} != {row[8]}")
        if key != "Total" and row[8] != groups[key]:
            raise SystemExit(f"Tableau A.5 {key}: {row[8]:,} against Tableau 16.2's {groups[key]:,}")
        urban_rural[key] = (row[2], row[5])

    t165 = after("Tableau 16. 5:")
    refugees = num(t165[t165.index("Total") + 5])
    ind = next(ls for ls in pages if any(l.startswith("Réfugiés et demandeur") for l in ls))
    k = next(j for j, l in enumerate(ind) if l.startswith("Réfugiés et demandeur"))
    if refugees != REFUGEES or num(ind[k + 3]) != REFUGEES:
        raise SystemExit(f"refugees: Tableau 16.5 {refugees:,}, indicators {ind[k + 3]}")

    t166 = after("Tableau 16. 6 :")
    i = next(j for j, l in enumerate(t166) if "Chargui" in l)
    per = {}
    for pc, (_n, _c, frag) in WILAYAS.items():
        if frag not in t166[i]:
            raise SystemExit(f"Tableau 16.6 row {t166[i]!r} should be {pc}")
        per[pc] = num(t166[i + 5])
        i += 7
    if t166[i] != "Total" or num(t166[i + 5]) != FOREIGNERS or sum(per.values()) != FOREIGNERS:
        raise SystemExit(f"Tableau 16.6: rows sum to {sum(per.values()):,}")
    print(f"Thème 16: {FOREIGNERS:,} foreigners in 8 groups (Tableau 16.2, checked against A.5), "
          f"{REFUGEES:,} refugees (16.5), per wilaya (16.6)")
    return groups, urban_rural, per


def desa():
    """{census group: {iso: DESA 2024 stock}} for Mauritania, every named origin accounted for."""
    with zipfile.ZipFile(DESA) as z:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
            for n in z.namelist():
                if n != "xl/styles.xml":            # openpyxl is slow on the stylesheet
                    out.writestr(n, z.read(n))
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name="Table 1", header=None, engine="openpyxl")
    hdr = next(i for i in range(2, 20)
               if any("of destination" in str(x) for x in df.iloc[i])
               and any("of origin" in str(x) for x in df.iloc[i]))
    cols = [str(x).strip() for x in df.iloc[hdr]]
    dcol = next(i for i, x in enumerate(cols) if "of destination" in x)
    ocol = next(i for i, x in enumerate(cols) if "of origin" in x)
    ccol = next(i for i, x in enumerate(cols) if x == "Location code of origin")
    ycol = next(i for i, x in enumerate(cols) if x.replace(".0", "") == "2024")
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "Mauritania"]
    code = pd.to_numeric(m[ccol], errors="coerce")
    ctry = m[code < 900]
    stock = dict(zip(ctry[ocol].astype(str).str.strip().str.rstrip("*").str.strip(),
                     pd.to_numeric(ctry[ycol]).astype(int)))
    known = set().union(*[set(g) for g in DESA_GROUPS.values()]) | DESA_NAMED_APART | DESA_LEFT_OUT
    if set(stock) != known:
        raise SystemExit(f"DESA's origins for Mauritania changed: new {sorted(set(stock) - known)}, "
                         f"gone {sorted(known - set(stock))}")
    world = int(pd.to_numeric(m.loc[m[ocol].astype(str).str.strip() == "World", ycol]).iloc[0])
    print(f"  UN DESA 2024: {world:,} migrants in Mauritania, {len(stock)} origins named; "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(stock.items(), key=lambda kv: -kv[1])[:6]))
    return {g: {iso: stock[name] for name, iso in names.items()} for g, names in DESA_GROUPS.items()}


def pew_table():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    return t[t["Year"] == 2020].set_index("Country")


def composition(pew, weights=None, regional=None):
    """{node: share}: `weights` {iso: w} through Pew's country rows, or `regional` Pew rows summed."""
    out = {}
    if regional:
        missing = [r for r in regional if r not in pew.index]
        if missing:
            raise SystemExit(f"Pew has no row {missing}")
        items = [("XX", {f: float(pew.loc[regional, f].sum()) for f in origin.FAMILIES}, 1.0)]
    else:
        items = []
        tot = float(sum(weights.values()))
        for iso, w in weights.items():
            pn = origin.PEW_BY_ISO.get(iso)
            if pn not in pew.index:
                raise SystemExit(f"Pew has no row {pn!r} for {iso}")
            items.append((iso, {f: float(pew.loc[pn, f]) for f in origin.FAMILIES}, w / tot))
    for iso, row, frac in items:
        for node, s in origin.composition(iso, row, OTHER_NODE).items():
            node = "islam" if node.startswith("islam") else node
            out[node] = out.get(node, 0.0) + frac * s
    return out


def refugee_composition():
    """{node: share} for the refugees in Hodh Chargui.

    Mali's 2022 census (`data/normalized/ml.csv`, 20 régions) pooled into the nine régions before the
    2023 reform that UNHCR's 2018 map uses (`sources/ml_geo.py` PARENT), each weighted by
    `ORIGIN_2018`, on the nodes `taxonomy/ml2022.py` maps them to, folded by `ML_NODE_FOLD`.
    """
    weights = dict(ORIGIN_2018)
    low = 1.0 - sum(weights.values())
    weights.update({r: low / len(ORIGIN_2018_LOW) for r in ORIGIN_2018_LOW})
    if (set(weights) != set(ML_PARENT.values())
            or len(weights) != len(ORIGIN_2018) + len(ORIGIN_2018_LOW)):
        raise SystemExit(f"ORIGIN_2018 should name each of Mali's old régions once: {sorted(weights)}")
    ml = pd.read_csv(ML_CSV)
    ml["old"] = ml["geo_id"].map(ML_PARENT)
    if ml["old"].isna().any():
        raise SystemExit(f"ml.csv régions with no old région: "
                         f"{sorted(ml.loc[ml['old'].isna(), 'geo_id'].unique())}")
    extra = set(ml["source_category"]) - set(ML_MAP) - {"Total"}
    if extra:
        raise SystemExit(f"ml.csv categories taxonomy/ml2022.py does not map: {sorted(extra)}")
    out, shown = {}, []
    for old, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        rows = ml[ml["old"] == old]
        cats = rows[rows["source_category"] != "Total"].groupby("source_category")["count"].sum()
        total = int(rows.loc[rows["source_category"] == "Total", "count"].sum())
        if abs(int(cats.sum()) - total) > 2 * len(rows):
            raise SystemExit(f"{old}: ml.csv's answers sum to {int(cats.sum()):,} against {total:,}")
        for cat, n in cats.items():
            node = ML_NODE_FOLD.get(ML_MAP[cat], ML_MAP[cat])
            out[node] = out.get(node, 0.0) + w * float(n) / float(cats.sum())
        if w >= 0.01:
            shown.append(f"{old} {w:.1%} of them, {cats['Musulman'] / cats.sum():.2%} Muslim")
    print("  refugees' origin, UNHCR 2018 x Mali RGPH 2022: " + "; ".join(shown))
    return out


def christian_witness(fcounts, pew, total):
    chr_ = int(fcounts[[c for c in fcounts.columns if c.startswith("christianity")]].sum().sum())
    drawn = chr_ / total
    pew_share = float(pew.loc["Mauritania", "Christians"]) / float(pew.loc["Mauritania", "Population"])
    r = drawn / pew_share
    print(f"\n  witness: the foreigner layer gives {chr_:,} Christians, {drawn:.3%} of the census; "
          f"Pew 2020 has {pew_share:.3%} for everyone living in Mauritania ({int(pew.loc['Mauritania', 'Christians']):,}); "
          f"ratio {r:.2f}, band {CHRISTIAN_BAND}")
    if not CHRISTIAN_BAND[0] <= r <= CHRISTIAN_BAND[1]:
        raise SystemExit("the foreigner layer's Christians are outside the band; read sources/mr.md §4 "
                         "before choosing a construction")
    return chr_


def main():
    if "--fetch" in sys.argv or not os.path.exists(THEME16) or not os.path.exists(DESA):
        fetch()
    groups, urban_rural, per = theme16()
    named = desa()
    pew = pew_table()

    # ---- the composition of each census group ----
    comps = {}
    for g, n in groups.items():
        if len(g) == 2:
            comps[g] = composition(pew, weights={g: 1.0})
            how = "Pew's own row"
        else:
            cover = sum(named[g].values()) / n
            use = cover >= COVER_BAR
            if use != USE_DESA[g]:
                raise SystemExit(f"{g}: DESA's named origins cover {cover:.2f} of the census group, "
                                 f"and USE_DESA says {USE_DESA[g]}")
            if use:
                comps[g] = composition(pew, weights=named[g])
                how = f"DESA's {len(named[g])} named origins ({sum(named[g].values()):,}, {cover:.0%} of it)"
            else:
                comps[g] = composition(pew, regional=REGIONAL[g])
                how = (f"Pew's {' + '.join(REGIONAL[g])} (DESA names {sum(named[g].values()):,}, "
                       f"{cover:.0%})")
        top = sorted(comps[g].items(), key=lambda kv: -kv[1])[:4]
        print(f"    {g:<13} {n:>6,}  {how}: " + ", ".join(f"{k} {v:.1%}" for k, v in top))

    # ---- refugees first, then everyone else at the national mix without them ----
    if per[REFUGEE_UNIT] < REFUGEES or urban_rural[REFUGEE_ISO][1] < REFUGEES:
        raise SystemExit("Hodh Chargui's foreigners or Mali's rural foreigners are fewer than the refugees")
    refugee = refugee_composition()
    for label, comp in (("Pew 2020's Mali row", comps[REFUGEE_ISO]), ("as drawn", refugee)):
        nm = 1.0 - comp.get("islam", 0.0)
        print(f"    refugees at {label}: {nm:.3%} not Muslim, {REFUGEES * nm:,.0f} of {REFUGEES:,}")
    rest = dict(groups)
    rest[REFUGEE_ISO] -= REFUGEES
    rest_n = sum(rest.values())
    mix = {}
    for g, n in rest.items():
        for node, s in comps[g].items():
            mix[node] = mix.get(node, 0.0) + (n / rest_n) * s
    nodes = sorted(set(mix) | set(refugee))
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str}).set_index("geo_id")
    rows = {}
    for pc in WILAYAS:
        others = per[pc] - (REFUGEES if pc == REFUGEE_UNIT else 0)
        rows[pc] = {n: others * mix.get(n, 0.0)
                    + (REFUGEES * refugee.get(n, 0.0) if pc == REFUGEE_UNIT else 0.0)
                    for n in nodes}
    fm = pd.DataFrame.from_dict(rows, orient="index")[nodes]
    fcounts = round_within_rows(fm)
    if not all(int(fcounts.loc[pc].sum()) == per[pc] for pc in WILAYAS):
        raise SystemExit("a wilaya's rounded foreign counts do not sum to its foreigners")
    print(f"  refugees: {REFUGEES:,} Malians in {lut.loc[REFUGEE_UNIT, 'name']}; the other "
          f"{rest_n:,} foreigners at the national mix without them")

    total = int(lut["pop"].sum())
    christians = christian_witness(fcounts, pew, total)

    # ---- nationals ----
    nat = {pc: int(lut.loc[pc, "pop"]) - per[pc] for pc in WILAYAS}
    if min(nat.values()) <= 0:
        raise SystemExit("a wilaya has no nationals left")
    out = pd.DataFrame({"geo_id": list(nat), "geo_level": "wilaya",
                        "geo_name": [lut.loc[pc, "name"] for pc in nat],
                        "source_category": "Muslim", "count": list(nat.values()),
                        "basis": "estimate", "year": 2023, "source_id": "mr_rgph2023_nationals",
                        "note": "no source asks religion; every Mauritanian national is drawn on "
                                "Islam (sources/mr.py), on the RGPH 2023 count less its foreign "
                                "residents"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    ext = fcounts.stack().rename("count").reset_index()
    ext.columns = ["geo_id", "node", "count"]
    ext = ext[ext["count"] > 0].copy()
    ext["geo_level"] = "wilaya"
    ext["geo_name"] = ext["geo_id"].map(lut["name"])
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_derived"
    ext["year"] = 2023
    ext["source_id"] = "rgph2023_foreigners_x_pew2020"
    ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
         "source_id"]].to_csv(OUT_FOREIGN, index=False, encoding="utf-8")

    fx = ext.groupby("node")["count"].sum().sort_values(ascending=False)
    muslim_f = int(fx.get("islam", 0))
    non_muslim = int(ext["count"].sum()) - muslim_f
    print(f"\nwrote {OUT} ({sum(nat.values()):,} nationals) and {OUT_FOREIGN} "
          f"({int(ext['count'].sum()):,} foreigners, {ext['node'].nunique()} nodes); {total:,} people")
    print(f"    drawn Muslim {(sum(nat.values()) + muslim_f) / total:.3%}; non-Muslim {non_muslim:,} "
          f"({non_muslim / total:.3%}); Pew 2020 for everyone living there: 99.185% Muslim")
    print("    foreign: " + ", ".join(f"{n} {int(v):,}" for n, v in fx.items()))
    big = ext[ext["node"] != "islam"].groupby("geo_name")["count"].sum().sort_values(ascending=False)
    print("    non-Muslim foreigners by wilaya: " + ", ".join(f"{k} {int(v):,}" for k, v in big.items()))

    got = dict(nationals=sum(nat.values()), foreigners=int(ext["count"].sum()),
               foreign_muslim=muslim_f, non_muslim=non_muslim, christians=christians,
               unaffiliated=int(fx.get("unaffiliated", 0)))
    print(f"\n  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
