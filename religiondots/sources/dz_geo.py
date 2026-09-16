"""Algeria: boundaries and populations for the 48 wilayas.

Writes data/geo/dz/dz_wilayas.gpkg and data/geo/dz/dz_lookup.csv.

Two sources:

  * **boundaries**: COD-AB `cod-ab-dza` v01 ADM1 (OCHA ROMENA, GAUL lineage with UNICEF p-codes,
    valid from 2021-01-20, reviewed 2024-12-19), 48 features.
  * **populations and areas**: ONS, *Annuaire Statistique de l'Algérie* no. 31, chapter III
    (`ons.dz/IMG/pdf/demographie.pdf`). Table 4 is the 2008 census's resident population of
    ordinary and collective households by wilaya (RGPH 2008, exhaustive count, 34,080,030); Table
    29 is density per km2 by wilaya at each census, which gives ONS's own area as population over
    density.

## WHY 2008, AND WHY THE 48 WILAYAS

The sixth census (RGPH, 25 September to 16 October 2022) was held and its wilaya results have not
been published: a press piece of 27 August 2023 (La Nation) says nothing had been released, and
ONS's own home page in October 2025 still links only the 2008 results, with a commented-out
`rgph2020` link beside them. So Table 4 is the last count of people by wilaya the office has
published, and it is drawn as counted, not scaled to a national estimate (`sources/dz.md` §5).

The 2008 census counts 48 wilayas. Ten southern wilayas were created from them in 2019, and the
Arab Barometer codes Algeria on the 48 in every wave through 2022 (`sources/dz.py`). COD-AB is on
the 48 as well, so the census, the survey and the polygons share one unit set and nothing is
dissolved.

## THE JOIN, AND WHY NEITHER NAME NOR P-CODE IS THE KEY

COD's p-codes run DZ001 to DZ048 in alphabetical order of its English names (Alger is DZ004),
not in the official wilaya order, so a p-code is not a wilaya number. The key is an authored table,
`WILAYAS`: official number -> COD p-code -> ONS's spelling. It is checked four ways:

  1. the table covers COD's 48 p-codes and ONS's 48 Table 4 rows exactly;
  2. **Table 4 lists the wilayas in official order**, so row n must be the wilaya the table names
     for number n (ONS prints no number beside the name; its order is the witness);
  3. **ONS's area (Table 4 population over Table 29 density) against COD's geometry**, by rank
     with a permutation null. Wilayas run from 755 km2 (Alger) to 603,180 km2 (Tamanrasset);
  4. **the four Saharan wilayas with under one person per km2 in 2008** (Illizi, Tamanrasset,
     Tindouf, Adrar, by ONS's own density column) must come out the four sparsest on COD's areas.

Usage:
    python sources/dz_geo.py --fetch    COD-AB geojson zip (5.6 MB) and the ONS chapter (2.2 MB)
    python sources/dz_geo.py            rebuild from data/raw/dz/
"""

import io
import os
import re
import ssl
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "dz")
OUT_DIR = os.path.join(ROOT, "data", "geo", "dz")
OUT = os.path.join(OUT_DIR, "dz_wilayas.gpkg")
LOOKUP = os.path.join(OUT_DIR, "dz_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

COD_URL = ("https://data.humdata.org/dataset/3d17274d-0812-4b21-b87a-4854af4eb244/resource/"
           "d2b759f7-5f43-48b8-8186-21bb096186d1/download/dza_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "dza_admin_boundaries.geojson.zip")
ONS_URL = "https://www.ons.dz/IMG/pdf/demographie.pdf"
ONS_PDF = os.path.join(RAW, "ons_annuaire31_demographie.pdf")

ONS_TOTAL = 34_080_030        # Table 4's own `Total` row, RGPH 2008

# official wilaya number -> (COD p-code, the name this map prints, ONS Table 4's spelling).
WILAYAS = {
    1: ("DZ001", "Adrar", "Adrar"),
    2: ("DZ014", "Chlef", "Chlef"),
    3: ("DZ025", "Laghouat", "Laghouat"),
    4: ("DZ034", "Oum El Bouaghi", "Oum El Bouaghi"),
    5: ("DZ006", "Batna", "Batna"),
    6: ("DZ008", "Béjaïa", "Bejaia"),
    7: ("DZ009", "Biskra", "Biskra"),
    8: ("DZ007", "Béchar", "Bechar"),
    9: ("DZ010", "Blida", "Blida"),
    10: ("DZ012", "Bouira", "Bouira"),
    11: ("DZ041", "Tamanrasset", "Tamanrasset"),
    12: ("DZ042", "Tébessa", "Tebessa"),
    13: ("DZ048", "Tlemcen", "Tlemcen"),
    14: ("DZ043", "Tiaret", "Tiaret"),
    15: ("DZ047", "Tizi Ouzou", "Tizi – Ouzou"),
    16: ("DZ004", "Algiers", "Alger"),
    17: ("DZ016", "Djelfa", "Djelfa"),
    18: ("DZ023", "Jijel", "Jijel"),
    19: ("DZ037", "Sétif", "Setif"),
    20: ("DZ036", "Saïda", "Saïda"),
    21: ("DZ039", "Skikda", "Skikda"),
    22: ("DZ038", "Sidi Bel Abbès", "Sidi-Bel-Abbes"),
    23: ("DZ005", "Annaba", "Annaba"),
    24: ("DZ021", "Guelma", "Guelma"),
    25: ("DZ015", "Constantine", "Constantine"),
    26: ("DZ027", "Médéa", "Médea"),
    27: ("DZ029", "Mostaganem", "Mostaganem"),
    28: ("DZ030", "M'Sila", "M'sila"),
    29: ("DZ026", "Mascara", "Mascara"),
    30: ("DZ033", "Ouargla", "Ouargla"),
    31: ("DZ032", "Oran", "Oran"),
    32: ("DZ017", "El Bayadh", "El-Bayadh"),
    33: ("DZ022", "Illizi", "Illizi"),
    34: ("DZ011", "Bordj Bou Arréridj", "Bordj Bou Arreridj"),
    35: ("DZ013", "Boumerdès", "Boumerdes"),
    36: ("DZ019", "El Tarf", "El-Tarf"),
    37: ("DZ044", "Tindouf", "Tindouf"),
    38: ("DZ046", "Tissemsilt", "Tissemsilt"),
    39: ("DZ018", "El Oued", "El-Oued"),
    40: ("DZ024", "Khenchela", "Khenchela"),
    41: ("DZ040", "Souk Ahras", "Souk-Ahras"),
    42: ("DZ045", "Tipaza", "Tipaza"),
    43: ("DZ028", "Mila", "Mila"),
    44: ("DZ002", "Aïn Defla", "Ain-Defla"),
    45: ("DZ031", "Naâma", "Naâma"),
    46: ("DZ003", "Aïn Témouchent", "Ain-Temouchent"),
    47: ("DZ020", "Ghardaïa", "Ghardaia"),
    48: ("DZ035", "Relizane", "Relizane"),
}

# Under one person per km2 in 2008 by ONS's own Table 29 (read before this was written).
SPARSEST = {33, 11, 37, 1}

# The five wilayas where COD's polygon area is outside 0.85-1.18x of the area ONS's Table 29
# implies (Table 4 population over printed density), measured 2026-09-15: Mila 0.37x, Djelfa
# 0.49x, El Oued 0.81x, Tizi Ouzou 0.83x, Chlef 0.84x. The rank test over all 48 still pins the
# pairing (rho +0.984, no permutation near it), so these are disagreements about where two wilaya
# lines run, or about Table 29's denominators, and not a permuted join. Which of the two is right
# was not settled here; `sources/dz_grid.py` prints Kontur people per wilaya against the census,
# which is the check that matters for where dots land. Named so that a sixth is a failure.
AREA_DISAGREE = {43, 17, 39, 15, 2}

EQUAL_AREA = "EPSG:6933"


def geo_id(n):
    return f"DZ{n:02d}"


def fold(s):
    s = str(s).lower()
    for a, b in (("é", "e"), ("è", "e"), ("ï", "i"), ("â", "a"), ("î", "i"), ("ô", "o")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z]", "", s)


def _ctx():
    # ons.dz's certificate chain does not verify from here (seen 2026-09-15); the bytes are checked
    # for the PDF magic and the %%EOF trailer below instead.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, dst, magic, minsize):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=900, context=_ctx()) as r:
        data = r.read()
    if not data.startswith(magic):
        raise SystemExit(f"{url} did not return the expected file; starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(COD_URL, COD_ZIP, b"PK", 1_000_000)
    _get(ONS_URL, ONS_PDF, b"%PDF", 1_000_000)


def _lines(page_text):
    return [ln.strip() for ln in page_text.splitlines()]


def _num(s):
    s = re.sub(r"[\s  ]", "", s)
    if re.fullmatch(r"\d+", s):
        return int(s)
    if re.fullmatch(r"\d+,\d+", s):
        return float(s.replace(",", "."))
    return None


def _find_page(doc, *needles):
    hits = [i for i, p in enumerate(doc) if all(n in p.get_text() for n in needles)]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} pages of {ONS_PDF} contain {needles}, expected exactly 1")
    return doc[hits[0]].get_text()


def ons_tables():
    """Table 4 (population, sex, growth) and Table 29 (density) as 48 rows in printed order."""
    import fitz

    with open(ONS_PDF, "rb") as fh:
        if b"%%EOF" not in fh.read()[-1024:]:
            raise SystemExit(f"{ONS_PDF} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(ONS_PDF)

    # ---- Table 4: name, male, female, total, growth, anchored on its own header and Total ----
    lines = _lines(_find_page(doc, "Tableau 4:", "Masculin", "RGPH 2008"))
    start = next(i for i, ln in enumerate(lines) if ln.startswith("(1998-2008) en %"))
    end = next(i for i, ln in enumerate(lines) if i > start and ln == "Total")
    body = [ln for ln in lines[start + 1:end] if ln]
    if len(body) != 48 * 5:
        raise SystemExit(f"Table 4 body has {len(body)} lines, expected 240 (48 x 5)")
    t4, off_by = [], []
    for k in range(48):
        name, m, f, tot, rate = body[5 * k:5 * k + 5]
        m, f, tot, rate = _num(m), _num(f), _num(tot), _num(rate)
        if _num(name) is not None or None in (m, f, tot, rate):
            raise SystemExit(f"Table 4 row {k + 1} does not parse: {body[5 * k:5 * k + 5]}")
        # ONS's own rows can miss by a person (Oum El Bouaghi prints 314,084 + 307,527 against
        # 621,612) while the national row adds exactly. A misread digit moves a row by ten or more.
        if abs(m + f - tot) > 2:
            raise SystemExit(f"Table 4 {name}: {m:,} + {f:,} != {tot:,}")
        if m + f != tot:
            off_by.append(f"{name} {m + f - tot:+d}")
        t4.append((name, tot, float(rate)))
    if off_by:
        print(f"  Table 4 rows whose sexes miss the printed total by a person or two: "
              f"{', '.join(off_by)} (the total is drawn)")
    total = [_num(x) for x in lines[end + 1:end + 4]]
    if total[2] != ONS_TOTAL or total[0] + total[1] != ONS_TOTAL:
        raise SystemExit(f"Table 4's Total row reads {total}, not {ONS_TOTAL:,}")
    if sum(t for _n, t, _r in t4) != ONS_TOTAL:
        raise SystemExit("Table 4's 48 rows do not sum to its own Total")

    # ---- Table 29: name then the densities at each census; 2008 is the last number ----
    lines = _lines(_find_page(doc, "Tableau 29:", "Densité", "1998*"))
    start = next(i for i, ln in enumerate(lines) if ln == "2008")
    rows, cur = [], None
    for ln in lines[start + 1:]:
        if not ln:
            continue
        if ln.startswith(("Source", "Total", "Ensemble", "*")):
            break
        v = _num(ln)
        if v is None and ln not in ("-", "--", "–"):
            cur = [ln, []]
            rows.append(cur)
        elif v is not None and cur is not None:
            cur[1].append(float(v))
    rows = rows[:48]
    if len(rows) != 48 or any(not vals for _n, vals in rows):
        raise SystemExit(f"Table 29 parsed {len(rows)} rows, expected 48 with values")
    t29 = [(n, vals[-1]) for n, vals in rows]
    return t4, t29


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, ONS_PDF):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")

    with zipfile.ZipFile(COD_ZIP) as zf:
        g = gpd.read_file(io.BytesIO(zf.read("dza_admin1.geojson")))
    if len(g) != 48:
        raise SystemExit(f"{len(g)} COD-AB ADM1 features, expected 48")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read COD-AB ADM1: {len(g)} wilayas, {g.crs}")

    t4, t29 = ons_tables()
    print(f"  ONS Table 4: 48 wilayas, {ONS_TOTAL:,} people, male + female = total in every row")

    # ---- witness 1: the authored table covers both sides exactly ----
    if sorted(v[0] for v in WILAYAS.values()) != sorted(g["adm1_pcode"]):
        raise SystemExit("WILAYAS' p-codes and COD's do not agree: "
                         f"{sorted(set(v[0] for v in WILAYAS.values()) ^ set(g['adm1_pcode']))}")
    print("  witness 1: the authored table covers all 48 COD p-codes")

    # ---- witness 2: Table 4's printed order is the official numbering ----
    bad = [(n, WILAYAS[n][2], t4[n - 1][0]) for n in WILAYAS
           if fold(WILAYAS[n][2]) != fold(t4[n - 1][0])]
    if bad:
        raise SystemExit(f"Table 4 row n is not wilaya n for: {bad}")
    bad29 = [(n, t4[n - 1][0], t29[n - 1][0]) for n in WILAYAS
             if fold(t4[n - 1][0]) != fold(t29[n - 1][0])]
    if bad29:
        raise SystemExit(f"Table 29's order differs from Table 4's at: {bad29}")
    print("  witness 2: Tables 4 and 29 list all 48 wilayas in official order, as WILAYAS numbers "
          "them")

    cod_name = dict(zip(g["adm1_pcode"], g["adm1_name"]))
    rows = []
    for n, (pc, name, ons) in WILAYAS.items():
        pop = t4[n - 1][1]
        dens = t29[n - 1][1]
        rows.append(dict(number=n, geo_id=geo_id(n), pcode=pc, name=name, ons_name=ons,
                         cod_name=cod_name[pc], pop=pop, growth_9808=t4[n - 1][2],
                         density_2008=dens, ons_area=pop / dens))
    lut = pd.DataFrame(rows)
    g = g.merge(lut, left_on="adm1_pcode", right_on="pcode", how="inner", validate="1:1")
    if len(g) != 48:
        raise SystemExit("the p-code merge lost wilayas")
    g["cod_area"] = g.to_crs(EQUAL_AREA).geometry.area / 1e6

    # ---- witness 3: ONS's area against COD's geometry ----
    rho = stats.spearmanr(g["ons_area"], g["cod_area"]).statistic
    rng = np.random.default_rng(0)
    a, b = g["ons_area"].to_numpy(), g["cod_area"].to_numpy()
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(5000)])
    beaten = int((perm >= rho).sum())
    g["area_ratio"] = g["cod_area"] / g["ons_area"]
    print(f"  witness 3: ONS area (Table 4 / Table 29) vs COD area over 48: rho = {rho:+.3f}; "
          f"{beaten} of 5,000 random pairings reach it (best {perm.max():+.3f})")
    # Table 29 prints density to 0.1, so a sparse wilaya's ONS area carries a large rounding
    # error (Illizi at 0.2/km2 is +/-25%). The band is asserted only where density is 10 or more.
    dense = g[g["density_2008"] >= 10]
    worst = dense.reindex((dense["area_ratio"] - 1).abs().sort_values(ascending=False).index)
    for _i, r in worst.head(4).iterrows():
        print(f"      {r['name']:<20} ONS {r['ons_area']:>9,.0f} km2   COD {r['cod_area']:>9,.0f} "
              f"km2   {r['area_ratio']:.2f}x")
    out_band = dense[((dense["area_ratio"] < 0.85) | (dense["area_ratio"] > 1.18))]
    if beaten:
        raise SystemExit(f"the area witness fails: {beaten} permutations reach rho")
    if set(out_band["number"]) != AREA_DISAGREE:
        raise SystemExit(f"wilayas outside 0.85-1.18x of ONS's area: "
                         f"{out_band[['number', 'name', 'area_ratio']].to_dict('records')}, not the "
                         f"five known ones {sorted(AREA_DISAGREE)}; read them before trusting COD")
    print(f"      the {len(AREA_DISAGREE)} outside 0.85-1.18x are the known ones (see "
          "AREA_DISAGREE); the rank test above is what pins the join")

    # ---- witness 4: the sparsest four ----
    g["density_cod"] = g["pop"] / g["cod_area"]
    order = g.sort_values("density_cod")
    print("  witness 4: sparsest on COD areas: "
          + ", ".join(f"{r['name']} {r['density_cod']:.2f}/km2" for _i, r in order.head(5).iterrows()))
    if set(order["number"].head(4)) != SPARSEST:
        raise SystemExit("the four sparsest wilayas on COD's areas are not the four ONS prints "
                         "under one person per km2; the join is permuted")

    same = int((g["cod_name"].map(fold) == g["ons_name"].map(fold)).sum())
    print(f"  and {same} of 48 COD names equal ONS's letter for letter (not asserted; COD "
          "truncates and hyphenates)")

    g["unit"] = g["geo_id"]
    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "geo_id", "pcode", "number", "pop", "geometry"]].to_file(
        OUT, layer="wilayas", driver="GPKG")
    print(f"\nwrote {OUT} (48 polygons)")
    lk = (g[["geo_id", "unit", "number", "pcode", "name", "pop", "growth_9808", "ons_area",
             "cod_area"]].sort_values("number").reset_index(drop=True))
    lk.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} (48 rows, {int(lk['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
