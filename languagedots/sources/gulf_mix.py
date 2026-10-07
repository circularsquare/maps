"""The Gulf states' "home mix" method, shared by sources/{ae,kw,qa,om,bh}_build.py.

None of the Gulf states asks anyone a language. Built as Saudi Arabia was (sources/sa_census.py,
sources/sa.md), under Anita's 2026-10-05 ruling for countries with no language question
(AGENT_BRIEF.md §2): citizens on the national Arabic variety, everyone else by nationality, each
nationality on its home country's language or language mix. EVERY ROW IS `derived`.

What this module adds to sa_census.py's pieces:
  * `desa(dest)`: UN DESA, International Migrant Stock 2024 (religiondots' copy, read-only), one
    destination's origins by sex. The only origin table the UAE, Qatar and Bahrain have; DESA's
    own `Others` row is drawn on `other` (unnamed origins, so an unnamed language), never spread.
  * `india(dest)`: Indians by state of origin as Saudi Arabia's, with the Keralite share taken
    for THIS destination from the Kerala Migration Survey 2023, Table 3.7 (KMS_DEST below).
  * `origin_mix(iso, dest, n)`: one nationality's language mix.

Gulf Arabic (sources/gulf.md §1): one node, `afroasiatic.gulf_arabic` (Glottolog gulf1241), for
the citizens of the UAE, Kuwait, Qatar and Bahrain (Bahrain's Sunni citizens; its Shia Baharna
are on `afroasiatic.baharna_arabic`, baha1259). Oman's citizens are on `afroasiatic.omani_arabic`
(oman1239), which Glottolog keeps apart. Saudi Arabia's own node stays as sa built it. A citizen
of one Gulf state living in another is drawn on their own state's node.
"""
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "taxonomy"), str(ROOT)]

import pandas as pd  # noqa: E402

import sa_census as sa  # noqa: E402  (its pieces; main() only runs as a script)
import origin_mix as _om  # noqa: E402

RD = ROOT.parent / "religiondots"
DESA_XLSX = RD / "data" / "raw" / "mr" / "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"

AR = "afroasiatic"
GULF_ARABIC = f"{AR}.gulf_arabic"
BAHARNA_ARABIC = f"{AR}.baharna_arabic"
OMANI_ARABIC = f"{AR}.omani_arabic"
OTHER = "other"

# Kerala Migration Survey 2023 (IIMAD, draft report, 2024; data/raw/gulf/KMS-2023-Report.pdf),
# Table 3.7, country of residence of Kerala's 2,154,275 emigrants (Table 3.1), both sexes.
KMS_DEST = {"AE": 0.386, "SA": 0.169, "OM": 0.064, "QA": 0.091, "KW": 0.058, "BH": 0.037}

# DESA's origin names for the Gulf destinations -> ISO 3166 alpha-2
DESA_ISO = {
    "India": "IN", "Bangladesh": "BD", "Pakistan": "PK", "Egypt": "EG", "Philippines": "PH",
    "Indonesia": "ID", "Sri Lanka": "LK", "Nepal": "NP", "Yemen": "YE", "Jordan": "JO",
    "Sudan": "SD", "Syrian Arab Republic": "SY", "State of Palestine": "PS", "Kuwait": "KW",
    "Lebanon": "LB", "Türkiye": "TR", "United Kingdom": "GB", "United Arab Emirates": "AE",
    "Eritrea": "ER", "United States of America": "US", "Nigeria": "NG", "France": "FR",
    "United Republic of Tanzania": "TZ", "South Sudan": "SS", "Thailand": "TH", "Ethiopia": "ET",
    "Somalia": "SO", "Morocco": "MA", "Afghanistan": "AF", "Saudi Arabia": "SA", "Tunisia": "TN",
    "Chad": "TD", "Netherlands": "NL", "Bahrain": "BH", "Qatar": "QA", "Oman": "OM",
    "Uganda": "UG", "Myanmar": "MM", "Iran (Islamic Republic of)": "IR", "Iraq": "IQ",
}

# single-language overrides of fr_build.COUNTRY_LANG (node ids): sa_census.OVERRIDE plus the Gulf
OVERRIDE = dict(sa.OVERRIDE)
OVERRIDE.update({"AE": GULF_ARABIC, "KW": GULF_ARABIC, "QA": GULF_ARABIC, "BH": GULF_ARABIC,
                 "SA": sa.SAUDI_ARABIC, "OM": OMANI_ARABIC})
# sa_census.OVERRIDE draws Myanmar as Rohingya because GASTAT's Myanmar nationals are the
# Rohingya; nothing says that of Myanmar nationals elsewhere in the Gulf (Oman's are domestic
# workers, 99% women), so outside Saudi Arabia they take Myanmar's drawn mix.
OVERRIDE.pop("MM")

HOME_MIN = 20_000   # an origin on this map with this many people takes its drawn mix (sa's rule)
DRAWN = {p.stem.upper() for p in (ROOT / "countries").glob("*.py") if not p.stem.startswith("_")}

_cache = {}


def desa(dest, year=2024):
    """{iso: (both, men, women)} of UN DESA's named origins in `year`, and the `Others` triple."""
    if "desa" not in _cache:
        with zipfile.ZipFile(DESA_XLSX) as z:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
                for n in z.namelist():
                    if n != "xl/styles.xml":            # openpyxl is slow on the stylesheet
                        out.writestr(n, z.read(n))
        buf.seek(0)
        _cache["desa"] = pd.read_excel(buf, sheet_name="Table 1", header=None, engine="openpyxl")
    df = _cache["desa"]
    hdr = next(i for i in range(2, 20)
               if any("of destination" in str(x) for x in df.iloc[i])
               and any("of origin" in str(x) for x in df.iloc[i]))
    cols = list(df.iloc[hdr])
    names = [str(x).strip() for x in cols]
    dcol = next(i for i, x in enumerate(names) if "of destination" in x)
    ocol = next(i for i, x in enumerate(names) if "of origin" in x)
    ccol = next(i for i, x in enumerate(names) if x == "Location code of origin")
    ycols = [i for i, x in enumerate(cols) if str(x) in (str(year), f"{year}.0")]
    if len(ycols) != 3:
        raise SystemExit(f"DESA: expected three {year} columns, got {ycols}")
    body = df.iloc[hdr + 1:]
    clean = lambda s: s.astype(str).str.strip().str.rstrip("*").str.strip()  # noqa: E731
    m = body[clean(body[dcol]) == dest]
    name = clean(m[ocol])
    world = tuple(int(pd.to_numeric(m.loc[name == "World", y]).iloc[0]) for y in ycols)
    code = pd.to_numeric(m[ccol], errors="coerce")
    keep = m[(code < 900) | (name == "Others")]
    out, others = {}, None
    for (_i, r), nm in zip(keep.iterrows(), clean(keep[ocol])):
        v = tuple(int(pd.to_numeric(r[y])) for y in ycols)
        if nm == "Others":
            others = v
        elif nm not in DESA_ISO:
            raise SystemExit(f"DESA origin {nm!r} for {dest} has no ISO code in DESA_ISO")
        else:
            out[DESA_ISO[nm]] = v
    tot = tuple(sum(v[j] for v in out.values()) + (others or (0, 0, 0))[j] for j in range(3))
    if any(abs(tot[j] - world[j]) > 3 for j in range(3)):
        raise SystemExit(f"DESA {dest}: origins sum to {tot}, World {world}")
    for k, (b, mm, f) in out.items():
        if abs(mm + f - b) > 2:
            raise SystemExit(f"DESA {dest} {k}: men + women != both")
    print(f"  UN DESA {year}, {dest}: {world[0]:,} migrants, {len(out)} named origins, `Others` "
          f"{others[0] if others else 0:,}")
    return out, others or (0, 0, 0), world


def india(dest):
    """India's mix for Indians in `dest`: Keralites at KMS 2023's share for the destination,
    the rest by MEA emigration clearances 2011-17 by state (sa_census.ECR_2011_17), each state
    at its 2011 census mother-tongue mix. `kerala_share` is of the destination's Indians."""
    key = ("india", dest)
    if key not in _cache:
        d, _ = sa.india_mix()
        _cache[key] = d
    return _cache[key]


def india_mix(dest, n_indians):
    d = india(dest)
    kerala_n = sa.KMS_EMIGRANTS * KMS_DEST[dest]
    k_share = kerala_n / n_indians
    if not 0 < k_share < 0.6:
        raise SystemExit(f"{dest}: Keralites would be {k_share:.1%} of Indians")
    rest = sum(sa.ECR_2011_17.values())
    parts = [(k_share, sa.mix_from(d[d["geo_name"] == "KERALA"]))]
    parts += [((1 - k_share) * v / rest, sa.mix_from(d[d["geo_name"] == s]))
              for s, v in sa.ECR_2011_17.items()]
    print(f"  India in {dest}: Keralites {kerala_n:,.0f} (KMS 2023, {KMS_DEST[dest]:.1%} of Kerala's "
          f"emigrants), {k_share:.1%} of {n_indians:,.0f} Indians; the rest by ECR clearances 2011-17")
    return sa.combine(parts)


def pakistan():
    if "pk" not in _cache:
        _cache["pk"] = sa.pakistan()
    return _cache["pk"]


def home_mix(cc):
    if cc not in _cache:
        _cache[cc] = sa.home_mix(cc)
    return _cache[cc]


def origin_mix(iso, dest, n):
    """One nationality's language mix, {node: share}."""
    import fr2023
    from fr_build import COUNTRY_LANG
    if iso == "IN":
        m = india_mix(dest, n)
    elif iso == "PK":
        m = pakistan()
    elif iso in OVERRIDE:
        m = {OVERRIDE[iso]: 1.0}
    elif _om.gulf_route(iso, dest) is not None:   # overrides, immigration countries (origin_mix)
        m = _om.gulf_route(iso, dest)
    elif iso in DRAWN and n >= HOME_MIN:
        m = home_mix(iso)
    elif iso in DRAWN and COUNTRY_LANG.get(iso) == "Arabic":
        # a small Arab origin on this map takes its drawn mix's largest language: COUNTRY_LANG's
        # plain "Arabic" would put Sudanese and Chadians on a node no Gulf citizen is drawn on,
        # and a whole mix would spread a few hundred people over twenty languages
        hm = home_mix(iso)
        m = {max(hm, key=hm.get): 1.0}
    else:
        v = COUNTRY_LANG[{"GB": "UK", "GR": "EL"}.get(iso, iso)]
        items = [(v, 1.0)] if isinstance(v, str) else list(v.items())
        m = {fr2023.NAMES[lab]: s for lab, s in items}
    if abs(sum(m.values()) - 1) > 1e-9:
        raise SystemExit(f"{iso}: mix sums to {sum(m.values())}")
    return m


def blend(weights, dest, fixed=None):
    """{iso or node-key: people} -> {node: share}. Keys starting `node:` are drawn on that node
    as they are (an unnamed `Others` row, say). `fixed` = {iso: mix} computed once by the caller
    (India's, whose Keralite share needs the country's whole Indian count)."""
    tot = sum(weights.values())
    out = {}
    for k, w in weights.items():
        if w <= 0:
            continue
        if fixed and k in fixed:
            m = fixed[k]
        else:
            m = {k[5:]: 1.0} if k.startswith("node:") else origin_mix(k, dest, w)
        for n, s in m.items():
            out[n] = out.get(n, 0.0) + w / tot * s
    return out


def spread(units, mixes, round_within_rows=sa.round_within_rows):
    """units: DataFrame index geo_id with one column per part (people); mixes: {part: {node: share}}
    -> integer DataFrame geo_id x node, each row summing to its rounded part total."""
    nodes = sorted({n for m in mixes.values() for n in m})
    fm = pd.DataFrame({gid: {n: sum(r[p] * mixes[p].get(n, 0.0) for p in mixes) for n in nodes}
                       for gid, r in units.iterrows()}).T[nodes]
    fc = round_within_rows(fm)
    for gid, r in units.iterrows():
        want = int(round(sum(r[p] for p in mixes)))
        if int(fc.loc[gid].sum()) != want:
            raise SystemExit(f"{gid}: rounded rows sum to {int(fc.loc[gid].sum()):,}, want {want:,}")
    return fc


def report(out, total, label):
    tot = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\n{label}: {len(out)} rows, {out['geo_id'].nunique()} units, {total:,} people, "
          f"{out['source_category'].nunique()} nodes")
    for n, c in tot.head(20).items():
        print(f"    {n:<50} {c:>11,}  {c / total:6.2%}")
