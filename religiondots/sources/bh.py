"""Bahrain: the 2020 census counts religion (Muslim, Others) by nationality and sex for the whole
country, and the population by governorate, nationality group and sex. This joins the two.

Reads, all fetched into data/raw/bh/ by --fetch, from the Information & eGovernment Authority's open
data portal (`www.data.gov.bh`, Opendatasoft, `api/explore/v2.1/catalog/datasets/<id>/records`):

  * `population-by-religion-nationality-and-sex-census-2020`: Muslim and Others for Bahraini and
    non-Bahraini, by sex; 1,501,635 people, the national figure UNSD's table 28 also prints;
  * `population-by-governorate-nationality-and-sex-census-2020`: the four governorates (Capital,
    Muharraq, Northern, Southern) by Bahraini / non-Bahraini and sex;
  * `population-by-governorate-nationality-groups-and-sex-census-2020`: the same four by eight
    nationality groups (Bahraini, Gulf Co-operative Countries, Other Arabs, Asian, African,
    European, North American, Others) and sex;
  * UN DESA, *International Migrant Stock 2024* (data/raw/mr/, shared), Bahrain's migrants by
    origin and sex, used only as the mix of origins INSIDE each census nationality group;
  * Pew Research Center, *Religious Composition 2010-2020* (data/raw/estimates/pew.zip): each
    origin's 2020 composition, and Bahrain's own 2020 row for its Christian / Hindu ratio only.

Writes data/normalized/bh.csv: one row per governorate and category. `sources/bh.md` is the record
in prose.

## WHAT WAS COUNTED, AND WHAT IS CARRIED

The census counted every resident's religion, but the portal prints it only for the country, by
nationality (Bahraini or not) and sex, and only as Muslim or `Others` although the form codes
Christian and Jewish too (sources.md §11ao). Nothing found crosses religion with a governorate.

  * **Bahrainis**: each governorate's Bahraini men and women at the national Bahraini shares by
    sex (99.74% and 99.61% Muslim). The 2,295 Bahraini `Others` stay on `other.bh`, unsplit.
    Tier `derived`.
  * **Bahraini Muslims, Shia against Sunni** (Anita, ask 055, 2026-10-03; sources.md
    §bh-2026-10-03c): nobody counted sect. Each governorate starts at its Shia share of mosques
    (the Ja'fari Endowments' 2016 count against the Sunni Endowments' of about 2022), and one
    logit shift over all four moves those shares until the country's Bahraini Muslims are Shia
    and Sunni as Arab Barometer wave I (2009) found them: 249 Shia, 183 Sunni, 3 "Muslim" of 435.
    The 3/435 stay on bare `islam`. A mosque register ranks places and does not count people, so
    the registers set only the spread and the survey the level. Witnesses: OSM's Shia-tagged
    mosques per governorate against the Ja'fari register, and the 2017 Washington Institute poll's
    62% Shia. Tier `modelled`. Foreign Muslims are not split (their sect is not measured).
  * **Non-Bahrainis, Muslim against Others**: each nationality group and sex gets a non-Muslim
    share from its DESA origins through Pew (Gulf nationals on Islam), and the Asian and African
    groups' shares are moved together by one logit shift per sex until the country's non-Bahraini
    `Others` equal the census count for that sex exactly. So the national totals are the census's
    and only their spread over governorates rests on the group mix. Tier `derived`.
  * **Non-Bahraini `Others`, split**: each group and sex's non-Muslims take the non-Muslim mix of
    its DESA origins through Pew, then the Gulf rule the UAE and Oman use
    (`origin_religion.gulf_christian_hindu`, sources.md §gulf-2026-10-03 and §bh-2026-10-03b):
    Christians / (Christians + Hindus) of the whole layer raked to Pew 2020's Bahrain row (0.550)
    by moving Indians, inside the Asian rows, from Hindu to Christian. Every other family
    (Buddhists, unaffiliated, Jews, `Other religions`) stays as the origins give it; until
    2026-10-03 every family was raked to Pew's row, which is Pew's shared Gulf template. Then
    `Other religions` resolved per origin by `taxonomy/origin_religion.py`. Christians and
    Muslims are not split into churches or branches. Tier `modelled`.

## CHECKS

  * the religion table: Bahraini and non-Bahraini by sex sum to 1,501,635 (UNSD's 2020 row) and
    Muslims to 1,111,533;
  * the governorate table and the nationality-group table agree cell for cell (Bahraini, and the
    seven foreign groups summed), and both close on the religion table's nationality-by-sex totals;
  * DESA: Bahrain's named origins and `Others` sum to the world stock, by sex, and the named
    origins are the pinned set, each assigned to one census group;
  * the logit shift solves exactly; the Gulf rule moves no more Indians than are Hindu, keeps every
    row's total, and is over `origin_religion.GULF_MATERIAL` of the country;
  * **witness**: the drawn Christian share of all non-Muslims against the 2001 census, the last
    one that printed Christians (58,315 of 122,211, UNSD), within `WITNESS_BAND`;
  * the sect split: both registers sum to their printed totals; the survey file returns 249 / 183
    / 3; the shift closes; OSM's Shia-tagged mosques fall in every governorate at `OSM_BAND` of
    the Ja'fari register's count; the drawn Shia share is within `WINEP_BAND` of the 2017 poll.

Usage:
    python sources/bh.py --fetch    the three portal tables and OSM's mosques (DESA, Pew and the
                                    Arab Barometer are shared)
    python sources/bh.py            rebuild data/normalized/bh.csv
"""

import io
import json
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

import numpy as np
import pandas as pd

from afrobarometer import round_within_rows
import origin_religion as origin

RAW = os.path.join(ROOT, "data", "raw", "bh")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
DESA = os.path.join(ROOT, "data", "raw", "mr",
                    "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
OUT = os.path.join(ROOT, "data", "normalized", "bh.csv")

PORTAL = "https://www.data.gov.bh/api/explore/v2.1/catalog/datasets/{}/records?limit=100"
TABLES = {
    "religion": "population-by-religion-nationality-and-sex-census-2020",
    "governorate": "population-by-governorate-nationality-and-sex-census-2020",
    "groups": "population-by-governorate-nationality-groups-and-sex-census-2020",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

GOVS = ["Capital", "Muharraq", "Northern", "Southern"]
SEXES = ["Male", "Female"]
NATS = ["Bahraini", "Non-Bahraini"]
GROUPS = ["Gulf Co-operative Countries", "Other Arabs", "Asian", "African", "European",
          "North American", "Others"]
CENSUS_TOTAL = 1_501_635
CENSUS_MUSLIM = 1_111_533               # UNSD table 28, 2020, and the portal's four Muslim cells

# DESA 2024's named origins for Bahrain, each to the census nationality group it falls in. Somalia
# and Sudan are Arab League members, so `Other Arabs`; Türkiye `Asian`. The census `Others` group
# (Oceania, Latin America and anyone unlisted; 1,787 people) has no DESA origin and takes the
# European and North American origins pooled.
DESA_YEAR = 2024.0
DESA_WORLD = (840_202, 616_321, 223_881)
DESA_GROUP = {
    "Kuwait": ("KW", "Gulf Co-operative Countries"), "Qatar": ("QA", "Gulf Co-operative Countries"),
    "Saudi Arabia": ("SA", "Gulf Co-operative Countries"),
    "United Arab Emirates": ("AE", "Gulf Co-operative Countries"),
    "Egypt": ("EG", "Other Arabs"), "Morocco": ("MA", "Other Arabs"), "Sudan": ("SD", "Other Arabs"),
    "Tunisia": ("TN", "Other Arabs"), "Somalia": ("SO", "Other Arabs"), "Jordan": ("JO", "Other Arabs"),
    "Lebanon": ("LB", "Other Arabs"), "State of Palestine": ("PS", "Other Arabs"),
    "Syrian Arab Republic": ("SY", "Other Arabs"), "Yemen": ("YE", "Other Arabs"),
    "Afghanistan": ("AF", "Asian"), "Bangladesh": ("BD", "Asian"), "India": ("IN", "Asian"),
    "Nepal": ("NP", "Asian"), "Pakistan": ("PK", "Asian"), "Sri Lanka": ("LK", "Asian"),
    "Indonesia": ("ID", "Asian"), "Philippines": ("PH", "Asian"), "Thailand": ("TH", "Asian"),
    "Türkiye": ("TR", "Asian"),
    "Eritrea": ("ER", "African"), "Ethiopia": ("ET", "African"), "South Sudan": ("SS", "African"),
    "Chad": ("TD", "African"), "Nigeria": ("NG", "African"),
    "United Kingdom": ("GB", "European"), "France": ("FR", "European"),
    "Netherlands": ("NL", "European"),
    "United States of America": ("US", "North American"),
}
DESA_UNNAMED = "Others"
POOLED_FOR_OTHERS = ("European", "North American")
GULF_ON_ISLAM = {"KW", "QA", "SA", "AE"}
# groups whose non-Muslim share the calibration moves: the worker streams, where DESA's origin mix
# is least likely to be right (its African women are 2,027 against the census's 15,826)
SHIFTED = ("Asian", "African")

OTHER_NODE = "other.bh"
FAMILIES_NM = ["Christians", "Religiously_unaffiliated", "Buddhists", "Hindus", "Jews",
               "Other_religions"]
FAMILY_NODE = {"Christians": "christianity", "Religiously_unaffiliated": "unaffiliated",
               "Buddhists": "buddhism", "Hindus": "hinduism", "Jews": "judaism"}
# the Gulf rule's one moved origin (origin_religion.gulf_christian_hindu)
INDIA, INDIA_GROUP = "IN", "Asian"

# 2001 census (UNSD table 28): Christian 58,315, other 63,896; the last count that printed Christians
WITNESS_2001 = 58_315 / (58_315 + 63_896)
WITNESS_BAND = 0.10

# ---- Bahraini Muslims by sect (ask 055; sources.md §scout-2026-10-03-sect-registers, §bh-2026-10-03c)
# Ja'fari (Shia) Endowments Directorate, official statistic reported by Bahrain Mirror,
# 2 August 2016 (bahrainmirror.com/news/32904.html): mosques 753 and ma'tams 619 by governorate.
JAFARI_MOSQUES = {"Capital": 332, "Muharraq": 44, "Northern": 344, "Southern": 33}
JAFARI_MATAMS = {"Capital": 306, "Muharraq": 71, "Northern": 211, "Southern": 31}
# Sunni Endowments, the Ministry's answer to MP Mohammed Buhmoud, about 2022 (Al-Watan,
# alwatannews.net/bahrain/article/1000143): 229 jami' and 282 masjid, of which 250 masjid are placed
SUNNI_JAMI = {"Capital": 48, "Muharraq": 72, "Northern": 34, "Southern": 75}
SUNNI_MASJID = {"Capital": 44, "Muharraq": 96, "Northern": 55, "Southern": 55}
# Arab Barometer wave I, Bahrain, January-May 2009 (Justin Gengler; Bahrain Center for Studies and
# Research), q711, unweighted, citizens 18+: read from the file and asserted
AB1 = os.path.join(ROOT, "data", "raw", "arabbarometer", "ABI_English.sav")
AB1_BH = {"shiite muslim (lebanon & bahrain)": 249, "sunni muslim (lebanon & bahrain)": 183,
          "muslim": 3}
# The Washington Institute's 2017 poll of 1,000 Bahraini citizens (Pollock, "Sunnis and Shia in
# Bahrain: New Survey Shows Both Conflict and Consensus"): 62% Shia, 38% Sunni. Witness only.
WINEP_2017_SHIA, WINEP_BAND = 0.62, 0.10
# OSM's Muslim places of worship, Shia-tagged, per governorate, against the Ja'fari register
OSM_MOSQUES = os.path.join(RAW, "osm_mosques.json")
OVERPASS = "https://overpass-api.de/api/interpreter"
OSM_UA = "religiondots-map-build/1.0"
OSM_Q = ('[out:json][timeout:180];area["ISO3166-1"="BH"][admin_level=2]->.a;'
         'nwr(area.a)[amenity=place_of_worship][religion=muslim];out center tags;')
OSM_BAND = (0.6, 1.25)
OSM_SNAP_M = 2000
SECT_CATS = {"shia": "Bahraini, Muslim, Shia", "sunni": "Bahraini, Muslim, Sunni",
             "none": "Bahraini, Muslim, no sect given"}

# note_public's figures, measured 2026-10-03 under the Gulf rule (sources.md §bh-2026-10-03b) and
# the sect split (§bh-2026-10-03c), and asserted
NOTE = {'Bahraini, Muslim, Shia': 406452, 'Bahraini, Muslim, Sunni': 298717,
        'Bahraini, Muslim, no sect given': 4898, 'Non-Bahraini, Muslim': 401464, 'christianity': 193586,
        'hinduism': 158651, 'buddhism': 12404, 'unaffiliated': 10952, 'sikhism': 6750,
        'other.bh': 4322, 'Bahraini, Others': 2295, 'jainism': 555, 'judaism': 357,
        'indigenous.african': 104, 'druze': 71, 'indigenous': 57}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for key, ds in TABLES.items():
        dst = os.path.join(RAW, f"census2020_{key}.json")
        if os.path.exists(dst) and os.path.getsize(dst) > 500:
            continue
        url = PORTAL.format(ds)
        print("  GET", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            data = r.read()
        d = json.loads(data)
        if "results" not in d or d.get("total_count") != len(d["results"]):
            raise SystemExit(f"{ds}: not a complete records answer ({len(data):,} bytes)")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)


def fetch_osm():
    """OSM's Muslim places of worship in Bahrain, tags and a centre point (a witness only)."""
    import urllib.parse
    os.makedirs(RAW, exist_ok=True)
    data = urllib.parse.urlencode({"data": OSM_Q}).encode()
    req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": OSM_UA})
    print("  POST", OVERPASS, "(Muslim places of worship in BH)")
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    if not body.lstrip().startswith(b"{") or b'"elements"' not in body:
        raise SystemExit(f"Overpass did not return JSON: {body[:300]!r}")
    with open(OSM_MOSQUES + ".part", "wb") as fh:
        fh.write(body)
    os.replace(OSM_MOSQUES + ".part", OSM_MOSQUES)


def survey_level():
    """Arab Barometer wave I's Bahraini answers to q711, asserted against AB1_BH."""
    import pyreadstat
    df, _meta = pyreadstat.read_sav(AB1, apply_value_formats=True, usecols=["country", "q711"])
    df = df.astype(object)
    got = df.loc[df["country"] == "bahrain", "q711"].value_counts().to_dict()
    if got != AB1_BH:
        raise SystemExit(f"Arab Barometer I, Bahrain, q711: {got}, pinned {AB1_BH}")
    n = sum(got.values())
    shia = got["shiite muslim (lebanon & bahrain)"] / n
    sunni = got["sunni muslim (lebanon & bahrain)"] / n
    print(f"\n  Arab Barometer I (2009), Bahraini citizens: {n} answers, Shia {shia:.1%}, Sunni "
          f"{sunni:.1%}, 'Muslim' {1 - shia - sunni:.1%} (unweighted; the file has no weight)")
    return shia, sunni


def register_shares(matams=False):
    """Each governorate's Shia share of mosques (ma'tams added to the Shia side if `matams`)."""
    if sum(JAFARI_MOSQUES.values()) != 753 or sum(JAFARI_MATAMS.values()) != 619:
        raise SystemExit("the Ja'fari register no longer sums to 753 mosques, 619 ma'tams")
    if sum(SUNNI_JAMI.values()) != 229 or sum(SUNNI_MASJID.values()) != 250:
        raise SystemExit("the Sunni register no longer sums to 229 jami' and 250 placed masjid")
    out = {}
    for g in GOVS:
        j = JAFARI_MOSQUES[g] + (JAFARI_MATAMS[g] if matams else 0)
        out[g] = j / (j + SUNNI_JAMI[g] + SUNNI_MASJID[g])
    return out


def osm_witness():
    """OSM's Shia-tagged mosques per governorate against the Ja'fari register's count."""
    import geopandas as gpd
    if not os.path.exists(OSM_MOSQUES):
        fetch_osm()
    with open(OSM_MOSQUES, encoding="utf-8") as fh:
        els = json.load(fh)["elements"]
    rows = []
    for e in els:
        lat, lon = (e["lat"], e["lon"]) if "lat" in e else (e["center"]["lat"], e["center"]["lon"])
        den = (e.get("tags", {}).get("denomination") or "").strip().lower()
        rows.append({"den": den, "geometry": gpd.points_from_xy([lon], [lat])[0]})
    pts = gpd.GeoDataFrame(rows, crs=4326)
    units = gpd.read_file(os.path.join(ROOT, "data", "geo", "bh", "bh_units.gpkg"))[["unit", "geometry"]]
    pts_m, units_m = pts.to_crs(32639), units.to_crs(32639)
    j = gpd.sjoin_nearest(pts_m, units_m, how="left", max_distance=OSM_SNAP_M, distance_col="d")
    j = j[~j.index.duplicated()]
    lost = int(j["unit"].isna().sum())
    shia = j[j["den"].isin(["shia", "shi'a", "shiite", "jafari", "ja'fari", "twelver"])]
    per = shia.groupby("unit").size().reindex(GOVS, fill_value=0)
    print(f"\n  OSM witness: {len(pts)} Muslim places of worship, {len(shia)} tagged Shia, "
          f"{int((j['den'] == 'sunni').sum())} Sunni, {int((j['den'] == '').sum())} untagged; "
          f"{lost} more than {OSM_SNAP_M} m from a governorate")
    bad = []
    for g in GOVS:
        r = per[g] / JAFARI_MOSQUES[g]
        print(f"      {g:<10} OSM Shia {per[g]:>4}   Ja'fari register {JAFARI_MOSQUES[g]:>4}   ratio {r:5.2f}")
        if not OSM_BAND[0] <= r <= OSM_BAND[1]:
            bad.append(g)
    if bad:
        raise SystemExit(f"OSM's Shia mosques do not follow the Ja'fari register in {bad} "
                         f"(band {OSM_BAND})")


def sect_split(bm):
    """Bahraini Muslims per governorate (`bm`, a Series) -> DataFrame of SECT_CATS columns, floats.

    Named answers share 432/435; the governorates' register shares are moved by one logit shift so
    the Shia total is the survey's 249/435 of all Bahraini Muslims."""
    shia, sunni = survey_level()
    named = shia + sunni
    prior = register_shares()
    n = {g: float(bm[g]) * named for g in GOVS}
    target = shia * float(bm.sum())
    d, q = logit_shift(n, prior, target, GOVS)
    alt_prior = register_shares(matams=True)
    _d2, alt = logit_shift(n, alt_prior, target, GOVS)
    print(f"  sect: register Shia shares moved by one logit shift {d:+.3f} to the survey's level")
    print(f"      {'':<10} {'mosques':>8} {'drawn':>8}   (mosques + ma'tams, same level: "
          "prior -> drawn)")
    for g in GOVS:
        print(f"      {g:<10} {prior[g]:8.1%} {q[g]:8.1%}   {alt_prior[g]:6.1%} -> {alt[g]:6.1%}")
    out = pd.DataFrame({SECT_CATS["shia"]: {g: n[g] * q[g] for g in GOVS},
                        SECT_CATS["sunni"]: {g: n[g] * (1 - q[g]) for g in GOVS},
                        SECT_CATS["none"]: {g: float(bm[g]) * (1 - named) for g in GOVS}})
    drawn = out[SECT_CATS["shia"]].sum() / (out[SECT_CATS["shia"]].sum() + out[SECT_CATS["sunni"]].sum())
    print(f"  witness: Shia {drawn:.1%} of Bahraini Muslims naming a sect; the Washington "
          f"Institute's 2017 poll {WINEP_2017_SHIA:.0%} (band {WINEP_BAND:.0%} points)")
    if abs(drawn - WINEP_2017_SHIA) > WINEP_BAND:
        raise SystemExit("the drawn Shia share is outside the 2017 poll's band")
    osm_witness()
    return out


def records(key):
    with open(os.path.join(RAW, f"census2020_{key}.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    return pd.DataFrame(d["results"])


def cell(df, col, val):
    if set(df[col]) - set(val):
        raise SystemExit(f"unexpected {col} labels: {sorted(set(df[col]) - set(val))}")
    if set(val) - set(df[col]):
        raise SystemExit(f"missing {col} labels: {sorted(set(val) - set(df[col]))}")


def load_census():
    rel = records("religion")
    cell(rel, "religion", ["Muslim", "Others"])
    cell(rel, "nationality", NATS)
    cell(rel, "sex", SEXES)
    if len(rel) != 8 or rel.duplicated(["religion", "nationality", "sex"]).any():
        raise SystemExit("the religion table is not 8 distinct cells")
    rel = rel.set_index(["nationality", "sex", "religion"])["population"].astype(int).sort_index()
    if rel.sum() != CENSUS_TOTAL or rel.xs("Muslim", level="religion").sum() != CENSUS_MUSLIM:
        raise SystemExit(f"religion table: {rel.sum():,} people, "
                         f"{rel.xs('Muslim', level='religion').sum():,} Muslims")

    gov = records("governorate")
    cell(gov, "governorate", GOVS)
    cell(gov, "nationality", NATS)
    cell(gov, "sex", SEXES)
    if len(gov) != 16 or gov.duplicated(["governorate", "nationality", "sex"]).any():
        raise SystemExit("the governorate table is not 16 distinct cells")
    gov = gov.set_index(["governorate", "nationality", "sex"])["population"].astype(int).sort_index()

    grp = records("groups")
    cell(grp, "governorate", GOVS)
    cell(grp, "nationality_groups", ["Bahraini"] + GROUPS)
    cell(grp, "sex", SEXES)
    if len(grp) != 64 or grp.duplicated(["governorate", "nationality_groups", "sex"]).any():
        raise SystemExit("the nationality-group table is not 64 distinct cells")
    grp = grp.set_index(["governorate", "nationality_groups", "sex"])["population"].astype(int).sort_index()

    for g in GOVS:
        for s in SEXES:
            if grp[(g, "Bahraini", s)] != gov[(g, "Bahraini", s)]:
                raise SystemExit(f"{g}/{s}: Bahrainis differ between the two governorate tables")
            if sum(grp[(g, x, s)] for x in GROUPS) != gov[(g, "Non-Bahraini", s)]:
                raise SystemExit(f"{g}/{s}: the foreign groups do not sum to the non-Bahrainis")
    for n in NATS:
        for s in SEXES:
            if sum(gov[(g, n, s)] for g in GOVS) != rel[(n, s)].sum():
                raise SystemExit(f"{n}/{s}: governorates {sum(gov[(g, n, s)] for g in GOVS):,} "
                                 f"against the religion table's {rel[(n, s)].sum():,}")
    print(f"  census 2020: {CENSUS_TOTAL:,} people; the governorate and nationality-group tables "
          f"agree cell for cell and close on the religion table's nationality by sex")
    for n in NATS:
        for s in SEXES:
            print(f"      {n:<13} {s:<7} {rel[(n, s)].sum():>8,}   Muslim "
                  f"{rel[(n, s, 'Muslim')] / rel[(n, s)].sum():7.2%}   Others {rel[(n, s, 'Others')]:>8,}")
    return rel, gov, grp


def desa_mixes():
    """{sex: {iso: stock}} for Bahrain, 2024, named origins only."""
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
    cols = list(df.iloc[hdr])
    names = [str(x).strip() for x in cols]
    dcol = next(i for i, x in enumerate(names) if "of destination" in x)
    ocol = next(i for i, x in enumerate(names) if "of origin" in x)
    ccol = next(i for i, x in enumerate(names) if x == "Location code of origin")
    ycols = [i for i, x in enumerate(cols) if str(x) in (str(DESA_YEAR), str(int(DESA_YEAR)))]
    if len(ycols) != 3:
        raise SystemExit(f"expected three {DESA_YEAR:g} columns (both sexes, men, women), got {ycols}")
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "Bahrain"]
    name = m[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    world = tuple(int(pd.to_numeric(m.loc[name == "World", y]).iloc[0]) for y in ycols)
    if world != DESA_WORLD:
        raise SystemExit(f"DESA 2024 world stock for Bahrain is {world}, pinned {DESA_WORLD}")
    code = pd.to_numeric(m[ccol], errors="coerce")
    ctry = m[(code < 900) | (name == DESA_UNNAMED)]
    cname = ctry[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    if set(cname) != set(DESA_GROUP) | {DESA_UNNAMED}:
        raise SystemExit(f"DESA's origins for Bahrain changed: "
                         f"{sorted(set(cname) ^ (set(DESA_GROUP) | {DESA_UNNAMED}))}")
    out = {}
    for j, key in enumerate(["Both", "Male", "Female"]):
        stock = dict(zip(cname, pd.to_numeric(ctry[ycols[j]]).astype(int)))
        if abs(sum(stock.values()) - world[j]) > 2:
            raise SystemExit(f"DESA's origins sum to {sum(stock.values()):,}, world {world[j]:,}")
        out[key] = {DESA_GROUP[k][0]: v for k, v in stock.items() if k != DESA_UNNAMED}
    for k in out["Both"]:
        if abs(out["Male"][k] + out["Female"][k] - out["Both"][k]) > 2:
            raise SystemExit(f"DESA {k}: the sexes do not sum")
    named = sum(out["Both"].values())
    print(f"  UN DESA 2024: {world[0]:,} migrants in Bahrain ({world[1]:,} men, {world[2]:,} women); "
          f"33 named origins {named:,}, used only as the mix inside each census group")
    return out


def pew_table():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    return t[t["Year"] == 2020].set_index("Country")


def origin_families(pew, iso):
    """Pew 2020's seven families for one origin, as shares; Gulf nationals all Muslim."""
    if iso in GULF_ON_ISLAM:
        return {f: (1.0 if f == "Muslims" else 0.0) for f in origin.FAMILIES}
    pn = origin.PEW_BY_ISO[iso]
    if pn is None or pn not in pew.index:
        raise SystemExit(f"Pew has no row for {iso}")
    row = {f: float(pew.loc[pn, f]) for f in origin.FAMILIES}
    tot = sum(row.values())
    return {f: v / tot for f, v in row.items()}


def other_split(iso):
    """Pew's `Other_religions` for one origin, resolved to nodes (origin_religion.OTHER)."""
    split = origin.OTHER.get(iso, {None: 1.0})
    s = sum(split.values())
    return {(OTHER_NODE if n is None else n): v / s for n, v in split.items()}


def group_priors(desa, pew):
    """{(group, sex): (families share dict, Other_religions node split)} from DESA x Pew."""
    out = {}
    for s in SEXES:
        stock = desa[s]
        members = {g: [iso for _k, (iso, gg) in DESA_GROUP.items() if gg == g] for g in GROUPS}
        members["Others"] = [iso for g in POOLED_FOR_OTHERS for iso in members[g]]
        for g in GROUPS:
            w = {iso: stock[iso] for iso in members[g]}
            tot = sum(w.values())
            if tot <= 0:
                raise SystemExit(f"DESA has no {s} migrants in group {g}")
            fam = dict.fromkeys(origin.FAMILIES, 0.0)
            oth = {}
            for iso, n in w.items():
                f = origin_families(pew, iso)
                for k, v in f.items():
                    fam[k] += n / tot * v
                for node, v in other_split(iso).items():
                    oth[node] = oth.get(node, 0.0) + n / tot * f["Other_religions"] * v
            ot = sum(oth.values())
            out[(g, s)] = (fam, {k: v / ot for k, v in oth.items()} if ot > 0 else {OTHER_NODE: 1.0})
    return out


def logit_shift(n, p, target, shifted):
    """Shift logit(p) by one delta over `shifted` keys so sum(n * p) == target."""
    fixed = sum(n[k] * p[k] for k in n if k not in shifted)
    keys = [k for k in shifted if 0 < p[k] < 1]
    lp = {k: np.log(p[k] / (1 - p[k])) for k in keys}

    def total(d):
        return fixed + sum(n[k] / (1 + np.exp(-(lp[k] + d))) for k in keys)

    lo, hi = -20.0, 20.0
    if not total(lo) <= target <= total(hi):
        raise SystemExit(f"no logit shift reaches {target:,} (range {total(lo):,.0f}-{total(hi):,.0f})")
    for _ in range(200):
        mid = (lo + hi) / 2
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    q = dict(p)
    for k in keys:
        q[k] = 1 / (1 + np.exp(-(lp[k] + d)))
    if abs(sum(n[k] * q[k] for k in n) - target) > 1e-6:
        raise SystemExit("the logit shift did not close")
    return d, q


def gulf_rule(prior, nm_rows, priors, desa, pew):
    """The Gulf rule, as the UAE and Oman (origin_religion.gulf_christian_hindu, sources.md
    §gulf-2026-10-03): Christians / (Christians + Hindus) of the foreign non-Muslims raked to Pew
    2020's Bahrain row, by moving Indians from Hindu to Christian. Every other family, and each
    group and sex's non-Muslim total, stay as the origins give them.

    `prior` is non-Muslims by (group, sex) x family. Indians are inside the Asian group's rows; their
    part of each is India's weight in that row's DESA mix times India's non-Muslim share, over the
    row's. The layer is handed to the shared function as one key for Indians (both sexes, one
    composition) and one per row for everyone else."""
    fam_node = dict(FAMILY_NODE, Other_religions="other")
    india = origin_families(pew, INDIA)
    in_nm = sum(india[f] for f in FAMILIES_NM)
    in_comp = {fam_node[f]: india[f] / in_nm for f in FAMILIES_NM}
    in_rows = {}
    for s in SEXES:
        k = (INDIA_GROUP, s)
        members = [iso for _n, (iso, g) in DESA_GROUP.items() if g == INDIA_GROUP]
        w = desa[s][INDIA] / sum(desa[s][iso] for iso in members)
        in_rows[k] = nm_rows[k] * w * in_nm / (1.0 - priors[k][0]["Muslims"])
    people = {INDIA: sum(in_rows.values())}
    comps = {INDIA: dict(in_comp)}
    for k in prior.index:
        rest = prior.loc[k] - (pd.Series({f: in_comp[fam_node[f]] for f in FAMILIES_NM})
                               * in_rows.get(k, 0.0))
        if (rest < -1e-6).any():
            raise SystemExit(f"{k}: India's part is more than the row")
        n = float(rest.sum())
        if n > 0:
            people[k] = n
            comps[k] = {fam_node[f]: float(v) / n for f, v in rest.items()}
    bh = {f: float(pew.loc["Bahrain", f]) for f in ("Christians", "Hindus")}
    ratio = bh["Christians"] / (bh["Christians"] + bh["Hindus"])
    before = (prior["Christians"].sum() / (prior["Christians"].sum() + prior["Hindus"].sum()))
    try:
        moved = origin.gulf_christian_hindu(people, comps, INDIA, ratio)
    except ValueError as e:
        raise SystemExit(f"the Gulf rule: {e}")
    if moved < origin.GULF_MATERIAL * CENSUS_TOTAL:
        raise SystemExit(f"the Gulf rule moves {moved:,.0f}, under {origin.GULF_MATERIAL:.0%} of the "
                         "country; it should not be applied here")
    x = moved / people[INDIA]
    out = prior.copy()
    for k, n in in_rows.items():
        out.loc[k, "Hindus"] = out.loc[k, "Hindus"] - n * x
        out.loc[k, "Christians"] = out.loc[k, "Christians"] + n * x
    if (out < -1e-6).any().any() or (out.sum(axis=1) - nm_rows).abs().max() > 1e-6:
        raise SystemExit("the Gulf rule broke a row")
    c_in = sum(v for node, v in comps[INDIA].items() if node.startswith("christianity"))
    print(f"\n  Gulf rule: Christians / (Christians + Hindus) {before:.3f} -> Pew's Bahrain {ratio:.3f}; "
          f"{moved:,.0f} of {people[INDIA]:,.0f} Indian non-Muslims moved from Hindu to Christian "
          f"(now {comps[INDIA]['hinduism']:.1%} Hindu, {c_in:.1%} Christian of them)")
    return out


def main():
    need = [os.path.join(RAW, f"census2020_{k}.json") for k in TABLES]
    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in need):
        fetch()
    if "--fetch" in sys.argv or not os.path.exists(OSM_MOSQUES):
        fetch_osm()
    for p in (PEW, DESA, AB1):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (shared; see sources/mr.py and estimates.py)")
    rel, gov, grp = load_census()
    desa = desa_mixes()
    pew = pew_table()
    priors = group_priors(desa, pew)

    # ---- Muslim against Others, per group and sex ----
    n_gs = {(g, s): float(sum(grp[(v, g, s)] for v in GOVS)) for g in GROUPS for s in SEXES}
    p_nm = {}
    print("\n  non-Muslim share by group and sex: DESA x Pew prior -> as drawn")
    for s in SEXES:
        n = {g: n_gs[(g, s)] for g in GROUPS}
        p = {g: 1.0 - priors[(g, s)][0]["Muslims"] for g in GROUPS}
        target = float(rel[("Non-Bahraini", s, "Others")])
        prior_tot = sum(n[g] * p[g] for g in GROUPS)
        d, q = logit_shift(n, p, target, SHIFTED)
        print(f"    {s}: census Others {target:,.0f} of {sum(n.values()):,.0f} "
              f"({target / sum(n.values()):.1%}); the prior gives {prior_tot:,.0f} "
              f"({prior_tot / sum(n.values()):.1%}); logit shift {d:+.3f} on {', '.join(SHIFTED)}")
        for g in GROUPS:
            print(f"        {g:<28} {n[g]:>9,.0f}   {p[g]:6.1%} -> {q[g]:6.1%}")
            p_nm[(g, s)] = q[g]

    # ---- the non-Muslims' split: Pew per origin, then the Gulf rule on Indians ----
    nm_rows = pd.Series({(g, s): n_gs[(g, s)] * p_nm[(g, s)] for g in GROUPS for s in SEXES})
    prior = pd.DataFrame({k: {f: priors[k][0][f] for f in FAMILIES_NM} for k in nm_rows.index}).T
    prior = prior.div(prior.sum(axis=1), axis=0).mul(nm_rows, axis=0).fillna(0.0)
    nm_total = float(nm_rows.sum())
    split = gulf_rule(prior, nm_rows, priors, desa, pew)
    bh = {f: float(pew.loc["Bahrain", f]) for f in FAMILIES_NM}
    print("\n  non-Bahraini non-Muslims by family: DESA x Pew prior -> Gulf rule (Pew's Bahrain row "
          "for comparison)")
    for f in FAMILIES_NM:
        print(f"      {f:<26} {prior[f].sum():>10,.0f} ({prior[f].sum() / nm_total:6.1%}) -> "
              f"{split[f].sum():>10,.0f} ({split[f].sum() / nm_total:6.1%})   Pew "
              f"{bh[f] / sum(bh.values()):6.1%}")
    share = split.div(nm_rows.where(nm_rows > 0, 1.0), axis=0)

    # ---- per governorate, nationality and sex ----
    recs, idx = [], []
    for v in GOVS:
        for s in SEXES:
            nb = grp[(v, "Bahraini", s)]
            sh = rel[("Bahraini", s, "Muslim")] / rel[("Bahraini", s)].sum()
            recs.append({"Bahraini, Muslim": nb * sh, "Bahraini, Others": nb * (1 - sh)})
            idx.append((v, "Bahraini", s))
            c = {}
            for g in GROUPS:
                n = grp[(v, g, s)]
                c["Non-Bahraini, Muslim"] = c.get("Non-Bahraini, Muslim", 0.0) + n * (1 - p_nm[(g, s)])
                o = n * p_nm[(g, s)]
                for f in FAMILIES_NM:
                    x = o * share.loc[(g, s), f]
                    if f == "Other_religions":
                        for node, y in priors[(g, s)][1].items():
                            k = f"Non-Bahraini, Others, {node}"
                            c[k] = c.get(k, 0.0) + x * y
                    else:
                        k = f"Non-Bahraini, Others, {FAMILY_NODE[f]}"
                        c[k] = c.get(k, 0.0) + x
            recs.append(c)
            idx.append((v, "Non-Bahraini", s))
    m = pd.DataFrame(recs, index=pd.MultiIndex.from_tuples(idx))
    if m.drop(columns=[c for c in m if c.startswith("Bahraini")]).iloc[1::2].isna().any().any():
        raise SystemExit("a non-Bahraini cell came out NaN")
    m = m.fillna(0.0)
    cats = ["Bahraini, Muslim", "Bahraini, Others", "Non-Bahraini, Muslim"] + sorted(
        c for c in m.columns if c.startswith("Non-Bahraini, Others"))
    m = m[cats]
    counts = round_within_rows(m)
    want = pd.Series({k: gov[k] for k in m.index})
    if not (counts.sum(axis=1) == want).all():
        print(pd.DataFrame({"got": counts.sum(axis=1), "want": want, "float": m.sum(axis=1)}))
        raise SystemExit("rounded counts do not sum to the governorate table's cells")
    for n in NATS:
        for s in SEXES:
            got = counts.xs((n, s), level=(1, 2)).sum()
            mus = int(got[f"{n}, Muslim"])
            if abs(mus - rel[(n, s, "Muslim")]) > 4:
                raise SystemExit(f"{n}/{s}: {mus:,} Muslims drawn against {rel[(n, s, 'Muslim')]:,}")

    # ---- witness ----
    tot = counts.sum()
    nm_cols = [c for c in cats if "Others" in c]
    chr_ = int(tot["Non-Bahraini, Others, christianity"])
    nm = int(tot[nm_cols].sum())
    print(f"\n  witness: Christians {chr_:,} of {nm:,} non-Muslims ({chr_ / nm:.1%}); the 2001 census "
          f"printed {WITNESS_2001:.1%} (band {WITNESS_BAND:.0%} points)")
    if abs(chr_ / nm - WITNESS_2001) > WITNESS_BAND:
        raise SystemExit("the Christian share of non-Muslims is outside the 2001 witness band")

    by_gov = counts.groupby(level=0).sum()
    print("\n  per governorate, drawn:")
    for v in GOVS:
        r = by_gov.loc[v]
        t = r.sum()
        mus = r["Bahraini, Muslim"] + r["Non-Bahraini, Muslim"]
        print(f"      {v:<10} {t:>9,}  Muslim {mus / t:6.1%}  Christian "
              f"{r['Non-Bahraini, Others, christianity'] / t:6.1%}  Hindu "
              f"{r['Non-Bahraini, Others, hinduism'] / t:6.1%}")

    # ---- Bahraini Muslims by sect (ask 055) ----
    sect = round_within_rows(sect_split(by_gov["Bahraini, Muslim"]))
    if not (sect.sum(axis=1) == by_gov["Bahraini, Muslim"]).all():
        raise SystemExit("the sect split does not sum to each governorate's Bahraini Muslims")
    by_gov = pd.concat([by_gov.drop(columns=["Bahraini, Muslim"]), sect], axis=1)

    # ---- write ----
    long = by_gov.stack().rename("count").reset_index()
    long.columns = ["geo_id", "source_category", "count"]
    long = long[long["count"] > 0].copy()
    long["geo_level"] = "governorate"
    long["geo_name"] = long["geo_id"]
    long["tier"] = np.where(long["source_category"].str.startswith("Non-Bahraini, Others")
                            | long["source_category"].isin([SECT_CATS["shia"], SECT_CATS["sunni"]]),
                            "modelled", "derived")
    long["basis"] = "census"
    long["year"] = 2020
    long["source_id"] = np.where(
        long["source_category"].str.startswith("Bahraini, Muslim"),
        "census2020_religion_x_governorate_x_endowments_mosques_x_arabbarometer1",
        "census2020_religion_x_governorate_groups_x_undesa2024_x_pew2020")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    long[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "basis", "year",
          "source_id"]].to_csv(OUT, index=False, encoding="utf-8")
    fx = long.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {long['geo_id'].nunique()} governorates, {int(long['count'].sum()):,} people")
    print("    " + ", ".join(f"{k} {int(v):,}" for k, v in fx.items()))
    got = {k.split(", ")[-1] if k.startswith("Non-Bahraini, Others") else k: int(v)
           for k, v in fx.items()}
    print(f"\n  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
