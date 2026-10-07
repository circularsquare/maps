"""Maldives: the 2014 census asked foreign residents their religion and did not ask Maldivians.
Maldivians are drawn on Islam; foreign residents on the census's own answers, which are printed for
Malé and for the atolls together, spread over the 20 atolls by citizenship and sex.

Reads data/raw/mv/ (the census's Release I and II tables), the UNSD Demographic Yearbook table 28
cache (`tools/oracle.py`), data/raw/estimates/pew.zip and the UN DESA migrant stock workbook that
`sources/mr.py --fetch` downloads; writes data/normalized/mv.csv. `sources/mv.md` is the record in
prose.

## WHO WAS ASKED

The 2014 form (M6 nationality, M7 religion) sends Maldivians from M6 past M7, so only foreign
residents answered (`sources.md` §scout-2026-09-14-asia-oceania, where a scout read the form). The
constitution of 2008 says a non-Muslim may not become a citizen (Article 9(d)). Every Maldivian is
drawn on Islam, which nobody asked: the tabulation codes them all Islam (UNSD's 2014 row has 1,228
`Not Stated` among 402,071 people, and every one of them fits inside the foreign residents).

## THE FOREIGN RESIDENTS' ANSWERS, AND WHERE THEY ARE PRINTED

UNSD's table 28 carries the census's religion table for Malé (`Urban`, which is Malé: 153,904, the
same as Table PP3's Malé row) and for the atolls (`Rural`), by sex. Less the Maldivians in each
(PP3), every cell is the foreign residents' own answer:

    Malé    24,523 foreign residents, measured as printed
    atolls  39,114, measured for the atolls together and spread to each atoll here

## SPREADING THE ATOLLS' ANSWERS

The census prints foreign residents by citizenship (India, Sri Lanka, Bangladesh, others), sex and
three places (Malé; the administrative islands, where Maldivians live; the resort and industrial
islands) nationally (MG15), and by country of birth, sex and atoll for each of the two kinds of
island (MG14). So:

  1. **The religion of each citizenship.** An iterative fit of sex x place x citizenship x religion
     to two printed margins: MG15's sex x place x citizenship, and the census's sex x (Malé,
     atolls) x religion. The seed is Pew Research Center's 2020 composition of each citizenship's
     home country (`others` from UN DESA's 2015 named origins for the Maldives, through Pew), mapped
     to the census's six answers. The seed only decides how the printed totals are shared out
     between citizenships; the totals are the census's.
  2. **Each atoll's foreign residents by sex, kind of island and citizenship.** The administrative
     islands' foreign residents per atoll by sex are printed (PP3). The resort and industrial
     islands' are printed by place of usual residence (MG14), scaled to the census's count of the
     people found there (MG15). Inside each, MG14's countries of birth give the citizenship mix,
     raked to MG15's citizenship totals.
  3. Each atoll takes the sum over its sex x island x citizenship cells, then a rounding that keeps
     both each atoll's foreign residents and the atolls' printed religion totals exact.

## PEOPLE

Base: place of enumeration, 402,071 in 21 units, which is UNSD's universe and PP3's. Maldivians per
atoll from MG4 (by place of enumeration, administrative and other islands together); foreign
residents per atoll as above. 1,228 foreign residents who did not state a religion are not drawn.

Usage:
    python sources/mv.py --fetch    the census tables (about 0.2 MB) if missing
    python sources/mv.py            rebuild data/normalized/mv.csv
"""

import io
import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

import origin_religion as origin
import oracle

RAW = os.path.join(ROOT, "data", "raw", "mv")
BASE_URL = "https://statisticsmaldives.gov.mv/mbs/wp-content/uploads/2015/12/"
FILES = {   # file -> magic bytes. Release I (PP) and Release II: Migration (MG), 2014 census
    "PP2.xls": b"\xd0\xcf\x11\xe0",
    "PP3.xls": b"\xd0\xcf\x11\xe0",
    "PP5-Updated-20181014.xlsx": b"PK\x03\x04",
    "MG4.xlsx": b"PK\x03\x04",
    "MG14.xlsx": b"PK\x03\x04",
    "MG15.xlsx": b"PK\x03\x04",
}
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
DESA = os.path.join(ROOT, "data", "raw", "mr",
                    "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
OUT = os.path.join(ROOT, "data", "normalized", "mv.csv")

# The census's atoll letters (as printed in brackets after each atoll's name) -> COD-AB pcode and
# name. sources/mv_geo.py asserts each pcode's COD name and witnesses the pairing by island names.
ATOLLS = [("HA", "MV001", "Haa Alifu"), ("HDh", "MV002", "Haa Dhaalu"), ("Sh", "MV003", "Shaviyani"),
          ("N", "MV004", "Noonu"), ("R", "MV006", "Raa"), ("B", "MV007", "Baa"),
          ("Lh", "MV005", "Lhaviyani"), ("K", "MV009", "Kaafu"), ("AA", "MV008", "Alifu Alifu"),
          ("ADh", "MV020", "Alifu Dhaalu"), ("V", "MV010", "Vaavu"), ("M", "MV013", "Meemu"),
          ("F", "MV011", "Faafu"), ("Dh", "MV012", "Dhaalu"), ("Th", "MV014", "Thaa"),
          ("L", "MV015", "Laamu"), ("GA", "MV016", "Gaafu Alifu"), ("GDh", "MV017", "Gaafu Dhaalu"),
          ("Gn", "MV018", "Gnaviyani"), ("S", "MV019", "Seenu")]
MALE = ("MALE", "MV021", "Male")
CODES = [a[0] for a in ATOLLS]
PCODE = {a[0]: a[1] for a in ATOLLS + [MALE]}
NAME = {a[0]: a[2] for a in ATOLLS + [MALE]}

CATS = ["Islam", "Hindu", "Christian", "Buddhist", "Other", "Not Stated"]
NATIONALS = "Maldivian (not asked)"
CIT = ["India", "Sri Lanka", "Bangladesh", "Others"]
CIT_ISO = {"India": "IN", "Sri Lanka": "LK", "Bangladesh": "BD"}
SEXES = ["Male", "Female"]
PLACES = ["Male'", "admin", "nonadmin"]

# Pew's seven families -> the census's answers. The form offers no "no religion": people with none
# would tick "other" or leave it blank, so Pew's unaffiliated are seeded half on each.
FAM_TO_CAT = {"Muslims": {"Islam": 1}, "Hindus": {"Hindu": 1}, "Christians": {"Christian": 1},
              "Buddhists": {"Buddhist": 1}, "Other_religions": {"Other": 1}, "Jews": {"Other": 1},
              "Religiously_unaffiliated": {"Other": 0.5, "Not Stated": 0.5}}
SEED_FLOOR = 0.002      # every citizenship x answer cell starts at least this, so the fit can move it

# UNSD 2014, as printed by tools/oracle.py on 2026-10-03, asserted against the cache.
UNSD = {
    ("Total", "Both"): dict(Islam=377651, Hindu=10163, Christian=6403, Buddhist=5337, Other=1289, NS=1228),
    ("Urban", "Male"): dict(Islam=79317, Hindu=3025, Christian=1326, Buddhist=1329, Other=209, NS=232),
    ("Urban", "Female"): dict(Islam=65700, Hindu=884, Christian=1019, Buddhist=533, Other=199, NS=131),
    ("Rural", "Male"): dict(Islam=130302, Hindu=5145, Christian=2572, Buddhist=3095, Other=462, NS=735),
    ("Rural", "Female"): dict(Islam=102332, Hindu=1109, Christian=1486, Buddhist=380, Other=419, NS=130),
}

# note_public's figures, measured 2026-10-03 and asserted
NOTE = dict(foreign=63637, foreign_muslim=39217, hindu=10163, christian=6403, buddhist=5337,
            other=1289, not_stated=1228, male_non_muslim=8524)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0 Safari/537.36"}
    for name, magic in FILES.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 5_000:
            continue
        print("  GET", BASE_URL + name)
        with urllib.request.urlopen(urllib.request.Request(BASE_URL + name, headers=ua),
                                    timeout=120) as r:
            data = r.read()
        if not data.startswith(magic):
            raise SystemExit(f"{name} did not return the expected file ({data[:8]!r})")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)


def say(ok, msg, bad):
    print(("  ok  " if ok else "  !!  ") + msg)
    if not ok:
        bad.append(msg)


def num(v):
    """A count cell: an int, or None for a blank. Anything else stops."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    s = str(v).strip()
    if s in ("", "-"):
        return None
    if re.fullmatch(r"\d+", s):
        return int(s)
    raise SystemExit(f"not a count: {v!r}")


def sheet(name):
    df = pd.read_excel(os.path.join(RAW, name), header=None)
    rows = [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row]
            for row in df.itertuples(index=False)]
    return [r for r in rows if any(x is not None and str(x).strip() for x in r)]


def code_of(label):
    m = re.search(r"\(([A-Za-z]+)\)\s*$", label)
    return m.group(1) if m else None


def sex_blocks(rows, first_label):
    """{'Both': rows, 'Male': rows, 'Female': rows}, split on the bare sex headings."""
    out, cur = {}, None
    for r in rows:
        lab = str(r[0]).strip() if r[0] is not None else ""
        if lab in ("Both Sexes", "Male", "Female") and all(x is None or not str(x).strip()
                                                             for x in r[1:]):
            cur = "Both" if lab == "Both Sexes" else lab
            out[cur] = []
            continue
        if cur:
            out[cur].append(r)
    if set(out) != {"Both", "Male", "Female"} or str(out["Both"][0][0]).strip() != first_label:
        raise SystemExit(f"sex blocks not found as expected: {list(out)}")
    return out


def read_mg14():
    """{sex: {place: {code: [all, India, Sri Lanka, Bangladesh, Maldives, Others, NS]}}}, place
    'atolls', 'admin', 'nonadmin' (nonadmin derived as atolls - admin and checked where printed),
    plus 'Male'' and 'Republic' under code None. Place of usual residence, country of birth."""
    blocks = sex_blocks(sheet("MG14.xlsx"), "Republic")
    out = {}
    for sex, rows in blocks.items():
        d = {"Republic": None, "Male'": None, "atolls": {}, "admin": {}, "nonadmin_printed": {}}
        place = None
        for r in rows:
            lab = str(r[0]).strip()
            vals = [num(x) for x in r[1:8]]
            if lab == "Republic":
                d["Republic"] = vals
            elif lab == "Male'":
                d["Male'"] = vals
            elif lab.startswith("Atolls"):
                place = "atolls"
            elif lab.startswith("Administrative"):
                place = "admin"
            elif lab.startswith("Non-") or lab.startswith("Non "):
                place = "nonadmin_printed"
            elif lab == "Not Stated":
                d["not_stated"] = vals
            else:
                c = code_of(lab)
                if c is None or place is None:
                    raise SystemExit(f"MG14: unexpected row {lab!r}")
                d[place][c] = vals
        for p in ("atolls", "admin"):
            if list(d[p]) != CODES:
                raise SystemExit(f"MG14 {sex} {p}: atolls {list(d[p])}")
        d["nonadmin"] = {}
        for c in CODES:
            der = [a - b for a, b in zip(d["atolls"][c], d["admin"][c])]
            pr = d["nonadmin_printed"].get(c)
            if pr is not None and any(v is not None for v in pr) and pr != der:
                raise SystemExit(f"MG14 {sex} {c}: non-administrative row {pr} != atoll - admin {der}")
            if min(der) < 0:
                raise SystemExit(f"MG14 {sex} {c}: negative non-administrative cell {der}")
            d["nonadmin"][c] = der
        out[sex] = d
    return out


def read_mg15():
    """{sex: {place: {citizenship or 'Total': n}}} for Republic, Male', atolls, admin, nonadmin.
    Place of enumeration, country of citizenship."""
    blocks = sex_blocks(sheet("MG15.xlsx"), "Republic")
    out = {}
    for sex, rows in blocks.items():
        d, place = {}, None
        for r in rows:
            lab = str(r[0]).strip()
            n = num(r[1])
            key = {"Republic": "Republic", "Male'": "Male'"}.get(lab)
            if key is None and lab.startswith("Atolls"):
                key = "atolls"
            elif key is None and lab.startswith("Administrative"):
                key = "admin"
            elif key is None and (lab.startswith("Non-") or lab.startswith("Non ")):
                key = "nonadmin"
            if key:
                place = key
                d[place] = {"Total": n}
            elif lab in CIT and place:
                d[place][lab] = n
            else:
                raise SystemExit(f"MG15: unexpected row {lab!r}")
        out[sex] = d
    return out


def read_pp3():
    """{'Male'': row, 'atolls': row, 'admin': {code: row}, 'nonadmin': row}; row = [total, m, f,
    maldivian, m, f, foreign, m, f]. Place of enumeration."""
    out = {"admin": {}}
    started = False
    for r in sheet("PP3.xls"):
        lab = str(r[0]).strip() if r[0] is not None else ""
        if lab == "Republic":
            started = True
        if not started or not lab or num(r[1]) is None:
            continue
        vals = [num(x) for x in r[1:10]]
        if lab == "Republic":
            out["Republic"] = vals
        elif lab == "Male'":
            out["Male'"] = vals
        elif lab.startswith("Atolls"):
            out["atolls"] = vals
        elif lab.startswith("Administrative"):
            out["admin_total"] = vals
        elif lab.startswith("Non Administrative"):
            out["nonadmin"] = vals
        elif lab in ("Resorts", "Industrial Islands and Others"):
            out[lab] = vals
        else:
            c = code_of(lab)
            if c not in CODES:
                raise SystemExit(f"PP3: unexpected row {lab!r}")
            out["admin"][c] = vals
    if list(out["admin"]) != CODES:
        raise SystemExit(f"PP3 atolls {list(out['admin'])}")
    return out


def read_mg4():
    """{code or 'Male'': Maldivians enumerated there}, administrative and other islands together."""
    rows = sex_blocks(sheet("MG4.xlsx"), "Republic")["Both"]
    out = {}
    for r in rows:
        lab = str(r[0]).strip()
        if lab == "Republic" or lab.startswith("Atolls"):
            out[lab.split(" ")[0]] = num(r[1])
        elif lab == "Male'":
            out["Male'"] = num(r[1])
        else:
            c = code_of(lab)
            if c not in CODES:
                raise SystemExit(f"MG4: unexpected row {lab!r}")
            out[c] = num(r[1])
    return out


def read_unsd(bad):
    got = oracle.oracle("Maldives", "2014")
    if not got:
        raise SystemExit("UNSD has no Maldives 2014 row; run `python tools/oracle.py --fetch`")
    sexed = {s: oracle.oracle("Maldives", "2014", sex=s) for s in ("Male", "Female")}
    lab = {"Islam": "Islam", "Hindu": "Hindu", "Christian": "Christian", "Buddhist": "Buddhist",
           "Other": "Other", "NS": "Not Stated"}
    for (area, sex), cells in UNSD.items():
        src = got[area] if sex == "Both" else sexed[sex][area]
        src = {k.strip(): v for k, v in src.items()}
        say(all(src[lab[k]] == v for k, v in cells.items())
            and sum(cells.values()) == src["Total"],
            f"UNSD 2014 {area} {sex}: the six answers as pinned, summing to {src['Total']:,}", bad)
    return {(a, s): {lab[k]: v for k, v in c.items()} for (a, s), c in UNSD.items()}


def pew_table():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    return t[t["Year"] == 2020].set_index("Country")


def desa_others(year="2015"):
    """{iso: stock} of UN DESA's named origins for the Maldives other than India, Sri Lanka and
    Bangladesh, and DESA's unnamed `Others`, for `year`."""
    if not os.path.exists(DESA):
        raise SystemExit(f"missing {DESA}; run `python sources/mr.py --fetch`")
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
    ycol = next(i for i, x in enumerate(cols) if x.replace(".0", "") == year)
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "Maldives"]
    code = pd.to_numeric(m[ccol], errors="coerce")
    names = m[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    stock = dict(zip(names[code < 900], pd.to_numeric(m.loc[code < 900, ycol]).astype(int)))
    unnamed = int(pd.to_numeric(m.loc[names == "Others", ycol]).iloc[0])
    iso_of = {v: k for k, v in origin.PEW_BY_ISO.items() if v}
    iso_of.update({"United States of America": "US", "Russian Federation": "RU"})
    out = {}
    for name, n in stock.items():
        if name in ("Bangladesh", "India", "Sri Lanka"):
            continue
        iso = iso_of.get(name)
        if iso is None:
            raise SystemExit(f"DESA origin {name!r} has no ISO here")
        out[iso] = n
    return out, unnamed


def seed_row(pew, weights):
    """Census-answer shares for a weighted mix of Pew country rows."""
    v = dict.fromkeys(CATS, 0.0)
    tot = float(sum(weights.values()))
    for iso, w in weights.items():
        pn = origin.PEW_BY_ISO.get(iso)
        if pn not in pew.index:
            raise SystemExit(f"Pew has no row {pn!r} for {iso}")
        row = {f: float(pew.loc[pn, f]) for f in origin.FAMILIES}
        s = sum(row.values())
        for f, x in row.items():
            for cat, k in FAM_TO_CAT[f].items():
                v[cat] += (w / tot) * k * x / s
    arr = np.array([v[c] for c in CATS]) + SEED_FLOOR
    return arr / arr.sum()


def fit(seed, A, B, tol=1e-9, iters=10_000):
    """IPF of X[sex, place, cit, cat] to A[sex, place, cit] and B[sex, area, cat], area 0 = Malé
    (place 0), area 1 = the atolls (places 1 and 2)."""
    X = np.broadcast_to(seed[None, None, :, :], A.shape + (len(CATS),)).copy()
    for it in range(iters):
        X *= (A / X.sum(axis=3))[..., None]
        X[:, 0] *= (B[:, 0] / X[:, 0].sum(axis=1))[:, None, :]
        at = X[:, 1:].sum(axis=(1, 2))
        X[:, 1:] *= np.where(at > 0, B[:, 1] / np.where(at > 0, at, 1), 0)[:, None, None, :]
        err = np.abs(X.sum(axis=3) - A).max()
        if err < tol:
            return X, it
    raise SystemExit(f"the fit did not converge (last error {err:.3g})")


def rake2(seed, rows, cols, iters=5_000):
    m = seed.astype(float).copy()
    for _ in range(iters):
        rs = m.sum(axis=1)
        m *= np.where(rs > 0, rows / np.where(rs > 0, rs, 1), 0)[:, None]
        cs = m.sum(axis=0)
        m *= np.where(cs > 0, cols / np.where(cs > 0, cs, 1), 0)[None, :]
        if np.abs(m.sum(axis=1) - rows).max() < 1e-7:
            return m
    raise SystemExit("a citizenship rake did not converge")


def largest_remainder(values, target):
    v = np.asarray(values, dtype=float)
    base = np.floor(v).astype(int)
    short = int(target) - int(base.sum())
    if short:
        base[np.argsort(-(v - base))[:short]] += 1
    return base


def round_both(m, rows, cols):
    """Integers keeping every row total and every column total: floor, then hand out the remainders
    by largest fraction where both the row and the column are still short."""
    base = np.floor(m).astype(int)
    frac = m - base
    rneed = rows - base.sum(axis=1)
    cneed = cols - base.sum(axis=0)
    for i, j in sorted(np.ndindex(m.shape), key=lambda ij: -frac[ij]):
        if rneed[i] > 0 and cneed[j] > 0:
            base[i, j] += 1
            rneed[i] -= 1
            cneed[j] -= 1
    if rneed.any() or cneed.any():
        raise SystemExit(f"rounding left rows {rneed} and columns {cneed} short")
    return base


def main():
    if "--fetch" in sys.argv or any(not os.path.exists(os.path.join(RAW, f)) for f in FILES):
        fetch()
    bad = []
    unsd = read_unsd(bad)
    pp3, mg4, mg14, mg15 = read_pp3(), read_mg4(), read_mg14(), read_mg15()

    # ---- 1. the tables agree with each other
    print("\n1. the census tables against each other")
    say(pp3["Republic"][0] == 402071 and pp3["Republic"][3] == 338434 and pp3["Republic"][6] == 63637,
        "PP3: 402,071 residents, 338,434 Maldivians, 63,637 foreign", bad)
    say(pp3["Male'"][0] == 153904 and sum(unsd[("Urban", s)][c] for s in SEXES for c in CATS) == 153904,
        "UNSD's Urban is Malé: 153,904 in both", bad)
    say(sum(pp3["admin"][c][6] for c in CODES) == pp3["admin_total"][6] == 16004
        and pp3["nonadmin"][6] == 23110 and pp3["atolls"][6] == 39114,
        "PP3 atolls: 16,004 foreign on administrative islands (the 20 rows sum to it), 23,110 on "
        "the others, 39,114 together", bad)
    say(mg4["Male'"] == pp3["Male'"][3] and sum(mg4[c] for c in CODES) == pp3["atolls"][3]
        and all(mg4[c] >= pp3["admin"][c][3] for c in CODES),
        "MG4's Maldivians by place of enumeration: Malé equals PP3, the atolls sum to PP3's "
        f"{pp3['atolls'][3]:,}, each atoll at least its administrative islands' count", bad)
    for sex, i in (("Male", 7), ("Female", 8)):
        for pl, pp in (("Male'", pp3["Male'"]), ("atolls", pp3["atolls"]),
                       ("admin", pp3["admin_total"]), ("nonadmin", pp3["nonadmin"])):
            g = mg15[sex][pl]
            say(g["Total"] == pp[i] == sum(g[c] for c in CIT),
                f"MG15 {sex} {pl}: {g['Total']:,} = PP3, = its four citizenships", bad)
    say(all(mg15["Both"][pl][c] == mg15["Male"][pl][c] + mg15["Female"][pl][c]
            for pl in ("Male'", "admin", "nonadmin") for c in CIT),
        "MG15: both sexes = men + women in every cell", bad)
    for sex in ("Both", "Male", "Female"):
        r = mg14[sex]
        tot = r["Republic"][0]
        parts = r["Male'"][0] + sum(r["atolls"][c][0] for c in CODES) + r["not_stated"][0]
        say(tot == parts and all(sum(v[1:]) == v[0] for v in r["atolls"].values()),
            f"MG14 {sex}: Malé + atolls + not stated = {tot:,}; birth countries sum to each row", bad)
    for sex, i in (("Male", 7), ("Female", 8)):
        d = [mg14[sex]["admin"][c][0] - pp3["admin"][c][i] for c in CODES]
        print(f"  ..  MG14 (usual residence) less PP3 (enumerated), administrative islands, {sex}: "
              f"{sum(d):+,} in all, per atoll {min(d):+,} to {max(d):+,}; PP3's are the ones used")

    # ---- 2. the foreign residents' answers, by sex, Malé and atolls
    print("\n2. foreign residents' answers (UNSD less PP3's Maldivians, all coded Islam)")
    B = np.zeros((2, 2, len(CATS)))
    for si, sex in enumerate(SEXES):
        for ai, (area, pp) in enumerate((("Urban", pp3["Male'"]), ("Rural", pp3["atolls"]))):
            cells = dict(unsd[(area, sex)])
            cells["Islam"] -= pp[4 + si]
            B[si, ai] = [cells[c] for c in CATS]
            say(min(cells.values()) >= 0 and sum(cells.values()) == pp[7 + si],
                f"{sex:<6} {'Malé' if ai == 0 else 'atolls':<6} {pp[7 + si]:>6,}: "
                + ", ".join(f"{c} {cells[c]:,}" for c in CATS), bad)
    tot_ns = int(B[..., CATS.index("Not Stated")].sum())
    say(tot_ns == unsd[("Total", "Both")]["Not Stated"],
        f"UNSD's {tot_ns:,} not stated all fit inside the foreign residents", bad)
    if bad:
        raise SystemExit(f"{len(bad)} check(s) failed")

    # ---- 3. each citizenship's answers: the fit
    print("\n3. each citizenship's answers, fitted to both printed margins")
    pew = pew_table()
    others, unnamed = desa_others()
    named = sum(others.values())
    seed = np.vstack([seed_row(pew, {CIT_ISO[c]: 1.0}) for c in CIT[:3]] + [seed_row(pew, others)])
    print(f"  seed for `others`: UN DESA 2015's {len(others)} named origins ({named:,}; DESA also has "
          f"{unnamed:,} unnamed), through Pew 2020")
    A = np.array([[[mg15[sex][pl][c] for c in CIT] for pl in PLACES] for sex in SEXES], dtype=float)
    X, it = fit(seed, A, B)
    P = X / X.sum(axis=3, keepdims=True)
    print(f"  converged in {it} rounds")
    for si, sex in enumerate(SEXES):
        for pi, pl in enumerate(PLACES):
            for ci, c in enumerate(CIT):
                if A[si, pi, ci] >= 300:
                    print(f"    {sex:<6} {pl:<8} {c:<10} {int(A[si, pi, ci]):>6,}: "
                          + ", ".join(f"{k} {P[si, pi, ci, j]:.1%}" for j, k in enumerate(CATS)
                                      if P[si, pi, ci, j] >= 0.005))

    # ---- 4. each atoll's foreign residents by sex, island kind and citizenship
    print("\n4. each atoll's foreign residents by sex, kind of island and citizenship")
    N = np.zeros((len(CODES), 2, 2, len(CIT)))       # atoll, sex, admin/nonadmin, citizenship
    for si, sex in enumerate(SEXES):
        rows_admin = np.array([pp3["admin"][c][7 + si] for c in CODES], dtype=float)
        usual = np.array([mg14[sex]["nonadmin"][c][0] for c in CODES], dtype=float)
        rows_non = largest_remainder(usual * mg15[sex]["nonadmin"]["Total"] / usual.sum(),
                                     mg15[sex]["nonadmin"]["Total"]).astype(float)
        for ki, (kind, rows) in enumerate((("admin", rows_admin), ("nonadmin", rows_non))):
            birth = np.array([mg14[sex][kind][c] for c in CODES], dtype=float)
            # India, Sri Lanka, Bangladesh, Others; born in the Maldives and not stated spread pro rata
            s4 = birth[:, [1, 2, 3, 5]]
            fallback = s4.sum(axis=0) / s4.sum()
            s4 = np.where(s4.sum(axis=1, keepdims=True) > 0, s4, fallback[None, :]) + 1e-6
            cols = np.array([mg15[sex][kind][c] for c in CIT], dtype=float)
            N[:, si, ki, :] = rake2(s4, rows, cols)
        print(f"  {sex}: resort and industrial islands, usual residence {int(usual.sum()):,} "
              f"scaled to the {mg15[sex]['nonadmin']['Total']:,} enumerated there")

    E = np.einsum("askc,skcr->ar", N, P[:, 1:])      # expected answers per atoll
    rows_tot = np.rint(N.sum(axis=(1, 2, 3))).astype(int)
    cols_tot = B[:, 1].sum(axis=0).astype(int)
    if abs(E.sum(axis=0) - cols_tot).max() > 1e-4:
        raise SystemExit("the atolls' expected answers do not sum to the printed totals")
    F = round_both(E, rows_tot, cols_tot)

    # ---- 5. witnesses: how much the seed decides
    def spread(seed_alt):
        Xa, _ = fit(seed_alt, A, B)
        Pa = Xa / Xa.sum(axis=3, keepdims=True)
        return np.einsum("askc,skcr->ar", N, Pa[:, 1:])
    nm = [i for i, c in enumerate(CATS) if c not in ("Islam", "Not Stated")]
    base_nm = E[:, nm].sum(axis=1)
    print("\n5. how much the seed decides: non-Muslim foreign residents per atoll, moved")
    flat = np.ones((len(CIT), len(CATS))) / len(CATS)
    for label, alt in (("every citizenship seeded alike", flat),
                       ("others seeded on Pew's All Asia-Pacific",
                        np.vstack([seed[:3], seed_row_regional(pew, "All Asia-Pacific")]))):
        d = spread(alt)[:, nm].sum(axis=1) - base_nm
        print(f"    {label}: {np.abs(d).sum() / 2:,.0f} of {base_nm.sum():,.0f} move between atolls; "
              f"largest {CODES[int(np.abs(d).argmax())]} {d[np.abs(d).argmax()]:+,.0f}")

    # ---- 6. write
    rows = []
    def add(code, cat, n, basis, tier, src, note):
        if n:
            # one level: Malé is a first-level unit beside the atolls in COD-AB and the census
            rows.append(dict(geo_id=PCODE[code], geo_level="atoll",
                             geo_name=NAME[code], source_category=cat, count=int(n), basis=basis,
                             year=2014, source_id=src, note=note, tier=tier))
    nat_note = ("not asked: the form skips religion for Maldivians, and the constitution bars "
                "non-Muslims from citizenship; drawn on Islam (sources/mv.py)")
    add("MALE", NATIONALS, pp3["Male'"][3], "estimate", "modelled", "mv_census2014_nationals", nat_note)
    for j, cat in enumerate(CATS):
        add("MALE", cat, int(B[:, 0, j].sum()), "count", "measured", "mv_census2014_foreign",
            "foreign residents' answers, census 2014 (UNSD table 28 Urban less PP3's Maldivians)")
    for i, c in enumerate(CODES):
        add(c, NATIONALS, mg4[c], "estimate", "modelled", "mv_census2014_nationals", nat_note)
        for j, cat in enumerate(CATS):
            add(c, cat, F[i, j], "nationality_derived", "derived", "mv_census2014_foreign",
                "foreign residents' answers for the atolls together (UNSD table 28 Rural less PP3's "
                "Maldivians), spread by sex, kind of island and citizenship (sources/mv.py)")
    out = pd.DataFrame(rows)
    per_unit = out.groupby("geo_id")["count"].sum()
    expect = {PCODE["MALE"]: pp3["Male'"][0]}
    expect.update({PCODE[c]: mg4[c] + rows_tot[i] for i, c in enumerate(CODES)})
    say(all(per_unit[k] == v for k, v in expect.items()) and per_unit.sum() == 402071,
        f"every unit sums to its people; {int(per_unit.sum()):,} in {len(per_unit)} units", bad)
    by_cat = out.groupby("source_category")["count"].sum()
    say(by_cat.get(NATIONALS, 0) == 338434 and all(
        by_cat.get(c, 0) == (unsd[("Total", "Both")][c] - (338434 if c == "Islam" else 0)) for c in CATS),
        "the categories sum to UNSD's national row (Islam less the Maldivians)", bad)
    if bad:
        raise SystemExit(f"{len(bad)} check(s) failed")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT + ".part", index=False, encoding="utf-8")
    os.replace(OUT + ".part", OUT)

    f = out[out["source_category"] != NATIONALS]
    nonm = f[~f["source_category"].isin(["Islam", "Not Stated"])]
    print(f"\nwrote {OUT}: {len(out)} rows")
    print("  foreign non-Muslims by unit: " + ", ".join(
        f"{n} {int(v):,}" for n, v in nonm.groupby("geo_name")["count"].sum()
        .sort_values(ascending=False).items()))
    got = dict(foreign=int(f["count"].sum()), foreign_muslim=int(by_cat["Islam"]),
               hindu=int(by_cat["Hindu"]), christian=int(by_cat["Christian"]),
               buddhist=int(by_cat["Buddhist"]), other=int(by_cat["Other"]),
               not_stated=int(by_cat["Not Stated"]),
               male_non_muslim=int(nonm[nonm["geo_id"] == PCODE["MALE"]]["count"].sum()))
    print(f"  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


def seed_row_regional(pew, region):
    if region not in pew.index:
        raise SystemExit(f"Pew has no row {region!r}")
    row = {f: float(pew.loc[region, f]) for f in origin.FAMILIES}
    s = sum(row.values())
    v = dict.fromkeys(CATS, 0.0)
    for fam, x in row.items():
        for cat, k in FAM_TO_CAT[fam].items():
            v[cat] += k * x / s
    arr = np.array([v[c] for c in CATS]) + SEED_FLOOR
    return arr / arr.sum()


if __name__ == "__main__":
    main()
