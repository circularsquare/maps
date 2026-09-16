"""Taiwan: religion by county and city, from seven rounds of the Taiwan Social Change Survey.

Reads the ARDA open copies of TSCS 1994, 1999, 2004, 2009, 2014 (religion module), 2015 and 2018
(religion module) and writes data/normalized/tw.csv. `sources/tw.md` is this country's record;
`sources.md` §11i (2026-09-06) closed Taiwan and §scout-2026-09-14-taiwan-belarus-gabon reopened it.

**NO TAIWANESE CENSUS ASKS ABOUT RELIGION** (§11i, the 2020 census read), and the Ministry of the
Interior's figures are a roll of registered religious bodies that certify their own members (6.9%
of the population, §11i). Every row this file writes is `modelled` in §7's sense.

## WHERE EACH RESPONDENT IS

TSCS draws townships (PPS) inside strata, then people from the household register, so the
sampling township is where the population base (the register) counts that person too.

  * 1994, 1999, 2004: `LIVES`, the three-digit postcode of the respondent's township.
  * 2009: `WHRLIVEP`, the postcode, and `WHRLIVEM`, the county label (pre-2010 counties). The
    two agree on every respondent both place, and that is asserted.
  * 2014: only `ZIP`, a county label for the sampling area; the cluster is stratum and PSU.
  * 2015: `ZIP`, the postcode, and `ZIP2`, the county label. Agree everywhere, asserted.
  * 2018: `ZIP`, the sampling area's postcode, and `V4CITY`, the county of present residence.
    They differ for about one respondent in eighteen (V4 says 16.3% were interviewed away from
    the registered address). The sampling area is used, because it is the register's geography.

Postcodes go to counties through Chunghwa Post's table of the 368 three-digit codes (2016,
unambiguous: every code sits in one county). The 2010 mergers joined whole counties (Taipei
County became New Taipei City; Taichung, Tainan and Kaohsiung counties joined their cities) and
Taoyuan was upgraded whole in 2014, so every old code lands in exactly one of today's 22 units.

**TRAP: THE CARDS ARE NOT CODED ALIKE.** 1994 and 2015 used a short card, the other five the
long one. In 1994 code 9 is *None*; in 2015 code 9 is *Cihui Tang* and *None* is 10. Each round
has its own code table here and every code's label is checked against the answer it is mapped to.

## THE LEVEL AND THE PATTERN COME FROM DIFFERENT ROUNDS

The national split between Buddhism and folk religion moves with the card: the short cards put
Buddhism at 38.5% (1994) and 19.9% (2015), the long cards, where the interviewer codes *worships
the gods* as folk religion, at 11% to 22%. So:

  * **Level**: the two religion-module rounds on the same long card, 2014 and 2018, weighted.
  * **Pattern**: all seven rounds, as each county's observed count over the count its own
    respondents would give at their round's national shares (indirect standardisation). A county
    sampled mostly in 1994 is then not read as Buddhist because 1994's card was.

## WHICH ANSWERS CARRY THEIR OWN GEOGRAPHY

Townships nest inside counties, so the resampling unit is the township (spec §12, "WHERE THE
SAMPLING UNITS NEST INSIDE THE DRAWN UNITS, THE NULL REGROUPS THEM"). Each split halves every
round's townships at random; the statistic is the median Spearman, over the splits, between the
two halves' county indices; the null deals townships to counties at random inside each round,
keeping each county's township count. The spatial chi-square (observed against expected) is a
veto, and so is one township holding over half of an answer (`stability.CELL_CAP`), nationally or
inside the county with the highest index. That second cap refuses `Buddhism and Taoism, or the
three teachings`: Hsinchu County's 35 answers are 26 from 2009 postcode 303 (Hukou), 25 of them
code 102, and 9 from 2004 postcode 310. Carried: none, folk religion, Buddhism, Taoism,
Protestant. Answers that fail share each county's remainder at their national proportions
(`lapop.build`'s rule), unless spec §12's 2x rule sends the tail flat (it does not, 1.16x).

The carried indices are pulled toward 1 by gamma-Poisson empirical Bayes before the fit, so a
county of 98 respondents (Taitung) is not drawn at the edge of what 98 people can show; the fit
closes every county on the register and every carried answer on its 2014-2018 level.

## THREE COUNTIES ARE NEVER SAMPLED

Penghu, Kinmen and Lienchiang appear in no round. Ecuador's line (Galápagos, Anita 2026-09-08):
nothing measured them, so they are not drawn and are in `gap=`.

Usage:
    python sources/tw.py --fetch    seven Stata files from OSF, the postcode table, the MOI base
    python sources/tw.py            rebuild data/normalized/tw.csv
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import stability as shared  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "tw")
POST_XML = os.path.join(RAW, "post_zip3.xml")
MOI_JSON = os.path.join(RAW, "odrp048_114.json")
OUT = os.path.join(ROOT, "data", "normalized", "tw.csv")

POST_URL = "https://www.post.gov.tw/post/download/1050812_行政區經緯度(toPost).xml"
MOI_URL = "https://www.ris.gov.tw/rs-opendata/api/v1/datastore/ODRP048/114"
OSF = "https://osf.io/download/{}"

SOURCE_ID = "tw_tscs_1994_2018"

# ISO 3166-2, which is also geoBoundaries' shapeISO (sources/tw_geo.py asserts the pairing).
UNITS = {
    "TW-TPE": ("臺北市", "Taipei City"),
    "TW-NWT": ("新北市", "New Taipei City"),
    "TW-TAO": ("桃園市", "Taoyuan City"),
    "TW-TXG": ("臺中市", "Taichung City"),
    "TW-TNN": ("臺南市", "Tainan City"),
    "TW-KHH": ("高雄市", "Kaohsiung City"),
    "TW-KEE": ("基隆市", "Keelung City"),
    "TW-HSZ": ("新竹市", "Hsinchu City"),
    "TW-CYI": ("嘉義市", "Chiayi City"),
    "TW-HSQ": ("新竹縣", "Hsinchu County"),
    "TW-MIA": ("苗栗縣", "Miaoli County"),
    "TW-CHA": ("彰化縣", "Changhua County"),
    "TW-NAN": ("南投縣", "Nantou County"),
    "TW-YUN": ("雲林縣", "Yunlin County"),
    "TW-CYQ": ("嘉義縣", "Chiayi County"),
    "TW-PIF": ("屏東縣", "Pingtung County"),
    "TW-ILA": ("宜蘭縣", "Yilan County"),
    "TW-HUA": ("花蓮縣", "Hualien County"),
    "TW-TTT": ("臺東縣", "Taitung County"),
    "TW-PEN": ("澎湖縣", "Penghu County"),
    "TW-KIN": ("金門縣", "Kinmen County"),
    "TW-LIE": ("連江縣", "Lienchiang County"),
}
ZH = {zh: iso for iso, (zh, _) in UNITS.items()}
NOT_DRAWN = ["TW-PEN", "TW-KIN", "TW-LIE"]

# The county labels the files print, pre- and post-2010, lower-cased. Anything else stops.
LABEL_UNIT = {
    "keelung city": "TW-KEE", "taipei city": "TW-TPE", "taipei county": "TW-NWT",
    "new taipei city": "TW-NWT", "taoyuan county": "TW-TAO", "taoyuan city": "TW-TAO",
    "hsinchu city": "TW-HSZ", "hsinchu county": "TW-HSQ", "miaoli county": "TW-MIA",
    "taichung city": "TW-TXG", "taichung county": "TW-TXG", "nantou county": "TW-NAN",
    "changhua county": "TW-CHA", "yunlin county": "TW-YUN", "chiayi city": "TW-CYI",
    "chiayi county": "TW-CYQ", "tainan city": "TW-TNN", "tainan county": "TW-TNN",
    "kaohsiung city": "TW-KHH", "kaohsiung county": "TW-KHH", "pingtung county": "TW-PIF",
    "ilan county": "TW-ILA", "hualien county": "TW-HUA", "taitung county": "TW-TTT",
}
NOT_A_UNIT = {"other foreign country", "other", "don't want to answer", "china"}

# ---- the harmonised answers, which are the source categories taxonomy/tw2018.py maps ----------
NONE = "No religious belief"
FOLK = "Folk religion"
BUDD = "Buddhism"
TAO = "Taoism"
SYNC = "Buddhism and Taoism, or the three teachings"
YGD = "Yiguan Dao"
PROT = "Protestant Christianity"
CATH = "Catholicism"
JAP = "Japanese religions"
CHN = "Other Chinese religions"
ISL = "Islam"
OTH = "Other"
ANSWERS = [NONE, FOLK, BUDD, TAO, SYNC, YGD, PROT, CATH, JAP, CHN, ISL, OTH]

# Every mapped code's label must match its answer's pattern (lower-cased), so a renumbered card
# fails here and not in the map.
LABEL_TEST = {
    NONE: r"no religi|^none$",
    FOLK: r"folk|self-identified|worships the gods|not clearly specified",
    BUDD: r"buddh|zong|chan and pure land",
    TAO: r"^taoism",
    SYNC: r"both buddhism and taoism|three religions|buddhism, taoism|polytheism|^other \(please specify\)$",
    YGD: r"yiguan",
    PROT: r"protestant",
    CATH: r"cathol",
    JAP: r"japan|nichiren|soka",
    CHN: r"cihui|tindi|tingde|local religion",
    ISL: r"islam",
    OTH: r"other|categori|unification|foreign",
}
SHORT_1994 = {1: BUDD, 2: TAO, 3: FOLK, 4: YGD, 6: ISL, 7: CATH, 8: PROT, 9: NONE, 10: OTH}
SHORT_2015 = {1: BUDD, 2: TAO, 3: FOLK, 4: YGD, 6: ISL, 7: CATH, 8: PROT, 9: CHN, 10: NONE,
              11: OTH}
LONG = {10: NONE, 20: FOLK, 21: FOLK, 22: FOLK, 23: FOLK, 24: FOLK,
        30: BUDD, 31: BUDD, 32: BUDD, 33: BUDD, 34: BUDD, 35: BUDD, 36: BUDD, 37: BUDD, 39: BUDD,
        40: TAO, 50: CATH, 60: PROT, 71: CHN, 72: YGD, 74: CHN, 75: CHN, 78: CHN,
        81: JAP, 82: JAP, 85: JAP, 91: ISL, 94: OTH, 95: OTH,
        101: SYNC, 102: SYNC, 103: SYNC, 110: OTH}
NO_ANSWER = {97, 98, 99, 996, 997, 998, 999}
# Labels that do not name their answer, pinned exactly. TSCS 2018's code 39 sits in the Buddhist
# block of the card (after `Other Buddhism`, before `Taoism`) and is the Buddhist who did not know
# the sect, which 2014 labels `Buddhism: Don't know`.
LABEL_EXCEPTIONS = {("TSCS181", 39): "do not know"}

ROUNDS = {
    1994: dict(fid="TSC94", osf="q2yvw", n=1862, rel="religion", card=SHORT_1994, zip="lives",
               label=None, weight=None, unplaced=2),
    1999: dict(fid="TSC99", osf="bv7nk", n=1925, rel="religion", card=LONG, zip="lives",
               label=None, weight=None, unplaced=4),
    2004: dict(fid="TSC04", osf="vgr7e", n=1881, rel="relbel", card=LONG, zip="lives",
               label=None, weight="weight", unplaced=0),
    2009: dict(fid="TSC09", osf="c7vw3", n=1927, rel="relbel", card=LONG, zip="whrlivep",
               label="whrlivem", weight="weight", unplaced=1),
    2014: dict(fid="TSCS142", osf="u8kxr", n=1934, rel="v15", card=LONG, zip=None, label="zip",
               weight="wr_19_5", unplaced=0),
    2015: dict(fid="TSCS151", osf="t48sd", n=2034, rel="v11", card=SHORT_2015, zip="zip",
               label="zip2", weight="wsel", unplaced=0),
    2018: dict(fid="TSCS181", osf="xgn86", n=1842, rel="v29", card=LONG, zip="zip",
               label="v4city", weight="wr_19_5", unplaced=0),
}
YEARS = sorted(ROUNDS)
LEVEL_ROUNDS = [2014, 2018]
# Rounds where the county label is a second reading of the same place as the postcode.
LABEL_AGREES = [2009, 2015]

# ---- the test --------------------------------------------------------------------------------
N_SPLITS = 200
N_NULL = 400
SEED = 0
MIN_UNITS = 8
STAB_ALPHA = shared.STAB_ALPHA
CELL_CAP = shared.CELL_CAP

# Pinned after the first run; a change is a change in what this country claims to know.
CARRIES = [NONE, FOLK, BUDD, TAO, PROT]
TAIL_FLAT = False
STANDOUTS = []
MOI_TOTAL = 23_299_132         # ODRP048, 民國114年, the 368 townships summed


# ---------------------------------------------------------------------------------------------
def _get(url, path, binary=True):
    import requests
    r = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(path + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(path + ".part", path)
    print(f"  {os.path.basename(path)}: {len(r.content):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for y, spec in ROUNDS.items():
        path = os.path.join(RAW, spec["fid"] + ".dta")
        if not os.path.exists(path):
            _get(OSF.format(spec["osf"]), path)
    if not os.path.exists(POST_XML):
        _get(POST_URL, POST_XML)
    fetch_moi()


def fetch_moi():
    import requests
    rows, page, pages = [], 1, 1
    while page <= pages:
        r = requests.get(MOI_URL, params={"page": page}, timeout=120,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        j = r.json()
        pages = int(j["totalPage"])
        rows += j["responseData"]
        page += 1
    with open(MOI_JSON + ".part", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    os.replace(MOI_JSON + ".part", MOI_JSON)
    print(f"  {os.path.basename(MOI_JSON)}: {len(rows)} rows")


def moi_counties():
    """MOI household-register population and land area by county, end of 2025 (民國114年).

    From ODRP048, population density by township: 368 townships, plus the Dongsha and Nansha
    island groups, which carry an area and no population and are left out.
    """
    with open(MOI_JSON, encoding="utf-8") as fh:
        rows = json.load(fh)
    if {r["statistic_yyy"] for r in rows} != {"114"}:
        raise SystemExit("ODRP048 file is not year 114")
    town = [r for r in rows if r["people_total"].isdigit()]
    rest = sorted(r["site_id"] for r in rows if not r["people_total"].isdigit())
    if len(town) != 368 or rest != ["南沙群島", "東沙群島"]:
        raise SystemExit(f"ODRP048: {len(town)} townships with people, others {rest}")
    df = pd.DataFrame(town)
    df["zh"] = df["site_id"].str[:3]
    bad = sorted(set(df["zh"]) - set(ZH))
    if bad:
        raise SystemExit(f"ODRP048 county prefixes not in UNITS: {bad}")
    df["pop"] = df["people_total"].astype(int)
    df["area"] = df["area"].astype(float)
    out = df.groupby("zh").agg(pop=("pop", "sum"), area=("area", "sum"), towns=("pop", "size"))
    out.index = [ZH[z] for z in out.index]
    if len(out) != 22:
        raise SystemExit(f"{len(out)} counties in ODRP048")
    return out.reindex(list(UNITS))


def zip_table():
    root = ET.parse(POST_XML).getroot()
    out = {}
    for rec in root:
        name = code = None
        for el in rec:
            if el.tag == "行政區名":
                name = el.text.strip()
            elif el.tag.endswith("碼郵遞區號"):
                code = int(el.text.strip())
        if name is None or code is None:
            raise SystemExit("a postcode record without a name or a code")
        if name[:3] not in ZH:
            raise SystemExit(f"postcode {code} {name}: county prefix not a unit")
        if code in out and out[code] != ZH[name[:3]]:
            raise SystemExit(f"postcode {code} is in two counties")
        out[code] = ZH[name[:3]]
    if len(out) != 368 or len(set(out.values())) != 22:
        raise SystemExit(f"postcode table: {len(out)} codes over {len(set(out.values()))} counties")
    return out


def read_round(year, zips):
    import re
    spec = ROUNDS[year]
    path = os.path.join(RAW, spec["fid"] + ".dta")
    with pd.io.stata.StataReader(path) as r:
        raw = r.read(convert_categoricals=False)
        labels = r.value_labels()
        lbl = {k.lower(): v for k, v in zip(raw.columns, r._lbllist)}
    raw.columns = [c.lower() for c in raw.columns]
    if len(raw) != spec["n"]:
        raise SystemExit(f"{spec['fid']}: {len(raw)} rows, expected {spec['n']}")

    # religion
    rel = raw[spec["rel"]].astype(float).astype(int)
    rlab = labels.get(lbl.get(spec["rel"], ""), {})
    card = spec["card"]
    for code in sorted(set(rel)):
        if code in NO_ANSWER:
            continue
        if code not in card:
            raise SystemExit(f"{spec['fid']} {spec['rel']}: code {code} "
                             f"({rlab.get(code)!r}) is on no answer")
        text = str(rlab.get(code, "")).lower()
        if (spec["fid"], code) in LABEL_EXCEPTIONS:
            if text != LABEL_EXCEPTIONS[(spec["fid"], code)]:
                raise SystemExit(f"{spec['fid']} code {code} is now {text!r}")
        elif not re.search(LABEL_TEST[card[code]], text):
            raise SystemExit(f"{spec['fid']} code {code} {text!r} does not read as {card[code]!r}")
    answer = rel.map(card)

    # residence
    unit = pd.Series(index=raw.index, dtype=object)
    if spec["zip"]:
        z = pd.to_numeric(raw[spec["zip"]], errors="coerce")
        unit = z.map(lambda v: zips.get(int(v)) if pd.notna(v) else None)
    lab_unit = None
    if spec["label"]:
        clab = labels.get(lbl.get(spec["label"], ""), {})

        def one(v):
            t = str(clab.get(v, "")).strip().lower()
            if t in LABEL_UNIT:
                return LABEL_UNIT[t]
            if t in NOT_A_UNIT:
                return None
            raise SystemExit(f"{spec['fid']} {spec['label']}: label {t!r} is neither a county "
                             "nor a known non-place")
        lab_unit = raw[spec["label"]].map(one)
        if not spec["zip"]:
            unit = lab_unit
    moved = 0
    if spec["zip"] and spec["label"]:
        both = unit.notna() & lab_unit.notna()
        moved = int((unit[both] != lab_unit[both]).sum())
        if year in LABEL_AGREES and moved:
            raise SystemExit(f"{spec['fid']}: postcode and county label disagree on {moved}")

    if spec["weight"]:
        w = pd.to_numeric(raw[spec["weight"]], errors="coerce")
        if w.isna().any() or (w <= 0).any():
            raise SystemExit(f"{spec['fid']} {spec['weight']}: missing or non-positive weights")
    else:
        w = pd.Series(1.0, index=raw.index)

    if year == 2014:
        cluster = ("2014:" + raw["r_stratum2014"].astype(int).astype(str) + ":"
                   + raw["psu"].astype(int).astype(str))
    else:
        cluster = str(year) + ":" + pd.to_numeric(raw[spec["zip"]], errors="coerce") \
            .fillna(-1).astype(int).astype(str)

    df = pd.DataFrame({"year": year, "code": rel, "answer": answer, "unit": unit,
                       "cluster": cluster, "w": w.astype(float)})
    unplaced = int(df["unit"].isna().sum())
    if unplaced != spec["unplaced"]:
        raise SystemExit(f"{spec['fid']}: {unplaced} respondents with no county, "
                         f"expected {spec['unplaced']}")
    no_answer = int(df["answer"].isna().sum())
    df = df[df["unit"].notna() & df["answer"].notna()].copy()
    df["w"] = df["w"] / df["w"].mean()
    per = df.groupby("cluster")["unit"].nunique()
    if (per > 1).any():
        raise SystemExit(f"{spec['fid']}: clusters in two counties: {list(per[per > 1].index)[:5]}")
    print(f"  {year} {spec['fid']:<8} n={spec['n']:>5}  placed and answered {len(df):>5}  "
          f"no county {unplaced}  no answer {no_answer}  "
          f"{df['cluster'].nunique():>3} clusters  weight {spec['weight'] or 'none'}"
          + (f"  sampling area and present residence differ for {moved}"
             if spec["zip"] and spec["label"] and year not in LABEL_AGREES else ""))
    return df


# ---------------------------------------------------------------------------------------------
def national(df, weighted=True):
    col = "w" if weighted else None
    g = df.groupby("answer")["w"].sum() if col else df.groupby("answer").size()
    return (g / g.sum()).reindex(ANSWERS, fill_value=0.0)


def standardised(df, units, weighted):
    """(observed, expected) as unit x answer arrays: each respondent's round share is expected."""
    val = df["w"] if weighted else pd.Series(1.0, index=df.index)
    d = df.assign(v=val)
    p = d.groupby(["year", "answer"])["v"].sum().unstack(fill_value=0.0) \
        .reindex(index=YEARS, columns=ANSWERS, fill_value=0.0)
    p = p.div(p.sum(axis=1), axis=0).fillna(0.0)      # a round absent from `df` expects nothing
    n = d.groupby(["year", "unit"])["v"].sum().unstack(fill_value=0.0) \
        .reindex(index=YEARS, columns=units, fill_value=0.0)
    E = n.to_numpy().T @ p.to_numpy()
    O = d.groupby(["unit", "answer"])["v"].sum().unstack(fill_value=0.0) \
        .reindex(index=units, columns=ANSWERS, fill_value=0.0).to_numpy()
    return O, E


class Clusters:
    """Township-level arrays for the split-half: counts, expected counts, unit and round."""

    def __init__(self, df, units):
        p = df.groupby(["year", "answer"]).size().unstack(fill_value=0) \
            .reindex(index=YEARS, columns=ANSWERS, fill_value=0).astype(float)
        p = p.div(p.sum(axis=1), axis=0)
        cl = df.groupby("cluster").agg(year=("year", "first"), unit=("unit", "first"))
        self.names = cl.index.to_numpy()
        self.year = cl["year"].to_numpy()
        self.unit = np.array([units.index(u) for u in cl["unit"]])
        self.K = df.groupby(["cluster", "answer"]).size().unstack(fill_value=0) \
            .reindex(index=cl.index, columns=ANSWERS, fill_value=0).to_numpy(float)
        self.n = self.K.sum(axis=1)
        self.Ek = self.n[:, None] * p.loc[self.year].to_numpy()
        self.n_units = len(units)

    def onehot(self, unit):
        U = np.zeros((len(unit), self.n_units))
        U[np.arange(len(unit)), unit] = 1.0
        return U

    def splits(self, rng):
        H = np.zeros((N_SPLITS, len(self.year)))
        for s in range(N_SPLITS):
            for y in YEARS:
                idx = np.flatnonzero(self.year == y)
                pick = rng.permutation(idx)[: len(idx) // 2 + rng.integers(0, 2) * (len(idx) % 2)]
                H[s, pick] = 1.0
        return H

    def permuted_units(self, rng):
        unit = self.unit.copy()
        for y in YEARS:
            idx = np.flatnonzero(self.year == y)
            unit[idx] = unit[rng.permutation(idx)]
        return unit

    def halves(self, H, unit):
        """(split, unit, answer) indices for halves A and B; nan where a unit has no one."""
        U = self.onehot(unit)
        out = []
        for mask in (H, 1.0 - H):
            UA = U[None, :, :] * mask[:, :, None]            # split x cluster x unit
            O = np.einsum("scu,ck->suk", UA, self.K)
            E = np.einsum("scu,ck->suk", UA, self.Ek)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.append(np.where(E > 0, O / E, np.nan))
        return out

    def median_rho(self, H, unit):
        IA, IB = self.halves(H, unit)
        k = len(ANSWERS)
        med = np.full(k, np.nan)
        for j in range(k):
            vals = []
            for s in range(H.shape[0]):
                ok = np.isfinite(IA[s, :, j]) & np.isfinite(IB[s, :, j])
                if ok.sum() >= MIN_UNITS:
                    r = shared.rho(IA[s, ok, j], IB[s, ok, j])
                    if np.isfinite(r):
                        vals.append(r)
            if vals:
                med[j] = float(np.median(vals))
        return med, IA, IB


def stability(df, units):
    """Township halves inside rounds, on standardised county indices. Returns the verdicts."""
    from scipy.stats import chi2

    rng = np.random.default_rng(SEED)
    cl = Clusters(df, units)
    H = cl.splits(rng)
    obs, IA, IB = cl.median_rho(H, cl.unit)
    null = np.full((N_NULL, len(ANSWERS)), np.nan)
    for i in range(N_NULL):
        null[i], _, _ = cl.median_rho(H, cl.permuted_units(rng))

    O, E = standardised(df, units, weighted=False)
    su, st = top_both_halves_standout(IA, IB)
    print(f"\n  split-half on {len(cl.names)} townships in {len(YEARS)} rounds over "
          f"{len(units)} counties: median Spearman of {N_SPLITS} within-round halvings of the "
          f"townships, on each county's observed/expected index; null deals townships to "
          f"counties within round, {N_NULL} draws; chi-square of observed against expected; "
          f"largest township's share of the answer.")
    print(f"    {'answer':<44}{'n':>6}{'median':>8}{'null95':>8}{'p':>8}{'chi2 p':>10}"
          f"{'cell':>6}  verdict")
    verdicts = {}
    for j, a in enumerate(ANSWERS):
        n = int(cl.K[:, j].sum())
        ok = E[:, j] > 0
        x2 = float((((O[ok, j] - E[ok, j]) ** 2) / E[ok, j]).sum())
        chi_p = float(chi2.sf(x2, max(int(ok.sum()) - 1, 1))) if n else float("nan")
        cell = float(cl.K[:, j].max() / n) if n else float("nan")
        p, q95 = shared.permutation_p(obs[j], null[:, j], STAB_ALPHA)
        if not np.isfinite(p):
            v = "no test possible"
        elif p < STAB_ALPHA and chi_p < STAB_ALPHA and cell <= CELL_CAP:
            v = "own geography"
        elif p < STAB_ALPHA and cell > CELL_CAP:
            v = "REFUSED: one township holds over half"
        elif p < STAB_ALPHA:
            v = "REFUSED: rank test passes, the counties do not differ"
        else:
            v = "not distinguishable from chance"
        # The national cell cap cannot see one township setting the TOP county. Hsinchu County's
        # `three teachings in one` was 26 of 35 answers from 2009 postcode 303 (Hukou), while that
        # township held 10% of the answer nationally. So for a pass, the county with the highest
        # index must not have over CELL_CAP of its answers in one township.
        top_note = ""
        if v == "own geography":
            with np.errstate(invalid="ignore", divide="ignore"):
                ix = np.where(E[:, j] > 0, O[:, j] / E[:, j], -1.0)
            top = int(np.argmax(ix))
            in_top = cl.K[cl.unit == top, j]
            top_cell = float(in_top.max() / in_top.sum()) if in_top.sum() else 0.0
            top_note = f"  (top county {UNITS[units[top]][1]}: largest township {top_cell:.0%})"
            if top_cell > CELL_CAP:
                v = "REFUSED: the top county's reading is one township"
        stand = top_note
        if v != "own geography" and su[j] >= 0:
            stand = f"  (top in both halves: {UNITS[units[su[j]]][1]} {st[j]:.0%})"
        verdicts[a] = dict(n=n, median=obs[j], q95=q95, p=p, chi_p=chi_p, cell=cell, verdict=v,
                           standout=(units[su[j]] if su[j] >= 0 else None), agree=st[j])
        print(f"    {a[:42]:<44}{n:>6,}{obs[j]:+8.3f}{q95:+8.3f}{p:8.4f}{chi_p:10.2e}"
              f"{cell:6.0%}  {v}{stand}")
    return verdicts


def top_both_halves_standout(IA, IB):
    sa = np.nan_to_num(IA, nan=0.0)
    sb = np.nan_to_num(IB, nan=0.0)
    return shared.top_both_halves(sa, sb)


# ---------------------------------------------------------------------------------------------
IPF_TOL = 1e-10
IPF_MAX = 1000


def compose(df, units, pop, level, carries, verdicts):
    """Shares for every drawn county, closing each to its population.

    The seed is the level times the county's weighted standardised index for each carried
    answer, plus one pooled column for the tail at its level. Iterative proportional fitting
    then makes every county sum to its register population and every carried answer's national
    total its level, keeping the seed's county-by-answer odds ratios, which is the pattern the
    seven rounds measured. The tail column is split inside each county at national proportions
    (`lapop.build`'s rule), or all goes flat under spec §12's 2x rule.

    Why a fit: level x index does not close by itself. The index is measured against each round's
    own national shares and the level is 2014-2018's, and on the first run the six carried
    answers alone came to more than the whole county in seven counties.
    """
    O, E = standardised(df, units, weighted=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = pd.DataFrame(np.where(E > 0, O / E, 0.0), index=units, columns=ANSWERS)
    # Empirical Bayes (Marshall 1991, the global gamma-Poisson estimator): each county's index is
    # pulled toward 1 in proportion to how little its respondents could show. The first build drew
    # Nantou 0.0% Protestant on 116 respondents and Hsinchu County 19.3% `Buddhism and Taoism`
    # (national 1.96%) on about 109 long-card ones. The test above is untouched; only the seed is.
    print("\n  empirical Bayes on the county index (prior mean 1, variance from the counties):")
    for j, a in enumerate(ANSWERS):
        if a not in carries:
            continue
        o, e = O[:, j], E[:, j]
        m = o.sum() / e.sum()
        s2 = float((e * (o / np.where(e > 0, e, np.nan) - m) ** 2)[e > 0].sum() / e.sum())
        between = s2 - m / (e.sum() / len(e))
        if between <= 0:
            eb = np.full(len(e), m)
            print(f"    {a[:42]:<44} no between-county variance left: every county at the mean")
        else:
            alpha, beta = m * m / between, m / between
            eb = (o + alpha) / (e + beta)
            moved = np.abs(eb - idx[a].to_numpy())
            worst = np.argsort(-moved)[:2]
            print(f"    {a[:42]:<44} prior strength {beta:6.1f} expected answers; moved most: "
                  + "; ".join(f"{UNITS[units[i]][1]} {idx[a].iloc[i]:.2f} to {eb[i]:.2f} "
                              f"(O {o[i]:.1f}, E {e[i]:.1f})" for i in worst))
        idx[a] = eb
    P = pop.reindex(units).astype(float).to_numpy()
    tail = [a for a in ANSWERS if a not in carries and level[a] > 0]
    tail_total = float(level[tail].sum())
    seed = np.column_stack([level[a] * idx[a].to_numpy() for a in carries]
                           + [np.full(len(units), tail_total)])
    col_target = np.array([level[a] for a in carries] + [tail_total]) * P.sum()
    if not np.isclose(col_target.sum(), P.sum()):
        raise SystemExit("the level does not sum to one over the carried answers and the tail")
    M = seed * P[:, None]
    for it in range(1, IPF_MAX + 1):
        M *= (P / M.sum(axis=1))[:, None]
        M *= (col_target / M.sum(axis=0))[None, :]
        err = max(np.abs(M.sum(axis=1) / P - 1).max(), np.abs(M.sum(axis=0) / col_target - 1).max())
        if err < IPF_TOL:
            break
    else:
        raise SystemExit(f"the fit did not converge in {IPF_MAX} passes ({err:.2e})")
    print(f"\n  fit: {it} passes; every county closes on the register, every carried answer on "
          f"its 2014-2018 level")
    share = pd.DataFrame(0.0, index=units, columns=ANSWERS)
    fitted = M / P[:, None]
    for j, a in enumerate(carries):
        share[a] = fitted[:, j]
    residual = pd.Series(fitted[:, -1], index=units)
    multiple = residual / tail_total
    unw_O, _ = standardised(df, units, weighted=False)
    found_none = pd.DataFrame(unw_O == 0, index=units, columns=ANSWERS)
    rows, worst = shared.residual_multiples(multiple, found_none, tail)
    flat = worst is not None and worst[2] >= shared.SMALL_CATEGORY_MULTIPLE
    print(f"\n  the tail ({', '.join(tail)}) is {residual.min():.1%} of "
          f"{UNITS[residual.idxmin()][1]} and {residual.max():.1%} of {UNITS[residual.idxmax()][1]}, "
          f"against {tail_total:.1%} nationally")
    for a, u, m in rows:
        print(f"    {a:<44} none found in {UNITS[u][1]:<18} drawn there at {m:.2f}x national")
    print(f"  2x rule: worst {worst[2]:.2f}x ({worst[0]}, {UNITS[worst[1]][1]})" if worst else
          "  2x rule: every tail answer was found in every county")
    if flat:
        print("  -> the tail goes FLAT: national shares everywhere, carried shares scaled")
        for u in units:
            k = (1.0 - tail_total) / float(share.loc[u, carries].sum())
            share.loc[u, carries] *= k
            for a in tail:
                share.loc[u, a] = level[a]
    else:
        for a in tail:
            share[a] = residual * (level[a] / tail_total)
    if not np.allclose(share.sum(axis=1), 1.0):
        raise SystemExit("a county's shares do not close")
    return share, idx, flat


def main():
    if "--fetch" in sys.argv:
        fetch()
    for path in [POST_XML, MOI_JSON] + [os.path.join(RAW, s["fid"] + ".dta") for s in ROUNDS.values()]:
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing; run with --fetch")

    zips = zip_table()
    moi = moi_counties()
    total = int(moi["pop"].sum())
    print(f"MOI register, end of 2025: {total:,} people in 22 counties "
          f"({int(moi['towns'].sum())} townships)")
    if MOI_TOTAL is not None and total != MOI_TOTAL:
        raise SystemExit(f"MOI total {total:,}, pinned {MOI_TOTAL:,}")

    print("\nTSCS rounds:")
    df = pd.concat([read_round(y, zips) for y in YEARS], ignore_index=True)
    sampled = [u for u in UNITS if u in set(df["unit"])]
    never = [u for u in UNITS if u not in sampled]
    if never != NOT_DRAWN:
        raise SystemExit(f"never-sampled counties are {never}, not {NOT_DRAWN}")
    gap = int(moi.loc[never, "pop"].sum())
    print(f"\n  {len(df):,} respondents on {len(sampled)} counties; never sampled: "
          + ", ".join(UNITS[u][1] for u in never)
          + f", {gap:,} people ({gap / total:.2%})")

    tab = df.groupby(["unit", "year"]).size().unstack(fill_value=0).reindex(index=sampled,
                                                                           fill_value=0)
    print("\n  respondents by county and round:")
    print("    " + tab.assign(total=tab.sum(axis=1))
          .rename(index=lambda u: UNITS[u][1]).to_string().replace("\n", "\n    "))

    print("\n  national shares by round (weighted where the round has weights), %:")
    by = pd.DataFrame({y: national(df[df["year"] == y]) * 100 for y in YEARS})
    print("    " + by.round(1).to_string().replace("\n", "\n    "))
    level = national(df[df["year"].isin(LEVEL_ROUNDS)])
    print(f"\n  level, {' and '.join(map(str, LEVEL_ROUNDS))} pooled and weighted "
          f"(n={int(df['year'].isin(LEVEL_ROUNDS).sum()):,}):")
    for a in ANSWERS:
        print(f"    {level[a] * 100:6.2f}%  {a}")

    verdicts = stability(df, sampled)
    carries = [a for a in ANSWERS if verdicts[a]["verdict"] == "own geography"]
    standouts = sorted(a for a in ANSWERS
                       if verdicts[a]["verdict"] != "own geography"
                       and verdicts[a]["agree"] >= shared.STANDOUT_AGREE
                       and verdicts[a]["chi_p"] < STAB_ALPHA and verdicts[a]["cell"] <= CELL_CAP)
    print(f"\n  -> carried on their own county index: {carries}")
    print(f"  -> standouts (one county tops both halves in {shared.STANDOUT_AGREE:.0%} of splits, "
          f"with the chi-square and the cell cap holding): {standouts}")
    if CARRIES is not None and carries != CARRIES:
        raise SystemExit(f"the test now carries {carries}, not {CARRIES}. That is a change in what "
                         "this country claims to know: read the table, then update CARRIES, the "
                         "docstring and sources/tw.md deliberately.")
    if STANDOUTS is not None and standouts != STANDOUTS:
        raise SystemExit(f"standouts now {standouts}, not {STANDOUTS}")
    if standouts:
        raise SystemExit(f"{standouts} qualify as standouts and compose() does not draw one yet")

    early = df[df["year"] <= 2004]
    late = df[df["year"] >= 2009]
    Oe, Ee = standardised(early, sampled, weighted=False)
    Ol, El = standardised(late, sampled, weighted=False)
    print("\n  printed, never deciding: 1994-2004 against 2009-2018, Spearman of county indices "
          "over counties sampled in both:")
    for j, a in enumerate(ANSWERS):
        ok = (Ee[:, j] > 0) & (El[:, j] > 0)
        if ok.sum() >= MIN_UNITS and a in carries:
            print(f"    {a:<44}{shared.rho(Oe[ok, j] / Ee[ok, j], Ol[ok, j] / El[ok, j]):+.3f} "
                  f"over {int(ok.sum())}")

    share, idx, flat = compose(df, sampled, moi["pop"], level, carries, verdicts)
    if TAIL_FLAT is not None and flat != TAIL_FLAT:
        raise SystemExit(f"the 2x rule now says flat={flat}, pinned {TAIL_FLAT}")

    rows = []
    n_by = df.groupby("unit").size()
    for u in sampled:
        p = int(moi.loc[u, "pop"])
        counts = (share.loc[u] * p).round().astype("int64")
        counts[counts.idxmax()] += p - int(counts.sum())
        for a in ANSWERS:
            if counts[a] <= 0:
                continue
            how = ("county index on the 2014 and 2018 level" if a in carries
                   else ("national share" if flat else "national share within the county's residual"))
            rows.append(dict(geo_id=u, geo_level="county", geo_name=UNITS[u][1],
                             source_category=a, count=int(counts[a]), basis="self_id",
                             year="1994-2018", source_id=SOURCE_ID,
                             note=f"TSCS 1994-2018 pooled, n={int(n_by[u])} in this county; {how}; "
                                  f"applied to the MOI register population, end of 2025"))
    out = pd.DataFrame(rows)
    # Spec §3.13 (Anita, 2026-09-15): `chinesefolk` is folk religion named, or a religious home
    # altar kept by someone who names no religion. sources/tw_altar.py splits the drawn answers.
    import tw_altar  # noqa: E402
    out = tw_altar.apply(out, sys.modules[__name__], zips, sampled)
    drawn = int(out["count"].sum())
    if drawn != total - gap:
        raise SystemExit(f"drawn {drawn:,} against {total - gap:,}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {drawn:,} people, {out['geo_id'].nunique()} counties)")

    print("\n  national, as drawn:")
    nat = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for a, c in nat.items():
        print(f"    {c / drawn * 100:6.2f}%  {c:>12,}  {a}")
    if carries:
        print("\n  carried answers by county, % of the county, with its respondents:")
        cols = carries
        print(f"    {'county':<18}{'n':>6}" + "".join(f"{c[:14]:>16}" for c in cols))
        for u in sorted(sampled, key=lambda x: -share.loc[x, cols[0]]):
            print(f"    {UNITS[u][1]:<18}{int(n_by[u]):>6}"
                  + "".join(f"{share.loc[u, c] * 100:15.1f}%" for c in cols))


if __name__ == "__main__":
    main()
