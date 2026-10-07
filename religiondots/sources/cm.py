"""Cameroon — religion in twelve units from five pooled Afrobarometer rounds, on COD-PS 2025.

Reads data/raw/afrobarometer/*.sav, data/geo/cm/cm_lookup.csv and cm_departments.csv, and as a
witness only data/raw/cm/cmr-2005-rec_TOME2.1_etat_structure.pdf; writes data/normalized/cm.csv.
`sources/cm.md` is the record; `sources/tz.py` is the construction this follows.

## CAMEROON ASKS, AND PRINTS THE ANSWER FOR THE WHOLE COUNTRY ONLY

The 2005 census (3e RGPH) asked every person's religion (Q12). BUCREP printed it as Tableau 5.8
of *Volume II Tome 01, État et structures de la population*: national, by sex and by urban and
rural, and nothing below the nation except one paragraph (printed p.101) giving the leading
religion's share in each region. The 2005 microdata is licensed (NADA catalog 89) and UNSD holds
no Cameroon row. The 4th census was enumerated from 24 April 2026 and has published nothing yet;
that is when this country should be rebuilt.

## WHAT IS DRAWN, AND WHERE EACH NUMBER COMES FROM

    row margin      unit populations        COD-PS 2025 (BUCREP's projection)   EXACT
    the composition each unit's own mix     Afrobarometer R5-R9 pooled           measured
    the national level                      neither                              computed

Nothing is fitted to a column margin. The census's national table is twenty years older than the
populations it would be fitted to, which is §6's rejected IPF, so it is read as a witness
(`census_witness`) and never used to draw. Every row is `modelled`.

## TWELVE UNITS, BECAUSE THE SURVEY SAMPLES YAOUNDÉ AND DOUALA APART

`sources/cm_geo.py` has the geometry. Every round lists Yaoundé and Douala under labels of their
own that change every round (`Yaounde`, `Centre-Yaoundé`, `Mfoundi`; `Douala`, `Littoral-Douala`,
`Wouri`), and round 8 shifts every REGION code by one against rounds 6, 7 and 9. So the decode
is by label (`NORM`), and it is checked against the survey's own department column in rounds 6,
7 and 9 (`check_locations`), which puts every city respondent in Mfoundi or Wouri and nobody else.

## TWO CHURCHES ARE DRAWN; THE REST OF CHRISTIANITY IS ONE COLOUR

The Afrobarometer trap (`sources/afrobarometer.py`, "THE ANSWER CARD IS THE SAME CARD AND THE
PROBING IS NOT") is here: the weighted share answering `Christian only` runs 6.7, 6.0, 10.4, 12.2
and 13.5% over rounds 5 to 9. **What makes Cameroon different is who that box takes from.**
Roman Catholic runs 40.5, 41.4, 29.5, 26.3, 36.8% (15.1 points) and Evangelical 7.8 to 2.7%, so
their levels measure the fieldwork and they are not drawn. Presbyterian runs 8.6 to 10.9% (2.2
points), Baptist 3.2 to 4.1% (0.9) and Lutheran 2.1 to 2.4% (0.3) while the catch-all doubles, so
the probing that moves Catholics does not move them. The census is the outside witness the
playbook asks for, at the family level: in 2005 it counted **Protestants at 26.3%**, and the
survey's Protestant bodies (Presbyterian, Baptist, Lutheran, Evangelical, Pentecostal, Adventist
and the small ones) come to about 27% without taking anything from `Christian only`, while its
Catholics fall 3 to 4 points short of the census's 38.4%. So `Christian only` is mostly Catholics
and unnamed others, and the Protestant churches' levels stand without it.

**A level that holds is not enough on its own, because `Christian only` is not spread evenly.**
Pooled, it is 7% of Christians in Ouest and 8 to 9% in Centre, Mfoundi and Nord-Ouest, but 29% in
Nord and 35% in Adamaoua. A church measured where a third of Christians name none is drawn short
exactly where it lives. So a church is drawn only if the unnamed share, averaged over where its
own respondents are, is no higher than the national one (`unnamed_where_they_live`). Presbyterians
(Nord-Ouest, Sud-Ouest, Sud, Centre) and Baptists (Nord-Ouest, Sud-Ouest, Littoral) pass;
Lutherans, the church of Adamaoua, Nord and Extrême-Nord, fail and are folded into `Christian`.
A reviewer (2026-09-14, `sources/cm.md` §4) argued for folding all three; this keeps its point
about the north and not its conclusion.

`LEVEL_RANGE_MAX` asserts the two stay level, `unnamed_where_they_live` that they still pass and
Lutherans still fail, and `CATHOLIC_RANGE_MIN` the Catholic swing that keeps the Catholic share
from being pooled over all five rounds, so a re-release that changes any of them re-opens the call.

## THE CATHOLIC CHURCH, FROM THE TWO ROUNDS THE DHS AGREES WITH (2026-10-03)

Namibia's rule (`sources/na.md` §3): a church whose answer drifts by round is taken from the rounds
that match an outside witness to its level. The witness is the two open DHS reports (EDS-MICS 2011,
EDS 2018), whose Tableau 3.1 names `Catholique` beside `Protestant` and `Autres chrétiens` for women
and men 15-49: Catholics are 51.3% and 54.1% of Christians (`dhs_witness`; the 2005 census had
55.5%). The survey's Catholics as a share of its Christians run 51.4 and 51.5% in rounds 5 and 6 and
42.1, 37.6 and 47.7% in rounds 7-9. So each unit's Catholics are its share of the `Christian` pool
(everything Christian but the two churches above) in rounds 5-6, times the pool's five-round share
(`carve_catholic`). The R5-against-R6 split-half passes (+0.657); over all five rounds the Catholic
share of the pool passes too (+0.706), which says the geography holds while the level moves.
`Christian only` stays in `Christian`: it is 8% of Christians in rounds 5-6, and the unnamed share
where Catholics live is under the national one, so they are not drawn short where they are. The
Church's diocesan statistics by cathedral region are printed beside it (`diocese_witness`).

Pentecostal (2.5-8.0% of Christians by round, R6 the low one while Evangelical peaks) and
Evangelical (3.4-11.0%) still move with the fieldwork and have no witness of their own; Adventist
fails the split-half. They stay in `Christian`.

## `Other` IS NOT PLACED

It passes the split-half (+0.63) at 1.04%, but all 58 of its respondents are in rounds 5 to 7 and
nobody chose it in rounds 8 or 9 although the box is on both cards. A pooled share that depends on
which rounds are in is Izala's case in `sources/ng.py`, so it goes in the tail. Asserted.

Usage:
    python sources/cm.py --fetch    the census volume (~7 MB) from CEPED's IREDA inventory; the
                                    Afrobarometer files are shared (`python sources/afrobarometer.py --fetch`)
    python sources/cm.py            rebuild data/normalized/cm.csv
"""

import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import afrobarometer as ab
import cab

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "cm", "cm_lookup.csv")
DEPTS = os.path.join(ROOT, "data", "geo", "cm", "cm_departments.csv")
RAW = os.path.join(ROOT, "data", "raw", "cm")
CENSUS_PDF = os.path.join(RAW, "cmr-2005-rec_TOME2.1_etat_structure.pdf")
CENSUS_URL = "https://ireda.ceped.org/inventaire/ressources/cmr-2005-rec_TOME2.1_etat_structure.pdf"
OUT = os.path.join(ROOT, "data", "normalized", "cm.csv")

COUNTRY = "Cameroon"
ROUNDS = [5, 6, 7, 8, 9]
RECENT = [8, 9]
SOURCE_ID = "cm_afrobarometer_2013_2022_codps2025"
YEARS = "2013-2022"

N_UNITS = 12
CODPS_2025 = 29_442_318

CHURCHES = ["Presbyterian", "Baptist"]
# Level-stable, split-half passing, and folded in because it lives where Christians name no church.
FOLDED_CHURCHES = ["lutheran"]
CATEGORIES = ["Christian", *CHURCHES, "Muslim", "Traditional/ethnic religion", "None", "Other"]

# Every answer Cameroonians give over the five rounds -> the category it is drawn as, keyed through
# `key()`, named one by one because an answer that falls through a default is dropped silently.
#   * `Protestant` is a Cameroon code on round 7's card only (17 people); it cannot say which
#     church, so it stays in `Christian` rather than in any of the three drawn.
#   * `Dutch Reformed`, `Zionist Christian Church`, `Coptic` and `New Apostolic Church` are other
#     countries' boxes on a continental card; 18 Cameroonians between them, all Christian.
#   * `Orthodox` (68) is Christian; the census counted Orthodox at 0.5% in 2005.
#   * `Atheist` and `Agnostic` join the card's own `None`; `Bahai` joins `Other`.
GROUP = {
    "christian only": "Christian", "roman catholic": "Christian", "orthodox": "Christian",
    "coptic": "Christian", "anglican": "Christian", "methodist": "Christian",
    "quaker/friends": "Christian", "mennonite": "Christian", "evangelical": "Christian",
    "pentecostal": "Christian", "independent": "Christian", "jehovah's witness": "Christian",
    "seventh day adventist": "Christian", "church of christ": "Christian",
    "zionist christian church": "Christian", "dutch reformed": "Christian",
    "new apostolic church": "Christian", "protestant": "Christian",
    "presbyterian": "Presbyterian", "baptist": "Baptist", "lutheran": "Christian",
    "muslim only": "Muslim", "sunni only": "Muslim", "shia": "Muslim", "ismaeli": "Muslim",
    "tijaniya brotherhood": "Muslim", "mouridiya brotherhood": "Muslim",
    "qadiriya brotherhood": "Muslim",
    "traditional/ethnic religion": "Traditional/ethnic religion",
    "none": "None", "atheist": "None", "agnostic": "None",
    "other": "Other", "bahai": "Other",
}
# The Protestant bodies, for the census witness only (the census's `Protestant` line).
PROTESTANT_KEYS = {"presbyterian", "baptist", "lutheran", "evangelical", "pentecostal",
                   "seventh day adventist", "methodist", "anglican", "church of christ",
                   "dutch reformed", "protestant", "quaker/friends", "mennonite"}

# Every REGION label over the five rounds, keyed through `gkey()`, -> the unit's pcode.
NORM = {
    "adamaoua": "CM001", "adamawa": "CM001",
    "centre": "CM002", "centreyaounde": "CM002007", "yaounde": "CM002007", "mfoundi": "CM002007",
    "est": "CM003", "east": "CM003",
    "extremenord": "CM004", "extremenorth": "CM004",
    "littoral": "CM005", "littoraldouala": "CM005004", "douala": "CM005004", "wouri": "CM005004",
    "nord": "CM006", "north": "CM006",
    "nordouest": "CM007", "northwest": "CM007",
    "ouest": "CM008", "west": "CM008",
    "sud": "CM009",
    "sudouest": "CM010",
}
REGION_OF = {"CM002007": "CM002", "CM005004": "CM005"}
# The survey's department spellings that do not fold onto COD-AB's: two typing slips and two
# spellings of the same names.
DEPT_ALIAS = {"kakey": "kadey", "mkam": "nkam", "koupeetmanengouba": "kupemanenguba",
              "ngoketundjia": "ngoketunjia"}
LOCATION_ROUNDS = [6, 7, 9]

# What is drawn on its own unit shares, asserted against the split-half so a change in the data
# stops the build. Set from the test's output, 2026-09-14.
CARRIES = ["Christian", "Presbyterian", "Baptist", "Muslim", "None"]
# Passes and is not placed; see the docstring.
NOT_PLACED = {"Other": "all of its respondents are in rounds 5-7; nobody chose it in 8 or 9"}
# Spec §12's small-category rule. Measured 2026-09-14: under the residual, Traditional would be
# drawn at 1.05x its national share in Littoral and Other at 1.71x in Ouest, the worst units where
# the survey found none; both under 2x, so the tail stays the residual. Asserted.
TAIL_FLAT = False
# The three churches must stay level across rounds; Catholics must still swing.
LEVEL_RANGE_MAX = 0.03
CATHOLIC_RANGE_MIN = 0.10
# Spec §12 (Norway): the pooled level against rounds 8-9.
LEVEL_GAP_MAX = 0.035

# ---- the Catholic Church, carved out of `Christian` (see the docstring) ----
OUT_CATEGORIES = ["Christian", "Catholic", *CHURCHES, "Muslim", "Traditional/ethnic religion",
                  "None", "Other"]
CATHOLIC_ROUNDS = [5, 6]
CATH_DHS_GAP_MAX = 0.03     # rounds 5 and 6 within 3 points of the DHS mean (gaps 1.3, 1.2)
CATH_SWING_MIN = 0.045      # rounds 7-9 at least 4.5 points under it (10.6, 15.1, 5.0)
CATH_DRAWN_GAP_MAX = 0.04   # the drawn Catholic share of Christians within 4 points of the DHS mean
DIOCESE_RHO_MIN = 0.60      # drawn Catholic share against the dioceses' by cathedral region
# The two open DHS reports, Tableau 3.1, the rows exactly as the PDF prints them (women: %, weighted
# n, unweighted n; men the same). The weighted n's are repeated as numbers for the arithmetic.
DHS = {
    "2011": ("FR260.pdf", 68, {
        "Catholique": ("Catholique 36,6 5 642 5 735 38,5 2 484 2 471", 5642, 2484),
        "Protestant": ("Protestant 33,8 5 221 5 415 30,3 1 953 2 104", 5221, 1953),
        "Autre chrétien": ("Autre chrétien 2,5 390 398 2,3 151 147", 390, 151)}),
    "2018": ("FR360.pdf", 103, {
        "Catholique": ("Catholique 38,0 5 179 5 061 38,8 2 375 2 333", 5179, 2375),
        "Protestant": ("Protestant 26,2 3 565 3 877 24,1 1 478 1 579", 3565, 1478),
        "Autres chrétiens": ("Autres chrétiens 7,4 1 013 1 073 5,6 346 367", 1013, 346)}),
}
DHS_URLS = {"FR260.pdf": "https://dhsprogram.com/pubs/pdf/FR260/FR260.pdf",
            "FR360.pdf": "https://dhsprogram.com/pubs/pdf/FR360/FR360.pdf"}
# GCatholic's region names (English) -> the census's ten regions.
DIOCESE_REGION = {"Adamaoua": "CM001", "Centre": "CM002", "East": "CM003", "Far North": "CM004",
                  "Littoral": "CM005", "North": "CM006", "North-West": "CM007", "West": "CM008",
                  "South": "CM009", "South-West": "CM010"}

# Tableau 5.8, "Ensemble, les deux sexes", printed p.97 (PDF page 124), per cent.
CENSUS_PAGE = 124
CENSUS_T58 = {"Catholique": 38.4, "Orthodoxe": 0.5, "Protestant": 26.3, "Autres chrétiens": 4.0,
              "Musulman": 20.9, "Animiste": 5.6, "Autres religions": 1.0, "Libre penseur": 3.2}
# Printed p.101 (PDF page 128): the leading religion's share in each region, 2005.
PROSE_PAGE = 128
CENSUS_PROSE = {
    ("Catholic", "CM002"): ("Centre,lescatholiquessont65,4%", 65.4),
    ("Catholic", "CM005"): ("Littoral(52,0%)", 52.0),
    ("Catholic", "CM003"): ("Est(42,4%)", 42.4),
    ("Catholic", "CM010"): ("Sud-Ouest(41,3%)", 41.3),
    ("Catholic", "CM008"): ("Ouest(34,7%)", 34.7),
    ("Protestant", "CM007"): ("Nord-Ouestpour49,3%", 49.3),
    ("Protestant", "CM009"): ("Sud(49,1%)", 49.1),
    ("Muslim", "CM001"): ("Adamaoua(71,5%)", 71.5),
    ("Muslim", "CM004"): ("Extrême-Nord(42,7%)", 42.7),
    ("Muslim", "CM006"): ("Nord(40,7%)", 40.7),
}


def key(s):
    """An Afrobarometer religion label reduced to what identifies the answer (`sources/tz.py`)."""
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'").replace("‘", "'")
    s = s.split("(")[0]
    s = re.sub(r"\s*/\s*", "/", s)
    return " ".join(s.split()).strip().casefold()


def gkey(s):
    """A REGION or department label as bare letters. Round 6 is read as LATIN1 (`ab._read`), so
    its UTF-8 accents arrive as `Ã©`; undo that first, then strip accents and everything else."""
    s = str(s)
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(CENSUS_PDF) and os.path.getsize(CENSUS_PDF) > 1_000_000:
        print(f"  have {os.path.basename(CENSUS_PDF)}")
        return
    r = requests.get(CENSUS_URL, headers={"User-Agent": ab.UA}, timeout=600)
    r.raise_for_status()
    if r.content[:4] != b"%PDF" or b"%%EOF" not in r.content[-2048:]:
        raise SystemExit("the census volume is not a complete PDF "
                         "([[reference_pdf_truncated_at_source]])")
    with open(CENSUS_PDF + ".part", "wb") as f:
        f.write(r.content)
    os.replace(CENSUS_PDF + ".part", CENSUS_PDF)
    print(f"  got  {os.path.basename(CENSUS_PDF)} ({len(r.content):,} bytes)")


def fetch_witnesses():
    """The two DHS reports and the GCatholic diocese table, into data/raw/cm/."""
    import requests
    import gcatholic

    for name, url in DHS_URLS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"  have {name}")
            continue
        r = requests.get(url, headers={"User-Agent": ab.UA}, timeout=600)
        r.raise_for_status()
        if r.content[:4] != b"%PDF" or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit(f"{name} is not a complete PDF ([[reference_pdf_truncated_at_source]])")
        with open(dst + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")
    gcatholic.fetch("CM", "cm")


def dhs_witness():
    """Catholics as a share of Christians in each DHS, re-read from Tableau 3.1; women and men
    pooled by weighted number. Returns {year: fraction}."""
    import fitz

    out = {}
    for year, (name, page, rows) in DHS.items():
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run with --fetch")
        t = " ".join(fitz.open(path)[page - 1].get_text().split())
        if "Tableau 3.1" not in t:
            raise SystemExit(f"{name} PDF page {page} is not Tableau 3.1")
        for lab, (line, _wn, _mn) in rows.items():
            if line not in t:
                raise SystemExit(f"DHS {year}: the row {line!r} is not on PDF page {page}")
        wn = {lab: wn + mn for lab, (_l, wn, mn) in rows.items()}
        out[year] = wn["Catholique"] / sum(wn.values())
    print("\n  DHS Tableau 3.1 re-read from both reports and equal; Catholics as a share of "
          "Christians, women and men 15-49 pooled: "
          + ", ".join(f"{y} {100 * v:.1f}%" for y, v in out.items()))
    return out


def carve_catholic(df, units, nm, dhs):
    """Each unit's Catholics as a share of the `Christian` pool in CATHOLIC_ROUNDS, with the checks
    that justify those rounds. Returns a Series over `units`."""
    chr_ = df[df["category"].isin(["Christian", *CHURCHES])]
    by_round = (chr_[chr_["k"] == "roman catholic"].groupby("round")["w"].sum()
                / chr_.groupby("round")["w"].sum())
    ref = float(np.mean(list(dhs.values())))
    print(f"\n  Catholics as a share of the survey's Christians by round, against the DHS mean "
          f"{100 * ref:.1f}%:")
    for r, v in by_round.items():
        print(f"    R{r}  {100 * v:5.1f}%  ({100 * (v - ref):+5.1f})"
              + ("  used" if r in CATHOLIC_ROUNDS else ""))
    off = {r: round(float(v), 3) for r, v in by_round.items()
           if r in CATHOLIC_ROUNDS and abs(v - ref) > CATH_DHS_GAP_MAX}
    if off:
        raise SystemExit(f"rounds {CATHOLIC_ROUNDS} no longer sit within "
                         f"{100 * CATH_DHS_GAP_MAX:.0f} points of the DHS Catholic share: {off}")
    near = {r: round(float(v), 3) for r, v in by_round.items()
            if r not in CATHOLIC_ROUNDS and ref - v < CATH_SWING_MIN}
    if near:
        raise SystemExit(f"rounds outside {CATHOLIC_ROUNDS} now come within "
                         f"{100 * CATH_SWING_MIN:.1f} points of the DHS: {near}; decide again")

    pool = df[df["round"].isin(CATHOLIC_ROUNDS) & (df["category"] == "Christian")]
    tot = pool.groupby("geo_id")["w"].sum().reindex(units, fill_value=0.0)
    cat = pool[pool["k"] == "roman catholic"].groupby("geo_id")["w"].sum().reindex(units,
                                                                                     fill_value=0.0)
    if (tot <= 0).any():
        raise SystemExit("a unit has no Christian-pool respondent in rounds 5-6")
    frac = cat / tot
    print(f"    Catholic share of the pool in rounds 5-6, by unit: "
          + ", ".join(f"{nm[u]} {100 * v:.0f}%" for u, v in frac.sort_values(ascending=False).items()))

    dfw = pool.assign(code=np.where(pool["k"] == "roman catholic", "Catholic", "rest of the pool"))
    dfw = dfw.rename(columns={"round": "wave"})[["wave", "geo_id", "code", "w"]]
    passed, _t = cab.stability(dfw, ["Catholic", "rest of the pool"], units,
                               f"{len(units)} units, the Christian pool in rounds 5-6")
    if "Catholic" not in passed:
        raise SystemExit("the Catholic share of the pool no longer passes the R5-R6 split-half")

    # the unnamed share where Catholics live, in the rounds used (sources/cm.md §4's second test)
    c56 = chr_[chr_["round"].isin(CATHOLIC_ROUNDS)]
    by = c56.groupby("geo_id")["w"].sum()
    un = c56[c56["k"] == "christian only"].groupby("geo_id")["w"].sum().reindex(by.index, fill_value=0)
    ufrac = un / by
    national = float(un.sum() / by.sum())
    w = c56[c56["k"] == "roman catholic"].groupby("geo_id")["w"].sum()
    where = float((w * ufrac.reindex(w.index)).sum() / w.sum())
    print(f"    `Christian only` in rounds 5-6: {100 * national:.1f}% of Christians nationally, "
          f"{100 * where:.1f}% averaged where the Catholics are")
    if where > national:
        raise SystemExit("Catholics now live where more Christians name no church than nationally; "
                         "their share would be drawn short where they are")
    return frac


def diocese_witness(share, pop, nm):
    """The drawn Catholic share by region against the Church's diocesan figures, summed by the
    region each cathedral stands in. A rank check, printed with its bar; not a level."""
    import gcatholic
    from scipy.stats import spearmanr

    g = gcatholic.by_seat("cm", rename=DIOCESE_REGION)
    unknown = sorted(set(g.index) - set(DIOCESE_REGION.values()))
    if unknown:
        raise SystemExit(f"GCatholic cathedral regions not in DIOCESE_REGION: {unknown}")
    reg = pd.Series({u: REGION_OF.get(u, u) for u in pop.index})
    drawn = (share["Catholic"] * pop).groupby(reg).sum() / pop.groupby(reg).sum()
    both = g.join(drawn.rename("drawn"), how="inner")
    rho = float(spearmanr(both["share"], both["drawn"]).statistic)
    rname = {"CM002": "Centre", "CM005": "Littoral", **{u: nm[u] for u in both.index
                                                         if u not in ("CM002", "CM005")}}
    print(f"\n  the Catholic Church's diocesan figures (GCatholic, 2016-2022), by the cathedral's "
          f"region, {len(both)} regions: Spearman {rho:+.3f} against the drawn Catholic share")
    print("    " + ", ".join(f"{rname[u]} {100 * r.share:.0f}/{100 * r.drawn:.0f}"
                             for u, r in both.sort_values("share", ascending=False).iterrows())
          + " (church/drawn, %)")
    print(f"    the Church claims {100 * g.catholics.sum() / g.population.sum():.1f}% of its own "
          "population figure nationally; a count of the baptised, not a level")
    if rho < DIOCESE_RHO_MIN:
        raise SystemExit(f"the drawn Catholic geography ranks the regions at {rho:+.2f} against the "
                         f"Church's dioceses, under {DIOCESE_RHO_MIN}")


def report_card():
    """The boxes the grouping relies on must be on every pooled round's card (value labels)."""
    import pyreadstat

    watch = ["christian only", "roman catholic", "presbyterian", "baptist", "lutheran",
             "muslim only", "none", "traditional/ethnic religion", "other"]
    print("\n  boxes on each round's showcard (value labels, not responses):")
    missing = []
    for rnd, name, _url, relname, _wt in ab.ROUNDS:
        if rnd not in ROUNDS:
            continue
        path = os.path.join(ab.AB_DIR, name)
        try:
            _d, meta = pyreadstat.read_sav(path, metadataonly=True)
        except pyreadstat._readstat_parser.ReadstatError:
            _d, meta = pyreadstat.read_sav(path, metadataonly=True, encoding="LATIN1")
        col = next(c for c in meta.column_names if c.upper() == relname.upper())
        have = {key(v) for v in meta.variable_value_labels.get(col, {}).values()}
        gone = [w for w in watch if w not in have]
        print(f"    R{rnd}: {'every box present' if not gone else 'MISSING ' + ', '.join(gone)}")
        missing += [(rnd, w) for w in gone]
    if missing:
        raise SystemExit(f"boxes missing from a pooled round's card: {missing}")


def check_locations(df, nm):
    """Rounds 6, 7 and 9 carry the department (`LOCATION.LEVEL.1`); it must agree with the unit."""
    d = pd.read_csv(DEPTS, dtype=str)
    if len(d) != 58:
        raise SystemExit(f"{DEPTS} has {len(d)} departments, expected 58; re-run cm_geo.py")
    dept_unit = dict(zip(d["department"].map(gkey), d["unit"]))
    print("\n  unit decoded from REGION against the department column (LOCATION.LEVEL.1):")
    for rnd in LOCATION_ROUNDS:
        sub = df[df["round"] == rnd]
        k = sub["LOCATION.LEVEL.1"].map(gkey).map(lambda s: DEPT_ALIAS.get(s, s))
        du = k.map(dept_unit)
        if du.isna().any():
            raise SystemExit(f"R{rnd}: department labels with no COD-AB department: "
                             f"{sorted(sub.loc[du.isna(), 'LOCATION.LEVEL.1'].astype(str).unique())}")
        bad = du != sub["geo_id"]
        print(f"    R{rnd}: {len(sub):,} of {len(sub):,} respondents' departments match a COD-AB "
              f"department; {int(bad.sum())} disagree with the unit from REGION")
        if bad.any():
            pairs = sub[bad].assign(du=du[bad]).groupby(["geo_id", "LOCATION.LEVEL.1", "du"]).size()
            raise SystemExit(f"R{rnd}: department and REGION disagree:\n{pairs.to_string()}")
    city = df[df["round"].isin(LOCATION_ROUNDS) & df["geo_id"].isin(REGION_OF)]
    print(f"    so the {len(city):,} Yaoundé and Douala respondents in those rounds are in Mfoundi "
          "and Wouri, and nobody labelled Centre or Littoral is")


def levels_by_round(raw):
    """The churches' levels across rounds, and the swings that fold Catholics away."""
    tot = raw.groupby("round")["w"].sum()
    t = raw.groupby(["k", "round"])["w"].sum().unstack(fill_value=0.0).div(tot, axis=1)
    show = ["roman catholic", "christian only", "presbyterian", "baptist", "lutheran",
            "evangelical", "pentecostal", "seventh day adventist", "other"]
    print("\n  weighted share of all respondents by round (%), and the range:")
    for k in show:
        row = t.loc[k] if k in t.index else pd.Series(0.0, index=tot.index)
        print(f"    {k:<24}" + "".join(f"{100 * row.get(r, 0):7.1f}" for r in ROUNDS)
              + f"   range {100 * (row.max() - row.min()):5.1f}")
    rng = {c: float(t.loc[c.lower()].max() - t.loc[c.lower()].min()) for c in CHURCHES}
    moved = {c: v for c, v in rng.items() if v > LEVEL_RANGE_MAX}
    if moved:
        raise SystemExit(f"a drawn church's level now moves more than {100 * LEVEL_RANGE_MAX:.0f} "
                         f"points across rounds: {moved}; the docstring's argument no longer holds")
    cath = float(t.loc["roman catholic"].max() - t.loc["roman catholic"].min())
    if cath < CATHOLIC_RANGE_MIN:
        raise SystemExit(f"Roman Catholic now moves only {100 * cath:.1f} points across rounds; "
                         "the reason for folding Catholics into Christian is weaker, decide again")
    return t


def unnamed_where_they_live(df, nm):
    """The share of Christians answering `Christian only`, averaged over where each church's own
    respondents are, against the national share. A drawn church must be at or under it; a folded
    one over it. Weighted throughout."""
    christian_keys = {k for k, v in GROUP.items() if v in ("Christian", *CHURCHES)}
    chr_ = df[df["k"].isin(christian_keys)]
    by = chr_.groupby("geo_id")["w"].sum()
    unnamed = chr_[chr_["k"] == "christian only"].groupby("geo_id")["w"].sum().reindex(by.index,
                                                                                         fill_value=0)
    frac = unnamed / by
    national = float(unnamed.sum() / by.sum())
    print(f"\n  `Christian only` as a share of Christians: national {100 * national:.1f}%; by unit "
          + ", ".join(f"{nm[u]} {100 * v:.0f}%" for u, v in frac.sort_values().items()))
    scores = {}
    for c in [x.lower() for x in CHURCHES] + FOLDED_CHURCHES:
        w = chr_[chr_["k"] == c].groupby("geo_id")["w"].sum()
        scores[c] = float((w * frac.reindex(w.index)).sum() / w.sum())
        print(f"    {c:<14} unnamed share where its respondents are: {100 * scores[c]:5.1f}%  "
              f"({'drawn' if c in [x.lower() for x in CHURCHES] else 'folded into Christian'})")
    wrong = [c for c in scores if (c in FOLDED_CHURCHES) == (scores[c] <= national)]
    if wrong:
        raise SystemExit(f"{wrong} are now on the other side of the national unnamed share "
                         f"({100 * national:.1f}%); CHURCHES and FOLDED_CHURCHES need deciding again")


def census_witness(df, frame_share, pop, nm):
    """The 2005 census: Tableau 5.8 nationally and the p.101 regional figures. Prints the survey
    beside them and asserts only what twenty years should not have moved."""
    import fitz

    if not os.path.exists(CENSUS_PDF):
        raise SystemExit(f"missing {CENSUS_PDF}; run with --fetch")
    doc = fitz.open(CENSUS_PDF)
    if doc.page_count < PROSE_PAGE:
        raise SystemExit("the census volume is short; a truncated download reads as fewer pages")
    t58 = " ".join(doc[CENSUS_PAGE - 1].get_text().split())
    if "Tableau 5. 8" not in t58:
        raise SystemExit(f"PDF page {CENSUS_PAGE} is not Tableau 5.8")
    for lab, v in CENSUS_T58.items():
        m = re.search(re.escape(lab) + r"\s+([\d.]+)", t58)
        if not m or float(m.group(1)) != v:
            raise SystemExit(f"Tableau 5.8 {lab}: transcribed {v}, the PDF has "
                             f"{m.group(1) if m else 'nothing'}")
    prose = re.sub(r"\s+", "", doc[PROSE_PAGE - 1].get_text())
    for (_what, _pc), (phrase, _v) in CENSUS_PROSE.items():
        if phrase not in prose:
            raise SystemExit(f"the p.101 figure {phrase!r} is not on PDF page {PROSE_PAGE}")
    print(f"\n  census 2005, Tableau 5.8 and the p.101 figures re-read from the PDF and equal")

    # the survey at the census's ten regions: each city merged back into its region by population
    reg = pd.Series({u: REGION_OF.get(u, u) for u in pop.index})
    raw = df.groupby(["geo_id", "k"])["w"].sum().unstack(fill_value=0.0)
    raw = raw.div(raw.sum(axis=1), axis=0)
    fam = pd.DataFrame({
        "Catholic": raw.get("roman catholic", 0.0),
        "Protestant": raw[[c for c in raw.columns if c in PROTESTANT_KEYS]].sum(axis=1),
        "Muslim": frame_share["Muslim"],
    })
    wr = fam.mul(pop, axis=0).groupby(reg).sum().div(pop.groupby(reg).sum(), axis=0)
    rname = {"CM002": "Centre", "CM005": "Littoral", **{u: nm[u] for u in wr.index
                                                         if u not in ("CM002", "CM005")}}
    print("  2005 census against the pooled survey at the census's regions (%):")
    print(f"    {'':<14}{'census':>8}{'survey':>8}")
    for (what, pc), (_phrase, v) in CENSUS_PROSE.items():
        print(f"    {what:<10} {rname[pc]:<14}{v:7.1f}{100 * wr.loc[pc, what]:8.1f}")
    print(f"    Catholic, survey, every region: " + ", ".join(
        f"{rname[u]} {100 * wr.loc[u, 'Catholic']:.1f}" for u in wr.sort_values("Catholic", ascending=False).index))

    top3 = set(wr["Muslim"].sort_values(ascending=False).index[:3])
    if top3 != {"CM001", "CM004", "CM006"}:
        raise SystemExit(f"the survey's three most Muslim regions are {sorted(top3)}, not the "
                         "census's Adamaoua, Extrême-Nord and Nord")
    if wr["Catholic"].idxmax() != "CM002":
        raise SystemExit("Centre is no longer the survey's most Catholic region, as in the census")
    for pc in ("CM007", "CM009"):
        if wr.loc[pc, "Protestant"] <= wr.loc[pc, "Catholic"]:
            raise SystemExit(f"{rname[pc]} is Protestant-led in the census and not in the survey")
    print("    asserted: the three northern regions are the most Muslim, Centre the most Catholic, "
          "Nord-Ouest and Sud Protestant-led")


def main():
    if "--fetch" in sys.argv:
        fetch()
        fetch_witnesses()
    dhs = dhs_witness()
    print("=== Afrobarometer, Cameroon ===")
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS, regroup=True, extra=["LOCATION.LEVEL.1"])
    print(f"\n  pooled: {len(raw):,} respondents with a religion answer over five rounds")
    raw["k"] = raw["category"].map(key)
    ct = pd.crosstab(raw["k"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print("\n  every answer as it arrives, keyed (this is what GROUP collapses):")
    print(ct.sort_values("all", ascending=False).to_string())

    report_card()
    unmapped = sorted(set(raw["k"]) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no category: {unmapped}; add them to GROUP deliberately")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = raw["k"].map(GROUP)
    ab.assert_one_wording(df, COUNTRY)
    levels_by_round(df)
    # The card's own `Other` box, not the grouped category (two Bahá'í answers in round 9 join it).
    other_late = int(((df["k"] == "other") & df["round"].isin(RECENT)).sum())
    if other_late:
        raise SystemExit(f"{other_late} respondents chose the Other box in rounds 8-9; NOT_PLACED's "
                         "reason no longer holds")

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} units, expected {N_UNITS}; re-run cm_geo.py")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    pop = lut.set_index("geo_id")["pop"].astype(float)
    if int(pop.sum()) != CODPS_2025:
        raise SystemExit(f"cm_lookup.csv sums to {int(pop.sum()):,}, not {CODPS_2025:,}")
    units = sorted(lut["geo_id"])

    df["geo_id"] = df["geo_raw"].map(gkey).map(NORM)
    if df["geo_id"].isna().any():
        raise SystemExit(f"REGION labels with no unit: "
                         f"{sorted(df.loc[df['geo_id'].isna(), 'geo_raw'].astype(str).unique())}")
    per_round = df.groupby("round")["geo_id"].nunique()
    print("\n  units present per round: " + ", ".join(f"R{r} {n}" for r, n in per_round.items()))
    if (per_round != N_UNITS).any():
        raise SystemExit("a pooled round does not sample all 12 units")
    clash = df.groupby(["round", "geo_raw"])["geo_id"].nunique()
    if (clash > 1).any():
        raise SystemExit("one REGION label names two units in a round")
    check_locations(df, nm)
    unnamed_where_they_live(df, nm)

    for rnd in ROUNDS:
        print(f"\n  R{rnd}:", end="")
        ab.held_out(df[df["round"] == rnd], pop, f"{COUNTRY} R{rnd}", pop_source="COD-PS 2025")
    ab.held_out(df, pop, COUNTRY, pop_source="COD-PS 2025")

    nat = ab.national(df).reindex(CATEGORIES).fillna(0.0)
    print(f"\n  the survey's national shares, pooled over R5-R9 (n={len(df):,}):")
    for c in CATEGORIES:
        print(f"    {c:<30}{nat[c]:8.3%}")
    rn = df.groupby(["round", "category"])["w"].sum().unstack(fill_value=0)
    print("\n  by round (survey weighting, %):")
    print((100 * rn.div(rn.sum(axis=1), axis=0)).round(1).reindex(columns=CATEGORIES).to_string())

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    passed, _table = cab.stability(dfw, CATEGORIES, units, f"{N_UNITS} units")
    carries = [c for c in passed if nat[c] >= ab.ELIGIBLE_FLOOR and c not in NOT_PLACED]
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {sorted(carries)}, not {sorted(CARRIES)}. "
                         "Read the table above, then edit CARRIES and the docstring deliberately.")
    stale = sorted(set(NOT_PLACED) - set(passed))
    if stale:
        raise SystemExit(f"NOT_PLACED names categories that no longer pass: {stale}")
    failing = [c for c in CATEGORIES if c not in passed and nat[c] >= ab.ELIGIBLE_FLOOR]
    if failing:
        raise SystemExit(f"categories over the 1% floor fail the split-half: {failing}; test them "
                         "for a standout unit (sources/tz.py::standouts) before drawing")

    # ---- compose ----
    frame, own, nraw, flat = ab.compose(df, nat, units, CATEGORIES, carries)
    if flat != TAIL_FLAT:
        raise SystemExit(f"the small-category rule now gives flat={flat}, against TAIL_FLAT="
                         f"{TAIL_FLAT}; read the multiples above and decide deliberately")
    counts = ab.round_within_rows(frame.mul(pop.reindex(units), axis=0))
    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a unit's drawn total is not its COD-PS population")
    drawn = counts.sum(axis=0) / counts.sum().sum()

    # ---- level: pooled against the recent rounds, both recomposed on COD-PS ----
    rec = df[df["round"].isin(RECENT)]
    rb = rec.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    rb = rb.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    recent = rb.div(rb.sum(axis=1), axis=0).mul(pop.reindex(units), axis=0).sum() / pop.sum()
    print("\n  national level: survey pool, as drawn, and rounds 8-9 alone recomposed the same way:")
    print(f"    {'category':<30}{'survey':>9}{'drawn':>9}{'R8-R9':>9}{'drawn-R8R9':>12}")
    for c in CATEGORIES:
        print(f"    {c:<30}{100 * nat[c]:8.2f}%{100 * drawn[c]:8.2f}%{100 * recent[c]:8.2f}%"
              f"{100 * (drawn[c] - recent[c]):+11.2f}")
    stale = [c for c in carries if abs(drawn[c] - recent[c]) > LEVEL_GAP_MAX]
    if stale:
        raise SystemExit(f"the pooled level differs from rounds 8-9 by more than "
                         f"{100 * LEVEL_GAP_MAX:.1f} points for {stale}; spec §12 (Norway)")

    share = counts.div(counts.sum(axis=1), axis=0)
    census_witness(df, share, pop.reindex(units), nm)
    print(f"    census 2005 nationally: Christian {sum(CENSUS_T58[k] for k in ('Catholique', 'Orthodoxe', 'Protestant', 'Autres chrétiens')):.1f}%, "
          f"Muslim {CENSUS_T58['Musulman']}%, animist {CENSUS_T58['Animiste']}%, free-thinker "
          f"{CENSUS_T58['Libre penseur']}%, other {CENSUS_T58['Autres religions']}%; drawn: "
          f"Christian {100 * sum(drawn[c] for c in ['Christian', *CHURCHES]):.1f}%, Muslim "
          f"{100 * drawn['Muslim']:.1f}%, traditional {100 * drawn['Traditional/ethnic religion']:.2f}%, "
          f"none {100 * drawn['None']:.1f}%, other {100 * drawn['Other']:.2f}%")

    # ---- carve the Catholic Church out of `Christian` (docstring; sources/na.md §3's rule) ----
    cath = carve_catholic(df, units, nm, dhs)
    frame8 = frame.copy()
    frame8["Catholic"] = frame["Christian"] * cath.reindex(units)
    frame8["Christian"] = frame["Christian"] - frame8["Catholic"]
    frame8 = frame8[OUT_CATEGORIES]
    if (frame8 < -1e-12).any().any() or (frame8.sum(axis=1) - 1).abs().max() > 1e-9:
        raise SystemExit("the eight shares are not a closed partition of every unit")
    counts7 = counts
    counts = ab.round_within_rows(frame8.mul(pop.reindex(units), axis=0))
    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a unit's drawn total is not its COD-PS population")
    if not ((counts["Christian"] + counts["Catholic"] - counts7["Christian"]).abs() <= 1).all():
        raise SystemExit("carving the Catholics out moved a unit's Christian total")
    all_chr = counts[["Christian", "Catholic", *CHURCHES]].sum().sum()
    cath_nat = float(counts["Catholic"].sum() / all_chr)
    ref = float(np.mean(list(dhs.values())))
    print(f"\n  Catholics as drawn: {100 * cath_nat:.1f}% of Christians and "
          f"{100 * counts['Catholic'].sum() / counts.sum().sum():.1f}% of everyone; DHS mean "
          f"{100 * ref:.1f}% of Christians")
    if abs(cath_nat - ref) > CATH_DRAWN_GAP_MAX:
        raise SystemExit(f"the drawn Catholic share of Christians is more than "
                         f"{100 * CATH_DRAWN_GAP_MAX:.0f} points from the DHS mean")
    share = counts.div(counts.sum(axis=1), axis=0)
    diocese_witness(share, pop.reindex(units), nm)

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    n56 = df[df["round"].isin(CATHOLIC_ROUNDS) & (df["category"] == "Christian")] \
        .groupby("geo_id").size()
    basis_note = {c: ("the unit's own measured share" if c in carries else
                      "the national share" if flat else
                      "the national proportion within the unit's remainder") for c in CATEGORIES}
    basis_note["Catholic"] = ("the unit's own Christian share (less Presbyterians and Baptists) "
                              "times its Catholics' share of it in rounds 5-6")
    basis_note["Christian"] = ("the unit's own measured share of Christians other than "
                               "Presbyterians and Baptists, less the Catholics")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("COD-PS 2025 population composed with the unit's own mix from Afrobarometer "
                   f"rounds 5-9 pooled (n={int(n_by[r.geo_id])} here); "
                   f"{basis_note[r.source_category]}"
                   + (f" ({int(n56[r.geo_id])} pool respondents here in rounds 5-6)"
                      if r.source_category == "Catholic" else "")), axis=1)
    total = int(out["count"].sum())
    if total != CODPS_2025:
        raise SystemExit(f"drawn {total:,} against COD-PS {CODPS_2025:,}")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    for c, n in counts.sum(axis=0).sort_values(ascending=False).items():
        print(f"    {100 * n / total:6.2f}%  {c}  ({n:,})")
    print("\n  as drawn, by unit, most Muslim first (pooled n in brackets):")
    print(f"    {'':<20}" + "".join(f"{c[:7]:>8}" for c in OUT_CATEGORIES))
    for u in share.sort_values("Muslim", ascending=False).index:
        s = share.loc[u]
        print(f"    {nm[u]:<20}" + "".join(f"{100 * s[c]:7.1f}%" for c in OUT_CATEGORIES)
              + f"{int(pop[u]):>12,}  n={int(n_by[u])}")
    zero = [(nm[u], c) for u in units for c in carries if counts.loc[u, c] == 0]
    print(f"  drawn at zero in a carried category (no pooled respondent gave it): {zero or 'none'}")
    print(f"  thinnest unit {nm[n_by.idxmin()]} n={int(n_by.min())}; median n={int(n_by.median())}; "
          f"total n={int(n_by.sum()):,}")


if __name__ == "__main__":
    main()
