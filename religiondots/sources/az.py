"""Azerbaijan: no census or survey asks religion in a way that places anyone, so the map is an
ETHNICITY MODEL. Nationality by unit from the census; Islam for every group the surveys find Muslim;
the Christian and Jewish groups by their own religion.

Reads data/geo/az/az_lookup.csv (`sources/az_geo.py`), the 2009 census's nationality by unit, the
2019 census's nationality for the country, and Kazakhstan's 2021 census religion by nationality
(`sources/kz_model.py`); writes data/normalized/az.csv. `sources/az.md` is the record; the rulings
are `ask/RULINGS.md` 2026-09-15 (Azerbaijan is a priority hole) and 2026-09-16 (draw a country with
no religion question on the best survey or compiler figure, with the method disclosed).

## WHAT ASKS, AND WHAT IT SAYS

Nothing asks religion with a place finer than the nation and a minority in it. The 2009 and 2019
census forms ask nationality and mother tongue, not religion (`sources.md` §11ao). Three surveys
with a place code ask religion, and none of them sees a non-Muslim:
  * EBRD LiTS III (2016), 1,510 adults in 75 PSUs over 8 of the 10 economic regions then held:
    1,510 MUSLIM, including all three Russians (`data/raw/lits/lits_iii.dta`, measured here). At the
    census's 0.7% Russian, about ten Orthodox answers were expected; zero is the instrument
    (`playbooks/lits.md`, "a zero where a known community lives").
  * DHS 2006 (report FR195, Table 3.1.1): women 15-49, 99.2% Muslim, 0.7% "Christian/no
    religion/other"; national only in the report, and the data file is behind a DHS account.
  * EVS 2017: 99.4% of those naming a religion are Muslim, and it is the source of Pew Research
    Center's 2020 estimate (Pew 2025, Appendix A); behind a GESIS account.
So the surveys settle the majority (every Azerbaijani, Lezgin, Talysh and other Muslim-heritage
respondent answers Muslim) and say nothing about the minorities, which is where the census's
nationality table takes over.

## THE MODEL

For each unit: the 2019 census's existing population (`az_geo.py`). Inside it, eight groups are
placed and given a religion; everyone else is drawn on Islam.

  group                2019 count   placed by                       religion
  Russians                 71,046   2009 nationality by unit        Kazakhstan 2021's Russians
  Ukrainians               13,947   "                               Kazakhstan 2021's Ukrainians
  Tatars                   17,712   "                               Kazakhstan 2021's Tatars
  Georgians                 8,442   2009 Georgians                  Georgian Orthodox
  Ingiloys                  1,817   2009 Georgians                  not known
  Udins                     3,540   2009 Udins                      Christian (the Udi church)
  Jews                      5,094   2009 Jews                       Judaism
  Armenians                   178   2009 Armenians outside Karabakh Armenian Apostolic
  other nationalities       5,039   2009 "other"                    not known

Each group's 2019 national count is spread over the units in proportion to its 2009 count there
(the 2019 census prints nationality for the country only; Volume B Table 28). Units with no
existing population in 2019 take nobody; for Armenians, the four partly held units (Aghdam,
Fuzuli, Tartar, Jabrayil) are left out as well, because their 2009 Armenians were the Karabakh
Armenians the 2009 census estimated and the 2019 census did not count.

**Russians, Ukrainians and Tatars take Kazakhstan's own census coefficients** (`kz_model.py`, the
2021 census volume's religion-by-nationality table), with refusals and the three smallest answers
(Catholic, Protestant, Judaism, Buddhism, other; 0.6% of Russians together) taken out and the rest
renormalised: Russians 92.7% Orthodox, 2.1% Muslim, 5.1% non-believers. This is spec §14.24's construction (a coefficient from this map's own
countries) and §14.12's good case: the Orthodox share is ancestry-shaped. The non-believer share is
attitude-shaped and is the weakest cell drawn. Kazakhstan's figures for Azerbaijanis themselves
(27% refused, 5% non-believers) are NOT used: Azerbaijan's own surveys answer for its majority.

Georgians, Udins, Jews and Armenians are religio-ethnic in Azerbaijan (spec §14.5): the census
already separates Muslim Ingiloys from Christian Georgians, Muslim Udins have been counted as
Azerbaijanis since the nineteenth century, and the Mountain Jews are a religious community by
definition. Ingiloys and "other nationalities" are drawn on `unknown`.

## WHAT IS NOT DRAWN

Pew's 2020 estimate, from EVS 2017, is 94.73% Muslim, 4.76% unaffiliated (484,645), 0.42%
Christian (42,730) and 0.08% Jewish (8,580). Its unaffiliated cell is EVS's "belong to no
denomination", against DHS's 0.7% for Christians, the unaffiliated and others together and LiTS's
zero; no source places it, so it is not drawn and is named in the note. The model draws more
Christians than Pew (the census counts the Russians; a survey of 1,800 meets about a dozen) and
fewer Jews (the census's 5,094 against Pew's 8,580).

No Sunni/Shia split: Pew's 2011 survey (37% Shia, 16% Sunni, 45% "just a Muslim") and CRRC 2012 are
national, the om/sa ruling (2026-09-15) applies, and an ethnicity-derived Sunni layer for the
Lezgins, Avars and Tsakhurs would mark the north's minorities and leave its Sunni Azerbaijanis
unmarked.

Usage:
    python sources/az.py            rebuild data/normalized/az.csv and print the witnesses
"""

import html
import io
import os
import re
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "az")
LOOKUP = os.path.join(ROOT, "data", "geo", "az", "az_lookup.csv")
ETH2009 = os.path.join(RAW, "mashke_azerbaijan_ethnic2009.htm")
ETH2009_URL = "https://pop-stat.mashke.org/azerbaijan-ethnic2009.htm"
T111 = os.path.join(RAW, "001_11-12en.xls")
T111_URL = "https://www.stat.gov.az/source/demoqraphy/en/001_11-12en.xls"
T117 = os.path.join(RAW, "001_17en.xls")
VOL_B = os.path.join(RAW, "Siyahiyaalinma-2019, Cild B.pdf")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "az.csv")

# 2019 census, Volume B Table 28 (printed pp.333-334), the country's nationality counts.
NAT2019 = {"Azerbaijani": 9_436_123, "Lezgi": 167_570, "Talish": 87_578, "Russian": 71_046,
           "Ukrainian": 13_947, "Avar": 48_636, "Turkish": 30_516, "Tat": 27_657, "Sakhur": 13_361,
           "Georgian": 8_442, "Ingiloy": 1_817, "Kurd": 4_105, "Tatarian": 17_712, "Griz": 2_076,
           "Jews": 5_094, "Udin": 3_540, "Khinalig": 3_466, "Budug": 1_044, "Armenian": 178,
           "Khaput": 2_462, "Other nations": 5_039}
NAT2019_TOTAL = 9_951_409
# The 2019 label -> table 1.11's label, for the witness.
T111_NAME = {"Azerbaijani": "Azerbaijanis", "Lezgi": "Lezgins", "Talish": "Talysh",
             "Russian": "Russians", "Ukrainian": "Ukrainians", "Avar": "Avars", "Turkish": "Turks",
             "Tat": "Tats", "Sakhur": "Tsakhurs", "Georgian": "Georgians", "Ingiloy": "Ingiloys",
             "Kurd": "Kurds", "Tatarian": "Tatars", "Griz": "Grysz", "Jews": "Jews", "Udin": "Udins",
             "Khinalig": "Khynalygs", "Budug": "Buduqlus", "Armenian": "Armenians",
             "Khaput": "Haputs", "Other nations": "other nationalities"}

# The modelled groups: 2019 label -> (2009 column that places it, religion rule).
GROUPS = {
    "Russian":       ("Russians",   "kz:Орыстар"),
    "Ukrainian":     ("Ukrainians", "kz:Украиндар"),
    "Tatarian":      ("Tatars",     "kz:Татарлар"),
    "Georgian":      ("Georgians",  "Georgian Orthodox"),
    "Ingiloy":       ("Georgians",  "Ingiloy, religion not known"),
    "Udin":          ("Udins",      "Udi Christian"),
    "Jews":          ("Jews",       "Jewish"),
    "Armenian":      ("Armenians",  "Armenian Apostolic"),
    "Other nations": ("Other",      "Other nationality, religion not known"),
}
# Kazakhstan 2021's religion columns kept, and the category each becomes here.
KZ_KEEP = {"Православие": "Orthodox (Kazakhstan's coefficient)",
           "Ислам": "Muslim",
           "Неверующие": "Non-believer (Kazakhstan's coefficient)"}
PARTLY_HELD = {"Aghdam", "Fuzuli", "Tartar", "Jabrayil"}     # COD-AB names; for Armenians only

# Measured 2026-10-03 and asserted, so note_public cannot drift from the data.
NOTE = dict(people=9_943_958, christians=95_717, orthodox=83_549, muslims=9_830_318,
            jews=5_099, unaffiliated=5_975, unknown=6_849, pew_unaffiliated=484_645,
            pew_christians=42_730, pew_jews=8_580)


def fold(s):
    s = str(s).replace("ə", "e").replace("Ə", "E")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("ı", "i")
    return re.sub(r"[^a-z]", "", s)


def read_2009():
    """2009 nationality by unit: DataFrame indexed by the Azerbaijani unit name, English columns.

    Tim Bespyatov's transcription of *Azərbaycan Respublikası əhalisinin siyahıyaalınması 2009-cu
    il, XIX cild* (State Statistical Committee, 2011); the volume itself is not online. Every unit
    total is checked against the committee's table 1.17 and every national column against 1.11.
    """
    if not os.path.exists(ETH2009):
        import urllib.request
        req = urllib.request.Request(ETH2009_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            open(ETH2009, "wb").write(r.read())
    h = open(ETH2009, encoding="utf-8-sig").read()
    rows = []
    for r in re.split(r"(?i)<tr[^>]*>", h)[1:]:
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.split(r"(?i)<t[dh][^>]*>", r)[1:]]
        rows.append(cells)
    eng = next(r for r in rows if len(r) > 2 and r[1] == "Total")
    cols = ["unit"] + eng[1:]
    body = [r for r in rows if len(r) == len(cols) and r[0] and not r[0].startswith("-")
            and r[1] not in ("Cəmi", "Total")]
    df = pd.DataFrame(body, columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c].str.replace("-", "0"), errors="raise").astype(int)
    df["unit"] = df["unit"].str.replace(r"\s+ş\.$", "", regex=True).str.strip()
    df = df.set_index("unit")
    other = df.drop(columns=["Total"]).sum(axis=1)
    if not (other == df["Total"]).all():
        raise SystemExit(f"2009 rows whose groups do not sum to Total: {list(df.index[other != df['Total']])}")
    return df


def kz_coefficients():
    """{kz nationality: {category: share}} from Kazakhstan 2021, over the kept columns only."""
    import kz_model

    coef = kz_model.read_coefficients()["total"]
    out = {}
    for nat in ("Орыстар", "Украиндар", "Татарлар"):
        row = coef[nat]
        kept = {KZ_KEEP[k]: float(row[k]) for k in KZ_KEEP}
        tot = sum(kept.values())
        dropped = float(row["total"]) - tot
        out[nat] = {k: v / tot for k, v in kept.items()}
        print(f"  Kazakhstan 2021, {nat}: {int(row['total']):,}; kept {int(tot):,} "
              f"({tot / float(row['total']):.1%}; refused, Catholic, Protestant, Judaism, Buddhism "
              f"and other dropped); " + ", ".join(f"{k} {v:.2%}" for k, v in out[nat].items()))
    return out


def witnesses_national(e09):
    t = pd.read_excel(T111, header=None)
    lab = t[1].astype(str).str.strip()
    s09, s19 = {}, {}
    for i in range(7, 28):
        s09[lab[i]] = t.iloc[i, 9]
        s19[lab[i]] = t.iloc[i, 10]
    bad = []
    for k, v in NAT2019.items():
        ref = s19[T111_NAME[k]]
        # 0.1 thousand: table 1.11 prints Budukhs at 1.1 against the volume's 1,044, as table
        # 1.17 differs from Table 3 by up to 76 people (az_geo.T117_TOL); a wrong row is thousands off.
        if abs(float(ref) * 1000 - v) > 100:
            bad.append((k, v, ref))
    if sum(NAT2019.values()) != NAT2019_TOTAL:
        bad.append(("sum", sum(NAT2019.values()), NAT2019_TOTAL))
    nat = e09.loc["Azərbaycan"]
    for col, name in (("Russians", "Russians"), ("Ukrainians", "Ukrainians"), ("Tatars", "Tatars"),
                      ("Georgians", "Georgians"), ("Udins", "Udins"), ("Jews", "Jews"),
                      ("Armenians", "Armenians"), ("Azerbaijanis", "Azerbaijanis"),
                      ("Lezgins", "Lezgins")):
        if abs(float(s09[name]) * 1000 - nat[col]) > 50.01:
            bad.append(("2009 " + col, int(nat[col]), s09[name]))
    if bad:
        raise SystemExit(f"nationality witnesses against table 1.11 fail: {bad}")
    print("  witness: the 2019 counts and the 2009 transcription's national row equal table 1.11")


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 74:
        raise SystemExit(f"{LOOKUP}: {len(lut)} units; re-run az_geo.py")
    e09 = read_2009()
    witnesses_national(e09)

    # join 2009 units to COD-AB on the Azerbaijani name; the witness is table 1.17's 2009 column
    key = {fold(n): u for n, u in zip(lut["name_az"], lut["geo_id"])}
    units09 = [n for n in e09.index if n not in ("Azərbaycan", "Naxçıvan MR")]
    j = {}
    for n in units09:
        k = fold(n)
        if k not in key:
            raise SystemExit(f"2009 unit {n!r} matches no COD-AB name")
        j[n] = key[k]
    if len(set(j.values())) != 74 or len(j) != 74:
        raise SystemExit(f"2009 units join {len(j)} rows onto {len(set(j.values()))} COD-AB units, not 74")
    t117 = pd.read_excel(T117, header=None)
    eng_2009 = {}
    for _i, r in t117.iloc[6:134].iterrows():
        nm = str(r[1]).strip()
        v = pd.to_numeric(r[5], errors="coerce")          # Pirallahi prints an ellipsis for 2009
        if nm != "nan" and pd.notna(v) and "economic region" not in nm and "including" not in nm:
            eng_2009[fold(re.sub(r"\s+(district|city)(\s+-\s+total)?$", "", nm))] = float(v)
    from az_geo import ALIAS, T117_ALIAS
    name_en = dict(zip(lut["geo_id"], lut["name"]))
    inv_alias = {v: k for k, v in ALIAS.items()}
    bad = []
    for n, u in j.items():
        en = inv_alias.get(name_en[u], name_en[u])
        en = T117_ALIAS.get(en, en)
        ref = eng_2009.get(fold(en))
        if ref is None or abs(ref * 1000 - e09.loc[n, "Total"]) > 100:
            bad.append((n, name_en[u], int(e09.loc[n, "Total"]), ref))
    if bad:
        raise SystemExit(f"2009 unit totals against table 1.17's 2009 column: {bad}")
    print(f"  witness: all 74 units' 2009 totals equal table 1.17's 2009 column, joined on the "
          "Azerbaijani name")

    e = e09.loc[units09].copy()
    e.index = [j[n] for n in e.index]
    pop = dict(zip(lut["geo_id"], lut["pop"]))
    name = dict(zip(lut["geo_id"], lut["name"]))
    kz = kz_coefficients()

    rel = {"Georgian Orthodox": {"Georgian Orthodox": 1.0},
           "Ingiloy, religion not known": {"Ingiloy, religion not known": 1.0},
           "Udi Christian": {"Udi Christian": 1.0},
           "Jewish": {"Jewish": 1.0},
           "Armenian Apostolic": {"Armenian Apostolic": 1.0},
           "Other nationality, religion not known": {"Other nationality, religion not known": 1.0}}
    rows = []
    placed = {u: 0.0 for u in pop}
    for grp, (col09, rule) in GROUPS.items():
        w = e[col09].astype(float).copy()
        w[[u for u in w.index if pop[u] == 0]] = 0.0
        if grp == "Armenian":
            w[[u for u in w.index if name[u] in PARTLY_HELD]] = 0.0
        if w.sum() <= 0:
            raise SystemExit(f"{grp}: no 2009 weight in any populated unit")
        n_u = NAT2019[grp] * w / w.sum()
        shares = kz[rule[3:]] if rule.startswith("kz:") else rel[rule]
        for u, n in n_u.items():
            if n <= 0:
                continue
            placed[u] += n
            for cat, s in shares.items():
                rows.append((u, cat, n * s))
    over = [name[u] for u in pop if placed[u] > pop[u]]
    if over:
        raise SystemExit(f"units where the placed groups exceed the population: {over}")
    for u in pop:
        if pop[u] > 0:
            rows.append((u, "Muslim", pop[u] - placed[u]))
    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    df = df.groupby(["geo_id", "source_category"], as_index=False)["count"].sum()

    # integer counts that keep each unit's total exact (largest remainder within a unit)
    out = []
    for u, g in df.groupby("geo_id"):
        fl = g["count"].astype(float)
        base = fl.astype(int)
        short = int(round(pop[u] - base.sum()))
        order = (fl - base).sort_values(ascending=False).index[:short]
        base.loc[order] += 1
        g = g.assign(count=base)
        if int(g["count"].sum()) != pop[u]:
            raise SystemExit(f"{u}: rounded to {int(g['count'].sum())}, not {pop[u]}")
        out.append(g)
    df = pd.concat(out)
    df = df[df["count"] > 0]
    df["geo_level"] = "unit"
    df["geo_name"] = df["geo_id"].map(name)
    df["basis"] = "model"
    df["year"] = 2019
    df["source_id"] = "az_census2019_ethnicity_model"
    df["note"] = ("2019 census existing population; groups placed on the 2009 census's nationality "
                  "by unit; religion within group per sources/az.py")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year", "source_id",
        "note"]].to_csv(OUT, index=False, encoding="utf-8")

    tot = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {df['geo_id'].nunique()} units, {int(df['count'].sum()):,} people")
    for k, v in tot.items():
        print(f"    {k:<45} {v:>10,}  {v / df['count'].sum():.3%}")
    print("\n  where the non-Muslims are (largest 12 units by people not on Islam):")
    nm = df[df["source_category"] != "Muslim"].groupby("geo_id")["count"].sum()
    for u, v in nm.sort_values(ascending=False).head(12).items():
        print(f"    {name[u]:<14} {v:>8,}  {v / pop[u]:.2%} of {pop[u]:,}")

    with zipfile.ZipFile(PEW) as z:
        nm_ = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(nm_)), thousands=",")
    p = t[(t["Country"] == "Azerbaijan") & (t["Year"] == 2020)].iloc[0]
    print(f"\n  Pew 2020 (EVS 2017): {int(p['Population']):,}; Muslims {p['Muslims'] / p['Population']:.2%}, "
          f"Christians {int(p['Christians']):,}, Jews {int(p['Jews']):,}, unaffiliated "
          f"{int(p['Religiously_unaffiliated']):,}")
    chris = int(tot[[k for k in tot.index if k in ("Orthodox (Kazakhstan's coefficient)",
                                                    "Georgian Orthodox", "Udi Christian",
                                                    "Armenian Apostolic")]].sum())
    got = dict(people=int(df["count"].sum()), christians=chris,
               orthodox=int(tot.get("Orthodox (Kazakhstan's coefficient)", 0)),
               muslims=int(tot["Muslim"]),
               jews=int(tot["Jewish"]),
               unaffiliated=int(tot.get("Non-believer (Kazakhstan's coefficient)", 0)),
               unknown=int(tot.get("Ingiloy, religion not known", 0)
                           + tot.get("Other nationality, religion not known", 0)),
               pew_unaffiliated=int(p["Religiously_unaffiliated"]),
               pew_christians=int(p["Christians"]), pew_jews=int(p["Jews"]))
    print(f"  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
