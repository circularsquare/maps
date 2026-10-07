"""Lebanon: Lebanese by pooled survey first / home language per mohafaza, Syrians and Palestinians
by OCHA's 2026 caza counts, on the 26 cazas -> data/normalized/lb.csv (node ids).

    python sources/lb_build.py

NO CENSUS SINCE 1932. The population base is the one religiondots uses for Lebanon (its
sources/lb.py): OCHA Lebanon's 2026 LRP population package (HDX), Lebanese 3,864,296 (CAS's
2018-19 labour force survey, carried forward), Syrians 1,120,000, Palestinian refugees 224,791
and migrant workers 164,097, each by caza. Read from religiondots' copy, read-only. Here the
Lebanese are drawn where OCHA says they LIVE, not where the register files them.

LEBANESE (and anyone the surveys reached): first or home language, pooled per mohafaza (six,
the pre-2014 ones every round shares), one respondent one vote:
  Arab Barometer II   (2010-11) q10191 first language   1,387 (South left out, below)
  Arab Barometer III  (2012-13) q1019_1                 1,200
  Arab Barometer IV   (2016)    q1019a                  1,500
  World Values Survey 7 (2018)  Q272 language at home   1,200 (religiondots' .dta, read-only)
  Arab Barometer VII  (2021-22) Q1012B ethnic group: a check only (Armenian 41 of 2,379).
Arab Barometer II's South: 11 of its 179 answered English, against 0 English in the South in
the other three rounds (0 of 408); a code slip, so that round's South is left out (Iraq's WVS 4
Kirkuk rule).

FROM MOHAFAZA TO CAZA. Each language's mohafaza count is spread over its cazas: Armenian by the
caza's share of the mohafaza's Armenian Orthodox and Armenian Catholic register (religiondots'
data/normalized/lb.csv, 2022 register carried to cazas), since Armenians live in a few places
(Bourj Hammoud in El Meten, Beirut, Anjar in Zahle) and their families are registered there;
every other non-Arabic answer by OCHA's resident Lebanese; Arabic takes the rest of each caza.
That moves people only inside the mohafaza the survey counted them in (AGENT_BRIEF section 4).

SYRIANS, PALESTINIANS: OCHA's caza counts, on Levantine Arabic (Syria is not drawn on this map;
sa_census.OVERRIDE puts Syrians and Palestinians there). Syrian Kurds are not split out: no
source counts them in Lebanon. MIGRANTS (164,097): not drawn, as religiondots; nothing gives
their nationalities by caza.

CHECKS: OCHA's caza columns sum to its national figures; register Armenians exist in every
mohafaza that has an Armenian answer; output sums to Lebanese + Syrians + Palestinians.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE), str(HERE / "sources")]
from rdlink import RD  # noqa: E402
import ab_firstlang as ab  # noqa: E402

OUT = HERE / "data" / "normalized" / "lb.csv"
OCHA_XLSX = RD / "data" / "raw" / "lb" / "05.-2026-lrp-population-package.xlsx"
WVS7 = RD / "data" / "raw" / "lb" / "WVS_Wave_7_Lebanon_Stata_v5.1.dta"
RD_LB = RD / "data" / "normalized" / "lb.csv"
OCHA = dict(lebanese=3_864_296, syrian=1_120_000, palestinian=224_791, migrant=164_097)
DRAWN_TOTAL = OCHA["lebanese"] + OCHA["syrian"] + OCHA["palestinian"]

PCODE = {
    "Beirut": "LB11", "Baalbek": "LB21", "El Hermel": "LB22", "Rachaya": "LB23",
    "West Bekaa": "LB24", "Zahle": "LB25", "Aley": "LB31", "Baabda": "LB32", "Chouf": "LB33",
    "Jbeil": "LB34", "Kesrwane": "LB35", "El Meten": "LB36", "Bent Jbeil": "LB41",
    "Hasbaya": "LB42", "Marjaayoun": "LB43", "El Nabatieh": "LB44", "Akkar": "LB51",
    "El Batroun": "LB52", "Bcharre": "LB53", "El Koura": "LB54", "El Minieh-Dennie": "LB55",
    "Tripoli": "LB56", "Zgharta": "LB57", "Saida": "LB61", "Jezzine": "LB62", "Sour": "LB63",
}
# the six pre-2014 mohafazat, by pcode prefix (Akkar in North, Baalbek-Hermel in Bekaa)
MOH = {"LB1": "Beirut", "LB2": "Bekaa", "LB3": "Mount Lebanon", "LB4": "Nabatieh",
       "LB5": "North", "LB6": "South"}
REGION = {
    "4501. Beirut": "Beirut", "4502. Mount Lebanon": "Mount Lebanon", "4503. North": "North",
    "4504. Bekaa": "Bekaa", "4505. South": "South", "4506. Nabataean": "Nabatieh",
    "Beirut": "Beirut", "Beqaa": "Bekaa", "Bekaa": "Bekaa", "Mount Lebanon": "Mount Lebanon",
    "Nabtieh": "Nabatieh", "El Nabatieh": "Nabatieh", "Northern": "North", "North": "North",
    "Southern": "South", "South": "South",
    "LB: Beirut": "Beirut", "LB: Bekaa": "Bekaa", "LB: Mount lebanon": "Mount Lebanon",
    "LB: El Nabatieh": "Nabatieh", "LB: North": "North", "LB: South": "South",
    "AKKAR": "North", "BAALBEK-EL HERMEL": "Bekaa", "BEIRUT": "Beirut", "BEKAA": "Bekaa",
    "EL NABATIYEH": "Nabatieh", "MOUNT LEBANON": "Mount Lebanon", "NORTH": "North",
    "SOUTH": "South",
}
AB_N = {"ABII": 1387, "ABIII": 1200, "ABIV": 1500, "ABVII": 2399}

LEVANTINE = "afroasiatic.levantine_arabic"
ARMENIAN = "indoeuropean.armenian.armenian"
ANSWER = {
    "1. Arabic": LEVANTINE, "Arabic": LEVANTINE,
    "17. Armenian": ARMENIAN, "Armenian": ARMENIAN, "Armenian; Hayeren": ARMENIAN,
    "2. English": "indoeuropean.germanic.english", "English": "indoeuropean.germanic.english",
    "3. French": "indoeuropean.romance.french", "French": "indoeuropean.romance.french",
    "Spanish": "indoeuropean.romance.spanish", "Ukrainian": "indoeuropean.slavic.east.ukrainian",
    "Kurdish": "indoeuropean.iranian.kurdish", "Syriac": "afroasiatic.syriac",
}


def read_ocha():
    """{pcode: {lebanese, syrian, palestinian, migrant}}, as religiondots' sources/lb.py."""
    import openpyxl
    wb = openpyxl.load_workbook(OCHA_XLSX, read_only=True, data_only=True)
    rows = list(wb["ALL POPULATION SUMMARY"].iter_rows(values_only=True))
    hdr = [i for i, r in enumerate(rows) if r[0] == "Governorate" and r[1] == "District"]
    if len(hdr) != 1:
        raise SystemExit("OCHA package: the district header row moved")
    h = [str(x or "") for x in rows[hdr[0]]]
    col = {}
    for key, pat in (("lebanese", "TOTAL LEBANESE"), ("palestinian", "TOTAL PALESTINIANS"),
                     ("syrian", "TOTAL SYRIANS"), ("migrant", "Migrants")):
        hits = [i for i, x in enumerate(h) if x.startswith(pat) and "2026" in x]
        if len(hits) != 1:
            raise SystemExit(f"OCHA package: column {pat} (2026) found {len(hits)} times")
        col[key] = hits[0]
    out = {}
    for r in rows[hdr[0] + 1:]:
        if r[1] is None or r[0] is None:
            break
        out[PCODE[r[1]]] = {k: float(r[c] or 0) for k, c in col.items()}
    if len(out) != 26:
        raise SystemExit(f"OCHA: {len(out)} cazas")
    for k, v in OCHA.items():
        if abs(sum(c[k] for c in out.values()) - v) > 0.5:
            raise SystemExit(f"OCHA {k}: cazas do not sum to {v:,}")
    return out


def int_split(total, weights):
    """an integer total over keys in proportion to weights, largest remainder"""
    out = {k: 0 for k in weights}
    if total:
        out.update(ab.shares_to_counts(weights, total))
    return out


def main():
    import pyreadstat
    ocha = read_ocha()
    cazas = sorted(ocha)
    moh_of = {c: MOH[c[:3]] for c in cazas}

    # Lebanese per caza, integers summing to OCHA's national figure
    leb = int_split(OCHA["lebanese"], {c: ocha[c]["lebanese"] for c in cazas})
    syr = int_split(OCHA["syrian"], {c: ocha[c]["syrian"] for c in cazas})
    pal = int_split(OCHA["palestinian"], {c: ocha[c]["palestinian"] for c in cazas})

    # survey pool per mohafaza
    pooled = {m: {} for m in MOH.values()}

    def add(m, a, n):
        if a not in ANSWER:
            raise SystemExit(f"answer {a!r} not mapped")
        pooled[m][ANSWER[a]] = pooled[m].get(ANSWER[a], 0) + n
    for wave in ("ABII", "ABIII", "ABIV"):
        for m, v in ab.answers("Lebanon", wave, REGION, AB_N[wave]).items():
            if wave == "ABII" and m == "South":
                print(f"  Arab Barometer II South left out: {v}")
                continue
            for a, n in v.items():
                add(m, a, n)
    df, _ = pyreadstat.read_dta(str(WVS7), apply_value_formats=True,
                                usecols=["N_REGION_WVS", "Q272"])
    if len(df) != 1200:
        raise SystemExit(f"WVS 7 Lebanon: {len(df)} rows")
    for g, a in zip(df["N_REGION_WVS"].astype(str), df["Q272"].astype(str)):
        if a in ab.NOT_DRAWN:
            continue
        add(REGION[g], a, 1)
    eth = ab.answers("Lebanon", "ABVII", REGION, AB_N["ABVII"])

    # register Armenians per caza (religiondots, read-only), for placement inside a mohafaza
    arm_reg = {c: 0 for c in cazas}
    with open(RD_LB, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_category"] in ("Lebanese, Armenian Orthodox", "Lebanese, Armenian Catholic"):
                arm_reg[r["geo_id"]] += int(r["count"])

    rows = []
    for m in MOH.values():
        cs = [c for c in cazas if moh_of[c] == m]
        lebm = sum(leb[c] for c in cs)
        v = pooled[m]
        tot = sum(v.values())
        cnt_m = ab.shares_to_counts(v, lebm)
        e = eth.get(m, {})
        print(f"  {m:<14} Lebanese {lebm:>9,}  n {tot:>4}: " + ", ".join(
            f"{k.split('.')[-1]} {n / tot:.2%}" for k, n in sorted(v.items(), key=lambda kv: -kv[1]))
            + f"   [AB VII ethnic Armenian {e.get('Armenian', 0)} of {sum(e.values())}]")
        per = {c: {} for c in cs}
        for node, n in cnt_m.items():
            if node == LEVANTINE:
                continue
            if node == ARMENIAN:
                w = {c: arm_reg[c] for c in cs}
                if not sum(w.values()):
                    raise SystemExit(f"{m}: Armenian answers but no register Armenians")
            else:
                w = {c: leb[c] for c in cs}
            for c, k in int_split(n, w).items():
                per[c][node] = k
        for c in cs:
            rest = leb[c] - sum(per[c].values())
            if rest < 0:
                raise SystemExit(f"{c}: minority languages exceed its Lebanese")
            per[c][LEVANTINE] = rest
            for node, k in per[c].items():
                if k:
                    rows.append(dict(geo_id=c, geo_level="caza", geo_name=c, source_category=node,
                                     count=k, tier="modelled", source_id="Lebanese", year=2026,
                                     note=f"Lebanese (OCHA 2026 residents) at {m}'s pooled "
                                          "survey answers"))
            for who, d in (("Syrian", syr), ("Palestinian", pal)):
                if d[c]:
                    rows.append(dict(geo_id=c, geo_level="caza", geo_name=c,
                                     source_category=LEVANTINE, count=d[c], tier="derived",
                                     source_id=who, year=2026, note=f"{who}s, OCHA 2026"))
    ab.report(rows, DRAWN_TOTAL)
    ab.write_csv(OUT, rows)
    print(f"  wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
