"""Brunei: census race by district, split into languages with published speaker estimates.

Writes data/normalized/bn.csv (node ids straight into `source_category`; taxonomy/bn2021.py is the
identity). The record is sources/bn.md.

THE BASE. BPP 2021 (Department of Economic Planning and Statistics), Annex A, Table A3,
"Population by Race, District and Sex", persons: Malays / Chinese / Others for the four districts,
440,715 people, read from religiondots' pinned copy of the tables workbook (read-only; religiondots'
sources/bn.py fetched and checked it). A1 (residential status by district) is read beside it.
Every district's rows sum to its A3 total (asserted). The census asked home language (E27) but
DEPS never tabulated it (sources/bn.md section 1), so this is the ethnicity route of AGENT_BRIEF
section 2 plus the estimate route (ask 019).

THE SPLIT, per district:
  Malays (the census's "Malay" holds the seven puak jati):
    Tutong   17,000 speakers (Ethnologue 18th ed., 2006 figure, via Wikipedia "Tutong language"),
             spread by the Tutong people's district shares on Wikipedia "Tutong people"
             (Tutong 64.7%, Brunei-Muara 24.5%, Belait 10.5%, Temburong 0.2%)
    Kedayan  30,000 speakers (Dewan Bahasa dan Pustaka Brunei 2006, via Wikipedia "Kedayan"),
             over Brunei-Muara, Tutong and Belait in proportion to their Malays (no district
             figure is published; Temburong left out, no Kedayan settlement there is reported)
    Dusun    10,000, the low end of the 10,000-20,000 range on Wikipedia "Dusun people (Brunei)",
             as speakers: the ethnic share is larger (Minority Rights Group 2018: 6.3%) but the
             language's vitality is 2 on a 0-6 scale (Noor Azam and Siti Ajeerah 2016) and the
             young often do not speak it. Two thirds Tutong, one third Belait ("a majority in
             Tutong District, some in several areas of Belait", UBD's Pronunciation of Dusun).
             Bisaya, a dialect of the same language (Glottolog brun1245), is inside this figure.
    Belait   200 ("fewer than 200" today, Wikipedia "Belait language"): 150 Belait, 50 Tutong
             (Kuala Balai and Labi; Kiudang)
    Murut    600 Lun Bawang (Joshua Project people group 15053, Brunei), Temburong
    the rest of each district's Malays: Brunei Malay
  Chinese, one national mix (no district source):
    English 16%, Mandarin 30% (Asia Harvest, "Brunei Chinese", people-groups.asiaharvest.org:
    "about 16 percent of the Chinese people use English as their first language while some 30
    percent of them are Mandarin speakers"); the other 54% over the dialects by Joshua Project's
    Brunei figures (Ethnologue-based): Min Nan 12,000, Cantonese 5,600, Min Dong 5,300,
    Hakka 2,700. Joshua Project's Min Bei 10,000 is left out: no other source names Min Bei
    speakers in Brunei, and every one names Hokkien (Min Nan) as the largest group.
  Others:
    Iban     "about 15,800 speakers of Iban in the districts of Belait, Tutong and Temburong"
             (Omniglot, "Iban language and alphabet"), placed over those three districts by
             their "settled Others" (Others less temporary residents, A3 - A1): the Iban are
             citizens and permanent residents, the temporary residents mostly foreign workers. Temburong is held at 2,000 (UBD IAS working paper 65:
             "less than 2,000" there; the plain split gave 2,548) and its excess spread over
             Belait and Tutong the same way.
    the rest: foreign residents, by nationality: Bangladesh 26,000, Indonesia 25,000,
             Philippines 23,000 (2024 private-sector work-pass figures as reported; the Minister
             of Home Affairs named the same three as the largest in 2022), India 10,000 (Government
             of India, via Wikipedia "Indians in Brunei"), Nepal 5,822 and the United Kingdom 2,238
             (UN DESA International Migrant Stock 2020); each origin at origin_mix.mix(iso, "bn").
             Only the shares are used: the counts are the census's Others.

Usage:  python sources/bn_build.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from rdlink import RD  # noqa: E402

XLS = RD / "data" / "raw" / "bn" / "bpp2021_excel_table_A-C.xls"
OUT = ROOT / "data" / "normalized" / "bn.csv"

DISTRICTS = ["Brunei Muara", "Belait", "Tutong", "Temburong"]
TOTAL = 440_715

# Table A3, persons (Malays, Chinese, Others), asserted equal to the workbook below
A3 = {
    "Brunei Muara": (222_772, 28_522, 67_236),
    "Belait":       (33_245, 11_068, 21_218),
    "Tutong":       (35_283, 2_318, 9_609),
    "Temburong":    (5_716, 224, 3_504),
}
# Table A1, temporary residents
TEMP = {"Brunei Muara": 63_560, "Belait": 12_386, "Tutong": 4_452, "Temburong": 814}

AN = "austronesian"
BRUNEI_MALAY = f"{AN}.malayic.brunei_malay"
KEDAYAN = f"{AN}.malayic.kedayan"
IBAN = f"{AN}.malayic.iban"
TUTONG = f"{AN}.north_borneo.tutong"
BELAIT = f"{AN}.north_borneo.belait"
DUSUN = f"{AN}.north_borneo.brunei_dusun"
MURUT = f"{AN}.north_borneo.lundayeh"
ST = "sinotibetan.sinitic"
ENGLISH = "indoeuropean.germanic.english"

TUTONG_SPEAKERS = 17_000
TUTONG_SHARE = {"Tutong": 64.7, "Brunei Muara": 24.5, "Belait": 10.5, "Temburong": 0.2}
KEDAYAN_SPEAKERS = 30_000
KEDAYAN_DISTRICTS = ["Brunei Muara", "Tutong", "Belait"]
DUSUN_SPEAKERS = 10_000
DUSUN_SHARE = {"Tutong": 2 / 3, "Belait": 1 / 3}
BELAIT_SPLIT = {"Belait": 150, "Tutong": 50}
MURUT_SPLIT = {"Temburong": 600}
IBAN_SPEAKERS = 15_800
IBAN_DISTRICTS = ["Belait", "Tutong", "Temburong"]
IBAN_CAP = {"Temburong": 2_000}     # UBD IAS working paper 65: "less than 2,000" in Temburong

CHINESE_MIX_FIXED = {ENGLISH: 0.16, f"{ST}.mandarin": 0.30}
CHINESE_DIALECTS = {f"{ST}.min_nan": 12_000, f"{ST}.cantonese": 5_600,
                    f"{ST}.min_dong": 5_300, f"{ST}.hakka": 2_700}
FOREIGN = {"BD": 26_000, "ID": 25_000, "PH": 23_000, "IN": 10_000, "NP": 5_822, "GB": 2_238}

LABEL = {
    "Malays": "BPP 2021 A3 Malays",
    "Chinese": "BPP 2021 A3 Chinese",
    "Others": "BPP 2021 A3 Others",
}


def check_workbook():
    import xlrd
    sh = xlrd.open_workbook(str(XLS)).sheet_by_name("A3")
    rows = {}
    for r in range(sh.nrows):
        lab = " ".join(str(sh.cell_value(r, 0)).split())
        if lab in ("Malays", "Chinese", "Others", "Jumlah/Total"):
            rows[lab] = sh.row_values(r)
    # persons columns: 1 (total), then 4, 7, 10, 13 for the districts in DISTRICTS order
    for i, d in enumerate(DISTRICTS):
        c = 4 + 3 * i
        got = tuple(int(rows[k][c]) for k in ("Malays", "Chinese", "Others"))
        if got != A3[d]:
            raise SystemExit(f"A3 {d}: workbook {got}, transcription {A3[d]}")
        if int(rows["Jumlah/Total"][c]) != sum(A3[d]):
            raise SystemExit(f"A3 {d}: races do not sum to the district total")
    if int(rows["Jumlah/Total"][1]) != TOTAL or sum(map(sum, A3.values())) != TOTAL:
        raise SystemExit("A3: national total is not 440,715")
    sh = xlrd.open_workbook(str(XLS)).sheet_by_name("A1")
    for r in range(sh.nrows):
        if " ".join(str(sh.cell_value(r, 0)).split()) == "Temporary Residents":
            got = {d: int(sh.cell_value(r, 4 + 3 * i)) for i, d in enumerate(DISTRICTS)}
            if got != TEMP:
                raise SystemExit(f"A1 temporary residents: workbook {got}, transcription {TEMP}")
    print("  OK A3 (race x district) and A1 (temporary residents) equal the transcription; "
          f"{TOTAL:,} people")


def integerise(parts, total):
    """Largest remainder: {node: float} -> {node: int} summing exactly to total."""
    fl = {k: int(v) for k, v in parts.items()}
    short = total - sum(fl.values())
    order = sorted(parts, key=lambda k: parts[k] - int(parts[k]), reverse=True)
    for k in order[:short]:
        fl[k] += 1
    return fl


def foreign_mix():
    from origin_mix import mix
    tot = sum(FOREIGN.values())
    out = {}
    for iso, n in FOREIGN.items():
        for node, s in mix(iso, "bn").items():
            out[node] = out.get(node, 0.0) + s * n / tot
    return out


def iban_split(settled):
    """Iban over IBAN_DISTRICTS by settled Others, a district over its cap held at the cap and
    the excess spread over the others the same way."""
    out, left, todo = {}, float(IBAN_SPEAKERS), list(IBAN_DISTRICTS)
    while todo:
        tot = sum(settled[d] for d in todo)
        share = {d: left * settled[d] / tot for d in todo}
        over = [d for d in todo if d in IBAN_CAP and share[d] > IBAN_CAP[d]]
        if not over:
            out.update(share)
            break
        for d in over:
            out[d] = float(IBAN_CAP[d])
            left -= IBAN_CAP[d]
            todo.remove(d)
    return out


def chinese_mix():
    rest = 1 - sum(CHINESE_MIX_FIXED.values())
    tot = sum(CHINESE_DIALECTS.values())
    out = dict(CHINESE_MIX_FIXED)
    for node, n in CHINESE_DIALECTS.items():
        out[node] = rest * n / tot
    return out


def main():
    check_workbook()
    tsum = sum(TUTONG_SHARE.values())
    ked_base = sum(A3[d][0] for d in KEDAYAN_DISTRICTS)
    settled = {d: A3[d][2] - TEMP[d] for d in DISTRICTS}
    if min(settled.values()) <= 0:
        raise SystemExit(f"settled Others not positive: {settled}")
    iban_by = iban_split(settled)
    cmix, fmix = chinese_mix(), foreign_mix()

    rows = []
    for d in DISTRICTS:
        malays, chinese, others = A3[d]
        # Malays
        m = {TUTONG: TUTONG_SPEAKERS * TUTONG_SHARE[d] / tsum,
             KEDAYAN: KEDAYAN_SPEAKERS * malays / ked_base if d in KEDAYAN_DISTRICTS else 0.0,
             DUSUN: DUSUN_SPEAKERS * DUSUN_SHARE.get(d, 0.0),
             BELAIT: float(BELAIT_SPLIT.get(d, 0)),
             MURUT: float(MURUT_SPLIT.get(d, 0))}
        m[BRUNEI_MALAY] = malays - sum(m.values())
        if m[BRUNEI_MALAY] < 0.25 * malays:
            raise SystemExit(f"{d}: the puak estimates leave under a quarter of the Malays: {m}")
        for node, n in integerise(m, malays).items():
            rows.append((d, node, "Malays", n, "derived" if node == BRUNEI_MALAY else "modelled"))
        # Chinese
        for node, n in integerise({k: v * chinese for k, v in cmix.items()}, chinese).items():
            rows.append((d, node, "Chinese", n, "modelled"))
        # Others
        iban = iban_by.get(d, 0.0)
        o = {k: v * (others - iban) for k, v in fmix.items()}
        o[IBAN] = o.get(IBAN, 0.0) + iban
        for node, n in integerise(o, others).items():
            rows.append((d, node, "Others", n, "modelled"))

    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "race", "count", "tier"])
    df = df[df["count"] > 0]
    df = df.groupby(["geo_id", "source_category", "tier"], as_index=False)["count"].sum()
    for d in DISTRICTS:
        got = int(df.loc[df["geo_id"] == d, "count"].sum())
        if got != sum(A3[d]):
            raise SystemExit(f"{d}: rows sum to {got:,}, A3 says {sum(A3[d]):,}")
    if int(df["count"].sum()) != TOTAL:
        raise SystemExit("rows do not sum to 440,715")
    df.insert(1, "geo_level", "district")
    df["source_id"] = "bn_bpp2021_A3_estimates"
    df["year"] = 2021
    df = df.sort_values(["geo_id", "count"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(OUT) + ".part", index=False)
    os.replace(str(OUT) + ".part", OUT)

    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\n  {len(df)} rows, {df['source_category'].nunique()} nodes, {int(nat.sum()):,} people")
    for node, n in nat.items():
        if n >= 300:
            print(f"    {n:>9,}  {100 * n / TOTAL:6.2f}%  {node}")
    print(f"    ({(nat < 300).sum()} nodes under 300)")
    print("\n  Iban by district:", {d: round(v) for d, v in iban_by.items()})
    piv = df.pivot_table(index="source_category", columns="geo_id", values="count", aggfunc="sum")
    top = [BRUNEI_MALAY, KEDAYAN, TUTONG, DUSUN, IBAN, f"{ST}.mandarin", ENGLISH]
    print(piv.loc[[t for t in top if t in piv.index], DISTRICTS].fillna(0).astype(int))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
