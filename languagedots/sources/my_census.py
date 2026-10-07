"""Malaysia: Census 2020 ethnic group (district) and sub-ethnic group (state), plus the 2010
census's Sabah and Sarawak sub-groups by district, for reading ethnicity as language.

    python sources/my_census.py --fetch    fetch the two 2010 PDFs (Wayback), then normalise
    python sources/my_census.py            normalise what is on disk

Malaysia's census asks ethnic group, never language (tier D; AGENT_BRIEF §2's ethnicity rule).

INPUTS:
  data/raw/my/population_district.parquet: OpenDOSM "Population Table: Administrative
      Districts" (storage.dosm.gov.my, CC BY 4.0, no key). Its 2020 rows are the census by
      district: Malay, Other Bumiputera, Chinese, Indians, Other citizens, Non-citizens, in
      thousands to one decimal (so each cell is rounded to 100 people). THE DISTRICT TABLE.
  ../religiondots/data/raw/my/JADUAL BANCI MALAYSIA 2020 MUKIM BANDAR PEKAN.xlsx (DOSM
      eStatistik, free login; religiondots' download, read-only): ethnic group by district /
      mukim, 2010 and 2020. Used for its 2010 column on 2020 districts (the check on the 2010
      PDFs) and as a second 2020 table. It is not the district table because its mukim, bandar
      and pekan rows do not add to the published district totals everywhere (Kota Bharu +103,861
      and Tanah Merah -103,198; Sepang and Ulu Langat swap 138,000; Jelebu, Port Dickson short)
      and some continuation sheets carry the wrong district heading.
  ../religiondots/data/raw/my/<STATE> JADUAL 1 HINGGA 16.xlsx, Table 5: population by sub-ethnic
      group, the state only. 16 states.
  ../religiondots/data/raw/my/JADUAL 1 HINGGA 29.xlsx, Table 4 (T): ethnic group by state (check).
  data/raw/my/PBT_Sabah_2010.pdf, PBT_Sarawak_2010.pdf: Census 2010, Table 11.1 / 12.1 "Total
      population by ethnic group, Local Authority area and state", which splits Other
      Bumiputera into Kadazan Dusun, Bajau, Murut (Sabah) and Iban, Bidayuh, Melanau (Sarawak)
      under each administrative-district heading. statistics.gov.my/portal/download_Population/
      files/population/04Jadual_PBT_negeri/, dead since; Wayback 2012-02-27.

OUTPUT data/normalized/my.csv: level, geo_id, geo_name, source_category, count, year
  district  2020 (and 2010) broad groups per religiondots unit MYS_ss_dd (160)
  state     2020 sub-ethnic leaves per state (Table 5)
  seed2010  Sabah and Sarawak 2010 heading groups: Malay, the three named groups, Other
            Bumiputera; geo_id is the 2020 district that borrows that group's mix

CHECKS (asserted unless said):
  * OpenDOSM: each of the 160 districts' 2020 total within 50 of religiondots' religion table
    (a different table of the same census; 50 is one rounded cell), its six groups within 300
    of its total; the 160 join by name, one to one;
  * districts sum per state to the national volume's Table 4 within 2,000 for every group but
    the Malay / Other Bumiputera split, which Table 4 files differently from Table 5 (Johor:
    4,841 more Malay) while OpenDOSM follows Table 5; Bumiputera as a whole is held;
  * the mukim workbook agrees with OpenDOSM within 100 on Bumiputera, Chinese, Indians and
    non-citizens in 143 of 160 districts (printed);
  * Table 5: leaves sum to the state's citizens, and its Bumiputera leaves to Table 4's
    Bumiputera, exactly in 15 states (Labuan 15 short, printed);
  * 2010 PDFs: every row's groups sum to its total; each 2010 heading equals the workbook's
    2010 column summed over the 2020 districts that borrow it, exactly in Sabah (3,117,405) and
    in Sarawak except where a boundary moved (Siburan, 32,299, from Kuching to Serian; 1,728
    Kapit to Belaga; 5,237 Miri to Marudi), printed and held under 2%.
"""
import csv
import os
import re
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import openpyxl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "my"
RD_RAW = ROOT.parent / "religiondots" / "data" / "raw" / "my"
RD_NORM = ROOT.parent / "religiondots" / "data" / "normalized" / "my.csv"
OUT = ROOT / "data" / "normalized" / "my.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
MUKIM = RD_RAW / "JADUAL BANCI MALAYSIA 2020 MUKIM BANDAR PEKAN.xlsx"
NATIONAL = RD_RAW / "JADUAL 1 HINGGA 29.xlsx"
PDFS = {
    "PBT_Sabah_2010.pdf": "https://web.archive.org/web/20120227090315id_/http://www.statistics.gov.my/"
                          "portal/download_Population/files/population/04Jadual_PBT_negeri/PBT_Sabah.pdf",
    "PBT_Sarawak_2010.pdf": "https://web.archive.org/web/20120227082205id_/http://www.statistics.gov.my:80/"
                            "portal/download_Population/files/population/04Jadual_PBT_negeri/PBT_Sarawak.pdf",
}
ODM_URL = "https://storage.dosm.gov.my/population/population_district.parquet"
STATE_FILES = {
    "01": "JOHOR", "02": "KEDAH", "03": "KELANTAN", "04": "MELAKA", "05": "NEGERI SEMBILAN",
    "06": "PAHANG", "07": "PULAU PINANG", "08": "PERAK", "09": "PERLIS", "10": "SELANGOR",
    "11": "TERENGGANU", "12": "SABAH", "13": "SARAWAK", "14": "W.P. KUALA LUMPUR",
    "15": "W.P. LABUAN", "16": "W.P. PUTRAJAYA",
}
GROUPS = ["Total", "Citizens", "Bumiputera", "Malay", "Other Bumiputera", "Chinese", "Indians",
          "Others", "Non-citizens"]
# Workbook headings that are not the religiondots name of the district.
ALIAS = {}

# 2020 districts -> the 2010 heading whose sub-group mix they borrow. Same name unless listed;
# the listed ones were carved out of these after 2010 (or, for Sabah, sit under the heading of
# the district they were split from in the 2010 table).
SEED_PARENT = {
    "12": {"Kalabakan": "TAWAU", "Telupid": "BELURAN"},
    "13": {"Tebedu": "SERIAN", "Pusa": "BETONG", "Kabong": "SARATOK", "Tanjung Manis": "DARO",
           "Sebauh": "BINTULU", "Bukit Mabong": "KAPIT", "Subis": "MIRI", "Beluru": "MARUDI",
           "Telang Usan": "MARUDI"},
}
SEED_COLS = {
    "12": ["Malay", "Kadazan Dusun", "Bajau", "Murut", "Other Bumiputera"],
    "13": ["Malay", "Iban", "Bidayuh", "Melanau", "Other Bumiputera"],
}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in list(PDFS.items()) + [("population_district.parquet", ODM_URL)]:
        if (RAW / name).exists():
            continue
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            (RAW / name).write_bytes(r.read())
        print("  fetched", name)


def num(v):
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(round(v))
    s = str(v).strip().replace(",", "")
    if s in ("", "-"):
        return 0
    return int(s)


ODM = RAW / "population_district.parquet"
ODM_COLS = {"overall": "Total", "bumi_malay": "Malay", "bumi_other": "Other Bumiputera",
            "chinese": "Chinese", "indian": "Indians", "other_citizen": "Others",
            "other_noncitizen": "Non-citizens"}


def district_opendosm(rd):
    """{geo_id: [9 values in GROUPS order]} for 2020, from OpenDOSM (thousands, one decimal)."""
    import pandas as pd
    d = pd.read_parquet(ODM)
    d = d[(d["date"].astype(str) == "2020-01-01") & (d["sex"] == "both") & (d["age"] == "overall")]
    ids = {}
    for (gid, name) in rd:
        ids[name.lower()] = ids.get(name.lower(), []) + [gid]
    out = {}
    for (state, district), g in d.groupby(["state", "district"]):
        cand = ids.get(district.lower(), [])
        if len(cand) != 1:
            raise SystemExit(f"OpenDOSM {state}/{district}: {len(cand)} religiondots districts")
        v = {ODM_COLS[e]: int(round(p * 1000)) for e, p in zip(g["ethnicity"], g["population"])}
        cit = sum(v[k] for k in ("Malay", "Other Bumiputera", "Chinese", "Indians", "Others"))
        out[cand[0]] = [v["Total"], cit, v["Malay"] + v["Other Bumiputera"], v["Malay"],
                        v["Other Bumiputera"], v["Chinese"], v["Indians"], v["Others"],
                        v["Non-citizens"]]
    assert len(out) == 160, len(out)
    return out


def rd_districts():
    out = {}
    with open(RD_NORM, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["geo_level"] != "district":
                continue
            k = (r["geo_id"], r["geo_name"])
            out[k] = out.get(k, 0) + int(float(r["count"]))
    return out


def norm(s):
    s = re.sub(r"\(samb\..*?\)", "", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def district_table(rd):
    """{geo_id: {year: [9 values]}} from the mukim workbook."""
    names = {}
    for (gid, name) in rd:
        names.setdefault(gid[4:6], {})[norm(ALIAS.get(name, name))] = gid
    wb = openpyxl.load_workbook(MUKIM, read_only=True, data_only=True)
    own, summed = {}, {}
    for ws in wb.worksheets:
        m = re.match(r"(\d+)\.2 ", ws.title)
        if not m:
            continue
        st = f"{int(m.group(1)):02d}"
        known = names[st]
        cur = None
        for row in ws.iter_rows(min_row=9, values_only=True):
            if not isinstance(row[0], str):
                continue
            name = norm(row[0])
            cells = list(row[1:19])
            blank = all(v is None or (isinstance(v, str) and not v.strip()) for v in cells)
            if blank:
                if name in known:
                    cur = known[name]
                continue
            vals = [num(v) for v in cells]
            v10, v20 = vals[0::2], vals[1::2]
            if name in known:                       # a district's own row, figures and all
                own[known[name]] = {2010: v10, 2020: v20}
                cur = known[name] if st not in ("12", "13") else None
                continue
            if st in ("12", "13") or cur is None:   # Sarawak daerah kecil: subsets
                continue
            acc = summed.setdefault(cur, {2010: [0] * 9, 2020: [0] * 9})
            for y, v in ((2010, v10), (2020, v20)):
                acc[y] = [a + b for a, b in zip(acc[y], v)]
    out = {}
    for gid in [g for (g, _) in rd]:
        if gid in summed and gid in own:
            # a heading with figures AND rows under it: the rows are its mukims; keep the own row
            out[gid] = own[gid]
        elif gid in own:
            out[gid] = own[gid]
        elif gid in summed:
            out[gid] = summed[gid]
        else:
            raise SystemExit(f"district {gid} not found in the mukim workbook")
    return out


def check_opendosm(odm, rd):
    """OpenDOSM's 2020 district totals against religiondots' religion table (same census)."""
    worst = 0
    for (gid, name), n in rd.items():
        tot = odm[gid][0]
        parts = sum(odm[gid][3:9])
        assert abs(parts - tot) <= 300, (gid, name, parts, tot)      # six cells rounded to 100
        assert abs(tot - n) <= 50, (gid, name, tot, n)                # one cell rounded to 100
        worst = max(worst, abs(tot - n))
    print(f"  OpenDOSM 2020: 160 districts, each total within {worst} of the religion table")


def check_districts(dist, rd):
    tot_ours = tot_rd = 0
    bad = []
    for (gid, name), n in sorted(rd.items()):
        v = dist[gid][2020]
        if sum(v[3:9]) != v[0]:
            # a few mukim rows' groups do not add to their own total in the workbook
            print(f"  {gid} {name}: groups add to {sum(v[3:9]):,}, total printed {v[0]:,}")
            assert abs(sum(v[3:9]) - v[0]) <= 0.001 * v[0], (gid, name, v)
        tot_ours += v[0]
        tot_rd += n
        if abs(v[0] - n) > 0.005 * n:
            bad.append((gid, name, v[0], n))
        elif v[0] != n:
            print(f"  {gid} {name}: workbook {v[0]:,} vs religion table {n:,}")
    for b in bad:
        print("  !! ", b)
    print(f"  districts 2020: {tot_ours:,} against the religion table's {tot_rd:,}")
    assert not bad, "districts off by more than 0.5%"
    assert abs(tot_ours - tot_rd) <= 0.0005 * tot_rd


def national_table4():
    wb = openpyxl.load_workbook(NATIONAL, read_only=True, data_only=True)
    ws = wb["4. (T)"]
    out = {}
    for row in ws.iter_rows(min_row=14, values_only=True):
        if isinstance(row[0], str) and isinstance(row[1], (int, float)):
            out[row[0].strip()] = [num(v) for v in row[1:10]]
    return out


def check_states(dist, t4):
    # districts sum per state, per group, to Table 4 (OpenDOSM rounds each cell to 100)
    order = list(STATE_FILES)
    names = list(t4)[1:]                   # first row is Malaysia
    for st, nm in zip(order, names):
        agg = [0] * 9
        for gid, v in dist.items():
            if gid[4:6] == st:
                agg = [a + b for a, b in zip(agg, v[2020])]
        ref = t4[nm]
        for g, a, b in zip(GROUPS, agg, ref):
            # Table 4 files some Bumiputera between Malay and Other Bumiputera differently from
            # Table 5 (Johor: 4,841 more Malay); OpenDOSM follows Table 5, so only the sum is held
            if g in ("Malay", "Other Bumiputera"):
                continue
            if abs(a - b) > 50 * 40:
                raise SystemExit(f"{nm} {g}: districts {a:,} vs Table 4 {b:,}")
        print(f"  {nm}: districts {agg[0]:,} vs Table 4 {ref[0]:,}")


def table5(st):
    """[(label, count)] leaves of a state's sub-ethnic table, and its subtotals."""
    wb = openpyxl.load_workbook(RD_RAW / f"{STATE_FILES[st]} JADUAL 1 HINGGA 16.xlsx",
                                read_only=True, data_only=True)
    ws = None
    for w in wb.worksheets:
        top = " ".join(str(c) for r in w.iter_rows(min_row=1, max_row=2, values_only=True)
                       for c in r if c)
        if "sub-etnik" in top or "sub-ethnic" in top:
            ws = w
            break
    if ws is None:
        raise SystemExit(f"{st}: no sub-ethnic table")
    rows = []
    for r in ws.iter_rows(values_only=True):
        if isinstance(r[0], str) and len(r) > 1 and isinstance(r[1], (int, float)):
            rows.append((re.sub(r"\s+", " ", r[0]).strip(), num(r[1])))
    return rows


SUBTOTAL_PREFIX = ("Jumlah Penduduk", "Warganegara Malaysia", "Bukan Warganegara")
SUBTOTAL_EXACT = ("Bumiputera", "Orang Asli Semenanjung", "Bumiputera Sabah", "Bumiputera Sarawak")


def state_leaves(t4):
    out = {}
    names = list(t4)[1:]
    for st, nm in zip(STATE_FILES, names):
        leaves = [(lab, n) for lab, n in table5(st)
                  if not lab.startswith(SUBTOTAL_PREFIX) and lab not in SUBTOTAL_EXACT]
        cit, bumi = t4[nm][1], t4[nm][2]
        got = sum(n for _, n in leaves)
        b = sum(n for lab, n in leaves if not lab.startswith(("Cina", "India", "Lain-lain")))
        if got != cit or b != bumi:
            print(f"  {nm} Table 5: leaves {got:,} vs citizens {cit:,}; Bumiputera {b:,} vs {bumi:,}")
        assert abs(got - cit) <= 0.001 * cit and abs(b - bumi) <= 0.001 * bumi, (nm, got, cit, b, bumi)
        out[st] = leaves
    return out


def pdf_groups(path):
    """{heading: [12 values summed over its local-authority rows]} from a 2010 PBT table."""
    import fitz
    numre = re.compile(r"^(\d{1,3}(,\d{3})*|-)$")
    lare = re.compile(r"^(M\.|D\.B|L\.B|L\.K|Majlis|Lembaga)")
    d = fitz.open(path)
    out, cur, first = {}, None, False
    for p in d:
        if "kumpulan etnik" not in p.get_text():
            continue
        rows = {}
        for w in p.get_text("words"):
            rows.setdefault(round(w[1] / 3), []).append(w)
        for y in sorted(rows):
            toks = [w[4] for w in sorted(rows[y], key=lambda w: w[0])]
            nums = [t for t in toks if numre.match(t)]
            name = " ".join(t for t in toks if not numre.match(t))
            if (not nums and name.isupper() and len(name) > 2 and not lare.match(name)):
                cur = name
                out.setdefault(cur, {})
                first = True
                continue
            if cur and len(nums) == 12 and (first or lare.match(name)):
                v = tuple(0 if t == "-" else int(t.replace(",", "")) for t in nums)
                # Total, Citizens, Bumiputera, Malay, X, Y, Z, Other Bumi, Chinese, Indians,
                # Others, Non-citizens
                assert v[0] == v[1] + v[11] and v[1] == v[2] + v[8] + v[9] + v[10] \
                    and v[2] == sum(v[3:8]), (path, cur, name, v)
                out[cur][v] = name          # identical rows (Sibu printed twice) count once
                first = False
    return {k: [sum(c) for c in zip(*v)] for k, v in out.items()}


def main():
    if "--fetch" in sys.argv:
        fetch()
    rd = rd_districts()
    assert len(rd) == 160, len(rd)
    odm = district_opendosm(rd)
    check_opendosm(odm, rd)
    t4 = national_table4()
    check_states({g: {2020: v} for g, v in odm.items()}, t4)
    dist = district_table(rd)           # the mukim workbook: 2010 on 2020 districts, and a check
    agree = sum(1 for g in odm if all(abs(odm[g][i] - dist[g][2020][i]) <= 100 for i in (2, 5, 6, 8)))
    print(f"  mukim workbook agrees with OpenDOSM within 100 on Bumiputera, Chinese, Indians and non-citizens in {agree} of 160 districts")
    leaves = state_leaves(t4)
    rows = []
    name_of = {gid: nm for (gid, nm) in rd}
    for gid in sorted(odm):
        for g, n in zip(GROUPS, odm[gid]):
            rows.append(("district", gid, name_of[gid], g, n, 2020))
    for st, ls in leaves.items():
        for lab, n in ls:
            rows.append(("state", f"MYS_{st}", st, lab, n, 2020))
    for st, pdf in (("12", "PBT_Sabah_2010.pdf"), ("13", "PBT_Sarawak_2010.pdf")):
        groups = pdf_groups(RAW / pdf)
        tot = sum(v[0] for v in groups.values())
        print(f"  {pdf}: {len(groups)} headings, {tot:,} people")
        wb10 = {}
        for gid, nm in sorted(name_of.items()):
            if gid[4:6] != st:
                continue
            head = SEED_PARENT[st].get(nm, nm.upper())
            if head not in groups:
                raise SystemExit(f"{nm}: no 2010 heading {head}")
            v = groups[head]
            wb10[head] = wb10.get(head, 0) + dist[gid][2010][0]
            for lab, n in zip(SEED_COLS[st], v[3:8]):
                rows.append(("seed2010", gid, head, lab, n, 2010))
        # each heading against the workbook's 2010 column summed over the 2020 districts that
        # borrow it: equal except where a boundary moved between 2010 and 2020
        for head, n in wb10.items():
            if n != groups[head][0]:
                print(f"    {head}: 2010 PDF {groups[head][0]:,} vs workbook 2010 {n:,}")
        moved = sum(abs(groups[h][0] - wb10[h]) for h in wb10) / 2
        assert sum(wb10.values()) == tot and moved < 0.02 * tot, (pdf, moved)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "geo_id", "geo_name", "source_category", "count", "year"])
        w.writerows(rows)
    print(f"  wrote {OUT} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
