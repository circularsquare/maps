"""Sri Lanka: CPH 2024 ethnic group by GN division, read as languages, with each group's retention
taken from CPH 2012's language-ability tables by district.

Writes data/normalized/lk.csv (one row per GN division x ethnic group x language drawn, all
`derived`) and data/normalized/lk_retention.csv (the 2012 shares used, per district and group).

THE TWO TABLES
- CPH 2024, `GN_Level_Population_by_Ethnic_Group.xlsx` (DCS): 10 ethnic groups + Other on 14,003
  GN divisions, 21,781,800 people. Same layout and the same `<10 moved to Other` disclosure rule
  as the religion workbook religiondots uses (`-` means zero, or up to nine people now in Other).
- CPH 2012 district reports, Table A30 (ability to speak Sinhala, Tamil, English by ethnic group)
  and Table A32 (ability to speak two or three of them), population aged 10+, all sectors, both
  sexes, one PDF each per district. Together they give, by inclusion-exclusion, how many of each
  ethnic group in each district speak each exact combination of the three languages.

THE READING (sources/lk.md has the reasoning)
An ethnic group is drawn on its heritage language unless the 2012 tables show its members cannot
speak it. Sinhalese -> Sinhala; Sri Lanka Tamil, Indian Tamil, Sri Lanka Moor -> Tamil; Burgher
-> English. Of those who cannot speak the heritage language, those who speak Sinhala or Tamil go
there, English-only go to English, and the few who speak none of the three stay on the heritage
language. Bilinguals stay on the heritage language: the census asks ability, not home language,
so this undercounts shift (a Moor family in Galle speaking Sinhala at home but able to speak Tamil
is drawn as Tamil). Malay -> Sri Lanka Malay, Chetty and Bharatha -> Tamil (no ability row of
their own: 2012 filed them under Other), Veddah -> the GN division's majority language of Sinhala
and Tamil, Other -> `other`.

District shares come from the group's own row in its district; rows with fewer than 200 people
aged 10+ use the group's national pooled shares instead (Burghers in Mannar are 18 people).

Usage:
    python sources/lk_census.py --fetch    the 2024 workbook (~7 MB) and 50 small PDFs
    python sources/lk_census.py            normalise from data/raw/lk/
"""
import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lk")
PDFS = os.path.join(RAW, "cph2012_lang")
OUT = os.path.join(ROOT, "data", "normalized", "lk.csv")
OUT_RET = os.path.join(ROOT, "data", "normalized", "lk_retention.csv")

XLSX = "GN_Level_Population_by_Ethnic_Group.xlsx"
URL = ("https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024/GNLevel/"
       "GN_Level_Population_by_Ethnic_Group")
PDF_URL = "https://www.statistics.gov.lk/PopHouSat/CPH2011/Pages/Activities/Reports/District/{}/{}.pdf"
# Kandy's reports alone are filed as "Table A30.pdf"
PDF_NAME = {"Kandy": "Table%20{}"}

NATIONAL = 21_781_800
EXPECTED_GND = 14_003

GROUPS = ["Sinhalese", "Sri Lanka Tamil", "Indian Tamil/ Malaiyaga Thamilar",
          "Sri Lanka Moor/Muslim", "Burgher", "Malay", "Sri Lanka Chetty", "Bharatha",
          "Veddahs", "Other"]
NAT_2024 = {"Sinhalese": 16140688, "Sri Lanka Tamil": 2665574,
            "Indian Tamil/ Malaiyaga Thamilar": 590087, "Sri Lanka Moor/Muslim": 2274372,
            "Burgher": 25159, "Malay": 22838, "Sri Lanka Chetty": 1753, "Bharatha": 553,
            "Veddahs": 1287, "Other": 59489}

# 2024 district code -> the 2012 report's folder name
DISTRICTS = {11: "Colombo", 12: "Gampaha", 13: "Kalutara", 21: "Kandy", 22: "Matale",
             23: "NuwaraEliya", 31: "Galle", 32: "Matara", 33: "Hambantota", 41: "Jaffna",
             42: "Mannar", 43: "Vavuniya", 44: "Mullaitivu", 45: "Kilinochchi",
             51: "Batticaloa", 52: "Ampara", 53: "Trincomalee", 61: "Kurunegala",
             62: "Puttalam", 71: "Anuradhapura", 72: "Polonnaruwa", 81: "Badulla",
             82: "Monaragala", 91: "Ratnapura", 92: "Kegalle"}

# 2024 group -> (2012 ability row, heritage language letter)
RETAINED = {"Sinhalese": ("Sinhalese", "S"), "Sri Lanka Tamil": ("Sri Lanka Tamil", "T"),
            "Indian Tamil/ Malaiyaga Thamilar": ("Indian Tamil", "T"),
            "Sri Lanka Moor/Muslim": ("Sri Lanka Moor", "T"), "Burgher": ("Burgher", "E")}
LANG = {"S": "Sinhala", "T": "Tamil", "E": "English"}
MIN_ROW = 200


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.makedirs(PDFS, exist_ok=True)
    h = {"User-Agent": "Mozilla/5.0"}
    dest = os.path.join(RAW, XLSX)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 1_000_000):
        r = requests.get(URL, timeout=300, verify=False, headers=h)
        r.raise_for_status()
        if r.content[:2] != b"PK":
            raise SystemExit(f"not an xlsx: {r.content[:60]!r}")
        open(dest, "wb").write(r.content)
        print(f"  {len(r.content):,} bytes -> {dest}")
    for d in DISTRICTS.values():
        for t in ("A30", "A32"):
            p = os.path.join(PDFS, f"{d}_{t}.pdf")
            if os.path.exists(p) and os.path.getsize(p) > 10_000:
                continue
            u = PDF_URL.format(d, PDF_NAME.get(d, "{}").format(t))
            r = requests.get(u, timeout=120, verify=False, headers=h)
            if r.status_code != 200 or r.content[:4] != b"%PDF":
                raise SystemExit(f"{u}: {r.status_code} {r.content[:40]!r}")
            open(p, "wb").write(r.content)
            print(f"  {d} {t} {len(r.content):,} bytes")


NUM = re.compile(r"^(?:[\d,]+(?:\.\d+)?|-)$")
ROW_LABEL = [("total", "Total"), ("all groups", "Total"), ("sinhal", "Sinhalese"),
             ("sri lanka ta", "Sri Lanka Tamil"), ("indian tamil", "Indian Tamil"),
             ("sri lanka m", "Sri Lanka Moor"), ("burgher", "Burgher"), ("malay", "Malay"),
             ("other", "Other"), ("all other", "Other"), ("not reported", "Not reported")]


def _num(s):
    return 0 if s == "-" else int(s.replace(",", ""))


def read_pdf(path, ncols):
    """The first block (all sectors, both sexes) of an A30/A32 page: {row: [N, c1, c2, ...]}.

    Text order varies between districts (labels before or after their numbers, stray header
    words), so the page is read as a word stream: a run of non-numbers is a label, the numbers
    that follow are its row, and each row must have exactly 1 + 2*ncols numbers (N, then a count
    and a percentage per column). Percentages are dropped.
    """
    import fitz
    words = fitz.open(path)[0].get_text().replace("\xa0", " ").split()
    low = [w.lower() for w in words]
    i = next(k for k, w in enumerate(low) if w.startswith("sectors"))
    if low[i + 1].startswith("both"):
        i += 1
    if low[i + 1].startswith("sex"):
        i += 1
    j = next((k for k, w in enumerate(low) if k > i and w in ("male", "males")), len(words))
    out, lab, nums = {}, [], []

    def close():
        name = " ".join(lab).lower() or ("total" if not out else "")  # Puttalam prints no label
        key = next((v for p, v in ROW_LABEL if name.startswith(p)), None)
        if key is None or len(nums) != 1 + 2 * ncols:
            return  # stray header words caught with a number
        if key in out:
            raise SystemExit(f"{path}: row {key} twice")
        vals = [_num(nums[0])] + [_num(x) for x in nums[1::2]]
        out[key] = vals

    for w in words[i + 1:j]:
        if NUM.match(w):
            nums.append(w)
        else:
            if nums:
                close()
                lab, nums = [], []
            lab.append(w)
    if nums:
        close()
    need = {"Total", "Sinhalese", "Sri Lanka Tamil", "Indian Tamil", "Sri Lanka Moor",
            "Burgher", "Malay", "Other"}
    if not need <= set(out):
        raise SystemExit(f"{path}: rows {sorted(out)} lack {sorted(need - set(out))}")
    return out


def partition(N, S, T, E, ST, SE, TE, STE, where):
    """Exact counts of each combination of speaking S, T, E, by inclusion-exclusion."""
    p = {"S": S - ST - SE + STE, "T": T - ST - TE + STE, "E": E - SE - TE + STE,
         "ST": ST - STE, "SE": SE - STE, "TE": TE - STE, "STE": STE}
    p["none"] = N - sum(p.values())
    bad = {k: v for k, v in p.items() if v < 0}
    if bad:
        raise SystemExit(f"{where}: negative combinations {bad} -- column order is not "
                         "S&T, S&E, T&E, all three")
    return p


def assign(p, H, district_tot):
    """Partition -> people per language drawn, for heritage language H."""
    out = {"S": 0, "T": 0, "E": 0}
    for combo, n in p.items():
        if combo == "none" or H in combo:
            out[H] += n
        elif combo in ("S", "T", "E"):
            out[combo] += n
        elif combo == "ST":            # only reached for H == E
            out["S" if district_tot["S"] >= district_tot["T"] else "T"] += n
        elif combo == "SE":            # H == T
            out["S"] += n
        elif combo == "TE":            # H == S
            out["T"] += n
        else:
            raise AssertionError(combo)
    return out


def retention():
    rows = []
    shares = {}
    pooled = {}
    for code, d in DISTRICTS.items():
        a30 = read_pdf(os.path.join(PDFS, f"{d}_A30.pdf"), 6)
        a32 = read_pdf(os.path.join(PDFS, f"{d}_A32.pdf"), 8)
        tot = a30["Total"]
        dt = {"S": tot[1], "T": tot[2], "E": tot[3]}
        for g, (row, H) in RETAINED.items():
            N, S, T, E = a30[row][:4]
            N2, ST, SE, TE, STE = a32[row][:5]
            if N != N2:
                raise SystemExit(f"{d} {row}: A30 N {N} != A32 N {N2}")
            p = partition(N, S, T, E, ST, SE, TE, STE, f"{d} {row}")
            got = assign(p, H, dt)
            assert sum(got.values()) == N
            pool = pooled.setdefault(g, {"S": 0, "T": 0, "E": 0})
            for k in got:
                pool[k] += got[k]
            shares[(code, g)] = (N, got)
    out = {}
    for (code, g), (N, got) in shares.items():
        use_nat = N < MIN_ROW
        src = pooled[g] if use_nat else got
        tot = sum(src.values())
        out[(code, g)] = {k: v / tot for k, v in src.items()}
        rows.append({"district_code": code, "district": DISTRICTS[code], "group": g,
                     "aged10_2012": N, "basis": "national pooled" if use_nat else "district",
                     **{LANG[k]: round(out[(code, g)][k], 5) for k in "STE"}})
    for g, pool in pooled.items():
        tot = sum(pool.values())
        print(f"  {g:34s} 2012 aged 10+ {tot:>10,}: " + ", ".join(
            f"{LANG[k]} {pool[k] / tot:.2%}" for k in "STE"))
    return out, rows


def _cell(v, where):
    if isinstance(v, (int, float)) and v == v:
        if float(v) != int(v):
            raise SystemExit(f"{where}: non-integer {v!r}")
        return int(v)
    if str(v).strip() == "-":
        return 0
    raise SystemExit(f"{where}: unrecognised cell {v!r}")


def split_int(n, shares):
    """n people over languages by shares, largest remainder, so the parts sum to n exactly."""
    raw = {k: n * s for k, s in shares.items()}
    base = {k: int(v) for k, v in raw.items()}
    left = n - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:left]:
        base[k] += 1
    return base


def main():
    import pandas as pd
    ret, ret_rows = retention()

    raw = pd.read_excel(os.path.join(RAW, XLSX), sheet_name=0, header=None, dtype=object)
    head = [str(x).strip() if x == x else "" for x in raw.iloc[3].tolist()]
    want = [""] * 6 + ["Total"] + GROUPS
    if head != want:
        raise SystemExit(f"header row 3 is {head!r}\n expected {want!r}")
    nat = [_cell(v, "national") for v in raw.iloc[4].tolist()[6:]]
    if nat[0] != NATIONAL or dict(zip(GROUPS, nat[1:])) != NAT_2024:
        raise SystemExit(f"national row {nat}")

    body = raw.iloc[5:].to_numpy()
    out, seen_d, tot_sum = [], {}, 0
    gsum = dict.fromkeys(GROUPS, 0)
    lsum = {}
    for r in body:
        d, ds, gn = int(r[0]), int(r[2]), int(r[4])
        dname = str(r[1]).strip()
        seen_d.setdefault(d, dname)
        if d not in DISTRICTS:
            raise SystemExit(f"district code {d} {dname} not in DISTRICTS")
        if DISTRICTS[d].replace(" ", "").lower()[:3] != dname.replace(" ", "").lower()[:3]:
            raise SystemExit(f"district {d}: workbook {dname!r} vs 2012 folder {DISTRICTS[d]!r}")
        geo_id = f"{d:02d}{ds:02d}{gn:03d}"
        name = str(r[5]).strip()
        vals = [_cell(v, f"{geo_id}") for v in r[6:17]]
        total, counts = vals[0], dict(zip(GROUPS, vals[1:]))
        if sum(counts.values()) != total:
            raise SystemExit(f"{geo_id}: groups sum {sum(counts.values())} != total {total}")
        tot_sum += total
        tamilish = (counts["Sri Lanka Tamil"] + counts["Indian Tamil/ Malaiyaga Thamilar"]
                    + counts["Sri Lanka Moor/Muslim"])
        for g, n in counts.items():
            gsum[g] += n
            if n == 0:
                continue
            if g in RETAINED:
                parts = {LANG[k]: v for k, v in split_int(n, ret[(d, g)]).items()}
            elif g == "Malay":
                parts = {"Sri Lanka Malay": n}
            elif g in ("Sri Lanka Chetty", "Bharatha"):
                parts = {"Tamil": n}
            elif g == "Veddahs":
                parts = {"Sinhala" if counts["Sinhalese"] >= tamilish else "Tamil": n}
            else:
                parts = {"Other": n}
            for lang, c in parts.items():
                if c == 0:
                    continue
                lsum[lang] = lsum.get(lang, 0) + c
                out.append({"geo_id": geo_id, "geo_level": "gnd", "geo_name": name,
                            "source_category": f"{g} > {lang}", "count": c,
                            "tier": "derived"})

    n_gnd = len({o["geo_id"] for o in out})
    print(f"  {len(body):,} GN rows, {n_gnd:,} with people; total {tot_sum:,}")
    if len(body) != EXPECTED_GND:
        raise SystemExit(f"expected {EXPECTED_GND} GN divisions, got {len(body)}")
    if tot_sum != NATIONAL or gsum != NAT_2024:
        raise SystemExit(f"GN rows do not sum to the national line: {tot_sum} {gsum}")
    if sum(o["count"] for o in out) != NATIONAL:
        raise SystemExit("languages drawn do not sum to the national total")
    if len(seen_d) != 25:
        raise SystemExit(f"{len(seen_d)} districts")
    for lang, c in sorted(lsum.items(), key=lambda x: -x[1]):
        print(f"  {lang:16s} {c:>11,}  {c / NATIONAL:6.2%}")
    moved = {}
    for o in out:
        g, lang = o["source_category"].split(" > ")
        if g in RETAINED and lang != LANG[RETAINED[g][1]]:
            moved[o["source_category"]] = moved.get(o["source_category"], 0) + o["count"]
    for k, v in sorted(moved.items(), key=lambda x: -x[1]):
        print(f"  moved off heritage: {k:45s} {v:>9,}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(OUT_RET, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ret_rows[0]))
        w.writeheader()
        w.writerows(ret_rows)
    print(f"  -> {OUT} ({len(out):,} rows), {OUT_RET}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
