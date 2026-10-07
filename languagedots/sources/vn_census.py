"""Vietnam 2019 census, ethnic group by province and urban/rural, with the 2024 survey's home
language shares applied -> data/normalized/vn.csv (and vn_retention_2024.csv beside it).

    python sources/vn_census.py [--fetch]

NO LANGUAGE QUESTION. Vietnam's censuses ask ethnic group (dan toc), not language. Anita's
2026-10-05 ruling (AGENT_BRIEF section 2, ethnicity only) allows reading each group as its
language, every row `tier=derived`, after a retention check. sources/vn.md is the record.

1. The counts. "Ket qua toan bo Tong dieu tra dan so va nha o 2019" (GSO, now NSO), Bieu 2
   (Table 2), PDF pages 44-210: population by ethnic group, urban/rural and sex, for the
   country, 6 regions and 63 provinces. Each block is a total row and 56 rows: Kinh, the 53
   minorities, `Nguoi nuoc ngoai` (foreigners) and `Khong xac dinh` (not stated). Written at
   geo_level `province_ur`, geo_id `<GSO code>-u` / `<GSO code>-r`, so the urban and rural
   parts of a province can be placed apart (countries/vn.py, sources/vn_geo.py).

2. The retention shares. "Ket qua Dieu tra thu thap thong tin ve thuc trang kinh te - xa hoi
   cua 53 dan toc thieu so nam 2024" (NSO and the Ethnic Minorities Committee, published June
   2026), Bieu 3.9, PDF pages 178-179: for each of the 53 minorities, the share of its
   HOUSEHOLDS mainly using, in family communication, the household's own ethnic language,
   Vietnamese (tieng Kinh), or another ethnic language. Each minority's census count in a
   province is split on those national shares into three categories:
       "<group>"                                    the group's own language
       "<group> / Vietnamese at home"               drawn as Vietnamese
       "<group> / another minority language at home" drawn on seasia_other (unnamed)
   Integer counts: own and Vietnamese rounded, the third takes the remainder (clipped at 0,
   the rounding then taken from Vietnamese), so every unit's group total is the census's.

Checks, all asserted:
  * every block's 56 rows sum to its total row, in all nine columns; total = urban + rural and
    total = male + female for every row;
  * the 63 provinces sum to the national block for every group; the national total is
    96,208,984 (the volume's own figure);
  * the retention table has 53 groups; own + other = 100 and Kinh + other ethnic = other,
    each within 0.15 (one-decimal rounding);
  * after the split, every province's urban and rural totals are the census's, to the person.
"""
import argparse
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "vn"
PDF_2019 = RAW / "Ket-qua-toan-bo-2019.pdf"
PDF_2024 = RAW / "vn_53dtts_2024.pdf"
URL_2019 = ("https://www.nso.gov.vn/wp-content/uploads/2019/12/"
            "Ket-qua-toan-bo-Tong-dieu-tra-dan-so-va-nha-o-2019.pdf")
URL_2024 = ("https://www.nso.gov.vn/wp-content/uploads/2026/06/Ket-qua-Dieu-tra-thu-thap-thong-"
            "tin-ve-thuc-trang-kinh-te-xa-hoi-cua-53-dan-toc-thieu-so-nam-2024.pdf.pdf")
OUT = HERE / "data" / "normalized" / "vn.csv"
OUT_RET = HERE / "data" / "normalized" / "vn_retention_2024.csv"
# religiondots' province layer: GSO codes and names (read-only)
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402
LOOKUP = RD_GEO / "vn" / "vn_lookup.csv"

NATIONAL = 96_208_984
T2_PAGES = range(43, 210)        # 0-based: PDF pages 44-210
T39_PAGES = (177, 178)           # 0-based: PDF pages 178-179
KINH = "Kinh"
FOREIGN = "Người nước ngoài"
NOT_STATED = "Không xác định"
HOME_VI = " / Vietnamese at home"
HOME_OTHER = " / another minority language at home"

NUM = re.compile(r"^(\d{1,3}(?:[  ]\d{3})*|-)$")
REGIONS = {"Trung du và miền núi phía Bắc Northern Midlands and Mountains",
           "Đồng bằng sông Hồng Red River Delta",
           "Bắc Trung Bộ và Duyên hải miền Trung North and South Central Coast",
           "Tây Nguyên - Central Highlans", "Đông Nam Bộ - Southeast",
           "Đồng bằng sông Cửu Long Mekong River Delta"}
COUNTRY = "TOÀN QUỐC - ENTIRE COUNTRY"


def nfc(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for url, path in ((URL_2019, PDF_2019), (URL_2024, PDF_2024)):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=600).read()
        if data[:4] != b"%PDF" or b"%%EOF" not in data[-2048:]:
            raise SystemExit(f"vn: {url} is not a whole PDF")
        path.write_bytes(data)


def _n(tok):
    return 0 if tok == "-" else int(tok.replace(" ", "").replace(" ", ""))


def read_table2():
    """Every block of Table 2 as {area: [(label, [9 ints]), ...]}, the total row first."""
    import fitz
    doc = fitz.open(PDF_2019)
    lines = []
    for i in T2_PAGES:
        ls = [nfc(l) for l in doc[i].get_text().splitlines()]
        n_f = 0
        for k, l in enumerate(ls):         # the column header ends at the third "Female"
            if l == "Female":
                n_f += 1
                if n_f == 3:
                    break
        if n_f != 3:
            raise SystemExit(f"Table 2: no column header on PDF page {i + 1}")
        lines += [l for l in ls[k + 1:] if l]
    rows, label, nums = [], [], []
    for l in lines:
        if NUM.match(l):
            nums.append(_n(l))
            if len(nums) == 9:
                rows.append((nfc(" ".join(label)), nums))
                label, nums = [], []
        else:
            if nums:
                raise SystemExit(f"Table 2: a label interrupts numbers at {label} {nums}")
            label.append(l)
    if len(rows) != 70 * 57:
        raise SystemExit(f"Table 2: {len(rows)} rows, expected 70 blocks x 57")
    blocks = {}
    for b in range(70):
        blk = rows[b * 57:(b + 1) * 57]
        blocks[blk[0][0]] = blk
    return blocks


def check_blocks(blocks):
    groups = None
    for area, blk in blocks.items():
        head, body = blk[0][1], blk[1:]
        labels = [r[0] for r in body]
        if groups is None:
            groups = labels
            if groups[0] != KINH or groups[-2:] != [FOREIGN, NOT_STATED] or len(groups) != 56:
                raise SystemExit(f"Table 2: unexpected group list {groups[:3]}...{groups[-3:]}")
        elif labels != groups:
            raise SystemExit(f"Table 2: {area}'s groups differ from the country's")
        for c in range(9):
            s = sum(r[1][c] for r in body)
            if s != head[c]:
                raise SystemExit(f"Table 2: {area} column {c} sums to {s:,}, total row {head[c]:,}")
        for lab, v in blk:
            if v[0] != v[3] + v[6] or v[0] != v[1] + v[2] or v[3] != v[4] + v[5] or v[6] != v[7] + v[8]:
                raise SystemExit(f"Table 2: {area} / {lab} fails total = urban + rural = M + F")
    if blocks[COUNTRY][0][1][0] != NATIONAL:
        raise SystemExit(f"Table 2: national total {blocks[COUNTRY][0][1][0]:,}")
    provs = [a for a in blocks if a != COUNTRY and a not in REGIONS]
    if len(provs) != 63:
        raise SystemExit(f"Table 2: {len(provs)} provinces")
    for gi, g in enumerate(groups):
        s = sum(blocks[p][gi + 1][1][0] for p in provs)
        if s != blocks[COUNTRY][gi + 1][1][0]:
            raise SystemExit(f"Table 2: provinces sum to {s:,} {g}, the country to "
                             f"{blocks[COUNTRY][gi + 1][1][0]:,}")
    print(f"  Table 2: 70 blocks x 56 groups reconcile; 63 provinces sum to the country, "
          f"{NATIONAL:,}")
    return groups, provs


def read_retention():
    """Bieu 3.9: {group: (own, kinh, other_ethnic)} as fractions summing to 1."""
    import fitz
    doc = fitz.open(PDF_2024)
    ls = []
    for i in T39_PAGES:
        ls += [nfc(l) for l in doc[i].get_text().splitlines() if l.strip()]
    out = {}
    pat = re.compile(r"^(\d{1,2})\.\s*(.+)$")
    val = re.compile(r"^(\d{1,3},\d|-)$")
    k = 0
    while k < len(ls):
        m = pat.match(ls[k])
        if m and k + 4 < len(ls) and all(val.match(x) for x in ls[k + 1:k + 5]):
            v = [0.0 if x == "-" else float(x.replace(",", ".")) for x in ls[k + 1:k + 5]]
            own, other, kinh, oth_eth = v
            if abs(own + other - 100) > 0.15 or abs(kinh + oth_eth - other) > 0.15:
                raise SystemExit(f"Bieu 3.9: {m.group(2)} does not add up: {v}")
            tot = own + kinh + oth_eth
            out[nfc(m.group(2))] = (own / tot, kinh / tot, oth_eth / tot, v)
            k += 5
        else:
            k += 1
    if len(out) != 53:
        raise SystemExit(f"Bieu 3.9: {len(out)} groups, expected 53")
    return out


# Bieu 3.9 spells three groups differently from Table 2 (and drops a letter of one)
RET_ALIAS = {"Đê": "Ê Đê", "Khơ mú": "Khơ Mú", "Gié Triêng": "Gié Triêng"}


def key(name):
    s = nfc(name).lower()
    s = re.sub(r"^(tp\.?|thành phố)\s*", "", s)
    s = s.replace("-", " ").replace(".", " ")
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not PDF_2019.exists() or not PDF_2024.exists():
        fetch()

    blocks = read_table2()
    groups, provs = check_blocks(blocks)

    ret_raw = read_retention()
    ret = {}
    for g, v in ret_raw.items():
        g2 = g
        if g not in groups:
            cand = [x for x in groups if x.endswith(g) or key(x) == key(g)]
            g2 = RET_ALIAS.get(g) if RET_ALIAS.get(g) in groups else (cand[0] if len(cand) == 1 else None)
        if g2 is None or g2 not in groups:
            raise SystemExit(f"Bieu 3.9 group {g!r} not in Table 2")
        ret[g2] = v
    minorities = [g for g in groups if g not in (KINH, FOREIGN, NOT_STATED)]
    if sorted(ret) != sorted(minorities):
        raise SystemExit(f"Bieu 3.9 vs Table 2: {set(minorities) ^ set(ret)}")
    pd.DataFrame([dict(group=g, own_pct=v[3][0], other_pct=v[3][1], kinh_pct=v[3][2],
                       other_ethnic_pct=v[3][3], own=round(v[0], 6), kinh=round(v[1], 6),
                       other_ethnic=round(v[2], 6)) for g, v in ret.items()]
                 ).to_csv(OUT_RET, index=False)

    lut = pd.read_csv(LOOKUP, dtype=str)
    code = {key(n): u for u, n in zip(lut["unit"], lut["name"])}
    if len(code) != 63:
        raise SystemExit(f"vn_lookup.csv: {len(code)} names")
    miss = [p for p in provs if key(p) not in code]
    if miss:
        raise SystemExit(f"provinces not in religiondots' lookup: {miss}")
    if len({code[key(p)] for p in provs}) != 63:
        raise SystemExit("two provinces share a code")

    out = []
    for p in provs:
        u = code[key(p)]
        for (lab, v) in blocks[p][1:]:
            for suf, c in (("u", v[3]), ("r", v[6])):
                if c == 0:
                    continue
                gid = f"{u}-{suf}"
                if lab in ret:
                    own_s, kinh_s, oth_s, _ = ret[lab]
                    own = round(c * own_s)
                    kinh = round(c * kinh_s)
                    oth = c - own - kinh
                    if oth < 0:
                        kinh += oth
                        oth = 0
                    parts = ((lab, own), (lab + HOME_VI, kinh), (lab + HOME_OTHER, oth))
                else:
                    parts = ((lab, c),)
                for cat, n in parts:
                    if n > 0:
                        out.append((gid, "province_ur", p, cat, n, "derived"))
    df = pd.DataFrame(out, columns=["geo_id", "geo_level", "geo_name", "source_category",
                                    "count", "tier"])
    # every province's urban and rural totals are the census's
    tot = df.groupby("geo_id")["count"].sum()
    for p in provs:
        u = code[key(p)]
        h = blocks[p][0][1]
        for suf, c in (("u", h[3]), ("r", h[6])):
            if c and tot.get(f"{u}-{suf}", 0) != c:
                raise SystemExit(f"{p} {suf}: {tot.get(f'{u}-{suf}', 0):,} after the split, "
                                 f"census {c:,}")
    if df["count"].sum() != NATIONAL:
        raise SystemExit(f"vn.csv sums to {df['count'].sum():,}")
    df.to_csv(OUT, index=False)
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"  wrote {OUT.name}: {len(df):,} rows on {df['geo_id'].nunique()} province halves, "
          f"{df['count'].sum():,} people")
    print(nat.head(15).to_string())


if __name__ == "__main__":
    main()
