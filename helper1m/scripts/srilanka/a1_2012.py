"""Reader for CPH 2012 district report Table A1, "Population by divisional secretariat
division, sex and sector" (one single-page PDF per district).

The text layer comes out as a token stream: the DS name (possibly over two lines), then
twelve cells (total / urban / rural / estate, each both sexes, male, female), a cell being
a number or "-". Some districts print fewer sector columns. The first row is the district.

read(district) -> (district_total, [(ds_name, total), ...]) with checks:
both sexes = male + female on every row, DS rows sum to the district row.
"""
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
DIR = HELPER / "data" / "srilanka" / "raw" / "cph2012"
NUM = re.compile(r"^(\d{1,3}(,\d{3})*|-)$")
FURNITURE = re.compile(r"^(Census of Population and Housing|Divisional secretariat division"
                       r"|All sectors|Urban Sector|Rural Sector|Estate Sector)|^((both sexes|male|female)\s*)+$",
                       re.I)
HEADER = {"both sexes", "both", "sexes", "male", "female", "total", "urban", "rural", "estate",
          "district / ds division", "ds division", "district"}


def _tokens(path):
    import fitz
    doc = fitz.open(path)
    toks = []
    for page in doc:
        for line in page.get_text().split("\n"):
            t = line.strip().replace("‐", "-").replace("–", "-")
            if t:
                toks.append(t)
    return toks


def read(district):
    toks = _tokens(DIR / f"{district}_A1.pdf")
    rows, name, vals = [], [], []

    def flush():
        if name and vals:
            nums = [None if v == "-" else int(v.replace(",", "")) for v in vals]
            rows.append((" ".join(name), nums))

    for t in toks:
        if NUM.match(t):
            vals.append(t)
            continue
        if t.lower().startswith("table a1") or t.lower() in HEADER or FURNITURE.search(t):
            continue
        if vals:                       # a name after values starts a new row
            flush()
            name, vals = [], []
        name.append(t)
    flush()

    out = []
    for nm, nums in rows:
        if re.search(r"Source|Census of|sector", nm, re.I) and len(nums) < 3:
            continue                   # page furniture (footnote marks, headers)
        tot, m, f = (nums + [None] * 3)[:3]
        if tot is None or m is None or f is None or tot != m + f:
            raise SystemExit(f"{district}: row {nm!r} {nums} fails both = male + female")
        if len(nums) % 3:
            raise SystemExit(f"{district}: row {nm!r} has {len(nums)} cells")
        out.append((re.sub(r"\s+", " ", nm).strip(), tot))
    (dname, dtot), ds = out[0], out[1:]
    s = sum(v for _, v in ds)
    if s != dtot:
        raise SystemExit(f"{district}: DS rows sum to {s:,}, district row {dtot:,}")
    return dtot, ds


if __name__ == "__main__":
    import download
    grand = n = 0
    for d in download.DISTRICTS_2012:
        tot, ds = read(d)
        grand += tot
        n += len(ds)
        print(f"{d:<14} {tot:>10,}  {len(ds):>2} DS  " + "; ".join(x for x, _ in ds))
    print(f"total {grand:,} in {n} DS divisions")
