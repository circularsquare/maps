"""Andorra: language at home from the World Values Survey wave 7 (2018), national shares on the
2018 population -> data/normalized/ad.csv.

    python sources/ad_wvs.py [--fetch]

NO CENSUS (Andorra keeps a population register, which records nationality, not language). Under
AGENT_BRIEF §2's survey ruling (2026-10-05) the open source with a home-language item is the World
Values Survey wave 7, Andorra 2018 (Institut d'Estudis Andorrans; 1,004 face-to-face interviews,
residents 18+, unweighted: W_WEIGHT is "No weighting"), item Q272 "Language at home". Its
frequencies are read off the International Household Survey Network catalogue page for the same
file (catalog.ihsn.org/catalog/11550/variable/F1/V312, saved as data/raw/ad/ihsn_11550_Q272.html),
the route religiondots used for Q289 (../religiondots/sources/ad.py). National only.

POPULATION: 76,177 on 31 December 2018, the Observatori Social's series religiondots pinned
(../religiondots/sources/ad.py POP), the survey's year.

Rows `modelled`: survey share x population, largest-remainder rounded.

CHECKS: the page is Q272 of the WVS-7 Andorra file; the six codes sum to N = 1,004 and equal the
pinned counts.
"""
import html
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "ad" / "ihsn_11550_Q272.html"
OUT = HERE / "data" / "normalized" / "ad.csv"
URL = "https://catalog.ihsn.org/catalog/11550/variable/F1/V312?name=Q272"
N, POP = 1_004, 76_177
Q272 = {810: ("Catalan; Valencian", 355), 1270: ("Spanish; Castilian", 369),
        1400: ("French", 53), 3530: ("Portuguese", 113), 9000: ("Other", 34),
        9040: ("Other European", 80)}
ROW = re.compile(r"<tr>\s*<td>(-?\d+)</td>\s*<td>([^<]*)</td>\s*<td>(\d+)</td>", re.S)


def fetch():
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"})
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(urllib.request.urlopen(req, timeout=120, context=ctx).read())


def main():
    if "--fetch" in sys.argv or not RAW.exists():
        fetch()
    t = RAW.read_text(encoding="utf-8")
    assert "WVS_Wave_7_Andorra" in t and re.search(r"<h2>[^<]*\(Q272\)</h2>", t)
    got = {int(c): (html.unescape(lab).strip(), int(n)) for c, lab, n in ROW.findall(t)}
    assert got == Q272, got
    assert sum(n for _, n in Q272.values()) == N
    raw = {lab: POP * n / N for lab, n in Q272.values()}
    cnt = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:POP - sum(cnt.values())]:
        cnt[k] += 1
    df = pd.DataFrame([dict(geo_id="AD", geo_level="country", geo_name="Andorra",
                            source_category=k, count=v, tier="modelled", year=2018,
                            note=f"WVS-7 Q272: {dict(Q272.values())[k]} of {N}")
                       for k, v in cnt.items()])
    assert df["count"].sum() == POP
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {POP:,} people")
    print(df[["source_category", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
