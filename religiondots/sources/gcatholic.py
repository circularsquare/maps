"""The Catholic Church's own diocesan statistics, from GCatholic.org, as a witness to where Catholics are.

Each diocese page on gcatholic.org carries the Annuario Pontificio's line for that diocese ("Population:
306,830 Catholics (70.8% of 433,300 total)", with the year it refers to) and its cathedral's place as
"town, region, country". `fetch(cc, code)` reads the country page, then every Latin and Eastern
diocese on it, and writes `data/raw/<cc>/gcatholic_dioceses.csv`. `by_seat(cc)` sums dioceses by the
region their cathedral stands in.

WHAT IT IS WORTH. These are the Church's own counts of the baptised against its own population
estimate, and they miss what people say when asked in both directions: Nigeria's dioceses claim
14.6% of the population as Catholic against 8-11% in the four DHS surveys, Cameroon's 26.9% against
37-39%. So they are never a level. Summed
by the cathedral's region they are a rank witness to a survey's geography, with two known blurs:
a diocese can cross a state or region line, and each diocese is filed whole under its seat.

robots.txt (2026-10-03) disallows only /temp/ for general agents. One request per second.

Used by: `sources/ng.py`, `sources/cm.py`.

Usage:
    python sources/gcatholic.py NG ng     fetch Nigeria into data/raw/ng/gcatholic_dioceses.csv
"""

import csv
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
COLS = ["id", "name", "catholics", "population", "asof", "seat"]


def path(cc):
    return os.path.join(ROOT, "data", "raw", cc, "gcatholic_dioceses.csv")


def fetch(code, cc):
    """Every diocese on gcatholic.org's country page `code` (ISO alpha-2, upper case)."""
    import requests

    out = path(cc)
    if os.path.exists(out):
        print(f"  have {out}")
        return
    get = lambda u: requests.get(u, headers={"User-Agent": UA}, timeout=120)
    page = get(f"https://gcatholic.org/dioceses/country/{code}.htm")
    page.raise_for_status()
    ids = re.findall(r"\['[^']*','s\d\d-(?:dioc|metr)','/dioceses/diocese/([a-z0-9]+)','([^']+)'",
                     page.text)
    if not ids:
        raise SystemExit(f"no dioceses found on gcatholic.org's {code} page; its layout has changed")
    rows = []
    for did, name in ids:
        time.sleep(1.0)
        d = get(f"https://gcatholic.org/dioceses/diocese/{did}.htm").text
        txt = " ".join(re.sub(r"<[^>]+>", " ", d).split())
        m = re.search(r"Population:\s*([\d,]+) Catholics \(([\d.]+)% of ([\d,]+) total\)", txt)
        st = re.search(r"\((\d{4}\.\d\d\.\d\d)\)\s*Area", txt)
        seat = re.search(r"'/churches/[^']+','[^']*','([^']+)'\]", d)
        rows.append({"id": did, "name": name,
                     "catholics": m.group(1).replace(",", "") if m else "",
                     "population": m.group(3).replace(",", "") if m else "",
                     "asof": st.group(1) if st else "",
                     "seat": seat.group(1) if seat else ""})
        print(f"    {name}: {rows[-1]['catholics'] or '-'} of {rows[-1]['population'] or '-'} "
              f"({rows[-1]['asof']}), seat {rows[-1]['seat']}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".part", out)
    print(f"  wrote {out} ({len(rows)} dioceses)")


def by_seat(cc, rename=None):
    """Catholics and population summed by the region of each diocese's cathedral, as a DataFrame
    indexed by region name (`rename` maps GCatholic's region names onto the caller's). Dioceses with
    no statistics line (a personal or Eastern-rite see with no territory) are left out."""
    import pandas as pd

    p = path(cc)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; fetch it with `python sources/gcatholic.py <CODE> {cc}`")
    d = pd.read_csv(p, dtype=str).fillna("")
    d = d[d["catholics"] != ""].copy()
    d["region"] = d["seat"].str.split(", ").str[-2]
    if rename:
        d["region"] = d["region"].replace(rename)
    for c in ("catholics", "population"):
        d[c] = d[c].astype(int)
    g = d.groupby("region")[["catholics", "population"]].sum()
    g["dioceses"] = d.groupby("region").size()
    g["share"] = g["catholics"] / g["population"]
    return g


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    fetch(sys.argv[1].upper(), sys.argv[2].lower())
