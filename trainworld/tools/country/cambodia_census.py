"""T-2000: Cambodia's 2019 census by district for admin_totals "KH", from citypopulation.de (no open machine-readable
district table exists, research/next-countries-data.md). Its admin page lists every district and commune with the
1998, 2008 and 2019 counts (household population) under NIS codes, which COD-AB KHM's ADM2 pcodes repeat after "KH".
Each district's figure is checked against the sum of its communes on the same page; a province row is the sum of its
communes, for admin_totals' parent check.

    python tools/country/cambodia_census.py [provinces]     default 01,02,13,22,24 (the north-west)

Writes data/raw/admin_totals/KH/kh_districts_2019.csv (code, name, pop) and keeps the page beside it.
"""
import os
import re
import subprocess
import sys
from collections import defaultdict

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(TW, "data", "raw", "admin_totals", "KH")
URL = "https://www.citypopulation.de/en/cambodia/admin/"
PAGE = os.path.join(DIR, "citypopulation_khm_admin.html")
OUT = os.path.join(DIR, "kh_districts_2019.csv")


def main():
    provs = (sys.argv[1] if len(sys.argv) > 1 else "01,02,13,22,24").split(",")
    os.makedirs(DIR, exist_ok=True)
    if not os.path.exists(PAGE):
        subprocess.run(["curl", "-sS", "-L", "--fail", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "-o", PAGE, URL], check=True)
    html = open(PAGE, encoding="utf-8", errors="replace").read()
    dist, names, comm = {}, {}, defaultdict(int)
    for tr in re.findall(r'<tr [^>]*AdministrativeArea.*?</tr>', html, flags=re.S):
        m = re.search(r'id="i(\d+)"', tr)
        if not m:
            continue
        code = m.group(1)
        pops = [int(p.replace(",", "")) for p in re.findall(r'<td class="rpop[^"]*">([\d,]+)</td>', tr)]
        if len(code) == 4:
            dist[code] = pops[-1]
            names[code] = re.search(r'<span itemprop="name">(.*?)</span>', tr, flags=re.S).group(1)
        elif len(code) == 6:
            comm[code[:4]] += pops[-1]
    mine = sorted(c for c in dist if c[:2] in provs)
    bad = [(c, dist[c], comm.get(c)) for c in mine if dist[c] != comm.get(c)]
    if bad or not mine:
        raise SystemExit(f"{len(mine)} districts; a district unequal to its communes: {bad}")
    with open(OUT + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write("code,name,pop\n")
        for p in provs:
            f.write(f"KH{p},province {p},{sum(v for c, v in comm.items() if c[:2] == p)}\n")
        for c in mine:
            f.write(f"KH{c},{names[c]},{dist[c]}\n")
    os.replace(OUT + ".part", OUT)
    print(f"{len(mine)} districts of provinces {','.join(provs)}, {sum(dist[c] for c in mine):,} people -> {OUT}")


if __name__ == "__main__":
    main()
