"""T-2956: Egypt's population by kism and markaz for admin_totals "EG", from citypopulation.de's admin page, which
lists every governorate, kism and markaz with the 1996, 2006 and 2017 census counts and CAPMAS's 2023-07-01 estimate
under CAPMAS codes; COD-AB EGY's pcodes repeat them after "EG" (adm1 EGnn, adm2 EGnnnn). GHS-POP put 5.9M in Luxor
governorate against CAPMAS's 1.4M (notes/T-2897.md). The newest column is taken; each governorate's figure is checked
against the sum of its kism and markaz on the same page.

    python tools/country/egypt_census.py

Writes data/raw/admin_totals/EG/eg_markaz_2023.csv (code, name, pop; every level) and eg_governorates_2023.csv (the 27
governorates, which admin_totals reads: 19 of COD-AB's 365 kism and markaz have no row and 32 rows no polygon, the
kism made since COD-AB's file), and keeps the page beside them.
"""
import os
import re
import subprocess
import sys
from collections import defaultdict

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(TW, "data", "raw", "admin_totals", "EG")
URL = "https://www.citypopulation.de/en/egypt/admin/"
PAGE = os.path.join(DIR, "citypopulation_egy_admin.html")
OUT = os.path.join(DIR, "eg_markaz_2023.csv")
GOV = os.path.join(DIR, "eg_governorates_2023.csv")  # the one admin_totals reads: COD-AB's kism lag CAPMAS's


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(DIR, exist_ok=True)
    if not os.path.exists(PAGE):
        subprocess.run(["curl", "-sS", "-L", "--fail", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "-o", PAGE, URL], check=True)
    html = open(PAGE, encoding="utf-8", errors="replace").read()
    gov, units, names, sums = {}, {}, {}, defaultdict(int)
    for tr in re.findall(r'<tr [^>]*AdministrativeArea.*?</tr>', html, flags=re.S):
        m = re.search(r'id="i(\d+)"', tr)
        if not m:
            continue
        code = m.group(1)
        pops = [int(p.replace(",", "")) for p in re.findall(r'<td class="rpop[^"]*">([\d,]+)</td>', tr)]
        if not pops:
            continue
        name = re.search(r'<span itemprop="name">(.*?)</span>', tr, flags=re.S).group(1)
        if len(code) <= 2:
            gov[code.zfill(2)] = pops[-1]
            names[code.zfill(2)] = name
        elif len(code) == 4:
            units[code] = pops[-1]
            names[code] = name
            sums[code[:2]] += pops[-1]
    # the page's governorate figure and its units' sum differ by under 0.5% in a few (Alexandria, Aswan, Port Said:
    # 2023 estimates rounded per unit); the row is the sum, for admin_totals' exact parent check
    bad = [(g, gov[g], sums.get(g)) for g in gov if abs(gov[g] - sums.get(g, 0)) > 0.005 * gov[g]]
    if bad or len(gov) != 27:
        raise SystemExit(f"{len(gov)} governorates; one off its kism and markaz by over 0.5%: {bad}")
    for g in sorted(gov):
        if gov[g] != sums[g]:
            print(f"  EG{g} {names[g]}: page {gov[g]:,}, its units {sums[g]:,}")
    with open(OUT + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write("code,name,pop\n")
        f.write(f"EG,Egypt,{sum(sums.values())}\n")
        for g in sorted(gov):
            f.write(f"EG{g},\"{names[g]}\",{sums[g]}\n")
        for c in sorted(units):
            f.write(f"EG{c},\"{names[c]}\",{units[c]}\n")
    os.replace(OUT + ".part", OUT)
    with open(GOV + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write("code,name,pop\n")
        for g in sorted(gov):
            f.write(f"EG{g},\"{names[g]}\",{sums[g]}\n")
    os.replace(GOV + ".part", GOV)
    print(f"{len(gov)} governorates, {len(units)} kism and markaz, {sum(sums.values()):,} people -> {OUT}")


if __name__ == "__main__":
    main()
