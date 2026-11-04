"""T-2079: Turkey's people by ilce for admin_totals "TR": the ADNKS address register at 31 December 2022 (COD-PS TUR,
HDX, CC BY-IGO) plus COD-PS's own table of Syrians under temporary protection by ilce, whom ADNKS leaves out. Codes are
COD-PS's ADM2 pcodes (TUR001001), which COD-AB TUR's tur_admin2 repeats; a province row is the sum of its ilce, the
national row the sum of all.

The 2023 earthquake (6 February 2023) came after the count: the check compares each province's 2022 register total with
the register's 2023-12-31 figure on citypopulation.de's Turkey admin page (provinces matched by name) and prints the
provinces moving most. Provinces in SCALE_2023 have their ilce scaled by that ratio (resident and protected alike).

    python tools/country/turkey_census.py

Writes data/raw/admin_totals/TR/tr_ilce_2022.csv (code, name, pop) and keeps the downloads beside it.
"""
import os
import re
import subprocess
import sys
import unicodedata

import pandas as pd

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(TW, "data", "raw", "admin_totals", "TR")
XLSX = os.path.join(DIR, "tur_admpop_2022.xlsx")
XLSX_URL = ("https://data.humdata.org/dataset/79d8aa16-f74e-4881-99a7-81cbd976b8c6/resource/"
            "71da478e-b15b-4add-a3de-2ca67b160fd3/download/tur_admpop_2022.xlsx")
PAGE = os.path.join(DIR, "citypopulation_tur_admin.html")
PAGE_URL = "https://www.citypopulation.de/en/turkey/admin/"
OUT = os.path.join(DIR, "tr_ilce_2022.csv")
# the earthquake provinces whose register lost over 4% in 2023 (Malatya 0.914, Hatay 0.916, Kahramanmaras 0.948,
# Adiyaman 0.952); elsewhere 2023 moves within about 6% either way, the usual drift of a year
SCALE_2023 = ["TUR044", "TUR031", "TUR046", "TUR002"]


def fold(s):
    s = unicodedata.normalize("NFKD", str(s).replace("ı", "i").replace("İ", "I")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def fetch(path, url, agent=None):
    if os.path.exists(path):
        return
    cmd = ["curl", "-sS", "-L", "--fail", "-o", path + ".part", url] + (["-A", agent] if agent else [])
    subprocess.run(cmd, check=True)
    os.replace(path + ".part", path)


def provinces_2023():
    """{folded province name: register total at 2023-12-31} from citypopulation's admin page (last column)."""
    fetch(PAGE, PAGE_URL, "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    html = open(PAGE, encoding="utf-8", errors="replace").read()
    out = {}
    for tr in re.findall(r'<tr class="rname"[^>]*AdministrativeArea.*?</tr>', html, flags=re.S):
        name = re.search(r'<span itemprop="name">(.*?)</span>', tr, flags=re.S).group(1)
        pops = [int(p.replace(",", "")) for p in re.findall(r'<td class="rpop[^"]*">([\d,]+)</td>', tr)]
        out[fold(name)] = pops[-1]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(DIR, exist_ok=True)
    fetch(XLSX, XLSX_URL)
    x = pd.ExcelFile(XLSX)
    res = x.parse("tur_admpop_adm2_2022")[["ADM1_EN", "ADM1_PCODE", "ADM2_EN", "ADM2_PCODE", "T_TL"]]
    sutp = x.parse("tur_admpop_sutp_adm2_2022")[["ADM2_PCODE", "T_TL"]].rename(columns={"T_TL": "sutp"})
    nat = int(x.parse("tur_admpop_adm0_2022").T_TL.iloc[0])
    if res.T_TL.sum() != nat:
        raise SystemExit(f"ilce sum {res.T_TL.sum():,} against the national row {nat:,}")
    df = res.merge(sutp, on="ADM2_PCODE", how="left").fillna({"sutp": 0})
    df["pop"] = df.T_TL + df.sutp
    print(f"register 2022 {df.T_TL.sum():,.0f} + under temporary protection {df.sutp.sum():,.0f} = {df['pop'].sum():,.0f}")

    p23 = provinces_2023()
    prov = df.groupby(["ADM1_PCODE", "ADM1_EN"])[["T_TL", "sutp"]].sum().reset_index()
    prov["y2023"] = prov.ADM1_EN.map(lambda n: p23.get(fold(n)))
    miss = prov[prov.y2023.isna()].ADM1_EN.tolist()
    if miss:
        print("no 2023 figure for:", miss)
    prov["ratio"] = prov.y2023 / prov.T_TL
    print("provinces by their 2023 register total over 2022's (furthest first):")
    for r in prov.reindex((prov.ratio - 1).abs().sort_values(ascending=False).index).head(14).itertuples():
        print(f"  {r.ADM1_PCODE} {r.ADM1_EN:16} 2022 {r.T_TL:>11,.0f}  2023 {r.y2023:>11,.0f}  x{r.ratio:.3f}  protected {r.sutp:>9,.0f}")
    for code in SCALE_2023:
        k = float(prov.loc[prov.ADM1_PCODE == code, "ratio"].iloc[0])
        df.loc[df.ADM1_PCODE == code, "pop"] *= k
        print(f"  {code} scaled x{k:.3f}")

    df["pop"] = df["pop"].round().astype(int)
    units = pd.DataFrame({"code": df.ADM2_PCODE, "name": df.ADM2_EN, "pop": df["pop"]})
    parents = df.groupby(["ADM1_PCODE", "ADM1_EN"])["pop"].sum().reset_index()
    parents = pd.DataFrame({"code": parents.ADM1_PCODE, "name": parents.ADM1_EN, "pop": parents["pop"]})
    out = pd.concat([pd.DataFrame({"code": ["TUR"], "name": ["Turkiye"], "pop": [int(units["pop"].sum())]}), parents, units])
    out.to_csv(OUT + ".part", index=False)
    os.replace(OUT + ".part", OUT)
    print(f"{OUT}: {len(units)} ilce, {len(parents)} provinces, {units['pop'].sum():,} people")


if __name__ == "__main__":
    main()
