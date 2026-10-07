"""Parse the NSC workbook "Численность постоянного населения областей, районов,
городов, айылных аймаков и айылов (сел)" (stat.gov.kg operational/825) into one
row per SOATE code, with the kind of territory the code stands for.

SOATE codes are 14 digits: 417 | OO oblast | RRR rayon | AAA aiyl aimak | VVV place.
  RRR 2xx/3xx is a rayon, 4xx a city of oblast significance.
  AAA 8xx is an aiyl aimak, 5xx/6xx a town or urban-type settlement (pgt),
  and 800 under a city or town is the "villages under the city" pseudo-unit.
Talas's codes are written with spaces ("41707 000 000 00 0"), so digits are
pulled out rather than strings compared.
"""
import re

import pandas as pd

BISHKEK, OSH = "41711000000000", "41721000000000"
NARYN_CITY = "41704000000010"      # coded without its 4xx rayon segment
ARAVAN = "41706211800000"          # Aravan rayon, coded with an 800 segment


def classify(code):
    o, r, a, v = code[3:5], code[5:8], code[8:11], code[11:14]
    if o == "00":
        return "national"
    if code == ARAVAN:
        return "rayon"
    if code == NARYN_CITY:
        return "city"
    if r == a == v == "000":
        return "oblast"
    if o in ("11", "21"):
        return "capital_part"          # Bishkek/Osh sub-rows; the city row is inclusive
    if r[0] == "4":
        if a == "000" and v == "010":
            return "city"
        if a[0] == "5" and v[0] == "0":
            return "city_pgt"
        if a == "800" and v == "000":
            return "city_pseudo"
        return "city_village"
    if r == "000":                     # Naryn city's own villages
        return "city_village" if a == "800" or v != "000" else "other"
    if a == "000" and v == "000":
        return "rayon"
    if a[0] == "8" and v == "000":
        return "aa"
    if a[0] in "456":                  # 4xx: Kok-Jangak, a city inside Suzak rayon, in 2024
        if v[0] == "0" and v != "000":
            return "town"
        if v == "800":
            return "town_pseudo"
        return "town_village"
    if a[0] == "8":
        return "village"
    return "other"


def parse(path):
    """Return a DataFrame: sheet, code, name, pop (float, NaN for '-'), kind."""
    x = pd.ExcelFile(path)
    rows = []
    for s in x.sheet_names:
        df = pd.read_excel(path, sheet_name=s, header=None, dtype=object)
        for r in df.itertuples(index=False):
            if pd.isna(r[0]):
                continue
            code = re.sub(r"\D", "", str(r[0]))
            if len(code) != 14 or not code.startswith("417"):
                continue
            name = re.sub(r"\s+", " ", str(r[1])).strip() if pd.notna(r[1]) else ""
            pop = pd.to_numeric(r[2], errors="coerce") if len(r) > 2 else float("nan")
            rows.append((s, code, name, pop))
    out = pd.DataFrame(rows, columns=["sheet", "code", "name", "pop"])
    out = out.drop_duplicates("code")
    out["kind"] = out.code.map(classify)
    out.loc[out.code == ARAVAN, "code"] = "41706211000000"   # give it a normal rayon code
    out["oblast"] = out.code.str[:5] + "000000000"
    out["rayon"] = out.code.str[:8] + "000000"
    # the territory a place row belongs to
    out["unit"] = None
    k = out.kind
    out.loc[k.isin(["aa", "town", "city", "city_pgt"]), "unit"] = out.code
    out.loc[k == "village", "unit"] = out.code.str[:11] + "000"
    out.loc[k == "city_village", "unit"] = out.code.str[:8] + "000010"
    out.loc[(k == "city_village") & (out.code.str[:8] == "41704000"), "unit"] = NARYN_CITY
    cv = (k == "capital_part") & (out.code.str[11:14] != "000")
    out.loc[cv, "unit"] = out.loc[cv, "oblast"]
    # villages under a town belong to the town row that shares their first 11 digits
    # (two towns can share 11 digits, 600010 and 600020; villages go to the first)
    towns = out[k == "town"].set_index(out[k == "town"].code.str[:11]).code
    towns = towns[~towns.index.duplicated()]
    tv = k.isin(["town_village", "town_pseudo"])
    out.loc[tv, "unit"] = out.loc[tv, "code"].str[:11].map(towns)
    return out.reset_index(drop=True)
