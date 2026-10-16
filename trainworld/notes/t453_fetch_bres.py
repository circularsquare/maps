"""BRES open access, employment by LSOA 2021 x SIC 2007 division, all England (T-453 stage 1)."""
import os, io, requests, pandas as pd
OUT = r"C:\Users\anita\projects\maps\trainworld\data\raw\t453\bres_lsoa_sic2_england.csv"
IND = {"45": 146800685, "47": 146800687, "55": 146800695, "56": 146800696, "90": 146800730, "91": 146800731,
       "92": 146800732, "93": 146800733, "94": 146800734, "95": 146800735, "96": 146800736}
URL = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv?date=latest&geography=2092957699TYPE151"
       "&industry=" + ",".join(str(v) for v in IND.values()) +
       "&employment_status=4&measure=1&measures=20100&select=date_name,geography_code,industry_code,obs_value&recordoffset={}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
parts, off = [], 0
while True:
    r = requests.get(URL.format(off), headers={"User-Agent": "trainworld-research/1.0"}, timeout=600)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype={"INDUSTRY_CODE": str})
    parts.append(df)
    print(off, len(df), flush=True)
    if len(df) < 25000:
        break
    off += len(df)
out = pd.concat(parts)
out.to_csv(OUT, index=False)
print(len(out), out.DATE_NAME.iloc[0], out.groupby("INDUSTRY_CODE").OBS_VALUE.sum().to_dict())
