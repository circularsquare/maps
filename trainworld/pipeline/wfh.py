"""T-042: work from home, from the ACS 2019-2023 5-year table-based summary files (no API key).

- B08301 (means of transportation to work), by tract: workers (_001) and worked from home (_021).
  Home side: each block's population times (1 - its tract's work-from-home share).
- B08126 (means of transportation to work by industry), summed over the city's tracts: the share
  working from home in each of 14 industries. Job side: each LODES sector's jobs times
  (1 - its industry's share), mapped CNS -> ACS industry below.

The nationwide .dat files (88 and 185 MB) are cut down to the city's tracts once and cached in
data/raw/ as small CSVs, so a rebuild does not need them.

ACS 2023 uses Connecticut's planning regions as county equivalents (tract codes unchanged since
2020), while the 2020 blocks use the old counties, so Connecticut tracts are matched on state plus
the 6-digit tract code.
"""
import os

import numpy as np
import pandas as pd

ACS_SF = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-{t}.dat"
INDUSTRIES = ["agriculture and mining", "construction", "manufacturing", "wholesale", "retail",
              "transport, warehousing, utilities", "information", "finance, insurance, real estate",
              "professional, management, admin", "education, health care, social assistance",
              "arts, accommodation, food", "other services", "public administration", "armed forces"]
# LODES CNS sector -> index in INDUSTRIES
CNS_TO_ACS = {"CNS01": 0, "CNS02": 0, "CNS03": 5, "CNS04": 1, "CNS05": 2, "CNS06": 3, "CNS07": 4,
              "CNS08": 5, "CNS09": 6, "CNS10": 7, "CNS11": 7, "CNS12": 8, "CNS13": 8, "CNS14": 8,
              "CNS15": 9, "CNS16": 9, "CNS17": 10, "CNS18": 10, "CNS19": 11, "CNS20": 12}
CNS = list(CNS_TO_ACS)


def tract_key(geoid11):
    """Join key for a 2020 tract GEOID: Connecticut by state + tract code, others whole."""
    g = pd.Series(geoid11).astype(str)
    return np.where(g.str[:2] == "09", "09-" + g.str[5:11], g)


def extract(table, cols, tract_keys, raw, fetch):
    """Rows of one ACS table for the given tracts, from a cached extract or the nationwide file."""
    path = os.path.join(raw, f"acs2023_5y_{table}_tracts.csv")
    want = set(tract_keys)
    if os.path.exists(path):
        d = pd.read_csv(path, dtype={"key": str})
        if want <= set(d["key"]):
            return d
    src = fetch(ACS_SF.format(t=table))
    out = []
    for ch in pd.read_csv(src, sep="|", usecols=["GEO_ID"] + cols, dtype={"GEO_ID": str}, chunksize=200_000):
        ch = ch[ch["GEO_ID"].str.startswith("1400000US")]
        ch = ch.assign(key=tract_key(ch["GEO_ID"].str[9:20]))
        out.append(ch[ch["key"].isin(want)])
    d = pd.concat(out, ignore_index=True)
    assert not d["key"].duplicated().any(), "tract key not unique"
    # tracts ACS lacks (renumbered since 2020) stay as empty rows, so the extract is complete
    lack = sorted(want - set(d["key"]))
    if lack:
        d = pd.concat([d, pd.DataFrame({"key": lack})], ignore_index=True)
    d.to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)
    return d


def tract_shares(blocks, raw, fetch):
    """Per block: its tract's work-from-home share (B08301). Returns (share array, stats)."""
    keys = tract_key(blocks["geoid"].str[:11].to_numpy())
    d = extract("b08301", ["B08301_E001", "B08301_E021"], keys, raw, fetch)
    d = d.set_index("key")
    workers = d["B08301_E001"].reindex(keys).to_numpy(np.float64)
    wfh = d["B08301_E021"].reindex(keys).to_numpy(np.float64)
    missing = np.isnan(workers)
    region = np.nansum(d["B08301_E021"]) / np.nansum(d["B08301_E001"])
    # a block whose 2020 tract ACS 2023 lacks (Suffolk renumbered some tracts after 2020) takes
    # its county's share; Connecticut keys have no county, so the region's
    cty = d[~d.index.str.startswith("09-")]
    cty = cty.groupby(cty.index.str[:5])[["B08301_E001", "B08301_E021"]].sum()
    cshare = (cty["B08301_E021"] / cty["B08301_E001"]).to_dict()
    fallback = np.array([cshare.get(g, region) for g in blocks["geoid"].str[:5]])
    share = np.where((~missing) & (workers > 0), wfh / np.where(workers > 0, workers, 1), fallback)
    stats = {"tracts": int(d["B08301_E001"].notna().sum()), "blocks_without_tract": int(missing.sum()),
             "pop_without_tract": float(blocks["pop"].to_numpy()[missing].sum()),
             "workers": float(d["B08301_E001"].sum()), "worked_from_home": float(d["B08301_E021"].sum()),
             "region_share": float(region)}
    return share, stats


def sector_rates(blocks, raw, fetch):
    """Work-from-home share per ACS industry over the city's tracts (B08126), and per CNS sector."""
    keys = tract_key(blocks["geoid"].str[:11].to_numpy())
    tot = [f"B08126_E{i:03d}" for i in range(2, 16)]
    home = [f"B08126_E{i:03d}" for i in range(92, 106)]
    d = extract("b08126", tot + home, keys, raw, fetch)
    t = d[tot].sum().to_numpy(np.float64)
    h = d[home].sum().to_numpy(np.float64)
    rate = np.where(t > 0, h / np.where(t > 0, t, 1), 0.0)
    by_cns = {c: float(rate[i]) for c, i in CNS_TO_ACS.items()}
    table = [{"industry": n, "workers": int(t[i]), "from_home": int(h[i]), "share": round(float(rate[i]), 4)}
             for i, n in enumerate(INDUSTRIES)]
    return by_cns, table


def commuters(blocks, raw, fetch):
    """Home-end and work-end commuter weights per block (before T-014's moves; build_city applies
    the moves to both). Home end: the block's resident job holders (LODES RAC, column `rac`, T-054;
    population where a city has none) x (1 - its tract's work-from-home share). Work end: sum over
    sectors of jobs x (1 - sector rate)."""
    share, hstats = tract_shares(blocks, raw, fetch)
    by_cns, table = sector_rates(blocks, raw, fetch)
    base = blocks["rac"] if "rac" in blocks else blocks["pop"]
    home = base.to_numpy(np.float64) * (1.0 - share)
    work = np.zeros(len(blocks))
    for c in CNS:
        work += blocks[c].to_numpy(np.float64) * (1.0 - by_cns[c])
    return home, work, by_cns, {"tracts": hstats, "industries": table}
