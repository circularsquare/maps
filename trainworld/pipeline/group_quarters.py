"""T-234: people in institutional group quarters (prisons and jails, juvenile facilities, nursing
homes, other institutions) are not homes: the census counts them where they are held or cared for,
but they do not commute, shop or take children to school. Their number is taken off each block's
population before anything is built on it.

Source: the 2020 Census redistricting file (PL 94-171) table P5, group quarters population by
major type, per block, from each state's summary file on www2.census.gov (no login or key; the
Census API refused keyless requests on 2026-10-13). Every US state has it; a small per-state table
is cached in data/raw/pl2020_p5_<st>.csv.gz and the state zip can go after. Removed: P5_003N correctional facilities for adults,
P5_004N juvenile facilities, P5_005N nursing facilities/skilled-nursing facilities, P5_006N other
institutional facilities. Kept: the non-institutional types (P5_008N college/university student
housing, P5_009N military quarters, P5_010N other non-institutional), whose residents travel.

`build_city.py` does this as it reads the blocks (`blocks_for_state`).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import io
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PL_ZIP = "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/{name}/{ab}2020.pl.zip"
PL_STATES = {"01": ("Alabama", "al"), "02": ("Alaska", "ak"), "04": ("Arizona", "az"), "05": ("Arkansas", "ar"),
             "06": ("California", "ca"), "08": ("Colorado", "co"), "09": ("Connecticut", "ct"), "10": ("Delaware", "de"),
             "11": ("District_of_Columbia", "dc"), "12": ("Florida", "fl"), "13": ("Georgia", "ga"),
             "15": ("Hawaii", "hi"), "16": ("Idaho", "id"), "17": ("Illinois", "il"), "18": ("Indiana", "in"),
             "19": ("Iowa", "ia"), "20": ("Kansas", "ks"), "21": ("Kentucky", "ky"), "22": ("Louisiana", "la"),
             "23": ("Maine", "me"), "24": ("Maryland", "md"), "25": ("Massachusetts", "ma"), "26": ("Michigan", "mi"),
             "27": ("Minnesota", "mn"), "28": ("Mississippi", "ms"), "29": ("Missouri", "mo"), "30": ("Montana", "mt"),
             "31": ("Nebraska", "ne"), "32": ("Nevada", "nv"), "33": ("New_Hampshire", "nh"), "34": ("New_Jersey", "nj"),
             "35": ("New_Mexico", "nm"), "36": ("New_York", "ny"), "37": ("North_Carolina", "nc"),
             "38": ("North_Dakota", "nd"), "39": ("Ohio", "oh"), "40": ("Oklahoma", "ok"), "41": ("Oregon", "or"),
             "42": ("Pennsylvania", "pa"), "44": ("Rhode_Island", "ri"), "45": ("South_Carolina", "sc"),
             "46": ("South_Dakota", "sd"), "47": ("Tennessee", "tn"), "48": ("Texas", "tx"), "49": ("Utah", "ut"),
             "50": ("Vermont", "vt"), "51": ("Virginia", "va"), "53": ("Washington", "wa"), "54": ("West_Virginia", "wv"),
             "55": ("Wisconsin", "wi"), "56": ("Wyoming", "wy"), "72": ("Puerto_Rico", "pr")}
# P5 group quarters population by major type (2020 PL 94-171), and the block total for a check
ALL = ["P1_001N", "P5_001N", "P5_002N", "P5_003N", "P5_004N", "P5_005N", "P5_006N",
       "P5_007N", "P5_008N", "P5_009N", "P5_010N"]
REMOVED = {"P5_003N": "correctional facilities for adults", "P5_004N": "juvenile facilities",
           "P5_005N": "nursing and skilled-nursing facilities", "P5_006N": "other institutional facilities"}
KEPT = {"P5_008N": "college and university student housing", "P5_009N": "military quarters",
        "P5_010N": "other non-institutional facilities"}
SOURCE = ("2020 Census redistricting file (PL 94-171), table P5 group quarters population by major "
          "type, per block (state summary files): institutional types (adult correctional, juvenile, "
          "nursing, other institutional) are taken off each block's population")
SOURCE_URL = "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/"


def state_table(st, fetch):
    """Every 2020 block of a state (2-digit FIPS) with P1_001N and P5, from the state's PL 94-171
    summary file: the geographic header (SUMLEV 750 = block, LOGRECNO, GEOCODE = block GEOID),
    segment 1 (P1) and segment 3 (P5). Kept as a small table in data/raw/pl2020_p5_<st>.csv.gz;
    the state zip is needed only to make it and can be deleted after."""
    import build_city as bc
    out = os.path.join(bc.RAW, f"pl2020_p5_{st}.csv.gz")
    if os.path.exists(out):
        return pd.read_csv(out, dtype={"geoid": str})
    name, ab = PL_STATES[st]
    zpath = fetch(PL_ZIP.format(name=name, ab=ab))
    with zipfile.ZipFile(zpath) as z:
        def member(suffix):
            m = next(m for m in z.namelist() if m.lower().endswith(suffix))
            return io.TextIOWrapper(z.open(m), encoding="latin-1")
        geo = pd.read_csv(member("geo2020.pl"), sep="|", header=None, dtype=str, usecols=[2, 7, 9],
                          names=None, quoting=3)
        geo.columns = ["sumlev", "logrecno", "geoid"]
        geo = geo[geo["sumlev"] == "750"]
        s1 = pd.read_csv(member("000012020.pl"), sep="|", header=None, dtype=str, usecols=[4, 5], quoting=3)
        s1.columns = ["logrecno", "P1_001N"]
        s3 = pd.read_csv(member("000032020.pl"), sep="|", header=None, dtype=str, usecols=[4] + list(range(5, 15)), quoting=3)
        s3.columns = ["logrecno"] + [f"P5_{i:03d}N" for i in range(1, 11)]
    d = geo.merge(s1, on="logrecno").merge(s3, on="logrecno")
    assert len(d) == len(geo), "PL segments do not cover every block"
    d = d[["geoid"] + ALL]
    for k in ALL:
        d[k] = d[k].astype(np.int64)
    assert (d["geoid"].str.len() == 15).all()
    d.to_csv(out + ".part", index=False, compression="gzip")
    os.replace(out + ".part", out)
    print(f"  PL 94-171 {ab.upper()}: {len(d):,} blocks, population {d['P1_001N'].sum():,}, "
          f"group quarters {d['P5_001N'].sum():,} ({d['P5_002N'].sum():,} institutional)", flush=True)
    return d


def by_block(counties, fetch):
    """One row per 2020 block of the counties (5-digit FIPS): geoid, every ALL column, and
    gq_inst, the sum of the REMOVED types."""
    frames = []
    for st in sorted({c[:2] for c in counties}):
        d = state_table(st, fetch)
        frames.append(d[d["geoid"].str[:5].isin(counties)])
    d = pd.concat(frames, ignore_index=True)
    d["gq_inst"] = d[list(REMOVED)].sum(axis=1)
    assert (d["P5_002N"] == d["gq_inst"]).all(), "P5 institutional types do not add up to P5_002N"
    assert (d["gq_inst"] <= d["P1_001N"]).all(), "more people in institutions than on the block"
    return d


def remove(blocks, counties, fetch):
    """`blocks` (geoid, pop = census population): returns it with pop less the institutional group
    quarters, and pop_census, gq_inst beside it; the census counts must match P1 exactly."""
    gq = by_block(counties, fetch)
    m = blocks.merge(gq[["geoid", "P1_001N", "gq_inst"]], on="geoid", how="left")
    missing = m["P1_001N"].isna()
    assert not missing.any(), f"{int(missing.sum())} blocks missing from the PL file"
    assert (m["P1_001N"].astype(np.int64).to_numpy() == blocks["pop"].to_numpy()).all(), "TIGER POP20 differs from PL P1_001N"
    out = blocks.copy()
    out["pop_census"] = blocks["pop"].to_numpy()
    out["gq_inst"] = m["gq_inst"].astype(np.int64).to_numpy()
    out["pop"] = out["pop_census"] - out["gq_inst"]
    return out


def cap_job_holders(df):
    """On a block with people in institutions, its resident job holders (LODES RAC, `rac`) can be
    no more than its remaining residents (`pop`, after `remove`). LODES puts job holders on such
    blocks (Rikers Island: 676 on blocks with 27 residents outside the jails), from administrative
    addresses and its noise; inmates and nursing-home residents hold almost no jobs. Other blocks
    keep LODES as it is. `rac_lodes` keeps the LODES figure."""
    out = df.copy()
    out["rac_lodes"] = df["rac"].to_numpy()
    inst = df["gq_inst"].to_numpy() > 0
    out["rac"] = np.where(inst, np.minimum(df["rac"].to_numpy(), df["pop"].to_numpy()), df["rac"].to_numpy())
    return out


def summary(counties, fetch, names=None):
    """Per county: census population, each group-quarters type, and what is removed."""
    gq = by_block(counties, fetch)
    g = gq.groupby(gq["geoid"].str[:5])[ALL + ["gq_inst"]].sum()
    if names:
        g.insert(0, "name", [names.get(k, k) for k in g.index])
    return g


POP_SOURCE = ("2020 Census, TIGER/Line TABBLOCK20 POP20 less institutional group quarters (PL 94-171 P5, "
              "T-234), spread over each block polygon (TIGER2020PL) by land area")
HOME_RULE = ("LODES RAC jobs by home block, capped on blocks with institutional group quarters at the "
             "block's remaining residents (T-234), x (1 - tract work-from-home share)")
