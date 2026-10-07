"""Latinobarómetro, read one country at a time: wave, `ciudad` (place) code and label, religion, weight.

A second instrument beside LAPOP for the places LAPOP never sampled. Corporación Latinobarómetro
(Santiago) puts every wave's Stata file on its site as a direct zip with no form; the data-use
note on `latinobarometro.org/agregados` allows non-commercial research and publication and
forbids re-publishing the data files, so the zips stay in data/raw and only shares leave it.
sources.md §scout-2026-10-03-gaps asked for it; `sources/co.md` §11 and `sources/ve.md` §14 are
what it found.

THE PLACE COLUMN IS `ciudad`, NOT `reg`. `reg` changes meaning between waves (21 Venezuelan states in
2005, eight regions in 2020) and holds design regions for Colombia. `ciudad` is a nine-digit code,
country (3) + department or state (3) + municipality (3, 000 when the wave records only the
department), and the same code carries the same label in every wave that labels it (checked over
564 Colombian and Venezuelan codes, 2000-2024; the only differences are accents). The department
digits follow the alphabetical order of each country's units in Spanish, not the national code
(Colombia 001 Amazonas, 012 Chocó; Venezuela 001 Amazonas, 009 Delta Amacuro, 016 Nueva Esparta).
2018's file leaves `ciudad` unlabelled; its codes take the label the same code has in other waves.

TRAPS FOUND, each asserted where it is used:
  * 2007: 65 Colombian respondents carry `ciudad` 600008002, a Paraguayan code (Itapúa, Capitán
    Meza). Their real place is unknown; `colombia_rows` drops exactly them.
  * 1995-2002 code `Ninguna` as 15 and `Otras` as 16; from 2003 they are 97 and 96. CARD_WAVES starts
    in 2003 and `load` asserts the label of every code it groups, wave by wave.
  * The card's Protestant boxes swap from wave to wave, as LAPOP's do: Colombia's `Protestante` is
    112 respondents and `Evangélica sin especificar` 30 in 2010, and 2 and 179 in 2015. Only the
    groups in GROUPS are compared with anything.

Usage:
    python sources/latinobarometro.py --fetch    download the Stata zips for CARD_WAVES (~100 MB)
"""

import io
import os
import re
import ssl
import sys
import urllib.request
import warnings
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "latinobarometro")
SITE = "https://www.latinobarometro.org/documents/LAT-{y}/{f}"

# wave -> Stata zip on the site (Spanish edition), as linked from latinobarometro.org/latinobarometro-<y>
FILES = {
    2003: "latinobarometro-2003-dta.zip", 2004: "latinobarometro-2004-dta.zip",
    2005: "latinobarometro-2005-dta.zip", 2006: "latinobarometro-2006-dta.zip",
    2007: "latinobarometro-2007-dta.zip", 2008: "latinobarometro-2008-dta.zip",
    2009: "latinobarometro-2009-dta.zip", 2010: "latinobarometro-2010-dta.zip",
    2011: "latinobarometro-2011-dta.zip", 2013: "latinobarometro-2013-dta.zip",
    2015: "latinobarometro-2015-dta.zip", 2016: "latinobarometro2016-dta.zip",
    2017: "latinobarometro2017-dta.zip", 2018: "latinobarometro-2018-esp-dta-v20190303.zip",
    2020: "latinobarometro-2020-esp-stata-v1-0.zip", 2023: "latinobarometro-2023-stata-v1-0.zip",
    2024: "latinobarometro-2024-stata-v20250817.zip",
}
CARD_WAVES = sorted(FILES)
# the religion question, by wave
REL = {2003: "p91st", 2004: "p90st", 2005: "s2", 2006: "s2", 2007: "s4", 2008: "s5", 2009: "s7",
       2010: "S9", 2011: "S18", 2013: "S14", 2015: "S16", 2016: "S8", 2017: "S9", 2018: "S5",
       2020: "s10", 2023: "S1", 2024: "S1"}
# self-identified race or ethnicity, where a wave asks it: a witness that uses no place name
RACE = {2016: "S9", 2017: "S10", 2018: "S6", 2020: "s12", 2023: "S7", 2024: "S7"}
COUNTRY = {"co": 170, "ve": 862}

# The answer groups compared with LAPOP. LAPOP's card puts Adventists and Baptists under
# `Evangélica y Pentecostal` and Lutherans, Methodists and `Cristiano` under traditional Protestant,
# so only the two boxes together match anything on this card. Witnesses (7) and Mormons (8) are
# outside both cards' Protestant groups.
GROUPS = {"cath": [1], "prot": [2, 3, 4, 5, 6, 10], "none": [12, 13, 14, 97]}
# what every grouped code's label must start with, folded, in every wave read
LABEL_STARTS = {1: "catol", 2: "evangelica", 3: "evangelica", 4: "evangelica", 5: "evangelica",
                6: "adventista", 10: "protestante", 12: "creyente", 13: "agnostico", 14: "ateo",
                97: "ningun"}

warnings.filterwarnings("ignore", module="pandas.io.stata")


def fold(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def unmojibake(s):
    """Older files store UTF-8 labels that pandas decodes as Latin-1."""
    s = str(s)
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def fetch(waves=CARD_WAVES):
    os.makedirs(RAW, exist_ok=True)
    ctx = ssl._create_unverified_context()      # the host's certificate chain is incomplete
    for y in waves:
        p = os.path.join(RAW, FILES[y])
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        req = urllib.request.Request(SITE.format(y=y, f=FILES[y]),
                                     headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, context=ctx, timeout=300).read()
        with open(p + ".part", "wb") as f:
            f.write(data)
        os.replace(p + ".part", p)
        print(f"  {y}: {len(data):,} bytes -> {p}")


def _dta(y):
    p = os.path.join(RAW, FILES[y])
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing -- run python sources/latinobarometro.py --fetch")
    zf = zipfile.ZipFile(p)
    names = [n for n in zf.namelist() if n.lower().endswith(".dta") and "__macosx" not in n.lower()]
    esp = [n for n in names if not re.search(r"eng", n, re.I)] or names
    if len(esp) != 1:
        raise SystemExit(f"{y}: expected one Spanish .dta in {p}, found {esp}")
    return io.BytesIO(zf.read(esp[0]))


def load(cc, waves):
    """One country's rows: wave, ciudad, ciudad_label, rel, rel_label, wt, race_label.

    Asserts every grouped code's label in every wave. `ciudad_label` is filled from the same code
    in another wave where the wave has none (2018).
    """
    code = COUNTRY[cc]
    frames = []
    for y in waves:
        it = pd.read_stata(_dta(y), iterator=True, convert_categoricals=False)
        vlab = it.value_labels()
        lbl_of = dict(zip(it._varlist, it._lbllist))
        d = it.read()
        cols = {c.lower(): c for c in d.columns}
        ctry, city = cols["idenpa"], cols["ciudad"]
        wt = cols.get("wt")
        rel = REL[y]
        d = d[d[ctry] == code]
        clab = vlab.get(lbl_of.get(city, ""), {})
        rlab = vlab.get(lbl_of.get(rel, ""), {})
        for k, start in LABEL_STARTS.items():
            if k in rlab and not fold(unmojibake(rlab[k])).startswith(start):
                raise SystemExit(f"{y}: religion code {k} is labelled {unmojibake(rlab[k])!r}, "
                                 f"expected {start}...")
        used = set(d[rel].dropna().astype(int)) & set(LABEL_STARTS)
        if used - set(rlab):
            raise SystemExit(f"{y}: grouped codes with no label: {sorted(used - set(rlab))}")
        race = RACE.get(y)
        if race is not None:
            race = cols.get(race.lower())
            slab = vlab.get(lbl_of.get(race, ""), {})
        f = pd.DataFrame({
            "wave": y,
            "ciudad": d[city].astype(float).fillna(-1).astype("int64"),
            "ciudad_label": d[city].map(lambda k: unmojibake(clab.get(k, ""))),
            "rel": d[rel].astype(float).fillna(-9).astype(int),
            "rel_label": d[rel].map(lambda k: unmojibake(rlab.get(k, ""))),
            "wt": pd.to_numeric(d[wt], errors="coerce").fillna(1.0) if wt else 1.0,
            "race_label": (d[race].map(lambda k: unmojibake(slab.get(k, ""))) if race else ""),
        })
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    known = out[out["ciudad_label"] != ""].groupby("ciudad")["ciudad_label"].first()
    out.loc[out["ciudad_label"] == "", "ciudad_label"] = out["ciudad"].map(known).fillna("")
    return out


def group_shares(d, unit_col, w_col="wt"):
    """Weighted share of each GROUPS group per unit, over respondents with an answer (rel > 0)."""
    d = d[d["rel"] > 0]
    tot = d.groupby(unit_col)[w_col].sum()
    out = pd.DataFrame({g: d[d["rel"].isin(cs)].groupby(unit_col)[w_col].sum()
                        .reindex(tot.index, fill_value=0.0) / tot for g, cs in GROUPS.items()})
    out["n"] = d.groupby(unit_col).size()
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--fetch" in sys.argv:
        fetch()
    else:
        print(__doc__)
