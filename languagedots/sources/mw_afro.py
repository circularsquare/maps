"""Malawi: home language by ethnic group, from six pooled Afrobarometer rounds (2008-2022).

    python sources/mw_afro.py --fetch   extract Malawi's rows from religiondots' six merged .sav
                                        files (read-only) -> data/raw/mw/ab_mw_language.csv

This is the retention check of AGENT_BRIEF §2 for Malawi's ethnicity-only census: for each
respondent, the ethnic group they name and the language they speak at home. sources/mw_census.py
reads the extract and turns it into shares per (census major group, zone). The record is
sources/mw.md.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
AB_DIR = RD / "data" / "raw" / "afrobarometer"
RAW = HERE / "data" / "raw" / "mw"
EXTRACT = RAW / "ab_mw_language.csv"

# (round, file, language column, verbatim column, weight column, district column or None,
#  ethnic group column, its verbatim column): the same columns sources/ng_afro.py checked.
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Q3OTHER", "Withinwt", "DISTRICT", "Q79", "Q79OTHER"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "Q2OTHER", "withinwt", None, "Q84", "Q84OTHER"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt", "LOCATION.LEVEL.1",
     "Q87", "Q87OTHER"),
    # R7: Q2A "Respondent's mother tongue", not Q2B "Language spoken in home" (Anita's ruling
    # on ask 018, 2026-10-05: lingua francas at R7's mother-tongue question)
    (7, "r7_merged_data_34ctry.release.sav", "Q2A", "Q2AOTHER", "withinwt", "LOCATION.LEVEL.1",
     "Q84", "Q84OTHER"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q2OTHER", "withinwt_hh", None, "Q81", "Q81OTHER"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "LOCATION.LEVEL.1", "Q84A", "Q84AOTHER"),
]
ETH_LABEL = ("tribe or ethnic group", "ethnic community")
LANG_LABEL = ("language of respondent", "language spoken in home", "mother tongue")


def say(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        raise SystemExit(msg)


def _read(path, cols=None, metadataonly=False):
    import pyreadstat
    kw = dict(metadataonly=True) if metadataonly else dict(usecols=cols)
    try:
        return pyreadstat.read_sav(str(path), **kw)
    except Exception:  # noqa: BLE001  R6 is not valid UTF-8
        return pyreadstat.read_sav(str(path), encoding="LATIN1", **kw)


def fetch():
    out = []
    for rnd, name, q, qo, wt, dist, eth, etho in ROUNDS:
        p = AB_DIR / name
        if not p.exists():
            raise SystemExit(f"{p} missing: religiondots' `python sources/ng.py --fetch` "
                             "downloads the six merged rounds")
        _, meta = _read(p, metadataonly=True)
        up = {c.upper(): c for c in meta.column_names}
        want = ["COUNTRY", "REGION", "RESPNO", q, qo, wt, eth, etho] + ([dist] if dist else [])
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say(any(t in lab for t in LANG_LABEL), f"R{rnd} {q} is the home-language question")
        elab = str(meta.column_names_to_labels.get(up[eth.upper()], "")).casefold()
        say(any(t in elab for t in ETH_LABEL), f"R{rnd} {eth} is the ethnic group")
        df, meta = _read(p, [up[c.upper()] for c in want])
        c = {k: up[k.upper()] for k in want}
        clab = meta.variable_value_labels.get(c["COUNTRY"], {})
        sub = df[df[c["COUNTRY"]].map(clab).astype(str).str.strip().str.casefold() == "malawi"]
        say(len(sub) >= 1199, f"R{rnd}: {len(sub):,} Malawian respondents")
        vl = meta.variable_value_labels
        w = pd.to_numeric(sub[c[wt]], errors="coerce")
        say(0.98 <= w.sum() / len(sub) <= 1.02, f"R{rnd} {wt} is a within-country weight")
        dl = vl.get(c[dist], {}) if dist else {}
        o = pd.DataFrame({
            "round": rnd,
            "respno": sub[c["RESPNO"]].astype(str),
            "region": sub[c["REGION"]].map(vl.get(c["REGION"], {})),
            "district": (sub[c[dist]].map(dl) if dl else sub[c[dist]]) if dist else "",
            "lang": sub[c[q]].map(vl.get(c[q], {})),
            "verbatim": sub[c[qo]].astype(str).str.strip(),
            "eth": sub[c[eth]].map(vl.get(c[eth], {})),
            "eth_verbatim": sub[c[etho]].astype(str).str.strip(),
            "w": w,
        })
        say(o["lang"].notna().all() and o["eth"].notna().all(),
            f"R{rnd}: every language and ethnic code has a label")
        out.append(o)
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        print(__doc__)
