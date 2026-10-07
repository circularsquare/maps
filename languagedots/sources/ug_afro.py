"""Uganda: Afrobarometer R4-R9 (2008-2022), home language and ethnic group, extract only.

    python sources/ug_afro.py      -> data/raw/ug/ab_ug_language.csv

Read-only on religiondots' merged .sav files. The model that uses the extract is
sources/ug_census.py; the record is sources/ug.md.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
AB_DIR = HERE.parent / "religiondots" / "data" / "raw" / "afrobarometer"
EXTRACT = HERE / "data" / "raw" / "ug" / "ab_ug_language.csv"

# (round, file, language, its verbatim, weight, district column, ethnic group, its verbatim)
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Q3OTHER", "Withinwt", "DISTRICT", "Q79", "Q79OTHER"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav", "Q2",
     "Q2OTHER", "withinwt", None, "Q84", "Q84OTHER"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt",
     "LOCATION.LEVEL.1", "Q87", "Q87OTHER"),
    # R7: Q2A "Respondent's mother tongue", not Q2B "Language spoken in home" (Anita's ruling
    # on ask 018, 2026-10-05: lingua francas at R7's mother-tongue question)
    (7, "r7_merged_data_34ctry.release.sav", "Q2A", "Q2AOTHER", "withinwt",
     "LOCATION.LEVEL.1", "Q84", "Q84OTHER"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav", "Q2", "Q2OTHER",
     "withinwt_hh", None, "Q81", "Q81OTHER"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav", "Q2", "Q2OTHER",
     "withinwt_hh", "LOCATION.LEVEL.1", "Q84A", "Q84AOTHER"),
]
N_RESP = {4: 2431, 5: 2400, 6: 2400, 7: 1200, 8: 1200, 9: 2400}
LANG_LABEL = ("language of respondent", "language spoken in home", "mother tongue")
ETH_LABEL = ("tribe or ethnic group", "ethnic community")


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def main():
    import pyreadstat
    out = []
    for rnd, name, q, qo, wt, loc, eth, etho in ROUNDS:
        p = AB_DIR / name
        try:
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True)
            enc = {}
        except Exception:  # noqa: BLE001  R6 is not valid UTF-8
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True, encoding="LATIN1")
            enc = {"encoding": "LATIN1"}
        up = {c.upper(): c for c in meta.column_names}
        want = [x for x in dict.fromkeys(["COUNTRY", "REGION", "RESPNO", "URBRUR", q, qo, wt,
                                           loc, eth, etho]) if x]
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say(any(t in lab for t in LANG_LABEL), f"R{rnd} {q} is the home-language question "
            f"({lab!r})")
        elab = str(meta.column_names_to_labels.get(up[eth.upper()], "")).casefold()
        say(any(t in elab for t in ETH_LABEL), f"R{rnd} {eth} is the ethnic group ({elab!r})")
        df, meta = pyreadstat.read_sav(str(p), usecols=[up[c.upper()] for c in want], **enc)
        c = {k: up[k.upper()] for k in want}
        vl = meta.variable_value_labels
        sub = df[df[c["COUNTRY"]].map(vl.get(c["COUNTRY"], {})).astype(str).str.strip()
                 .str.casefold() == "uganda"]

        def lab_of(k):
            if not k:
                return ""
            m = vl.get(c[k], {})
            return sub[c[k]].map(m) if m else sub[c[k]]
        say(len(sub) == N_RESP[rnd], f"R{rnd}: {len(sub):,} Ugandan respondents")
        w = pd.to_numeric(sub[c[wt]], errors="coerce")
        say(0.98 <= w.sum() / len(sub) <= 1.02, f"R{rnd} {wt} is a within-country weight "
            f"(mean {w.sum() / len(sub):.3f})")
        o = pd.DataFrame({
            "round": rnd, "respno": sub[c["RESPNO"]].astype(str),
            "region": lab_of("REGION"), "district": lab_of(loc), "urb": lab_of("URBRUR"),
            "lang": lab_of(q), "verbatim": sub[c[qo]].astype(str).str.strip(),
            "eth": lab_of(eth), "eth_verbatim": sub[c[etho]].astype(str).str.strip(),
            "w": w,
        })
        say(o["lang"].notna().all(), f"R{rnd}: every answer code has a label")
        out.append(o)
    a = pd.concat(out, ignore_index=True)
    EXTRACT.parent.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


if __name__ == "__main__":
    main()
