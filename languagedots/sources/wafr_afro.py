"""Afrobarometer extract for the West African ethnicity-only countries (bj, tg, lr, gm).

    python sources/wafr_afro.py <cc> [<cc> ...]   -> data/raw/<cc>/<cc>_afro.csv

Reads religiondots' merged .sav files (read-only), one row per respondent of the country, every
round it is in. Columns: round, region, district (where the round carries one), lang (the
round's FIRST-LANGUAGE question, see below), home (R7's "language spoken in home", else the
same as lang), eth (ethnic group), interview (language of interview), w (within-country
weight, rescaled to mean 1 per round).

WHICH QUESTION IS "lang". R4 Q3, R5 Q2 and R6 Q2 ask "Language of respondent"; R7 asks both
Q2A "Respondent's mother tongue" and Q2B "Language spoken in home"; R8 and R9 ask only
"Language spoken in home" (Q2). The first-language reading of ask 018 is R4-R6 plus R7's Q2A;
R8-R9 are extracted too, so the per-country scripts can show what the lingua-franca reading
would give (their FIRST_ROUNDS switch). Same files and columns as sources/ng_afro.py and
sources/ne_census.py; R9's interview language is Q102, not Q103.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
AFB = HERE.parent / "religiondots" / "data" / "raw" / "afrobarometer"

# (round, file, first-language q, home q, ethnic q, interview q, district col, weight)
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Q3", "Q79", "Q103", "DISTRICT", "Withinwt"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "Q2", "Q84", "Q103", None, "withinwt"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2", "Q87", "Q103", "LOCATION.LEVEL.1",
     "withinwt"),
    (7, "r7_merged_data_34ctry.release.sav", "Q2A", "Q2B", "Q84", "Q103", "LOCATION.LEVEL.1",
     "withinwt"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q2", "Q81", "Q103", None, "withinwt_hh"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q2", "Q84A", "Q102", "LOCATION.LEVEL.1", "withinwt_hh"),
]
COUNTRY = {"bj": {"benin"}, "tg": {"togo"}, "lr": {"liberia"}, "gm": {"gambia", "the gambia"}}


def _read(path, **kw):
    try:
        return __import__("pyreadstat").read_sav(str(path), **kw)
    except Exception:  # noqa: BLE001  some rounds are not valid UTF-8
        return __import__("pyreadstat").read_sav(str(path), encoding="LATIN1", **kw)


def extract(cc):
    out = []
    for rnd, f, q1, qh, qe, qi, qd, wt in ROUNDS:
        _, meta = _read(AFB / f, metadataonly=True)
        lab = {c: str(meta.column_names_to_labels.get(c, "")).casefold()
               for c in meta.column_names}
        assert "language" in lab[q1] or "tongue" in lab[q1], (rnd, q1, lab[q1])
        assert "language" in lab[qh], (rnd, qh, lab[qh])
        assert "ethnic" in lab[qe] or "tribe" in lab[qe], (rnd, qe, lab[qe])
        assert "interview" in lab[qi], (rnd, qi, lab[qi])
        cols = ["COUNTRY", "REGION", q1, qh, qe, qi, wt] + ([qd] if qd else [])
        df, _ = _read(AFB / f, usecols=sorted(set(cols)), apply_value_formats=True)
        cn = df["COUNTRY"].astype(str).str.strip().str.casefold()
        s = df[cn.isin(COUNTRY[cc])]
        if not len(s):
            print(f"  R{rnd}: not in this round")
            continue
        w = pd.to_numeric(s[wt], errors="coerce").fillna(0.0)
        w = w * len(s) / w.sum()
        out.append(pd.DataFrame({
            "round": rnd, "question": lab[q1], "region": s["REGION"].astype(str).str.strip(),
            "district": s[qd].astype(str).str.strip() if qd else "",
            "lang": s[q1].astype(str).str.strip(), "home": s[qh].astype(str).str.strip(),
            "eth": s[qe].astype(str).str.strip(), "interview": s[qi].astype(str).str.strip(),
            "w": w}))
        print(f"  R{rnd}: {len(s):,} respondents ({lab[q1]})")
    a = pd.concat(out, ignore_index=True)
    p = HERE / "data" / "raw" / cc / f"{cc}_afro.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    a.to_csv(p, index=False)
    print(f"wrote {p} ({len(a):,} respondents)")
    return a


# ---------------------------------------------------------------- ask 018: R7 mother tongue
#
# Anita, 2026-10-05 (ask 018): in survey-built African countries a lingua franca (English,
# French, Pidgin, Hausa outside Hausaland, Fulfulde as a second language...) is drawn at its
# share of Afrobarometer R7's Q2A "mother tongue" answers. R7 has ~1,200-1,600 respondents a
# country, so a region's own Q2A share is shrunk towards a prior by K_SHRINK respondents:
#     share_r = (Q2A answers naming it in r + K * prior_r) / (respondents in r + K)
# (weighted). A region of 400 respondents keeps 89% of its own reading; one of 30 keeps 38%.
# The prior is the national Q2A share for a language spread thinly everywhere (English,
# French); a script can pass a regional prior instead for a language with a homeland (Hausa).
K_SHRINK = 50
R7_FILE = "r7_merged_data_34ctry.release.sav"


def r7_mother(country, groups):
    """R7 Q2A (mother tongue) and Q2B (home) by REGION for one country.

    country: R7's COUNTRY label ("Nigeria"); groups: {name: [Q2A/Q2B labels]}.
    Returns a DataFrame indexed by REGION with weighted columns n, and <name>_A, <name>_B (the
    weighted count naming it as mother tongue / home language), plus a "_national" row.
    """
    _, meta = _read(AFB / R7_FILE, metadataonly=True)
    lab = {c: str(meta.column_names_to_labels.get(c, "")).casefold() for c in meta.column_names}
    assert "tongue" in lab["Q2A"] and "home" in lab["Q2B"], (lab["Q2A"], lab["Q2B"])
    df, _ = _read(AFB / R7_FILE, usecols=["COUNTRY", "REGION", "Q2A", "Q2B", "withinwt"],
                  apply_value_formats=True)
    s = df[df["COUNTRY"].astype(str).str.strip() == country].copy()
    assert len(s), f"{country} is not in R7"
    s["w"] = pd.to_numeric(s["withinwt"], errors="coerce").fillna(0.0)
    s["w"] *= len(s) / s["w"].sum()
    s["REGION"] = s["REGION"].astype(str).str.strip()
    out = pd.DataFrame({"n": s.groupby("REGION")["w"].sum()})
    for g, labels in groups.items():
        for q, suf in (("Q2A", "_A"), ("Q2B", "_B")):
            m = s[q].astype(str).str.strip().isin(labels)
            out[g + suf] = s[m].groupby("REGION")["w"].sum().reindex(out.index).fillna(0.0)
    out.loc["_national"] = out.sum()
    return out


def shrink(t, name, prior=None, k=K_SHRINK):
    """Each region's R7 mother-tongue share of `name`, shrunk to `prior` (a Series by REGION,
    or None for the national share). Returns a Series by REGION (no "_national" row)."""
    nat = t.loc["_national", name + "_A"] / t.loc["_national", "n"]
    r = t.drop(index="_national")
    p = nat if prior is None else prior.reindex(r.index)
    return (r[name + "_A"] + k * p) / (r["n"] + k)


if __name__ == "__main__":
    for cc in sys.argv[1:]:
        print(cc)
        extract(cc)
