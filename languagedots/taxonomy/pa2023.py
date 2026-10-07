"""Panama, Censo 2023: indigenous people (P08) and Afro-descendant group (P09) -> shares of
language nodes, from MICS 2013's mother tongue by group (sources/pa_mics.py,
data/normalized/pa_mics_shares.csv). The census asks no language question; this is AGENT_BRIEF
section 2's ethnicity rule with a measured retention source. Every row `modelled`.

WHICH MICS VECTOR EACH CENSUS GROUP TAKES (sources/pa.md section 2):
  Kuna, Ngabe, Embera, Wounaan, Bugle   their own, inside or outside a comarca (the census
                                        corregimiento's province 10, 11 or 12 is inside)
  Naso, Teribe                          MICS "Naso/Teribe", all (n=41; too few to split)
  Bokota                                MICS Bugle, all: MICS's 23 "Bokota" sit mostly in the
                                        Embera comarca and speak Embera, a miscode
  Bri Bri                               Bribri 100% (MICS has one person)
  Otro grupo indigena                   MICS "Neither" vector: MICS's 42 "Otro" give no
                                        language of their own (65% Spanish, 31% "other")
  not indigenous, Afroantillano         MICS "Negro antillano", all (8.0% English)
  not indigenous, Afrocolonial          MICS "Negro colonial", all
  not indigenous, Negro, Afrodescendiente, Afropanameno
                                        MICS "Negro", all (1.9% English)
  not indigenous, anything else         MICS "Neither", by inside / outside

WHICH LANGUAGES ARE KEPT. For an indigenous group, only its own language, Spanish, English and
the neighbour languages it plausibly speaks (Ngabere for Bugle, Bokota and Naso; Embera and
Wounaan for each other) are kept, then renormalised; the scattered other codes (6% "Kuna" among
Bugle, 5% among Naso, no Kuna anywhere near) look like keying slips. For the non-indigenous
vectors every answer is kept.

Nodes: Kuna -> co.txt's chibchan.tule (Tule (Kuna), the same language across the border);
Ngabere cr.txt; Buglere bugl1243 and Naso (Teribe, teri1250) new under Chibchan; Embera co.txt's
generic chocoan.embera.embera (Panama's is Northern Embera); Wounaan co.txt; English, `other`
and `americas_other` (MICS "otra lengua indigena").
"""
import pandas as pd
from pathlib import Path

SHARES = Path(__file__).resolve().parent.parent / "data" / "normalized" / "pa_mics_shares.csv"

NODE = {"es": "indoeuropean.romance.spanish", "kuna": "chibchan.tule",
        "ngab": "chibchan.ngabere", "bugl": "chibchan.buglere", "emb": "chocoan.embera.embera",
        "woun": "chocoan.wounaan", "naso": "chibchan.naso",
        "en": "indoeuropean.germanic.english", "oth": "other", "oind": "americas_other"}
CODES = NODE                     # build.py reads the node values
EXTRA_NODES = ["chibchan.bribri"]

# census P08 label -> (MICS group, split by comarca?, languages kept or None for all)
IND = {
    "Kuna": ("ind:Kuna", True, {"kuna"}),
    "Ngäbe": ("ind:Ngäbe", True, {"ngab"}),
    "Emberá": ("ind:Emberá", True, {"emb", "woun"}),
    "Wounaan": ("ind:Wounaan", True, {"woun", "emb"}),
    "Buglé": ("ind:Buglé", True, {"bugl", "ngab"}),
    "Bokota": ("ind:Buglé", False, {"bugl", "ngab"}),
    "Naso": ("ind:Naso/Teribe", False, {"naso", "ngab"}),
    "Teribe": ("ind:Naso/Teribe", False, {"naso", "ngab"}),
}
AFRO = {"Afroantillano(a)": "afro:Negro(a) antillano(a)",
        "Afrocolonial": "afro:Negro(a) colonial",
        "Negro(a)": "afro:Negro(a)", "Afrodescendiente": "afro:Negro(a)",
        "Afropanameño(a)": "afro:Negro(a)"}
COMARCAS = {"10", "11", "12"}

_tab = None


def _vec(group, scope, keep=None):
    global _tab
    if _tab is None:
        _tab = pd.read_csv(SHARES)
    s = _tab[(_tab["group"] == group) & (_tab["scope"] == scope)]
    if s.empty:
        raise SystemExit(f"pa2023: no MICS vector for {group} / {scope}")
    v = dict(zip(s["lang"], s["share"]))
    if keep is not None:
        v = {k: x for k, x in v.items() if k in keep | {"es", "en"}}
    t = sum(v.values())
    return {NODE[k]: x / t for k, x in v.items() if x > 0}


def shares(indigenous, afro, province):
    """-> {node: share} for one census (P08, P09) cell in a corregimiento of `province`."""
    scope = "inside" if province in COMARCAS else "outside"
    if indigenous == "Bri Bri":
        return {"chibchan.bribri": 1.0}
    if indigenous in IND:
        grp, split, keep = IND[indigenous]
        return _vec(grp, scope if split else "all", keep)
    if indigenous in ("Otro grupo indígena", "Ninguno", "No declarado"):
        if indigenous == "Otro grupo indígena" or afro not in AFRO:
            return _vec("neither", scope)
        return _vec(AFRO[afro], "all")
    raise SystemExit(f"pa2023: P08 label {indigenous!r} not planned")


def resolve(code):
    return NODE.get(code)
