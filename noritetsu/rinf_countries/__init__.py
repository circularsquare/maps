"""rinf.py's per-country settings, one file per country: rinf_countries/<cc>.py defines COUNTRY.

One file each so that several agents can add RINF countries at once without editing the same
file: a new country is a new file here, and rinf.py itself is shared code, changed only
through whoever holds the shared files (HANDOFF.md, "How the work is run").

What a COUNTRY dict may hold is in rinf.py's docstring ("A NEW COUNTRY"); be.py, at.py and
nl.py are worked examples of the three ways a RINF id becomes a public line number.
"""
import importlib
import re


def osm_ref_default(ref):
    """An OSM relation's ref as a bare line number: "L50A", "L 36", "Lijn 36" -> "50A", "36"."""
    r = (ref or "").strip()
    r = re.sub(r"^(?:[A-Za-z]{1,6}\.?\s*)(?=\d)", "", r)
    return r.replace(" ", "").upper() or None


def country(cc):
    try:
        mod = importlib.import_module(f"rinf_countries.{cc}")
    except ModuleNotFoundError:
        raise SystemExit(f"no RINF settings for {cc!r}: add rinf_countries/{cc}.py")
    return mod.COUNTRY
