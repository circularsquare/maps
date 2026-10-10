"""Poland: RINF carries PKP Polskie Linie Kolejowe alone (IM 0051), and its line id is "PL" +
the IM code 0051 + PLK's three-digit line number: PL0051001 is line 1, PL0051131 the coal
trunk line 131, PL0051573 a connecting curve. PLK's number is the line's public name
("linia kolejowa nr 1", as PLK's line list Id-12 and pl.wikipedia write it) and Wikidata's
P1671, so the id IS the number (`rule_certain`, as in Austria). OSM's route=railway relations
carry the same numbers in `ref`, and near the borders also Czech, German, Belarusian and
Ukrainian ones, which `rule_certain` keeps from renaming anything.

The name is PLK's form on the number, the English name "Line N". Wikidata's English labels are
not fetched (`langs` is Polish only): of the 74 lines that had one, several named a historic
railway the modern line only partly follows ("Prussian Eastern Railway" on 203 Tczew -
Kostrzyn, "Berlin-Wrocław railway" on 273 Wrocław - Szczecin), called a running line
"Former", or were boilerplate ("railway line nr 52 (Poland)"), and rinf.py would have used
them as the whole English name."""
import re


# Passenger stops OSM has no station node for, which rinf.py would make junctions (rinf's
# `stop_name` hook: placed at RINF's own coordinate). Olecko: PKP Intercity's TLK "Gryf"
# (18106/81107 Szczecin - Suwałki) calls there and runs on over line 39 to Suwałki (PKP
# Intercity feed, 19 days each way in 22 Sep - 31 Dec 2026), but OSM has no station at Olecko,
# so line 39's west end was a junction and prune_dead_track deleted the whole line (2026-10-07).
STOPS = {"PL01195": "Olecko"}


def pl_stop_name(p):
    return STOPS.get(p.get("uopid"))


def pl_ref(lid):
    m = re.fullmatch(r"PL0051(\d{3})", lid or "")
    return str(int(m.group(1))) if m else None


COUNTRY = {
    # join a line's pieces where RINF leaves a stretch out, over its own OSM relation
    # (rinf.fill_holes; trialled 2026-10-05)
    "fill_holes": True,
    "stop_name": pl_stop_name,
    "iso3": "POL", "wikidata": "Q36", "langs": ["pl"],
    "ref": pl_ref, "rule_certain": True,
    "name": "Linia kolejowa nr {ref}", "name_en": "Line {ref}",
    "im": {"0051_IM": "PKP Polskie Linie Kolejowe"},
}
