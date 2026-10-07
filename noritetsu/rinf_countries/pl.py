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


def pl_ref(lid):
    m = re.fullmatch(r"PL0051(\d{3})", lid or "")
    return str(int(m.group(1))) if m else None


COUNTRY = {
    # join a line's pieces where RINF leaves a stretch out, over its own OSM relation
    # (rinf.fill_holes; trialled 2026-10-05)
    "fill_holes": True,
    "iso3": "POL", "wikidata": "Q36", "langs": ["pl"],
    "ref": pl_ref, "rule_certain": True,
    "name": "Linia kolejowa nr {ref}", "name_en": "Line {ref}",
    "im": {"0051_IM": "PKP Polskie Linie Kolejowe"},
}
