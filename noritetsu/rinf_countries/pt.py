"""Portugal: Infraestruturas de Portugal's RINF id is its line number plus one digit for the part
of the line: "081" is line 8 (Linha do Norte), "251" and "252" are the two parts of line 25
(Linha da Beira Baixa, Entroncamento side and Abrantes - Guarda), "011" and "012" line 1
(Linha do Minho, São Bento - Campanhã and on to Valença). Four-digit ids are private sidings
with three-digit numbers: "1041" is 104, Ramal da Colpor. So the number is the id less its last
digit, and it IS IP's number: OSM's route=railway relations carry the same one as `ref`
("8", "20", "104") with the line's name, and Wikidata's P1671 uses it too (33 Linha de Vendas
Novas, 68 Variante de Alcácer).

Portuguese lines are known by name, not number, so names come from those relations ("Linha do
Norte"), else Wikidata's label, else `PT_NAMES` below.

Numbers are KEYED zero-padded to two digits ("08") and shown bare ("8"): Wikidata's P1671 "1"
and "3" are the Almada light-rail lines (Linha 1, Linha 3, Metro Transportes do Sul), and rinf.py
looks Wikidata up by key before OSM's name, so an unpadded key named Linha do Minho "Linha 1".
OSM refs are padded the same way so they still confirm the rule.
"""
import re

from rinf_countries import osm_ref_default


def pt_key(num):
    """"8" -> "08", "33" -> "33", "104" -> "104"; anything else unchanged."""
    return num.zfill(2) if num and num.isdigit() else num


def pt_ref(line_id):
    m = re.fullmatch(r"(\d{2,3})\d", line_id or "")
    return pt_key(str(int(m.group(1)))) if m else None


def pt_osm_ref(ref):
    r = osm_ref_default(ref)
    return pt_key(r) if r else r


def pt_display(key):
    return str(int(key)) if key and key.isdigit() else key


# IP's names for numbers whose OSM relation carries no ref (read off the relation or the RINF
# points at the line's ends), used only when nothing above names the line.
PT_NAMES = {"21": "Ramal da Lousã", "30": "Ramal do Pego"}


def pt_id_name(lid, _uop):
    k = pt_ref(lid)
    return PT_NAMES.get(k) if k else None


# Never lines: IP numbers private sidings and freight terminals 101-183, four-digit ids (65 of
# the 66 are section nature 20, a link). A siding that leaves the main line and rejoins it
# gets the main line's ways in build_model, so it read as ridden and came out as a 0.3 km
# "line" (Terminal de Loulé, Ramal Cacia Portucel). Numbered lines with no passenger train,
# which survived because they run between two stops or beside a ridden main line: 65 Ramal
# do Barreiro-Terra, 66 Ramal Barreiro-Quimigal, 81 the Tadim terminal; 3 Concordância de São
# Gemil (0% under an OSM passenger route: the Leixões trains run Leça do Balio - Contumil, not
# to Ermesinde); 62 Ramal da Figueira da Foz, the 1.9 km stub left at Pampilhosa after the
# line closed in 2009, serving the Valouro siding; 63 Linha da Matinha, the freight line
# beside the Norte out of Santa Apolónia (41%, and only where it lies alongside).
PT_FREIGHT = {"651", "661", "811", "031", "621", "631"}


def pt_skip(lid):
    return bool(re.fullmatch(r"\d{4}", lid or "")) or lid in PT_FREIGHT


COUNTRY = {
    "iso3": "PRT", "wikidata": "Q45", "langs": ["pt", "en"],
    "ref": pt_ref, "osm_ref": pt_osm_ref, "ref_display": pt_display,
    # The id is IP's number; OSM agrees wherever it has one, and the rule must stand where a
    # line's trace also lies near another numbered relation (Norte beside Cintura in Lisbon).
    "rule_certain": True,
    "id_name": pt_id_name,
    "skip_line": pt_skip,
    # Bifurcação de Águas de Moura-Sul has two neighbours on the Linha do Sul, but two other
    # lines join there; merged through it, the Intercidades' Águas de Moura - Pinheiro and the
    # freight-only Praias do Sado - Águas de Moura were one 26 km section, 41% ridden, and the
    # Sul lost 12.4 km of the Lisbon - Faro route.
    "cut_at_junctions": True,
    "generic_label": r"^Linha \d+$",
    "im": {"0094_IM": "Infraestruturas de Portugal"},
}
