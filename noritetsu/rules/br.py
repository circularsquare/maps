"""Brazil's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (br_register.py) are the passenger track of the train lines; OSM route
relations are the services over them, and the metros, light rail, trams and monorails.
br_sources.md has the reasoning.
"""
import re

# Named trains, not lines (their track counts through the register lines it lies on):
#   - Vale's two long-distance trains, Vitória - Belo Horizonte (with its Itabira connection)
#     and São Luís - Parauapebas: one train each way a day / three a week, the project's
#     "single long-distance train". Their railways are register lines, which count.
#   - tourist trains that run on fewer than four days a week (Anita's "more often than about
#     once a week", read as Australia's build did: daily heritage lines count, 2-3-day ones
#     do not): Trem Republicano (Itu - Salto), CPTM's Expresso Turístico, São João del-Rei -
#     Tiradentes (Friday to Sunday), Trem das Águas, Trem da Serra da Mantiqueira, Trem de
#     Guararema, the Campinas - Jaguariúna steam train (weekends), Trem da Vale (Ouro Preto -
#     Mariana).
#   - trains of the neighbours the extract carries a few km of: the Tren Ecológico de la Selva
#     (Iguazú, Argentina), the Expreso Oriental (Bolivia).
# Counted, though tagged tourism: the Maria Fumaça of Giordani Turismo (Bento Gonçalves -
# Carlos Barbosa, four days a week, two trains a day), the Corcovado rack railway and the
# Bonde de Santa Teresa (daily).
NAMED = re.compile(
    r"Estrada de Ferro Vit[oó]ria a Minas|Estrada de Ferro Caraj[aá]s|Trem Republicano"
    r"|Expresso Tur[ií]stico|S[aã]o Jo[aã]o del.Rei|Trem das [AÁ]guas|Serra da Mantiqueira"
    r"|Trem de Guararema|Campinas.Jaguari[uú]na|Trem da Vale|Tren Ecol[oó]gico"
    r"|Expreso Oriental",
    re.IGNORECASE)
COUNTED = re.compile(r"Giordani", re.IGNORECASE)

# PROPOSED (not yet read by build_model; the diff is in br_sources.md): route relations the
# OSM half leaves out. Teresina's Linha 1 routes run on from the city over the disused
# Teresina - Parnaíba railway to Luís Correia, 359 km with 15 stops no train has called at
# since the 1980s; the city line (13.6 km) is a register line in br_register.py.
SKIP_ROUTES = {420628, 10570394}


def looks_like_service(tags, name, name_en):
    text = " ".join([name or "", name_en or "", tags.get("network") or "",
                     tags.get("operator") or ""])
    if COUNTED.search(text):
        return False
    if NAMED.search(text):
        return True
    return "tourism" in set((tags.get("service") or "").split(";")) or \
        "long_distance" in set((tags.get("service") or "").split(";"))
