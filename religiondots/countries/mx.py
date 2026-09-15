# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mx_counts():
    """INEGI 2020 at municipio, allocated to 23 categories."""
    import mx2020
    return _allocated_counts("mx", "municipio", mx2020)


ENTRY = {
    "mx": dict(
        name="Mexico",
        source="Censo de Poblacion y Vivienda 2020 (INEGI)",
        basis="self-identification",
        note_public=(
            "INEGI separates people with no religion from believers with no affiliation, "
            "which most censuses do not: 9.5 million against 3.1 million, and folding the "
            "second into the first would overstate Mexican irreligion by a third. The "
            "denominations are thin by comparison — 23 categories, and everything except "
            "Catholic is derived from state-level shares. 'Other religions' is a single "
            "248,000-person bucket holding Buddhists, Hindus and Orthodox Christians "
            "together."),
        how="census, 2020",
        fill="from the same census at state level",
        grain="municipios, 51,000 people on average",
        counts=_mx_counts,
        # Counts are on municipio; AGEBs carry their municipio's code in the first five
        # characters of CVEGEO, so no spatial join. 81,451 AGEBs against 2,469 municipios,
        # and INEGI builds them to a population target — urban ones to about 2,500 people
        # — which is what §8.2 asks for. Both urban and rural AGEBs are present and every
        # municipio has at least one, so nothing falls through.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mx" / "mg2020" / "conjunto_de_datos" / "00a.shp",
        place_unit=lambda g: g["CVE_ENT"].astype(str) + g["CVE_MUN"].astype(str),
        note="INEGI is self_id; every category except Catolica is allocated from entidad "
             "level (spec §3.9).",
    ),
}
