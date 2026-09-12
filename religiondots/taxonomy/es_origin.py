"""INE's Spanish nationality names -> ISO 3166-1 alpha-2, for Spain's foreign half.

The composition tables themselves live in `taxonomy/origin_religion.py`, keyed on ISO2 and
shared with every other country that has a foreign half (Greece is the second). This file is
only the translation layer, because INE writes its nationality axis in Spanish and nothing
else does.

See `origin_religion.py` for what the model can and cannot do; the short version is that it
counts a migrant as their country of origin says, which is an upper bound on adherence, and
that it stops at the passport — anyone who naturalised is in CIS's universe instead
(taxonomy/es2026.py).

INE's own residual rows — `Resto de África`, `Resto de Asia` and the rest — have no ISO code
and no Pew row, so they carry a composition of their own below: each is the tail of a named
regional list with the big countries already removed, which is a different mixture from the
region as a whole.
"""

INE_TO_ISO = {
    # --- Europe, EU
    "Alemania": "DE", "Austria": "AT", "Bélgica": "BE", "Bulgaria": "BG", "Chipre": "CY",
    "Croacia": "HR", "Dinamarca": "DK", "Eslovenia": "SI", "Estonia": "EE",
    "Finlandia": "FI", "Francia": "FR", "Grecia": "GR", "Hungría": "HU",
    "Irlanda": "IE", "Italia": "IT", "Letonia": "LV", "Lituania": "LT",
    "Luxemburgo": "LU", "Malta": "MT", "Países Bajos": "NL", "Polonia": "PL",
    "Portugal": "PT", "República Checa": "CZ", "República Eslovaca": "SK",
    "Rumanía": "RO", "Suecia": "SE",
    # --- Europe, non-EU
    "Albania": "AL", "Andorra": "AD", "Armenia": "AM", "Belarús": "BY",
    "Bosnia y Herzegovina": "BA", "Georgia": "GE", "Islandia": "IS",
    "Liechtenstein": "LI", "Macedonia": "MK", "Macedonia del Norte": "MK",
    "Moldavia": "MD", "Noruega": "NO", "Reino Unido": "GB", "Rusia": "RU",
    "Serbia": "RS", "Serbia y Montenegro (Antigua Yugoslavia)": "RS", "Suiza": "CH",
    "Turquía": "TR", "Ucrania": "UA",
    # --- Africa
    "Angola": "AO", "Argelia": "DZ", "Benin": "BJ", "Burkina Faso": "BF",
    "Cabo Verde": "CV", "Camerún": "CM", "Congo": "CG", "Costa de Marfil": "CI",
    "Egipto": "EG", "Etiopía": "ET", "Gambia": "GM", "Ghana": "GH", "Guinea": "GN",
    "Guinea Ecuatorial": "GQ", "Guinea-Bissau": "GW", "Kenia": "KE", "Liberia": "LR",
    "Mali": "ML", "Marruecos": "MA", "Mauritania": "MR", "Nigeria": "NG",
    "República Democrática del Congo": "CD", "Senegal": "SN", "Sierra Leona": "SL",
    "Sudáfrica": "ZA", "Togo": "TG", "Túnez": "TN",
    # --- Americas
    "Argentina": "AR", "Bolivia": "BO", "Brasil": "BR", "Canadá": "CA", "Chile": "CL",
    "Colombia": "CO", "Costa Rica": "CR", "Cuba": "CU", "Dominica": "DM",
    "Ecuador": "EC", "El Salvador": "SV", "Estados Unidos de América": "US",
    "Guatemala": "GT", "Honduras": "HN", "México": "MX", "Nicaragua": "NI",
    "Panamá": "PA", "Paraguay": "PY", "Perú": "PE", "República Dominicana": "DO",
    "Uruguay": "UY", "Venezuela": "VE",
    # --- Asia
    "Arabia Saudí": "SA", "Bangladesh": "BD", "China": "CN", "Corea": "KR",
    "Filipinas": "PH", "India": "IN", "Indonesia": "ID", "Irán": "IR", "Iraq": "IQ",
    "Israel": "IL", "Japón": "JP", "Jordania": "JO", "Kazajstán": "KZ", "Líbano": "LB",
    "Nepal": "NP", "Pakistán": "PK", "Siria": "SY", "Tailandia": "TH", "Vietnam": "VN",
    # --- Oceania
    "Australia": "AU", "Nueva Zelanda": "NZ",
    # --- INE's own residuals and the stateless: no ISO code, see RESIDUAL below
    "Resto de África": None, "Resto de América Central y Caribe": None,
    "Resto de América del Sur": None, "Resto de Asia": None,
    "Resto de Nacionalidades Europeas": None, "Resto de Oceanía": None,
    "APÁTRIDAS": None,
}

# Compositions for the rows with no ISO code, in the same seven families as Pew's file.
RESIDUAL = {
    "Resto de Nacionalidades Europeas": dict(Christians=72, Muslims=12,
                                             Religiously_unaffiliated=16),
    "Resto de África": dict(Christians=48, Muslims=45, Religiously_unaffiliated=2,
                            Other_religions=5),
    "Resto de América Central y Caribe": dict(Christians=88, Religiously_unaffiliated=9,
                                              Other_religions=3),
    "Resto de América del Sur": dict(Christians=86, Religiously_unaffiliated=11,
                                     Other_religions=3),
    "Resto de Asia": dict(Muslims=55, Christians=15, Buddhists=12,
                          Religiously_unaffiliated=13, Other_religions=5),
    "Resto de Oceanía": dict(Christians=88, Religiously_unaffiliated=8,
                             Other_religions=4),
    # Spain's 3,631 apátridas are predominantly Sahrawi and Palestinian.
    "APÁTRIDAS": dict(Muslims=93, Christians=4, Religiously_unaffiliated=3),
}

# The Christian split for those residuals, since they have no ISO row in the shared table.
RESIDUAL_CHRISTIAN = {
    "Resto de Nacionalidades Europeas": {"christianity.orthodox.canonical": 0.55,
                                         "christianity.catholic.latin": 0.35,
                                         "christianity.protestant": 0.10},
    "Resto de África": {"christianity.protestant": 0.55,
                        "christianity.catholic.latin": 0.40,
                        "christianity.oriental": 0.05},
    "Resto de América Central y Caribe": {"christianity.catholic.latin": 0.62,
                                          "christianity.protestant": 0.38},
    "Resto de América del Sur": {"christianity.catholic.latin": 0.78,
                                 "christianity.protestant": 0.22},
    "Resto de Asia": {"christianity.orthodox.canonical": 0.40,
                      "christianity.catholic.latin": 0.30,
                      "christianity.protestant": 0.30},
    "Resto de Oceanía": {"christianity.protestant": 0.66,
                         "christianity.catholic.latin": 0.34},
    "APÁTRIDAS": {"christianity.catholic.latin": 0.50,
                  "christianity.orthodox.canonical": 0.30,
                  "christianity.protestant": 0.20},
}

OTHER_NODE = "other.es"


def composition(ine_name, pew_row=None):
    """{node: share} for one INE nationality row.

    `pew_row` is the seven Pew percentages, or None/empty when Pew has no row — which
    happens two ways, and both are handled here rather than by the caller: INE's own
    `Resto de …` residuals have no ISO code at all, and a few microstates have one but sit
    below Pew's 100,000 threshold.
    """
    import origin_religion as origin

    iso = INE_TO_ISO.get(ine_name, "MISSING")
    if iso == "MISSING":
        raise KeyError(f"no ISO code for INE nationality {ine_name!r}")
    if iso is None:
        # A residual row. Its Christian split is INE-specific, so it is registered in the
        # shared table under the row's own name — which no ISO code can collide with.
        origin.CHRISTIAN.setdefault(ine_name, RESIDUAL_CHRISTIAN[ine_name])
        return origin.composition(ine_name, RESIDUAL[ine_name], OTHER_NODE)
    if not pew_row:
        pew_row = origin.REGIONAL[iso]
    return origin.composition(iso, pew_row, OTHER_NODE)


def pew_name(ine_name):
    """Pew's country name for an INE nationality, or None if it has no Pew row."""
    import origin_religion as origin

    iso = INE_TO_ISO.get(ine_name, "MISSING")
    if iso == "MISSING":
        raise KeyError(f"no ISO code for INE nationality {ine_name!r}")
    if iso is None:
        return None
    return origin.PEW_BY_ISO.get(iso)
