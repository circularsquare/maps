"""Spain, INE's Encuesta de Caracteristicas Esenciales de la Poblacion y las Viviendas (ECEPOV)
2021, "lengua inicial" (the first language a person spoke), by province -> node.

Keyed by the labels INE prints in its 52 provincial tables "Personas segun la lengua inicial mas
frecuente" (sources/es_ecepov.py writes them into es.csv's `province` rows). Each province's
table names only the languages frequent there, plus "Otra"; the full set is below.

Calls (sources/es.md says more):
  * "Valenciano" -> its own node, `valencian` (bo.txt made it beside Catalan), because INE prints
    it as its own label in the three Valencian provinces and "Catalan" everywhere else. In the
    Balearics INE files Mallorquin, Menorquin and Eivissenc under "Catalan" (the release's own
    footnote), so they are on `catalan`.
  * "Arabe" stays on `arabic` everywhere, including Ceuta and Melilla where the speech is Moroccan
    Arabic: the table says Arabic, and spec 3.1 maps the label printed.
  * Combinations ("Castellano y catalan", "Castellano, ingles y frances") name two or three first
    languages for one person. Each person is shared equally across the languages named (spec
    3.6); es.csv carries the shares as "<combination> | <language>" rows, tier derived.
  * "Otra" is the unnamed remainder and sits on `other`, except for two derived splits of it that
    sources/es_ecepov.py makes and es.md argues for:
      - "Otra | nationality: <node>": foreign nationals' "Otra", shared out by the province's
        foreign residents by nationality (Padron 2022, INE table 03005), each nationality on the
        first language most of its people speak (ORIGIN below), counting only languages the
        province's table does not already name. Capped at the cell; any rest stays on `other`.
      - "Otra | regional: <node>": Spanish nationals' "Otra" above the level the rest of Spain
        shows, in the three provinces where a local language INE did not list explains it:
        Illes Balears (Catalan written in as Mallorquin etc. but not recoded in this table),
        Asturias (Asturian), Melilla (Tarifit, the Riffian Berber of the city's Muslim half).
        And in Lleida, Aranese (Occitan, fr.txt's node): Idescat's EULP 2023 share for the Val
        d'Aran times Aran's population, taken out of Lleida's Spanish nationals' "Otra".
Nobody is "not stated": ECEPOV imputes, and the tables have no such row.
"""
IE = "indoeuropean"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"
SL = f"{IE}.slavic"
IA = f"{IE}.indoaryan"
NC = "nigercongo"

SPANISH = f"{RO}.spanish"
CATALAN = f"{RO}.catalan"
VALENCIAN = f"{RO}.valencian"
GALICIAN = f"{RO}.galician"
BASQUE = "isolate.basque"
ASTURIAN = f"{RO}.asturian"
OCCITAN = f"{RO}.occitan"
TARIFIT = "afroasiatic.berber.tarifit"
BERBER = "afroasiatic.berber"
ARABIC = "afroasiatic.arabic"
GW_KRIOL = "creole.portuguese_based.guinea_bissau_kriol"

NAMES = {
    "Castellano": SPANISH,
    "Catalán": CATALAN,
    "Valenciano": VALENCIAN,
    "Gallego": GALICIAN,
    "Euskera": BASQUE,
    "Rumano": f"{RO}.romanian",
    "Árabe": ARABIC,
    "Inglés": f"{GE}.english",
    "Francés": f"{RO}.french",
    "Italiano": f"{RO}.italian",
    "Alemán": f"{GE}.continental.german",
    "Otra": "other",
}

# Combinations INE prints: each splits equally across its members (spec 3.6).
COMBOS = {
    "Castellano y catalán": ["Castellano", "Catalán"],
    "Castellano y valenciano": ["Castellano", "Valenciano"],
    "Castellano y gallego": ["Castellano", "Gallego"],
    "Castellano y euskera": ["Castellano", "Euskera"],
    "Castellano y rumano": ["Castellano", "Rumano"],
    "Castellano y árabe": ["Castellano", "Árabe"],
    "Castellano e inglés": ["Castellano", "Inglés"],
    "Castellano y francés": ["Castellano", "Francés"],
    "Castellano e italiano": ["Castellano", "Italiano"],
    "Castellano y alemán": ["Castellano", "Alemán"],
    "Francés y árabe": ["Francés", "Árabe"],
    "Castellano, inglés y francés": ["Castellano", "Inglés", "Francés"],
    "Castellano, inglés e italiano": ["Castellano", "Inglés", "Italiano"],
}

# Where Spanish nationals' "Otra" stands far above the rest of Spain, the local language INE's
# table for that province does not list (sources/es.md, "Otra by province").
REGIONAL_OTRA = {"07": CATALAN, "33": ASTURIAN, "52": TARIFIT}

# INE table 03005's nationalities -> ISO 3166 alpha-2. Each nationality's languages come from
# the shared origin table, sources/origin_mix.py (Morocco's 20.2% Berber from Idescat's EULP
# 2023 is an override there); sources/es_ecepov.py drops the languages a province's table
# already names. Remainder rows ("Resto de ...", "APATRIDAS") name no country and are left
# out, so their share of "Otra" stays on `other`. "Serbia y Montenegro (Antigua Yugoslavia)"
# is origin_mix's YU (Serbo-Croatian).
ORIGIN = {
    'Alemania': 'DE', 'Austria': 'AT', 'Bélgica': 'BE', 'Bulgaria': 'BG',
    'Chipre': 'CY', 'Croacia': 'HR', 'Dinamarca': 'DK', 'Eslovenia': 'SI',
    'Estonia': 'EE', 'Finlandia': 'FI', 'Francia': 'FR', 'Grecia': 'GR',
    'Hungría': 'HU', 'Irlanda': 'IE', 'Italia': 'IT', 'Letonia': 'LV',
    'Lituania': 'LT', 'Luxemburgo': 'LU', 'Malta': 'MT', 'Países Bajos': 'NL',
    'Polonia': 'PL', 'Portugal': 'PT', 'República Checa': 'CZ', 'República Eslovaca': 'SK',
    'Rumanía': 'RO', 'Suecia': 'SE', 'Albania': 'AL', 'Andorra': 'AD',
    'Armenia': 'AM', 'Belarús': 'BY', 'Bosnia y Herzegovina': 'BA', 'Georgia': 'GE',
    'Islandia': 'IS', 'Liechtenstein': 'LI', 'Macedonia del Norte': 'MK', 'Moldavia': 'MD',
    'Noruega': 'NO', 'Reino Unido': 'GB', 'Rusia': 'RU', 'Serbia y Montenegro (Antigua Yugoslavia)': 'YU',
    'Serbia': 'RS', 'Suiza': 'CH', 'Turquía': 'TR', 'Ucrania': 'UA',
    'Angola': 'AO', 'Argelia': 'DZ', 'Benin': 'BJ', 'Burkina Faso': 'BF',
    'Cabo Verde': 'CV', 'Camerún': 'CM', 'Congo': 'CG', 'Costa de Marfil': 'CI',
    'Egipto': 'EG', 'Etiopía': 'ET', 'Gambia': 'GM', 'Ghana': 'GH',
    'Guinea': 'GN', 'Guinea Ecuatorial': 'GQ', 'Guinea-Bissau': 'GW', 'Kenia': 'KE',
    'Liberia': 'LR', 'Mali': 'ML', 'Marruecos': 'MA', 'Mauritania': 'MR',
    'Nigeria': 'NG', 'República Democrática del Congo': 'CD', 'Senegal': 'SN', 'Sierra Leona': 'SL',
    'Sudáfrica': 'ZA', 'Togo': 'TG', 'Túnez': 'TN', 'Costa Rica': 'CR',
    'Cuba': 'CU', 'Dominica': 'DM', 'El Salvador': 'SV', 'Guatemala': 'GT',
    'Honduras': 'HN', 'Nicaragua': 'NI', 'Panamá': 'PA', 'República Dominicana': 'DO',
    'Canadá': 'CA', 'Estados Unidos de América': 'US', 'México': 'MX', 'Argentina': 'AR',
    'Bolivia': 'BO', 'Brasil': 'BR', 'Chile': 'CL', 'Colombia': 'CO',
    'Ecuador': 'EC', 'Paraguay': 'PY', 'Perú': 'PE', 'Uruguay': 'UY',
    'Venezuela': 'VE', 'Arabia Saudí': 'SA', 'Bangladesh': 'BD', 'China': 'CN',
    'Corea': 'KR', 'Filipinas': 'PH', 'India': 'IN', 'Indonesia': 'ID',
    'Irán': 'IR', 'Iraq': 'IQ', 'Israel': 'IL', 'Japón': 'JP',
    'Jordania': 'JO', 'Kazajstán': 'KZ', 'Líbano': 'LB', 'Nepal': 'NP',
    'Pakistán': 'PK', 'Siria': 'SY', 'Tailandia': 'TH', 'Vietnam': 'VN',
    'Australia': 'AU', 'Nueva Zelanda': 'NZ',
}

EXTRA_NODES = sorted({TARIFIT, BERBER, ASTURIAN, OCCITAN})


def resolve(label):
    """A table label, a "<combination> | <member>" share, or an "Otra | ...: <node>" split."""
    if label in NAMES:
        return NAMES[label]
    if " | " in label:
        head, tail = label.split(" | ", 1)
        if head in COMBOS:
            if tail not in COMBOS[head]:
                raise KeyError(f"es2021: {tail!r} is not a member of {head!r}")
            return NAMES[tail]
        if head == "Otra" and (tail.startswith("nationality: ") or tail.startswith("regional: ")):
            node = tail.split(": ", 1)[1]
            if not node[:1].islower():   # a node id (origin_mix's, or regional)
                raise KeyError(f"es2021: Otra split onto unknown node {node!r}")
            return node
    raise KeyError(f"es2021: unmapped label {label!r}")
