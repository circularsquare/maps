"""
INEGI Censo 2020 religion classification -> religiondots taxonomy.

**Branch-level mapping, like ca2021.py / cz2021.py / br2010.py.** No leaves are created;
the category's own name travels with the row in `source_category` (spec §2.4).

The source is 4 categories at municipio and 23 below them at entidad, so everything except
`Católica` arrives here already allocated (§3.9) and carries `tier=derived`. countries.py
turns that into `may_ring=False`: allocation spreads a total and cannot establish that
anyone is present (§3.10).

INEGI's own four groups, and what this file does with each:

    Población con religión católica            97,864,218   one category, measured
    ...protestante/cristiano evangélico         14,095,307   10 categories
    ...sin religión o sin adscripción           13,314,516   3 categories
    ...otras religiones                            248,169   9 categories

The third group is the interesting one and Mexico is the reason `unchurched` exists on the
tree at all after Czechia: INEGI separates `Ninguna religión` (9.5M) from `Sin adscripción
religiosa (creyente)` (3.1M) — believers with no affiliation — and they are not the same
people. Folding the second into `unaffiliated` would overstate Mexican irreligion by a third.

Mexico is also spec §3.11's worked example: `Población con otras religiones` hides Orthodox
Christians, Buddhists and Hindus inside a 248,169-person bucket that INEGI will not split.
Nothing here invents that split; `Origen oriental` stays a lump and says so.

`No especificado` (491,814, 0.39%) is not in this file at all — the ITER extract does not
publish it, which sources/mx.md §5 records. So the Mexican total is 125,522,210 against a
census population of 126,014,024, and the missing people are missing upstream of here.
"""

EXCLUDED = {}   # the four parent groups are not in the allocated file; nothing to exclude

REVIEW = {
    "Iglesia del Dios Vivo, Columna y Apoyo de la Verdad, la Luz del Mundo":
        "La Luz del Mundo -> pentecostal.oneness. Founded 1926 in Guadalajara, "
        "nontrinitarian and baptising in Jesus' name, which is the Oneness position; INEGI "
        "files it under Protestant/evangelical. 190,005 people and the largest "
        "Mexican-founded church, so it is worth being right about.",
    "Cristiana":
        "-> christianity.protestant. 6.8M people who answered 'cristiana' inside INEGI's "
        "Protestant/evangelical group — so it means non-Catholic Christian, not Christian "
        "in general. Largest single Protestant answer in Mexico and it names no body.",
    "Evangélica":
        "-> christianity.protestant. Same shape as 'Cristiana'; INEGI lists the two "
        "separately and neither names a body, so both land on the same node.",
    "Sin adscripción religiosa (creyente)":
        "-> unchurched. INEGI's own gloss is a believer with no affiliation, which is "
        "exactly the node Czechia's 'věřící - nehlásící se k žádné církvi' created. 3.1M "
        "people, and putting them in `unaffiliated` would overstate irreligion badly.",
    "Origen oriental":
        "-> other.mx, NOT buddhism. 29,985 people covering Buddhism, Hinduism, Sikhism and "
        "the rest in one INEGI bucket. Splitting it needs an external estimate (§3.11) and "
        "this file will not guess.",
    "Cultos populares":
        "-> other.mx. Folk devotion, largely Santa Muerte. §3.3 would give a syncretism its "
        "own node, but INEGI's bucket is not one tradition, so it stays residual.",
    "Raíces afro":
        "-> afrodiasporic. 40,799. The node was added for Brazil's Umbanda and Candomblé; "
        "INEGI gives no named tradition, so this sits on the family rather than a child.",
    "Raíces étnicas":
        "-> indigenous. INEGI names no people, as with Brazil's 'Tradições indígenas'.",
}

MAP = {
    # ---------------------------------------------------------------- Catholic
    "Católica": "christianity.catholic.latin",

    # ---------------------------------------------------------------- Protestant / evangelical
    "Cristiana": "christianity.protestant",
    "Evangélica": "christianity.protestant",
    "Otro Protestante/cristiano evangélico": "christianity.protestant",
    "Testigo de Jehová": "christianity.witnesses",
    "Pentecostal": "christianity.pentecostal",
    "Adventista del Séptimo Día": "christianity.adventist",
    "Presbiteriana": "christianity.reformed.presbyterian",
    "Bautista": "christianity.baptist",
    "Iglesia de Jesucristo de los Santos de los Últimos Días (Mormón)":
        "christianity.latterday",
    "Iglesia del Dios Vivo, Columna y Apoyo de la Verdad, la Luz del Mundo":
        "christianity.pentecostal.oneness",

    # ---------------------------------------------------------------- no religion
    "Ninguna religión": "unaffiliated",
    "Sin adscripción religiosa (creyente)": "unchurched",
    "Ateos/Agnósticos": "secular",

    # ---------------------------------------------------------------- other religions
    "Judía": "judaism",
    "Islámica": "islam",
    "Espiritualista": "spiritualism",
    "Raíces afro": "afrodiasporic",
    "Raíces étnicas": "indigenous",
    "New Age y Escuelas esotéricas": "esoteric",
    "Origen oriental": "other.mx",
    "Cultos populares": "other.mx",
    "Otras religiones o movimientos religiosos": "other.mx",
}


def resolve(category):
    """religiondots branch for an INEGI category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
