"""Senegal RGPH 1988 religion (Direction de la Prévision et de la Statistique, résultats définitifs
Tableau 1.15; the Diourbel regional report's Tableau 1.12) -> religiondots taxonomy.

Nine régions of 1988 and Diourbel's three départements, 6,896,808 residents of ordinary households.
sources/sn.md is the write-up. The household form's P11 has eight codes, and for Muslims the
answer is the brotherhood: 1 KH Khadr, 2 LA Layène, 3 MO Mouride, 4 TI Tidiane, 5 AM Muslims in
none of these; 6 CA Catholic, 7 AC other Christian; 8 AR other (the manual: *Juifs, Bouddhistes,
Animistes etc.*). No code for no religion and none for no answer.

The national table prints the five Muslim codes, Christians together and `Autres`; Diourbel's
report prints all eight in counts, in capitals. The two label sets are both mapped here, and
Diourbel's `AUTRES` (Muslims of no listed brotherhood) differs from the national `Autres` (other
religions) only in case, so resolve() matches keys exactly and the foot of this file asserts it.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Tidiane":
        "-> islam.tijaniyya, a new node. 3,260,497 people with Diourbel's `TIDIANE`, 47.28% of "
        "residents, and 80.2% of Saint-Louis région (Matam included). Code 4 `TI`. One code for "
        "every branch of the order.",
    "Mouride":
        "-> islam.mouride, a new node. 2,047,728 with Diourbel's `MOURIDE`, 29.69%; 91.55% of "
        "Mbacké département.",
    "Khadriya":
        "-> islam.qadiriyya, a new node. 806,271 with Diourbel's `KHADRIA`, 11.69%; 32.0% of "
        "Ziguinchor. The national table prints Diourbel's 3.7 under Layène; sources/sn.py uses the "
        "Diourbel report's counts, which settle it.",
    "Layène":
        "-> islam.layene, a new node. 41,681, 0.60%, 74.9% of them in Dakar région. Kaolack prints "
        "`-` and is drawn as zero (under 0.05%, fewer than about 400 people). The founder's "
        "Mahdist claim is why the four orders sit under `islam` and not `islam.sunni` (branches.py).",
    "Autres mus.":
        "-> islam, no branch. 353,325, 5.12%. Code 5 `AM`, which the manual defines as Muslims "
        "*qui n'appartiennent pas à ces confréries*: Muslims of no brotherhood and of any other, "
        "reformist movements and Shia included, which the 1988 form does not separate. 16.0% of "
        "Ziguinchor and 11.0% of Kolda.",
    "Chrétiens":
        "-> christianity, the bare branch. 307,722 with Diourbel's two rows, 4.46%. The form "
        "separates Catholics (code 6) from other Christians (code 7, *protestants, luthériens, "
        "témoins de Jéhovah*), but the national table prints one column, so nothing finer exists "
        "outside Diourbel.",
    "Autres":
        "-> other.sn. 79,584 with Diourbel's row, 1.15%, and 7.7% of Ziguinchor, which holds 38.5% "
        "of the count. Code 8 `AR`: the manual lists Jews, Buddhists and animists, and the report "
        "(printed p27) calls it *animisme principalement*. Not drawn as traditional religion "
        "because the code also holds every other religion and, with no code for no religion or "
        "no answer, may hold people with none; the report's phrase is a description, not a count.",
    "KHADRIA": "Diourbel report Tableau 1.12: -> islam.qadiriyya, as `Khadriya`.",
    "LAYENNE": "-> islam.layene, as `Layène`.",
    "MOURIDE": "-> islam.mouride, as `Mouride`.",
    "TIDIANE": "-> islam.tijaniyya, as `Tidiane`.",
    "AUTRES":
        "Diourbel's `AUTRES` is the row under TOTAL MUSULMAN, Muslims of no listed brotherhood: "
        "-> islam, as `Autres mus.`. It is NOT the national table's `Autres`.",
    "CATHOLIQUE":
        "-> christianity, not christianity.catholic. 3,775 people in the Diourbel région, about "
        "four dots; the other nine régions print Christians as one column, so drawing Diourbel's "
        "Catholics apart would add a legend row for one région.",
    "AUTRES CHRETIENS": "-> christianity, with CATHOLIQUE.",
    "AUTRE RELIGION": "-> other.sn, as `Autres`.",
}

MAP = {
    "Khadriya": "islam.qadiriyya",
    "Layène": "islam.layene",
    "Mouride": "islam.mouride",
    "Tidiane": "islam.tijaniyya",
    "Autres mus.": "islam",
    "Chrétiens": "christianity",
    "Autres": "other.sn",
    "KHADRIA": "islam.qadiriyya",
    "LAYENNE": "islam.layene",
    "MOURIDE": "islam.mouride",
    "TIDIANE": "islam.tijaniyya",
    "AUTRES": "islam",
    "CATHOLIQUE": "christianity",
    "AUTRES CHRETIENS": "christianity",
    "AUTRE RELIGION": "other.sn",
}

# spec §7a-i-1: every row is measured at the node it is drawn on.
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))


# The two `autres` differ only in case and must not meet.
assert resolve("AUTRES") == "islam" and resolve("Autres") == "other.sn"
