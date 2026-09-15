"""Sierra Leone 2015 Population and Housing Census religion (Statistics Sierra Leone, Thematic
Report on Population Structure and Population Distribution, Table 5.3) -> religiondots taxonomy.

Six cells on the 14 districts of 2015, 7,076,119 people in households, about 505,000 a district.
sources/sl.md is the write-up. The six are the form's eleven codes (P05 on the household
questionnaire, `Religion (P07)` on the code list) with the six Christian codes added together:
01 Catholic, 02 Anglican, 03 Methodist, 04 SDA, 05 Pentecostal, 06 Other Christian -> Christianity;
07 Islam; 08 Bahai; 09 Traditional; 10 Other; 11 No Religion. There is no code for no answer.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Christianity":
        "-> christianity, the bare branch. 1,547,129 people, 21.86%. The form asked six Christian "
        "codes, but Stats SL printed them only nationally (National Analytical Report Table 3.25: "
        "Catholic 7.0, Anglican 1.2, Methodist 3.0, SDA 0.7, Pentecostal 5.3, other Christian 4.7, "
        "once a stray `SDA 8.0` row with a sex ratio of 210.2 is left out). No district split "
        "exists, so nothing is spread from the national one: Pujehun is 4.8% Christian, and a "
        "uniform 7.0% Catholic share cannot even fit there. Kono (43.5%), Kailahun (34.0%) and "
        "Western Area Urban (31.3%) hold 47% of the country's Christians.",
    "Islam":
        "-> islam, no branch. 5,452,061 people, 77.05%. The 2004 census coded Ahmadi, Sunni and "
        "Shia Muslims apart (its Table 8A), and 2015 dropped them for one Islam code, so a branch "
        "would be an inference. The 2004 split survives only for the nation, Bo District, Bo Town, "
        "Western Area Urban and Bombali (sources.md §11aq), which is not a district layer.",
    "Bahai":
        "-> bahai. 2,527 people, 0.036%. Printed as 0.1 in Kailahun, Kono and both Western Area "
        "districts and 0.0 in the other ten, so at one decimal every district's figure could be "
        "anywhere from 0.05% to 0.15% or from 0 to 0.05%; drawn as printed. The national row's "
        "0.5 is a misprint (sources/sl.py::check).",
    "Traditional":
        "-> indigenous.african. 2,591 people, 0.037%, printed 0.2 in Kailahun and Kono, 0.1 in "
        "Tonkolili and 0.0 elsewhere. **Read it as a floor**, per §11b's continental rule: the "
        "form takes one code per person, so a Muslim or Christian who also takes part in "
        "traditional practice is counted once, under Islam or Christianity.",
    "Other":
        "-> other.sl. 51,963 people, 0.73%. Neither report says what it holds. The code list has "
        "no code for no answer and no table prints a religion non-response, so blank answers may "
        "be inside it; nothing here measures how many, and no dots are moved on that guess "
        "(playbooks/census_table.md, the Burkina Faso trap). It is 1.8% of Kono, 1.4% of Kailahun "
        "and Bonthe, 1.2% of Bombali and 0.1% of Pujehun.",
    "No Religion":
        "-> unaffiliated. 19,848 people, 0.28%. The form offers `09 Traditional` beside `11 No "
        "Religion`, so under the 2026-09-14 no-religion rule this box is drawn as printed "
        "(Guinea-Bissau, Chad). **87.0% of these people are under 15** (Table 3.25), against "
        "40.9% of the whole household population, so the cell is mostly children recorded "
        "without a religion rather than adults who gave none. Nothing goes to `secular`, which "
        "needs a separately counted atheist answer. Port Loko is the highest at 0.8%.",
}

MAP = {
    "Christianity": "christianity",
    "Islam": "islam",
    "Bahai": "bahai",
    "Traditional": "indigenous.african",
    "Other": "other.sl",
    "No Religion": "unaffiliated",
}

# spec §7a-i-1: every row is measured at the node it is drawn on.
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
