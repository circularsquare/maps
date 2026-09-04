"""
INS RPL 2021 religion classification -> religiondots taxonomy.

**Branch-level mapping**, like cz2021.py and pl2021.py. 23 categories, the shortest list of
any drawn country, and the reason is legal rather than statistical: **Romania's category
list is its list of state-recognised cults.** There are 18 recognised cults plus the
Metropolis of Bessarabia, and INS publishes almost exactly those, with everything else
swept into `Alta religie`. So the granularity ceiling here is a statute, not a question
design — which is a different failure mode from every other source in the project and is
worth saying out loud (spec §2.3).

What that buys, though, is unusual: the recognised list includes several bodies no other
census names at all.

  * `Crestina de Rit Vechi` — the Lipovans, Russian Old Believers who settled the Danube
    delta after the Nikonian reforms. 28,362 people, and the largest Old Believer
    population any census in the world publishes. This is the second use of
    `christianity.orthodox.oldbeliever`, which Poland introduced a day earlier.
  * `Unitariana (Biserica Unitariana Maghiara)` — the Hungarian Unitarian Church of
    Transylvania, the oldest continuously existing Unitarian body anywhere (1568, the
    Edict of Torda). 47,992 people.
  * `Ortodoxa Sarba`, `Armeana`, and BOTH Lutheran churches kept apart — the Slovak-and-
    Hungarian `Evanghelica Lutherana` and the Transylvanian Saxon `Evanghelica de
    Confesiune Augustana`, which are two churches of the same confession separated by
    ethnicity rather than doctrine. Filing them to one node loses that, and there is no
    node for "Lutheran, Saxon", so it is lost here. Recorded rather than solved.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "POPULATIA REZIDENTA TOTAL":
        "the unit's own population total, not a category.",
    "Informatie nedisponibila":
        "2,658,165 people — 13.95% of Romania — for whom religion could not be "
        "established. NOT a refusal like Poland's: INS built the 2021 census largely "
        "from administrative registers, and religion is in none of them, so this is a "
        "variable that is simply absent for one person in seven. Excluded from the dots, "
        "so the Romanian map draws 16.4M of 19.1M people. It belongs in note_public.",
}

REVIEW = {
    "Reformata":
        "-> christianity.reformed. This is the Reformed Church in Romania, the Hungarian "
        "Calvinist church of Transylvania — 495,433 people and the third largest body in "
        "the country. `christianity.reformed.continental` would be wrong: it is neither "
        "Dutch nor German but the Hungarian Reformed line, which branches.py has no node "
        "for, so it sits on the parent.",
    "Crestina dupa Evanghelie":
        "-> christianity.plymouth. The Christians According to the Gospel are the "
        "Romanian Brethren, from the Open Brethren mission of the 1890s, which is what "
        "the plymouth node holds. 36,374 people.",
    "Evanghelica (Biserica Evanghelica Romana)":
        "-> christianity.pietist. The Romanian Evangelical Church is a recognised cult of "
        "Romanian rather than immigrant origin, out of the same early-20th-century "
        "evangelical revival as the Brethren. Not `christianity.protestant`, which is for "
        "answers that name NO body, and this names one. 7,690 people; arguable.",
    "Unitariana (Biserica Unitariana Maghiara)":
        "-> unitarianuniversalist, which is a ROOT in branches.py, so the Hungarian "
        "Unitarian Church leaves the Christian family. Same call as pl2021.py makes for "
        "the Polish Brethren, and for the same reason — but it is a bigger loss here, "
        "because 47,992 Transylvanian Unitarians are a Reformation church with an "
        "unbroken line from 1568, not a modern liberal congregation.",
    "Ateu / Agnostic":
        "-> secular, both, following cz2021.py's treatment of ateismus and agnosticismus. "
        "`Fara religie` is separately `unaffiliated`: INS asks them apart and they are "
        "different answers — 71,430 no-religion against 57,229 atheist and 25,485 "
        "agnostic — so merging them would throw away a distinction the source paid for.",
    "Ortodoxa Sarba":
        "-> christianity.orthodox.canonical, with the Romanian Orthodox. The Serbian "
        "Orthodox Diocese of Timişoara is a canonical jurisdiction of the Serbian "
        "Patriarchate; it is a different church but not a different communion, and "
        "branches.py splits Orthodoxy by communion rather than by jurisdiction.",
}

MAP = {
    # ---------------------------------------------------------------- Orthodox
    "Ortodoxa (Biserica Ortodoxa Româna)": "christianity.orthodox.canonical",
    "Ortodoxa Sarba": "christianity.orthodox.canonical",
    # The Lipovans of the Danube delta — the largest Old Believer population any census
    # anywhere publishes.
    "Crestina de Rit Vechi": "christianity.orthodox.oldbeliever",
    "Armeana (Arhiepiscopia Bisericii Armene)": "christianity.oriental",

    # ---------------------------------------------------------------- Catholic
    "Romano-Catolica": "christianity.catholic.latin",
    "Greco-Catolica (Biserica Romana Unita cu Roma)": "christianity.catholic.eastern",

    # ---------------------------------------------------------------- Reformation
    "Reformata": "christianity.reformed",
    "Evanghelica Lutherana (Biserica Evanghelica Lutherana din România)":
        "christianity.lutheran",
    "Evanghelica de Confesiune Augustana": "christianity.lutheran",

    # ---------------------------------------------------------------- free churches
    "Baptista (Cultul Crestin Baptist)": "christianity.baptist",
    "Crestina dupa Evanghelie": "christianity.plymouth",
    "Evanghelica (Biserica Evanghelica Romana)": "christianity.pietist",
    "Penticostala (Cultul Crestin Penticostal - Biserica lui Dumnezeu Apostolica)":
        "christianity.pentecostal.trinitarian",
    "Adventista de Ziua a Saptea": "christianity.adventist",
    "Martorii lui Iehova": "christianity.witnesses",

    # ---------------------------------------------------------------- other traditions
    "Musulmana (Cultul Musulman)": "islam",
    "Mozaica (Federatia Comunitatilor Evreiesti din România - Cultul Mozaic)": "judaism",
    "Unitariana (Biserica Unitariana Maghiara)": "unitarianuniversalist",

    # ---------------------------------------------------------------- no religion
    "Fara religie": "unaffiliated",
    "Ateu": "secular",
    "Agnostic": "secular",

    # ---------------------------------------------------------------- residual
    "Alta religie (asociatii religioase sau grupari religioase)": "other.ro",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
