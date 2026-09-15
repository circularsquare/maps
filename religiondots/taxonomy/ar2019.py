"""
CEIL-CONICET, Segunda Encuesta Nacional sobre Creencias y Actitudes Religiosas (2019) -> taxonomy.

**Six answers on six survey regions, and the whole country is `modelled`** (§7b: nobody was
counted). Shares are Tabla 5 of Mallimaci, Esquivel & Giménez Béliveau (2020); sources/ar.py
has the construction and the stability test.

    62.9%  Católica                        -> christianity.catholic.latin
    18.9%  Sin filiación religiosa         -> unaffiliated
    15.3%  Evangélica                      -> christianity.protestant
     1.4%  Testigos de Jehová/Mormones     -> christianity
     1.2%  Otras                           -> other.ar
     0.3%  No sabe                         -> unknown

**Every cell sits at the depth the REGIONAL table prints it, spec §2.7.** Two of the six have
finer splits that the survey publishes for the country and for nothing smaller, and those
splits are therefore not drawn:

    Sin filiación religiosa 18.9  =  Atea 6.0 + Agnóstica 3.2 + Ninguna 9.7   (Tabla 1)
    Evangélica 15.3               =  Pentecostales 13.0 + Otros evangélicos 2.3  (report p14)

Laying either national split on every region at the same ratio was considered and refused:
it would assert that Patagonia's evangelicals are as Pentecostal as the NOA's, which nothing
measures, and §2.7 says the answer that cannot distinguish is the finished answer. The
splits belong in `note_public`, which is where they are.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Evangélica":
        "-> christianity.protestant, not christianity.evangelical and not "
        "christianity.pentecostal. In Argentine usage `evangélico` is the whole non-Catholic "
        "Protestant field, and the first survey's own report defines the box that way: "
        "*Pentecostal, Baptista, Luterana, Metodista, Adventista e Iglesia Universal del Reino "
        "de Dios*. The card offers no separate Protestant answer, so this is Brazil's "
        "`Evangélicas` exactly (br2010.py): the only non-Catholic Christian box, holding the "
        "mission churches and the Pentecostals together. `christianity.evangelical` is for a "
        "source that names Evangelical BESIDE Protestant (gt2023.py), which this one does not. "
        "85% of the box is Pentecostal nationally (13.0 of 15.3) and that split is not drawn; "
        "see the docstring. Drawn on its own regional shares under sources/ar.py's OVERRIDE.",
    "Sin filiación religiosa":
        "-> unaffiliated. The regional cell holds three national answers, Atea 6.0, "
        "Agnóstica 3.2 and Ninguna 9.7, and nothing regional separates them. The house "
        "precedent for a cell that merges no religion with a stated atheist position is "
        "`unaffiliated`: cy2021.py `Atheist/No Religion`, ca2021.py `No religion and secular "
        "perspectives`, es2026.py `Indiferente, no creyente`. Sending the whole cell to "
        "`secular` would put 9.7% who answered *ninguna* into a stated non-theistic position; "
        "splitting it at the national ratio is refused in the docstring.",
    "Testigos de Jehová/Mormones":
        "-> christianity, the bare family. The survey prints Jehovah's Witnesses and the "
        "Latter-day Saints as ONE answer in 2019 (the 2008 report still separated them, 1.2% "
        "and 0.9%), and the two nodes' nearest common ancestor in the tree is the family "
        "itself. §2.7: a cell that cannot distinguish stays at the parent. 1.4% nationally, "
        "about 640,000 people, drawn at the national rate inside each region's residual "
        "because its regional order does not survive the 2008 wave (sources/ar.py).",
    "Otras":
        "-> other.ar. 1.2% nationally. Neither report says what is inside it. It has to hold "
        "every respondent who named a religion other than the five on the card, which in "
        "Argentina means its Jewish and Muslim communities among others, and none of them can "
        "be drawn separately from this source.",
}

MAP = {
    "Católica": "christianity.catholic.latin",
    "Sin filiación religiosa": "unaffiliated",
    "Evangélica": "christianity.protestant",
    "Testigos de Jehová/Mormones": "christianity",
    "Otras": "other.ar",
    "No sabe": "unknown",
}

# No COLUMNS dict (spec §7a-i-1). Every row in this country is `modelled`, and the roll-up is
# about where a DERIVED row was counted. Nothing here was counted anywhere.


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
