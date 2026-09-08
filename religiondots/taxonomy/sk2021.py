"""SODB 2021 (Slovakia) religion classification -> religiondots taxonomy.

Eleven categories at obec level, all eleven drawn. **One new node, `other.sk`**; the other
ten needed none, which is unusual here and has a specific cause: Slovakia's registered
churches are Czechoslovakia's, and Czechia was drawn first (§9b). `cz2021.py` had already had
to decide every one of these bodies, so this file is mostly a translation of its calls into
Slovak.

    55.76%  Rímskokatolícka cirkev            -> christianity.catholic.latin
    23.79%  bez náboženského vyznania         -> unaffiliated
     7.83%  ostatné                           -> other.sk  (Anita's call — see below)
     5.27%  Evanjelická cirkev a. v.          -> christianity.lutheran
     4.00%  Gréckokatolícka cirkev            -> christianity.catholic.eastern
     1.56%  Reformovaná kresťanská cirkev     -> christianity.reformed
     0.93%  Pravoslávna cirkev                -> christianity.orthodox.canonical
     0.34%  Kresťanské zbory                  -> christianity.plymouth
     0.30%  Jehovovi svedkovia                -> christianity.witnesses
     0.17%  Apoštolská cirkev                 -> christianity.pentecostal.trinitarian
     0.06%  Evanjelická cirkev metodistická   -> christianity.methodist

**`ostatné` IS THE ONLY REAL DECISION IN THIS FILE, AND IT WAS ANITA'S — 2026-09-08.** The
GIS service publishes nine named churches, a no-religion cell and one residual. UNSD
Demographic Yearbook table 28 carries the same census with **21** categories, agrees with the
service to the person on the total and on all nine named churches, and breaks the residual
out: `Not Stated` **353,797** and `Other Religions` **64,990**, plus ten small named bodies.
`Kresťanské zbory` + `ostatné` equals the sum of those twelve rows exactly, which is what
establishes that the residual really does contain the not-stated. So **83% of `ostatné` is
`nezistené`.**

It was first built EXCLUDED on spec §3.5, which does not draw a derived non-response
residual. Anita reversed that: *"we should definitely draw these points as an
other(slovakia), tons of countries have an other."* **The argument for drawing is §6.12** —
excluding it left 7.83% of Slovakia as a hole, and a hole on a dot map reads as an absence of
*people*, not as an absence of work. 426,496 people who exist and were counted were being
shown as nobody.

**But this node is not comparable with its siblings and the label says so.** It is
`Other or not stated (Slovakia)`, not `Other religion (Slovakia)`, because unlike `other.ro`,
`other.me` or `other.mw` it is not a residual of religions somebody named — it is that plus
the non-response, merged by the publisher before release and unseparable at obec level.

**AND THE MIX IS NOT CONSTANT ACROSS UNITS, WHICH IS THE THING THAT WOULD MISLEAD.** Measured
on the drawn data: the cell runs **0.0% to 58.8%** between municipalities (median 3.8%, p90
8.9%, p99 15.1%), and the largest values are Slovakia's Roma settlements and its two city
centres — **Košice-Luník IX 58.8%**, Pavlovce nad Uhom 28.7%, Jasov 26.9%, Bratislava-Staré
Mesto 16.4% — against 0.7-2.7% in the Orava and Kysuce villages. Those are non-response
geographies, not unusual-religion geographies. So this node is mostly *did not answer* where
it is dense and mostly *some other faith* where it is sparse, and **its density must not be
read as a religious fact.** `note_public` says that on the map.

**This is still not Kazakhstan's case (§9aq), and the difference is the questionnaire.**
There an offered `Отказываюсь указать` box that people actively ticked was drawn *as a
refusal*, because an offered answer somebody picked is an answer. Slovakia's form has no
refusal box — `nezistené` is what the office computes for a form left blank. Drawing it here
is a §6.12 call about holes, not a §3.5 reclassification of non-response into an answer.

**And note the neighbour does the opposite.** `cz2021.py` excludes Czechia's `Neuvedeno` —
3,162,540 people, **30.05%** — so the Czech map draws 7.36M of 10.52M. The two countries
therefore handle non-response differently, which matters because the Czech/Slovak border is
the comparison this country exists for. Czechia's is excludable because it is a clean
non-response cell; Slovakia's is not, because it has real religions merged into it. If
Czechia is ever revisited this asymmetry is the thing to settle.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "spolu":
        "the unit's own population total, not a category.",
}

REVIEW = {
    "Rímskokatolícka cirkev v Slovenskej republike (rímskokatolícke)":
        "-> christianity.catholic.latin, matching cz2021.py's `Církev římskokatolická`. "
        "Slovakia is the most Catholic country drawn in this part of Europe at 55.8%, "
        "against Czechia's 7.0% — the sharpest religious contrast across any land border on "
        "this map, and between two halves of a state that existed until 1993.",
    "Gréckokatolícka cirkev na Slovensku (gréckokatolícke)":
        "-> christianity.catholic.eastern, with cz2021.py's `Církev řeckokatolická` and "
        "ro2021.py's `Greco-Catolica`. Byzantine-rite Catholics in communion with Rome, "
        "concentrated in Prešov and the Rusyn east. At 4.00% and 218,235 people this is the "
        "**largest Greek Catholic population on this map after Romania's**, and unlike "
        "Romania's it was never dissolved and re-founded — it was suppressed in 1950 and "
        "restored in 1968.",
    "Evanjelická cirkev augsburského vyznania na Slovensku (evanjelické)":
        "-> christianity.lutheran, with cz2021.py's three Augsburg-confession churches. "
        "286,907 people, and the historic church of the Slovak national revival; its "
        "geography is the central-northern uplands rather than the Catholic west and east.",
    "Reformovaná kresťanská cirkev na Slovensku (kalvínske)":
        "-> christianity.reformed, with ro2021.py's `Reformata`. **This is an ethnic "
        "geography as much as a religious one**: the Reformed church in Slovakia is "
        "overwhelmingly Hungarian, and its 85,271 people sit along the southern border in "
        "the same districts that return a Hungarian mother tongue. The same church, the "
        "same minority and the same border as Romania's Reformed — drawn on both sides now.",
    "Pravoslávna cirkev na Slovensku (pravoslávne)":
        "-> christianity.orthodox.canonical, with cz2021.py's `Pravoslavná církev v českých "
        "zemích`. The two are literally one body — the autocephalous Orthodox Church of the "
        "Czech Lands and Slovakia — so the node must match, and it does. Canonical is not "
        "arguable here the way mk2021.py's and me2023.py's are: no rival jurisdiction "
        "claims these 50,677 people.",
    "Kresťanské zbory na Slovensku":
        "-> christianity.plymouth. **cz2021.py settles this and it is the reason to check a "
        "neighbour's file before inventing a node**: `Křesťanské sbory` is the Czech half of "
        "the same Plymouth Brethren body and is already mapped there. The name translates as "
        "'Christian Congregations', which reads like a generic residual and is not one — "
        "taking it as `christianity` would have been the natural wrong call. 18,553 people.",
    "Apoštolská cirkev na Slovensku":
        "-> christianity.pentecostal.trinitarian, with cz2021.py's `Apoštolská církev`. "
        "Again the same body across the former federation. Oneness Pentecostals are not "
        "separately counted in Slovakia and this church is trinitarian, so the child is "
        "right rather than the `christianity.pentecostal` parent.",
    "Evanjelická cirkev metodistická, Slovenská oblasť":
        "-> christianity.methodist. 3,018 people, the smallest named cell here, and its own "
        "name says it is the Slovak district of the church cz2021.py maps as `Evangelická "
        "církev metodistická`.",
    "Náboženská spoločnosť Jehovovi svedkovia v Slovenskej republike":
        "-> christianity.witnesses, with cz2021.py and ro2021.py.",
    "ostatné":
        "-> other.sk, a per-source residual (§3.11), and the arguable call in this file. "
        "See the module docstring for the reversal and its reason. The short form: 83% of "
        "this cell is `nezistené` rather than a religion, so the node is labelled `Other or "
        "not stated (Slovakia)` and must never be pooled with another country's `other`; "
        "and because the cell varies 0.0-58.8% between municipalities with its peaks in "
        "Roma settlements and city centres, its DENSITY is a map of the census's reach "
        "rather than of anybody's religion.",
    "bez náboženského vyznania":
        "-> unaffiliated, with cz2021.py's `Bez náboženské víry`. **Slovakia offers no "
        "atheist or agnostic box and no believing-without-belonging box**, where the Czech "
        "form offers all three, so `secular` and `unchurched` are both unlit for this "
        "country (coverage.py, §6.12). That is a property of the form and it matters for "
        "reading the two side by side: Czechia's 47.8% no-religion and Slovakia's 23.8% are "
        "answers to differently-shaped questions, which is spec §3.1a's rule — compare the "
        "answer LISTS before comparing across a border.",
}

MAP = {
    "Rímskokatolícka cirkev v Slovenskej republike (rímskokatolícke)":
        "christianity.catholic.latin",
    "Gréckokatolícka cirkev na Slovensku (gréckokatolícke)":
        "christianity.catholic.eastern",
    "Evanjelická cirkev augsburského vyznania na Slovensku (evanjelické)":
        "christianity.lutheran",
    "Reformovaná kresťanská cirkev na Slovensku (kalvínske)":
        "christianity.reformed",
    "Pravoslávna cirkev na Slovensku (pravoslávne)":
        "christianity.orthodox.canonical",
    "Kresťanské zbory na Slovensku":
        "christianity.plymouth",
    "Náboženská spoločnosť Jehovovi svedkovia v Slovenskej republike":
        "christianity.witnesses",
    "Apoštolská cirkev na Slovensku":
        "christianity.pentecostal.trinitarian",
    "Evanjelická cirkev metodistická, Slovenská oblasť":
        "christianity.methodist",
    "bez náboženského vyznania":
        "unaffiliated",
    "ostatné":
        "other.sk",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
