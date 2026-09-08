"""INE Censo 2024 religion classification -> religiondots taxonomy.

Thirteen named categories plus Ninguna and a non-response line, on 346 comunas, for
residents aged 15 or over.

**Chile names bodies that most censuses of its size do not.** Jehovah's Witnesses (121,805),
the Latter-day Saints (94,266), the Orthodox (10,912) and the Bahá'í (2,010) each get their
own cell, and at four and three figures. The question was written with the Oficina Nacional
de Asuntos Religiosos rather than inherited from 2002, and it shows: this is a deeper list
than North Macedonia's and far deeper than Sri Lanka's, on a country of 18M.

**What it does NOT split is the big one.** `Evangélica o protestante` is 2,466,607 people,
16.2%, in a single cell — and Chilean Protestantism is overwhelmingly Pentecostal, with the
Iglesia Metodista Pentecostal and the Iglesia Evangélica Pentecostal the largest bodies in
the country after the Catholic Church. The tree can hold Pentecostals apart and Chile has
more of them per head than almost anywhere; no table here separates them. See REVIEW.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Población de 15 años o más":
        "the unit's own universe total, not a category.",
    "Religión o credo no declarado":
        "87,515 people, 0.58% of the 15+ universe, who did not answer question 31. A "
        "non-response, not an answer, and spec §3.5 keeps those off the tree rather than "
        "filling them. Chile is drawn at 81.8% of its population — this plus the 15+ "
        "universe — and countries.py says so.",
}

REVIEW = {
    "Católica":
        "-> christianity.catholic.latin, matching mx2020.py's identical `Católica`. Chile "
        "lists `Católica Ortodoxa` as a separate option, so the bare word here means the "
        "Latin church and not Catholicism in general. 8,168,978 people, 53.7% of the 15+ "
        "population — and worth noting against 70.0% in 2002 and 76.9% in 1992, which is "
        "one of the fastest declines any census on this map records.",
    "Evangélica o protestante":
        "-> christianity.protestant, the 'named no body' node. 2,466,607 people, 16.2%, and "
        "the least satisfying mapping in this file because the category hides the most "
        "interesting thing about Chilean religion. Chile's Protestants are perhaps 80-90% "
        "Pentecostal — the Iglesia Metodista Pentecostal and Iglesia Evangélica Pentecostal "
        "between them are the second and third largest religious bodies in the country — and "
        "`christianity.pentecostal` exists on the tree, drawn for the US, Brazil and "
        "elsewhere. Filing Chile there anyway would assert a split the census does not make, "
        "for a proportion taken from general knowledge rather than measured, which is what "
        "§2 forbids at ingest time. The parent says what the source says; the distinction "
        "survives in `source_category` if a later table ever supports it.",
    "Católica Ortodoxa":
        "-> christianity.orthodox, whose label in the tree is 'Eastern Orthodox', and that "
        "is a slightly stronger claim than the source makes. 10,912 people. Chile's Orthodox "
        "are overwhelmingly Antiochian — Palestinian and Syrian Christian immigration from "
        "the late 19th century, under the Greek Orthodox Patriarchate of Antioch — which is "
        "Eastern; there is a smaller Russian presence, also Eastern, and an Armenian "
        "community which is Oriental and would belong elsewhere. INE's single cell cannot "
        "separate them. Filed on the Eastern node because that is where the large majority "
        "is and where mk2021.py files North Macedonia's bare `Orthodox`.",
    "Otros cristianos y tradiciones relacionadas con Cristo":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots' and "
        "renders as a Christianity `unspecified` row. 199,426 people, 1.3%. **This is a "
        "DERIVED category and INE says so**: it does not appear on the questionnaire. It is "
        "built by coding the free-text answers given under option 11 (`Otra religión o "
        "credo`) and pulling out the ones that name a Christian movement not matching any of "
        "the listed options. So it is a residual within Christianity produced by INE's own "
        "coding, which is exactly what the root node means, and it must not be confused with "
        "`Otras religiones o credos`, which is what stayed in option 11 afterwards.",
    "Ninguna":
        "-> unaffiliated. 3,903,308 people, **25.7% of the 15+ population**, against 8.3% "
        "declaring no religion in 2002. INE's own wording is 'declaran no profesar ninguna "
        "de las religiones o credos anteriormente señalados', which is an absence of "
        "affiliation rather than a stated position, so nothing goes to `secular` — Chile "
        "offers no agnostic or atheist option to separate out. Note the age gradient behind "
        "it: 96.0% of over-65s profess a religion against 63.9% of 15-29s.",
    "Otras religiones o credos":
        "-> other.cl. 89,856 people, 0.59%. Question 31's option 11, an open write-in, MINUS "
        "the Christian answers INE recoded into `Otros cristianos`. So unlike most residuals "
        "on this map it has been actively reduced by the source before publication, and what "
        "is left is genuinely non-Christian and unclassified. Per source, per spec §3.11.",
}

MAP = {
    "Católica": "christianity.catholic.latin",
    "Evangélica o protestante": "christianity.protestant",
    "Judía": "judaism",
    "Musulmana": "islam",
    "Iglesia de Jesucristo de los Santos de los Últimos Días": "christianity.latterday",
    "Católica Ortodoxa": "christianity.orthodox",
    "Budista": "buddhism",
    "Hinduista": "hinduism",
    "Fe Bahá'í": "bahai",
    "Testigo de Jehová": "christianity.witnesses",
    "Otros cristianos y tradiciones relacionadas con Cristo": "christianity",
    "Otras religiones o credos": "other.cl",
    "Ninguna": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
