"""
Zensus 2022 religion classification -> religiondots taxonomy.

**The shallowest mapping in the project, and the only one where that is a property of the
instrument rather than of the table.** Three categories, and the reason there are three is
that nobody was asked: Zensus 2022 has no religion question, and the figures are read off
the Melderegister, which records religious-body membership because it determines church-tax
liability. So the data can see the two churches that levy church tax and nothing else.
`basis` is `roll` (spec §3.1) and sources/de.py's docstring has the full argument.

Croatia is drawn shallow ON PURPOSE — DZS publishes 54 named churches at the same geography
and joining them has not been done (hr2021.py). Germany is different: there is no deeper
table anywhere, in this census or in 2011, and `sources/de.md` §2 records why the 2011
attempt failed. Nothing here is deferred work.

WHAT THIS COSTS, stated plainly because the map cannot say it and the about panel must:
Germany's roughly four million Muslims, its two million Orthodox Christians, its Jewish
communities and its Freikirchen are all inside `unrecorded`, indistinguishable from the
people who belong to nothing. That is not a modelling choice — the register never knew.
Drawing them would mean inventing both a magnitude and a location, which spec §14.3 forbids
in exactly these words: never model at a finer resolution, or a stronger claim, than the
source publishes its magnitude at.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Einwohnerzahl":
        "the unit's own population total, not a category. Note it is NOT the sum of the "
        "three categories: the Cell-Key disclosure method perturbs the category cells and "
        "leaves the Einwohnerzahl untouched, so the three fall short of it by 174 people "
        "nationally (sources/de.py reconciles this).",
}

REVIEW = {
    "Römisch-katholische Kirche (öffentlich-rechtlich)":
        "-> christianity.catholic, the PARENT, not christianity.catholic.latin. Croatia's "
        "`Katolici` is filed the same way for the same reason: the category is the Roman "
        "Catholic Church as a legal body, and Germany's Eastern Catholics pay their church "
        "tax through it, so it is not the Latin rite alone. destatis is explicit that it "
        "EXCLUDES the Alt-Katholiken ('nicht aber der Altkatholiken und verwandter "
        "Gruppen'), who are a separate public-law body and therefore sit, uncounted, "
        "inside `unrecorded` — christianity.catholic.independent exists and gets nothing.",

    "Evangelische Kirche (öffentlich-rechtlich)":
        "-> christianity.protestant, whose own note in branches.py already named Germany "
        "as a case it would be wanted for. The category is the EKD, which destatis "
        "defines as 'der Zusammenschluss der zwanzig selbständigen lutherischen, "
        "reformierten und unierten Landeskirchen'. That spans three of the tree's "
        "families, and the tree has NO Protestant super-node on purpose — Lutheran and "
        "Reformed are siblings, not children of one. So this is the second kind of thing "
        "`christianity.protestant` holds: not 'the answer Protestant with no body named' "
        "(Czechia, Croatia) but 'a named body that spans the Protestant families'. "
        "christianity.united was considered and rejected: that node is for churches formed "
        "BY union, like the Uniting Church in Australia, and the EKD is a federation whose "
        "twenty members stayed independent. Splitting the 19.1M across lutheran / reformed "
        "/ united by Landeskirche would be possible — the Landeskirchen have territories — "
        "but destatis publishes one number and the split would be an allocation inventing "
        "structure the source does not have (spec §3.10).",

    "Sonstige, keine, ohne Angabe":
        "-> unrecorded, a node added for this source (branches.py, 2026-09-04) and the "
        "single most consequential call in this file: 42,845,220 people, 51.8% of Germany. "
        "Every existing home for it asserts something false. `unaffiliated` is a person "
        "reporting no religion, and nobody was asked. `other.de` would be a religion the "
        "source named but the tree cannot place, and the source named nothing. "
        "`unchurched` is a positive report of belief without institution. destatis' own "
        "definition is the argument for a separate node — the bucket holds people in OTHER "
        "public-law bodies too, because 'für diese anderen öffentlich-rechtlichen "
        "Religionsgesellschaften liegen nur in sehr begrenztem Umfang Einträge im "
        "Melderegister vor'. So it is three different things at once (another body, no "
        "body, no entry) and its composition is a fact about the register rather than "
        "about the people. It gets the greyest treatment in the §6.3a family for that "
        "reason.",
}

MAP = {
    "Römisch-katholische Kirche (öffentlich-rechtlich)": "christianity.catholic",
    "Evangelische Kirche (öffentlich-rechtlich)": "christianity.protestant",
    "Sonstige, keine, ohne Angabe": "unrecorded",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
