"""Swiss Volkszählung 2000 religion classification -> religiondots taxonomy.

Eighteen categories plus the universe total, at commune. **The deepest Protestant breakdown on
this map outside the United States** — seven separate cells where most European censuses have
one — and the reason Switzerland is worth drawing on a source this old.

    36.82%  Keine Zugehörigkeit                    -> unaffiliated
    30.00%  Römisch-katholische Kirche             -> christianity.catholic.latin
    18.74%  Evangelisch-reformierte Kirche         -> christianity.reformed
     6.03%  Islamische Gemeinschaften              -> islam
     2.46%  Christlich-orthodoxe Kirchen           -> christianity.orthodox.canonical
     0.99%  Übrige protestantische Kirchen         -> christianity.protestant
     0.90%  Ohne Angabe                            -> EXCLUDED (non-response)
     0.61%  Hinduistische Vereinigungen            -> hinduism
     0.59%  Neupietistisch-evangelikale Gemeinden  -> christianity.pietist
     0.51%  Neuapostolische Kirchen                -> christianity.other
     0.50%  Buddhistische Vereinigungen            -> buddhism
     0.39%  Zeugen Jehovas                         -> christianity.witnesses
     0.38%  Pfingstgemeinden                       -> christianity.pentecostal
     0.27%  Andere christliche Gemeinschaften      -> christianity
     0.24%  Christkatholische Kirche               -> christianity.catholic.independent
     0.20%  Jüdische Glaubensgemeinschaft          -> judaism
     0.19%  Übrige Kirchen und Religionsgemeinsch. -> other.ch
     0.15%  Evangelisch-methodistische Kirche      -> christianity.methodist

**THE PERCENTAGES ABOVE ARE 2024's AND THE CATEGORIES ARE 2000's, AND THAT IS THE WHOLE
COUNTRY.** `ch_rescale.py` fits current Strukturerhebung canton totals onto the 2000 census's
commune-by-category structure (spec §3.4). So the *shares* here are what Switzerland is now —
the collapse from 41.8% Catholic and 33.0% Reformed in 2000 is real and is the largest
religious change on this map inside one generation — while the *split inside each survey
group* is 2000's and is carried forward. Read the seven Protestant cells as a 2000 partition
of a 2024 magnitude, and see `ch_rescale.py` for what that can and cannot see.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Religionen - Total":
        "the unit's own population total, not a category.",
    "Ohne Angabe":
        "no answer given. 4.33% of the 2000 census and 0.90% of the 2024 survey — the fall "
        "is what a voluntary interview does that a mandatory form does not. spec §3.5 "
        "applies and these people are not drawn. **It is NOT `Keine Zugehörigkeit`**, which "
        "is a separate answer taken by 11.11% in 2000 and 36.82% now; conflating a refusal "
        "with a report of no religion is the §3.1 error this project exists to avoid.",
}

REVIEW = {
    "Evangelisch-reformierte Kirche":
        "-> christianity.reformed. The cantonal Landeskirchen of the Swiss "
        "Reformation — Zwingli's Zurich, Calvin's Geneva — federated as the Evangelical "
        "Reformed Church of Switzerland. This is the **single largest Reformed body on this "
        "map** and the tradition's place of origin, which is why it takes the named family "
        "node rather than `christianity.protestant` — that node holds the ANSWER "
        "'Protestant' and is deliberately not a parent of anything (branches.py). Its "
        "collapse is the country's story: 33.0% in 2000, 18.7% now.",
    "Römisch-katholische Kirche":
        "-> christianity.catholic.latin, and NOT the `christianity.catholic` parent, "
        "because BFS counts the Christ Catholics separately and this cell therefore means "
        "the Roman church specifically.",
    "Christkatholische Kirche":
        "-> christianity.catholic.independent. **The Old Catholic church, and this is the "
        "call most likely to be questioned.** It broke with Rome over papal infallibility "
        "in 1871, keeps Catholic orders and rites, and is not in communion — which is "
        "branches.py's definition of `independent` exactly, and is why it is not filed "
        "under `.latin` with the Roman church or under `.eastern` with the Uniate ones. "
        "Switzerland is one of the three countries where it is a recognised public-law "
        "church rather than a remnant, alongside Germany and Austria; 18,343 people, "
        "concentrated in the Jura and Solothurn.",
    "Christlich-orthodoxe Kirchen":
        "-> christianity.orthodox.canonical. Serbian, Greek, Russian and Romanian "
        "jurisdictions, all canonical, none separated by the source. **This is the fastest "
        "growing large category in the country** — 1.81% in 2000 against 2.46% now — and "
        "the growth is Balkan and Eritrean migration. Note what §3.4 cannot do here: the "
        "2024 magnitude is right, but its split against Pentecostals and the free churches "
        "inside `Autres communautés chrétiennes` is 2000's, and Orthodoxy has grown far "
        "faster than they have, so this cell is if anything understated.",
    "Neupietistisch-evangelikale Gemeinden":
        "-> christianity.pietist. BFS's own grouping for the Swiss free "
        "churches of pietist descent — the Freie Evangelische Gemeinden, the Chrischona "
        "and Evangelical Alliance congregations, the Darbysts of the Bernese Jura. Not "
        "`pentecostal`, which BFS counts separately in the next cell, and not "
        "`christianity.protestant` unqualified, which would lose the distinction the "
        "source paid to make.",
    "Pfingstgemeinden":
        "-> christianity.pentecostal. Kept apart from the evangelical cell "
        "above because BFS keeps them apart; almost no other European census on this map "
        "can express the difference at all.",
    "Neuapostolische Kirchen":
        "-> christianity.other, **which is au2021.py's and lt2021.py's call for the same "
        "body** and is followed here rather than improved on. The New Apostolic Church "
        "descends from the Catholic Apostolic (Irvingite) movement, has a living apostolate "
        "of its own, and belongs to no Protestant family the tree holds; "
        "`christianity.restorationist` is scoped to the Stone-Campbell movement and is not "
        "it. **Switzerland makes this the largest instance by far** — 38,611 people against "
        "Lithuania's 412 — which is an argument for a node of its own if a third source "
        "ever counts it, and not an argument for diverging from two existing ones now.",
    "Zeugen Jehovas":
        "-> christianity.witnesses.",
    "Evangelisch-methodistische Kirche":
        "-> christianity.methodist. Small — 11,393 — and named because BFS "
        "names it.",
    "Übrige protestantische Kirchen und Gemeinschaften":
        "-> christianity.protestant, which holds the ANSWER 'Protestant' where no body is "
        "named — not a parent of the Protestant families, which are its siblings. 74,683 "
        "people. **Unlike most countries' single Protestant cell this one really is a "
        "leftover**: it is the residual of a seven-cell Protestant question, so it means "
        "'some other Protestant church' rather than 'Protestant' tout court.",
    "Andere christliche Gemeinschaften":
        "-> christianity, the ROOT, for an answer that is Christian and names no church. "
        "Sits beside a named Protestant residual, a named Catholic residual and a named "
        "Orthodox cell, so it is what is left when none of those fits — the same call "
        "mk2021.py makes for `Christians` and cz2021.py for `křesťanství`.",
    "Islamische Gemeinschaften":
        "-> islam, the parent, with no branch. Switzerland's Muslims are overwhelmingly "
        "Bosniak, Kosovar, Albanian and Turkish — Hanafi Sunni — and the census does not "
        "say so. 6.03% now against 4.27% in 2000.",
    "Keine Zugehörigkeit":
        "-> unaffiliated and NOT `secular`. A report of no religious affiliation, not a "
        "stated atheist or humanist position, which is exactly branches.py's distinction "
        "between the two. **It is now the largest single answer in Switzerland at 36.82%**, "
        "having been 11.11% in 2000 — the sharpest secularisation any country on this map "
        "records between two measurements of its own population.",
    "Übrige Kirchen und Religionsgemeinschaften":
        "-> other.ch, a per-source residual (§3.11). See branches.py.",
    "Jüdische Glaubensgemeinschaft":
        "-> judaism, the parent. No denominational split is published; Switzerland's "
        "communities run from the Orthodox congregations of Zurich and Basel to the liberal "
        "ones, and nothing in the source separates them.",
    "Buddhistische Vereinigungen":
        "-> buddhism, the parent. No school is named. The population is Thai, Tibetan, "
        "Vietnamese and Swiss convert in unknown proportions, so no branch could be chosen "
        "without inventing one.",
    "Hinduistische Vereinigungen":
        "-> hinduism. 46,130 people and **larger than the Jewish and Buddhist communities "
        "combined**, which is not what most readers will expect: it is overwhelmingly Sri "
        "Lankan Tamil, from the asylum migration of the 1980s and 1990s, and it is the "
        "second-largest Tamil Hindu population in Europe after Britain's.",
}

MAP = {
    "Evangelisch-reformierte Kirche": "christianity.reformed",
    "Evangelisch-methodistische Kirche": "christianity.methodist",
    "Neupietistisch-evangelikale Gemeinden": "christianity.pietist",
    "Pfingstgemeinden": "christianity.pentecostal",
    "Neuapostolische Kirchen": "christianity.other",
    "Zeugen Jehovas": "christianity.witnesses",
    "Übrige protestantische Kirchen und Gemeinschaften": "christianity.protestant",
    "Römisch-katholische Kirche": "christianity.catholic.latin",
    "Christkatholische Kirche": "christianity.catholic.independent",
    "Christlich-orthodoxe Kirchen": "christianity.orthodox.canonical",
    "Andere christliche Gemeinschaften": "christianity",
    "Jüdische Glaubensgemeinschaft": "judaism",
    "Islamische Gemeinschaften": "islam",
    "Buddhistische Vereinigungen": "buddhism",
    "Hinduistische Vereinigungen": "hinduism",
    "Übrige Kirchen und Religionsgemeinschaften": "other.ch",
    "Keine Zugehörigkeit": "unaffiliated",
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
