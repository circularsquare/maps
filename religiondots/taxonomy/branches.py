"""
The internal nodes of the religion tree — the containment relation of spec.md §2.1.

Source-independent. Leaves are contributed by per-source mapping files (usrc2020.py and
whatever follows it); this file is only the structure they hang from, so that two sources
naming the same body land in the same place.

Every entry is (id, label, note). `note` is for the reasoning that would otherwise be lost —
why a node exists, or why something contested sits where it does. Order here is display
order within a parent.

Rules, from spec.md §2:
  - containment, not descent. "Every Jesuit is a Latin Catholic" — a fact about people now.
    Historical descent is a separate DAG and does not live here.
  - a node earns its place by being *countable* somewhere, not by existing. Depth is uneven
    and that is correct; the US runs deep because ASARB enumerates 372 bodies by county.
  - nothing is created silently at ingest. A source category with no home goes to unmapped.
"""

BRANCHES = [
    # ---------------------------------------------------------------- Christianity
    ("christianity", "Christianity", ""),

    ("christianity.catholic", "Catholic", ""),
    ("christianity.catholic.latin", "Latin Catholic", ""),
    ("christianity.catholic.eastern",
     "Eastern Catholic",
     "The 23 sui iuris churches in communion with Rome — Ukrainian Greek, Maronite, Chaldean, "
     "Melkite, Syro-Malabar. Added 2026-08-27 when Canada arrived: ASARB has no category for "
     "them, StatCan enumerates nine. Distinct from independent Catholic bodies, which are NOT "
     "in communion, and from Eastern Orthodoxy, which is a different communion again."),
    ("christianity.catholic.independent",
     "Independent Catholic",
     "Bodies using Catholic orders and rites outside communion with Rome — Polish National, "
     "Old Roman, Liberal Catholic and the small ecumenical Catholic jurisdictions. Distinct "
     "from Eastern Catholic churches, which ARE in communion."),

    ("christianity.orthodox", "Eastern Orthodox", ""),
    ("christianity.orthodox.canonical",
     "Canonical Orthodox",
     "Jurisdictions in communion with the recognised patriarchates."),
    ("christianity.orthodox.oldcalendarist",
     "Old Calendarist",
     "Broke over the 1924 calendar reform; out of communion with the canonical churches."),
    ("christianity.orthodox.other",
     "Other Orthodox jurisdictions",
     "Self-declared autocephalous or independent bodies recognised by nobody. Kept separate "
     "rather than folded into canonical, because the distinction is the whole content of "
     "these groups' identity."),

    ("christianity.oriental", "Oriental Orthodox",
     "The non-Chalcedonian churches — Coptic, Ethiopian, Eritrean, Armenian, Syriac, "
     "Malankara. A separate communion from Eastern Orthodoxy since 451, and conflating the "
     "two is the commonest error in religion taxonomies."),

    ("christianity.anglican", "Anglican", ""),
    ("christianity.anglican.continuing",
     "Continuing Anglican",
     "Bodies that left the Episcopal Church, mostly after 1977 and 2003. A crowded field of "
     "small jurisdictions, which is why they get their own node rather than sitting beside "
     "the Episcopal Church as peers."),

    ("christianity.lutheran", "Lutheran", ""),

    ("christianity.reformed", "Reformed", ""),
    ("christianity.reformed.presbyterian", "Presbyterian", ""),
    ("christianity.reformed.continental",
     "Continental Reformed",
     "Dutch, German and related Reformed churches, as against the Scottish Presbyterian line."),
    ("christianity.reformed.congregational", "Congregational", ""),

    ("christianity.baptist", "Baptist", ""),
    ("christianity.baptist.national",
     "National Baptist conventions",
     "The historically African-American conventions."),
    ("christianity.baptist.freewill", "Free Will Baptist", ""),
    ("christianity.baptist.primitive", "Primitive Baptist", ""),
    ("christianity.baptist.oldregular", "Old Regular Baptist", ""),
    ("christianity.baptist.landmark",
     "Landmark Baptist",
     "The ABA/BMA line, holding to Baptist church succession."),
    ("christianity.baptist.independent",
     "Independent Fundamental Baptist",
     "The separatist fundamentalist associations."),
    ("christianity.baptist.reformed", "Reformed Baptist", ""),

    ("christianity.methodist", "Methodist", ""),
    ("christianity.methodist.african",
     "African Methodist",
     "AME, AME Zion, CME and the smaller bodies of the same origin — separated over racial "
     "exclusion, not doctrine, which is why they are a branch rather than scattered."),
    ("christianity.methodist.holiness",
     "Holiness Methodist",
     "Bodies that left Methodism over the holiness movement, keeping Methodist polity."),
    ("christianity.methodist.unitedbrethren",
     "United Brethren and Evangelical",
     "The German-speaking American parallel to Methodism; most of it merged into the UMC."),

    ("christianity.holiness", "Holiness", ""),

    ("christianity.pentecostal", "Pentecostal", ""),
    ("christianity.pentecostal.trinitarian", "Trinitarian Pentecostal", ""),
    ("christianity.pentecostal.oneness",
     "Oneness Pentecostal",
     "Rejects the Trinity and baptises in Jesus' name. The 1916 split is the deepest "
     "division inside Pentecostalism and is invisible if these sit among the others."),
    ("christianity.pentecostal.charismatic",
     "Charismatic and neo-charismatic",
     "Later movements — Vineyard, Calvary Chapel — outside the classical Pentecostal bodies."),

    ("christianity.restorationist",
     "Stone-Campbell Restorationist",
     "Churches of Christ, Christian Churches, Disciples. 'Restorationist' is also used for "
     "the Latter Day Saints; kept to the Stone-Campbell movement here and the LDS branch "
     "named for itself, to avoid the ambiguity."),

    ("christianity.adventist", "Adventist", ""),
    ("christianity.latterday", "Latter Day Saints", ""),

    ("christianity.anabaptist", "Anabaptist", ""),
    ("christianity.anabaptist.amish", "Amish", ""),
    ("christianity.anabaptist.mennonite", "Mennonite", ""),
    ("christianity.anabaptist.hutterite", "Hutterite", ""),
    ("christianity.anabaptist.brethren",
     "Schwarzenau Brethren",
     "The German Baptist Brethren line — Church of the Brethren, Old German Baptist, "
     "Dunkard, Brethren in Christ. Not to be confused with the Plymouth Brethren, who are "
     "unrelated, or with the Moravians, who call themselves Unity of the Brethren."),
    ("christianity.anabaptist.apostolic",
     "Apostolic Christian",
     "The Fröhlich line, Swiss Anabaptist in origin."),

    ("christianity.friends", "Friends (Quakers)", ""),
    ("christianity.moravian", "Moravian", ""),
    ("christianity.plymouth", "Plymouth Brethren", ""),
    ("christianity.pietist",
     "Pietist",
     "The Scandinavian free-church line — Evangelical Covenant, Evangelical Free."),
    ("christianity.nondenominational",
     "Non-denominational",
     "By construction the least satisfying node on the tree, and the second largest in the "
     "United States at 21.1M. It is a real answer people give, not a failure to classify."),

    ("christianity.witnesses", "Jehovah's Witnesses", ""),
    ("christianity.christianscience", "Christian Science", ""),
    ("christianity.swedenborgian", "Swedenborgian", ""),
    ("christianity.messianic",
     "Messianic Judaism",
     "CONTESTED PLACEMENT. Filed under Christianity because the theology is Christian; "
     "adherents generally identify as Jewish, and Jewish institutions do not accept the "
     "claim. Flagged rather than settled — see review notes in usrc2020.py."),
    ("christianity.other",
     "Other Christian",
     "Bodies with no branch to belong to, not a residual. The computed residual is a "
     "different thing and is named '…, other or unspecified' — see spec.md §3.2."),

    # ---------------------------------------------------------------- other families
    ("judaism", "Judaism", ""),
    ("islam", "Islam", ""),
    ("hinduism", "Hinduism", ""),
    ("buddhism", "Buddhism", ""),
    ("sikhism", "Sikhism", ""),
    ("jainism", "Jainism", ""),
    ("zoroastrianism", "Zoroastrianism", ""),
    ("bahai", "Bahá'í", ""),
    ("shinto", "Shinto", ""),
    ("daoism", "Daoism", ""),
    ("unification", "Unification Church", ""),
    ("newthought", "New Thought", ""),
    ("spiritualism", "Spiritualism", ""),
    ("unitarianuniversalist",
     "Unitarian Universalism",
     "Christian in origin, not now in self-description. Top level rather than under "
     "Christianity for that reason."),
    ("hebrewisraelite",
     "Hebrew Israelite",
     "CONTESTED PLACEMENT. Not a branch of Judaism, and Judaism does not recognise the "
     "claim; not straightforwardly Christian either. Own family, flagged."),
    ("secular",
     "Secular and ethical",
     "Organised non-theistic bodies and stated secular positions — Ethical Culture, and "
     "Canada's Atheist / Agnostic / Humanist answers. Distinct from `unaffiliated`, which is "
     "the absence of an answer rather than a position."),

    # --- added 2026-08-27, when Canada became the second source (spec §2.4: the tree grows
    #     where a source reaches). ASARB has no category for any of these.
    ("unaffiliated",
     "No religion",
     "People who report no religion. NOT the same quantity as the US residual, which is "
     "'absent from every membership roll' — see spec §3.1 on the basis clash at the border. "
     "Held as a real node because Canada, Australia, NZ, Ireland and the UK all measure it "
     "directly and it is the largest single answer in several of them."),
    ("paganism",
     "Pagan and nature religions",
     "Wicca, Neopaganism, Druidry, nature religions. StatCan groups these explicitly; the ABS "
     "buries them inside Other Religious Groups (§3.9)."),
    ("chinesefolk",
     "Chinese religions",
     "Chinese folk religion, Confucianism and the syncretic practice that census categories "
     "usually cannot separate — the §3.3 combination node for China's own tradition. Daoism "
     "is a sibling top-level family rather than a child, because sources enumerate it "
     "separately where they enumerate anything at all."),
    ("indigenous",
     "Indigenous and traditional religions",
     "Traditional religions of indigenous peoples, kept as one family with regional children "
     "rather than scattered, because sources report them that way."),
    ("indigenous.northamerican", "North American Indigenous spirituality", ""),
    ("druze",
     "Druze",
     "Its own family. Sometimes filed under Islam by sources and by convention; the Druze "
     "themselves generally do not accept that placement, and StatCan lists it separately."),
]
