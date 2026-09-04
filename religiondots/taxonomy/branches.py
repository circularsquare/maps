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
    ("christianity.orthodox.oldbeliever",
     "Old Believer",
     "Broke over Patriarch Nikon's 1653-66 revision of the Russian service books — three "
     "centuries before the Old Calendarists and over a different question, so folding the "
     "two together would be wrong twice. spec §R2 names Old Believers as one of the test "
     "cases the map has to be able to show. Added 2026-09-03 with Poland, which publishes "
     "two of them by name: Wschodni Kościół Staroobrzędowy (343) and Staroprawosławna "
     "Cerkiew Staroobrzędowców (235)."),
    ("christianity.orthodox.other",
     "Other Orthodox jurisdictions",
     "Self-declared autocephalous or independent bodies recognised by nobody. Kept separate "
     "rather than folded into canonical, because the distinction is the whole content of "
     "these groups' identity."),

    ("christianity.oriental", "Oriental Orthodox",
     "The non-Chalcedonian churches — Coptic, Ethiopian, Eritrean, Armenian, Syriac, "
     "Malankara. A separate communion from Eastern Orthodoxy since 451, and conflating the "
     "two is the commonest error in religion taxonomies."),

    ("christianity.churchofeast",
     "Church of the East",
     "The Assyrian Church of the East and the Ancient Church of the East — the Dyophysite "
     "(historically 'Nestorian') line, which is a THIRD communion, neither Eastern Orthodox "
     "nor Oriental Orthodox. It separated in 431, twenty years before Chalcedon split the "
     "other two from each other. The ABS gets this right and gives it its own group (222), "
     "distinct from 221 Oriental and 223 Eastern; most sources fold it into one or the "
     "other, which is wrong twice over. Added 2026-09-03 with Australia."),

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

    ("christianity.united",
     "United and uniting churches",
     "Churches formed by union across the Protestant families, where the result is not a "
     "branch of any one of them — the Uniting Church in Australia (673,383, its third "
     "largest Christian body), the United Church of Canada, the Church of South India. "
     "Filing the Uniting Church under Methodist because Methodism was its largest strand "
     "would lose the Presbyterians and Congregationalists who are equally in it. Added "
     "2026-09-03 with Australia."),

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

    ("christianity.hussite",
     "Hussite",
     "The Czechoslovak Hussite Church, 23,610 in 2021 and the third largest Christian body "
     "in Czechia after the Roman Catholics and the Czech Brethren. Founded 1920 out of "
     "Catholic modernism and claiming the Hussite reformation; it is neither Catholic nor "
     "any of the Protestant families, which is why it needs a node rather than a shelf. "
     "Added 2026-09-02 when Czechia arrived."),

    ("christianity.protestant",
     "Protestant, unspecified",
     "For sources that collect 'Protestant' or 'Evangelical' as an answer without naming a "
     "body — Czechia's 'protestantská/evangelická víra' (27,149). Deliberately NOT a "
     "parent of the Protestant families: the tree has no Protestant super-node, because "
     "Lutheran, Reformed, Baptist and the rest are siblings rather than children of one. "
     "This node holds the ANSWER, not the category. Added 2026-09-02; Korea, Germany and "
     "Brazil will all want it."),

    ("christianity.biblestudent",
     "Bible Students",
     "The part of Russell's Bible Student movement that did NOT follow Rutherford after "
     "1917 — Free Bible Students, the Laymen's Home Missionary Movement, and the smaller "
     "associations. They are the older body and the Witnesses are the split, so filing "
     "them under `christianity.witnesses` would invert the descent as well as merge two "
     "groups that have been separate for a century. GUS gives them a classification group "
     "of their own (`nurt badaczy Pisma Świętego`) holding four bodies, of which one is "
     "the Witnesses and three are these. Added 2026-09-03 with Poland."),
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
    ("spiritualism.kardecist",
     "Kardecist Spiritism",
     "Allan Kardec's codification, and by far the largest branch of Spiritism in the world "
     "— 3.85M in Brazil in 2010, its fifth largest religion, against a few tens of "
     "thousands of Anglo-American Spiritualists everywhere else. Same 19th-century movement "
     "as `spiritualism` and institutionally quite separate from it, which is why it is a "
     "child rather than the same node. Added 2026-09-03 with Brazil."),
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

    # ---- India, added 2026-09-03 with Census 2011.
    #
    # This is the largest indigenous-religion population any source on the map reaches:
    # 7.94M people in the census's `Other religions and persuasions`, of whom 7.79M are in
    # 83 NAMED religions in the C-01 Appendix. Nowhere else does a census name Sarna,
    # Sanamahi or Donyi-Polo at all, so the depth here is earned by countability in the
    # §2.4 sense and not by enthusiasm.
    #
    # Five children exist because five groups are large and regionally distinct; everything
    # else in the tail sits on `indigenous.indian` itself (§6.6: a branch that carries dots
    # is a category). The tail is genuinely long — 60-odd names between 100 and 20,000
    # people, most of them one people in one district — and giving each a node would add
    # sixty categories nobody can see to satisfy a taxonomy nobody reads.
    ("indigenous.indian",
     "Adivasi and tribal religions (India)",
     "The traditional religions of India's Scheduled Tribes and other Adivasi peoples. "
     "India's census does not offer them a box: the form lists six religions, and every "
     "one of these 7.9M people had to be written in under `Other religions and "
     "persuasions`, which is why the count is a floor and not a measurement. Many more "
     "Adivasi are recorded as Hindu, some as Christian, and the boundary is politically "
     "live — the demand for a separate `Sarna` code on the census form has been running "
     "since the 1951 census and was refused again for 2011."),
    ("indigenous.indian.sarna",
     "Sarna",
     "The sacred-grove religion of the Chotanagpur plateau — Oraon, Munda, Ho and Santal "
     "peoples across Jharkhand, Odisha and West Bengal. 4,957,467 as `Sarna` plus the "
     "local names for the same religion (see the REVIEW note in in2011.py), which makes "
     "it the largest single indigenous religion on the map by an order of magnitude."),
    ("indigenous.indian.gondi",
     "Gondi (Koya Punem)",
     "The religion of the Gond peoples of central India — Madhya Pradesh, Chhattisgarh, "
     "Maharashtra, Telangana. 1,026,344 under `Gond / Gondi`, and the associated Koyatur "
     "and Budhadeo write-ins."),
    ("indigenous.indian.donyipolo",
     "Donyi-Polo",
     "The Sun-and-Moon religion of the Tani peoples of Arunachal Pradesh — Nyishi, Adi, "
     "Apatani, Galo — codified into an organised religion with congregational worship "
     "from the 1970s in response to Christian missions. 331,370 under `Doni Polo / "
     "Sidonyi Polo`, and effectively all of Arunachal's smaller named traditions."),
    ("indigenous.indian.sanamahi",
     "Sanamahi",
     "Meitei Sanamahism, the pre-Hindu religion of Manipur, revived from the early 20th "
     "century against the Vaishnavism imposed in the 18th. 222,422 in the Appendix, and "
     "the Annexure separately records 41,673 people who wrote `Meitei` under Hinduism — "
     "the same revival showing up on both sides of a boundary the census cannot see."),
    ("indigenous.indian.khasi",
     "Niam Khasi and Niamtre",
     "The traditional religion of the Khasi and Jaintia hills of Meghalaya, in a state "
     "that is 75% Christian. Niam Khasi (`Khasi`, 138,512) and Niamtre (84,276, the "
     "Jaintia/Pnar form) are sister traditions and are kept in one node because the census "
     "names them at the same level and no source separates their adherents further."),

    ("druze",
     "Druze",
     "Its own family. Sometimes filed under Islam by sources and by convention; the Druze "
     "themselves generally do not accept that placement, and StatCan lists it separately."),

    # --- added 2026-09-02, when Czechia arrived. Every one of these is earned by a
    #     countable Czech category (spec §2: a node earns its place by being countable
    #     somewhere, not by existing).
    ("rastafari",
     "Rastafari",
     "Its own family rather than a branch of Christianity. 190 in Czechia, which is small, "
     "but the UK and Jamaica both enumerate it and it will recur."),

    ("scientology",
     "Scientology",
     "397 in Czechia, where it is a recognised category on the census form. Not filed under "
     "`esoteric`: it is a single organisation with a membership, not a current."),

    ("esoteric",
     "Esoteric, New Age and Western occult",
     "The Western esoteric current and the new religious movements that come out of it — "
     "Czechia's Hnutí Grálu, Společenství Josefa Zezulky, New Age, esoterismus and "
     "satanismus. Distinct from `paganism`, which reconstructs pre-Christian religion, and "
     "from `spiritualism` and `newthought`, which are specific 19th-century movements "
     "rather than the wider current."),

    ("unchurched",
     "Believing, no church",
     "People who report religious belief AND explicitly no institutional affiliation. "
     "960,201 in Czechia — 9.1% of the country and its third largest answer. NOT "
     "`unaffiliated`, which is a report of no religion, and not `secular`, which is a "
     "stated non-theistic position. Czechia is the first source to measure it directly and "
     "it is the classic 'believing without belonging' category."),

    # --- added 2026-09-04, when Germany became the first register-only source.
    ("unrecorded",
     "Religion not recorded",
     "The residual of a source that reads religion off an ADMINISTRATIVE REGISTER rather "
     "than asking anybody: people for whom the register holds no religious body. "
     "Germany's 'Sonstige, keine, ohne Angabe' — 42,845,220 people, 51.8% of the "
     "country and the largest single node on the map after the US. It is a sixth member "
     "of the §6.3a grey family and it is NOT any of the other five. Not `unaffiliated`, "
     "which is a person reporting no religion — nobody was asked, and this bucket "
     "contains Germany's roughly four million Muslims, its Orthodox Christians, the "
     "Jewish communities and the Freikirchen alongside everyone who belongs to nothing. "
     "Not `other.<source>`, which is a religion the source named but the tree cannot "
     "place; here the source named nothing. Not `unchurched`, which is a positive report "
     "of belief without institution. The German register can only see bodies that levy "
     "church tax, so the category is a property of the INSTRUMENT and not of the people "
     "in it — which is why it needs its own node rather than a home in one of the "
     "others. Any register-basis source with the same shape belongs here: Austria and "
     "the Nordic countries are the obvious next ones."),

    ("parody",
     "Parody and protest answers",
     "Jedi, Sith and pastafarianism, tabulated by ČSÚ because respondents wrote them in. "
     "Kept on the tree rather than dropped because Jedi alone is 21,023 people in 2,512 "
     "Czech municipalities — the 13th largest category in the country, ahead of Jehovah's "
     "Witnesses — and silently discarding that many visible answers would misstate the map. "
     "Filing them under a religion would assert something false, so they get a family whose "
     "label says what they are. England and Wales, Australia and New Zealand all have Jedi "
     "write-ins too."),

    ("other",
     "Other, by source",
     "A CONTAINER, not a claim. spec §3.11: residual buckets are per source and are never "
     "merged, because one country's 'other' holds Orthodox Christians and another's holds "
     "Bahá'í and Wiccans. This node exists only so the per-source residuals below it have a "
     "root; it asserts nothing about what they have in common, and nothing should ever be "
     "mapped to it directly."),
    ("other.cz",
     "Other religion (Czechia)",
     "ČSÚ's 'Jiné' (21,308), plus 'věřící - hlásící se k církvi - název neuveden' (65,567), "
     "the people who say they belong to a church but not which."),
    ("other.ca",
     "Other religions and spiritual traditions (Canada)",
     "StatCan's own residual. Added 2026-09-02: `ca2021.py` has mapped to this id since "
     "Canada was ingested, but the node was never declared here, so it was missing from "
     "religions.json and the viewer did not know it."),

    # --- added 2026-09-03, when Brazil arrived.
    ("afrodiasporic",
     "Afro-diasporic religions",
     "The religions of the African diaspora in the Americas — Candomblé, Umbanda, Santería, "
     "Vodou, Quimbanda. One family with named children rather than a bucket, because the "
     "distinctions inside it are the interesting part: Brazil enumerates Umbanda and "
     "Candomblé separately and they are not the same religion. Not filed under "
     "`indigenous`, which is for the traditional religions of indigenous peoples."),
    ("afrodiasporic.umbanda",
     "Umbanda",
     "Brazilian, syncretic by construction — Kardecist Spiritism, Catholic saints and "
     "Bantu/Yoruba orixás. 407,333 in 2010. spec §3.3: the syncretism gets a node rather "
     "than being split across its ingredients."),
    ("afrodiasporic.candomble",
     "Candomblé",
     "The older and more directly Yoruba/Fon/Bantu of the two, 167,366 in 2010. Umbanda is "
     "the larger by a factor of 2.4 and the two are often reported together — IBGE's own "
     "parent category is 'Umbanda e Candomblé' — but they are distinct traditions."),

    ("japanesenew",
     "Japanese new religions",
     "Sekai Kyūsei Kyō (Igreja Messiânica Mundial), Seicho-no-Ie, Perfect Liberty, "
     "Tenrikyo, Soka Gakkai. Brazil has the largest Japanese diaspora in the world and "
     "enumerates them: 103,716 Messiânica alone in 2010. Distinct from `shinto`, which is "
     "the shrine tradition, and from `buddhism`, though individual bodies here derive from "
     "one or the other. Japan itself will need this node heavily (sources.md §2)."),

    # --- added 2026-09-03, when Australia, Ireland and Mexico were wired. Australia's 148
    #     categories reach further than any source so far except ASARB, and four of these
    #     exist because it names things nobody else does.
    ("christianity.maori",
     "Maori Christian churches",
     "Ratana and Ringatu — churches founded by Maori prophets, Christian in theology and "
     "Maori in authority and practice, and belonging to neither the missionary "
     "denominations that preceded them nor to `indigenous`. 3,246 in Australia; New "
     "Zealand, where they were founded, counts far more."),

    ("alevism",
     "Alevism",
     "The Anatolian tradition of Turkey's Alevi and Kurdish Alevi communities. Usually "
     "filed as a branch of Shi'a Islam, a placement many Alevis reject; given its own "
     "family for the same reason as Druze. 25,657 in England and Wales, where it is a "
     "published write-in category."),

    ("ravidassia",
     "Ravidassia",
     "Followers of Guru Ravidass, who declared themselves a religion separate from Sikhism "
     "in 2010. Before that they were counted Sikh and many still are, so the boundary is "
     "live rather than settled. 9,583 in England and Wales."),

    ("mandaeism",
     "Mandaeism",
     "An ancient Gnostic religion of southern Iraq and Iran venerating John the Baptist, "
     "and one of the smallest surviving religions with a continuous tradition. 9,182 in "
     "Australia, which after the Iraq war holds one of the largest Mandaean communities in "
     "the world — larger than what is left in Iraq."),

    ("yazidism",
     "Yazidism",
     "Its own religion, not a branch of Islam and not Zoroastrianism, though sources file "
     "it as both. 4,125 in Australia."),

    ("caodaism",
     "Caodaism",
     "The Vietnamese syncretic religion founded 1926 — spec §3.3's case, where the "
     "syncretism is the tradition and splitting it across its ingredients would describe "
     "nobody. 677 in Australia; Vietnam's own census counts it in the millions and "
     "sources.md §2 flags it as a reason Vietnam is worth having."),

    ("indigenous.maori",
     "Maori traditional religion",
     "The pre-Christian Maori religion and the part of Stats NZ's `Maori Religions, Beliefs "
     "and Philosophies` that is not one of the named prophetic churches. Ratana, Ringatu "
     "and Paimarire are Christian and sit under `christianity.maori` instead; this is the "
     "remainder, 5,496 in New Zealand."),

    ("indigenous.australian",
     "Australian Aboriginal traditional religions",
     "7,391 in the 2021 census. The ABS names no individual people, as IBGE does not for "
     "Brazil."),

    ("other.au",
     "Other religion (Australia)",
     "The ABS residual leaves — `Religious Groups, nec`, `Other Spiritual Beliefs`, "
     "`Multi Faith` and the nfd rows. Per source, per spec §3.11."),
    ("other.ie",
     "Other religion (Ireland)",
     "CSO's `Other stated religion (nec)`, 22,163."),
    ("other.uk",
     "Other religion (United Kingdom)",
     "The residual leaves of three different classifications — ONS's `Other religions` and "
     "`Own Belief System`, NRS's `Other religion`, NISRA's `Other Religions: Other "
     "Religions`, and NISRA's `Mixed Catholic / Protestant`, which is a statement about "
     "Northern Ireland rather than a denomination. Per source, per spec §3.11."),
    ("other.nz",
     "Other religion (New Zealand)",
     "Stats NZ's `Other Religions, Beliefs and Philosophies` nfd/nec, plus the POLITICAL "
     "IDEOLOGIES it tabulates as religious affiliations because people wrote them in — "
     "Socialism, Marxism, Maoism, Libertarianism, 47 people between them. Not `secular`: a "
     "political programme is not a stated non-theistic position."),
    ("other.mx",
     "Other religion (Mexico)",
     "INEGI's `Origen oriental` (Buddhism, Hinduism and the rest in one bucket), `Cultos "
     "populares` (largely Santa Muerte) and `Otras religiones o movimientos religiosos`. "
     "Mexico is spec §3.11's own worked example of a residual an external estimate could "
     "shrink; nothing here has shrunk it yet."),

    ("other.br",
     "Other religion (Brazil)",
     "IBGE's residual leaves — 'outras religiosidades', 'outras religiões orientais', "
     "'religiosidade não determinada ou mal definida' and the multiple-affiliation "
     "declaration. Per source, per spec §3.11."),
    ("other.hr",
     "Other religion (Croatia)",
     "DZS's `Ostale religije, pokreti i svjetonazori` (37,066) — 'other religions, "
     "movements and life philosophies' — and `Istočne religije` (3,392), which lumps every "
     "Dharmic and East Asian tradition into one cell. The second is filed here reluctantly: "
     "it plainly means Buddhism, Hinduism and their neighbours, but the tree has no node "
     "for 'some Eastern religion, unspecified' and asserting any one of them would be "
     "inventing a fact. See the REVIEW note in hr2021.py."),
    ("other.hu",
     "Other religion (Hungary)",
     "KSH's `Más vallási közösséghez, felekezethez tartozó` — 29,977 people at settlement "
     "level, where it is the whole NON-CHRISTIAN remainder and nothing finer exists. At "
     "vármegye level the same census splits it into Muslim (7,983), Buddhist (11,042), "
     "Hindu (3,307) and a 7,645 remainder that keeps this node, and those three leave for "
     "islam / buddhism / hinduism. So unlike Croatia's `Istočne religije` this bucket is "
     "not opaque — it is measured coarsely and resolved by allocation (spec §3.10), and "
     "what stays here is only KSH's own residual."),
    ("other.mk",
     "Other religion (North Macedonia)",
     "SSO's `Друго` — 1,221 people, 0.07%, in a census whose named list is otherwise "
     "entirely Christian bodies plus Islam. Small because North Macedonia's religious "
     "map really is two large communities and a long thin tail, not because the question "
     "was coarse: the census names Jehovah's Witnesses and Evangelical Methodists "
     "separately at four figures and under."),
    ("other.ee",
     "Other religion (Estonia)",
     "Statistics Estonia's `Other religion` (8,100) plus `Religion unknown` (1,530) — the "
     "latter being people who DO feel an affiliation but whose religion was not recorded, "
     "which is the same kind of answer as Czechia's 'believer, church not named' and is "
     "filed the same way. Neither is `Refused to answer` or `Religious affiliation "
     "unknown`, which are universe residuals and off the tree entirely."),
    ("other.ro",
     "Other religion (Romania)",
     "INS's `Alta religie (asociatii religioase sau grupari religioase)` — 23,956 people "
     "in the religious associations and groupings that are not one of the 18 state-"
     "recognised cults. Romania's list of named categories IS the recognition list, so "
     "this residual is a legal artefact rather than a statistical one. Per source, per "
     "spec §3.11."),
    ("other.pl",
     "Other religion (Poland)",
     "GUS's own residual leaves — 'inne - niesklasyfikowane' (839) and 'własne "
     "(indywidualne) wierzenia religijne' (108), the people who wrote in a belief that "
     "belongs to no body. Per source, per spec §3.11."),
    ("other.in",
     "Other religion (India)",
     "Two different things, both small. First, the 149,668 people (1.9%) in `Other "
     "religions and persuasions` whose answer is not one of the 83 names the C-01 Appendix "
     "lists — the Appendix's floor is 100 people nationally, so this is the sub-100 tail. "
     "Second, a handful of named Appendix categories that are real religions with no home "
     "in the tree and too few adherents to earn a root: the Sant Nirankari Mission (1,781) "
     "and Dera Sacha Sauda (139), both Punjab movements of Sikh derivation whose members "
     "declined all six census religions. Adding a root for either would mean a 31st entry "
     "in ROOT_HSL, whose indigo→magenta wedge is already at its 4°-apart limit (§6.3), for "
     "groups that draw one dot. Per source, per spec §3.11."),

    # --- added 2026-09-03 with the §3.5a re-basing, for Pew rather than for ASARB.
    ("other.us",
     "Other religion (United States)",
     "Pew's two irreducible lines, and only Pew's — ASARB names every body it counts and has "
     "never needed a residual. First, what is left of `Other world religions` (776,032 "
     "adults) after the ASARB rolls of the roots it covers are subtracted; Sikhs, Daoists, "
     "Bahá'ís and Zoroastrians are one published line at n=36,908 and cannot be separated "
     "(spec §3.5a, §4.4). Second, `Other in the Unitarian and other liberal faiths family` "
     "(221,896), a write-in residual whose siblings run from Unitarian Universalism to "
     "humanism, so no root can claim it. Per source, per spec §3.11; see "
     "taxonomy/us_pew2024.py."),
]


# ---------------------------------------------------------------------------------------
# LINEAGE — the descent relation, at the coarsest useful grain (spec §2.1, §6.5)
# ---------------------------------------------------------------------------------------
"""
The second of §2.1's two relations, in the smallest form that is useful today. Not the full
`from` DAG with dates and edge kinds — that is still §10's job. This is one thing only: for
a node whose children are many, **the order they descend in**, cut into named groups.

Two problems it exists to solve, both of them real:

1. **Colour was allocated in size order.** The viewer sorted a parent's children biggest
   first and then walked the hue wheel down that list, so the two largest bodies always came
   out adjacent in hue — in the US, Catholic (62m) at red and Baptist (24m) at orange, the
   two you most need to tell apart, 18° from each other. Ordering by descent instead puts
   them 144° apart, and it does so for a reason rather than by luck: bodies are large
   because they are distinct traditions, so descent order tends to separate the big ones.

2. **The panel is meant to be a genealogy** (spec §10), and a list sorted by membership is
   the one order that hides descent completely.

The groups are the standard historical divisions, and the order inside each is roughly the
order of separation. Where a body has more than one parent — Baptists out of English
Separatism with Mennonite influence, Methodists out of Anglicanism by way of the Moravians —
it is placed on its main line and the other edge is a note here, because a linear order can
only carry one. That is the honest limit of a list, and it is why this is not the DAG.

The last group in each family is the one Anita named: bodies that do not descend from any
single line. A union of three traditions, a non-denominational congregation, and a source
answer that names no body at all are not a lineage and are not pretended to be one.

Anything not listed keeps size order and sorts after everything listed, so a new branch is
never silently reordered — `build_tree.py` names it instead.
"""

"""
Reader-facing notes for the legend — added 2026-09-04.

`note` above is for whoever maintains the tree, and nothing in the viewer has ever shown it.
This is the other kind: one or two sentences written FOR A READER, which the legend hangs on
the row as its tooltip.

It exists because the legend truncates. `.row .lb` is `nowrap` with `text-overflow: ellipsis`,
so a label long enough to carry a caveat is a label that gets cut off mid-caveat — the visible
text has to stay short, and the explanation needs somewhere else to live. Every legend row
already carries a `title`, which until now repeated the label it was truncating.

Only add a node here when the LABEL ALONE WOULD MISLEAD. Most do not need one: "Lutheran"
means Lutheran. The test is whether a reader who reads the label and nothing else comes away
believing something false.
"""

PUBLIC_NOTE = {
    "unrecorded":
        "Not a report of no religion — nobody was asked. The source is an administrative "
        "register that records only the churches entitled to church tax, so this category "
        "holds Muslims, Orthodox Christians, Jews and free-church members alongside everyone "
        "who belongs to nothing, and cannot tell them apart.",
}

LINEAGE = {
    "christianity": [
        ("Ancient communions", [
            # Separated by the councils and the schism, not the Reformation: Ephesus 431,
            # Chalcedon 451, and 1054. Ordered by the date they parted.
            "christianity.churchofeast",
            "christianity.oriental",
            "christianity.orthodox",
            "christianity.catholic",
        ]),
        ("Reformation", [
            # Hussite first: 1415 and a century early. The Czechoslovak Hussite Church is a
            # 1920 body claiming that reformation, so it sits at the head of this line
            # rather than inside the ancient communions it left.
            "christianity.hussite",
            "christianity.lutheran",
            "christianity.reformed",
            "christianity.anglican",
        ]),
        ("Separatist and believers' churches", [
            # The radical wing: baptism on profession, and a church separate from the state.
            "christianity.anabaptist",
            "christianity.baptist",     # English Separatists, with Dutch Mennonite contact
            "christianity.friends",
            "christianity.plymouth",
        ]),
        ("Pietist and Wesleyan revival", [
            # A single chain, and worth reading as one: Pietism renews the Moravians, the
            # Moravians convert Wesley, Methodism throws off Holiness, Holiness throws off
            # Pentecostalism. Methodism's other parent is Anglican, in the group above.
            "christianity.pietist",
            "christianity.moravian",
            "christianity.methodist",
            "christianity.holiness",
            "christianity.pentecostal",
        ]),
        ("Restorationist and adventist", [
            # 19th-century America and the claim to restore the apostolic church or to read
            # the end of the age. Jehovah's Witnesses come out of the Millerite adventists,
            # which is why they follow them here.
            "christianity.restorationist",
            "christianity.latterday",
            "christianity.adventist",
            # Bible Students before Witnesses: Russell's movement is the parent and the
            # Witnesses are the 1917 split out of it, so this is descent order.
            "christianity.biblestudent",
            "christianity.witnesses",
            "christianity.swedenborgian",
            "christianity.christianscience",
        ]),
        ("No single line", [
            # Bodies that are a union of several traditions, congregations that decline the
            # question, churches founded outside the missionary denominations, and the
            # source answers that name no body. See the note on `christianity.protestant`:
            # there is no Protestant super-node, and this group is not one either.
            "christianity.united",
            "christianity.nondenominational",
            "christianity.maori",
            "christianity.messianic",
            "christianity.protestant",
            "christianity.other",
        ]),
    ],

    # Ordered by the date the movement separated, which for Judaism is also the order the
    # names are usually given in. Chabad is a Hasidic court inside Orthodoxy rather than a
    # movement beside it, and is placed with Orthodox for that reason.
    "judaism": [
        ("Traditional", ["judaism.orthodox", "judaism.chabad", "judaism.conservative"]),
        ("Liberal", ["judaism.reform", "judaism.reconstructionist"]),
        ("No single line", ["judaism.independent"]),
    ],

    # The three vehicles, oldest first. Vajrayana is carried inside Mahayana historically and
    # is counted beside it, which is the §2.1 split showing up in one line.
    "buddhism": [
        ("Vehicles", ["buddhism.theravada", "buddhism.mahayana", "buddhism.vajrayana"]),
    ],
}
