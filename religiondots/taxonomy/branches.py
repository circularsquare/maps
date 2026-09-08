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
    # --- added 2026-09-08 with Tonga. FOUR SIBLING CHURCHES, ALL FOUR A CENSUS CELL, and
    #     together 53.4% of the country. Grouped rather than filed flat under
    #     `christianity.methodist` so that four Tonga-only children do not sit permanently
    #     in the Methodist legend of every other country.
    ("christianity.methodist.tongan",
     "Tongan Methodist churches",
     "**Tonga is 53.4% Methodist and the census counts that as four separate churches.** "
     "The Wesleyan mission arrived in 1826 and converted the kingdom; every division since "
     "has been about who controls the church rather than about doctrine, and each one left "
     "a body that is still here two centuries later. Nothing else on this map divides a "
     "single Protestant tradition into four countable national churches, which is why they "
     "are grouped here instead of collapsing onto `christianity.methodist`: on the branch "
     "they would draw as one colour over half the country and Tonga's actual religious "
     "geography would disappear.\n\n"
     "The line runs: the **Free Church of Tonga** was created in 1885 by King George Tupou "
     "I and his adviser Shirley Baker, to be free of the Wesleyan conference in Australia; "
     "**Queen Sālote Tupou III reunited most of it with the Wesleyan remnant in 1924** as "
     "the Free Wesleyan Church, and those who refused the union carried the Free Church "
     "name on; a further separation in the later 1920s produced the **Church of Tonga**, "
     "and the **Constitutional Church of Tonga** is of the same family. Accounts of the "
     "1920s disagree about which body is the continuation of which, so what is asserted "
     "here is only what they agree on: one mission, one 1885 break, one 1924 reunion that "
     "did not hold, and four churches now.\n\n"
     "Not a home for the later revival breakaways. The Tokaikolo Christian Church (1978) "
     "and Mo'ui Fo'ou 'ia Kalaisi left the Free Wesleyan Church but are charismatic in "
     "practice, and §2.1's containment is a fact about people now — see the REVIEW notes "
     "in `taxonomy/to2021.py`."),
    ("christianity.methodist.tongan.freewesleyan",
     "Free Wesleyan Church of Tonga",
     "**33,953 people, 34.2%, the largest church in Tonga and the church of the monarchy** "
     "(Siasi Uēsiliana Tau'atāina 'o Tonga). Formed by Queen Sālote Tupou III's 1924 "
     "reunion of the Free Church with the Wesleyan Methodist remnant.\n\n"
     "**Its geography is that it has none, which is the point.** It is 34.1% of Tongatapu, "
     "34.6% of Vava'u, 33.1% of Ha'apai, 36.3% of 'Eua and 30.7% of the Niuas — a spread of "
     "six points across a country whose other churches swing by thirty. Every other body "
     "here has a stronghold; the national church is the one that is the same everywhere."),
    ("christianity.methodist.tongan.free",
     "Free Church of Tonga",
     "**11,244 people, 11.3%** (Siasi 'o Tonga Tau'atāina). Carries the name of the church "
     "King George Tupou I founded in 1885, kept by the part that refused the 1924 reunion. "
     "Strongest in Vava'u (16.0%) and on 'Eua (17.6%) against 10.0% on Tongatapu, and it is "
     "**51.3% of Ha'atafu and 48.4% of Tufuvai**."),
    ("christianity.methodist.tongan.tonga",
     "Church of Tonga",
     "**6,782 people, 6.8%** (Siasi 'o Tonga), out of a further separation in the later "
     "1920s. **It is the Ha'apai church**: 20.1% of that division against 6.8% nationally, "
     "**37.1% of Lulunga district and 29.8% of Ha'ano**, reaching 44.3% of Ha'afeva. Those "
     "are the small outer islands between Tongatapu and Vava'u, and no other body on this "
     "map is concentrated there."),
    ("christianity.methodist.tongan.constitutional",
     "Constitutional Church of Tonga",
     "**1,152 people, 1.2%** (Siasi Konisitūtone Tau'atāina 'o Tonga), the smallest of the "
     "four and of the same 1920s family. Thinly spread, with local pockets: 21.8% of Hunga "
     "in Vava'u and 16.2% of Faleloa in Ha'apai."),

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

    # --- added 2026-09-08 with Angola.
    ("christianity.newapostolic",
     "New Apostolic Church",
     "The Irvingite line: the Catholic Apostolic Church of the 1830s, reorganised at "
     "Hamburg in 1863 around a living apostolate that the older body would not renew. It "
     "is neither Pentecostal, evangelical nor Stone-Campbell restorationist, and until "
     "Angola no source here counted enough of it to be worth its own node. "
     "**Angola counts 515,929, 1.50% of the country** -- and its geography is the "
     "argument for the node as much as its size. The peaks are Moxico Leste 8.0%, Moxico "
     "5.4% and Cuando 4.9%, running to 23.8% in Ninda and 17.2% in Lumbala Nguimbo: the "
     "Zambian border, and Zambia is one of the church's largest countries anywhere. A "
     "church that had arrived through Lisbon would sit in Luanda; this one came overland "
     "from the east. "
     "**Two counted peers are elsewhere on the tree and were deliberately not moved.** "
     "`ee2021.py` files Estonia's New Apostolic Church on `christianity.restorationist`, "
     "which is the wrong restorationism, and `au2021.py` files Australia's on "
     "`christianity.other`; both predate this node. Each is a one-line change plus a "
     "re-scatter of that country, and doing it while adding Angola would alter two "
     "countries nobody asked about -- the same restraint `christianity.africaninstituted."
     "harrist` showed towards Benin."),

    ("christianity.adventist", "Adventist", ""),
    # --- added 2026-09-08 with Samoa. The smallest kind of thing this map exists to show:
    #     a church that one census names and no other census on earth does.
    ("christianity.adventist.sisdac",
     "Samoa Independent Seventh Day Adventist Church",
     "`Aso Fitu` — Samoan for *the seventh day* — is SBS's own column for the **Samoa "
     "Independent Seventh Day Adventist Church, 1,962 people, 0.95% of Samoa**. It broke "
     "from the Seventh-day Adventist Church in Samoa and kept the Adventist sabbath and "
     "doctrine, which is why it sits under `christianity.adventist` rather than beside it.\n\n"
     "**It is counted here because Samoa counts it, and nowhere else because nobody else "
     "does.** No other census on this map, and no census the UNSD Yearbook holds for any "
     "other country, has a cell for it. Merging it into `christianity.adventist` would lose "
     "the only place it is visible, and the precedent for a node this size is Ekalesia Niue "
     "at 981 people. It is 11.7% of Safune on Savai'i and 20.1% of Eva, against 3.0% of the "
     "largest district, so it is a village-scale body: at the 25-district grain Samoa is "
     "drawn on it will not show a geography of its own."),
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
     "The Scandinavian free-church line — Evangelical Covenant, Evangelical Free. **Widened "
     "2026-09-06 with Switzerland**, whose census counts *Neupietistisch-evangelikale "
     "Gemeinden* as a category of its own: the Freie Evangelische Gemeinden, the Chrischona "
     "congregations and the Darbysts of the Bernese Jura are the same pietist free-church "
     "descent on the continent rather than in Scandinavia, and BFS groups them under a name "
     "that says so. Kept distinct from `christianity.pentecostal`, which BFS counts "
     "separately, and from `christianity.evangelical`, which holds an answer rather than a "
     "family."),
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
    # --- added 2026-09-05 with Lithuania, the first source anywhere to count them.
    ("judaism.karaite",
     "Karaite Judaism",
     "Jews who hold the written Torah alone and not the rabbinic oral law, separated in the "
     "eighth or ninth century — a thousand years before any of the movements beside it here, "
     "which is why LINEAGE puts it first. Lithuania's census offers `Karaimų` as its own "
     "answer and 255 people took it: the Karaims of Trakai, a Turkic-speaking community "
     "brought from Crimea in 1397 and one of the smallest recognised religious communities "
     "in Europe. **Many Lithuanian Karaims describe themselves as a distinct people and not "
     "as Jews**, and filing them under Judaism is a claim about the religion rather than "
     "about the community — see the REVIEW note in lt2021.py, which is where to argue with "
     "it. No other source on this map separates them from `judaism`."),

    # --- added 2026-09-07 with Israel. THESE FOUR ARE AN OBSERVANCE AXIS AND NOT MOVEMENTS,
    #     which is the one thing to understand before using or extending them.
    #
    #     Every other child of `judaism` here is a movement you join: Orthodox, Conservative,
    #     Reform, Reconstructionist, Chabad. These four are the answer to a different
    #     question — how observant is this household — which is the question Israel actually
    #     asks and the movements are the question it does not. An Israeli Hiloni Jew has not
    #     left Orthodoxy for a liberal movement; the liberal movements are marginal in Israel
    #     and the axis simply runs elsewhere. So they are grouped separately in LINEAGE
    #     rather than being slotted in beside Orthodox, and a country must use one axis or
    #     the other and never both.
    #
    #     THEY ARE ALSO NOT A PARTITION OF JUDAISM IN GENERAL. CBS asks the lifestyle
    #     question of the whole population, not of Jews — Umm al-Fahm is 99.8% Muslim and
    #     returns 47% Traditional — so `sources/il.py` applies the split ONLY to units that
    #     are at least 85% Jewish and leaves everyone else on the parent `judaism`. Those
    #     rows are `derived` in §7's sense, not `modelled`: the people are counted, and
    #     counted as Jews — only the branch is inferred, from two tables CBS published per
    #     unit. See il.py's docstring for why that threshold and why the tier.
    ("judaism.haredi",
     "Haredi",
     "Ultra-Orthodox. The one category here that is unambiguous across sources and the one "
     "the census measures best: Bene Beraq returns 83.8% and Israel as a whole 6.4%. Note "
     "that Haredi is Orthodox in doctrine — the reason it is not filed under "
     "`judaism.orthodox` is that this whole group answers a different question (see above), "
     "not that anyone disputes the theology."),
    ("judaism.dati",
     "Religious (dati)",
     "Observant and largely, though not only, the Religious Zionist stream. CBS's own "
     "English is 'Religious / Very religious', collapsing two answers into one cell. In "
     "movement terms most of these people are Orthodox too, which is again why the axis is "
     "kept separate rather than merged into it."),
    ("judaism.masorti",
     "Traditional (masorti)",
     "**A FALSE FRIEND, AND THE MOST LIKELY THING ON THIS BRANCH TO BE GOT WRONG.** In "
     "Israel *masorti* means traditional-but-not-strictly-observant — keeping some practice, "
     "not committed to full halakhic observance — and it is the largest single answer among "
     "Mizrahi Jews. Everywhere ELSE in the Jewish world 'Masorti' is the name of the "
     "Conservative movement, which is `judaism.conservative` and is a different object with "
     "a different history. Mapping an Israeli masorti row to `judaism.conservative` would be "
     "wrong in the same way §12's `animismus` example is wrong: the string is shared and the "
     "place decides the meaning."),
    ("judaism.hiloni",
     "Secular (hiloni)",
     "Israel's largest Jewish answer, 53.2% nationally. **This is not irreligion in the "
     "sense the grey family means** (§6.3a), and the difference is a fact about the source "
     "rather than a judgement: Israel's population register has no irreligion box at all, so "
     "a secular Israeli Jew is registered as a Jew and appears on this map inside Judaism "
     "however they describe themselves. Drawing them grey would assert something the "
     "register never asked. What the label carries is the household's observance, which is "
     "the only thing anyone measured."),

    ("islam", "Islam", ""),

    # --- added 2026-09-05 with Russia, and they are the first two. Twenty countries in,
    #     `islam` had no children at all: Australia's 148 categories, the Philippines' 129
    #     and ASARB's 372 all name Islam once and stop, and the UK's write-in tail reaches
    #     Alevism without ever reaching Sunni. Sreda's Arena is the first source on this map
    #     that ASKS the question — it offers Sunni, Shia and "Muslim, neither" as three
    #     separate answers — which is why the nodes arrive now rather than earlier.
    ("islam.sunni",
     "Sunni Islam",
     "The larger branch, roughly nine tenths of Muslims worldwide, and almost never "
     "enumerated: a census that asks about religion at all normally stops at 'Muslim'. "
     "2.50M in Russia, and 48.6% of Dagestan."),
    ("islam.shia",
     "Shia Islam",
     "313,477 in Russia, concentrated where the tradition actually is — Dagestan's 2.0% is "
     "the Azeri community around Derbent, which has been Shia since the Safavids. Note that "
     "a source offering Sunni and Shia as answers does NOT thereby measure them well: most "
     "Russian Muslims chose neither, and 4.66% of the country — nearly twice the Sunni "
     "figure — answered 'I profess Islam, but am neither Sunni nor Shia', which stays on "
     "the parent `islam`."),
    # --- added 2026-09-06 with Pakistan.
    ("islam.ahmadiyya",
     "Ahmadiyya",
     "The movement founded by Mirza Ghulam Ahmad in Qadian, 1889. **Placed under `islam` "
     "deliberately, and the placement is the whole content of the node.** Ahmadis identify "
     "as Muslim; Pakistan's constitution declares them non-Muslim, its penal code makes it "
     "a criminal offence for them to say otherwise, and its census accordingly prints "
     "`Qadiani/Ahmadi` as a PEER of `Muslim` rather than a subdivision of it. spec §2 says "
     "this tree is containment as it holds for people now, not as a state's law defines it, "
     "so filing them outside Islam would be adopting Pakistan's legal classification as "
     "this map's taxonomy. It is not neutral to do that. "
     "**The count is a floor by a large and unknown margin**, for reasons that are specific "
     "and documented rather than general: registering as Ahmadi has legal consequences — a "
     "separate electoral roll, and a declaration disavowing the movement's founder required "
     "to obtain a passport — and the community has organised census boycotts on exactly that "
     "ground since 1974. 191,737 in Pakistan 2017 and 0.09% of the country; every "
     "independent estimate is several times that. Read it the way §11b says to read an "
     "African `Traditionalist` cell. "
     "`Qadiani` is the state's term and is used by others pejoratively; it is kept in "
     "`source_category` because that is what the census prints (§2.4) and is not used as a "
     "label here."),

    # --- the four Sunni schools and the Ja'fari, added 2026-09-08 with Türkiye, which is
    #     the first source on this map to enumerate a madhhab at all (sources.md §11ac).
    #     `islam.sunni`'s own note said the school is "almost never enumerated: a census
    #     that asks about religion at all normally stops at 'Muslim'", and it stayed true
    #     for eighty-one countries. The Diyanet's Q11 asks it outright.
    ("islam.sunni.hanafi",
     "Hanafi",
     "The oldest and largest of the four Sunni schools of law, from Abu Hanifa in 8th-century "
     "Kufa, and the school the Ottoman state ran on. **77.5% of Muslims in Türkiye** and the "
     "plurality in eleven of its twelve statistical regions, running above 90% in Doğu "
     "Karadeniz and Orta Anadolu. Where it thins is the southeast, and what replaces it there "
     "is Shafi'i rather than anything non-Sunni."),
    ("islam.sunni.shafii",
     "Shafi'i",
     "From al-Shafi'i, d. 820. **11.1% of Muslims in Türkiye, and the country's sharpest "
     "religious geography by a distance**: 48.7% in Ortadoğu Anadolu, which is the one region "
     "of the country where it leads Hanafi, 42.0% in Güneydoğu Anadolu and 35.2% in Kuzeydoğu "
     "Anadolu, against 0.2% in Doğu Karadeniz and 1.0% in Batı Karadeniz. That is the Kurdish "
     "southeast, and it is drawn here from the state's own survey rather than inferred from "
     "who lives there, which is the difference §14.5 turns on."),
    ("islam.sunni.maliki",
     "Maliki",
     "From Malik ibn Anas, d. 795; the school of North and West Africa and the Gulf. **0.03% "
     "of Muslims in Türkiye**, which is the whole of what any source on this map has ever "
     "measured about it. Drawn because the column is real and published, not because the "
     "number is large."),
    ("islam.sunni.hanbali",
     "Hanbali",
     "From Ahmad ibn Hanbal, d. 855; the smallest of the four and the school of the Arabian "
     "peninsula. **0.1% of Muslims in Türkiye.** As with `maliki`, a real published column "
     "and a very small one."),
    ("islam.shia.jaafari",
     "Ja'fari (Twelver)",
     "The school of the Twelver Shia, named for Ja'far al-Sadiq, d. 765. **1.0% of Muslims in "
     "Türkiye, and 4.6% in Kuzeydoğu Anadolu** — Iğdır and Kars, on the Azerbaijani and "
     "Iranian border, which is where Turkey's Azeri Shia are and has been since the Safavids. "
     "The same shape Russia's Shia figure has in Dagestan (`islam.shia`). "
     "**It is not a count of Alevis and must not be read as one.** The Turkish questionnaire "
     "offers Caferi as one of its options and offers no Alevi option at all; Alevis are not in "
     "this node, and the country's note says where they are instead."),

    ("hinduism", "Hinduism", ""),
    # --- added 2026-09-06 with Vietnam, which counts it as one of its sixteen recognised
    #     religions under its own name.
    # --- added 2026-09-07 with Mauritius, which is the first source anywhere on this map to
    #     divide Hinduism at all. India's census does not; Guyana's does not; before this the
    #     family had one child and it was Vietnam's Cham Balamon.
    #
    #     THREE OF THE FOUR ARE COMMUNITIES AND THE FOURTH IS A MOVEMENT, and the difference
    #     matters (spec §2.1's two relations). Marathi, Tamil and Telugu Hindus in Mauritius
    #     are the descendants of indentured labourers from three different parts of India who
    #     have kept separate temples, priesthoods, languages and festival calendars for a
    #     century and a half; Arya Samaj is a doctrinal reform movement that anyone can join.
    #     Statistics Mauritius counts all four as answers to its religion question, which is
    #     why they are religion nodes here rather than an ethnicity read as religion (§14.5).
    ("hinduism.tamil",
     "Tamil Hindu",
     "**63,950 in Mauritius, 5.2% of the country** — the second largest Hindu community "
     "there and the only place on this map that counts it. Shaivite and Murugan-centred, "
     "with its own kovils and its own priesthood, and the Cavadee at Thai Poosam is a public "
     "holiday. Mauritian Tamils arrived partly before the indenture system and are "
     "concentrated in Port Louis and the Plaines Wilhems towns rather than in the cane "
     "districts. Nothing here separates Tamil Hindus anywhere else — India's census asks "
     "religion and language in different tables and crosses them at no useful geography."),
    ("hinduism.telugu",
     "Telugu Hindu",
     "**25,216 in Mauritius, 2.0%.** Andhra-origin, arrived under indenture, and organised "
     "since 1947 around the Andhra Maha Sabha; Ugadi is its new year and is separately "
     "recognised. Kept apart from `hinduism.tamil` because Mauritius keeps them apart, and "
     "because the two communities' temples, languages and festivals are different — reading "
     "them as one 'South Indian' block is the error this node exists to avoid."),
    ("hinduism.marathi",
     "Marathi Hindu",
     "**19,052 in Mauritius, 1.5%.** Maharashtrian-origin, organised around the Mauritius "
     "Marathi Mandali Federation, with Ganesh Chaturthi as its principal festival — the "
     "immersion at Grand Baie and along the north coast is the largest Marathi public "
     "observance outside India."),
    ("hinduism.aryasamaj",
     "Arya Samaj",
     "**7,422 in Mauritius, 0.6%**, where the census calls it `Vedic/Hindu Vedic & Aryan`. "
     "**The one of Mauritius's four Hindu cells that is a MOVEMENT rather than a "
     "community**: Dayananda Saraswati's 1875 reform, holding the Vedas alone as authority "
     "and rejecting image worship, caste birth-right and priestly mediation. It reached "
     "Mauritius in 1910 and the split with the Sanatanist majority was bitter enough to run "
     "through Mauritian politics for decades. It is doctrinal, so it sits beside the three "
     "community nodes rather than inside any of them, and a Bhojpuri-speaking Arya Samajist "
     "is in this node and not in the parent."),
    ("hinduism.chambalamon",
     "Cham Balamon",
     "Chăm Bà la môn — the Hindu religion of the Cham of Ninh Thuận and Bình Thuận, 56,427 "
     "in 2009 and 64,547 in 2019, and **the last living Hindu tradition of Southeast Asia's "
     "Indianised kingdoms**. Descended from the Shaivism of Champa, and a thousand years "
     "separated from Indian practice: its Basaih priesthood is hereditary, its Kate festival "
     "follows the Cham calendar, and it has absorbed the local goddess Po Nagar. Filed under "
     "Hinduism because that is what it descends from and what Vietnam's own category calls "
     "it (`Bà La Môn` = Brahmanism), and given a node rather than the parent because reading "
     "it as Indian Hinduism would be wrong about almost everything except the ancestry. "
     "**Distinct from Cham Bani**, the syncretic Cham Islam of the same two provinces, which "
     "Vietnam's census counts inside `Hồi giáo` and which nothing here can separate out — "
     "see taxonomy/vn2009.py."),
    ("buddhism", "Buddhism", ""),
    # Authored 2026-09-05 with China. The three vehicles existed only as ASARB leaves, so
    # they carried ASARB's own labels — including its misspelling, `Vajarayana Buddhist`,
    # which is fine as a SOURCE CATEGORY (spec §3.1's note on group code 892) and wrong as
    # a node name. Nobody noticed while the node held a few thousand American adherents;
    # China puts 6.3M Tibetan Buddhists on it, and a source's typo should not be the label
    # on the second-largest thing drawn in the country.
    ("buddhism.theravada",
     "Theravada Buddhism",
     "The southern transmission — Sri Lanka, Myanmar, Thailand, Laos, Cambodia, and in "
     "China the Dai of Xishuangbanna and Dehong, who are the tradition's northern edge "
     "rather than an outpost of Chinese Buddhism. Note that Sri Lanka is NOT drawn here: "
     "its census offers one cell labelled `Buddhist` and lk2024.py keeps those 15.2M "
     "people on the parent, because the school is knowable and not stated."),
    ("buddhism.mahayana",
     "Mahayana Buddhism",
     "The northern transmission — China, Korea, Japan, Vietnam. **This node was nearly empty "
     "until 2026-09-08 and is now one of the largest on the map**, because China's Han "
     "Buddhists arrived: about 55 million people, carved out of China's grey at province "
     "level from three pooled waves of the Chinese General Social Survey, which asks *which "
     "religion do you belong to* and gets 佛教 from roughly one adult in twenty (spec "
     "§14.16). They are heaviest in Zhejiang, Fujian and Jiangxi. "
     "**It is still an underestimate and by a wide margin, because the question is the "
     "narrow one.** Ask instead about belief in Buddha or a bodhisattva and the same "
     "instrument family returns 33% rather than 4%; spec §3.1 requires one basis and this "
     "map is drawn on self-identification everywhere, so the people who keep the practice "
     "without claiming the label are in China's `unknown` grey. Note also that most censuses "
     "which ask about Buddhism at all ask it as one word, so many of the world's Mahayana "
     "Buddhists sit on the parent node rather than here — Sri Lanka's are Theravada and "
     "still on the parent for the same reason."),
    ("buddhism.won",
     "Won Buddhism",
     "원불교, 84,141 in South Korea in 2015 — founded 1916 by Sotaesan as a reform of "
     "Buddhism for the modern world: no temple statuary, a single Irwonsang circle in place "
     "of images, laity and clergy on the same footing, women ordained from the start. "
     "**It is filed under Buddhism because that is what it says it is** — the movement's own "
     "self-description is Buddhist reform, and spec §2.1's relation is containment, so the "
     "claim its adherents make about themselves is the one that decides. Scholars routinely "
     "class it instead with the Korean new religions, and that reading is not wrong; if the "
     "tree ever needs to follow it, the node is `eastasiannew.korean.won` and nothing else "
     "changes. It sits beside the three vehicles rather than inside Mahayana because it is a "
     "20th-century founding, not a transmission."),
    ("buddhism.vajrayana",
     "Vajrayana Buddhism",
     "Tibetan Buddhism and its Himalayan and Mongolian extensions. Counted inside Mahayana "
     "historically and beside it here, which is spec §2.1's two relations showing up in one "
     "node. **China is essentially all of it** — 6.3M, derived from Tibetan, Yugur, Monba "
     "and Pumi nationality (taxonomy/cn2000.py), covering the plateau, western Sichuan, "
     "Qinghai and southern Gansu. Mongols and Tu are NOT in it: see cn2000.py's REVIEW."),

    # --- added 2026-09-06 with Vietnam. Five bodies that all say they are Buddhist, all
    #     founded in the Mekong Delta or Saigon between 1849 and 1939, and all counted
    #     separately by Vietnam's census. **They are siblings here and not a chain**, because
    #     spec §2.1's containment relation is about people now: a Hòa Hảo follower is not a
    #     Bửu Sơn Kỳ Hương follower. Four of them ARE a single documented descent, and that
    #     belongs in LINEAGE at the bottom of this file, which is exactly the split §2.1
    #     draws and the first time a source has supplied a whole lineage at once.
    ("buddhism.hoahao",
     "Hòa Hảo Buddhism",
     "Phật giáo Hòa Hảo — 1,433,252 in 2009 and Vietnam's third largest religion. Founded "
     "1939 at Hòa Hảo village in An Giang by Huỳnh Phú Sổ, and the last and much the largest "
     "of the Bửu Sơn Kỳ Hương line: lay practice at home rather than in temples, no images, "
     "no ordained clergy, and the Four Debts of Gratitude in place of monastic vows. It is "
     "**intensely local even by this map's standards** — 936,974 of it, 65%, is in An Giang "
     "alone, where it is 44% of the province. Filed under Buddhism because that is what it "
     "calls itself, which is `buddhism.won`'s reason."),
    ("buddhism.tuan",
     "Tứ Ân Hiếu Nghĩa",
     "Đạo Tứ Ân Hiếu Nghĩa, 41,280 in 2009 — 'the Four Debts and Filial Righteousness', "
     "founded 1867 by Ngô Lợi in the Bảy Núi hills of An Giang out of the Bửu Sơn Kỳ Hương "
     "teaching, and organised as much as an anti-colonial settlement movement as a religion. "
     "Almost all of it is still in An Giang."),
    ("buddhism.buuson",
     "Bửu Sơn Kỳ Hương",
     "10,824 in 2009 — 'Strange Fragrance from the Precious Mountain', founded 1849 by Đoàn "
     "Minh Huyên during a cholera epidemic in the Mekong Delta, and **the root of the line "
     "that produced Tứ Ân Hiếu Nghĩa and Hòa Hảo**. What survives under the name is the "
     "small remnant that followed neither, so the node is the body and not the lineage; the "
     "lineage is in LINEAGE below."),
    ("buddhism.talon",
     "Hiếu Nghĩa Tà Lơn",
     "Phật giáo Hiếu Nghĩa Tà Lơn, 401 people — recognised by Vietnam in 2010 and therefore "
     "**counted in the 2019 census and not the 2009 one**, so it has a national figure and "
     "no geography anywhere. Named for Tà Lơn (Bokor mountain, in Cambodia), where its "
     "founder is held to have received the teaching; the last of the Bửu Sơn Kỳ Hương line "
     "to be registered."),
    ("buddhism.tinhdo",
     # Named in Vietnamese like the four beside it, and not "Pure Land Buddhist Home-practice
     # Association", which was the first label and is 44 characters: `.row .lb` is nowrap with
     # ellipsis, so it shipped as "Pure Land Buddhist Home-pra…" in the legend (§6.3a's rule
     # about labels that get cut off). The English is in the note, which the tooltip carries.
     "Tịnh Độ Cư Sĩ",
     "Tịnh Độ Cư Sĩ Phật Hội Việt Nam — the Pure Land Buddhist Home-practice Association of "
     "Vietnam, 11,093 in 2009 — founded 1934 in Sa Đéc by Minh Trí "
     "as a lay Pure Land association, and unusual in running free herbal-medicine clinics as "
     "its central practice rather than as charity beside it. **Not part of the Bửu Sơn Kỳ "
     "Hương line** — it is a Pure Land body and so Mahayana by descent — and it is a node of "
     "its own rather than sitting on `buddhism.mahayana`, because that node is the generic "
     "answer and would then be labelled, in the legend, by the only body that ever landed "
     "on it."),
    ("sikhism", "Sikhism", ""),
    ("jainism", "Jainism", ""),
    ("zoroastrianism", "Zoroastrianism", ""),
    ("bahai", "Bahá'í", ""),

    # Added 2026-09-07 with Nepal, which is the only source on this map that counts it.
    ("bon",
     "Bon",
     "Yungdrung Bon, the religion of the Tibetan plateau that predates Buddhism's arrival "
     "there in the seventh century, with its own canon, monasteries, lineages and abbots. "
     "**A root of its own and deliberately not a child of `buddhism`.** Modern Bon and "
     "Tibetan Buddhism have converged far enough that Bon is sometimes counted as a fifth "
     "Tibetan school, and the Dalai Lama has recognised it alongside the four — but Bonpos "
     "do not describe themselves as Buddhists, the two trace to different founders "
     "(Tonpa Shenrab against the Buddha), and NSO Nepal puts `Bon` in a cell beside "
     "`Bouddha` rather than inside it. Filing it under Buddhism would contradict the only "
     "source that counts it. "
     "**67,223 people in Nepal, and the only Bon this map DRAWS.** India's census does name "
     "it — 697 people — but only as a C-01 Annexure write-in under `Buddhist`, and "
     "`in2011.py` excludes the whole Annexure because each of its sects undercounts the "
     "real community by one to three orders of magnitude (sources/in.md §4). So India's "
     "Bon are recorded in `in.csv` and drawn nowhere, and Nepal is the only source here "
     "that puts Bon on the map at all. Tibet is the population that would really fill this "
     "node, and China's source cannot see religion at all (§14.6); Bhutan is not drawn. "
     "**Its geography is NOT the Tibetan border, which is the thing worth knowing about "
     "this cell.** The expectation is Mustang, Dolpa and Humla — the trans-Himalayan "
     "districts with Yungdrung Bon monasteries — and the census says otherwise: **Gandaki "
     "province holds 2.17% against 0.09% in Bagmati**, and the top districts are Manang "
     "(6.1%), Gorkha (5.7%), Lamjung (4.7%), Syangja (3.3%) and Kaski (2.6%). Those are "
     "the Gurung (Tamu) middle hills, not the Tibetan plateau. The likeliest reading is "
     "that most of the cell is the **Tamu shamanic tradition** — the *pye-ta lhu-ta*, "
     "served by *pachyu* and *klepri* priests — which Gurung communities describe as Bon "
     "and which is related to, but not the same institution as, the monastic Yungdrung Bon "
     "of Dolpa. Shey Phoksundo in Dolpa, which is the Yungdrung heartland, is 12.4% and "
     "is a much smaller number of people. **The census offers one box and this map cannot "
     "separate the two**; the note is here so the cluster over Gorkha is not misread as an "
     "error."),
    ("shinto", "Shinto", ""),
    ("daoism", "Daoism", ""),
    ("confucianism",
     "Confucianism",
     "Added 2026-09-05 with South Korea, which is **the only source on this map that counts "
     "it as a religion** — 유교, 75,703 people in 2015, with a cell of its own beside "
     "Buddhism and the two Christianities. Everywhere else it is either not asked or folded "
     "into a folk-religion residual: `chinesefolk`'s note names it as one of the things that "
     "category carries, and China's census asks nothing at all. "
     "A root rather than a child of anything, on Anita's call 2026-09-05, because there is "
     "nothing on the tree for it to descend from — the alternative was `chinesefolk`, which "
     "would be wrong twice over for a Korean answer. "
     "**What the number means is narrower than the word.** Confucian practice — ancestral "
     "rites, the lineage hall — is near-universal in Korea and is not what these 75,703 "
     "people are reporting; they are reporting Confucianism as their RELIGION, which is a "
     "much smaller and more deliberate claim, and most of them are the membership of "
     "Seonggyungwan and the local hyanggyo. Read it as the institutional core, not the "
     "practice."),
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

    # ---- added 2026-09-05 with Russia.
    ("indigenous.northeurasian",
     "North Eurasian traditional religions",
     "One Arena answer — 'I practice the traditional religion of my ancestors, worship the "
     "gods and the forces of nature' — and 1.76M people. Regional child of `indigenous`, on "
     "the pattern of `indigenous.african` and `indigenous.indian`. "
     "**It holds two different populations and the smaller one is the famous one.** "
     "Concentrated, in the ethnic republics: North Ossetia 29.4% (Uatsdin, the Ossetian "
     "faith), Sakha 13.0%, the Altai Republic 13.0%, Karachay-Cherkessia 12.2%, Tuva 8.0%, "
     "Mari El 5.6% — the Mari native religion being one of the last indigenous religions of "
     "Europe with a continuous practice. Those are real, distinct traditions with nothing in "
     "common but their position, and no Russian source separates them. "
     "**But that is only 27% of the node, and the correction matters.** This entry first "
     "claimed the category was 'the surviving ethnic religions of Russia's non-Russian "
     "peoples' with Rodnovery 'a small part of the total'. Measured against the 2021 census: "
     "the share of peoples who HAVE such a tradition explains only **R² = 0.27** of the "
     "answer across the 79 measured subjects — against R² = 0.96 for ethnicity and the Islam "
     "answer — and **73% of the people giving it live in subjects where those peoples are "
     "under 10% of the population**, at a median 0.69%. So most of this node is ethnic "
     "Russians answering a question whose wording ('worship the gods and the forces of "
     "nature') invites a broad reading: Rodnovery, the Slavic neo-pagan reconstruction, "
     "which belongs nearer `paganism`, plus a general nature-religion answer. "
     "The geographic label is therefore doing real work — it covers both halves without "
     "asserting either, which naming a people would not."),

    # ---- added 2026-09-05 with Ghana, the first African source on the map.
    ("indigenous.african",
     "African traditional religions",
     "The traditional religions of sub-Saharan Africa — Akan, Ewe, Ga and Dagomba practice "
     "in Ghana, and their neighbours across the region. A regional child of `indigenous` "
     "on the pattern already set by India, the Philippines, Aotearoa and Australia, and "
     "not a family of its own: §12 says a new top-level root costs the whole palette a "
     "degree, and nothing here needs one. 999,319 people in Ghana, 3.25%. "
     "**The count is a floor and the reason is the same everywhere in Africa.** Ghana's "
     "form offers `Traditionalist` as one box against four Christian ones, and traditional "
     "practice very often coexists with a Christian or Muslim affiliation rather than "
     "replacing it — so the people who would answer both are counted in the other column. "
     "**Angola is the extreme case, and it shows where the missing people went.** INE's "
     "`Animista` is 44,370, **0.13% of the country** — five times below Kenya, the next "
     "lowest of the seven African sources here, and twenty-five times below Ghana. Beside "
     "it Angola's `Sem religião` is 12.10%, the second highest on the continent after Côte "
     "d'Ivoire, and **its geography is pastoralist rather than urban**: Iona 60.6%, Virei "
     "53.3% and Curoca 48.6% in Namibe and Cunene, against 14.9% in Luanda province. Those "
     "are Kuvale, Himba and Mucubal transhumance districts and some of the least "
     "missionised ground in the country. A herder who keeps the ancestors appears to be "
     "answering *no religion* rather than *animist*, so for Angola the floor is a long way "
     "below the number. Nothing is moved between the two cells, because the census "
     "publishes no split and any transfer would be invented; see ao2024.py. "
     "**One child, added for Benin, which is the first source here to name an African "
     "tradition individually** (spec §2: a node earns its place by being countable "
     "somewhere). Everything else sits on this node, including Benin's own `Autres "
     "traditionnelles` — 260,448 people whose geography is not Vodun's at all, being the "
     "Atacora highlands in the north-west rather than the southern plateau: Boukoumbé 54.1%, "
     "Cobly 42.1%, Tanguiéta 36.6%. Those are the traditions of the Bètammaribè (Otammari) "
     "and their neighbours, and INStaD's cell exists because Benin has two large and "
     "unrelated traditional-religion complexes and its census can see both."),

    # ---- added 2026-09-07 with Benin, and it is the first named African tradition here.
    ("indigenous.african.vodun",
     "Vodun",
     "The religion of the Fon, Adja, Ewe and Yoruba-speaking peoples of the Bight of Benin — "
     "the *vodun* themselves, the deities, and the priesthoods and initiatory societies that "
     "serve them. **1,160,279 people in Benin, 11.6% of the country, and it is the first "
     "time any source on this map has counted an African tradition under its own name.** "
     "INStaD spells it `Vodoun`, which is the Beninese official spelling; `Vodou` is the "
     "Haitian descendant of the same religion and `Voodoo` is a term this map does not use. "
     "**Benin is the one place where the census can ask this, and the reason is political "
     "rather than statistical.** Vodun was suppressed under Kérékou's Marxist government and "
     "then recognised outright in 1996; 10 January is a national holiday. Where every other "
     "African census on this map offers one undifferentiated `Traditionalist` box, RGPH-4 "
     "offers `Vodoun` and `Autres traditionnelles` as separate answers, and the two land in "
     "different halves of the country. "
     "**Its heartland is the Adja country and not, as usually assumed, Abomey.** The six "
     "highest communes are Djakotomey 69.1%, Toviklin 66.1%, Lalo 56.2%, Aplahoué 55.3% and "
     "Klouékanmè 50.5% — the whole of the Couffo department, which is 56.5% Vodun — with "
     "Athiémé 42.8% in the Mono beside it. The Fon plateau that the kingdom of Dahomey ruled "
     "from Abomey is lower: Agbangnizoun 39.2%, Abomey itself 23.5%. It falls to 0.1% in the "
     "Muslim north. "
     "**And the count is a floor for the usual reason** — the box is exclusive of "
     "`Catholique` and `Islam`, and in Benin more than almost anywhere the same person is "
     "commonly both. See the REVIEW note in bj2013.py, and `sources/bj.md` §7, which has the "
     "one piece of direct evidence this map has ever had for that."),

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

    # ---- Nepal, added 2026-09-07 with NPHC 2021.
    #
    # A regional child on the pattern of `indigenous.african` and `indigenous.indian`,
    # holding the two traditional-religion boxes NSO prints beside Hinduism and Buddhism
    # rather than inside them. Both are large enough to be countable in the §2.4 sense and
    # both are named on the census form, which is what earns the depth — nothing here is
    # a category this project invented.
    #
    # NOT here: `Bon`, which NSO also names and which is a root of its own — see its note.
    # An organised religion with a canon and a monastic hierarchy is not a folk tradition,
    # and the census's own three-way split (Bouddha / Bon / Kirat) is the evidence that
    # Nepal does not read them as one kind of thing either.
    ("indigenous.himalayan",
     "Himalayan traditional religions",
     "The traditional religions of Nepal's Janajati peoples, named on the census form and "
     "counted separately from Hinduism and Buddhism. 1,026,252 people between the two "
     "children, which puts this family within a hundred people of "
     "`indigenous.indian.gondi` and behind only `indigenous.indian.sarna`. **Unlike "
     "India's, it is not a write-in under `Other`: Nepal prints these as boxes on the "
     "form**, which is why they arrive as counts rather than as a residual to be split."),
    ("indigenous.himalayan.kirat",
     "Kirat Mundhum",
     "The religion of the Kirati peoples of eastern Nepal — Limbu, Rai, Yakkha, Sunuwar — "
     "whose scripture, the *Mundhum*, is an oral corpus recited by *phedangma* and other "
     "ritual specialists. 924,204 people, 3.17% of Nepal: the third-largest count of a "
     "named indigenous religion on this map, after Sarna (4.96M) and Gondi (1.03M), and "
     "the only one of the three a census asked about directly. "
     "**Its geography is the sharpest of any category in Nepal.** Koshi province is 16.8% "
     "Kirat against 0.01% in Sudurpashchim, Karnali and Lumbini alike; Panchthar district "
     "is **55.7%** and Taplejung 44.2%; and Mahakulung in Solukhumbu reaches **87.3%**. "
     "That is Limbuwan and the Rai hills of the far east, and the boundary against the "
     "Hindu middle hills is abrupt rather than gradual. "
     "The Kirat count has risen at every census since 1991, which is a revival movement "
     "asserting a distinct identity against enumeration as Hindu — the same dynamic "
     "`indigenous.indian.sanamahi` records for Manipur."),
    ("indigenous.himalayan.prakriti",
     "Prakriti",
     "Literally *nature*. The census box for nature worship. 102,048 people, 0.35%, and "
     "**it is one region rather than a scatter**: Rukum East is 16.6% and Rolpa 8.2%, with "
     "Thawang in Rolpa at 45.0% and Sunchhahari at 30.7%. That is the **Kham Magar** "
     "country of the mid-western hills — not the Tharu Terai, which is the other place a "
     "reader might expect it and where it is near zero. "
     "**It is a self-description rather than a named tradition**, which is why it sits "
     "beside Kirat rather than under it: no single ritual system, priesthood or corpus is "
     "meant by it, and NSO offers no gloss. Kept on the tree anyway, and not folded into a "
     "residual, because the census names it and a residual would lose the one thing it "
     "says — that these people answered the question and answered it with something other "
     "than the five world religions on the form."),

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

    # --- added 2026-09-06 with Vietnam, and it is the node spec §14.7 specified for China
    #     without knowing which country would need it first.
    ("unknown",
     "Religion unknown",
     "**A SEVENTH member of the §6.3a grey family, and it is the one that reports nothing at "
     "all.** People the source counted and whose religion it did not establish. Not a report "
     "of no religion; not a religion the tree could not place; not a positive report of "
     "belief outside an institution. The claim is only that these people are here.\n\n"
     "**Vietnam is the first, at 70,195,530 people — 81.8% of the country.** Its census asks "
     "which of the state-recognised religious organisations a person belongs to, and that "
     "many answered none of them. The answer is real and it does not mean irreligion: "
     "ancestor veneration is close to universal, the village đình and the mother-goddess "
     "rites of đạo Mẫu are everywhere, and most of the people who would call themselves "
     "Buddhist in conversation are in here rather than in the 4.6 million the census counts. "
     "The bucket's composition is decided by Vietnam's list of registered organisations, "
     "which is a fact about the question.\n\n"
     "**Why not `unrecorded`, which is the near neighbour.** That node is the residual of a "
     "source that never asked — Germany's church-tax register — and §6.3a-i's argument for "
     "splitting it from `unaffiliated` was precisely that being asked and not being asked are "
     "different. Vietnam's respondents WERE asked; the answer set was too narrow for their "
     "answer to mean what it says. Folding the two together would blur the distinction that "
     "ramp exists to draw, and §14.7 says so outright: *`unrecorded` means 'the register only "
     "ever saw church-tax bodies' and is Germany-shaped*.\n\n"
     "**The test for admitting anything else here**, and it is deliberately narrower than "
     "'we are not sure': the source COUNTED these people, and what they practise is not "
     "determinable from it at any geography. Not a small residual — that is `other.<source>` "
     "— and not a suppressed cell. **China is the case it was specified for** (§14.7), and it "
     "was BUILT on 2026-09-07 (§14.13): **1.21 billion people, 97.5% of the country**, being "
     "the 1.14 billion Han plus every nationality no religion is claimed for. That makes this "
     "node much the largest on the map — bigger than every other node put together — and it "
     "is worth naming why that is allowed. China's census never asked about religion at all, "
     "so unlike Vietnam this is not the residual of an answer; it is the residual of a "
     "question that was never put. The test above still passes on the letter of it — the "
     "source counted these people and what they practise is not determinable from it at any "
     "geography — and the alternative was the map's previous state, in which a billion people "
     "did not appear.\n\n"
     "**The colour is Anita's, 2026-09-06: `#68665a`** — a neutral grey with a warm cast, which "
     "is what §14.7's *slightly yellow* was asking for. It is the THIRD quietest of the seven "
     "rather than the first, which departs from §6.3a's 'the biggest node is the darkest' on "
     "purpose: at 70,195 dots this is Vietnam's settlement pattern, and §14.7's argument for "
     "drawing the residual at all is that the people are visible as people. See spec §6.3a-ii, "
     "including the two earlier attempts and why an HSL lightness cannot be compared across "
     "hues.\n\n"
     "**And it is NOT in the viewer's `no religion` control** (`NO_RELIGION_IDS`), for "
     "`unrecorded`'s reason and more strongly: hiding it behind a switch labelled 'no "
     "religion' asserts in one click the exact thing the node exists to avoid asserting."),

    # --- added 2026-09-07 with Myanmar. Anita's call: draw the non-enumerated as their own
    #     category rather than applying the state's own assumption about their religion.
    ("unenumerated",
     "Not enumerated",
     "**AN EIGHTH member of the §6.3a grey family, and the only one where the source did not "
     "COUNT the people.** Every other member is a residual of some answer: `unaffiliated` said "
     "none, `secular` stated a position, `parody` refused the question, `unchurched` reported "
     "belief without institution, `other.<source>` gave a religion the tree cannot place, "
     "`unrecorded` was never asked because the instrument is a register, and `unknown` was "
     "asked from too narrow a list. **This one was never reached at all.** The number is the "
     "state's own estimate of how many people it did not enumerate, and no religion is "
     "published for them at any geography.\n\n"
     "**Myanmar is the first, at 1,206,353 people — and Rakhine State is 1,090,000 of them, "
     "34% of that state's population.** The 2014 census report says why in its own words: "
     "*\"In Rakhine, an estimated 1.09 million people were not enumerated in the Census "
     "because they were not allowed to self-identify using a name not recognized by the "
     "Government.\"* That is the Rohingya. The other two entries are Kayin (69,753) and "
     "Kachin (46,600), areas that were not enumerated because they were not under government "
     "control.\n\n"
     "**Why not `unknown`, which is the near neighbour.** §6.3a-ii admits a category there on "
     "a deliberately narrow test: *the source COUNTED these people, and what they practise is "
     "not determinable from it.* Vietnam's 70 million were counted. Myanmar's 1.2 million "
     "were not — they are an estimate of a headcount, which is a weaker claim than any other "
     "figure on this map, and the distinction between *counted, religion unclear* and *not "
     "counted at all* is exactly the kind §6.3a-i created this family to keep.\n\n"
     "**And why not `islam`, which the source itself suggests.** The report states its own "
     "assumption — *\"It is assumed that the non-enumerated population in Rakhine is mainly "
     "affiliated with the Islamic faith\"* — and applies it at the Union level to publish a "
     "second set of national percentages (Islam 2.3% → 4.3%). It publishes no such breakdown "
     "at State level, which is where this map draws. Assigning them to `islam` would be "
     "inventing a magnitude the source does not publish at the resolution it is used, which "
     "is §14 rule 1. The assumption is quoted in Myanmar's `note_public` instead, so a reader "
     "is told what the state itself concluded and this node claims only what is certain.\n\n"
     "**The alternative was not drawing them, and it is much worse.** Enumerated Rakhine is "
     "2,098,807 people of whom 28,731 are Muslim, so a map built from the enumerated columns "
     "alone renders Rakhine **96.2% Buddhist** — the census's own exclusion reproduced as a "
     "finding, which is §14.2's second risk in its purest form.\n\n"
     "**Every row on this node is `modelled` (§7).** The tiers are about whether anybody was "
     "counted, and here nobody was; 1,090,000 is a round number in the source because it is "
     "an estimate. So `inferred dots: hidden` empties this node completely, which is the "
     "honest test — it shows the census exactly as the state published it, with the hole "
     "where the hole is.\n\n"
     "**It is NOT in the viewer's `no religion` control** (`NO_RELIGION_IDS`), for "
     "`unrecorded`'s and `unknown`'s reason and more strongly still: these people were not "
     "asked and did not decline, and hiding them behind a switch labelled 'no religion' "
     "would assert in one click the thing this node exists to refuse to assert."),

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
     "Other",
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

    # --- added 2026-09-06 with Jamaica.
    ("afrodiasporic.revival",
     "Revival Zion and Pukkumina",
     "STATIN's `Revivalist` — **36,296 people, 1.35% of Jamaica**, and the third Afro-"
     "diasporic tradition this map can name. Revival Zion and Pukkumina (Pocomania) came out "
     "of the **Great Revival of 1860-61**, when a Christian revival met the surviving "
     "Afro-Jamaican Myal practice and produced something that is neither: Christian scripture "
     "and hymnody, spirit possession, drumming, water rites and a hierarchy of messenger "
     "spirits. Jamaicans distinguish Zion (the 'heavenly' side, closer to the churches) from "
     "Pukkumina (the 'ground' side, closer to the ancestors); the census does not, and one "
     "node is therefore the honest grain.\n\n"
     "**Filed beside Umbanda and Candomblé rather than under Christianity, and the parallel "
     "is exact.** All three are New World religions in which an African inheritance and a "
     "Christian one fused rather than one absorbing the other, and spec §3.3 says the "
     "syncretism gets a node instead of being split across its ingredients. Putting Revival "
     "under `christianity` would make the same mistake in Jamaica that filing Umbanda under "
     "`christianity.catholic` would make in Brazil.\n\n"
     "**It is separate from `rastafari`**, which STATIN also counts separately and which is a "
     "different and later thing — 1930s, Ethiopianist, and not a possession religion. And it "
     "is the tradition Jamaican popular music came out of, which is the reason a 1.35% "
     "category is worth a node at all."),

    # --- added 2026-09-07 with Trinidad and Tobago.
    ("afrodiasporic.orisha",
     "Orisha (Shango)",
     "CSO's `Orisha` — **11,918 people, 0.90% of Trinidad and Tobago, and the first census "
     "count of an orisha religion under its own name anywhere on this map.** Yoruba orisha "
     "worship carried to Trinidad in the nineteenth century, largely by liberated Africans "
     "settled after 1838, and long known locally as **Shango** for the orisha of thunder. "
     "It is the direct sibling of Candomblé and Santería: the same West African pantheon, "
     "the same drumming and possession, the same overlay of Catholic saints on individual "
     "orishas.\n\n"
     "**Read the number as a floor, not a count.** In Trinidad, Orisha practice and "
     "Spiritual Baptist practice overlap heavily — many people take part in both, and much "
     "of the literature treats them as one religious complex — while a census offers one "
     "box. So this counts people who chose Orisha *over* the alternatives. The same caution "
     "`rastafari` carries in Jamaica, for a different reason.\n\n"
     "**Its geography is the eastern and southern coast**: Mayaro/Rio Claro 2.1%, Point "
     "Fortin 1.8%, Port of Spain 1.6%, against 0.3% in Penal/Debe. Legally recognised in "
     "Trinidad and Tobago since the Orisa Marriage Act of 1999."),
    ("afrodiasporic.spiritualbaptist",
     "Spiritual Baptist (Shouter)",
     "CSO's `Baptist-Spiritual Shouter` — **75,002 people, 5.67%, more numerous than "
     "Trinidad's Anglicans and nearly five times its ordinary Baptists**, which the census "
     "counts separately as `Baptist-Other`.\n\n"
     "A Trinidadian religion in which Baptist Protestantism and West African — mainly "
     "Yoruba and Kongo — practice fused rather than one absorbing the other: scripture, "
     "hymnody and baptism alongside spirit possession, the *mourning* ground, bell-ringing, "
     "candles, water rites and head-tying.\n\n"
     "**Filed beside Revival Zion, Umbanda and Candomblé rather than under Christianity, "
     "and the parallel with Jamaica's Revivalists is structural.** Both are New World "
     "religions with a genuinely double descent, and spec §3.3 says the syncretism gets a "
     "node instead of being split across its ingredients. Putting this under "
     "`christianity.baptist` would make the same mistake in Trinidad that filing Umbanda "
     "under `christianity.catholic` would make in Brazil — and it would additionally throw "
     "away a distinction CSO went to the trouble of publishing.\n\n"
     "**The objection is recorded rather than dismissed**: Spiritual Baptists "
     "overwhelmingly describe themselves as Christians and as Baptists, and many would "
     "reject this placement. What decides it is that the tree is a genealogy of traditions "
     "and not a register of self-description. The same call, made the same way, as "
     "`afrodiasporic.revival`.\n\n"
     "**Trinidad treats it as a matter of history, not of ethnography.** The Shouters "
     "Prohibition Ordinance banned the religion outright from 1917 to 1951 — the drumming "
     "and bell-ringing were the stated offence — and **30 March is a public holiday, "
     "Spiritual Baptist / Shouter Liberation Day.** Its geography is Afro-Trinidadian: "
     "13.0% in Point Fortin, 10.6% in Tobago, 9.6% in San Juan/Laventille, against 2.6% in "
     "Penal/Debe."),

    ("eastasiannew",
     "East Asian new religions",
     "The 19th- and 20th-century foundings of Japan, Korea and China — shinshūkyō and "
     "sinheung jonggyo — studied together because they are one phenomenon: charismatic "
     "founders, syncretic of Buddhism, Confucianism, folk practice and Christianity, and "
     "often millenarian. **Was `japanesenew`, a root, until 2026-09-05, when Korea arrived "
     "counting three of these separately and the Japanese-only name stopped fitting.** "
     "Renaming rather than adding a second root was the cheap move (§12): the wedge this "
     "sits in is at 3° spacing and a new family costs every other small group a degree. "
     "It also FIXED something: IBGE's own category is `Novas religiões orientais` — new "
     "ORIENTAL religions — and its residual `Outras novas religiões orientais` was landing "
     "on the Japanese node for want of a parent. It now lands on the parent, which is what "
     "the source actually says. "
     "Distinct from `shinto`, the shrine tradition, and from `buddhism`, though individual "
     "bodies here derive from one or the other. "
     "**Two existing roots arguably belong under this and are deliberately left alone**: "
     "`unification` is Korean (Sun Myung Moon, 1954) and `caodaism` Vietnamese (1926), so "
     "both are East Asian new religions by any reading. Moving them would restructure four "
     "drawn countries at once for no gain today, and Anita's call on 2026-09-05 was that "
     "this is not a blocker and can change later. If it does, they become "
     "`eastasiannew.korean.unification` and `eastasiannew.vietnamese`."),
    ("eastasiannew.japanese",
     "Japanese new religions",
     "Sekai Kyūsei Kyō (Igreja Messiânica Mundial), Seicho-no-Ie, Perfect Liberty, "
     "Tenrikyo, Soka Gakkai. Brazil has the largest Japanese diaspora in the world and "
     "enumerates them: 103,716 Messiânica alone in 2010. Australia and New Zealand name "
     "Tenrikyo and Mahikari. Japan itself will need this node heavily (sources.md §2)."),
    ("eastasiannew.korean",
     "Korean new religions",
     "Added 2026-09-05 with South Korea, which is the only census on this map that counts "
     "any of them — and counts three separately, at sigungu level, for 110,241 people. "
     "The 2015 census is the last one to ask, so this is the only measurement there will be "
     "for the foreseeable future."),
    ("eastasiannew.korean.cheondogyo",
     "Cheondogyo",
     "천도교, 65,964 in 2015. The institutional continuation of Donghak — the 1860 movement "
     "whose peasant rebellion of 1894 triggered the Sino-Japanese War, and whose leaders "
     "wrote and signed the 1919 Declaration of Independence. Not a sect of anything: its "
     "doctrine of innaecheon, that each person bears the divine, was formulated against both "
     "Confucian hierarchy and Catholic missionary teaching."),
    ("eastasiannew.korean.daesun",
     "Daesun Jinrihoe",
     "대순진리회, 41,176 in 2015 — the largest of the Jeungsanist bodies, which descend from "
     "Kang Jeungsan (1871-1909) and hold that he reordered the cosmos. Counted here because "
     "KOSIS gives it a cell of its own; the census figure is far below the movement's own "
     "claims, which is ordinary for new religions and is not adjudicated here (§3.1)."),
    ("eastasiannew.korean.daejong",
     "Daejonggyo",
     "대종교, 3,101 in 2015. The Dangun religion — revived in 1909 around the founding "
     "ancestor of the Korean nation, suppressed by the Japanese colonial authorities, and "
     "central to the independence movement in Manchuria. An ethnic-national religion rather "
     "than a syncretic new one, and the weakest fit in this family; it is here rather than "
     "under `indigenous` because it is a 20th-century founding with a doctrine and a "
     "hierarchy, not a continuous folk tradition."),
    ("eastasiannew.vietnamese",
     "Vietnamese new religions",
     "Added 2026-09-06 with Vietnam, and it is the third national grouping in this family "
     "after Japan's and Korea's. Holds the two surviving **Minh đạo** bodies — the Vietnamese "
     "branches of the Chinese Xiantiandao / vegetarian-hall tradition, transplanted to Cochin"
     "china in the 19th century — which Vietnam's census counts separately for 1,075 people "
     "between them. "
     "**`caodaism` belongs here and is deliberately still a root.** It is 807,915 people, it "
     "is the largest Vietnamese new religion by two orders of magnitude, and the Minh đạo "
     "halls below are among the traditions it drew its séance practice from. Anita's call on "
     "2026-09-05 was that moving it is not a blocker; it stays out because four drawn "
     "countries map to `caodaism` by id and the gain today is nil. **If it moves, the id is "
     "`eastasiannew.vietnamese.caodai`** — which supersedes the suggestion in this family's "
     "own note above, written before there was a Vietnamese grouping node to be a child of. "
     "The Vietnamese bodies here are NOT the Mekong Delta Buddhist reform lineage, which is "
     "under `buddhism` because those five say they are Buddhist and these two do not."),
    ("eastasiannew.vietnamese.minhsu",
     "Minh Sư Đạo",
     "Giáo hội Phật đường Nam Tông Minh Sư Đạo, 709 in 2009 and 260 in 2019 — the Vietnamese "
     "Xiantiandao (先天道) line, brought south by Ming loyalist refugees and organised into "
     "vegetarian halls (Phật đường) from the 1860s. The oldest of the five Minh branches and "
     "the direct ancestor of much of Caodaist ritual."),
    ("eastasiannew.vietnamese.minhly",
     "Minh Lý Đạo",
     "Hội thánh Minh Lý đạo - Tam Tông Miếu, 366 in 2009 and 193 in 2019 — founded 1924 in "
     "Saigon, two years before Caodaism and out of the same Minh-branch séance milieu. "
     "`Tam Tông` is the three teachings: Buddhism, Daoism and Confucianism, held together "
     "rather than chosen between."),

    # --- added 2026-09-03, when Australia, Ireland and Mexico were wired. Australia's 148
    #     categories reach further than any source so far except ASARB, and four of these
    #     exist because it names things nobody else does.
    ("christianity.maori",
     "Maori Christian churches",
     "Ratana and Ringatu — churches founded by Maori prophets, Christian in theology and "
     "Maori in authority and practice, and belonging to neither the missionary "
     "denominations that preceded them nor to `indigenous`. 3,246 in Australia; New "
     "Zealand, where they were founded, counts far more. "
     "**Of the three `Locally founded churches` this is the closest kin to the African "
     "Instituted Churches** — both prophet-founded, both healing-centred, and both religious "
     "and anti-colonial political movements at once; Ratana and Kimbanguism are a standard "
     "comparison. See the group's note in LINEAGE."),

    # --- added 2026-09-04 with the Philippines, whose 129 categories are four levels deep
    #     on Philippine evangelicalism and reach bodies no other source names at all.
    ("christianity.filipinoindependent",
     "Filipino independent churches",
     "Churches founded in the Philippines, outside every tradition the missionaries "
     "brought, and belonging to none of them afterwards. The same idea as "
     "`christianity.maori` and a far larger population: Iglesia ni Cristo alone is 2.8M, "
     "the third largest religious body in the country. "
     "WHAT IS NOT HERE MATTERS AS MUCH AS WHAT IS. The Aglipayan church is Filipino, "
     "independent and larger than everything below this node except INC, and it sits on "
     "`christianity.catholic.independent` instead, because it kept Catholic orders, rites "
     "and self-description and that node exists for exactly that. Nor is this a home for "
     "every church a Filipino founded: Jesus Is Lord, Christ's Commission Fellowship and "
     "Victory are Filipino-founded and are squarely inside the global evangelical and "
     "charismatic streams, so they are filed there. What is left, and what this node is "
     "for, are the bodies whose theology is their own — nontrinitarian restorationist, or "
     "gathered around a Filipino founder — and which no classification outside the "
     "Philippines has a shelf for."),
    ("christianity.filipinoindependent.inc",
     "Iglesia ni Cristo",
     "Founded 1914 by Felix Manalo; 2,806,524 in the 2020 census, 2.6% of the country and "
     "its third largest religious body after the Catholics and the Muslims. Nontrinitarian "
     "— it holds Christ to be man and not God — which is why it is not filed among the "
     "Protestant families it is otherwise sometimes grouped with. "
     "`Church of Christ` is INC's own English name AND A SEPARATE CENSUS CATEGORY of "
     "429,921 people that is not it; see the REVIEW note in ph2020.py before assuming "
     "either way."),
    ("christianity.filipinoindependent.mcgi",
     "Members Church of God International",
     "The census calls it `Most Holy Church of God in Christ Jesus`, its registered name; "
     "the Philippines knows it as MCGI or `Ang Dating Daan` after its television "
     "programme, the longest-running religious broadcast in the country. 9,585 in 2020. "
     "Nontrinitarian, and descended from the same 1920s Manila milieu as INC — its parent "
     "body was a 1928 split from it — which is why the two sit together here."),

    # --- added 2026-09-05 with Kenya, whose census is the first anywhere on this map to
    #     count either of these as a category of its own.
    # --- added 2026-09-08 with the Solomon Islands, as the Melanesian member of the
    #     `christianity.maori` / `christianity.filipinoindependent` /
    #     `christianity.africaninstituted` set: churches founded in the islands by islanders,
    #     outside every tradition the missions brought and belonging to none of them after.
    ("christianity.melanesianindependent",
     "Melanesian independent churches",
     "The same idea as `christianity.maori`, `christianity.filipinoindependent` and "
     "`christianity.africaninstituted`, for Melanesia. **What is NOT here defines it:** the "
     "South Sea Evangelical Church is a Solomon Islands body and 17.3% of that country, and "
     "it sits on `christianity.evangelical` instead, because it descends from the Queensland "
     "Kanaka Mission rather than from a break with one. Vanuatu's Neil Thomas Ministries is "
     "ni-Vanuatu-founded and squarely inside the global charismatic stream, so it is filed "
     "there. This node is for the churches that came out of a mission and then stopped "
     "belonging to it."),
    ("christianity.melanesianindependent.cfc",
     "Christian Fellowship Church",
     "**Silas Eto's church, 16,179 people, 2.2% of the Solomon Islands — and it is one "
     "island group.** 13,629 of those 16,179 are in Western Province, where it is 14.5%, "
     "and it reaches **77.2% of Kusaghe ward, 62.6% of Roviana Lagoon and 57.2% of "
     "Kolombaghea**. Everywhere else in the country it rounds to nothing.\n\n"
     "Eto was a Methodist teacher and catechist on New Georgia who was known as the **Holy "
     "Mama**; he broke with the Methodist mission in 1960 and took much of the New Georgia "
     "membership with him. The church kept a Methodist shape and added a devotion to Eto "
     "himself that the mission would not have, which is why it belongs here and not under "
     "`christianity.methodist`: containment is a fact about people now (§2.1), and its "
     "members are not Methodists.\n\n"
     "SINSO prints it as its own cell at ward level in 2019, which is what makes it "
     "countable — a national census naming an indigenous breakaway church and following it "
     "down to the ward."),

    # --- note rewritten 2026-09-08. What it said was written when Kenya and then Zimbabwe
    #     were the whole node; ci, ao, sz and za were each added without touching it, and it
    #     had drifted to saying the opposite of the dots ("mostly Zimbabwean", "three
    #     countries supply it, each with one undivided cell"). Every figure below is
    #     recomputed from the seven contributing countries' counts(), not carried over.
    ("christianity.africaninstituted",
     "African Instituted Churches",
     "The churches founded in Africa by Africans, outside the mission denominations and "
     "belonging to none of them afterwards. The name is not settled: Independent, "
     "Initiated, Instituted and Indigenous are all in use and all abbreviate to AIC. The "
     "same idea as `christianity.maori`, `christianity.filipinoindependent` and "
     "`christianity.melanesianindependent`, on a much larger scale. "
     "**Two currents account for nearly all of what is counted here.** The older are the "
     "Ethiopian churches, secessions from mission churches from the 1880s onwards, made "
     "over who was allowed to lead a congregation rather than over doctrine, and keeping "
     "the liturgy they left with. The larger are the prophetic and healing churches, each "
     "gathered around a named founder and marked by faith healing, dreams, white robes and "
     "worship in the open air or at a river rather than in a building: the Zionist and "
     "Apostolic churches of southern Africa, the Aladura churches of Yorubaland, and in "
     "Kenya the Legio Maria, the African Israel Nineveh Church, the Nomiya Luo Church and "
     "the Akorino. A third and much newer current is African-founded and Pentecostal in "
     "practice; this node takes a body on where it was founded rather than on how it "
     "worships, which is what puts `christianity.africaninstituted.bomdeus` here and not "
     "on `christianity.pentecostal`. "
     "**Seven countries supply it and two of them are 78% of it.** Of **25,899,969** "
     "people on the node and its children, South Africa is **14,158,453 (54.7%)** and "
     "Zimbabwe **6,112,503 (23.6%)**, then Kenya 3,292,573 (12.7%), Angola 1,099,234 "
     "(4.2%), Benin 676,032 (2.6%), Eswatini 420,690 (1.6%) and Côte d'Ivoire 140,484 "
     "(0.5%). Read the other way round it is a large share of a small country as often as "
     "a small share of a large one: **40.3% of Zimbabwe**, where ZIMSTAT's `Apostolic "
     "Sect` is the largest single religious answer in the country, **39.3% of Eswatini** "
     "and **25.8% of South Africa**, against 7.0% of Kenya, 6.8% of Benin, 3.3% of Angola "
     "and 0.5% of Côte d'Ivoire. "
     "**It is coarse, and 95.2% of it is coarse in the same way.** That share sits on the "
     "bare parent, in six census cells that between them cover thousands of separate "
     "churches: South Africa's `African Independent Church/African Initiated Church`, "
     "Zimbabwe's `Apostolic Sect` (the Vapostori, founded by Johane Marange and Johane "
     "Masowe in the 1930s, and dozens of distinct churches that no source separates), "
     "Kenya's `African Instituted Churches`, Eswatini's `Zionists` and `Apostles` as two "
     "cells, and Benin's `Chrétien céleste`. The four children hold the remaining 4.8%: "
     "Côte d'Ivoire's Harrist church and Angola's Kimbanguist, Tocoist and Bom Deus. "
     "Benin's cell names a single church too, the Celestial Church of Christ, and sits on "
     "the parent only because it was ingested before this node had any children; the "
     "Harrist note prices promoting it. "
     "**The node is still smaller than the thing it is for.** Nigeria, the home of the "
     "Aladura churches, is not on this map at all, so West Africa reaches it through Benin "
     "and Côte d'Ivoire only. Ghana's census has no cell for the Musama Disco Christo "
     "Church or the Twelve Apostles, so they are inside its `Other Christian`, **12.3% of "
     "the country**, and stay on `christianity.other`; see the REVIEW note in gh2021.py, "
     "which predicted this node and said where its people would be found in the meantime. "
     "Malawi's identically shaped `Other Christian Denominations` is **26.6%** and "
     "mw2018.py records the same problem. Mozambique's `Sião/Zione` would land here if it "
     "is ever ingested."),

    # --- added 2026-09-07 with Côte d'Ivoire. The FIRST child this node has had.
    ("christianity.africaninstituted.harrist",
     "Harrist Church",
     "The church of **William Wadé Harris** — 140,482 people in Côte d'Ivoire, 0.48%, and "
     "**the first African Initiated Church anywhere on this map that a census named and "
     "counted as its own cell at a drawn geography** (Angola's three followed the next "
     "day, and Benin's Celestial Church of Christ is a fifth still sitting on the parent). "
     "Harris was a Grebo teacher and "
     "catechist from Liberia who walked the lagoon coast of the Ivory Coast and the Gold "
     "Coast between 1913 and 1915 in a white robe with a bamboo cross, telling people to "
     "burn their fetishes and be baptised. He is usually credited with more conversions "
     "than any missionary in African history — something over a hundred thousand in "
     "eighteen months — and the Catholic and Methodist missions inherited most of them "
     "when the colonial authorities deported him. What stayed organised under his own name "
     "is the **Église Harriste**, one of Côte d'Ivoire's state-recognised confessions. "
     "**It is the oldest thing on this node.** Harris was preaching two decades before "
     "Johane Marange founded the Vapostori and three before the Celestial Church of Christ; "
     "the parent's centre of mass is southern-African and mid-century, and this child is "
     "West African and pre-war. "
     "**Its geography is its own evidence**: 2.4% in La Mé, 1.7% in Grands-Ponts, 1.6% in "
     "Agnéby-Tiassa, and **a printed 0.0% in seven northern régions** — Bafing, Folon, "
     "Hambol, Kabadougou, Poro, Tchologo, Worodougou. The 1913-15 itinerary, still legible "
     "in a 2021 census. "
     "**One peer is NOT a child and should be**: Benin's `Chrétien céleste` (676,032, the "
     "Celestial Church of Christ, §9ai) is an individually-named AIC of the same standing "
     "and 4.8x the size, and it currently sits on the parent because it was ingested before "
     "this node existed. Promoting it is a one-line change in bj2013.py plus a re-scatter of "
     "Benin, and it was deliberately NOT done here rather than silently altering a country "
     "nobody asked about."),

    # --- added 2026-09-08 with Angola, which triples this node's named children in one go.
    #     Ordered by founding: Kimbangu 1921, Toco 1949, Lutumba 1981.
    ("christianity.africaninstituted.kimbanguist",
     "Kimbanguist Church",
     "L'Église de Jésus-Christ sur la Terre par son envoyé Simon Kimbangu — **409,254 "
     "people in Angola, 1.19%**, and the oldest of the three prophetic churches this node "
     "gained with that country. Simon Kimbangu was a Baptist catechist at Nkamba in the "
     "Belgian Congo who began healing and preaching in April 1921; the colonial "
     "administration arrested him that September and he died in prison at Elisabethville "
     "in 1951, having spent thirty years inside. The church organised around his sons and "
     "was legalised in 1959. "
     "**Angola's count is the church's spread across a border it never recognised.** "
     "Nkamba is in Bas-Congo, a few kilometres from Angolan territory, and the Angolan "
     "figures fall away from it exactly as distance would predict: Zaire province 9.43% "
     "and Uíge 7.45% against 1.19% nationally, peaking at **Lufíco 34.7%, Nova Esperança "
     "33.1%, Alto Zaza 29.6% and Nóqui 28.1%** — Nóqui and Lufíco sit on the Congo river "
     "opposite Matadi. One people, the Bakongo, with a colonial line drawn through them, "
     "and the census drawing the line's irrelevance. "
     "The DRC holds far more Kimbanguists than Angola does and publishes no census "
     "religion figures, so **this is a large minority of the church, seen from the only "
     "side that counts it**."),

    ("christianity.africaninstituted.tocoist",
     "Tocoist Church",
     "A Igreja do Nosso Senhor Jesus Cristo no Mundo — **350,936 people in Angola, "
     "1.02%**, and the largest church founded by an Angolan. **Simão Gonçalves Toco** "
     "(1918-1984) was a Baptist mission pupil and choirmaster from Sadi Zulumongo in "
     "Maquela do Zombo, northern Uíge; the church dates itself to the descent of the Holy "
     "Spirit on his choir at Léopoldville on **25 July 1949**, and the Belgian and "
     "Portuguese authorities deported and then confined him for most of the following "
     "quarter-century, first to the Azores and then to southern Angola. It is prophetic "
     "and Bakongo in idiom, out of the Baptist mission rather than the Catholic one, and "
     "it is the closest thing Angola has to a national church of its own. "
     "**The census puts it back where it started.** Uíge province is 6.22% against 1.02% "
     "nationally, peaking at Nsosso 22.9%, Bungo 16.7% and Mucaba 15.4% — and **Maquela "
     "do Zombo, the municipality Toco was born in, is 11.1%**. Luanda holds the largest "
     "absolute number, 99,447, at 1.19%, which is the twentieth-century migration rather "
     "than the origin. A prophet's birthplace, legible in a 2024 census."),

    ("christianity.africaninstituted.bomdeus",
     "Bom Deus",
     "A Igreja Fraternidade Evangélica de Pentecostes na África em Angola — **339,044 "
     "people, 0.98%**. Begun by **Simão Lutumba in 1981** as the Angolan arm of the "
     "Congolese Nzambe Malamu and separate from it since the 1990s. Pentecostal in "
     "practice and African-founded in origin; this node is defined by the second, which is "
     "the same reason Zimbabwe's Vapostori sit here rather than on "
     "`christianity.pentecostal`. "
     "**It is the modern one of the three, and it looks it.** Kimbanguism and Tocoism are "
     "still sitting on the prophetic map of the 1920s and 1940s, each with a province "
     "above 6% and municipalities above 20%. Bom Deus has no region at all: Cuanza Norte "
     "2.55%, Icolo e Bengo 2.02%, Lunda Norte 1.85%, Malanje 1.82%, and not one "
     "municipality in the country above 6.4%. A church founded in Luanda in the 1980s and "
     "grown nationally through the war and after it, rather than one that spread out from "
     "a prophet's village."),

    ("christianity.evangelical",
     "Evangelical, unspecified",
     "For sources that collect `Evangelical` as an answer distinct from both `Protestant` "
     "and a named body. Like `christianity.protestant` it holds an ANSWER rather than a "
     "church, and it is deliberately NOT a parent of anything — evangelicalism cuts across "
     "the Baptist, Holiness, Pentecostal and independent families rather than containing "
     "them. "
     "Added for Kenya, where KNBS puts `Evangelical Churches` beside `Protestant` and it is "
     "**9,648,690 people, 20.4% of the country** — the Africa Inland Church, the Baptists, "
     "the Pentecostal Assemblies of God, Deliverance Church and the rest of the Evangelical "
     "Alliance of Kenya, against the mainline ACK, PCEA and Methodists in the Protestant "
     "cell. `mk2021.py` wanted this node for 678 Macedonian `Евангелисти` and was right not "
     "to add it then; spec §2's rule is that a node earns its place by being countable, and "
     "Kenya makes it countable fourteen thousand times over."),

    # --- added 2026-09-06 with Malawi.
    ("christianity.sdabaptistapostolic",
     "Adventist, Baptist or Apostolic",
     "NSO Malawi's `SDA/Baptist/Apostolic` — **1,644,829 people, 9.4% of the country**, and "
     "the only cell on this map that merges three traditions the tree keeps in three "
     "different places. Seventh-day Adventists belong to `christianity.adventist`, Baptists "
     "to `christianity.baptist`, and Malawi's Apostolic churches — the African Apostolic "
     "Church, the Apostolic Faith Mission and the Zion-adjacent bodies of the Shire valley "
     "— to `christianity.africaninstituted` or `christianity.pentecostal` depending on the "
     "body. **The census merged them and this node holds the merge**, because sending 1.6 "
     "million people to any one of those four asserts a division NSO did not make, and "
     "sending them to `christianity.other` would file them as bodies with no branch when "
     "the truth is that they have three. "
     "It holds an ANSWER rather than a church, like `christianity.protestant` and "
     "`christianity.evangelical` beside it, and like them it is deliberately NOT a parent "
     "of anything. "
     "**Why NSO grouped them is not recorded, and the grouping is not arbitrary**: all "
     "three are Sabbath-or-prophecy churches outside the Presbyterian/Anglican/Catholic "
     "mission settlement that shaped Malawi. Its geography backs that up — the cell is "
     "**29.1% in Neno, 21.5% in Thyolo, 21.0% in Mwanza and 17.6% in Chikwawa**, against "
     "9.4% nationally and 2.2% in Machinga: one contiguous block over the Shire highlands "
     "and valley, which is where the Seventh-day Adventist Malamulo mission has been since "
     "1902 and where the Apostolic churches are strongest. A merge of three unrelated "
     "things would look like noise; this does not. Read the cell as a category, not as an "
     "arithmetic sum. If a Malawian source ever separates them the rows still carry their "
     "`source_category` and can be moved (spec §2.4)."),

    ("indigenous.philippine",
     "Philippine indigenous religions",
     "The census's `Tribal religion`, 251,548 people, and one cell is all it gives them: "
     "the traditional religions of the Cordillera peoples of northern Luzon, the Lumad of "
     "Mindanao, the Mangyan, the Aeta and the Palawan groups are a single answer with no "
     "names under it. Compare India, which the tree runs five children deep because the "
     "census Appendix names 83 traditions. Depth follows what a source counts (spec §2.4), "
     "and the Philippines counts one."),

    # --- added 2026-09-07 with Myanmar, on indigenous.philippine's precedent: a national
    #     child where a source gives its own country's traditions exactly one cell.
    ("indigenous.myanmar",
     "Myanmar indigenous religions",
     "DOP's `Animist`, **408,045 people**, 0.8% of the enumerated population — and one cell "
     "is all the census gives them. **Its geography is almost entirely Shan State, which "
     "holds 383,072 of the 408,045 — 93.9% of the national figure and 6.6% of that state.** "
     "Kayah is next at 1.9%, and no other state reaches 0.2%. That is the eastern uplands and "
     "the traditional religions of the hill populations there, which the census does not name "
     "individually.\n\n"
     "**Read the figure as a floor**, for the reason `indigenous.african` carries for the "
     "whole of that continent: the box is exclusive of the Buddhist one, and **nat "
     "propitiation is close to universal in Myanmar and is practised alongside Buddhism "
     "rather than instead of it.** A census that offers one religion per person counts a nat "
     "shrine keeper who calls himself Buddhist as Buddhist. So this cell is the people for "
     "whom the traditional religion is the WHOLE answer, not the people who practise one.\n\n"
     "`Animist` is DOP's word and is kept in `source_category` per §2.4. §12's rule applies "
     "here — the same string means opposite things in different countries, and the units it "
     "sits in are what decide: this one is concentrated in the upland minority states, so it "
     "is an outsider's label for tribal religion and belongs on `indigenous`, not the "
     "Western neo-animist self-description that sends Czechia's to `paganism`. No child "
     "nodes: no source on this map names a Myanmar tradition individually."),

    # --- added 2026-09-08 with Vanuatu, on the indigenous.philippine / indigenous.myanmar
    #     precedent: a national child where a source gives its own country's traditions
    #     exactly one cell. Vanuatu's differs from both in one important way — see below.
    ("indigenous.vanuatu",
     "Vanuatu customary beliefs (kastom)",
     "VNSO's `Customary beliefs`, **9,080 people, 3.09%** — and unlike almost every other "
     "node under `indigenous` this one is **a printed census category standing beside the "
     "churches on equal terms, not an outsider's residual**. Vanuatu has counted it in every "
     "census since 1989 (6,484 → 10,365 → 8,600 → 9,080), so it is also the only indigenous "
     "religion on this map with a four-census time series.\n\n"
     "**Its geography is one island.** 7,757 of the 9,080 are in Tafea province — 17.3% of "
     "it — and within Tafea it is Tanna: **South West Tanna 30.3%, Middle Bush Tanna 25.3%, "
     "North Tanna 19.1%, West Tanna 16.7%**. That is the highest indigenous-religion share "
     "of any unit on this map outside India, and it is where the **John Frum** movement and "
     "the Prince Philip movement are. The census does not name either, so neither is a node: "
     "`kastom` is what it counts and kastom is what this holds.\n\n"
     "**Read it as a floor, for the reason `indigenous.african` and `indigenous.myanmar` "
     "carry.** The box is exclusive of the church boxes, and customary practice in Vanuatu "
     "runs alongside church membership rather than instead of it, so this is the people for "
     "whom kastom is the WHOLE answer and not the people who keep it."),

    # --- added 2026-09-08 with the Solomon Islands, on the same precedent as
    #     `indigenous.vanuatu`: a census that gives its own country's traditions one cell.
    ("indigenous.solomon",
     "Solomon Islands custom beliefs",
     "SINSO's `Custom Beliefs or Animism`, **4,115 people, 0.57%** — a printed census "
     "category standing beside the churches, as in Vanuatu, but a tenth of the share. The "
     "Solomons are 96% Christian on the 2019 count and this is what is left beside it.\n\n"
     "**It is not spread thinly; it is a few places.** `Waneagu/Taelanasina` in Malaita is "
     "21.5% and `Tetekanji` in Guadalcanal 21.2%, with `Gulalofou` (12.3%) and `Vulolo` "
     "(9.8%) behind them. That is the Kwaio interior of east Malaita and the Weather Coast "
     "of Guadalcanal, the two parts of the country where the missions reached least, and "
     "the ward tier is what makes them visible at all — at province level Malaita is 1.2%.\n\n"
     "**Read it as a floor**, for the reason `indigenous.african` and `indigenous.vanuatu` "
     "carry: the box is exclusive of the church boxes, so this counts the people for whom "
     "custom is the whole answer and not the much larger number who keep it alongside a "
     "church."),

    # --- added 2026-09-08 with Laos. UNLIKE every sibling above, no source PRINTS this
    #     category: it is the census's `no religion` cell, filed here on the evidence set
    #     out below. The argument is Anita's to reverse in one line of taxonomy/la2015.py.
    ("indigenous.laos",
     "Traditional religions of Laos",
     "**2,038,393 people, 31.45% of Laos, and the census calls this cell `no religion`.** "
     "It is the largest single decision on the Laos map and the only node on this map filed "
     "against its source's own English label, so the whole case is here.\n\n"
     "**The 2005 census defined religion as any spiritual system with WRITTEN DOCTRINES.** "
     "That is the *Socio-Economic Atlas of the Lao PDR*'s account of the instrument, in "
     "Section F.5: *\"According to this definition only Buddhism, Christianity, Baha'i and "
     "Islam are therefore identified as religions.\"* Animism was not measured and found "
     "absent; it was defined out of the category `religion` and had nowhere to go but the "
     "residual. The atlas then says what the residual is: *\"it might be suggested that a "
     "more appropriate term for the 'other' category would be Animism. The majority of "
     "non-Lao ethnic groups are essentially Animists.\"* The 2015 census kept the shape and "
     "renamed the cell; **the report's own summary calls it *\"no religion or being "
     "animist\"***, and the Lao subtitle on LSB's own map service reads *\"following other "
     "religions or not following any religion\"*.\n\n"
     "**Its geography is the opposite of irreligion's.** Dakcheung in Xekong is 96.5%, "
     "Samuoi in Salavan 93.1%, Ta Oi 92.1%, May in Phongsaly 92.0%. Vientiane Capital, the "
     "urban and educated end of the country and where a secular answer would concentrate, "
     "is **5.9%** and Champasak is **2.0%**. A category that runs 96% in the Katuic uplands "
     "and 2% on the Mekong is tracking the ethno-linguistic map, not schooling or "
     "urbanisation.\n\n"
     "**And it tracks it in the data as well as on the eye.** LSB publishes the ten "
     "ethno-linguistic categories on the same 8,499 villages, and the rate inside them is "
     "**9.2% for Lao-Tai against 65.1% Mon-Khmer, 79.1% Hmong-Mien and 77.4% "
     "Sino-Tibetan** — an ecological split, so read it as the shape and not as a "
     "measurement of individuals. The fact that needs no assumption at all: **84.5% of the "
     "2.04 million are in villages that are less than half Lao-Tai.**\n\n"
     "**The external estimate that names what the census will not is Pew's**, and it is "
     "unusually clean. *How the Global Religious Landscape Changed From 2010 to 2020* "
     "(2025) puts Laos's religiously unaffiliated at **under 0.1%** and its `other "
     "religions` at **34.2%, 2,510,000 people**, which makes Laos the tenth-largest `other "
     "religions` population in the world. That is spec §3.11's first bullet exactly: an "
     "external national estimate naming a category the census refuses to, and it bounds the "
     "genuinely non-religious part of this cell at a few thousand people nationally.\n\n"
     "**THE STRONGEST EVIDENCE CAME LAST AND IT IS THE 2005 CENSUS.** Anita asked whether "
     "31% irreligion is realistic at national level, and the previous census answers it: "
     "**2005 called this same cell `another religion` and it was 31.04%.** The box was "
     "renamed and the people did not change, which is checkable rather than asserted — the "
     "2005 `another religion` share against the 2015 `no religion` share, across the **137 "
     "districts present in both censuses, correlates at r = 0.9714 with the median district "
     "moving -0.0 points.** A census does not relabel a box and find the new label "
     "distributed exactly like the old one across 137 districts unless it is the same "
     "population, and in 2005 that population was recorded as *following another religion*. "
     "**And the urban gradient runs backwards inside Vientiane Capital**, the one place a "
     "secular population could be: Xaythany on the northern hill fringe is 11.99%, and "
     "**Chanthabuly, the historic city centre, is 2.41%** with Hadxaifong at 0.89%. A "
     "thirteen-fold gradient inside one municipality, at its minimum in the core.\n\n"
     "**What the node does NOT claim.** No tradition is named under it, because no source "
     "names one: this is the Khmu, Hmong, Akha, Katu, Ta Oi, Brao and Lamet religions and "
     "several dozen others in one box, and depth follows what a source counts (§2.4). "
     "**Read it as a ceiling rather than a floor**, which reverses the reading every other "
     "node in this family carries: `indigenous.myanmar` and `indigenous.vanuatu` are boxes "
     "that stood BESIDE the churches, so they undercount people who keep both, while this "
     "one is a residual that also holds however many Lao really do report no religion. The "
     "5.9% in Vientiane Capital is where to look for them.\n\n"
     "**The alternative was `unknown`** (§6.3a-ii), which Vietnam uses next door for a "
     "residual of the same kind, and it was rejected because the evidence above is a good "
     "deal more specific than *\"what they practise is not determinable\"* — four "
     "independent sources agree on what the cell is, one of them being the census's own "
     "report. Filing it there would render the Lao uplands as one grey block and say "
     "nothing about a religious geography that is among the sharpest on this map."),

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

    ("modekngei",
     "Modekngei",
     "**Palau's own religion, and the only one on this map indigenous to Micronesia.** "
     "Founded around 1915 by Temedad on Babeldaob, it fuses Palauan *chelid* spirit belief "
     "with Christian elements and a healing practice, and was suppressed by the Japanese "
     "administration in the 1930s and 40s. **1,733 people in the 2005 census, 8.7% of "
     "Palau**, which makes it a larger share of its country than any other indigenous "
     "religion drawn here. "
     "**A root rather than a child of `indigenous`, on caodaism's precedent** (spec §3.3): "
     "it is a named, organised, founded religion whose syncretism IS the tradition, not a "
     "diffuse body of traditional practice like `indigenous.philippine` or "
     "`indigenous.myanmar`. Filing it under `indigenous` would group it with things it does "
     "not resemble; splitting it across Christianity and Palauan tradition would describe "
     "nobody."),

    ("christianity.reformed.congregational.cicc",
     "Cook Islands Christian Church",
     "The national church of the Cook Islands and the direct descendant of the London "
     "Missionary Society mission of 1821. **7,356 people, 49.1% of the country**, the "
     "largest single body there. Filed under `congregational` with the LMS's other Pacific "
     "daughters rather than at the parent, because these are three distinct national "
     "churches in three countries and nothing else on the map files them elsewhere; "
     "ge2014.py's caution about adding a child for one country does not bite here, since no "
     "other source counts these bodies at all."),

    ("christianity.reformed.congregational.ekt",
     "Ekalesia Kelisiano Tuvalu",
     "The Congregational Christian Church of Tuvalu, LMS-descended by way of Samoan "
     "missionaries from 1861, and the established church under Tuvalu's constitution. "
     "**9,023 people, 85.9% of the country — the largest share any single church holds in "
     "any country on this map.** Its sibling is the Cook Islands Christian Church, and "
     "Samoa's own CCCS would be a third if Samoa is drawn."),

    # --- added 2026-09-08 with Samoa, which `.ekt`'s note above had already named as the
    #     fourth of this set: "Samoa's own CCCS would be a third if Samoa is drawn".
    # --- added 2026-09-08 with Kiribati, the fifth and last of the Pacific Congregational set.
    ("christianity.reformed.congregational.kpc",
     "Kiribati Protestant Church",
     "**34,464 people, 31.3% of Kiribati at the 2015 census**, and the second body in the "
     "country after the Catholics. Founded as the Gilbert Islands Protestant Church out of "
     "**two Congregational missions rather than one**: the American Board of Commissioners "
     "for Foreign Missions from 1857, by way of Hawaiian pastors, and the London Missionary "
     "Society from 1870, by way of Samoan and Tuvaluan ones. Both parents are Congregational, "
     "which is what puts it here beside `.cccs`, `.cicc`, `.ekt` and `.niue` rather than on a "
     "node of its own kind. Renamed from Gilbert Islands to Kiribati Protestant Church in "
     "1979 with the country.\n\n"
     "**Its geography is the mission partition of the Gilberts and it is still almost "
     "perfect.** The chain runs Catholic in the north and Protestant in the south, and the "
     "two ends invert: **Butaritari is 82.5% Catholic and 13.2% Protestant; Arorae, 600 km "
     "south, is 98.0% Protestant and 1.4% Catholic.** Tamana beside it is 95.8%. Nothing in "
     "between is mixed by accident — the share falls almost monotonically down the chain.\n\n"
     "**THE 2014 UNION IS NOT IN THIS NODE AND THE CENSUS YEAR IS WHY.** In 2014 the church "
     "reconstituted itself as the **Kiribati Uniting Church**, a union of Congregationalists, "
     "Evangelicals, Anglicans and Presbyterians, and roughly ten thousand members — mainly "
     "Congregationalists — refused and re-formed a separate Kiribati Protestant Church. The "
     "2015 census still counts one cell called `KPC`; the 2020 census counts KUC at 21% and "
     "KPC at 8% separately. This map draws 2015, because 2015 is the newest year Kiribati "
     "publishes religion with a geography, so it draws the body whole. `christianity.united` "
     "would be the right home for the KUC half if a year that separates them is ever drawn."),
    ("christianity.reformed.congregational.cccs",
     "Congregational Christian Church of Samoa",
     "The Ekalesia Fa'apotopotoga Kerisiano Samoa, **55,411 people and 27.0% of Samoa** — "
     "the largest of the London Missionary Society's Pacific daughters and the mother of "
     "the rest of them. John Williams landed the LMS mission at Sapapali'i on Savai'i in "
     "1830, and the Samoan teachers it trained carried it on to the Cook Islands, Tuvalu, "
     "Niue and the Gilberts, which is why `.cicc`, `.ekt` and `.niue` are its siblings here "
     "rather than its parents.\n\n"
     "**Malua, its theological college village on Upolu, is 100.0% Congregational** — 424 "
     "of 424 people, with not one person in any other category. It reaches 55.2% of "
     "Aleipata Itupa i Luga and 50.0% of Lepa, and it is the largest body in every one of "
     "Samoa's four statistical regions."),
    ("christianity.reformed.congregational.niue",
     "Ekalesia Niue",
     "The Congregational church of Niue, LMS-descended from 1846 through Samoan and "
     "Rarotongan teachers. **981 people, 61.7% of Niue**, which is the smallest population "
     "any node on this map is built from."),

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
     "political programme is not a stated non-theistic position. "
     "WIDENED 2026-09-07 for §7a-i-1: as a roll-up target it stands for Stats NZ's RESIDUAL "
     "GROUPS generally — the ones tabulated apart from the six religions the SA2 table names "
     "— so `Spiritualism and New Age Religions` (21,180) rolls here too. That group spans "
     "Spiritualist, Pagan, Wiccan, Rastafari and Scientologist with no family covering it, "
     "and this node claims only what is true of it: Stats NZ did not put these people with "
     "the named religions. `Māori Religions, Beliefs and Philosophies` pointedly does NOT "
     "roll here — see nz2023.COLUMNS."),
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
    # ---- THE MICROSTATE TIER, all nine from UNSD table 28 (sources/micro.py) ------------
    # Every one of these is a residual the Demographic Yearbook prints, at national level,
    # in a country small enough that the whole map is 20 to 80 dots. They are grouped here
    # rather than scattered because they share one instrument and one caveat: the DYB's
    # category names are the OFFICE'S, but the classification into them is UNSD's, so an
    # `Other` here is not necessarily the office's own word.
    ("other.pw",
     "Other religion (Palau)",
     "1,613 people, 8.1%, beside a list that already names Modekngei, the Adventists, the "
     "Witnesses and the Latter-day Saints separately. Palau's 2005 tail is large for a "
     "country of 20,000 and is not opaque so much as unenumerated: the census names nine "
     "categories where the 1995 one named twelve, and the Baha'i, Assembly of God and "
     "Church of Christ counted in 1995 are inside this cell in 2005."),
    ("other.ck",
     "Other religion (Cook Islands)",
     "`Other Religions`, 1,201 people, 8.0%, beside seven named churches. Distinct from "
     "`Unknown` (323), which the same table keeps separate and which is off the tree."),
    ("other.tv",
     "Other religion (Tuvalu)",
     "270 people, 2.6%, in a country where one church holds 85.9%. The named tail around "
     "it is unusually specific for its size: Brethren, Adventists, Baha'i, Assembly of God, "
     "Witnesses, Latter-day Saints and 53 Catholics."),
    ("other.nu",
     "Other religion (Niue)",
     "131 people, 8.2%. Niue's whole population is 1,591, so this node is built from fewer "
     "people than any other on the map and draws nothing at 1 dot = 1,000; it exists so the "
     "category is not silently dropped."),
    ("other.sb",
     "Other religion (Solomon Islands)",
     "SINSO's `Other religions`, **14,953 people, 2.1%**, printed beside thirteen named "
     "bodies plus `Custom Beliefs or Animism`, `No Religion or Atheism` and a refusal cell. "
     "The census names the Baha'i Faith (3,104) and Muslims (1,100) separately, which most "
     "censuses on this map do not, so this residual is genuinely what was left after a long "
     "list rather than a place where the non-Christian religions were put. Its geography is "
     "unremarkable, which is the tell that it is a tail and not a hidden body."),
    ("other.vu",
     "Other religion (Vanuatu)",
     "**35,270 people, 12.0% — the third-largest cell in Vanuatu, ahead of the Anglicans.** "
     "Table 3.5 heads the column `Other churches`, which would put it on `christianity`; "
     "Volume 2 of the same census heads it plainly `Other` and then says what is in it: "
     "**\"the category 'Other' includes 88 different religions ranging from one member to "
     "more than 2,000 members\"**. *Religions*, not churches, and 88 of them. So the "
     "Christian reading is the volume-1 header's and is not supported by the volume that "
     "describes the contents, and the cell goes where a residual that cannot be shown to be "
     "Christian belongs. §14.4 rule 1: the Baha'i, the Muslims and the Jehovah's Witnesses "
     "Vanuatu certainly has are inside this number, and nothing published says in what "
     "proportion. The one body big enough to name is unnamed too — 'more than 2,000 "
     "members' is the largest of the 88 and the census does not say which it is."),
    ("other.ms",
     "Other religion (Montserrat)",
     "`Other Religions`, 251 people, 5.8%, in the 2001 census taken six years after the "
     "Soufriere Hills eruption that removed two thirds of the island's population."),

    # --- added 2026-09-08 with Laos, on the other.mu / other.sk pattern: a printed row
    #     that pools an answer with a non-answer.
    ("other.la",
     "Other or not stated (Laos)",
     "**133,296 people, 2.06%, and it is a residual of a residual.** LSB's Table 3.5 prints "
     "one row, `Others/not stated`, at 137,640; the village-level services separate the "
     "Muslims (1,603) and the Baha'is (2,121) out of it, and what is left is this. It is "
     "not published as a category anywhere at any geography, and `sources/la.py` derives it "
     "as each village's population less its five published categories.\n\n"
     "**Its shape says non-response rather than religion.** It is 1-5% in 3,934 of the "
     "8,499 villages and under 0.1% in 944 of them, which is a thin national film and not a "
     "community anywhere; only 12 villages are more than half, and all twelve are small. "
     "Compare `indigenous.laos` beside it, which runs 96% in one district and 2% in "
     "another. **So read this as the census's own reach and not as Laos's other faiths**, "
     "the more so because the four religions the instrument recognised are all drawn "
     "separately already (§3.11, and the same call as `other.mu`).\n\n"
     "Per source, and never merged with another country's residual."),
    ("other.bm",
     "Other religion (Bermuda)",
     "**Two columns, not one**: `Other Religions` (4,399) and `Other` (1,527), which the "
     "Demographic Yearbook prints side by side without saying how they differ. Both are "
     "residuals and both land here, 5,926 people or 9.2%. Bermuda's named list is the "
     "deepest in this tier at 23 categories and already separates the Muslims, Jews, "
     "Baha'is, Rastafari and Ethiopian Orthodox, so whatever the distinction was, it is not "
     "one of those."),
    ("other.ag",
     "Other religion (Antigua and Barbuda)",
     "116 people, 0.15%, and the smallest residual in this tier as a share, because the "
     "2001 census names 21 categories including the Moravians, the Spiritualists and the "
     "Rastafari separately."),
    ("other.dm",
     "Other religion (Dominica)",
     "252 people, 0.37%. Kept apart from `Other Evangelical Churches` (4,882), which names "
     "a family and goes to `christianity.evangelical` rather than here."),
    ("other.mh",
     "Other religion (Marshall Islands)",
     "5,632 people, **11.1%, and by far the least satisfying node in this tier**: the 1999 "
     "census publishes four categories in total, so this cell is everything that is not "
     "Protestant, Assembly of God or Catholic. It certainly contains the country's "
     "Bukot nan Jesus and Baha'i communities and probably its Latter-day Saints, none of "
     "which the source names. Read Marshall Islands' colours as a four-way split and "
     "nothing finer."),
    ("other.py",
     "Other religion (Paraguay)",
     "Two of CPV2002's fifty-four categories, kept together because neither names a body: "
     "`Relig. no incluidas en las anteriores` (1,208) and `Otra religion No Especificada` "
     "(6,139), 7,347 people between them and **0.19% of the universe**. That is a small "
     "residual by the standards of this tier, and the reason is the length of the list it "
     "is left over from: Paraguay names fifteen Protestant bodies, three Orthodox cells, "
     "Umbanda, Reyukai, Shinto and the Bahai separately, so what falls out of the bottom "
     "is genuinely a tail. The cell that deserves suspicion in Paraguay is not this one, "
     "it is `Otras - Evangelica` at 186,107 (taxonomy/py2002.py)."),
    # --- added 2026-09-08 with Türkiye.
    ("other.tr",
     "Other religion or none (Türkiye)",
     "**The odd one out in this family, and the name says why: it is not only a religion "
     "cell.** The Diyanet's question 10 offers Islam, Christianity, Judaism, other religions "
     "and 'I belong to no religion', and the report publishes the answers as three numbers — "
     "İslamiyet 99.2%, this 0.4%, no answer 0.5% — describing the middle one in its own words "
     "as *belongs to a religion other than Islam or belongs to no religion*. So one cell of "
     "about 341,000 people holds Türkiye's Christians, its Jews and its irreligious together, "
     "and nothing in 293 pages separates them. "
     "**It is also national.** There is no regional religion table anywhere in the report, so "
     "this share sits at the same rate in all twelve regions and carries no geography at all; "
     "it must not be read as saying Türkiye's Christians are evenly spread, when in fact "
     "almost all of them are in Istanbul. KONDA's independent 2006 national survey splits the "
     "same space as Orthodox 0.06%, Catholic 0.01%, Protestant and others 0.057%, Jewish "
     "0.013%, other religion 0.04% and no religion 0.47% — which is not drawn here, because "
     "mixing two instruments to decompose a cell is exactly what §14 rule 1 forbids."),
    ("other.bg",
     "Other religion (Bulgaria)",
     "NSI's `Друго` — 6,451 people, 0.10%, and a genuine tail rather than a coarse cell: "
     "the 2021 census puts it beside Judaism and beside four named Christian bodies, so it "
     "is what is left after those and not a bucket standing in for them. Bulgaria's real "
     "opacity is elsewhere and is much larger, in a `Християнско` column that merges "
     "4.2M Orthodox, Protestants, Catholics and Armenians at municipal level "
     "(taxonomy/bg2021.py)."),
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
    ("other.md",
     "Other religion (Moldova)",
     "BNS's `Alte religii` — 4,720 people, 0.20%, at the 2024 census. Wider than Romania's "
     "neighbouring residual, because the 2024 list is shorter than the 2014 one: the 2014 "
     "census named Judaism and the Lutheran church of the Augsburg Confession as their own "
     "categories and 2024 does not, so both are inside this cell along with everything "
     "else outside the nine bodies it does name. Per source, per spec §3.11."),
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

    # --- added 2026-09-04 with Chile.
    ("other.cl",
     "Other religion (Chile)",
     "INE's `Otras religiones o credos` — 89,856 people, 0.59% of the 15+ universe. Unusual "
     "among the residuals here in having been REDUCED by the source before publication: it "
     "is question 31's option 11, a free-text write-in, minus the answers INE then recoded "
     "into `Otros cristianos y tradiciones relacionadas con Cristo` (199,426). So what is "
     "left is genuinely non-Christian and genuinely unclassified, rather than the usual "
     "mixture of the two. Chile's Mapuche religion is somewhere inside it, and there is no "
     "way to bring it out: the census asks about religion and about indigenous belonging in "
     "separate questions and does not cross them."),

    # --- added 2026-09-04 with Sri Lanka.
    ("other.lk",
     "Other religion (Sri Lanka)",
     "DCS's `Other` — 63,494 people, 0.29% — and the only residual on this map that is NOT "
     "purely a residual. The 2024 census moves any religion with fewer than 10 people in a "
     "GN division into this cell, so it holds genuine other-religion answers mixed with "
     "the suppressed tail of Buddhist, Hindu, Islam, Roman Catholic and Other Christian, "
     "in an unknown proportion. Sri Lanka's real small groups — the Malay Muslims, the "
     "Bahá'ís, the Parsis of Colombo, the Veddas' own practice — are somewhere inside it "
     "and cannot be brought out at this geography. That the bucket is this small is a fact "
     "about the question, which offers six answers, and not about the country. See spec "
     "§3.8 and sources/lk.py."),

    # --- added 2026-09-04 with the Philippines.
    ("other.ph",
     "Other religion (Philippines)",
     "PSA's `Other religious affiliations` — 1,893,134 people, 1.74%, and the fourth "
     "largest category in the country. It is a residual of an unusually lopsided kind. "
     "The census names 129 bodies and 126 of them are Christian; the entire non-Abrahamic "
     "world gets three cells — Islam, Buddhist, and Tribal religion — and there is no "
     "Hindu, Jewish, Sikh, Chinese folk or Bahá'í option anywhere on the form. So every "
     "Filipino Hindu, every Chinese Filipino who practises the folk religion, and every "
     "Jew in the country is inside this number, together with the tail of small Christian "
     "bodies that did not make the list of 129. The bucket is not large because the "
     "country is unclassifiable; it is large because the question was written for "
     "Philippine Christianity and asked in four levels of detail there and none anywhere "
     "else (sources/ph.md §7). Per source, per spec §3.11."),

    # --- added 2026-09-05 with Ghana.
    ("other.gh",
     "Other religion (Ghana)",
     "GSS's `Other Religion` — 328,721 people, 1.07%. Everything that is not one of the "
     "four Christian boxes, Islam, Traditionalist or No Religion, which in Ghana means the "
     "Bahá'í community, the Hindu Monastery of Africa at Odorkor, the Buddhists, the Jewish "
     "community of Sefwi Wiawso, Eckankar and the Rastafari. None of them is separable at "
     "any geography: the form has one cell for all of it. Per source, per spec §3.11."),
    ("other.ke",
     "Other religion (Kenya)",
     "KNBS's `Other Religion` — 467,083 people, 0.99%. Smaller in scope than most residuals "
     "here, because Kenya's list is unusually generous at the top: Hindus, Traditionists and "
     "the Orthodox all have cells of their own, so this one is not carrying them. What is "
     "inside is the Sikhs and Jains of Nairobi and Kisumu, the Bahá'ís, the Buddhists, and "
     "the tail of small movements. Per source, per spec §3.11."),

    # --- added 2026-09-08 with Mongolia.
    ("other.mn",
     "Other religion (Mongolia)",
     "NSO's `Бусад` (Other) -- 13,699 people, 0.66% of the population aged 15 and over. The "
     "2020 census offers six answers and five of them are named, so this is the sixth box "
     "and there is nothing behind it in the published tables: no volume breaks it down and "
     "the question has no write-in. What is in it is the Bahá'í community, a small Mormon "
     "and Jehovah's Witness presence built since 1990, and the tail. "
     "It is not evenly spread, which is the one interesting thing about it: 3.5% of Uvs's "
     "religious population and 2.7% of Dornogovi's against 0.1% in Övörkhangai. On a cell "
     "this small at a 10% sample those differences are within reach of sampling noise and "
     "should not be read as a map of anything."),

    # --- added 2026-09-05 with Russia.
    ("other.ru",
     "Other religion (Russia)",
     "Two of Arena's answers, 875,000 people between them and neither placeable. `Other` "
     "(0.61%) is the survey's own residual. The second is `I follow Eastern religions and "
     "spiritual practices (Hinduism, Krishnaism, other)` (0.06%, 90,990 people) and it is "
     "filed here for exactly the reason hr2021.py files Croatia's `Istočne religije` here: "
     "it plainly means Hinduism, Buddhism and their neighbours, the tree has no node for "
     "'some Eastern religion, unspecified', and choosing one of them would be inventing a "
     "fact. Note that Arena asks about Buddhism separately and 659,032 people took it, so "
     "this category is the remainder AFTER the one Eastern religion Russia has in numbers "
     "— which makes 'Krishnaism' in the answer text a fair guide to what is actually in "
     "it. Per source, per spec §3.11."),

    # --- added 2026-09-05 with Lithuania.
    ("other.lt",
     "Other religion (Lithuania)",
     "Statistics Lithuania's `Kitų` — 15,353 people, 0.55%, and unusually large for a "
     "residual on a form this detailed. Roman and Greek Catholic, Orthodox, Old Believer, "
     "Lutheran, Reformed, Baptist, Pentecostal, Adventist, New Apostolic, Sunni, Jewish and "
     "Karaim all have cells of their own, so this one is what is left after sixteen "
     "answers: the Baltic-faith revival (Romuva, state-recognised since 2025 after two "
     "decades of refusals), the Jehovah's Witnesses, the Latter-day Saints, and the "
     "Buddhist and Hindu communities of Vilnius. None is separable at any geography. Per "
     "source, per spec §3.11."),

    # --- added 2026-09-05 with Serbia.
    ("other.rs",
     "Other religion (Serbia)",
     "RZS's two smallest cells put together — `Источњачке вероисповести` (Eastern "
     "religions, 1,207) and `Остале вероисповести` (other religions, 500). 1,707 people, "
     "0.026%, the thinnest tail of any country on this map. Orthodox, Catholic, "
     "Protestant, other Christian, Muslim and Jewish all have cells of their own, so this "
     "one carries Serbia's Buddhists and Hindus, its Bahá'ís, and whatever else the form "
     "left nowhere else to go. Per source, per spec §3.11."),
    ("other.id",
     "Other religion (Indonesia)",
     "BPS's `Lainnya` — 292,889 people, 0.12%. **The most understated residual on this "
     "map, and knowably so.** Indonesia's census offers the six religions the state "
     "recognises and this one cell for everything else, so `Lainnya` has to hold Judaism, "
     "Sikhism, Bahá'í and Shinto TOGETHER WITH the whole of Aliran Kepercayaan — the "
     "indigenous belief systems, Kejawen, Sunda Wiwitan, Parmalim, Kaharingan, Marapu. In "
     "2010 those had no legal standing on the form (registration came only with the 2017 "
     "Constitutional Court ruling), so their adherents overwhelmingly recorded one of the "
     "six instead; Kaharingan was administratively counted as Hinduism outright, which is "
     "most of why Central Kalimantan draws Hindu. The number is therefore a FLOOR by a "
     "large and unknown margin, in the same way every African `Traditionalist` cell is "
     "(sources.md §11b). Not mapped to an indigenous node despite that, because the cell "
     "is genuinely mixed and the source gives no composition. Per source, per spec §3.11."),
    ("other.kr",
     "Other religion (South Korea)",
     "KOSIS's `기타` — 98,185 people, 0.20%. Small, because the list above it is unusually "
     "generous: Buddhism, both Christianities, Won Buddhism, Confucianism, Cheondogyo, "
     "Daesun Jinrihoe and Daejonggyo all have cells of their own, so this residual is not "
     "carrying any of the things a Korean religion map is about. What is in it is the rest "
     "of the Jeungsanist and Donghak tail, the Unification Church (which the census does not "
     "name), Islam, and the small foreign-resident communities. Per source, per spec §3.11."),
    ("other.gy",
     "Other religion (Guyana)",
     "The Bureau of Statistics' `Other` — 6,324 people, 0.85%. Unusually SMALL for a "
     "residual, because the thirteen categories above it are unusually generous: Hindus, "
     "Muslims, Rastafarians and Bahá'ís all have cells of their own, so this one is not "
     "carrying any of them. What it does carry is the Chinese community's Buddhists and "
     "the tail of small bodies. **What it does not carry, and should, is Amerindian "
     "traditional religion** — the 2012 form offers no box for it, so the nine Amerindian "
     "nations of Regions 1, 7, 8 and 9 answered one of the Christian categories instead. "
     "That is Ghana's exclusive-category undercount (sources.md §11b) in South America, "
     "and it is why Guyana's interior draws almost entirely Catholic and Anglican. Not "
     "corrected, because correcting it would mean inventing a magnitude (§14.4). Per "
     "source, per spec §3.11."),
    ("other.et",
     "Other religion (Ethiopia)",
     "The 2007 census's `Other` — 470,682 people, 0.64%. The residual of a six-cell "
     "question, so it is doing more work than its size suggests: with only Orthodox, "
     "Protestant, Catholic, Islam and Traditional above it, this one cell holds Ethiopia's "
     "**Beta Israel** remnant, the Jehovah's Witnesses, the Bahá'ís, the Latter-day Saints "
     "and the small Hindu and Buddhist communities of Addis Ababa — none of them separable "
     "at any geography, because the form has one box for all of it. Unlike Kenya's and "
     "Guyana's residuals, which are small because the lists above them are generous, this "
     "one is small despite a short list, which is the more interesting fact: it says the "
     "five named categories really do cover Ethiopia. Per source, per spec §3.11."),
    ("other.pk",
     "Other religion (Pakistan)",
     "PBS's `Others` — 43,253 people, 0.021%, and **the smallest residual on this map by an "
     "order of magnitude**, both absolutely against the country's size and as a share. That "
     "is not because the 2017 form was generous: it offers six cells and no `not stated` at "
     "all. It is what a residual looks like when 96.5% of a country answers the first box. "
     "Inside it are Pakistan's Sikhs, Parsis, Bahá'ís, Buddhists, Kalasha and Jews — the "
     "Sikhs and Parsis being the notable loss, since both are historic Pakistani "
     "communities with real geographies (Nankana Sahib, Peshawar; Karachi) that this cell "
     "cannot show. **The 2023 census fixed exactly that** and gives Sikh and Parsi cells of "
     "their own, so this residual is a fact about the 2017 form rather than about Pakistan; "
     "see sources/pk.md §6. Per source, per spec §3.11."),

    # --- added 2026-09-06 with Bangladesh.
    ("other.bd",
     "Other religion (Bangladesh)",
     "BBS's `Other religion` — 202,167 people, 0.14%, the residual of a five-cell question "
     "and the smallest share of any residual on this map bar Pakistan's. **Its geography is "
     "the interesting thing and it is not evenly spread at all.** Nationally 0.14%, but it "
     "reaches **15.3% in Ruma upazila**, 7.8% in Thanchi and 5.1% in Rowangchhari — all in "
     "the Chittagong Hill Tracts, all among the least Muslim units in the country, and all "
     "sitting beside the Christian and Buddhist peaks rather than instead of them. A "
     "five-cell question cannot express what that is, and the near-certain answer is the "
     "**indigenous religion of the Hill Tracts peoples** — the Mru, Khumi, Khyang, Bawm and "
     "Pankhua traditions, which are neither of the two Buddhisms nor Christianity and have "
     "no box on the form. This is the same pattern §9u found in Bore woreda, where "
     "Ethiopia's `Other` spikes to 21.8% and is almost certainly Waaqeffanna: **a residual "
     "with a sharp geography is a missing category, not a mixture.** Elsewhere the cell "
     "holds Bangladesh's Bahá'ís, its small Sikh and Jain communities, and the Santal and "
     "Oraon traditional practice of the northwest — Patnitala in Rajshahi at 3.9% is that "
     "one showing through. None of it is separable at any geography. Per source, per spec "
     "§3.11."),

    # --- added 2026-09-06 with Portugal.
    ("other.pt",
     "Other non-Christian religion (Portugal)",
     "INE's `Outra não cristã` — 24,366 people, 0.28%. The residual of the NON-CHRISTIAN "
     "half of an eleven-cell question, which is a narrower job than most residuals here "
     "have: Buddhists, Hindus, Jews and Muslims each have a cell, so this one is not "
     "carrying any of them. **And its geography is sharp enough to name what is in it.** "
     "Nationally 0.28%, but 2.99% in Odemira and 3.84% in the freguesia of São Teotónio — "
     "the same Alentejo coastal strip where the Hindu share reaches 17.1% and the Buddhist "
     "9.3%, and nowhere else in the country comes close. That is the intensive "
     "berry-and-greenhouse belt and its South Asian and Southeast Asian workforce, so the "
     "near-certain content is **Sikhism**, which has a real Portuguese community (Lisbon "
     "and the Odemira farms) and no box on the form — with Bahá'ís and the Chinese "
     "community's folk practice behind it. §9u's Bore woreda and §9r's Chittagong Hill "
     "Tracts again: **a residual with a sharp geography is a missing category, not a "
     "mixture.** Not mapped to Sikhism despite that, because the cell is genuinely mixed "
     "and INE publishes no composition (§14.4). Per source, per spec §3.11."),
    ("other.xk",
     "Other religion (Kosovo)",
     "ASK's `Others` — 7,175 people, 0.45%, the residual of a four-religion question. With "
     "Islam, Orthodoxy and Catholicism named and nothing else, this cell carries **the "
     "Kosovo Protestant Evangelical Church** — the country's fourth legally recognised "
     "community, and the only substantial body with no box of its own — together with the "
     "Bahá'ís and the small Jewish community of Prizren. Its geography is unremarkable "
     "(Prishtinë 1.14%, Suharekë 1.07%), which by §9r's rule is itself the finding: a "
     "residual spread evenly is a mixture rather than a missing category, unlike "
     "Bangladesh's and Ethiopia's. Per source, per spec §3.11."),
    ("other.ge",
     "Other religion (Georgia)",
     "Geostat's `Other` — 1,429 people, 0.04%, **the smallest residual on this map in both "
     "absolute and relative terms**, and it is small because the list above it is unusually "
     "generous for a twelve-cell form: Orthodox, Muslim, Armenian Apostolic, Catholic, "
     "Jehovah's Witnesses, Yazidi, Protestant and Jewish all have cells of their own. What "
     "is left is Georgia's Bahá'ís, its Hare Krishna community, and the tail. One of the "
     "eleven regions has it withheld as '0-10' — Geostat's disclosure control, bounded by "
     "its own codelist, so the shortfall is at most ten people. Per source, per spec "
     "§3.11."),

    # --- added 2026-09-06 with Spain.
    ("other.es",
     "Believer of another religion (Spain)",
     "CIS's `Creyente de otra religión` — one cell for every religion that is not "
     "Catholicism, offered to Spanish citizens in every monthly barómetro, and never "
     "followed by a question asking which. **The widest residual on this map after "
     "Vietnam's, and the only one that is a residual of a single named religion rather "
     "than of a list.** 3.2% of Spanish citizens, ~1.36 million people. It does not "
     "arrive here whole: sources/es.py first takes UCIDE's province-level count of "
     "Spanish-citizen Muslims out of it, because leaving Ceuta and Melilla — majority "
     "Muslim and majority Spanish-citizen — drawn as an unnamed grey would be a false "
     "statement about the two most Muslim places in Spain rather than an honest "
     "silence. What remains is Spain's own evangelical Protestants, the Orthodox who "
     "have naturalised, roughly 110,000 Jehovah's Witnesses, ~45,000 Jews and a "
     "Buddhist tail — and none of it is separable, because the question that would "
     "separate it is not asked. Its geography is flat once the Muslim share is out, "
     "which by §9r's rule is the finding: this part of it is a mixture, not a missing "
     "category. Per source, per spec §3.11."),

    # --- added 2026-09-06 with Greece.
    ("other.gr",
     "Other religion (Greece)",
     "Two ESS answers together — `Eastern religions` and `Other Non-Christian religions` — "
     "which is 0.2% of Greek citizens and a few dozen respondents across three pooled "
     "rounds. The tree has no node for 'some Eastern religion, unspecified' and choosing one "
     "would invent a fact, so both sit here, as ru2012.py and hr2021.py do for the same "
     "shape. **Its content is largely knowable even though it is not separable**: Greece's "
     "Bahá'ís, its small Hindu and Sikh communities from the South Asian labour migration to "
     "Attica, and the Hellenic-polytheist revival, which is legally recognised and is the "
     "one of the three that is specific to this country. Per source, per spec §3.11."),

    # --- added 2026-09-06 with Bosnia and Herzegovina.
    ("other.ba",
     "Other religion (Bosnia and Herzegovina)",
     "BHAS's `Ostali` — 40,655 people, 1.15%, the residual of a question that names only "
     "Islam, Catholicism and Orthodoxy. That is a narrow list for a European census, so "
     "this cell is doing more work than most residuals here: it holds **the Jewish "
     "community of Sarajevo**, which is Sephardic and dates from the 1560s and whose "
     "surviving few hundred are among the oldest continuously present communities in the "
     "Balkans; the Protestant and evangelical bodies, the Adventists and the Jehovah's "
     "Witnesses; and everyone who wrote in something the three boxes could not take. "
     "**Its geography is sharp, and by §9r's rule that means a missing category rather "
     "than a mixture — but this one does not resolve.** Velika Kladuša is **7.42%** "
     "`Ostali`, 2,998 people and six times the national rate, with Tuzla second at 3.51%; "
     "no other unit above 2,000 people passes 3.3%. Velika Kladuša is a 96%-Muslim "
     "municipality in the Cazin Krajina, so the peak is not a Christian minority, and "
     "BHAS publishes no composition of the cell at any geography. **Named here as an open "
     "question rather than guessed at (§14.4)**: something specific is in it and this "
     "source cannot say what. Per source, per spec §3.11."),

    # --- added 2026-09-06 with Montenegro.
    ("other.me",
     "Other religion (Montenegro)",
     "MONSTAT's `Ostale vjere` — **310 people, 0.05%, the smallest per-source residual on "
     "this map** and smaller in relative terms than Georgia's, which held the record. It is "
     "small because the 2023 form is generous for a country of 620,000: Orthodoxy, "
     "Catholicism, Islam, Protestantism, Jehovah's Witnesses, other Christian, Buddhism, "
     "atheist and agnostic all have cells of their own, so a Montenegrin has to be some "
     "distance from all of those to land here. What is in it is the Jewish community — "
     "recognised by the state in 2012 and numbering a few hundred — the Bahá'ís, and the "
     "tail. **Read it against the suppression rather than on its own**: `Ostale vjere` is "
     "withheld as `z` in 17 of the 23 municipalities, so the 310 drawn is a floor and the "
     "true figure is unknowable from this table. A residual this small in a country with "
     "this much disclosure control is measuring the control as much as the tail. Per "
     "source, per spec §3.11."),

    # --- added 2026-09-08 with Slovakia.
    ("other.sk",
     "Other or not stated (Slovakia)",
     "**THE LABEL IS DELIBERATELY NOT 'OTHER RELIGION', AND THIS NODE IS NOT COMPARABLE "
     "WITH ITS SIBLINGS.** Every other `other.<cc>` on this map is a residual of religions "
     "somebody named. Slovakia's `ostatné` is a residual of religions AND of people who did "
     "not answer, merged before publication, and the non-response is the large half: UNSD "
     "table 28 publishes the same census with 21 categories and puts `nezistené` at "
     "**353,797 — 83% of this cell** — against roughly 72,699 in named and unnamed "
     "religions. 426,496 people, 7.83% of the country. Anita's call, 2026-09-08: drawn "
     "rather than dropped, because the alternative left 7.8% of Slovakia as a hole in the "
     "map and a hole reads as an absence of people (§6.12). "
     "**AND THE MIX IS NOT CONSTANT, WHICH IS WHY THE NATIONAL SPLIT CANNOT BE APPLIED PER "
     "UNIT.** The cell runs from 0.0% to 58.8% between municipalities (median 3.8%, p90 "
     "8.9%), and the top of the list is Slovakia's Roma settlements and its two city "
     "centres — Košice-Luník IX **58.8%**, Pavlovce nad Uhom 28.7%, Jasov 26.9%, Bratislava-"
     "Staré Mesto 16.4% — against 0.7-2.7% in the Orava and Kysuce villages. Those are "
     "non-response geographies, not unusual-religion geographies, so this node is mostly "
     "'did not answer' where it is large and mostly 'some other faith' where it is small. "
     "**Do not read its density as a religious fact**; read it as the census's own reach. "
     "Per source, per spec §3.11, and never merged with another country's residual."),

    # --- added 2026-09-06 with Malawi.
    ("other.mw",
     "Other religion (Malawi)",
     "NSO's `Other Denomination` — 992,304 people, 5.65%, and **the only residual on this "
     "map whose contents are partly known and still cannot be drawn apart.** Table E5 gives "
     "it as one column at district; Table 3.4, on page 19 of the same report, splits the "
     "identical national figure into **Buddhism 5,506, Hinduism 3,211 and Other "
     "non-Christian 983,587**. So 8,717 Buddhists and Hindus are inside this node and there "
     "is no geography anywhere in the report that would let them out of it — §3.9's "
     "category/geography trade, made across two tables of one publication rather than "
     "inside one. "
     "The remaining 983,587 is the third-largest residual here in relative terms, and it is "
     "large because the question is a DENOMINATION question: the ten cells are eight "
     "Christian groupings, Islam and No Religion, so a Malawian Bahá'í, Rastafarian, Sikh "
     "or Jew has nowhere else to go, and neither does anyone whose answer the coder could "
     "not place. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Mauritius.
    ("other.mu",
     "Other religion (Mauritius)",
     "Statistics Mauritius's `Other & Not stated` — 6,931 people, 0.56%, and **it is the "
     "only residual on this map that knowingly mixes an answer with a non-answer.** Every "
     "other source here keeps `not stated` in a cell of its own and this project takes it "
     "off the tree (§3.5); Mauritius pools the two and publishes no split at any geography, "
     "so the choice is between drawing an unknown number of non-responders as a religion or "
     "deleting an unknown number of real adherents. It is drawn, and this note is the "
     "marking §3.5 asks for: **read 6,931 as a ceiling on Mauritius's other religions, not "
     "a count of them.** "
     "What is genuinely inside it is small and knowable in kind if not in number: the "
     "Bahá'ís, who have been in Mauritius since the 1950s, the few hundred Sikhs, and the "
     "island's tiny Jewish community — the Beau Bassin detainees of 1940-45 were interned "
     "here and most left after the war. Per source, per spec §3.11."),

    # --- added 2026-09-06 with Switzerland.
    ("other.ch",
     "Other church or religious community (Switzerland)",
     "BFS's `Übrige Kirchen und Religionsgemeinschaften` — 14,226 people, 0.19%. **Small "
     "because the list above it is the longest any European census on this map offers**: "
     "seven Protestant cells, two Catholic, Orthodox, other Christian, Jewish, Islamic, "
     "Buddhist and Hindu all have boxes of their own, so a Swiss respondent has to be "
     "outside eighteen named answers to land here. What is in it is the Sikh gurdwaras of "
     "the Zurich and Geneva regions, the Bahá'ís — whose European Continental House of "
     "Worship stands at Langenhain rather than here, but whose Swiss community is old — the "
     "Yazidis, and the small esoteric and new religious movements Switzerland has an "
     "unusual number of. **Read it as 2000's residual carrying a 2024 magnitude**, per "
     "`ch_rescale.py`: the group it sits in has grown, and which parts of it grew is not "
     "knowable from this source. Per source, per spec §3.11."),

    # --- added 2026-09-06 with Jamaica.
    ("other.jm",
     "Other religion (Jamaica)",
     "STATIN's `Other religion` — **169,014 people, 6.30%**, and **the largest residual on "
     "this map relative to the length of the list above it**. Seventeen named bodies get a "
     "cell, including four Church of God bodies separately, and one in sixteen Jamaicans "
     "still lands outside all of them. That is a fact about how many small churches Jamaica "
     "has rather than about how coarse the question is.\n\n"
     "**What is NOT in it, and this is the trap the country turns on.** It does not hold "
     "Jamaica's Bahá'ís, Hindus, Muslims or Jews: STATIN excluded those four religions from "
     "the parish tables altogether — 4,124 people — so they are absent from the drawn "
     "universe rather than pooled here (`sources/jm.md` §3). A reader who assumes a residual "
     "catches everything the list misses would be wrong about exactly those four.\n\n"
     "**What is in it** is the long tail of Jamaican Christianity — the independent and "
     "storefront churches, the Church of God bodies too small for their own cell, the "
     "Disciples of Christ, the Salvation Army — together with the Kumina societies, whose "
     "Kongo-derived practice has no box anywhere on the form.\n\n"
     "**AND ITS GEOGRAPHY IS SHARP, WHICH BY §9r's RULE MAKES IT A MISSING CATEGORY RATHER "
     "THAN A MIXTURE.** 6.31% nationally, but **2.92% in Kingston and 14.18% in "
     "Westmoreland** — a fivefold spread, with the peak at the western end of the island and "
     "the trough in the capital. That is the opposite of what a genuine grab-bag looks like "
     "and it says something specific is inside. STATIN publishes no composition of the cell, "
     "so **what** is not determinable from this source, and it is left as an open question "
     "rather than guessed at (§14.4) — the same shape as Bosnia's Velika Kladuša peak in "
     "`other.ba`, found the same day. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Liechtenstein.
    ("other.li",
     "Other religious community (Liechtenstein)",
     "The Amt für Statistik's `Other religious communities` — **128 people, 0.34%, and the "
     "smallest residual on this map in absolute terms by two orders of magnitude.** In a "
     "country of 37,622 that is not a rounding cell but a countable group of individuals, "
     "and the census publishes it down to single people in a commune. "
     "**What matters about it is what the form does NOT ask.** There is no Jewish cell "
     "anywhere in Liechtenstein's religion question, so the country's Jewish residents are "
     "in here — which is why `coverage.py` leaves Liechtenstein unlit for Judaism rather "
     "than lit-with-no-dots: the question was not put (§6.12). The Bahá'ís, the Hindu and "
     "Sikh residents of the industrial communes, and the small new religious movements are "
     "here too. A residual this small in a country this small is a list of people rather "
     "than a category, and nothing separates them. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Saint Vincent.
    ("other.vc",
     "Other religion (Saint Vincent)",
     "Two cells together — the SVG Statistical Office's `Other religion` (4,672 people, "
     "4.28%) and its `Traditional` (74, 0.07%). Seventeen bodies are named above them, "
     "including Presbyterian at 294 people and Salvation Army at 287, so this is the tail "
     "of an unusually generous list rather than a coarse bucket.\n\n"
     "**`Traditional` is folded in here, and what it is NOT was established rather than "
     "assumed.** The obvious reading is the Kalinago (Carib) population's practice — Saint "
     "Vincent has the largest surviving indigenous community in the eastern Caribbean — and "
     "the data rules it out. The same sheet carries ethnicity, so §12's Philippines "
     "co-location technique applies directly: Traditional's share correlates with the "
     "Indigenous share at **r = -0.03** across the 219 populated enumeration districts, and "
     "**every one of the eight most-indigenous districts has zero Traditional**, including "
     "the three Sandy Bay districts at 79-86% Indigenous. Its own largest cells sit in "
     "Georgetown, the Northern Grenadines and Kingstown, which are 75-83% Black and 0-1% "
     "Indigenous.\n\n"
     "The plausible remaining readings are African-derived practice and the **Spiritual "
     "Baptist / 'Converted'** tradition for which Saint Vincent is the home — criminalised "
     "here from 1912 to 1965 — but the form's separate `Baptist` cell may already absorb "
     "the Shakers, and 74 people across 25 districts carries no signal either way. "
     "**`afrodiasporic` is the node that would be wanted if a source ever resolved it**, "
     "beside Jamaica's Revival Zion; recorded per §2.4 so the fix is a lookup. Not asserted "
     "now (§14.4). Per source, per spec §3.11."),

    # --- added 2026-09-07 with France.
    ("other.fr",
     "Other religion (France)",
     "Two ESS answers together — `Eastern religions` and `Other Non-Christian religions` — "
     "0.55% of French citizens, and **the residual on this map that is most clearly hiding "
     "a specific large group rather than a tail.** France has the largest Buddhist "
     "population in Europe: the Vietnamese, Cambodian and Lao communities who arrived as "
     "refugees after 1975, their pagodas in Île-de-France and around Bordeaux, and a "
     "sizeable French-convert Zen and Tibetan following. Pew puts French Buddhists at "
     "0.71% of the country and this map draws 0.13%, almost all of it from the foreign "
     "half — **because a Buddhist who holds a French passport has nowhere to go on the ESS "
     "form.** The tree has no node for 'some Eastern religion, unspecified' and choosing "
     "one would invent a fact (§14.4), so both answers sit here, as gr2024.py, ru2012.py "
     "and hr2021.py do for the same shape.\n\n"
     "**What else is in it is knowable and also not separable**: France's Sikhs, "
     "concentrated in Bobigny and the Seine-Saint-Denis; its Hindus, largely Tamil from "
     "Sri Lanka and the former comptoirs, whose Ganesha temple in La Chapelle draws the "
     "largest Hindu procession in Europe; the Bahá'ís; and the neo-druid and Wiccan "
     "revival. **The cell is a limitation of the instrument and not of the country**, which "
     "is the opposite of Bosnia's and worth distinguishing: a census with three boxes has "
     "decided what it will not see, whereas ESS offers eight denominations to a sample of "
     "two thousand and simply cannot resolve what is under one percent.\n\n"
     "**AND IT HOLDS A SECOND, UNRELATED THING IN ONE PLACE.** France's five overseas "
     "régions are drawn from Pew rather than from ESS (§9ag), and Pew's `Other religions` "
     "lands here too — which in **Guyane is 9.2%, about 26,700 people, by a distance the "
     "largest single thing this residual holds anywhere in France.** It is not a tail: it "
     "is the traditional practice of the Businenge (Maroon) communities of the Maroni, "
     "descended from people who escaped the Surinamese plantations in the seventeenth and "
     "eighteenth centuries, and of the Kalina, Wayana, Teko, Wayampi and Palikur. "
     "`afrodiasporic` and `indigenous` are the two nodes that would be wanted and Pew "
     "publishes no split between them, so the proportions would have to be guessed and the "
     "guess would be the only fact that mattered (§14.4). **Recorded here rather than "
     "resolved**, per §2.4, so that a source which ever separates them is a lookup and not "
     "an excavation. Per source, per spec §3.11."),

    # --- added 2026-09-08 with Italy.
    ("other.it",
     "Other religion (Italy)",
     "Two ESS answers together — `Eastern religions` and `Other Non-Christian religions` — "
     "0.44% of Italian citizens across the pooled rounds, and **the smallest residual on "
     "the map that is hiding the clearest single thing.** Italy has the largest Soka Gakkai "
     "membership in Europe: the Istituto Buddista Italiano Soka Gakkai signs a state "
     "*intesa*, takes 91,006 otto per mille choices, and claims roughly 90,000 members, "
     "which is on its own larger than everything this node draws. Beside it sit the Unione "
     "Buddhista Italiana's Theravada and Tibetan sanghas, the Unione Induista Italiana, and "
     "the Bahá'í community — **all four of which the Italian state recognises by name and "
     "funds by name, while the only survey that asks Italians their religion offers them "
     "one box marked `Eastern religions`.** The tree has no node for 'some Eastern "
     "religion, unspecified' and choosing one would invent a fact (§14.4), so both answers "
     "sit here, as gr2024.py, fr2024.py, ru2012.py and hr2021.py do for the same shape.\n\n"
     "**It is the instrument's limitation and not the country's**, and Italy makes that "
     "unusually easy to prove: the confessions in this cell are individually enumerated in "
     "the 8x1000 returns, so a reader can see the categories the state counts and the "
     "single box the survey offers, side by side, in the same country. What that return "
     "cannot do is supply a magnitude — it counts spending votes, not adherents, and "
     "overstates the Waldensians twentyfold while understating the Orthodox by forty "
     "(sources.md §11l-ii). So the names are knowable, the numbers are not, and the two "
     "facts do not meet. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Benin.
    ("other.bj",
     "Other religion (Benin)",
     "INStaD's `Autres religions` — 259,446 people, 2.59%, and it is a genuine tail rather "
     "than a hidden large group, because RGPH-4 has already given its own cells to the two "
     "things that would otherwise be buried in it: `Vodoun` and `Chrétien céleste` are "
     "counted by name. What is left is small and mixed — the Bahá'ís, who have been in "
     "Benin since the 1950s and are one of the country's recognised confessions; the "
     "Eckankar following, unusually visible in Cotonou; a Hindu and a Buddhist presence "
     "measured in thousands; and the smaller prophetic and syncretic movements that neither "
     "`Autres chrétiens` nor `Chrétien céleste` claims. "
     "**Its geography says it is not noise, and also not one thing**: it peaks at 7.5% in "
     "Houéyogbé, 6.9% in Djidja and 6.8% in Ouèssè, and falls to 0.0% in Karimama — a "
     "southern, Fon-and-Adja distribution that tracks neither the Christian nor the Muslim "
     "pattern. Some of what makes it southern is very likely the Orisha and Egungun "
     "societies that a respondent might not file under `Vodoun`, and the census cannot say. "
     "Per source, per spec §3.11."),

    # --- added 2026-09-07 with Zimbabwe.
    ("other.zw",
     "Other religion (Zimbabwe)",
     "ZIMSTAT's `Other` — 124,017 people, 0.82%, and an unusually SMALL residual for a "
     "census with only eleven boxes, because the box that would normally swallow everything "
     "is `Apostolic Sect` and it is counted by name. Zimbabwe already gives cells to Islam, "
     "Judaism and Hinduism at 0.58%, 0.05% and 0.02%, so the tail this holds is genuinely "
     "the tail: the Bahá'ís, the Rastafari of Harare and Bulawayo, the small Greek and "
     "Armenian Orthodox communities of the old settler cities, and the Buddhist and Sikh "
     "presence that arrived with Indian and Chinese migration. "
     "**Its geography is Matabeleland**, at 2.6% in the South and 1.9% in the North against "
     "0.3% in Mashonaland Central — the same halves of the country that hold the mission "
     "churches and the highest `None`. Per source, per spec §3.11."),

    # --- added 2026-09-08 with Eswatini.
    ("other.sz",
     "Other religion (Eswatini)",
     "The CSO's `Other` in Table 3.2.1 — 3,363 people, 0.31%, and small for the same reason "
     "`other.zw` is: the 2017 census gives boxes of their own to Islam, Hinduism, the "
     "Baha'i Faith and Judaism, at 0.33%, 0.02%, 0.04% and 0.01%, so nothing large is "
     "hiding in here. What is left is the tail a Southern African census usually collects "
     "under this heading: Rastafari, the Greek Orthodox and Muslim traders' congregations "
     "of Mbabane and Manzini that did not use their own box, and the handful of Buddhist "
     "and Sikh households that came with Indian and Chinese migration. "
     "**It has no geography and nothing here claims one.** The CSO publishes Table 3.2.1 "
     "nationally only, so every dot on this node in Eswatini is `derived`: drawn at the "
     "national rate inside each region's own measured non-Christian total. Per source, per "
     "spec §3.11."),

    # --- added 2026-09-07 with the Central African Republic.
    ("other.cf",
     "Other religion (Central African Republic)",
     "ICASEES's `Autre réligion` — 171,441 people, 4.47%, and **the residual on this map "
     "that most clearly has the geography of a category the form does not offer.** RGPH03 "
     "asks five things — Catholique, Protestante, Musulmane, Autre réligion, Sans réligion "
     "— and **there is no traditional or animist box at all**, where Ghana, Kenya, "
     "Ethiopia, Malawi and Benin all have one. "
     "**Where this cell peaks says what is in it.** 23.7% in Topia, 21.0% in Moboma and "
     "Baleloko, 20.2% in Carnot, 19.5% in Lésse — every one of them in Lobaye or "
     "Mambéré-Kadéï, the south-western forest, which is the Aka homeland and the Gbaya and "
     "Ngbaka country. A fivefold concentration against the national 4.47%, in exactly the "
     "part of the country where traditional practice is strongest. **`Sans réligion` peaks "
     "in the same communes**, which is what a census with no traditional box does to two "
     "of its cells at once. "
     "**And it is still not mapped to `indigenous.african`**, for the reason taxonomy/"
     "cf2003.py argues at length: the cell genuinely also holds CAR's Bahá'ís, Jehovah's "
     "Witnesses, Kimbanguists and the Orthodox merchants of Bangui, so sending all of it "
     "to the tradition would miscount them — and at the same time the tradition's real "
     "size is larger than this cell, because whoever answered `Catholique` or "
     "`Protestante` as well is invisible. Mapping it across would be an overclaim and an "
     "undercount simultaneously. CAR's traditional religion is not drawn, and saying so is "
     "the only accurate thing available. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Côte d'Ivoire.
    ("other.ci",
     "Other religion (Côte d'Ivoire)",
     "ANStat's `Autres religions` — 53,051 people, 0.18%, and **the smallest residual on "
     "this map relative to its country**, which is only true because the cell above it is "
     "doing the work. `Autres religions chrétiennes` is 6,004,781 people, 20.5%, and the "
     "RGPH 2021's own methodology says the census collected `Evangélique`, `Céleste`, "
     "`Bouddhiste` and `Témoin de Jehova` as distinct modalities that appear in no printed "
     "table. So Côte d'Ivoire's Buddhists — who would normally land here — are inside a "
     "cell labelled *other CHRISTIAN religions*, and this node holds only what was left "
     "after that: the Bahá'ís, the small Hindu and Lebanese-Druze communities of Abidjan, "
     "and the syncretic movements that claimed neither a Christian nor a Muslim label. "
     "**A residual being unusually small is a fact about the table above it, not about the "
     "country.** Per source, per spec §3.11."),

    # --- added 2026-09-07 with Malaysia.
    ("other.my",
     "Other religion (Malaysia)",
     "DOSM's `Lain-lain` — 285,152 people, 0.88%, and **the most crowded residual on this "
     "map**, because the footnote printed on every sheet of Table 7 says exactly what is in "
     "it: *\"Others include Sikhism, Taoism, Confucianism, Bahai, Tribal/ folk/ other "
     "traditional Chinese religion, Animisme and Others.\"* Six named traditions in one "
     "cell. "
     "**Its geography is sharp, and unusually it resolves into TWO different missing "
     "categories rather than one.** §9r's Chittagong rule says a residual with a sharp "
     "geography is a missing category; Malaysia's has two peaks that are nothing like each "
     "other. "
     "**The Chinese one.** Timur Laut — George Town, the most Chinese district in the "
     "country and 52.5% Buddhist — is 3.65%, with Barat Daya 2.22% and Seberang Perai "
     "Selatan 1.99% behind it. That is **Chinese folk religion, Taoism and Confucianism**, "
     "and it is the sharpest loss in this cell: the tree HAS a `chinesefolk` node, "
     "Malaysia's Chinese population is roughly a fifth of the country, and DOSM's form "
     "gives temple practice no box, so a tradition this map draws in China, Vietnam and "
     "Singapore cannot be brought out here at all. "
     "**And the indigenous one.** Song, Sarawak reaches **10.14%**, and behind it comes a "
     "belt of peninsular interior districts — Cameron Highlands 4.36%, Batang Padang "
     "2.73%, Kuala Langat 2.47%, Gua Musang 2.38%, Raub 2.35%, Lipis 2.24%. Those are the "
     "**Orang Asli** districts and the interior Iban country, and the near-certain content "
     "is indigenous religion, which the footnote calls `Animisme`. "
     "**What makes the reading solid is where the cell is EMPTY.** Sabah has the largest "
     "indigenous population in Malaysia and its `Lain-lain` is essentially zero — Nabawan "
     "and Telupid record 0 people, Tambunan and Kalabakan 1 each — because Sabah's "
     "indigenous communities had overwhelmingly become Christian or Muslim by 2020. A "
     "residual that is 10% in interior Sarawak and 0.00% in interior Sabah is not measuring "
     "remoteness; it is measuring which traditions survived, which is a real finding rather "
     "than an artefact. "
     "Not split, because DOSM publishes no composition and the two contents overlap in no "
     "district (§14.4). Per source, per spec §3.11."),

    # --- added 2026-09-07 with Belize.
    ("other.bz",
     "Other religion (Belize)",
     "SIB's `Other` — 25,117 people, **6.32%**, and a wide residual because the list above "
     "it is short: nine Christian bodies and nothing else at all. No cell for Hinduism, "
     "none for Islam, none for any indigenous or Afro-Caribbean tradition, in a country "
     "that has all four. "
     "**Its geography is sharp, which by §9r's rule makes it a missing category rather "
     "than a mixture** — 11.9% in Orange Walk against 2.7% in Stann Creek, a 4.4x spread — "
     "**but unlike Bosnia's it does not resolve to one thing, and it is not guessed at.** "
     "The peak is the two northern districts, which are also the Mennonite and Mestizo "
     "north; the second is Toledo at 8.4%, which is the Maya south and where indigenous "
     "practice would be if it were anywhere. What is known to exist in Belize and has no "
     "box on this form: a Hindu and Muslim population concentrated in Belize City and the "
     "north, a Bahá'í community, Rastafari, Maya traditional practice in Toledo, and the "
     "Garifuna *dugu* of the Stann Creek coast. "
     "**The Garifuna case is the one that shows why nothing is assigned.** Stann Creek is "
     "the Garifuna district and it has the LOWEST `Other` in the country, 2.7% — which is "
     "consistent with the well-documented pattern of Garifuna practising dugu alongside "
     "Catholicism and answering a census with the church. So a tradition that certainly "
     "exists is probably not in this cell at all, and is instead invisible inside "
     "`christianity.catholic`. Naming that possibility is the honest thing this node can "
     "do; splitting on it would be §14.4. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Trinidad and Tobago.
    ("other.tt",
     "Other religion (Trinidad and Tobago)",
     "CSO's `Other` — 96,166 people, **7.27%**, which is wide for a question that already "
     "names fourteen bodies including three the rest of this map has never counted. "
     "**Its geography is sharp, and by §9r's rule that means a missing category rather than "
     "a mixture** — 14.4% in Point Fortin, 9.5% in Chaguanas, 9.0% in Siparia, against 4.5% "
     "in Port of Spain. It does not resolve onto one population: Point Fortin is "
     "Afro-Trinidadian and Chaguanas is Indo-Trinidadian, so whatever is in this cell has "
     "at least two different contents. "
     "**What is known to exist in Trinidad and has no box on this form**: the Baptist and "
     "evangelical bodies CSO's `Baptist-Other` does not reach, the Lutheran and Salvation "
     "Army congregations, Bahá'ís, the Sikh and Buddhist communities that came with later "
     "Indian and Chinese migration, and the Chinese temple practice of Port of Spain and "
     "San Fernando. **The Indo-Trinidadian peaks may also be sectarian Hindu answers** — "
     "the Kabir Panth, the Arya Samaj and the Sai movement all have a Trinidadian presence "
     "and none of them has a cell, so a respondent who names one may have been coded here "
     "rather than to `Hinduism`. That is a guess about a coding practice CSO does not "
     "publish, and it is written down as a guess. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Kazakhstan, and it is a MODELLED residual (spec §14.10).
    ("other.kz",
     "Other religion (Kazakhstan)",
     "BNS's `Другое` — **23,247 people, 0.12%**, and the smallest per-country residual on "
     "this map. The census offers Islam, Christianity (split three ways), Judaism, Buddhism, "
     "`other`, `refused to state` and `non-believer`, so this cell is what is left of a very "
     "full form: the Baha'i, Hare Krishna and new religious movements that Kazakhstan's "
     "registration law names separately, plus whatever else was written in. "
     "**Unlike every other `other.*` node here, these dots are MODELLED rather than "
     "counted** — Kazakhstan publishes religion nationally only, and its geography here is "
     "each region's ethnic composition applied to the national share (sources/kz.py). At "
     "0.12% of a modelled country it is the faintest claim on the map, and it is drawn "
     "because dropping it would silently move 23,247 people into nothing rather than "
     "marking them (§3.5)."),

    # --- added 2026-09-07 with Cambodia.
    ("other.kh",
     "Other religion (Cambodia)",
     "NIS's `Other` — 82,332 people, 0.53%, and **the sharpest residual geography on this "
     "map**: Ratanak Kiri is 23.2% and Mondul Kiri 21.2%, the two of them holding 70,000 of "
     "the 82,332 between them, against **0.0% in fifteen of the twenty-five provinces**. By "
     "§9r's rule a residual that sharp is a missing category rather than a mixture, and "
     "unlike Belize's or Trinidad's it resolves onto essentially ONE thing. "
     "**NIS says what it is, in the paragraph above the table**: *\"The category of 'Others' "
     "mainly refers to the local religious system of the highland tribal groups and a few "
     "minority religious groups from other countries.\"* That is the animist traditions of "
     "the north-eastern highlands — the Bunong of Mondul Kiri, the Tampuan, Jarai, Kreung, "
     "Brao and Kavet of Ratanak Kiri — spirit forests, buffalo sacrifice and ancestor "
     "practice, and the largest surviving non-Abrahamic, non-Buddhist tradition in mainland "
     "Southeast Asia outside the Vietnamese highlands. "
     "**It is NOT filed on `indigenous`, and the reason is the other 15%.** The cell is not "
     "empty away from the highlands: Kampong Chhnang has 4,739 (0.9%) with no highland "
     "population at all, Phnom Penh 2,282 and Kandal 1,202. Whatever those are — Chinese "
     "temple practice, Bahá'ís, Cao Đài along the Vietnamese border — they are not Bunong "
     "animism, and NIS's own wording (*\"mainly… and a few…\"*) says the cell has at least "
     "two contents. Filing the whole of it as indigenous would assert the magnitude of a "
     "category the source does not publish, and splitting it would invent one (§14.4). "
     "**So the node that is wanted is an `indigenous` child for the Cambodian highland "
     "traditions, and this records that it is wanted** (§2.4) — the moment any source counts "
     "them separately this becomes a lookup rather than an investigation. Until then the "
     "map can show that a fifth of two provinces answers none of the three named religions, "
     "and cannot show what they answer instead. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Suriname.
    ("other.sr",
     "Traditional religion and other (Suriname)",
     "ABS's `Traditional Religion +Others` — 28,549 people, 5.79%, and **the only residual "
     "on this map whose label names a tradition and then pools a remainder into the same "
     "cell.** That shape is why it sits here rather than at `indigenous.african`, where "
     "Ethiopia's and Ghana's `Traditional` cells go: mapping it there would assert that all "
     "28,549 practise a traditional religion, and the later census shows they do not.\n\n"
     "**Census 8 (2012) prints the equivalent district cell as `Andere godsdienst Jodendom "
     "Winti Jehova's Getuigen`** — Winti pooled with Judaism and the Jehovah's Witnesses — "
     "which is direct evidence from the same office that this cell mixes an "
     "Afro-Surinamese religion with two unrelated things.\n\n"
     "**What is certainly inside it.** *Winti*, the Afro-Surinamese religion of the Maroon "
     "and Creole populations — sibling to Vodou, Candomblé and Trinidad's Orisha, and the "
     "one this map would most want to name. It was **criminalised in Suriname until 1971** "
     "and is still widely practised alongside a church affiliation, so a single-answer "
     "census cell undercounts it twice over. Also the *indigenous* religion of the Kalina, "
     "Lokono, Trio and Wayana. And then the actual residual: Judaism — Paramaribo's "
     "Sephardic and Ashkenazi congregations are among the oldest in the Americas — the "
     "Bahá'ís, and the Jehovah's Witnesses.\n\n"
     "**Not split.** ABS publishes no composition of it at any geography, and the 2012 "
     "national denominational figures cannot be pushed down onto 62 ressorten without "
     "inventing their entire spatial structure — §14 rule 1, and the same refusal §11r made "
     "for Saudi Arabia. A Census 9 build would very likely be able to name Winti properly, "
     "and that is the thing to want here (§2.4). Per source, per spec §3.11."),

    # --- added 2026-09-07 with Myanmar.
    ("other.mm",
     "Other religion (Myanmar)",
     "DOP's `Other religion` — **82,825 people, 0.16% of the enumerated population**, and one "
     "of the smallest residuals on this map, because the 2014 census already gives boxes to "
     "Buddhist, Christian, Islam, Hindu, Animist and No religion. What is left is genuinely "
     "the tail.\n\n"
     "**Its geography is the minority states, by share rather than by count.** Kayah 1.20%, "
     "Chin 1.11%, Kayin 0.68% and Shan 0.46%, against 0.10% in Yangon and 0.11% in "
     "Ayeyawady — while in absolute terms the largest blocks are Shan (27,036), Bago (12,687) "
     "and Kayin (10,194). §9r's rule says a residual with a sharp geography is a missing "
     "category, and this one has two different peaks that do not resolve onto a single thing, "
     "so nothing is assigned.\n\n"
     "**What is known to exist in Myanmar and has no box on this form**: a Bahá'í community, "
     "the Sikh and Chinese temple populations that came with colonial-era migration and are "
     "concentrated in Yangon, Mandalay and the Bago plantation districts (where the census's "
     "Hindu count is also unusually high at 2.0%), and a very small Jewish remnant. **The "
     "upland peaks may instead be traditional practice that respondents or enumerators did "
     "not code to `Animist`** — Chin and Kayah are the two states with the highest share here "
     "AND high Christian shares, so a category boundary rather than a population is a real "
     "possibility. That is a guess about coding practice DOP does not publish, and it is "
     "written down as a guess. Per source, per spec §3.11."),

    # --- added 2026-09-07 with the Bahamas.
    ("other.bs",
     "Other religion (Bahamas)",
     "**TWO CELLS, AND THEY ARE NOT THE SAME KIND OF THING.** Together 2,171 people, 0.55% "
     "of the country.\n\n"
     "**1. BNSI's `Other Non-Christian Religion` — 1,799 people, 0.45%, and a NARROW "
     "residual.** The 2022 form already gives boxes to Bahá'í, Hindu, Muslim, Jewish and "
     "Rastafari, so this is the tail after five non-Christian answers rather than a bucket "
     "standing in for them. **Its geography is flat, which by §9r's rule makes it a mixture "
     "rather than a missing category**: 0.72% on Grand Bahama and Bimini, 0.66% on Andros, "
     "0.45% on New Providence, 0.35% on Eleuthera, 0.23% on Abaco — a threefold spread "
     "across the only six islands that report it at all, with no peak to point at. Nothing "
     "is assigned to it.\n\n"
     "**What is known to exist in the Bahamas and has no box on this form**: Obeah, the "
     "Bahamian folk practice, which is a set of practices carried alongside a church "
     "affiliation rather than an answer to `what is your religion`; Haitian Vodou, which "
     "matters here more than the 0.45% suggests, because the Bahamas has the largest "
     "Haitian-descended population of any country in the region outside Hispaniola and this "
     "cell plainly does not hold it — the same single-box undercount `tt2011.py` records "
     "for Trinidadian Orisha and `sr` for Winti; the Buddhist and Chinese temple practice "
     "of the Nassau Chinese community; and a small Sikh presence.\n\n"
     "**2. The per-island suppression residual, `Other Religion` — 372 people, 0.093%, and "
     "it is a disclosure artefact rather than a religion.** Only New Providence prints all "
     "24 named bodies; the other seventeen islands fold their smallest answers into one "
     "cell, with a starred footnote under each table naming exactly which. So this half's "
     "contents differ island by island and are mostly CHRISTIAN — Roman Catholics and "
     "Pentecostals on Ragged Island, Greek Orthodox and Jews on Long Island, Hindus and "
     "Muslims on Cat Island. It is pooled here because there is no node meaning `some "
     "religion, and which one depends where you are standing`, and because the alternative "
     "— dropping it — would delete 30.4% of Ragged Island and 11.8% of Mayaguana from the "
     "map rather than misfiling them.\n\n"
     "**On those same two islands the footnote says `None` was folded in**, so up to 41 "
     "non-religious Bahamians are drawn here as a religion, and Mayaguana and Ragged Island "
     "show no irreligion at all. Recorded, not corrected (§14.4); `sources/bs.py` prints "
     "every island's residual and footnote on each run. Per source, per spec §3.11."),

    # --- added 2026-09-07 with the Cayman Islands.
    ("other.ky",
     "Other religion (Cayman Islands)",
     "ESO's `Other` — 2,679 people, **3.89%**, and a NARROW residual: the 2021 form already "
     "names Hindu, Muslim, Jewish and Rastafari separately, so the non-Christian population "
     "of the territory is not pooled here.\n\n"
     "**Its geography is flat, which by §9r's rule makes it a mixture rather than a missing "
     "category** — 6.03% in East End, 4.19% in George Town, 3.62% in Bodden Town, 3.50% in "
     "West Bay, 3.36% on the Sister Islands, 2.26% in North Side. A twofold spread with no "
     "peak to point at, in a country where every OTHER category's geography tracks foreign "
     "birth sharply. Nothing is assigned to it.\n\n"
     "**What is known to exist in the Cayman Islands and has no box on this form**: the "
     "Bahá'í community, which has been established here since the 1960s; Buddhist practice "
     "among the Filipino, Chinese and other Asian workforce — over half the resident "
     "population was born abroad, and this is the one large migrant religion the form does "
     "not name; the Salvation Army and the Brethren assemblies, which are named cells in "
     "neighbouring Jamaica's census and not here; and the Church of Christ and smaller "
     "independent congregations. **Its highest share is East End**, the smallest and least "
     "expatriate district on Grand Cayman, which does not fit the migrant-religion reading "
     "and is not explained. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Barbados.
    ("other.bb",
     "Other religion (Barbados)",
     "BSS's `Other Non-Christian` — 622 people, **0.27%, the narrowest residual on this "
     "map**. That is not because Barbados is uniform but because the 2010 form is generous: "
     "it names Bahá'í, Hindu, Jewish, Muslim and Rastafari as separate answers, so the "
     "non-Christian field is almost entirely accounted for before this cell is reached. "
     "Fifteen Christian bodies are named too, and their leftovers go to "
     "`christianity.other` rather than here.\n\n"
     "**Six hundred people is too few to have a geography worth reading**, and none is "
     "claimed: the parish shares run 0.10% to 0.47% with the largest single cell at 275 "
     "people in St. Michael, which is where a fifth of Barbados lives. §9r's rule needs a "
     "peak to point at and there is not one.\n\n"
     "**What is known to exist in Barbados and has no box on this form**: Buddhist practice "
     "among the small Chinese and Guyanese-Chinese community, the Sikh presence that came "
     "with Indian migration alongside the counted Hindus and Muslims, and the Spiritual "
     "Baptist / Tie-head churches — though those last would more likely answer `Baptist` or "
     "`Other Christian`, which is where Trinidad's own Shouter Baptists would land if CSO "
     "did not give them a cell of their own (`tt2011.py`). Per source, per spec §3.11."),

    # --- added 2026-09-07 with Thailand.
    ("other.th",
     "Other religion (Thailand)",
     "NSO's `อื่น ๆ` — 55,545 people, **0.08%, and the narrowest residual on this map after "
     "Barbados's** — from a form that names Buddhism, Islam, Christianity, Hinduism, "
     "Confucianism and Sikhism as separate answers and keeps `no religion` in a cell of its "
     "own beside this one.\n\n"
     "**What is most likely in it is the animist practice of the highland peoples, and "
     "unlike Cambodia's cell that gets no argument here** — because NSO says nothing about "
     "what the category contains. `other.kh` could be read as Bunong and Tampuan animism "
     "because NIS wrote the sentence explaining it; nothing equivalent exists in the Thai "
     "volumes, and §3.11's floor is what stops the guess from being made anyway. The "
     "difference is worth stating rather than smoothing over: two neighbouring censuses "
     "produce the same-shaped residual and only one of them can be read.\n\n"
     "**Its geography cannot be read either, and that is a property of this country's "
     "build rather than of the cell.** Thailand is a spec §3.10 allocation — only Buddhist "
     "and Muslim are published per province — so every province's share of this node is its "
     "own region's share, and the four regions run 0.04% to 0.13%. Nothing finer is "
     "measured, so nothing finer is claimed. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Saint Lucia.
    ("other.lc",
     "Other religion (Saint Lucia)",
     "CSO's `Other` — 3,854 people, **2.24%**, and it is the last of question 1.5's 22 "
     "options rather than a residual the tabulator invented. The form already names Bahá'í, "
     "Buddhism, Hindu, Islam and Rastafari, so the non-Christian field is largely accounted "
     "for before this cell is reached; what it mostly holds is Christian bodies the "
     "twelve-name Christian list does not reach.\n\n"
     "**Its geography is flat, which by §9r's rule makes it a mixture rather than a missing "
     "category** — 2.85% in Gros Islet down to 1.15% in Canaries, a 2.5-fold spread across "
     "ten districts with no peak to point at, in a country where every large category has a "
     "sharp north-south gradient. Nothing is assigned to it.\n\n"
     "**What is known to exist in Saint Lucia and has no box on this form**: the Spiritual "
     "Baptist / Converted churches, which neighbouring Saint Vincent (`vc2012.py`) and "
     "Trinidad (`tt2011.py`) both count separately and which are present here — though "
     "many would answer `Baptist`, a named option, so this cell cannot be read as them; the "
     "Church of God bodies, which Barbados, Jamaica, the Bahamas, Cayman and Grenada all "
     "name and Saint Lucia does not; the Baptist-adjacent independent congregations; and "
     "Kali worship and other Indo-Caribbean practice alongside the counted Hindus. Per "
     "source, per spec §3.11."),

    # --- added 2026-09-07 with Grenada.
    ("other.gd",
     "Other religion (Grenada)",
     "CSO's `OTHER (SPECIFY)` — 1,716 people, **1.58%**, and one of the narrowest residuals "
     "here relative to how much the form names: **25 answers**, including `SPIRITUAL "
     "BAPTIST`, `MENNONITE`, `LUTHERAN`, `MORAVIAN`, `PRESBYTERIAN`, `INDEPENDENT "
     "BAPTISTE`, `EVANGELICAL`, `CHURCH OF GOD`, `BUDDHIST`, `BAHAI`, `HINDU`, `MUSLIM` "
     "and `RASTAFARIAN` — so both the Christian and the non-Christian fields are unusually "
     "well covered before this cell.\n\n"
     "**It is written-in, and Grenada does not publish what people wrote.** `(SPECIFY)` is "
     "in the category's own name: the form asked, so the answers exist in the microdata and "
     "are simply not tabulated. That makes this the one residual on the map whose contents "
     "are known to a statistics office and not to anyone else, which is worth recording as "
     "a thing that could be asked for rather than guessed at.\n\n"
     "**Its geography has a peak and it is not read as one.** St. John 4.19% and St. Mark "
     "4.01% against 0.48% in St. Andrew — an eightfold spread, and §9r's rule would make "
     "that a missing category rather than a mixture. But the two peaks are also the two "
     "parishes with the most `PRESBYTERIAN` (St. Mark 3.17%, seventeen times the national "
     "rate) and the most `SPIRITUAL BAPTIST` (St. Mark 4.77%), so whatever is being "
     "written in there is written in *beside* those and not instead of them, and nothing "
     "in the report says what it is. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Nicaragua.
    ("other.ni",
     "Other religion (Nicaragua)",
     "INIDE's `Otra` — 74,101 people, **1.63%**, against a question that names only seven "
     "things: Catholic, Evangelical, Moravian, Jehovah's Witness, Judaism, Islam and none. "
     "**That list is short in a specific direction — it names no Anglicans, no Baptists, no "
     "Adventists and no Latter-day Saints**, all four of which have a documented Nicaraguan "
     "presence, and it is the only census on this map that names the Moravians while naming "
     "none of the other Protestant bodies beside them.\n\n"
     "**Its geography is not merely sharp, it is one coastline, and by §9r's rule that "
     "makes it a missing category rather than a mixture.** Corn Island **44.05%**, Laguna "
     "de Perlas 21.44%, Bluefields 17.48%, Desembocadura de Río Grande 13.74%, Puerto "
     "Cabezas 10.19%, Kukra-Hill 8.46% — against 1.63% nationally and **0.02% in San José "
     "de Cusmapa and Ciudad Antigua**, a spread of more than two thousand to one. Every "
     "municipality above 8% is on the Caribbean, and Corn Island is the largest single "
     "`Otra` share of any unit in any country drawn here.\n\n"
     "**What is known to be there and has no box**: the Anglican church of the Mosquito "
     "Coast, which arrived with the British protectorate alongside the Moravians and is "
     "still the second historic church of Bluefields and the Corn Islands; the Jamaican "
     "Baptist mission that worked the same Creole coast from the 1840s, and for which Corn "
     "Island is the obvious candidate; Adventists, who are numerous on the coast; and the "
     "Rama and Garífuna communities of the southern lagoons. **The Anglican and Baptist "
     "reading is the strong one and it is still not acted on** — the cell is not split, "
     "because INIDE publishes no composition for it and §14.4 forbids inventing one. A "
     "residual that resolves this cleanly onto one region is worth saying out loud and is "
     "not worth guessing at.\n\n"
     "Note that this cell also has to hold everything the seven named boxes miss "
     "everywhere else in the country, where it runs at a few tenths of a percent — so it "
     "is a coastal category and a national dustbin at the same time. Per source, per spec "
     "§3.11."),

    # --- added 2026-09-07 with Germany's ESS split.
    ("other.de",
     "Other religion (Germany)",
     "Two ESS answers together — `Östliche Religionsgemeinschaft` and `Andere "
     "nicht-christliche Religionsgemeinschaft` — 1.04% of Germany between them, and the "
     "same shape `other.fr` and `other.gr` already hold for the same survey. The tree has "
     "no node for 'some Eastern religion, unspecified' and choosing one would invent a "
     "fact.\n\n"
     "**What is unusual here is that a census once counted what this cell contains, and "
     "the count is unusable.** Zensus 2011's Frage 8 offered Buddhism and Hinduism as "
     "their own answers — but it was the only voluntary question on the form, it was put "
     "only to people who had already said they belonged to no public-law body, and "
     "destatis' own verdict is that no reliable statement about world religions can be "
     "made from it (sources/de.md §2). So the boxes existed, were printed, were answered "
     "by a self-selected fraction, and left nothing this map can use.\n\n"
     "Its content is largely knowable even though it is not separable: Germany's "
     "Buddhists, both the Vietnamese and Thai communities and a substantial German "
     "convert population; the Hindu communities of the Tamil diaspora in Nordrhein-"
     "Westfalen and the Afghan Hindus; the Yazidis, of whom Germany has the largest "
     "diaspora anywhere and who are neither Eastern nor Christian and will be scattered "
     "across both answers; and the Bahá'ís. Per source, per spec §3.11."),

    # --- added 2026-09-07 with Peru.
    ("other.pe",
     "Other religion (Peru)",
     "INEI's `Otra` — 94,150 people, **0.41%**, and one of the smallest residuals on this "
     "map, because Peru's census does the unusual thing of naming five non-Catholic "
     "Christian bodies rather than sweeping them into one box. Católica, Evangélica, "
     "Cristiano, Adventista, Testigo de Jehová and Mormones all have their own answers, so "
     "what falls into `Otra` is genuinely the remainder rather than the usual mixture of "
     "'not Catholic' and 'not classified'.\n\n"
     "**Its geography is one thing and it is not the cities.** Nationally 0.41%; in Yavarí "
     "**20.7%**, San Pablo 18.9% and Pebas 12.0% — all three Amazon river districts in "
     "Loreto on the Brazilian and Colombian frontier — and Tournavista 19.2%, Puerto Inca "
     "9.1% (Huánuco) and Iberia 13.0% (Madre de Dios), which are the Amazon colonisation "
     "zones. That is a spread of fifty to one against a national figure under half a "
     "percent, and by §9r's rule a residual that sharp is a **missing category rather than "
     "a mixture**.\n\n"
     "**ONE CANDIDATE WAS LISTED HERE AND HAS BEEN MEASURED AWAY.** This node first offered "
     "the *indigenous religions of the Amazon* as plausible content, on the grounds that the "
     "census gives them no box — hedged with the observation that in the Awajún and "
     "Asháninka districts the answer that rises is `Ninguna` and not `Otra`. INEI serves the "
     "self-identified ethnicity variable `C5P25` over the same universe, so the hedge was "
     "testable, and `sources/pe_ethnicity.py` tested it: **`Otra` is 0.40% among people who "
     "identify as native or indigenous of the Amazon against 0.41% nationally — 0.98x, and "
     "a district correlation of r=0.07.** Flat. Whatever is in this cell, it is not them, "
     "and the `Ninguna`/`Otra` distinction is a real difference between two ways of having "
     "no listed religion rather than a coincidence of geography.\n\n"
     "**What is left is a better-shaped answer.** The **Israelitas del Nuevo Pacto "
     "Universal**, a Peruvian millenarian church founded in 1968 whose settlement colonies "
     "are concentrated in exactly these Amazon frontier districts and whose members are "
     "Andean migrants rather than Amazonian peoples, fit the geography and the ethnicity "
     "together. And the cell's most concentrated groups by share are the **Tusán (5.51%) "
     "and Nikkei (3.50%)** — Peru's Chinese and Japanese populations, where Buddhism, Daoism "
     "and Shinto have no box either — though at 22,534 and 14,307 people they are only 1.7% "
     "of the cell between them; its bulk is Mestizo (49,126) and Quechua (25,046). "
     "INEI publishes no breakdown, so it stays whole. Per source, per spec §3.11 and §14.4."),

    # --- added 2026-09-08 with Fiji.
    ("other.ki",
     "Other religion (Kiribati)",
     "Two cells of Kiribati's 2015 table, **918 people between them, 0.83%**. NSO's own "
     "`Other` is 832 of that, printed after thirteen named answers including the Baha'is "
     "(2,314), the Muslims (139) and `No religion`, so it is a genuine tail rather than a "
     "place the non-Christian religions were put.\n\n"
     "**The other 86 are `Te Ran`, and nothing published says what it is.** It is a printed "
     "census cell in 2015 and again in the 2020 census (89 people), so the office treats it "
     "as a body worth naming, but no census report, no Census Atlas note and no reachable "
     "secondary source identifies it; the name is Gilbertese. It is here rather than on "
     "`christianity.other` because filing it as Christian would be a claim this project "
     "cannot support (§14.4 rule 1), and it is not merged silently — the REVIEW note in "
     "`taxonomy/ki2015.py` says the search that failed. At 0.08% it draws no dot at 1 dot = "
     "1,000 people either way."),
    ("other.to",
     "Other religion (Tonga)",
     "TSD's `Other minor religious groups` — **714 people, 0.72%**, and it is a genuine tail "
     "rather than a place the non-Christian religions were put. G 20 prints the **Baha'i "
     "Faith (730), Hinduism (78), Islam (60) and Buddhist (58) as four lines of their own**, "
     "which very few censuses of a country this size do, and it also names the Salvation "
     "Army, the Anglicans and the Jehovah's Witnesses at three to six hundred people each. "
     "So the list was already twenty-one deep before anything reached this cell.\n\n"
     "Its geography says the same. The strongest district anywhere is **'Eua Fo'ou at "
     "2.26%** against 0.72% nationally, and that is 48 people; no district reaches three "
     "percent and most round to nothing. A residual hiding a real body would not look like "
     "that. Tonga is **99.88% drawn** — only 119 people in the country refused the "
     "question."),
    ("other.fj",
     "Other religion (Fiji)",
     "FBoS's `Other religion` — **1,294 people, 0.15%**, and one of the smallest residuals on "
     "this map for a reason worth stating: Fiji's 2007 census does not sweep non-Christians "
     "into it. **Hindu (232,103), Moslem (52,594) and Sikh (2,548) each have a printed line "
     "of their own**, which is why this cell is a rounding error rather than the 28% of the "
     "country that is not Christian. Very few censuses anywhere name Sikhs separately; Fiji "
     "does, because the Punjabi minority within the indenture-era migration is old enough and "
     "distinct enough to have been counted since the colonial censuses.\n\n"
     "What is left in it is genuinely small and genuinely unclassified: the **Bahá'í** "
     "community, which has a documented Fijian presence; **Buddhists and Chinese religion** "
     "among the small Chinese-Fijian population (4,704 people identified as full or part "
     "Chinese in the same census); and whatever iTaukei respondents meant who did not choose "
     "one of the eighteen named churches or `No religion`. FBoS publishes no breakdown of the "
     "cell, so it stays whole. Per source, per spec §3.11 and §14.4."),

    # --- added 2026-09-08 with Guatemala, the first country here drawn from LAPOP.
    ("other.gt",
     "Other religion (Guatemala)",
     "**Two LAPOP answers land here and neither is a residual in the usual sense**: `Otro` "
     "(1.14%) and `Religiones Orientales no Cristianas` (0.31%). They are kept apart in "
     "`source_category` so a later source can move either, and they are merged in the tree "
     "because nothing in the survey separates what is in them.\n\n"
     "This cell is small for the opposite reason to Peru's. `other.pe` is 0.41% because "
     "INEI names five non-Catholic Christian bodies; `other.gt` is 1.45% because LAPOP's "
     "card is a WORLDWIDE instrument, so it carries boxes for Judaism, Islam, Hinduism and "
     "the Witnesses that almost nobody in Guatemala ticks, while the categories a Guatemalan "
     "questionnaire would have printed are absent. **The bucket is small and the hole is "
     "somewhere else**: `Religiones Tradicionales` at 0.22% in a country that is 43.6% "
     "indigenous by its own census. sources.md §11ad measured that instrument failure "
     "against Suriname's census and `taxonomy/gt2023.py` carries the argument.\n\n"
     "What is plausibly inside it: the Baha'i community, which has a documented Guatemalan "
     "presence; the small Chinese and Korean populations of the capital; and respondents who "
     "would have said *costumbre* to an interviewer who had a box for it. None of that is "
     "separable here, so per spec §3.11 it stays whole."),

    # --- added 2026-09-08 with El Salvador, the second LAPOP country.
    ("other.sv",
     "Other religion (El Salvador)",
     "The same two LAPOP answers as `other.gt` — `Otro` (0.93%) and `Religiones Orientales "
     "no Cristianas` (1.42%) — kept apart in `source_category` and merged here because "
     "nothing in the survey separates what is in them.\n\n"
     "**At 2.35% this is the largest `other` cell of any LAPOP country in Central America**, "
     "and nearly two-thirds of it is the Eastern-religions box rather than the residual one, "
     "which is the reverse of Guatemala. What can be said about the contents is one thing "
     "and it is a reading rather than a measurement: **El Salvador has the region's oldest "
     "Palestinian and Lebanese communities**, arrived from the 1890s, and while the "
     "Levantine Christians among them would answer Orthodox or Catholic, a Salvadoran Muslim "
     "has no box on this card at all and `Religiones Orientales no Cristianas` is where an "
     "interviewer working from it would most likely land. No source here can split the cell, "
     "so per spec §3.11 and §14.4 it stays whole."),

    # --- added 2026-09-08 with Ecuador, the third AmericasBarometer country.
    ("other.ec",
     "Other religion (Ecuador)",
     "The same two LAPOP answers as `other.gt` and `other.sv` — `Otro` (2.05%) and "
     "`Religiones Orientales no Cristianas` (0.35%) — kept apart in `source_category` and "
     "merged here because nothing in the survey separates what is in them.\n\n"
     "**This cell is not comparable with the other two, because in Ecuador's pool the "
     "`Otro` box did not exist for half the waves and then changed meaning.** Code 77 is "
     "exactly zero in the 2010, 2012 and 2014 rounds across all 28 countries of the merge, "
     "on 29,374 / 27,254 / 36,725 valid answers, and appears from 2016. Meanwhile codes 6, "
     "10 and 12 — Mormons, Jews and Jehovah's Witnesses — are exactly zero in 2018 and 2023 "
     "on 15,107 and 25,649 answers. Zero Witnesses among 25,649 Latin Americans is a "
     "withdrawn box and not a measurement. So **this node is inflated at the late end of the "
     "pool by people who would have had their own answer earlier**, and "
     "`christianity.witnesses` and `christianity.latterday` are correspondingly floors. "
     "Ecuador's `Otro` runs 0%, 0%, 3.22%, 5.02% across its four rounds, and the 1.80-point "
     "rise between the last two is close to the 1.81% those three withdrawn cells hold — "
     "suggestive, on n=1,545, rather than a decomposition.\n\n"
     "What is plausibly inside it beyond that: Ecuador's Baha'i community, which is among "
     "the larger ones in South America; the Chinese and Lebanese populations of Guayaquil; "
     "and — the reason this cell is worth watching rather than dismissing — **the "
     "Kichwa, Shuar, Achuar and Waorani practice that `Religiones Tradicionales` fails to "
     "catch at 0.06% in a country that is 7.69% indigenous.** `taxonomy/ec2023.py` has that "
     "argument. None of it is separable here, so per spec §3.11 it stays whole."),

    # --- added 2026-09-08 with South Africa.
    ("other.za",
     "Other religion (South Africa)",
     "Stats SA's `Other` in Community Survey 2016 table 2.10a — 1,482,210 people, 2.70% of "
     "the answers given. The survey gives its own box to Islam, Hinduism, Judaism, "
     "Buddhism, Bahaism and Traditional African religion, and splits Christianity fourteen "
     "ways besides, so this cell is what is left after an unusually long card. The bodies "
     "with nowhere else to go are South Africa's Rastafari, who have no box and are the "
     "largest unlisted group in the country; the Sikh and Jain communities of Durban and "
     "Johannesburg; Chinese folk religion; and the Zoroastrians of the old Parsi trading "
     "families.\n\n"
     "**Do not read it as a small tail, because it is not small and those bodies do not "
     "fill it.** At 2.70% it is larger than Islam (1.62%) and about the size of Islam, "
     "Hinduism, Judaism, Buddhism and Bahaism put together (2.79%). Nothing on the list "
     "above plausibly reaches 1.5 million people, so most of this cell is unaccounted for "
     "rather than merely unnamed.\n\n"
     "**Its geography does not help either, and that is worth saying rather than "
     "glossing.** It runs 3.87% in Gauteng and 2.90% in Mpumalanga against 1.18% in North "
     "West and 1.27% in Northern Cape, a spread of about three to one, where this map's "
     "other residuals commonly run ten or twenty to one and point straight at what is "
     "inside them (§9r's Chittagong rule, and `other.ao` next door). A flat residual names "
     "nothing — but flatness is also what a cell looks like when it is distributed roughly "
     "with population, which is what a catch-all does, so it is not evidence either way.\n\n"
     "**And its size is one of the places the two Stats SA instruments disagree.** Census "
     "2022's equivalent `Other Faiths` cell is 1.0% of that count against 2.70% here. Six "
     "years does not move a residual by a factor of two and a half; see `sources/za.md` "
     "§3. Per spec §3.11, it is drawn whole because nothing separates it."),

    # --- added 2026-09-08 with Angola.
    ("other.ao",
     "Other religion (Angola)",
     "INE's `Outra religião` — 435,663 people, 1.26% of the 2+ population, and a small "
     "residual because the twenty boxes above it are unusually many. "
     "**Its geography names most of what is in it**, on §9r's Chittagong rule. Zaire "
     "province is **7.62%** against 1.26% nationally, peaking at **Luvo 13.9%, Mbanza "
     "Kongo 11.0% and Cuimba 10.3%** — Mbanza Kongo is the old capital of the Kongo "
     "kingdom and Luvo is the border post on the road to Matadi. That corner of Angola "
     "carries dozens of small Kongo prophetic churches, and the census names only three "
     "African-founded bodies, so the northern half of this cell is "
     "`christianity.africaninstituted` that could not be brought out. A second cluster in "
     "interior Cuanza Sul (Ebo 12.0%, Condé 10.8%) has no such obvious reading. "
     "Per source, per spec §3.11."),
    ("other.sg",
     "Other religion (Singapore)",
     "SingStat's `Other Religions` — 9,827 people, 0.28%, and one of the smallest residuals "
     "on this map because the eight boxes above it are unusually many. **Sikhism, which most "
     "censuses fold into a cell like this one, has its own at 12,051**, so this is a tail "
     "and not a store cupboard: Singapore's Jewish, Baha'i, Jain and Zoroastrian communities "
     "are the bodies with nowhere else to go, and none is separately counted anywhere. "
     "**Its geography is the wealthy central belt and nothing else.** 1.13% in the "
     "un-named `Others` planning areas, 0.96% in River Valley, 0.88% in Marine Parade and "
     "0.78% in Tanglin, against 0.15% in Ang Mo Kio — three times the national rate in the "
     "private-housing districts, which is where a small expatriate and old-mercantile "
     "population would be. Per source, per spec §3.11."),
    ("other.at",
     "Andere nichtchristliche Gemeinschaften (Austria)",
     "The Volkszählung 2001's cell for every non-Christian answer that is not Jewish or "
     "Muslim — 19,750 people, 0.25%. **It is a mixture and not a tail**, and UNSD's national "
     "table decomposes it exactly: Buddhists 10,402, Hindus 3,629, Sikhs 2,794, `other "
     "religions` 1,745, Bahá'í 760, Unification 297, Shinto 123. Buddhism is more than half "
     "of it, which is worth saying because Austria recognised Buddhism as a public-law "
     "Religionsgesellschaft in 1983, the first state in Europe to do so. Those figures exist "
     "at national level ONLY, so the cell is drawn whole rather than split; "
     "`taxonomy/at2001.py` has that argument. **Its geography is Vienna and almost nothing "
     "else**: 0.67% across the city against 0.09% in Burgenland, peaking at 1.18% in "
     "Mariahilf and 1.17% in Margareten, the dense inner districts inside the Gürtel. "
     "Per source, per spec §3.11."),
    ("other.cy",
     "Other religion (Cyprus)",
     "CYSTAT's `Other Religion` in the 2021 census — 4,545 people, 0.49%. **A tail rather "
     "than a store cupboard**, because Cyprus names Buddhism, Sikhism and Hinduism as their "
     "own answers, so the usual contents of a cell like this are already out of it. What is "
     "left is Bahá'í, Yazidi, and any Alevi or Ahmadi answer that did not go in the Muslim "
     "cell — and, oddly, **Judaism, which the census has no row for at all**: the same "
     "census counts a Jewish community in its ethnic/religious-group table and 885 Hebrew "
     "speakers in its language table, but the religion question's twelve categories do not "
     "include one. **It has no geography of its own**, because nothing in Cyprus does: the "
     "religion table is national and every community figure on this map is allocated from "
     "citizenship composition, so `other.cy` reads 0.29% to 0.95% across the island purely "
     "as a function of how many non-Cypriots live there. Per source, per spec §3.11."),
    ("other.bw",
     "Other religion, NEC (Botswana)",
     "Statistics Botswana's `Other religion (NEC)` in the 2011 census — 1,416 people, "
     "0.10% of everyone who answered, and **one of the smallest residuals on this map**. "
     "It is a tail rather than a store cupboard because the eight boxes above it are "
     "unusually generous for an African census: Islam, Hinduism, the Bahá'í Faith and "
     "Rastafari all have their own row, and so does Badimo, so the answers that usually "
     "fill a cell like this have somewhere else to go. What is left is Judaism, Buddhism, "
     "Sikhism and whatever an enumerator could not place. "
     "**Its geography is the seven towns and very little else**: two thirds of it is in "
     "Gaborone, Francistown, Lobatse, Selebi Phikwe, Orapa, Jwaneng and Sowa, which "
     "between them hold a quarter of the people the country draws. Per source, per "
     "spec §3.11."),

    # --- added 2026-09-08 with Armenia.
    ("other.am",
     "Other religion (Armenia)",
     "Armstat's `Այլ` in the 2022 census — 7,673 people, 0.26%, and it holds the nine "
     "Transcendental Meditation adherents of Ararat as well, because the tree has no node "
     "for that movement and nine people do not justify making one. **It is a tail in the "
     "marzes that print a long list and a store cupboard in the marzes that do not**, which "
     "is unusual and is a property of how Armstat typeset these tables rather than of "
     "Armenia: a marz prints only the columns it has people in, so Syunik's list runs to "
     "five religions and Yerevan's to fourteen, and every answer without a column in a given "
     "marz falls into that marz's residual. Armenia's 515 Muslims are the clearest case — "
     "483 are printed in the four marzes with a Muslim column and the other 32 are inside "
     "this cell somewhere among the other seven. Across all eleven marzes 169 people arrive "
     "here that the national table counts under a name. "
     "**Its geography is the Ararat plain**, at 0.66% in Armavir, 0.56% in Aragatsotn and "
     "0.46% in Ararat against 0.01% in Syunik, and those are the three marzes where "
     "Armenia's Yazidis live: the national table cuts religion by ethnicity and puts 3,246 "
     "of the 7,675 in this cell under `Yezidi`, so about two fifths of the residual is a "
     "Yazidi answer that was neither `Shar-fadinian` nor `Pagan`. Per source, per spec "
     "§3.11."),

    # --- added 2026-09-08 with Finland.
    ("other.fi",
     "Other religion (Finland)",
     "Two ESS answers together, `Eastern religions` and `Other Non-Christian religions`, "
     "pooled over rounds 5 to 11. The tree has no node for 'some Eastern religion, "
     "unspecified' and choosing one would invent a fact, so both sit here as they do for "
     "Greece and Croatia. **Finland is a country where this cell has a named occupant the "
     "instrument cannot reach**: the Buddhist and Hindu communities are small and mostly "
     "immigrant, but Suomenusko, the Finnish native-faith revival, is a registered religious "
     "community and is exactly the sort of answer that lands in `Other Non-Christian "
     "religions` rather than in a box of its own. Estonia's census names Maausk and Taarausk "
     "and this survey cannot, which is the difference between a country with a census "
     "question and one with only a register. Per source, per spec §3.11."),
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
    "unknown":
        "Not a report of no religion. These people were counted and the source does not "
        "establish what they practise — in Vietnam because the census asks only which "
        "state-registered organisation someone belongs to, so this holds ancestor "
        "veneration, folk religion and most Buddhist practice alongside the genuinely "
        "irreligious, and cannot tell them apart.",
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
            #
            # The New Apostolic Church is the one that is not American: Edward Irving's
            # London congregation of the 1830s, restoring the apostolate rather than the
            # apostolic congregation, and reorganised at Hamburg in 1863. Same claim, same
            # decade, different continent, so it sits at the head of the group beside
            # Stone-Campbell rather than getting a group of its own for one node.
            "christianity.newapostolic",
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
        # PROMOTED OUT OF `No single line` TO A GROUP OF THEIR OWN — 2026-09-06, Anita:
        # "have some yellow color be reserved for things similar to filipino independent
        # churches… is there anything its similar to we can just group it with?"
        #
        # There is, and this file had already said so twice without acting on it: both
        # `filipinoindependent` and `africaninstituted` carry the sentence "the same idea as
        # `christianity.maori`". All three were then drawn in three unrelated colours,
        # including a lime green for the Maori churches — so the tree asserted a grouping the
        # map did not show. This group is the assertion made visible: §6.14 draws a lineage
        # group as ONE colour, so the three now share a reserved gold (OVERVIEW_FLAT).
        #
        # WHAT THE GROUP IS. Churches founded by local converts in a mission field, outside
        # missionary control, and belonging to none of the imported families afterwards. It
        # is a standard bloc in the study of world Christianity — Barrett's *World Christian
        # Encyclopedia* calls it "Independents" — and Kenya's own census category, "African
        # Instituted Churches", is the same construction the Philippine and Maori ones are.
        #
        # AND WHAT IT IS NOT, because the group is easy to over-read. It is defined by a
        # church's RELATION TO A MISSION, which is a fact about colonial history rather than
        # about doctrine, and the three do not share a theology: the African and Maori
        # churches are broadly trinitarian mission-church breakaways that added prophecy and
        # healing, while the Philippine node is explicitly the nontrinitarian restorationist
        # remainder (see its own note — the trinitarian Filipino-founded churches are filed
        # with the global streams they belong to, and the Aglipayans sit under
        # `catholic.independent`). On doctrine `maori` and `africaninstituted` are the close
        # pair. A shared colour asserts the historical grouping, which is what "Ancient
        # communions" and "Reformation" assert too; it does not assert a creed.
        ("Locally founded churches", [
            # Oldest first: Ringatu 1868, the Aladura and Zionist churches from the 1880s,
            # Iglesia ni Cristo 1914.
            "christianity.maori",
            "christianity.africaninstituted",
            "christianity.filipinoindependent",
            # Added 2026-09-08 with the Solomon Islands. The Christian Fellowship Church is
            # Silas Eto's 1960 break with the Methodist mission on New Georgia, which is the
            # `maori` and `africaninstituted` shape exactly: a mission-church breakaway led
            # by a local prophet, belonging to nothing imported afterwards.
            "christianity.melanesianindependent",
        ]),
        ("No single line", [
            # Bodies that are a union of several traditions, congregations that decline the
            # question, and the source answers that name no body. See the note on
            # `christianity.protestant`: there is no Protestant super-node, and this group is
            # not one either.
            "christianity.united",
            "christianity.nondenominational",
            "christianity.messianic",
            # The nodes that hold an ANSWER rather than a body, last and together.
            # `sdabaptistapostolic` is the third of them and the only one that names the
            # traditions it merges — added 2026-09-06 with Malawi.
            "christianity.protestant",
            "christianity.evangelical",
            "christianity.sdabaptistapostolic",
            "christianity.other",
        ]),
    ],

    # Ordered by the date the movement separated, which for Judaism is also the order the
    # names are usually given in. Chabad is a Hasidic court inside Orthodoxy rather than a
    # movement beside it, and is placed with Orthodox for that reason.
    #
    # `Non-rabbinic` was added 2026-09-05 with Lithuania and goes FIRST, because the order
    # here is by date of separation and the Karaites left in the eighth century — a
    # thousand years before every other group below. The three groups under it were
    # implicitly rabbinic already; naming this one makes that explicit rather than changing
    # it.
    # `By observance` was added 2026-09-07 with Israel and breaks the ordering principle
    # above on purpose: it is not a date of separation, because these four are not
    # separations. They are how observant a household is, which is the question Israel asks
    # and the movements are the question it does not — see the block above them in BRANCHES.
    # It is placed immediately after `Traditional` so that Haredi and Dati sit next to
    # Orthodox in the legend and in the palette, which is where a reader will look for them
    # (§6.5 puts colour on descent, and in doctrine that is exactly where they descend from).
    "judaism": [
        ("Non-rabbinic", ["judaism.karaite"]),
        ("Traditional", ["judaism.orthodox", "judaism.chabad", "judaism.conservative"]),
        ("By observance", ["judaism.haredi", "judaism.dati",
                           "judaism.masorti", "judaism.hiloni"]),
        ("Liberal", ["judaism.reform", "judaism.reconstructionist"]),
        ("No single line", ["judaism.independent"]),
    ],

    # The three vehicles, oldest first. Vajrayana is carried inside Mahayana historically and
    # is counted beside it, which is the §2.1 split showing up in one line.
    "buddhism": [
        ("Vehicles", ["buddhism.theravada", "buddhism.mahayana", "buddhism.vajrayana"]),
        # Added 2026-09-06 with Vietnam, and it is the first time a source has handed this
        # file a WHOLE LINEAGE rather than one more member of an existing group. The four
        # are one documented descent in the Mekong Delta — Đoàn Minh Huyên's 1849 teaching,
        # Ngô Lợi's 1867 movement out of it, Huỳnh Phú Sổ's 1939 founding out of it again,
        # and the Tà Lơn body registered last — and they are listed in that order, which is
        # what this structure is for. In BRANCHES they are flat siblings under `buddhism`,
        # because containment is about people now and a Hòa Hảo follower is not a Bửu Sơn Kỳ
        # Hương one. Descent here, containment there: spec §2.1's two relations, and the
        # cleanest example of the split anywhere in this file.
        ("Bửu Sơn Kỳ Hương lineage", [
            "buddhism.buuson",
            "buddhism.tuan",
            "buddhism.hoahao",
            "buddhism.talon",
        ]),
        # Not a vehicle and not a transmission: a 1916 Korean founding that calls itself
        # Buddhist reform, and a 1934 Vietnamese lay Pure Land association. Their own group
        # so the three vehicles stay the three vehicles.
        ("Modern reform movements", ["buddhism.won", "buddhism.tinhdo"]),
    ],

    # Added 2026-09-05 with Russia, the first source that asks. The succession dispute of
    # 632 is the single division here, so one group and two members in the order they are
    # always named — which is also size order, and for once those do not conflict.
    "islam": [
        ("Branches", ["islam.sunni", "islam.shia"]),
        # Not a branch of the Sunni/Shia split and does not belong beside them: the
        # Ahmadiyya movement is a 19th-century messianic renewal, so it descends from
        # Islam as a whole rather than from either side of a 7th-century succession.
        ("Later movements", ["islam.ahmadiyya"]),
    ],

    # Added 2026-09-08 with Türkiye, the first source here that asks which school. Ordered
    # by the founder's death — Abu Hanifa 767, Malik 795, al-Shafi'i 820, Ibn Hanbal 855 —
    # which is the order they are always given in and is not size order: Shafi'i is second
    # in Türkiye and Maliki, second-largest worldwide, is a rounding error there.
    "islam.sunni": [
        ("Schools of law", [
            "islam.sunni.hanafi",
            "islam.sunni.maliki",
            "islam.sunni.shafii",
            "islam.sunni.hanbali",
        ]),
    ],
    "islam.shia": [
        ("Schools of law", ["islam.shia.jaafari"]),
    ],
}
