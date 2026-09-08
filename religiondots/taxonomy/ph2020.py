"""PSA 2020 Census of Population and Housing religious affiliation -> religiondots taxonomy.

129 named categories on 117 units — the second longest category list on this map after the
US Religion Census, and the only *self-identified* one that long (sources/ph.md). The list
is flat, alphabetical and mutually exclusive: PSA publishes no `Protestant` or `Christian`
parent anywhere, so **every level above the leaf is built here** and nothing in the source
constrains it.

## The shape of the problem, which is not the shape of the other countries'

Three facts about these 129 categories set everything below:

1. **126 of them are Christian, and 123 of those are Protestant or independent.** The whole
   non-Abrahamic world gets three cells — Islam, Buddhist, Tribal religion — and there is
   no Hindu, Jewish, Sikh, Chinese-folk or Bahá'í option at all. The depth follows the
   country's own religious politics rather than world religion, which is spec §2.2 seen
   from an angle no other source here shows.
2. **The tail is Filipino and undocumented.** Roughly sixty of the 129 are Philippine
   evangelical, Pentecostal or charismatic bodies under 50,000 people, and for most of
   them there is no encyclopedia article, no denominational yearbook entry and no
   published account of their doctrine — several are one founder's ministry, registered
   with PSA and known locally. Where the US Religion Census hands you a body's family, PSA
   hands you a name.
3. **The census's own five residuals are already the honest bucket.** `Other Baptists`,
   `Other Methodists`, `Other Protestants`, `Other Evangelical Churches` and `Other
   religious affiliations` mean PSA had somewhere to put the bodies it would not name, so
   a category that IS named is a body PSA judged large enough to name.

## How the undocumented tail is placed, and how confident that is

**It is placed from the body's name and from what the name implies about the stream it
belongs to, and that is weaker evidence than anything else in `taxonomy/`.** Said plainly
so it is not mistaken for research: a fellowship called `Jesus Christ Saves Global
Outreach` is filed as charismatic because Philippine independent ministries with names of
that shape are overwhelmingly charismatic, not because a source says this one is.

Two rules keep that from spreading further than it should:

- **A name that states a tradition is taken at its word** — anything called Baptist,
  Methodist, Pentecostal, Presbyterian, Holiness or Reformed goes to that family. This is
  reliable in the Philippines, where these names carry mission lineages rather than being
  decorative.
- **A name that states nothing goes to the stream, never to a sub-branch of it.** An
  independent ministry with a devotional name lands on `christianity.pentecostal.charismatic`
  — the node for exactly "later movements outside the classical Pentecostal bodies" — or on
  `christianity.nondenominational`, and never on something narrower that would assert a
  doctrine. The split between those two is itself a judgement: a *ministry* or *fellowship*
  built round a founder and healing or revival language reads charismatic; a *Bible church*,
  a *community church* or a mission agency reads non-denominational.

## The geographic check, what it settled, and where it is worthless

129 categories over 117 units is enough to ask a question no single body's name can answer:
**which other bodies does this one sit beside on the ground?** Two churches planted by the
same mission are in the same provinces sixty years later, so provincial co-location is real
evidence about lineage — and it is evidence the name cannot fake.

It has to be done to a share of the province's **named non-Catholic Christian population**,
not of its people. Raw correlation on population shares measures "is this province
Protestant" and nothing else: on that basis Bible Baptist correlates 0.68 with the Seventh
Day Adventists, which is a fact about Mindanao and not about Baptists.

**What it settled, where the neighbours are independently classified:**

- `Bible Baptist Church` → nearest is `Association of Fundamental Baptist Churches`
  (r=0.52). That is the separatist fundamentalist family it was filed in from its name, and
  it is the largest name-based call in the file, so the confirmation is worth having.
- `Universal pentecostal Church` → Pentecostal Church of God Asia Mission (0.56), United
  Pentecostal (0.55), Philippine Pentecostal Holiness (0.48). Classical Pentecostal, as
  filed.
- `Fundamental Grace Gospel Church` → `Things to Come` is in its top four. Those are the
  two Grace Movement bodies in the census and they were grouped on doctrine before this was
  run; finding them adjacent on the ground is the check working.
- **A Caraga cluster of Filipino independent churches**: `Church Body of Christ Filipinista`
  (75% of it in five units), `Philippine Ecumenical Christian Church` (73%) and the
  `Philippine Benevolent Missionaries Association` are mutually nearest neighbours across
  Agusan, Surigao and Dinagat. Three bodies filed on `christianity.filipinoindependent`
  from three separate arguments turn out to occupy one region, which is what a family of
  locally-founded churches should look like.
- `Crusaders of the Divine Church of Christ` is Isabela, Pangasinan, Tarlac, Nueva Ecija,
  Ilocos Norte — the Ilocano corridor, the same ground as the Aglipayans and INC. Also as
  filed.
- One call overturned: `Evangelical Christian Outreach Foundation`, see REVIEW.

**WHERE IT IS WORTHLESS, AND THIS IS THE MORE USEFUL HALF.** Run over all 56 charismatic /
non-denominational calls, the check "disagrees" with 22 of them — and almost every
disagreement is circular, because the nearest neighbours of a small undocumented ministry
are *other small undocumented ministries this file also placed by name*. The vote is then
counting my own guesses back to me. Two further limits:

- **It cannot separate charismatic from non-denominational at all**, because those bodies
  differ by doctrine and agree by geography: both are Metro Manila megachurch networks.
  The check puts `Christ's Commission Fellowship` among the charismatics, and CCF is
  documented as non-denominational. Geography lost, correctly.
- **Co-location has other causes.** `Free Believers in Christ Fellowship` sits nearest the
  Episcopal Church because both are in Mountain Province, which makes it Cordilleran and
  not Anglican.

So the rule this file follows: **act on the check only where the neighbours are bodies whose
classification is known independently of this file, and treat everything else as
description.** On that rule it changed exactly one mapping out of 129, and confirmed six.
That ratio is the honest measure of what the technique is worth.

Everything in REVIEW below is a call that could reasonably go the other way. It is long,
and it should be.

EXCLUDED holds categories that are deliberately not on the tree.
"""

EXCLUDED = {
    "Household Population":
        "the unit's own denominator, emitted by sources/ph.py alongside the 129 categories "
        "so the census's own base travels with the counts (spec §8.2). Not a religion.",
}

REVIEW = {
    # ---- the two that decide how the country reads -----------------------------------
    "Aglipay":
        "-> christianity.catholic.independent, THE SAME NODE AS `Iglesia Filipina "
        "Independiente`, WHICH MERGES THEM. They are one church: the IFI is the Aglipayan "
        "Church, founded 1902 by Gregorio Aglipay, and the census offered both names as "
        "separate response options. Respondents split 818,916 / 640,076 between them. "
        "Summed, 1,458,992 = 1.34% of the country, which makes it the FOURTH largest "
        "religious body in the Philippines and the largest Christian body after the "
        "Catholics, the Muslims and INC. sources/ph.py deliberately left them apart at "
        "ingest (§2.4, keep the source raw); this file is where they come back together, "
        "and it is the right place — the merge is a claim about the world, not about the "
        "file. If it is ever wrong it is wrong in one direction only: some `Aglipay` "
        "answers may mean the several small Aglipayan splinter jurisdictions rather than "
        "the IFI proper, and those are independent Catholic too, so they land here either "
        "way.",
    "Church of Christ":
        "-> christianity.restorationist (Stone-Campbell), 429,921 people, AND THIS IS THE "
        "LARGEST GENUINELY UNCERTAIN CALL IN THE FILE. `Church of Christ` is the English "
        "name of Iglesia ni Cristo, which is a separate census category of 2,806,524, so "
        "the two readings are: (a) Stone-Campbell Churches of Christ, which are real and "
        "numerous in the Philippines — `Christian Missions in the Philippines`, 183,584 "
        "and mapped to the same node, is the direct-support mission of that movement; or "
        "(b) an overflow of INC answers given in English. Filed as (a) because a Filipino "
        "respondent naming INC has the Filipino name available on the same form and "
        "overwhelmingly uses it, and because a 430,000-strong Stone-Campbell presence is "
        "consistent with a century of American Restoration Movement mission work. If (b) "
        "is right, `christianity.restorationist` is overstated by up to 430,000 and "
        "`christianity.filipinoindependent.inc` understated by the same. No published "
        "table separates them. "
        "TWO DISTRIBUTIONAL TESTS WERE RUN AND BOTH CAME BACK EMPTY, which is itself the "
        "finding. Geographic affinity puts it at r=0.40 with INC against r=0.35 with "
        "`Christian Missions in the Philippines` — a gap far too small to decide on, and "
        "its own three largest concentrations are Cotabato, Isabela and Cagayan while the "
        "Restoration Movement's Philippine heartland is the Manila ring, where it is also "
        "strong. Ubiquity is worse: `Church of Christ` is in all 117 units with 58.0% of "
        "itself in its top twenty, and INC is in all 117 with 57.9%, which looks like a "
        "match until you notice the Adventists are 117 and 54.0% and the Bible Baptists "
        "116 and 51.1%. **At 117 units of ~929,000 people each, every body over about "
        "400,000 is present everywhere**, so the test measures size and not spread. This "
        "is §3.9's coarse-geography trade arriving as a taxonomy problem: the question is "
        "decidable at municipality level and the Philippines does not publish one.",

    # ---- Catholic -------------------------------------------------------------------
    "Catholic Charismatic":
        "-> christianity.catholic.latin, the same node as the Roman Catholics, which "
        "merges 74,096 people back into the 85.6M they were carved out of. They are Roman "
        "Catholics: the Charismatic Renewal is a movement inside the Latin church, not a "
        "body beside it, and its members receive the Latin sacraments from Latin clergy. "
        "PSA separates them because the press release's 78.8% headline excludes them "
        "(sources/ph.md §5), which is a fact about the headline. The alternative — filing "
        "them on `christianity.pentecostal.charismatic` — would assert they had left the "
        "Catholic church, which they have not.",
    "Oblates of Mary Immaculate, Incorporated":
        "-> christianity.catholic.latin. 528 people, and the strangest row in the census: "
        "the OMI are a Catholic missionary ORDER, and an order is a slice of its parent "
        "rather than a layer beside it (spec §3.2). That argument was written about "
        "membership rolls, where double counting is the risk; here it arrives inside a "
        "self-id census where the 528 are genuinely NOT also counted as Roman Catholic, so "
        "the partition is real and only the placement is odd. They are Latin Catholics who "
        "answered with their order, and they are filed as Latin Catholics.",
    "Apostolic Catholic Church, Inc.":
        "-> christianity.catholic.independent. Founded 1992, Catholic in rite and orders, "
        "not in communion with Rome. 54,543. Not the Irvingite Catholic Apostolic Church, "
        "which is a different body with a near-identical name and no Philippine presence.",

    # ---- unions ---------------------------------------------------------------------
    "Iglesia Evangelica Unida de Cristo":
        "-> christianity.united, THE SAME NODE AS `Unida Evangelical Church`, which merges "
        "them — a second duplicate pair, smaller than Aglipay/IFI and the same kind of "
        "thing. Both name the Unida Church, founded 3 January 1932 out of the merger of "
        "six Filipino evangelical groups of Presbyterian and Methodist background. A union "
        "across Protestant families whose result is not a branch of any one of them is "
        "precisely what `christianity.united` is for, so it sits beside the UCCP rather "
        "than under Methodist or Reformed. 13,424 + 5,051 = 18,475.",

    "United Evangelical Church of the Philippines (Chinese)":
        "-> christianity.united. 39,778, and THE ONLY CATEGORY IN THE CENSUS DEFINED BY "
        "ETHNICITY — PSA puts `(Chinese)` in the label, which it does for no other body. "
        "Founded in Manila in 1929 as the Chinese United Evangelical Church, out of "
        "congregations from several mission backgrounds, and now the largest Chinese "
        "Filipino Protestant body; the union origin is why it sits beside the UCCP rather "
        "than on `christianity.nondenominational`, which is the other reading and would be "
        "closer to how it operates today. Worth holding beside `Other religious "
        "affiliations`: this is the census's one Chinese-Filipino cell, and it is a "
        "Protestant one, so the Chinese folk religion and the Buddhist-Daoist temple "
        "practice of the same community have nowhere to go but the residual.",

    # ---- the two that are neither Protestant nor Catholic ---------------------------
    "Union Espiritista Cristiana de Filipinas, Incorporated":
        "-> spiritualism.kardecist. 132,476 people, which makes the Philippines the second "
        "Kardecist country on this map after Brazil and by a long way the smaller. Founded "
        "1905 and explicitly in Allan Kardec's line, so it belongs on the Kardecist child "
        "rather than on `spiritualism`, the Anglo-American node. Filed outside Christianity "
        "despite `Cristiana` in the name, for the same reason Brazil's is: Kardecism is a "
        "distinct 19th-century movement and its Christian vocabulary does not make it a "
        "Christian denomination.",
    "Philippine Benevolent Missionaries Association":
        "-> christianity.filipinoindependent, AND THIS PLACEMENT IS CONTESTED. PBMA, "
        "founded 1965 on Dinagat by Ruben Ecleo Sr, 11,283 people, is usually grouped by "
        "Philippine writers with the Rizalista movements and the other folk religious "
        "sects rather than with the churches — it describes itself as a non-sectarian "
        "fraternal and charitable association, and its members venerate its founder. It is "
        "filed here because it is Filipino-founded, outside every imported tradition, and "
        "Christian in idiom, which is this node's definition; the honest alternative is a "
        "`filipinofolk` family of its own holding PBMA, the Rizalista sects and the Bell "
        "Church. That was not created because no source on this map counts the other "
        "members — the census has no cell for any of them — and a family with one occupant "
        "and no prospect of a second is a node that exists rather than one that is "
        "countable (spec §2). Revisit if a Philippine source ever enumerates the sects.",

    # ---- the residuals and the non-answers ------------------------------------------
    "Other Evangelical Churches":
        "-> christianity.protestant, the `Protestant, unspecified` node, together with "
        "`Other Protestants`. 254,489 + 332,173 = 586,662 people whose answer named a "
        "stream and no body, which is exactly what that node holds. NOT filed on "
        "`christianity.other`, which is for bodies that have no branch — these are not "
        "bodies at all.",
    "Other Baptists":
        "-> christianity.baptist, the parent, which is §6.6's branch that carries dots. "
        "361,332 people who are Baptists of an unnamed convention, and the parent is the "
        "only truthful place for them: any child would assert a polity the source did not "
        "record. Same for `Other Methodists` (49,179) on christianity.methodist.",
    "None":
        "-> unaffiliated. 43,931 people, 0.040% of the country, and the number is worth "
        "stopping on. Four hundredths of one percent is not a measurement of Philippine "
        "irreligion; it is a measurement of what a proxy-reported household question about "
        "affiliation produces in a country where affiliation is assumed. One household "
        "member answered for everyone (sources/ph.md §3), and 'none' is a harder answer to "
        "give on someone else's behalf than the name of the church the family belongs to. "
        "Drawn, because the census asked and 43,931 people are the answer, but no reading "
        "of this map should treat it as the country's unaffiliated population.",
    "Not reported":
        "-> unrecorded. 15,186 people, 0.014%. Nobody's religion was recorded, which is "
        "what that node is for; it is not a report of no religion and is deliberately not "
        "pooled with `None`.",

    # ---- calls made from the name, largest first ------------------------------------
    "Bible Baptist Church":
        "-> christianity.baptist.independent. 540,364 people, the largest Protestant body "
        "in the country and the largest call made from a name in this file. `Bible "
        "Baptist` in the Philippines is the Baptist Bible Fellowship International "
        "lineage, whose Philippine work is extensive and which is separatist fundamentalist "
        "in exactly the sense that node describes. If it is wrong it is wrong one step "
        "narrow — the body is certainly Baptist, and `christianity.baptist` would be the "
        "safe retreat.",
    "Alliance of Bible Christian Communities of the Philippines":
        "-> christianity.nondenominational. 236,408. Founded 1972 as the Association of "
        "Bible Churches of the Philippines, out of church planting by four interdenominational "
        "faith missions — OMF, SEND, World Team and TEAM. A network of independent Bible "
        "churches whose founding agencies were deliberately not denominational is the "
        "clearest case for that node in the file. NOT the Christian and Missionary "
        "Alliance, which is a separate category of 327,537 and a separate body.",
    "Church of God World Missions in the Philippines":
        "-> christianity.pentecostal.trinitarian. 172,440. The Philippine work of the "
        "Church of God (Cleveland, Tennessee), which is Holiness-Pentecostal; filed on the "
        "Pentecostal side because that is where the body's own identity sits and where "
        "usrc2020.py puts its American parent.",
    "Evangelical Christian Outreach Foundation":
        "-> christianity.nondenominational, 115,626, and THE ONE CALL THE GEOGRAPHIC CHECK "
        "OVERTURNED. It was filed charismatic on the naming rule, wrongly. Its nearest "
        "geographic neighbours are the Evangelical Free Church of the Philippines "
        "(r=0.63) and the Evangelical Presbyterian Church (r=0.56) — both independently "
        "classified, so not a circular vote — and its distribution is a tribal-mission "
        "one: 18% of the body is in Sarangani, where it is 3.7% of the province, with "
        "Davao del Sur, Davao Occidental and Oriental Mindoro behind it. That sent me "
        "back to look again, and the body turns out to be documented after all: ECOF, now "
        "Evangelical Christian Outreach Fellowship International, grew out of Rev. Keith "
        "Williams' Midwest Evangelistic Association work from 1954 among the tribal "
        "peoples of Mindanao, and it runs a Blaan congregation in Malungon, Sarangani — "
        "which is the Sarangani number. An independent faith mission with no "
        "denominational parent is what `christianity.nondenominational` is for.",
    "Fundamental Grace Gospel Church of the Christ in the Philippines, Inc.":
        "-> christianity.nondenominational, with `Things to Come` (11,380). Both are the "
        "Grace Movement — mid-Acts dispensationalism, out of the American Grace Gospel "
        "Fellowship and Things to Come Mission — which is a genuine tradition with a "
        "genuine distinctive and NO NODE ON THIS TREE, because no source before this one "
        "counted it. 66,974 people between them. A `christianity.grace` node is the right "
        "fix if a second source ever names it; until then they sit on the independent-"
        "evangelical node rather than being given a family of their own on one country's "
        "evidence.",
    "Filipino Assemblies of the First Born, Incorporated":
        "-> christianity.pentecostal.oneness. 14,443. Founded 1936 among Filipino migrant "
        "workers in Hawaii and California and brought home; Oneness in doctrine, which is "
        "the 1916 division the tree treats as the deepest inside Pentecostalism. Filed "
        "there rather than with the trinitarian Assemblies of God, whose name it echoes "
        "and whose theology it rejects.",
    "Worldwide Church of God":
        "-> christianity.adventist. 5,983, and a placement that is right for 2020 and will "
        "not stay right. Herbert Armstrong's body is Sabbatarian and descends from the "
        "Church of God (Seventh Day), which is Millerite, so Adventist is the family. But "
        "the church renounced Armstrong's distinctive doctrines after 1995 and renamed "
        "itself Grace Communion International, and what it is now is ordinary evangelical. "
        "Which of the two the 5,983 answered for is not knowable from a category label "
        "that uses the old name.",
    "Missionary Baptist Churches of the Philippines":
        "-> christianity.baptist, NOT christianity.baptist.landmark. 31,206. `Missionary "
        "Baptist` in American usage names the ABA/BMA Landmark line and the Philippine "
        "body plausibly descends from that mission, but plausibly is the whole of the "
        "evidence and Landmarkism is a specific ecclesiology to attribute to 31,000 people "
        "on a name. The parent asserts only what is certain.",
    "Christian and Missionary Alliance Church of the Philippines":
        "-> christianity.holiness. 327,537, one of the largest evangelical bodies in the "
        "country. The C&MA is Higher Life / Keswick holiness in origin and is filed there "
        "rather than under Pentecostal, though its Philippine churches are widely "
        "charismatic in practice — the tree records the body, not the worship style.",
    "Tribal religion":
        "-> indigenous.philippine. 251,548 people in one cell, covering the Cordillera "
        "peoples, the Lumad of Mindanao, the Mangyan, the Aeta and the Palawan groups with "
        "no distinction between them. It is also a floor rather than a count: the same "
        "peoples appear in the Catholic and Protestant rows in large numbers, and the "
        "census has no way to record a household that keeps both.",
    "Buddhist":
        "-> buddhism, the parent, with no school. 39,158, and almost all of it Chinese "
        "Filipino Mahayana, but the census offers one cell and in2011.py and lk2024.py "
        "make the same call for the same reason.",
    "Islam":
        "-> islam, with no branch. 6,981,710 people, 6.4%, and the second largest category "
        "in the country. Philippine Muslims are overwhelmingly Sunni of the Shafi'i school, "
        "in thirteen ethnolinguistic groups the census does not separate here; the "
        "religion question offers one cell and this file does not invent the rest.",
}


# ---------------------------------------------------------------------------------------
# MAP — 129 categories plus the denominator. Grouped by where they land, not alphabetically:
# the grouping is the argument, and a flat alphabetical dict would hide every duplicate pair
# and every case where two names go to one node.
# ---------------------------------------------------------------------------------------
MAP = {
    # ---- Catholic ---------------------------------------------------------------------
    "Roman Catholic, excluding Catholic Charismatics": "christianity.catholic.latin",
    "Catholic Charismatic": "christianity.catholic.latin",
    "Oblates of Mary Immaculate, Incorporated": "christianity.catholic.latin",

    # Independent Catholic. The first two are one church under two names (REVIEW).
    "Aglipay": "christianity.catholic.independent",
    "Iglesia Filipina Independiente": "christianity.catholic.independent",
    "Philippine Independent Catholic Church": "christianity.catholic.independent",
    "Apostolic Catholic Church, Inc.": "christianity.catholic.independent",

    # ---- the Reformation families, as the missions brought them ------------------------
    "Episcopal Church in the Philippines": "christianity.anglican",
    "Lutheran Church of the Philippines": "christianity.lutheran",
    "Christian Reformed Church in the Philippines, Incorporated":
        "christianity.reformed.continental",
    "Presbyterian Church in the Philippines": "christianity.reformed.presbyterian",
    "Evangelical Presbyterian Church": "christianity.reformed.presbyterian",

    # Unions across the families. The last two are one church under two names (REVIEW).
    "United Church of Christ in the Philippines": "christianity.united",
    "Unida Evangelical Church": "christianity.united",
    "Iglesia Evangelica Unida de Cristo": "christianity.united",
    "United Evangelical Church of the Philippines (Chinese)": "christianity.united",

    # ---- Methodist and Holiness --------------------------------------------------------
    "United Methodists Church": "christianity.methodist",
    "Iglesia Evangelista Methodista en Las Islas Filipinas (IEMELIF)":
        "christianity.methodist",
    "Ang Iglesia Metodista sa Pilipinas, Inc.": "christianity.methodist",
    "Other Methodists": "christianity.methodist",
    "Free Methodist Church": "christianity.methodist.holiness",
    "Wesleyan Church": "christianity.methodist.holiness",

    "Christian and Missionary Alliance Church of the Philippines": "christianity.holiness",
    "Salvation Army Philippines": "christianity.holiness",
    "Church of the Nazarene": "christianity.holiness",
    "Philippine Evangelical Holiness Church": "christianity.holiness",

    # ---- Baptist -----------------------------------------------------------------------
    # Conventions and associations, where the name says Baptist and nothing narrower.
    "Other Baptists": "christianity.baptist",
    "Baptist Conference of the Philippines": "christianity.baptist",
    "Southern Baptist Church": "christianity.baptist",
    "General Baptist Churches of the Philippines": "christianity.baptist",
    "Convention of the Philippine Baptist Church": "christianity.baptist",
    "Missionary Baptist Churches of the Philippines": "christianity.baptist",
    "Conservative Baptist Association in the Philippines": "christianity.baptist",
    "Association of Baptist Churches in Luzon, Visayas, and Mindanao": "christianity.baptist",
    # The separatist fundamentalist line, which in the Philippines is what "Bible",
    # "Fundamental" and "Independent" name (REVIEW on Bible Baptist Church).
    "Bible Baptist Church": "christianity.baptist.independent",
    "Association of Fundamental Baptist Churches in the Philippines":
        "christianity.baptist.independent",
    "Independent Baptist Churches of the Philippines": "christianity.baptist.independent",
    "Faith Baptist Church": "christianity.baptist.independent",
    "International Baptist Missionary Fellowship": "christianity.baptist.independent",
    "Higher Ground Baptist Mission of the Philippines, Inc.":
        "christianity.baptist.independent",

    # ---- classical Pentecostal ---------------------------------------------------------
    "Assemblies of God": "christianity.pentecostal.trinitarian",
    "Philippines General Council of the Assemblies of God":
        "christianity.pentecostal.trinitarian",
    "Pentecostal Church of God Asia Mission": "christianity.pentecostal.trinitarian",
    "Church of God World Missions in the Philippines": "christianity.pentecostal.trinitarian",
    "Church of the Foursquare Gospel in the Philippines, Incorporated":
        "christianity.pentecostal.trinitarian",
    "Philippine Pentecostal Holiness Church": "christianity.pentecostal.trinitarian",
    "Universal pentecostal Church": "christianity.pentecostal.trinitarian",
    # Oneness — the 1916 split, and the tree keeps it visible.
    "United Pentecostal Church (Philippines), Inc.": "christianity.pentecostal.oneness",
    "Filipino Assemblies of the First Born, Incorporated": "christianity.pentecostal.oneness",

    # ---- charismatic and neo-charismatic ------------------------------------------------
    # The Filipino independent ministries: founder-led, Full Gospel or revival language,
    # inside the global charismatic stream rather than outside it. Most of the docstring's
    # naming rule is spent here.
    "Jesus is Lord Church": "christianity.pentecostal.charismatic",
    "Victory Christian Fellowship of the Philippines, Inc.":
        "christianity.pentecostal.charismatic",
    "Victory Chapel Christian Fellowship": "christianity.pentecostal.charismatic",
    "International One Way Outreach": "christianity.pentecostal.charismatic",
    "Free Believers in Christ Fellowship": "christianity.pentecostal.charismatic",
    "Door of Faith": "christianity.pentecostal.charismatic",
    "Christ Faith Fellowship Philippines, Inc.": "christianity.pentecostal.charismatic",
    "Charismatic Full Gospel Ministries": "christianity.pentecostal.charismatic",
    "Jesus is Alive Community, Inc.": "christianity.pentecostal.charismatic",
    "Jesus Reigns Ministries": "christianity.pentecostal.charismatic",
    "March of Faith Church Sole, Inc.": "christianity.pentecostal.charismatic",
    "Christ the Living Stone Fellowship": "christianity.pentecostal.charismatic",
    "Zion Christian Community Church": "christianity.pentecostal.charismatic",
    "Jesus Christ Saves Global Outreach": "christianity.pentecostal.charismatic",
    "I Am Redeemer and Master Evangelical Church, Inc.": "christianity.pentecostal.charismatic",
    "Miracle Revival Church of the Philippines": "christianity.pentecostal.charismatic",
    "Miracle Life Fellowship International": "christianity.pentecostal.charismatic",
    "Jesus the Anointed One Church": "christianity.pentecostal.charismatic",
    "Bread of Life Ministries": "christianity.pentecostal.charismatic",
    "Light of the World Christian Center, Inc.": "christianity.pentecostal.charismatic",
    "Jesus First Christian Ministries, Incorporated": "christianity.pentecostal.charismatic",
    "Lord Jesus Our Redeemer Church Foundation International, Inc.":
        "christianity.pentecostal.charismatic",
    "Cathedral of Praise, Incorporated": "christianity.pentecostal.charismatic",
    "Love of Christ International Ministries": "christianity.pentecostal.charismatic",
    "Word International Ministries, Inc.": "christianity.pentecostal.charismatic",
    "River of God Church, Inc.": "christianity.pentecostal.charismatic",
    "Potter's House Christian Center": "christianity.pentecostal.charismatic",
    "Jesus Loves You Ministries, Inc.": "christianity.pentecostal.charismatic",
    "Lord of the Nations, Inc.": "christianity.pentecostal.charismatic",
    "Kingsway Fellowship International (Philippines), Inc.":
        "christianity.pentecostal.charismatic",
    "Take the Nation for Jesus Global Ministires (Corpus Christi)":
        "christianity.pentecostal.charismatic",
    "Don Stewart Ministries Miracle Revivals, Inc.": "christianity.pentecostal.charismatic",
    "Faith Tabernacle Church (Living Rock Ministries)": "christianity.pentecostal.charismatic",

    # ---- independent evangelical, non-denominational ------------------------------------
    # Bible churches, community churches, and the mission agencies people answered with.
    "Alliance of Bible Christian Communities of the Philippines":
        "christianity.nondenominational",
    "Evangelical Christian Outreach Foundation": "christianity.nondenominational",
    "Christ's Commission Fellowship": "christianity.nondenominational",
    "Fundamental Grace Gospel Church of the Christ in the Philippines, Inc.":
        "christianity.nondenominational",
    "Things to Come": "christianity.nondenominational",
    "Bible Centered Fellowship": "christianity.nondenominational",
    "Good News Christian Churches": "christianity.nondenominational",
    "Bethany Church of Philippines": "christianity.nondenominational",
    "Philippine Missionary Fellowship": "christianity.nondenominational",
    "Harvesters Christian Fellowship": "christianity.nondenominational",
    "Christ to the Philippines, Inc.": "christianity.nondenominational",
    "Word for the World": "christianity.nondenominational",
    "Way of Salvation": "christianity.nondenominational",
    "Philippine Evangelical Mission": "christianity.nondenominational",
    "World Missionary Evangelism": "christianity.nondenominational",
    "National Council of Christian Community Churches (NCCCC), Inc.":
        "christianity.nondenominational",
    "Asia Evangelistic Fellowship Philippines, Inc.": "christianity.nondenominational",
    "Ambassadors for Christ Philippine Evangelism, Inc.": "christianity.nondenominational",
    "Philippine Good News Ministries": "christianity.nondenominational",
    "Good News Worldwide Mission, Inc.": "christianity.nondenominational",
    "F.R.E.E. Mission Philippines, Inc.": "christianity.nondenominational",
    "FIFCOP Mission, Inc.": "christianity.nondenominational",
    "Jireh-Evangel Church Planting Philippines, Inc.": "christianity.nondenominational",

    # ---- the remaining imported families ------------------------------------------------
    "Christian Brethren International Pilipinas, Inc.": "christianity.plymouth",
    "Evangelical Free Church of the Philippines": "christianity.pietist",
    "Church of Christ": "christianity.restorationist",
    "Christian Missions in the Philippines": "christianity.restorationist",
    "Seventh Day Adventist": "christianity.adventist",
    "Worldwide Church of God": "christianity.adventist",
    "Church of Jesus Christ of the Latter Day Saints": "christianity.latterday",
    "Jehovah's Witness": "christianity.witnesses",

    # ---- Filipino independent churches ---------------------------------------------------
    "Iglesia ni Cristo": "christianity.filipinoindependent.inc",
    "Most Holy Church of God in Christ Jesus": "christianity.filipinoindependent.mcgi",
    "Church Body of Christ Filipinista": "christianity.filipinoindependent",
    "Crusaders of the Divine Church of Christ, Incorporated": "christianity.filipinoindependent",
    "Iglesia sa Dios Espiritu Santo, Incorporated": "christianity.filipinoindependent",
    "Philippine Ecumenical Christian Church": "christianity.filipinoindependent",
    "Philippine Benevolent Missionaries Association": "christianity.filipinoindependent",

    # ---- named a stream and no body ------------------------------------------------------
    "Other Protestants": "christianity.protestant",
    "Other Evangelical Churches": "christianity.protestant",

    # ---- outside Christianity ------------------------------------------------------------
    "Islam": "islam",
    "Buddhist": "buddhism",
    "Tribal religion": "indigenous.philippine",
    "Union Espiritista Cristiana de Filipinas, Incorporated": "spiritualism.kardecist",

    # ---- not a religion, or not named ----------------------------------------------------
    "None": "unaffiliated",
    "Not reported": "unrecorded",
    "Other religious affiliations": "other.ph",
}


def _key(cat):
    return " ".join(str(cat).split())


# Built once so a duplicated key in the literal above cannot hide behind dict semantics
# (build_tree.py reads the TEXT of the other mapping files for the same reason).
_LOOKUP = {_key(k): v for k, v in MAP.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return _LOOKUP.get(c)
