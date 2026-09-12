"""NSO Nepal NPHC 2021 religion (Table 1) -> religiondots taxonomy.

Ten categories plus the universe total, on 753 local levels. **An exact partition with no
`Other` box, no `Not stated` and no residual of any kind** — so unlike almost every other
source here, nothing is dropped except the universe row, and `other.np` does not exist
because Nepal never needed one.

**THREE NEW NODES, AND ALL THREE ARE BOXES ON THE FORM RATHER THAN WRITE-INS.** That is what
makes them different from India's, which are the only comparable thing on this map: India's
83 named Adivasi religions are all inside `Other religions and persuasions` and had to be
recovered from an appendix, so `indigenous.indian` is a floor. Nepal ASKS about Kirat, about
Prakriti and about Bon, and prints the answers beside Hinduism.

  Kirat     924,204  -> indigenous.himalayan.kirat
  Prakriti  102,048  -> indigenous.himalayan.prakriti
  Bon        67,223  -> bon                            (a root; see its note in branches.py)

**THE FIVE WORLD RELIGIONS ARRIVE UNDIVIDED AND STAY UNDIVIDED**, which is the whole of the
rest of this file. Nepal's form has one Hindu box, one Buddhist box, one Muslim box and one
Christian box, and every one of them covers populations this map can hold apart the moment
some source separates them. It does not invent the split here (§2.4, and the same call
lk2024.py, kh2019.py and in2011.py make on the same traditions).

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total Population":
        "the row's own population total, not a category.",
}

REVIEW = {
    "Hindu":
        "-> hinduism, the ROOT, undivided. 23,677,744 people, 81.19% — **the second-"
        "largest Hindu population on earth**, after India's and nearly twice Bangladesh's "
        "12.3M. (§9v's Bangladesh bullet called that 12.3M 'the largest Hindu population "
        "outside India and larger than Nepal's'; it was written before Nepal was drawn and "
        "is corrected in sources.md.) "
        "Nothing in NPHC 2021 names a sampradaya, a caste tradition or a sect, so there is "
        "no Vaishnava/Shaiva/Shakta distinction to be had, and nothing here asserts one. "
        "**Two things about the cell are worth stating because the map cannot show "
        "either.** Nepal was a Hindu kingdom until 2008 and the census category carries "
        "that history: the Hindu box has historically absorbed Janajati communities whose "
        "practice is not Brahminical, and every census since 1991 has seen some of them "
        "move out of it into Kirat, Prakriti and Bon — so the Hindu figure is falling "
        "(86.5% in 1991, 81.3% in 2011, 81.2% now) for reasons that are as much about "
        "identity assertion as about belief. And the boundary is not sharp in practice: "
        "the same household may keep a Mundhum ritual and a Hindu festival, and the census "
        "makes each person pick one.",
    "Bouddha":
        "-> buddhism, the PARENT, and deliberately not a vehicle. 2,393,549 people, 8.21%. "
        "**Nepal is the country where this call costs the most**, because it holds at "
        "least three unlike Buddhisms and the census separates none of them: the Tibetan "
        "Vajrayana of the northern border and of the Tamang, Sherpa, Gurung and Bhote "
        "peoples; the **Newar Vajrayana** of the Kathmandu valley, a Sanskritic tradition "
        "with married Vajracharya priests that exists nowhere else on earth; and the "
        "Theravada revival that arrived from Burma and Sri Lanka in the twentieth century "
        "and is now substantial among Newars. Filing all of it on `buddhism.vajrayana` "
        "would be true of most of it and false of a large minority, and the source offers "
        "no way to say which. lk2024.py, kh2019.py and in2011.py make the same call for "
        "the same reason; this is the fourth. "
        "Its geography is the northern rim and the valley rim: Rasuwa 70.8%, Mustang "
        "58.6%, Manang 50.6%, then Makwanpur 43.9%, Sindhupalchok 41.4% and Nuwakot 38.1% "
        "— the Tamang belt around Kathmandu, which is the largest part of the number.",
    "Islam":
        "-> islam, with no branch, because the census gives none. 1,483,066 people, 5.09%. "
        "Nepal's Muslims are overwhelmingly Sunni of the Hanafi school, with a small Shia "
        "presence, and NSO offers no cell for either. "
        "**The geography is the Terai and it is emphatic**: Rautahat 22.6%, Banke 18.7%, "
        "Kapilbastu 18.2%, Parsa 17.8%, Mahottari 15.6% and Bara 14.7%, against 0.00% in "
        "Bajhang and under 0.3% across Sudurpashchim and Karnali. Madhesh province alone "
        "is 13.3%. That is the Indian border plain, and the community is continuous with "
        "the Muslim population of Uttar Pradesh and Bihar across it — which is visible on "
        "this map for the first time now that both sides are drawn.",
    "Kirat":
        "-> indigenous.himalayan.kirat, a NEW node. 924,204 people, 3.17%. See the node's "
        "own note for what the Mundhum is and where the Kirati peoples are. "
        "**The argument for a node rather than a residual is countability (§2.4)**: this "
        "is a named box on a national census form answered by nearly a million people with "
        "a geography sharp enough to read at a glance — Panchthar 55.7%, Mahakulung "
        "87.3%, and effectively zero west of Bagmati. Nothing about it is inferred. "
        "**The one thing to hold loosely** is the boundary against Hinduism, which is "
        "politically live in exactly the way `indigenous.indian.sarna`'s is: the Kirat "
        "count has risen at every census since 1991 and the movement to record Kirat "
        "rather than Hindu is an explicit one. The number is a measurement of what people "
        "said, and what people say here is changing fast.",
    "Christian":
        "-> christianity, the ROOT, which is spec §6.6's 'branch that carries dots' and "
        "renders as a Christianity `unspecified` row. 512,313 people, 1.76%. NSO names no "
        "body at all, and the population is at least three unlike things — the "
        "Presbyterian and Baptist churches of the eastern hills, a large and mostly "
        "independent Pentecostal movement, and a small Catholic church. Filing it on "
        "`christianity.protestant` would assert a body the source does not name. "
        "**Nepal's Christian population is the fastest-growing in the country by a wide "
        "margin** — under 0.5% in 2001 and 1.4% in 2011 — and its geography is neither "
        "the capital nor the Terai but the central hills: Dhading 7.6%, Makwanpur 6.1%, "
        "Gorkha 6.0%. That is largely Tamang and Magar, i.e. the same Janajati populations "
        "the Kirat and Prakriti boxes draw from, and the three categories are working on "
        "overlapping ground. "
        "**Read the figure with the law in view.** Nepal's 2017 penal code criminalises "
        "conversion and 'hurting religious sentiment', and prosecutions of Christians "
        "under it are documented. A census answer given in that setting is a floor, not a "
        "measurement — §3.5's rule, in its sharpest form on this map outside China.",
    "Prakriti":
        "-> indigenous.himalayan.prakriti, a NEW node. 102,048 people, 0.35%. "
        "**The most arguable call in this file**, because `Prakriti` is not the name of a "
        "tradition — it is the Nepali word for *nature*, offered as a box, and NSO "
        "publishes no gloss on what it covers. Three options were live: fold it into "
        "`indigenous.himalayan` itself (§6.6), give it a node, or send it to a residual. "
        "**It gets a node because it is regionally concentrated enough to be about "
        "something**: Rukum East 16.6%, Rolpa 8.2%, Thawang 45.0%, which is Kham Magar "
        "country and not a scatter of odd answers across the country. A residual would "
        "lose that, and folding it onto the parent would put it in the same row as Kirat, "
        "which is a different and much larger thing. "
        "**What the node does NOT claim** is that these 102,048 people share a ritual "
        "system with each other, let alone with the Kirati. It says they answered the "
        "question and did not answer it with any of the five world religions on the form.",
    "Bon":
        "-> bon, a NEW ROOT and deliberately not a child of `buddhism`. 67,223 people, "
        "0.23%, and the only Bon this map draws — India's census names 697 in its C-01 "
        "Annexure under `Buddhist`, which in2011.py excludes and draws nowhere. The "
        "reasoning for the placement is in the node's note; the short version is that "
        "Bonpos do not "
        "self-describe as Buddhists and NSO prints `Bon` in a cell beside `Bouddha` "
        "rather than inside it, so filing it under Buddhism would contradict the only "
        "source that counts it. "
        "**The cell is probably two things and the census cannot separate them** — the "
        "monastic Yungdrung Bon of the trans-Himalaya, and the Tamu (Gurung) shamanic "
        "tradition of the Gandaki middle hills, which is where most of the number "
        "actually is. The node's note has the figures.",
    "Jain":
        "-> jainism. 2,398 people, 0.01% — a Marwari trading community, concentrated in "
        "Kathmandu and the eastern Terai towns. Small enough that §4.3's presence rings "
        "are what most of its local levels will draw.",
    "Bahai":
        "-> bahai. 537 people. **The smallest national count of anything on this map**, "
        "and worth keeping for exactly that reason: it is a real published census figure "
        "for a community that most sources fold into `Other`. NSO's own `Nepal` summary "
        "sheet prints its percentage as `0`, which is a 1-dp rounding and not a count — "
        "see sources/np.py, which asserts the count against that sheet's column B and "
        "would have failed on its percentage column.",
    "Sikha":
        "-> sikhism. NSO's spelling, kept verbatim as the source category. 1,496 people, "
        "chiefly around the Nanak Math in Kathmandu and the Terai border towns. "
        "**Nepal is the only country between India and China that counts Sikhs at all.**",
}

MAP = {
    "Hindu": "hinduism",
    "Bouddha": "buddhism",
    "Islam": "islam",
    "Kirat": "indigenous.himalayan.kirat",
    "Christian": "christianity",
    "Prakriti": "indigenous.himalayan.prakriti",
    "Bon": "bon",
    "Jain": "jainism",
    "Bahai": "bahai",
    "Sikha": "sikhism",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
