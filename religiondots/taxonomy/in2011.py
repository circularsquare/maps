"""
Census of India 2011, table C-01 and its Appendix -> religiondots taxonomy.

**Branch-level mapping, like cz2021.py.** Nothing here creates a leaf; the census's own
category name travels on every row in `source_category`, so deepening later costs nothing
(spec §2.4).

Two tables, and they are different in kind:

  C-01           eight columns, of which six are religions, one is the residual bucket and
                 one is the universe. The six are the only categories India publishes at a
                 fine geography, so they carry 99.1% of the map.
  C-01 Appendix  83 named religions inside that residual bucket, at state level, 7,788,066
                 people. This is the entire reason India is worth drawing at more than
                 Pew depth, and almost all of it is Adivasi.

**The Annexure is deliberately absent from this file.** It is arithmetically a partition —
`Religion:X` = the unspecified remainder + the named sects, per state, to within a few
hundred people nationally — so it *looks* usable for splitting Hindus into Lingayats or
Christians into Catholics. It is not, and the reason is in the numbers rather than in the
structure: it records 8,399 Catholics among 27.8M Christians, 573 Shia and 267 Sunni among
172.2M Muslims, 3,269 Digambar among 4.45M Jains. Nobody believes India has 573 Shia
Muslims. What the Annexure counts is people who wrote a SECT where the form asked for a
religion, which is a measure of insistence, not of membership — and it undercounts every
group it names, including the plausible-looking ones (Lingayat 2,663,229 against a Karnataka
community usually put near 10M). Mapping any of it would put figures on the map that are
wrong by one to three orders of magnitude in a direction the map cannot show. It is
normalised into in.csv with a note, and drawn nowhere. See sources/in.md §4.

**India's Muslims are divided by a second instrument, 2026-09-14.** `in_split.py` splits the
census `Muslim` column by Pew's *Religion in India* (2021) sect question, one share per Pew
region, and writes three `Muslim: ... (Pew 2021)` labels that are in no census table. The
Annexure is still not used for this; Pew counted people and the Annexure did not.

**Christians in two states, 2026-09-15.** `in_split_christian.py` splits the census `Christian`
column where a second source reaches: Kerala's Catholics by district from the Kerala Migration
Surveys (K.C. Zachariah, CDS Working Paper 468, Table 6), and Mizoram's churches from the state
statistics department's membership rolls for 2010-11, one share for the whole state. Kerala's
table names ten churches and only the Catholic total is drawn: the three rites disagree with the
dioceses' own rolls by up to fifty times from place to place while their sum holds, and nothing
independent reaches the others. The reasoning is in that file's docstring and sources/in.md §9.

**And Pew's regions for the rest, the same day.** The same script applies Pew's 2021 respondent file
(`qdenomrec`, weighted) to the East and South outside Kerala: Catholic and Baptist, of the three churches
the public file names. Catholics take one share pooled over both regions, whose own shares do not differ
(sources/in.md §13); Baptists keep the South's. North, Central and West have too few Christian respondents (24, 10 and 56) and stay
on `christianity`. The Northeast was drawn too for part of the day and is withdrawn: its one share failed
the Catholic rolls state by state and put Presbyterians in Nagaland, so its Christians outside Mizoram are
on `christianity` as well. sources/in.md §10 to §12.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

# Categories that are not religious affiliations and are kept off the tree entirely.
EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Religion not stated":
        "2,867,303 people, 0.24% of India — the smallest such residual of any country on "
        "the map, and the reason is worth knowing rather than celebrating. The religion "
        "question was answered by the head of household FOR the household, so it is not a "
        "personal self-identification and nobody had the opportunity to decline on their "
        "own behalf. Compare Poland's 20.5% refusal of a voluntary personal question. "
        "Excluded from the dots, so India draws 1.208bn of 1.211bn people.",
    "Other religions and persuasions":
        "the parent bucket, not a category. It is replaced by its 83 Appendix children "
        "wherever they can be allocated (spec §3.10) and by `other.in` for the 1.9% "
        "remainder the Appendix does not name. Drawing it as well would double every "
        "Adivasi religion in the country.",
}

# Defensible but arguable, recorded so the reasoning is not lost and can be overturned.
REVIEW = {
    "Muslim: Sunni (Pew 2021)":
        "-> islam.sunni, from Pew's QSECT (topline p. 23), one share per region: North 80, "
        "Central 79, West 50, East 43, South 38, Northeast 32. Uniform inside a region, because "
        "six regions is all Pew publishes, so no concentration inside a region can show. The "
        "Northeast's 32 sits beside 38% who did not know or refused, and that 38 stays on "
        "`islam` (spec §2.7a). QSECT offers the four madhhabs nowhere, so nothing goes below "
        "`islam.sunni` (§2.6).",
    "Muslim: Shi'a (Pew 2021)":
        "-> islam.shia, NOT islam.shia.jaafari: the card says Shi'a and names no school, and "
        "the Bohras and Khojas inside it are not Twelvers anyway. West 11, North 7, South 6, "
        "Central 5, Northeast 5, East 2. THE KNOWN COST: Lucknow's Shia get Central's 5% like "
        "the rest of Uttar Pradesh, Hyderabad's get the South's 6%, and the Gujarati Bohras "
        "are spread over the whole West. Kargil, where Muslims are mostly Shia, would have got "
        "the North's 7% and is left undivided instead because Pew selected no location in "
        "Ladakh (in_split.py). About 6% of India's Muslims against community figures often "
        "quoted at 10 to 15%; none of those was verified, and Pew is the only one that counted "
        "people.",
    "Muslim: Ahmadiyya (Pew 2021)":
        "NO LONGER EMITTED. Pew's volunteered Ahmadiyya share goes to islam with the unspecified "
        "remainder, not islam.ahmadiyya: folded back on Anita's call of 2026-09-14 ('lets move "
        "ahmadiya back for now'), after it had drawn 1,355,525 people for part of a day. No census "
        "counts India's Ahmadis and the best national figure is about 150,000, while Pew's code "
        "put 1.17M in the South on about fifteen respondents and 0 in the East, where Odisha has "
        "organised Ahmadi villages (sources/branches.md, 'India's Ahmadis'). in_split.py still "
        "splits the share out before adding it to `Muslim`, so Sunni and Shi'a did not move.",
    "Muslim":
        "THE UNSPLIT PART stays on `islam` as the census's own `Muslim`, `measured`, in two "
        "cases. (1) Inside a surveyed region, the answers that name no branch: some other sect, "
        "no sect in particular and DK/refused (spec §2.7a), the volunteered Ahmadiyya code "
        "since Anita folded it back on 2026-09-14, plus Pew's rounding, since three "
        "regions print to 99 and one to 101. (2) Everywhere Pew interviewed nobody: the Kashmir "
        "Valley's ten districts, Ladakh (Leh and Kargil), Manipur, Sikkim, Chandigarh, Dadra and "
        "Nagar Haveli, Daman and Diu, Lakshadweep, and the Andaman and Nicobar Islands. Kashmir "
        "Valley is read as the 2011 Kashmir division; Pew says only 'Kashmir districts'. "
        "Ladakh, Chandigarh and the two western UTs were in Pew's frame and got no location, "
        "which on the Galapagos line (queue.md) is still nothing measured. TIER: the split rows "
        "are `derived` and not `modelled`, unlike Turkiye's, because these Muslims were counted "
        "at the sub-district and the viewer removes `modelled` dots instead of rolling them up.",
    "Christian: Catholic, three rites (KMS 2008-2014)":
        "-> christianity.catholic, NOT the rite nodes the survey names. Kerala only, from Zachariah's "
        "WP468 Table 6: Syro-Malabar + Syro-Malankara + Latin as a share of each of the 14 districts' "
        "Christians (Thrissur 88.5, Kozhikode and Pathanamthitta 35.6), applied to the census. The "
        "rites fail against catholic-hierarchy's diocesan rolls (roll/survey spreads 51 for "
        "Syro-Malankara and 9.2 for Latin across six groups of see districts; Kannur and Kasaragod "
        "read Syro-Malabar 2.22 and Latin 0.27 under sees covering the same two districts), while "
        "the Catholic total spreads 2.3, so the survey swaps rites and keeps the total. Would be "
        "catholic.eastern and catholic.latin if a rite ever passed. A floor: Catholics who answered "
        "Dalit Christian or Others stay on `christianity`.",
    "Christian: Catholic (Pew 2021)":
        "-> christianity.catholic, NOT .latin: the card says Catholic, and India has three rites. Pew's "
        "respondent file, weighted: East 37.4 and South 39.4 on their own, which do not differ once Pew's "
        "design effect is allowed for (p 0.27), so both are drawn at the pooled 39.0 (sources/in.md §13). "
        "The South's pool includes Kerala's respondents and is applied only outside Kerala; against "
        "catholic-hierarchy's 2004 rolls it reads roll/survey 2.24 there against 1.48 in the East, so Tamil "
        "Nadu's and Andhra "
        "Pradesh's Catholics are probably under-drawn. By state the East spreads 3.8 (Bihar high on few "
        "Christians) and is kept. North 52.4, Central 62.2 and West 52.5 are not drawn (24, 10 and 56 "
        "respondents, under the 100 floor); North's roll/survey, 0.73, is the only one under 1. The "
        "Northeast's 16.2 is not drawn either, withdrawn 2026-09-15: by state its roll/drawn ran from "
        "0.18 in Nagaland to 2.21 in Assam with Arunachal (sources/in.md §11, §12).",
    "Christian: Baptist (Pew 2021)":
        "-> christianity.baptist. A volunteered code (DO NOT READ). South 13.0 (67 respondents), drawn; "
        "Northeast 33.6 (114), withdrawn 2026-09-15 with the rest of the Northeast's share, because one "
        "rate for Nagaland, Meghalaya, Assam, Arunachal Pradesh and Tripura fits none of them (sources/"
        "in.md §12); no Baptist answer in the other four regions. THE KNOWN COST: the South's 881,434 "
        "are spread over Tamil Nadu, Karnataka, Andhra Pradesh and Puducherry alike.",
    "Christian: Presbyterian (Pew 2021)":
        "NO LONGER EMITTED WITH A COUNT, since 2026-09-15. -> christianity.reformed.presbyterian, as "
        "Mizoram's Presbyterian Church of India roll is. A volunteered code; all 96 answers are in the "
        "Northeast (31.8%). For part of a day one share for the region put about 553,000 Presbyterians "
        "in Nagaland, whose churches the Nagaland Baptist Church Council organises (716,495 baptised "
        "members on the Baptist World Alliance's undated member page, per the scout), and 1.81M outside "
        "Mizoram from Meghalaya to Tripura. The review (sources/in.md §11) found the same share failing "
        "the Catholic rolls state by state and the region's total not holding either, since Mizoram's "
        "respondents sit inside a share applied only to its neighbours, so the Northeast was withdrawn "
        "(§12). The mapping stays for the label, which in_split_christian.py still names and no region "
        "now fills.",
    "Christian":
        "THE UNSPLIT PART stays on `christianity` as the census's own `Christian`, `measured`. "
        "(1) Kerala: the non-Catholic answers of WP468 Table 6 (Jacobite, Orthodox, Mar Thoma, CSI, "
        "Pentecost/Church of God/Brethren) have no independent figure to check them against, and "
        "three of the four Syrian churches share the 'Malankara' name the rite check shows being "
        "confused, so unchecked is not drawn; Dalit Christian is a caste, not a church; Others. "
        "(2) Mizoram: Isua Krista Kohhran, 1.2% of the rolls, whose family nothing read here names. "
        "(3) Pew's East and South: the answers the respondent file does not name, which are "
        "all other denominations (Church of North India, Church of South India, Orthodox, Pentecostal "
        "and the rest, recoded together in the public file), no denomination, don't know and refused. "
        "(4) Pew's North, Central and West: 24, 10 and 56 Christian respondents, under the 100 floor "
        "(in_split_christian.py docstring, point 3); their shares are printed by --dry-run. (4a) Pew's "
        "Northeast outside Mizoram, 326 respondents, withdrawn 2026-09-15 because its one share fails "
        "state by state; a split there needs a source by state (sources/in.md §12, queue.md D). (5) Where "
        "Pew interviewed nobody: Manipur, Sikkim, the Kashmir Valley, Ladakh, Chandigarh, Dadra and "
        "Nagar Haveli, Daman and Diu, Lakshadweep, and the Andaman and Nicobar Islands.",
    "Christian: Presbyterian Church of India (Mizoram roll 2010-11)":
        "-> christianity.reformed.presbyterian. The Mizoram Synod of the Presbyterian Church of India, "
        "550,560 members in 2010-11, 57.3% of the ten rolls. Its series is smooth except 2014-15 "
        "(325,214 with 30,396 women), an entry error in a year not read.",
    "Christian: Baptist Church of Mizoram (Mizoram roll 2010-11)":
        "-> christianity.baptist. The southern (Lunglei) Baptist church, 146,331. Spread over the "
        "whole state like every body here, which is the construction's known cost.",
    "Christian: Lairam Isua Krista Baptist Kohhran (Mizoram roll 2010-11)":
        "-> christianity.baptist. The Lai church of Lawngtlai, 24,795; Baptist by its own name. It is "
        "a Lawngtlai church drawn at a state share, so most of its dots land in Aizawl.",
    "Christian: United Pentecostal Church (North East India) (Mizoram roll 2010-11)":
        "-> christianity.pentecostal.oneness, as fj2007.py files `United Pentecostal`: the UPC of North "
        "East India is the Oneness (Jesus' name) church of that name, 70,497.",
    "Christian: United Pentecostal Church (Mizoram) (Mizoram roll 2010-11)":
        "-> christianity.pentecostal.oneness. 45,471. A later separation from the UPC (North East "
        "India) that kept the name; filed with it on the name alone, no doctrinal statement read.",
    "Christian: Evangelical Church of Maraland (Mizoram roll 2010-11)":
        "-> christianity.other, NOT christianity.evangelical: that node holds a census ANSWER "
        "'Evangelical', and this is a named body, the church of the Lakher Pioneer Mission among the "
        "Mara of Saiha, 37,383, with no Protestant family to belong to. Drawn at the state share, so "
        "it appears across Aizawl although it is a Saiha church.",
    "Christian: Salvation Army (Mizoram roll 2010-11)":
        "-> christianity.holiness.salvation-army, as ag2001, bm2010 and eight others. 36,395 in "
        "2010-11; the series steps to 55,791 in 2011-12 with no explanation, which is most of why "
        "2011-12's rolls sum to 106% of the census and 2010-11's to 100.5%.",
    "Christian: Seventh-day Adventist (Mizoram roll 2010-11)":
        "-> christianity.adventist. 19,235; a noisy series (26,858 the year before, 12,542 in 2013-14).",
    "Christian: Roman Catholic (Mizoram roll 2010-11)":
        "-> christianity.catholic.latin, as fr2024, ee2021 and be2024 file `Roman Catholic`: Mizoram is "
        "the Latin Diocese of Aizawl, 18,890. Unlike Kerala's cell, this one names the Roman church.",
    "Sari Dharma":
        "-> indigenous.indian.sarna, NOT a node of its own, and this is the largest "
        "judgement call in the file: 506,369 people, 100% of them in West Bengal. `Sari "
        "Dharma` (also Sari/Sarna Dharam) is the Santal and Oraon name for the same "
        "sacred-grove religion that Jharkhand's census respondents wrote as `Sarna`; the "
        "two are one religion under two regional spellings, and the perfect "
        "state-complementarity — Sarna 83% Jharkhand, Sari Dharma 100% West Bengal — is "
        "itself evidence of that rather than of two religions. Overturnable: if they are "
        "genuinely distinct, this merges 506k people into the wrong node. `Sarnam` "
        "(1,494) and `Saranath` (837) go the same way for the same reason.",
    "Pagan":
        "-> indigenous.indian.khasi, NOT `paganism`. 2,088 people, 62% in Meghalaya. `Pagan` in "
        "the Khasi hills is the colonial-era label for the traditional religion and is "
        "still used that way locally; it is not the Western neo-pagan revival that "
        "branches.py's `paganism` node describes. Mapping it there would file Khasi "
        "traditionalists with Wiccans. The 38% outside Meghalaya take the Khasi node too, "
        "because the mapping is one row for every state.",
    "Animist":
        "-> indigenous.indian. 4,130 people, 80% in Sikkim. The mirror of cz2021.py's "
        "`animismus`, which goes to `paganism` — there the write-in is a Western "
        "self-description, here it is an outsider's word for a tribal religion in a state "
        "full of them. Same string, opposite meaning, decided by where it was written.",
    "Non Christians":
        "-> indigenous.indian.khasi. 1,538 people, 96% in Meghalaya, a state that is 75% "
        "Christian. A negative self-description given in a place where the traditional "
        "religion is defined locally by not being the missionary one. `unaffiliated` "
        "would be a clear misreading; it says nothing about belief.",
    "Nirankari":
        "-> other.in. The Sant Nirankari Mission, 1,781 people, 64% Punjab. A distinct "
        "Sikh-derived movement that would deserve its own node if any source counted it "
        "properly; these are only the Nirankaris who declined all six census religions, "
        "and the great majority are recorded as Sikh or Hindu. Not mapped to `sikhism`, "
        "because answering `other` is exactly the datum.",
    "Dera Sarsa":
        "-> other.in. Dera Sacha Sauda, 139 people. Same reasoning as Nirankari at a "
        "twentieth of the size.",
    "ADI DHARM":
        "-> indigenous.indian. 82,255 people, but 65% in ODISHA rather than Punjab, which "
        "is the giveaway: this is `Adi Dharam`, the generic 'original religion' write-in "
        "used across the Adivasi belt, and not the Ad Dharm movement of Punjab's "
        "Ravidassia Dalits. `ravidassia` would have been the obvious wrong answer.",
    "Tadvi":
        "-> indigenous.indian. 1,786 people, 99% Maharashtra. The Tadvi Bhils are a Bhil "
        "group with substantial Muslim practice; those who wrote `Tadvi` under `Other "
        "religions` rather than answering Muslim are being taken at their word.",
    "A.C.":
        "-> other.in. 1,317 people, Maharashtra and Gujarat. The abbreviation is not "
        "expanded anywhere in the census documentation and no confident reading is "
        "available, so it goes to the residual rather than to a guess.",
    "Jews / Judaism":
        "-> judaism. 4,429 people, and the state split is the interesting part: 46% "
        "Manipur, which is the Bnei Menashe of Manipur and Mizoram rather than the older "
        "Cochin, Bene Israel and Baghdadi communities. Those are largely counted here too, "
        "but the Bnei Menashe are why the number is as large as it is.",
    "Atheist":
        "-> secular, not `unaffiliated`. 33,304 people who wrote the word. India's census "
        "offers no `no religion` box at all, so there is no `unaffiliated` figure for "
        "India anywhere — an absence worth stating, because on this map India will show as "
        "a country with no irreligion, and that is a property of the question.",
}

# 83 Appendix names + the six C-01 religions. Grouped by where they land.
MAP = {
    # ------------------------------------------------------------ C-01, the six religions
    # These carry 99.1% of India and are the only categories published at sub-district
    # level. Each is a whole family here: the census asks for the religion and clubs every
    # sect into it (see the Annexure note above), so `Muslim` really is all of Islam in
    # India and nothing finer is knowable from this source.
    "Hindu": "hinduism",
    "Muslim": "islam",
    "Christian": "christianity",
    "Sikh": "sikhism",
    "Buddhist": "buddhism",
    "Jain": "jainism",

    # ------------------------------------------------------------ Muslim branches, in_split.py
    # DERIVED, 2026-09-14. Not census categories: in_split.py divides each sub-district's
    # `Muslim` count by Pew's 2021 QSECT shares for its region and writes these labels, plus a
    # smaller `Muslim` for the answers that name no branch. See REVIEW and in_split.py.
    "Muslim: Sunni (Pew 2021)": "islam.sunni",
    "Muslim: Shi'a (Pew 2021)": "islam.shia",
    # No Ahmadiyya label: folded back into `Muslim` on Anita's call, 2026-09-14 (REVIEW).

    # ------------------------------------------------------------ Christian churches, in_split_christian.py
    # DERIVED, 2026-09-15. Not census categories: in_split_christian.py divides each Kerala and Mizoram
    # sub-district's `Christian` count and writes these labels, plus a smaller `Christian` for the rest.
    "Christian: Catholic, three rites (KMS 2008-2014)": "christianity.catholic",
    "Christian: Presbyterian Church of India (Mizoram roll 2010-11)": "christianity.reformed.presbyterian",
    "Christian: Baptist Church of Mizoram (Mizoram roll 2010-11)": "christianity.baptist",
    "Christian: Lairam Isua Krista Baptist Kohhran (Mizoram roll 2010-11)": "christianity.baptist",
    "Christian: United Pentecostal Church (North East India) (Mizoram roll 2010-11)":
        "christianity.pentecostal.oneness",
    "Christian: United Pentecostal Church (Mizoram) (Mizoram roll 2010-11)":
        "christianity.pentecostal.oneness",
    "Christian: Evangelical Church of Maraland (Mizoram roll 2010-11)": "christianity.other",
    "Christian: Salvation Army (Mizoram roll 2010-11)": "christianity.holiness.salvation-army",
    "Christian: Seventh-day Adventist (Mizoram roll 2010-11)": "christianity.adventist",
    "Christian: Roman Catholic (Mizoram roll 2010-11)": "christianity.catholic.latin",

    # ------------------------------------------------------------ Christian churches, Pew 2021 regions
    # DERIVED, 2026-09-15. in_split_christian.py divides each sub-district's `Christian` count in Pew's
    # East and South (outside Kerala) by the respondent file's weighted shares. The Northeast was withdrawn
    # the same day, so no row carries the Presbyterian label; its mapping is kept (REVIEW).
    "Christian: Catholic (Pew 2021)": "christianity.catholic",
    "Christian: Baptist (Pew 2021)": "christianity.baptist",
    "Christian: Presbyterian (Pew 2021)": "christianity.reformed.presbyterian",

    # ------------------------------------------------------------ Appendix: Sarna
    "Sarna": "indigenous.indian.sarna",
    "Sari Dharma": "indigenous.indian.sarna",
    "Sarnam": "indigenous.indian.sarna",
    "Saranath": "indigenous.indian.sarna",

    # ------------------------------------------------------------ Appendix: Gondi
    "Gond / Gondi": "indigenous.indian.gondi",
    "Koyatur": "indigenous.indian.gondi",
    "Budhadeo": "indigenous.indian.gondi",

    # ------------------------------------------------------------ Appendix: Donyi-Polo
    # The Tani religion of Arunachal Pradesh, plus the smaller Arunachal names that are
    # local forms of the same organised revival. Nocte, Rangfra and the Mishmi names are
    # separate peoples; they sit here rather than on the parent because Arunachal's
    # traditional religions were codified together and the census reports them together.
    "Doni Polo / Sidonyi Polo": "indigenous.indian.donyipolo",
    "Nani Intiya": "indigenous.indian.donyipolo",
    "Intaya": "indigenous.indian.donyipolo",
    "Nyarino": "indigenous.indian.donyipolo",
    "Rangfra": "indigenous.indian.donyipolo",
    "Dongi": "indigenous.indian.donyipolo",
    "Rangkho thak": "indigenous.indian.donyipolo",
    "Apo Rangang": "indigenous.indian.donyipolo",
    "Nocte": "indigenous.indian.donyipolo",
    "Idu / Idu Mishmi": "indigenous.indian.donyipolo",
    "Kaman  / Miju Mishmi / Kaman Mishmi / Miju": "indigenous.indian.donyipolo",
    "Hill Miri": "indigenous.indian.donyipolo",
    "Aka": "indigenous.indian.donyipolo",

    # ------------------------------------------------------------ Appendix: Sanamahi
    "Sanamahi": "indigenous.indian.sanamahi",
    "Heraka": "indigenous.indian.sanamahi",
    "Tikao Ragong": "indigenous.indian.sanamahi",
    "Chang Naga": "indigenous.indian.sanamahi",

    # ------------------------------------------------------------ Appendix: Meghalaya
    "Khasi": "indigenous.indian.khasi",
    "Niamtre": "indigenous.indian.khasi",
    "Niam Shnong": "indigenous.indian.khasi",
    "Songsarek": "indigenous.indian.khasi",
    "Garo": "indigenous.indian.khasi",
    "Traditional Religion": "indigenous.indian.khasi",
    "Non Christians": "indigenous.indian.khasi",
    "Pagan": "indigenous.indian.khasi",

    # ------------------------------------------------------------ Appendix: the rest of
    # the Adivasi tail. Sixty names, mostly one people in one district, none of them large
    # enough to earn a node of its own (§2.4). The people are real and the names are the
    # census's own, so they travel in `source_category` and can be split out later.
    "Addi Bassi": "indigenous.indian",
    "ADI DHARM": "indigenous.indian",
    "Adim dhamm": "indigenous.indian",
    "Adi": "indigenous.indian",
    "ADI KURUM": "indigenous.indian",
    "Bidin": "indigenous.indian",
    "Yumasam": "indigenous.indian",
    "Tribal Religion": "indigenous.indian",
    "Nature Religion": "indigenous.indian",
    "Animist": "indigenous.indian",
    "Santal": "indigenous.indian",
    "Ho": "indigenous.indian",
    "Munda": "indigenous.indian",
    "Oraon": "indigenous.indian",
    "Kharwar": "indigenous.indian",
    "Paharia": "indigenous.indian",
    "Birsa": "indigenous.indian",
    "Tana Bhagat": "indigenous.indian",
    "Sadri": "indigenous.indian",
    "Kisan": "indigenous.indian",
    "Marangboro": "indigenous.indian",
    "Swarna": "indigenous.indian",
    "Krupa": "indigenous.indian",
    "Dupub": "indigenous.indian",
    "Fralung": "indigenous.indian",
    "Bamanya": "indigenous.indian",
    "Bori": "indigenous.indian",
    "Baiga": "indigenous.indian",
    "Baigani Dharam": "indigenous.indian",
    "Bhumia": "indigenous.indian",
    "Korku": "indigenous.indian",
    "Bhil": "indigenous.indian",
    "Tadvi": "indigenous.indian",
    "Halba": "indigenous.indian",
    "Katkari": "indigenous.indian",
    "Pardhi": "indigenous.indian",
    "Bhoi": "indigenous.indian",
    "Bodo / Boro": "indigenous.indian",
    "Karbi / Mikir": "indigenous.indian",
    "Hajong": "indigenous.indian",
    "subba": "indigenous.indian",
    "Mannan": "indigenous.indian",
    "paniyar": "indigenous.indian",
    "Hidmaraj": "indigenous.indian",

    # ------------------------------------------------------------ Appendix: not indigenous
    "Parsi/Zorastrian": "zoroastrianism",
    "Bahai / Bahais": "bahai",
    "Jews / Judaism": "judaism",
    "Atheist": "secular",

    # ------------------------------------------------------------ residual
    # spec §3.11: residual buckets are per source and never merged across countries.
    #
    # The first entry is the Appendix's own floor: it names a religion only at 100+
    # adherents nationally, so 149,668 people (1.9% of the bucket) are in religions the
    # census recorded and did not publish. sources/in.py emits them as a category rather
    # than letting the allocation absorb them into the named ones; this is where they land.
    # tools/check_mapping.py caught them being dropped — they were exactly the gap between
    # the 1,207,838,006 that reached countries.py and the 1,207,987,674 that should have.
    "Other religions and persuasions, not separately named": "other.in",
    "Nirankari": "other.in",
    "Dera Sarsa": "other.in",
    "A.C.": "other.in",
}


# ------------------------------------------------------------------------ the Annexure
#
# All 47 write-in sects, excluded by name rather than by a `startswith` rule, so that a
# sect appearing in a reissue shows up as unmapped and fails the check instead of being
# swallowed. The reasoning is in the module docstring and in sources/in.md §4; the short
# version is that these count insistence rather than membership.
#
# They are listed in the census's own order of religion — Hindu, Muslim, Christian, Sikh,
# Buddhist, Jain — because that grouping is what makes the problem visible: the Christian
# block names nine denominations totalling 13,391 people out of 27.8M Christians.
_ANNEXURE_SECTS = [
    # under Hindu (966.3M)
    "Hindu", "Lingayat / Veer Shaiva", "Bathau / Bathew / Bathou",
    "Ghasidas / Satnam / Satnami", "Sanatan Dharma", "Ravidasi", "Meitei",
    "Balmiki / Walmiki / Valmiki", "Baishnav / Vaishnav", "Vishwa Karma", "Kabir Panthi",
    "Alakh/Mahima", "Nath Panthi", "Anukul Thakur", "Swami Narayan", "Bairagi",
    "Parnami/Pranami", "Sai", "Brahm Kumar / Brahm Kumari", "Anand Margi",
    # under Muslim (172.2M)
    "Islam/Muslim", "Bohra", "Agakhani", "Shia", "Sunni", "Ahmadia",
    # under Christian (27.8M)
    "Christian", "Catholic", "Unitarian", "Protestant", "Anglo Indian",
    "Seventh Day Adventists", "Orthodox", "Jacobite", "Marthomite", "Jehova Witness",
    # under Sikh (20.8M)
    "Sikh", "Nirmala",
    # under Buddhist (8.4M)
    "Buddhist", "Nav Buddhist / Neo Buddhist / Nav Baudha / Nav Boudha", "Hinayana", "Bon",
    # under Jain (4.45M)
    "Jain", "Digambar", "Samanar", "Swetamber / Shwetambar", "Moksha Margi",
]

_ANNEXURE_REASON = (
    "C-01 Annexure write-in sect. Recorded in in.csv, drawn nowhere. The Annexure is a "
    "true partition of its parent religion but every named sect is an undercount of the "
    "real community by one to three orders of magnitude, because the sect was recorded "
    "only where the respondent volunteered it instead of the religion. See the module "
    "docstring and sources/in.md §4.")

EXCLUDED.update({f"Sect: {s}": _ANNEXURE_REASON for s in _ANNEXURE_SECTS})

# The sub-district COLUMN each allocated category was split out of -> the node that column
# names (spec §7a-i-1). India has exactly one: the six census religions are published at the
# sub-district and are already `measured`, and only `Other religions and persuasions` is
# split from state structure.
#
# THIS IS THE ONE THAT LOOKS LIKE A LOSS AND IS NOT. Rolled up, 4.96M Sarna, 1.03M Gondi,
# 506k Sari Dharma, 331k Donyi-Polo and the rest all become one colour — because ORGI
# counted them as one cell. Their names come from the STATE table, and drawing them at the
# sub-district while claiming they were counted there is the thing this control exists to
# refuse. The named map is still the default; this is what is left when the reader asks
# what was measured.
#
# `Muslim` joined 2026-09-14 with in_split.py: the census counted Muslims at every sub-district,
# and only the branch comes from Pew's six regions, so a Sunni or Shi'a dot rolls back to the
# `islam` the census measured there.
#
# `Christian` joined 2026-09-15 with in_split_christian.py, the same way: counted at every
# sub-district, and only the church comes from a second source (Kerala's survey, Mizoram's rolls,
# and Pew's East and South).
COLUMNS = {
    "Other religions and persuasions": "other.in",
    "Muslim": "islam",
    "Christian": "christianity",
}


def resolve(category):
    """religiondots branch for a Census of India category, or None if off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
