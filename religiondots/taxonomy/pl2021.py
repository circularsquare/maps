"""
GUS NSP 2021 declared-affiliation classification -> religiondots taxonomy.

**Branch-level mapping, like cz2021.py and ca2021.py.** All 216 named categories map onto
branches from branches.py; no leaves are created. spec §2.4 defers meticulous cross-source
matching, and the category's own name travels with the row in `source_category`, so
deepening later costs nothing and redoes nothing.

Unlike Czechia, the source is NOT flat — GUS ships a 7-level classification and
`sources/pl.py` keeps only the leaves, so what arrives here is the bottom of that tree.
The group rows above the leaves (`chrześcijaństwo`, `katolicyzm`, `protestantyzm i
tradycja protestancka`, `nurt badaczy Pisma Świętego`) never reach this file; they are
recorded in `sources/pl.md` because they are GUS's own opinion about the shape of the
tree, and it disagrees with branches.py in two places that are worth naming:

  1. GUS puts `starokatolicyzm` under `katolicyzm`, so the Mariavites and the Polish
     National Catholic Church are Catholic in its tree. branches.py has the same view —
     `christianity.catholic.independent` is a child of `christianity.catholic` — so this
     one agrees.
  2. GUS puts the antitrinitarian bodies (`Kościół Unitariański`, `Wspólnota Unitarian
     Uniwersalistów`, `Jednota Braci Polskich`) under `inne chrześcijańskie`, i.e. inside
     Christianity. branches.py has `unitarianuniversalist` as a ROOT, beside Christianity
     rather than under it. This file follows branches.py, so those three change family.
     See REVIEW. Mexico produced the same kind of disagreement in the other direction
     (INEGI files Orthodox Christians under *otras religiones*), so this is now the
     expected shape rather than a surprise: source categories sit in different PLACES in
     the tree, not just under different names.

WHAT MAKES THE POLISH LIST DIFFERENT from every other source so far: the tail is
CONGREGATIONS, not denominations. `Zbór Ewangeliczny "Betel" w Warszawie` (2 people),
`Kościół Jezusa Chrystusa w Werbkowicach` (5), `Warsaw International Church` (24),
`Zbór Wolnych Chrześcijan w Jaworznie` (18) are single congregations that registered as
religious associations, and GUS tabulates each as its own category. About 60 of the 216
are of this kind. They are mapped to the tradition the congregation belongs to, which is
the honest reading, but it means a Polish "category" and an American one are not the same
size of thing at all.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

# Categories that are not religious affiliations and are kept off the tree entirely.
EXCLUDED = {
    "OGÓŁEM":
        "the unit's own population total, not a category.",
    "Ogółem":
        "the unit's own population total, not a category.",
    "Udzielający odpowiedzi na pytanie o wyznanie":
        "universe subtotal — everyone who answered the question at all, 30,212,506. It "
        "is the sum of `należący` and `nienależący`, so drawing it would double them.",
    "należący do wyznania":
        "universe subtotal — everyone who named a religion, 27,601,000. It is the sum of "
        "the 216 named categories, so drawing it would double the whole map.",
    "Odmawiający odpowiedzi na pytanie o wyznanie":
        "7,807,553 people — 20.53% of the country — who refused a VOLUNTARY question. "
        "Not a religion, and not 'no religion' either, which is its own category "
        "(nienależący do żadnego wyznania, 2,611,506). Excluded from the dots, so the "
        "Polish map draws 27.6M of 38.0M people. This is the single most important fact "
        "about the Polish map and belongs in note_public, not in a footnote. Czechia's "
        "equivalent share is 30%, so this is the same problem one size smaller.",
    "Nie ustalono":
        "15,059 people for whom GUS could not establish an answer at all — a coverage "
        "residual rather than a refusal, and far too small to matter either way.",
}

# Defensible but arguable, recorded so the reasoning is not lost and can be overturned.
REVIEW = {
    "Kościół Unitariański / Wspólnota Unitarian Uniwersalistów / Jednota Braci Polskich":
        "-> unitarianuniversalist, which is a ROOT in branches.py, so these three leave "
        "Christianity even though GUS files them under `inne chrześcijańskie`. Jednota "
        "Braci Polskich is the modern revival of the Polish Brethren — the Socinians, "
        "expelled in 1658 — who are the direct ancestors of Unitarianism, so filing the "
        "three together is the coherent choice. The cost is real: it takes 130 people out "
        "of the Christian family in the country whose Reformation produced them.",
    "Wyznawcy Słońca":
        "-> other.pl. 480 people, present in 88 gminy, and the biggest write-in GUS did "
        "not classify. Sun-worship reads as neopagan and `paganism` was tempting, but GUS "
        "had `pogaństwo - rekonstrukcjonizm i neopogaństwo` available and deliberately "
        "did not use it. Respecting the source's own placement rather than improving on it.",
    "teizm (ogólnie teizm, wiara w Boga, monoteizm)":
        "-> unchurched. 301 people who answered the 'which religion' question with 'I "
        "believe in God'. That is belief with no body, which is exactly what `unchurched` "
        "is for. Not `secular`, which is the opposite claim.",
    "deizm":
        "-> secular, following cz2021.py, which puts deismus there. A deist reports a "
        "position about God rather than membership of anything.",
    "satanizm":
        "-> esoteric, following cz2021.py. LaVeyan and most Polish Satanism is atheistic "
        "and genealogically part of Western occultism, so it is not paganism and not a "
        "theistic family of its own. 292 people.",
    "Zachodni Zakon Sufi":
        "-> esoteric, NOT islam. Inayat Khan's Sufi Order in the West is explicitly "
        "universalist and does not require its members to be Muslim, so filing 25 people "
        "under islam would overstate what the name implies. Arguable both ways.",
    "Karaimski Związek Religijny":
        "-> judaism. Karaite Judaism rejects the rabbinic tradition, and the Polish "
        "Karaims are a recognised ethnic minority as well as a religious one. Filed under "
        "judaism because that is the tradition; a `judaism.karaite` leaf would be better "
        "and is deferred with everything else under §2.4.",
    "Kościół Chrystusowy w Rzeczypospolitej Polskiej / w Polsce / Zrzeszenie Kościołów "
    "Chrystusowych / Warszawski Kościół Chrystusowy":
        "-> christianity.restorationist. The Polish Churches of Christ come out of the "
        "Stone-Campbell restoration, though the Polish bodies reached it via the "
        "Zjednoczony Kościół Ewangeliczny and read as generically evangelical today. "
        "`christianity.pietist` would also be defensible for the first two.",
    "Kościół Wolnych Chrześcijan / Stowarzyszenie Zborów Chrześcijan":
        "-> christianity.plymouth. These are the Polish Open Brethren, which is what the "
        "plymouth node holds. 1,776 people between them.",
    "Kościół Remonstrantów Polskich":
        "-> christianity.reformed.continental. The Remonstrants are the Dutch Arminian "
        "body, so Continental Reformed is right by descent even though Arminianism is the "
        "position the Synod of Dort defined itself against.",
    "Kościół Boga Żywego (...) Kościół Światło Świata":
        "-> christianity.pentecostal.oneness. This is La Luz del Mundo, a Mexican Oneness "
        "body; 28 people in Poland.",
    "Dom Izraela Polania / Mesjańskie Zbory Boże / Wspólnota \"Drzewo Oliwne\" / "
    "judaizm mesjanistyczny":
        "-> christianity.messianic, which branches.py already flags as a CONTESTED "
        "PLACEMENT: the theology is Christian, adherents generally identify as Jewish, "
        "and Jewish institutions do not accept the claim.",
}

# The two parody answers, and in Poland they are LARGE. Pastafarianism at 2,312 is bigger
# than 190 of the 216 categories and is present in 29 gminy; Jedi is 687. GUS tabulated
# them because respondents wrote them in.
#
# They get their own family rather than being dropped or filed under a religion, for the
# reason cz2021.py gives: dropping silently loses visible people, and filing them under
# `paganism` or `other.pl` would assert something false.
PARODY = ["pastafarianizm", "jediizm (religia Jedi)"]

MAP = {
    # ================================================================ no religion
    # GUS's level-2 row, and a real category rather than a universe subtotal: 2,611,506
    # people who answered the question and said they belong to nothing.
    "nienależący do żadnego wyznania": "unaffiliated",

    # ================================================================ Catholic
    # katolicyzm > Kościół katolicki
    "Kościół katolicki - obrządek łaciński (Kościół rzymskokatolicki)":
        "christianity.catholic.latin",
    "Kościół katolicki - obrządek bizantyjsko-ukraiński (Kościół greckokatolicki)":
        "christianity.catholic.eastern",
    "Kościół katolicki - obrządek ormiański (Kościół ormiańskokatolicki)":
        "christianity.catholic.eastern",
    "Kościół katolicki - obrządek bizantyjsko-słowiański (Kościół neounicki)":
        "christianity.catholic.eastern",
    "różne inne obrządki wschodnie Kościoła katolickiego":
        "christianity.catholic.eastern",

    # katolicyzm > starokatolicyzm. The Mariavites are Poland's own — a 1906 movement
    # around Feliksa Kozłowska's revelations, and the only Old Catholic body anywhere with
    # a Polish origin. Two of them, because it split again in 1935.
    "Kościół Starokatolicki Mariawitów": "christianity.catholic.independent",
    "Kościół Katolicki Mariawitów": "christianity.catholic.independent",
    "Kościół Polskokatolicki": "christianity.catholic.independent",
    "Katolicki Kościół Narodowy": "christianity.catholic.independent",
    "Polski Narodowy Katolicki Kościół": "christianity.catholic.independent",
    "Narodowy Kościół Katolicki": "christianity.catholic.independent",
    "Reformowany Kościół Katolicki": "christianity.catholic.independent",
    "Kościół Starokatolicki": "christianity.catholic.independent",
    "Kościół Dobrej Nadziei": "christianity.catholic.independent",
    "Powszechny Kościół Ludu Bożego": "christianity.catholic.independent",

    # ================================================================ Orthodox
    "Kościół Prawosławny": "christianity.orthodox.canonical",
    "Wschodni Kościół Staroobrzędowy": "christianity.orthodox.oldbeliever",
    "Staroprawosławna Cerkiew Staroobrzędowców": "christianity.orthodox.oldbeliever",

    # chrześcijańskie kościoły orientalne
    "Ormiański Kościół Apostolski": "christianity.oriental",
    "Ormiański Kościół Apostolski Katolikosatu Eczmiadzyńskiego": "christianity.oriental",
    "Kościół koptyjski": "christianity.oriental",
    "różne inne chrześcijańskie kościoły wschodnie": "christianity.oriental",

    # ================================================================ Reformation
    "Kościół Ewangelicko-Augsburski": "christianity.lutheran",
    "Kościół Ewangelicko-Reformowany": "christianity.reformed.continental",
    "Ewangeliczny Kościół Reformowany": "christianity.reformed",
    "Kościół Remonstrantów Polskich": "christianity.reformed.continental",
    "Kościół Ewangelicko-Prezbiteriański": "christianity.reformed.presbyterian",
    "Kościół Prezbiteriański": "christianity.reformed.presbyterian",
    "Kościół Anglikański": "christianity.anglican",

    # ================================================================ Separatist
    "Kościół Chrześcijan Baptystów": "christianity.baptist",
    "Zbór Ewangelicko-Baptystyczny w Katowicach": "christianity.baptist",
    "Biblijny Kościół Baptystyczny": "christianity.baptist.independent",
    "Kościół Wolnych Chrześcijan": "christianity.plymouth",
    "Stowarzyszenie Zborów Chrześcijan": "christianity.plymouth",
    "Zbór Wolnych Chrześcijan w Jaworznie": "christianity.plymouth",
    "Chrześcijańska Wspólnota Mennonitów": "christianity.anabaptist.mennonite",
    "kwakrzy": "christianity.friends",

    # ================================================================ Pietist / Wesleyan
    "Kościół Ewangelicznych Chrześcijan": "christianity.pietist",
    "Ewangeliczny Kościół Chrześcijański": "christianity.pietist",
    "Chrześcijańska Wspólnota Ewangeliczna": "christianity.pietist",
    "Kościół Ewangeliczny": "christianity.pietist",
    "Ewangeliczny Związek Braterski": "christianity.pietist",
    "Kościół Ewangeliczny \"Misja Łaski\"": "christianity.pietist",
    "Zbór Ewangeliczny \"Agape\" w Poznaniu": "christianity.pietist",
    "Zbór Ewangelii Łaski": "christianity.pietist",
    "Ursynowska Społeczność Ewangeliczna": "christianity.pietist",
    "Chrześcijańska Wspólnota Życie i Misja": "christianity.pietist",
    "Polski Ewangeliczny Kościół Braterski w Tarnowskich Górach": "christianity.pietist",
    "Chrześcijańska Wspólnota Braterska": "christianity.pietist",
    "Zbór Ewangeliczny \"Betel\" w Warszawie": "christianity.pietist",

    "Kościół Ewangelicko-Metodystyczny": "christianity.methodist",
    "Ewangeliczny Kościół Metodystyczny": "christianity.methodist",
    "Kościół Armia Zbawienia": "christianity.holiness",

    # ---- Pentecostal. The largest Protestant family in Poland after the Lutherans, and
    # the one the congregation-level tail is mostly made of.
    "Kościół Zielonoświątkowy": "christianity.pentecostal.trinitarian",
    "Kościół Boży w Chrystusie": "christianity.pentecostal.trinitarian",
    "Kościół Boży": "christianity.pentecostal.trinitarian",
    "Kościół Chrześcijan Wiary Ewangelicznej": "christianity.pentecostal.trinitarian",
    "Chrześcijańska Wspólnota Zielonoświątkowa": "christianity.pentecostal.trinitarian",
    "Ewangeliczna Wspólnota Zielonoświątkowa": "christianity.pentecostal.trinitarian",
    "Zbór Stanowczych Chrześcijan": "christianity.pentecostal.trinitarian",
    "Kościół Jezusa Chrystusa Wiary Chrześcijańskiej": "christianity.pentecostal.trinitarian",
    "Kościół Pentakostalny": "christianity.pentecostal.trinitarian",

    "Zbór Ewangelicznych Chrześcijan w Duchu Apostolskim": "christianity.pentecostal.oneness",
    "Kościół Boga Żywego, Będącego Wsparciem i Podporą Prawdy, Kościół Światło Świata":
        "christianity.pentecostal.oneness",

    "Kościół Chwały": "christianity.pentecostal.charismatic",
    "Wspólnota Chrześcijańska Wrocław dla Jezusa": "christianity.pentecostal.charismatic",
    "Kościół Nowego Przymierza w Lublinie": "christianity.pentecostal.charismatic",
    "Centrum Biblijne \"Jezus Jest Panem\"": "christianity.pentecostal.charismatic",
    "Centrum Chrześcijańskie \"Kanaan\"": "christianity.pentecostal.charismatic",
    "Centrum Chrześcijańskie \"Miecz Ducha\"": "christianity.pentecostal.charismatic",
    "Centrum Chrześcijańskie \"Nowa Fala\"": "christianity.pentecostal.charismatic",
    "Chrześcijańska Wspólnota \"Jezus Panem\"": "christianity.pentecostal.charismatic",
    "Kościół Chrześcijan Pełnej Ewangelii Obóz Boży": "christianity.pentecostal.charismatic",
    "Ruch Chrześcijański MT28": "christianity.pentecostal.charismatic",
    "Kościół \"Chrystus dla Wszystkich\"": "christianity.pentecostal.charismatic",
    "Kościół Chrześcijański \"Arka\" w Poznaniu": "christianity.pentecostal.charismatic",
    "Kościół Chrześcijański ZOE": "christianity.pentecostal.charismatic",
    "Kościół Chrześcijański \"Jezus Żyje\"": "christianity.pentecostal.charismatic",
    "Kościół Chrześcijański \"Wieczernik\"": "christianity.pentecostal.charismatic",
    "Kościół Chrześcijański \"Słowo Wiary\"": "christianity.pentecostal.charismatic",
    "Międzynarodowa Misja \"Centrum Służby Życia\" - Life Centre Ministries":
        "christianity.pentecostal.charismatic",
    "Chrześcijański Kościół \"Dobra Nowina\"": "christianity.pentecostal.charismatic",
    "Kościół \"Chrześcijańska Wspólnota Jordan\"": "christianity.pentecostal.charismatic",
    "Społeczność Chrześcijańska \"Miejsce Odnowienia\"": "christianity.pentecostal.charismatic",
    "Wspólnota Chrześcijańska \"Pojednanie\"": "christianity.pentecostal.charismatic",
    "Chrześcijański Kościół Pełnej Ewangelii \"Duch i Moc\"":
        "christianity.pentecostal.charismatic",
    "Chrześcijański Kościół \"Maranatha\" w Wiśle": "christianity.pentecostal.charismatic",
    "Chrześcijańskie Centrum \"Pan Jest Sztandarem\" - Kościół w Tarnowie":
        "christianity.pentecostal.charismatic",
    "Kościół Jezusa Chrystusa \"Syjon\" w Rzeszowie": "christianity.pentecostal.charismatic",
    "Misja Pokoleń": "christianity.pentecostal.charismatic",
    "Kościół \"Misja dla Polski\"": "christianity.pentecostal.charismatic",
    "Kościół Chrześcijański \"Otwarte Drzwi\"": "christianity.pentecostal.charismatic",

    # ================================================================ Restorationist
    "Kościół Chrystusowy w Rzeczypospolitej Polskiej": "christianity.restorationist",
    "Kościół Chrystusowy w Polsce": "christianity.restorationist",
    "Zrzeszenie Kościołów Chrystusowych": "christianity.restorationist",
    "Warszawski Kościół Chrystusowy": "christianity.restorationist",
    "Lokalny Kościół w Kwidzynie": "christianity.restorationist",
    "Miejscowy Kościół w Lublinie": "christianity.restorationist",
    "Kościół Nowoapostolski": "christianity.restorationist",
    "Kościół Jezusa Chrystusa Świętych w Dniach Ostatnich (Mormoni)": "christianity.latterday",

    # ---- Adventist and sabbatarian
    "Kościół Adwentystów Dnia Siódmego": "christianity.adventist",
    "Adwentyści Dnia Siódmego - Ruch Reformacyjny": "christianity.adventist",
    "Kościół Reformowany Adwentystów Dnia Siódmego": "christianity.adventist",
    "Kościół Chrześcijan Dnia Sobotniego": "christianity.adventist",

    # ---- nurt badaczy Pisma Świętego. GUS's own group, and the reason
    # christianity.biblestudent exists: one of these four is the Witnesses and three are
    # the Bible Students who did not follow Rutherford after 1917.
    "Świadkowie Jehowy": "christianity.witnesses",
    "Zrzeszenie Wolnych Badaczy Pisma Świętego": "christianity.biblestudent",
    "Świecki Ruch Misyjny \"Epifania\"": "christianity.biblestudent",
    "Stowarzyszenie Badaczy Pisma Świętego": "christianity.biblestudent",

    "Stowarzyszenie Chrześcijańskiej Nauki - Związek Wyznaniowy":
        "christianity.christianscience",

    # ================================================================ Messianic
    "judaizm mesjanistyczny": "christianity.messianic",
    "Dom Izraela Polania": "christianity.messianic",
    "Mesjańskie Zbory Boże (Dnia Siódmego)": "christianity.messianic",
    "Mesjańska Społeczność Wywołanych": "christianity.messianic",
    "Wspólnota \"Drzewo Oliwne\"": "christianity.messianic",

    # ================================================================ unspecified Christian
    "chrześcijaństwo (ogólna deklaracja wyznaniowa)": "christianity",
    "chrześcijanie (ochrzczeni, wierzący) niepraktykujący": "christianity",
    "protestantyzm (ogólna deklaracja wyznaniowa)": "christianity.protestant",
    "różne wspólnoty i kościoły protestancko-ewangeliczne niezarejestrowane w Polsce":
        "christianity.protestant",
    "Unia Ewangelikalna": "christianity.protestant",

    "chrześcijanie niezrzeszeni": "christianity.nondenominational",
    "Kościoły domowe - protestanckie": "christianity.nondenominational",
    "Zbór Chrześcijański": "christianity.nondenominational",
    "Zbór w Wodzisławiu Śląskim": "christianity.nondenominational",
    "Kościół Chrześcijański w Warszawie": "christianity.nondenominational",
    "Kościół \"Ekklesia\" w Warszawie": "christianity.nondenominational",
    "Warsaw International Church": "christianity.nondenominational",
    "Kościół Chrześcijan w Rybniku": "christianity.nondenominational",
    "Kościół w Radomiu": "christianity.nondenominational",
    "Kościół Jezusa Chrystusa w Werbkowicach": "christianity.nondenominational",
    "Związek Wyznaniowy \"Polska Chrześcijańska Służba\"": "christianity.nondenominational",

    "Chrześcijański Kościół Dobra": "christianity.other",
    "Kościół Chrześcijański w Duchu Prawdy i Pokoju": "christianity.other",
    "Uczniowie Ducha Świętego (Stowarzyszenie Panunistyczne)": "christianity.other",
    "Kościół Miłosierdzia Jezusowego": "christianity.other",
    "Kościół Miłosiernego Boga": "christianity.other",
    "Chrześcijański Związek Wyznaniowy \"Źródło\"": "christianity.other",

    # ================================================================ antitrinitarian
    # Out of Christianity and onto the unitarianuniversalist root — see REVIEW.
    "Kościół Unitariański": "unitarianuniversalist",
    "Wspólnota Unitarian Uniwersalistów": "unitarianuniversalist",
    "Jednota Braci Polskich": "unitarianuniversalist",

    # ================================================================ Islam
    "Muzułmański Związek Religijny": "islam",
    "Islamskie Zgromadzenie Ahl-Ul-Bayt": "islam",
    "Liga Muzułmańska": "islam",
    "różne afiliacje islamskie (ogólnie islam, muzułmanizm, sunnizm, szyizm itp.)": "islam",
    "Stowarzyszenie Jedności Muzułmańskiej": "islam",
    "Stowarzyszenie Muzułmańskie Ahmadiyya": "islam",

    # ================================================================ Judaism
    "Związek Gmin Wyznaniowych Żydowskich": "judaism",
    "Beit Polska - Związek Postępowych Gmin Żydowskich": "judaism",
    "Niezależna Gmina Wyznania Mojżeszowego": "judaism",
    "różne afiliacje judaistyczne (ogólnie judaizm, mozaizm, judaizm reformowany itp.)":
        "judaism",
    "Gmina Wyznaniowa Starozakonnych": "judaism",
    "Karaimski Związek Religijny": "judaism",

    # ================================================================ Buddhism
    "Buddyjski Związek Diamentowej Drogi Linii Karma Kagyu": "buddhism",
    "Buddyjska Wspólnota Zen Kannon": "buddhism",
    "różne afiliacje buddyjskie (ogólnie buddyzm, therawada, zen itp.)": "buddhism",
    "Związek Buddyjski Bencien Karma Kamtsang": "buddhism",
    "Ośrodek Wietnamskiego Buddyzmu": "buddhism",
    "Związek Buddystów Zen \"Bodhidharma\"": "buddhism",
    "Misja Buddyjska - Trzy Schronienia": "buddhism",
    "Szkoła Zen Kwan Um": "buddhism",
    "Kanzeon - Związek Buddyjski": "buddhism",
    "Związek Buddyjski Khordong": "buddhism",
    "Ligmincha Polska": "buddhism",
    "Międzynarodowa Wspólnota Dzogczen- Namdagling": "buddhism",
    "Sangha \"Dogen Zenji\"": "buddhism",
    "Związek Buddystów Czan": "buddhism",
    "Związek Buddyjski \"Dzogczien Kunzang Cziuling\"": "buddhism",
    "Wspólnota bez Bram Mumon-Kai Związek Buddyjski Zen Rinzai": "buddhism",
    "Instytut Śardza Ling": "buddhism",
    "Związek Buddyjski \"Yeshe Khorlo\"": "buddhism",
    "Związek Tybetańskiego Bon \"Sa Trik Er Sang\"": "buddhism",
    "Związek Buddyjski Dak Shang Kagyu": "buddhism",

    # ================================================================ Hinduism
    "Międzynarodowe Towarzystwo Świadomości Kryszny": "hinduism",
    "Związek Wyznaniowy Hindu Bhavan": "hinduism",
    "różne afiliacje hinduistyczne (ogólnie hinduizm, wisznuizm, śiwaizm itp.)": "hinduism",
    "Związek Ajapa Yoga": "hinduism",
    "Związek Hatha Jogi \"Brama Jogi\"": "hinduism",
    "Instytut Wiedzy o Tożsamości \"Misja Czaitanii\"": "hinduism",
    "Ruch Świadomości Babadżi Herakhandi Samadź": "hinduism",
    "Światowy Uniwersytet Duchowy Brahma Kumaris": "hinduism",
    "Radha Govind Society of Poland": "hinduism",

    # ================================================================ paganism
    # GUS's own level-4 group, and one of the better ones anywhere: Slavic native faith is
    # a Polish movement with a real 1930s lineage, not a generic "other".
    "Rodzima Wiara": "paganism",
    "Rodzimy Kościół Polski": "paganism",
    "Polski Kościół Słowiański": "paganism",
    "Zachodniosłowiański Związek Wyznaniowy \"Słowiańska Wiara\"": "paganism",
    "różne afiliacje pogańskie (ogólnie pogaństwo, neopogaństwo, rodzimowierstwo, Asatru itp.)":
        "paganism",
    "Wicca": "paganism",
    "Zgromadzenie Braci i Sióstr Politeistów": "paganism",

    # ================================================================ other traditions
    "Wiara Baha'I": "bahai",
    "Związek Wyznaniowy Singh Sabha Gurudwara": "sikhism",
    "sikhizm": "sikhism",
    "shintoizm": "shinto",
    "zaratusztrianizm": "zoroastrianism",
    "rastafarianizm": "rastafari",
    "taoizm": "daoism",
    "Związek Taoistów Tao Te King": "daoism",
    "Kościół Zjednoczeniowy - Ruch pod Wezwaniem Ducha Świętego dla Zjednoczenia "
    "Chrześcijaństwa Światowego": "unification",

    # ================================================================ esoteric
    "różne wierzenie ezoteryczne (okultyzm, spirytyzm, spirytualizm, antropozofia itp.)":
        "esoteric",
    "gnostycyzm": "esoteric",
    "satanizm": "esoteric",
    "panteizm": "esoteric",
    "Kościół Panteistyczny \"Pneuma\"": "esoteric",
    "Lectorium Rosicrucianum, Międzynarodowa Szkoła Złotego Różokrzyża": "esoteric",
    "Zakon Braci Zjednoczenia Energetycznego": "esoteric",
    "Medytacyjne Stowarzyszenie Najwyższej Mistrzyni Czing Hai": "esoteric",
    "Zachodni Zakon Sufi": "esoteric",
    "Związek Wyznaniowy Eckankar": "esoteric",

    # ================================================================ positions
    "deizm": "secular",
    "teizm (ogólnie teizm, wiara w Boga, monoteizm)": "unchurched",

    # ================================================================ parody
    "pastafarianizm": "parody",
    "jediizm (religia Jedi)": "parody",

    # ================================================================ residual
    "inne - niesklasyfikowane": "other.pl",
    "własne (indywidualne) wierzenia religijne": "other.pl",
    "Wyznawcy Słońca": "other.pl",
    "Związek Wyznaniowy \"Wierzę w Dobro Człowieka\"": "other.pl",
    "Związek Wyznaniowy Kwinarystów": "other.pl",
    "Polski Kościół Dialogu": "other.pl",
}


def _key(cat):
    """Normalise a source category for lookup.

    GUS pads two of its universe labels with a trailing "w tym:" ("of which") that appears
    on some sheets and not others, so `należący do wyznania` arrives in two spellings.
    Everything else matches verbatim.
    """
    c = " ".join(str(cat).split())
    for suffix in (" w tym:", " w tym"):
        if c.endswith(suffix):
            c = c[: -len(suffix)].strip()
    return c


def resolve(cat):
    """Source category -> taxonomy node id, or None if it is deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
