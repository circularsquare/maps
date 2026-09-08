"""BNS Kazakhstan, National Census 2021, volume ch.12 -> religiondots taxonomy.

Nine categories plus two subtotals, on 17 regions, and **every one of them is MODELLED** —
Kazakhstan publishes religion nationally only, and `sources/kz.py` distributes it across the
regions by their ethnic composition (spec §14.10). Nothing here is a count of anybody in the
region it is drawn in.

**THE CATEGORY LIST IS UNUSUALLY GOOD FOR A COUNTRY WITH NO GEOGRAPHY.** Kazakhstan splits
Christianity three ways on the form — Orthodox, Catholic, Protestant — which most censuses on
this map do not, and it offers `non-believer` and `refused to state` as separate boxes rather
than folding them together. The whole thing is an exact partition.

**THE LABELS BELOW ARE THIS PROJECT'S, NOT VERBATIM STRINGS.** The volume's header is
bilingual Kazakh/Russian and split over five lines, so `sources/kz.py` takes the columns by
x-POSITION on the page and names them; the names here are the Russian ones from that header.
A mapping keyed on a verbatim string would be keyed on nothing.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Всего":
        "the region's own population total, not a category. Carried in kz.csv because "
        "sources/kz.py and kz_grid.py both check against it.",
    "Христианство":
        "the PARENT of `Православие`, `Католицизм` and `Протестантизм`, which sum to it "
        "exactly on every row. Drawn would double-count; sources/kz.py does not emit it.",
}

REVIEW = {
    "Ислам":
        "-> islam, with no branch, because the census gives none. **13,297,775 people, "
        "69.31%** — modelled onto regions, so read every regional figure as 'this is what "
        "this region's ethnic mix implies', never as a count. Kazakhstan's Muslims are "
        "overwhelmingly Sunni of the Hanafi school; there is a small Shia Azerbaijani "
        "population and the Ahmadiyya are refused registration. Nothing in the census "
        "separates any of them. "
        "**The model's strongest cell.** The held-out urban/rural test predicts it to "
        "+1.7%/-2.3%, because Islam in Kazakhstan really is close to a function of "
        "ancestry: Kazakhs are 89.2% Muslim, Uzbeks 77.8%, Uyghurs 71.9%, Russians 2.0%.",
    "Православие":
        "-> christianity.orthodox, the BRANCH, which is §6.6's 'a branch that carries dots' "
        "and renders as an Orthodox `unspecified` row. **3,269,143 people, 17.04% — 99.1% "
        "of all Kazakhstani Christians.** The census names no jurisdiction. In fact almost "
        "all of them are the Russian Orthodox Church's Metropolitan District of Kazakhstan, "
        "and `christianity.orthodox.canonical` would very probably be true — but that is "
        "inference from general knowledge and not what the source says, so it stays on the "
        "branch. ru2012.py files Russia's answers one level deeper only because Arena names "
        "the church.",
    "Католицизм":
        "-> christianity.catholic, the branch and not `.latin`. 18,988 people. Kazakhstan "
        "has BOTH a Latin church (the Polish and German deportee descendants of the 1930s "
        "and 40s, and the reason Karaganda has a cathedral) and a Greek Catholic apostolic "
        "administration for Ukrainians. The census does not separate them and neither does "
        "this. ru2012.py's call on `I profess Catholicism`, for the same reason.",
    "Протестантизм":
        "-> christianity.protestant, the node for a Protestant answer that names no body. "
        "9,419 people. In Kazakhstan this is chiefly Baptists, Lutherans of German descent, "
        "Presbyterians and Pentecostals — **and it is the group Kazakhstan's registration "
        "law bears on hardest**, with unregistered house churches prosecuted. Read 9,419 as "
        "a floor for that reason as well as for the modelling.",
    "Неверующие":
        "-> secular, NOT `unaffiliated`, and this is Russia's precedent exactly. "
        "branches.py draws the line at whether a POSITION is stated: `unaffiliated` is a "
        "report of no religion, `secular` is a stated non-theistic position. `Неверующие` "
        "is 'non-believers' — the same wording ru2012.py files under `secular` for Arena's "
        "*I do not believe in God*, and ca2021.py for Canada's atheists. **Nothing in "
        "Kazakhstan goes to `unaffiliated` at all**, because the census offers no plain "
        "'no religion' box: it offers a refusal and a disbelief and they are different "
        "things. "
        "432,140 people, 2.25%. **This is the drawn cell the model is worst at** — the "
        "held-out test misses it by -13.0% urban and +41.8% rural, because non-belief is an "
        "urban behaviour inside every ethnic group at once and ancestry cannot see that. "
        "sources/kz.md §5.",
    "Иудаизм":
        "-> judaism. 7,192 people. Kazakhstan's Jewish population is what is left of a "
        "community built by wartime evacuation and the Gulag; it has fallen by roughly an "
        "order of magnitude since 1989 through emigration. The model puts it where Jews "
        "live, which the census does record — the ethnicity `Евреи` is inside the residual "
        "group here, so this cell's geography comes from the residual's composition and is "
        "the weakest of the drawn religions. Its held-out error is -21% urban / +127% "
        "rural on 7,192 people.",
    "Буддизм":
        "-> buddhism, the PARENT, not a vehicle. 15,458 people, and unusually for this map "
        "the group is identifiable: **82% of Kazakhstan's Buddhists are Koreans** (12,702 "
        "of 15,458), the descendants of the 1937 deportation of the Soviet Korean "
        "population from the Far East. That makes the tradition Mahayana in all likelihood, "
        "and 'in all likelihood' is not what the source says (§2.4). lk2024.py, kh2019.py, "
        "in2011.py and np2021.py all make the same call.",
    "Другое":
        "-> other.kz. 23,247 people, 0.12% — the smallest residual on this map, because "
        "the form it is left over from is unusually full. See the node's note.",
    "Отказались указать":
        "-> unknown. **2,112,653 people, 11.01%, and it IS drawn — which reverses this "
        "file's first version and is Anita's question, 2026-09-07:** *\"is there any way "
        "we could try to draw the 11% who refused to state?\"* "
        "**The fact that decides it is on the census form.** Question 11 of the 2021 "
        "individual questionnaire (`Переписной лист 3-И`) reads: `1. Ислам / 2. "
        "Христианство (2.1 Православие, 2.2 Католицизм, 2.3 Протестантизм) / 3. Иудаизм / "
        "4. Буддизм / 5. Другое (укажите) / 6. Отказываюсь указать / 7. Неверующий`. "
        "**`Отказываюсь указать` — 'I decline to state' — is option SIX: printed, numbered, "
        "first person, and actively chosen by 2.1 million people out of seven offered.** "
        "It is not a blank, not item non-response, not an enumerator's residual code. "
        "**That is what separates it from tt2011.py's Trinidad `Not Stated`**, which is the "
        "ordinary derived residual §3.5 is written about. An offered answer that somebody "
        "picked is an answer. "
        "**-> `unknown` and not a node of its own.** branches.py defines `unknown` as *the "
        "one that reports nothing at all — people the source counted and whose religion it "
        "did not establish… the claim is only that these people are here*, and names "
        "Vietnam as **the first** rather than the only. A refusal is the plainest possible "
        "instance of that: the source counted them, their religion is not established, and "
        "nothing else is claimed. Vietnam's cell arose differently (its answer set never "
        "reached what those people practise) and the two countries never appear together, "
        "so one row carries one meaning in each. "
        "**Drawing it is §3.5 satisfied rather than bent.** The rule is *marked, not "
        "filled, and never redistributed* — and these people are not redistributed into "
        "any religion, they are drawn where they are, in their own colour, saying exactly "
        "what the census recorded. Excluding them marked the absence in a legend line; "
        "drawing them marks it on the map, which is stronger. **Kazakhstan is now 100.00% "
        "drawn.** "
        "**Two things to carry, both in countries.py's public note.** The cell is very "
        "unevenly spread across ethnicities — 9.25% of Kazakhs, 7.48% of Russians, 26.7% "
        "of Kurds, **65.7% of the residual `other nationalities`** (which carries the "
        "24,806 people who gave no ETHNICITY either; a household that declined one "
        "question declined the other) — so the model puts more of it in some regions than "
        "others for reasons about ancestry rather than about belief. And **Kazakhstan "
        "requires religious groups to register, refuses registration to Jehovah's "
        "Witnesses and Ahmadi Muslims, and prosecutes unregistered worship**, so some part "
        "of an 11% refusal is plausibly *about* that and nothing published says which part. "
        "**It is the model's second-worst cell** (-7.8% urban, +15.0% rural on the held-out "
        "test), behind `Неверующие`. That is a reason to label it carefully, which "
        "note_public now does, and was never a reason to leave 2.1 million people off the "
        "map. "
        "**NOT in the viewer's `no religion` control**, which `unknown` already stays out "
        "of — putting a refusal behind a button labelled *no religion* would assert in one "
        "click the exact thing the cell refuses to say.",
}

MAP = {
    "Ислам": "islam",
    "Православие": "christianity.orthodox",
    "Католицизм": "christianity.catholic",
    "Протестантизм": "christianity.protestant",
    "Иудаизм": "judaism",
    "Буддизм": "buddhism",
    "Другое": "other.kz",
    "Неверующие": "secular",
    "Отказались указать": "unknown",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
