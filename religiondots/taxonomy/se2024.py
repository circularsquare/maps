"""ESS `rlgblg` x `rlgdnase` -> religiondots taxonomy. Sweden's Swedish-citizen half.

The European Social Survey asks `rlgblg` — do you consider yourself as belonging to any
particular religion or denomination — and only then which one. For Sweden the second question
is `rlgdnase`, a country-specific list, and **not `rlgdnse`, which does not exist in any
round**. Its nine answers arrive in Swedish even with `metadataLanguage:"en"`, because ESS
translates the harmonised variables and leaves the country-specific ones in the field
language; the keys below are therefore Swedish and are the source's own spelling.

    1  Katolska kyrkan                      -> christianity.catholic.latin
    2  Svenska kyrkan                       -> christianity.lutheran
    3  Annan protestantisk forsamling       -> christianity.protestant
       (t.ex. frikyrka)
    4  Ortodoxa kyrkan                      -> SPLIT THREE WAYS, see below
    5  Annan kristen forsamling             -> christianity
    6  Judisk                               -> judaism
    7  Islam                                -> islam.sunni
    8  Osterlandsk religion (t.ex.          -> other.se
       Buddhism, Hinduism, Sikh, Shinto,
       Tao etc.)
    9  Annan icke-kristen religion          -> other.se
    (rlgblg = No)                           -> unaffiliated
    (refusal / don't know / no answer)      -> excluded, spec §3.5

THE ONE CALL THAT IS NOT A LOOKUP IS ANSWER 4, AND SWEDEN IS WHERE THE USUAL SHORTCUT BREAKS.

Every other country on this map that collects an undifferentiated `Orthodox` answer sends it
to `christianity.orthodox` on an explicit arithmetic argument: Austria says the Armenians are
1.0% of its cell, the UK's note says filing it on `oriental` instead "would be wrong for 98%",
and Canada sends its unspecified remainder to Eastern because that is where the overwhelming
majority of a Canadian Orthodox answer belongs. **In Sweden that argument runs the other way.**

MUCF, which pays the state grant to faith communities and therefore counts them, publishes
`betjanade` per community. Grouped by communion (`MUCF_BETJANADE` below, 2024):

    Eastern Orthodox        69,153   45.85%   Serbian, Greek, Romanian, Macedonian,
                                              Georgian, Russian, Bulgarian, Finnish
    Oriental Orthodox       72,133   47.82%   the two Syriac jurisdictions, Eritrean,
                                              Coptic, Armenian, Ethiopian
    Church of the East       9,542    6.33%   Osterns assyriska kyrka

So the non-Chalcedonian half is the LARGER half, and putting the whole answer on
`christianity.orthodox` would assert the wrong communion for about half of the people in it,
across a schism that is the thing `christianity.oriental`'s own node text calls the commonest
error in religion taxonomies. `christianity`, the bare root, is the other available single
answer and it says only true things, at the price of merging Sweden's Orthodox into the same
cell as answer 5.

**This module takes the third option and splits the answer on MUCF's own counts, per spec
§3.11.** Three existing nodes, no new legend row, and `sources/se.py` emits three source
categories off the one ESS code so the operation is visible in `se.csv` rather than hidden
inside `resolve()`.

Two things about the split that are named rather than fitted:

  * **`betjanade` counts residents, not citizens**, so this ratio is borrowed across the
    citizen/foreign line that the rest of the build keeps. It is very probably CONSERVATIVE
    in the direction that matters: the Syriac and Assyrian population arrived from Tur Abdin
    in the 1960s to 1980s and from Lebanon, Syria and Iraq later, and is heavily naturalised
    and now in its third generation, while the Romanian, Bulgarian and Polish Orthodox are
    post-2007 EU movers with little reason to naturalise. So the citizen half is if anything
    MORE Oriental than 47.82% and the foreign half less. Left unadjusted because nothing
    publishes the adjustment.
  * **Some Assyrians will have answered 5 rather than 4.** The Church of the East is not
    called Orthodox in its own name even though Swedish usage groups it with the Orthodox
    churches, so 6.33% is a floor for that node rather than an estimate of it.

Both halves go on the BARE parents, `christianity.orthodox` and `christianity.oriental`,
rather than on `.canonical` and the named Oriental churches. Every body in MUCF's Eastern list
is in fact a canonical patriarchate and `.canonical` would be true of the population, but the
respondent named no church and this split is an imputation on both sides; the parent is the
granularity the evidence supports. Finland's fi2024.py uses `.canonical` for the same ESS
answer and does not split it, because Finland's Orthodox are one church.

AND THE THING THIS FILE CANNOT DO. Sweden's frikyrkor are Equmeniakyrkan, Pingst, the
Evangeliska Frikyrkan, EFS and Svenska Alliansmissionen, and MUCF counts every one of them
separately: 96,460 / 101,925 / 41,255 / 29,571 / 17,619 in 2024. ESS offers them as a single
`Annan protestantisk forsamling` with frikyrka as a for-instance, and has no geography for any
of them, so the free churches that are the most distinctive thing about religious Sweden are
one undifferentiated node here. MUCF's numbers are national and cannot fix it.
"""

# MUCF (formerly SST), `Trossamfund - antal betjanade`, the 2024 column, read from
# www.mucf.se/rapporter-och-statistik/trossamfund-antal-betjanade on 2026-09-11. `Betjanade`
# is defined in lagen (2024:487) as a member of a faith community or a person who takes part
# in its activities regularly, confirmed by an auditor, and it is the basis of the state
# grant. Two listed bodies have no figure (`Ryska Ortodoxa Kyrkan (Moskva-patriarkatet)` and
# `Svenska prosteriet`, which merged into the Serbian church in 2025) and are absent here.
MUCF_BETJANADE = {
    "christianity.orthodox": {
        "Serbisk-ortodoxa Kyrkan": 24105,
        "Grekisk-ortodoxa Metropolitdomet": 20670,
        "Rumanska Ortodoxa Forsamlingen": 9826,
        "Makedonska Ortodoxa Stiftelse": 7533,
        "Georgiska ortodoxa kyrkan": 2625,
        "Ryska Ortodoxa Kyrkan (Kristi Forklarings forsamling)": 2549,
        "Bulgariska Ortodoxa Forsamlingen": 1353,
        "Finska Ortodoxa kyrkan": 492,
    },
    "christianity.oriental": {
        "Syrisk-Ortodoxa Patriarkatets stallforetradarskap": 26045,
        "Syrisk Ortodoxa Arkestiftet": 20550,
        "Eritreanska Ortodoxa Tewahdo Kyrkan": 10841,
        "Koptisk-ortodoxa Metropolitdomet": 5157,
        "Armeniska apostoliska kyrkan": 5002,
        "Etiopiska Ortodoxa kyrkan": 4538,
    },
    "christianity.churchofeast": {
        "Osterns assyriska kyrka": 9542,
    },
}

ORTHODOX_ANSWER = "Ortodoxa kyrkan"

# The three source categories sources/se.py writes into se.csv in place of answer 4, and the
# share of it each takes. Derived rather than typed, so the arithmetic cannot drift from the
# table above.
_ORTH_TOTALS = {node: sum(v.values()) for node, v in MUCF_BETJANADE.items()}
_ORTH_ALL = sum(_ORTH_TOTALS.values())
ORTHODOX_SPLIT = {
    f"{ORTHODOX_ANSWER} (Eastern Orthodox)":
        _ORTH_TOTALS["christianity.orthodox"] / _ORTH_ALL,
    f"{ORTHODOX_ANSWER} (Oriental Orthodox)":
        _ORTH_TOTALS["christianity.oriental"] / _ORTH_ALL,
    f"{ORTHODOX_ANSWER} (Church of the East)":
        _ORTH_TOTALS["christianity.churchofeast"] / _ORTH_ALL,
}

# The ten answers the pooled ESS rounds produce, before the Orthodox split. sources/se.py
# asserts the pool against this set in BOTH directions: `pool - SOURCE` catches a category
# that appeared, and `SOURCE - pool` catches one that vanished, which is what an off-by-one on
# the denomination axis looks like (fi.py §8).
SOURCE = {
    "Katolska kyrkan",
    "Svenska kyrkan",
    "Annan protestantisk församling (t.ex. frikyrka)",
    ORTHODOX_ANSWER,
    "Annan kristen församling",
    "Judisk",
    "Islam",
    "Österländsk religion (t.ex. Buddhism, Hinduism, Sikh, Shinto, Tao etc.)",
    "Annan icke-kristen religion",
    "No religion",
}

EXCLUDED = {
    "__refused__":
        "Everyone whose `rlgblg` is Refusal, Don't know or No answer, and everyone who "
        "answered Yes to `rlgblg` and then declined to name a denomination. spec §3.5 marks "
        "a refusal rather than filling it, so these people are not drawn and the share they "
        "represent is stated in `gap` instead. Small in Sweden; sources/se.py prints it.",
}

REVIEW = {
    "Svenska kyrkan":
        "-> christianity.lutheran, the branch, and NOT a new node for the Church of Sweden. "
        "fi2024.py's call for the same reason: almost every Swedish Lutheran is in that one "
        "church, so a named node would be accurate and would also be a single-country legend "
        "row for a body no other source here counts, which AGENT_BRIEF §3 sends to Anita "
        "rather than deciding alone. **The loss inside this cell is EFS**, the Evangeliska "
        "Fosterlands-Stiftelsen, a low-church revival movement of 29,571 that works inside "
        "the Church of Sweden and has its own congregations; its members answer 2 like "
        "everybody else, exactly as Finland's Laestadians do. `note_public` names it rather "
        "than the code pretending it is not there.",
    ORTHODOX_ANSWER:
        "-> SPLIT between christianity.orthodox (45.85%), christianity.oriental (47.82%) and "
        "christianity.churchofeast (6.33%), on MUCF's own `betjanade` counts. The module "
        "docstring has the argument and the numbers. In one line: Sweden is the country where "
        "the non-Chalcedonian half of an undifferentiated Orthodox answer is the LARGER half, "
        "because of Sodertalje, so the shortcut every other country here takes would be wrong "
        "for about half the cell. The split is an imputation from a national register applied "
        "uniformly across the lan, so it moves nobody geographically; what it does is stop "
        "75,000 people being filed in a communion half of them left in 451.",
    "Annan protestantisk församling (t.ex. frikyrka)":
        "-> christianity.protestant, `Protestant, unspecified`, and the precedent is "
        "ru2012.py rather than fi2024.py. Arena's 'I profess the Protestant (Lutheran, "
        "Baptist, Evangelical, Anglican)' goes to the same node because the parenthesis is "
        "examples offered to the respondent rather than categories counted, which is exactly "
        "the shape of `(t.ex. frikyrka)`. **fi2024.py is the tempting wrong model**: Finland "
        "lists `Free church` as its own code beside `Other Protestant denomination` and sends "
        "it to christianity.evangelical, because `Vapaakirkko` is the name of an actual body. "
        "Sweden's head noun is `protestantisk` and frikyrka is a for-instance, so "
        "christianity.evangelical would be reading a stream into an example. **This is where "
        "Swedish religion loses most detail**: Equmeniakyrkan, Pingst, the Evangeliska "
        "Frikyrkan and Svenska Alliansmissionen are 257,000 betjanade between them on MUCF's "
        "2024 figures and are one cell here.",
    "Annan kristen församling":
        "-> christianity, the root, following fi2024.py, gr2024.py, ru2012.py and mk2021.py. "
        "The answer names no body and the list it sits at the end of has already ruled out "
        "Catholics, the Church of Sweden, the Protestant free churches and the Orthodox, so "
        "there is nothing left to name and the parent is honest. Not `christianity.other`, "
        "which branches.py reserves for bodies that have no branch rather than for people who "
        "name none.",
    "Katolska kyrkan":
        "-> christianity.catholic.latin. The Catholic Church in Sweden is one diocese, "
        "Stockholm, and it is Latin rite. **The Eastern-rite Catholics are the exception and "
        "they are real here**: the Chaldean, Syriac and Melkite communities are inside that "
        "diocese and are Iraqi, Syrian and Lebanese in origin. ESS offers one Catholic code "
        "and six pooled rounds reach 107 Catholic citizens in total, so they are not "
        "separable in this half. The foreign half does better, because origin_religion.py "
        "splits Iraq, Syria and Lebanon by their own compositions and reaches "
        "christianity.catholic.eastern from there. Said here rather than silently rolled in.",
    "Islam":
        "-> islam.sunni. Sweden's Muslim population is majority Sunni, from Bosnia, Turkey, "
        "Somalia, Syria and Afghanistan. **The Shia are not a rounding error and are not "
        "separable in this half**: MUCF's own figures put Islamiska Shiasamfunden i Sverige "
        "at 35,388 betjanade against roughly 168,000 across the Sunni organisations, so "
        "something like a sixth of organised Swedish Islam is Shia, mostly Iraqi, Iranian, "
        "Afghan Hazara and Lebanese. ESS offers one Islam code. The foreign half splits Iraq, "
        "Iran, Lebanon and Afghanistan by their own Pew compositions, so Sweden's Shia dots "
        "come from the foreign residents only and the citizen Shia are inside islam.sunni.",
    "Judisk":
        "-> judaism, the root, and not one of the movements. Judiska centralradet reports "
        "6,176 betjanade and ESS reaches ten Jewish respondents in six rounds, seven of them "
        "in Stockholm. That fails at the 21 lan and passes both tests at the 8 riksomraden "
        "(p = 0.004, chi-square p = 0.011, and 0.013 on a Monte Carlo version of it), so the "
        "cell takes its riksomrade's share inside each lan. The cost is Gothenburg: no "
        "respondent in Vastsverige named it, so that region's congregation gets none of the "
        "citizen share.",
    "Österländsk religion (t.ex. Buddhism, Hinduism, Sikh, Shinto, Tao etc.)":
        "-> other.se with `Annan icke-kristen religion`, for gr2024.py's and hr2021.py's "
        "reason: the tree has no node for 'some Eastern religion, unspecified', the answer "
        "names five traditions and counts them as one, and picking Buddhism over Hinduism "
        "would invent a fact about a person the survey deliberately did not ask. Sveriges "
        "Buddhistiska Gemenskap reports 11,183 betjanade, nationally and with no geography.",
    "Annan icke-kristen religion":
        "-> other.se. **Sweden's named occupants of this cell are Mandaeans and Yazidis**, "
        "both largely Iraqi: MUCF funds two Mandaean bodies at 4,397 and 10,741 betjanade, "
        "which is one of the largest Mandaean populations outside Iraq and Iran. `mandaeism` "
        "and `yazidism` are both on the tree and neither is reachable from this instrument, "
        "which offers one box. The Alevitiska Riksforbundet, 3,624, is here too and "
        "`islam.alevi` does not exist on the tree at all (Turkiye's §11ac left Alevis inside "
        "the Islam parent for the same reason).",
    "No religion":
        "-> unaffiliated, and it is much the largest cell in the file at 68% of Swedish "
        "citizens. It is not a missing value: it is everyone who answered NO to `rlgblg`. "
        "branches.py's line is whether a POSITION is stated — `unaffiliated` is a report of "
        "not belonging, `secular` is a stated non-theistic stance — and 'I do not belong to a "
        "religion' is plainly the first. **Nothing in Sweden reaches `secular` at all**, "
        "because ESS never offers atheist or agnostic as a denomination; fi2024.py, gr2024.py "
        "and ge2014.py make the same call for the same reason. **And this cell is not the "
        "complement of Church of Sweden membership**: 54% of Sweden is on the church's own "
        "rolls and 68% of Swedish citizens say they belong to no religion, so most of the "
        "overlap is people who are members and do not consider themselves to belong. "
        "sources/se.md §4 has the comparison.",
}

MAP = {
    "Katolska kyrkan": "christianity.catholic.latin",
    "Svenska kyrkan": "christianity.lutheran",
    "Annan protestantisk församling (t.ex. frikyrka)": "christianity.protestant",
    f"{ORTHODOX_ANSWER} (Eastern Orthodox)": "christianity.orthodox",
    f"{ORTHODOX_ANSWER} (Oriental Orthodox)": "christianity.oriental",
    f"{ORTHODOX_ANSWER} (Church of the East)": "christianity.churchofeast",
    "Annan kristen församling": "christianity",
    "Judisk": "judaism",
    "Islam": "islam.sunni",
    "Österländsk religion (t.ex. Buddhism, Hinduism, Sikh, Shinto, Tao etc.)": "other.se",
    "Annan icke-kristen religion": "other.se",
    "No religion": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
