"""
ČSÚ Sčítání 2021 religious-belief classification -> religiondots taxonomy.

**Branch-level mapping, like ca2021.py.** All 78 categories map onto branches from
branches.py; no leaves are created. spec §2.4 defers meticulous cross-source matching, and
the category's own name travels with the row in `source_category`, so deepening later costs
nothing and redoes nothing.

Czechia is flat — there is no parent/child structure in the source at all — so this is a
plain dict rather than ca2021.py's ancestor walk.

Three things about this list make it harder than its length suggests (sources/cz.md §5):

  1. **Tradition and institution are separate rows describing overlapping people.** `islám`
     (5,132) and `Ústředí muslimských obcí` (112) both map to `islam`, and that is correct —
     they aggregate to one node here. What must NOT happen is treating the pair as a
     hierarchy, because the tradition row is 46x the institution row and is not its parent.
  2. **Some categories are positions, not religions** — ateismus, agnosticismus, deismus.
     They go to `secular`, which branches.py defines as exactly that.
  3. **Three are parody answers**, and one of them is large. See PARODY below.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

# Categories that are not religious affiliations and are kept off the tree entirely.
EXCLUDED = {
    "Celkem":
        "the unit's own population total, not a category.",
    "Neuvedeno":
        "3,162,540 people — 30.05% of the country — who did not answer a VOLUNTARY "
        "question. Not a religion, and not 'no religion' either, which is its own "
        "category (Bez náboženské víry, 5.03M). Excluded from the dots, so the Czech map "
        "draws 7.36M of 10.52M people. This is the single most important fact about the "
        "Czech map and belongs in note_public, not in a footnote. The share runs 11.4%-"
        "81.3% between municipalities, so it is not even a uniform haircut.",
}

# Defensible but arguable, recorded so the reasoning is not lost and can be overturned.
REVIEW = {
    "Kněžské bratrstvo svatého Pia X.":
        "SSPX -> christianity.catholic.latin. It uses the Latin rite and claims to BE "
        "Roman Catholic; its canonical status is irregular rather than schismatic-by-"
        "self-description. Not christianity.catholic.independent, which is the Old "
        "Catholic / Polish National line that does not claim communion with Rome.",
    "satanismus":
        "-> esoteric. LaVeyan and most Czech Satanism is atheistic and genealogically "
        "part of Western occultism, so it is not paganism and not a theistic family of "
        "its own. 998 people.",
    "animismus":
        "-> paganism. A Czech write-in 'animismus' is a Western neo-animist self-"
        "description, not the traditional religion of an indigenous people, so "
        "`indigenous` would be the wrong family. 42 people.",
    "Višva Nirmala Dharma":
        "Sahaja Yoga -> hinduism. A Hindu-derived new religious movement; could equally "
        "sit under `esoteric` with the other NRMs.",
    "Obec křesťanů v České republice":
        "The Christian Community -> christianity.other. Anthroposophical in origin "
        "(founded with Steiner's help) but Christian in liturgy and self-description.",
    "Českobratrská církev evangelická":
        "-> christianity.reformed. It is a 1918 union of Reformed and Lutheran churches "
        "and is the largest Protestant body in the country; filed Reformed because that "
        "is the dominant strand, but the Lutheran half is real.",
    "Církev bratrská":
        "-> christianity.pietist. The Czech member of the International Federation of "
        "Free Evangelical Churches, which is what branches.py's pietist node holds.",
    "věřící - hlásící se k církvi - název neuveden":
        "-> other.cz. 'Believer, belongs to a church, church not named' — overwhelmingly "
        "Christian in Czechia, but the source does not say so and this file will not "
        "assert it. 65,567 people.",
}

# The three parody / protest answers. Jedi is NOT negligible: at 21,023 it is the 13th
# largest category in the country, ahead of Jehovah's Witnesses, and present in 2,512 of
# 6,254 municipalities. ČSÚ tabulated them because respondents wrote them in.
#
# They get their own family rather than being dropped or filed under a religion. Dropping
# them would silently lose 24,235 visible people; filing them under `paganism` or
# `other.cz` would assert something false. `parody` is honest and lets the viewer decide.
PARODY = ["Jedi", "Sith", "pastafariánství"]

MAP = {
    # ---------------------------------------------------------------- no religion
    "Bez náboženské víry": "unaffiliated",
    "ateismus": "secular",
    "agnosticismus": "secular",
    "deismus": "secular",

    # Believing without belonging — 960,201 people, 9.1% of the country, and NOT the same
    # as `unaffiliated`, which is a report of no religion.
    "věřící - nehlásící se k žádné církvi ani náboženské společnosti": "unchurched",

    # ---------------------------------------------------------------- Catholic
    "Církev římskokatolická": "christianity.catholic.latin",
    "katolická víra (katolík)": "christianity.catholic",
    "Kněžské bratrstvo svatého Pia X.": "christianity.catholic.latin",
    "Církev řeckokatolická": "christianity.catholic.eastern",
    "Starokatolická církev v ČR": "christianity.catholic.independent",

    # ---------------------------------------------------------------- Orthodox
    "Pravoslavná církev v českých zemích": "christianity.orthodox.canonical",
    "Ruská pravoslavná církev, podvorje patriarchy moskevského a celé Rusi v České republice":
        "christianity.orthodox.canonical",
    "Církev Svatého Řehoře Osvětitele": "christianity.oriental",

    # ---------------------------------------------------------------- Protestant
    "Českobratrská církev evangelická": "christianity.reformed",
    "Evangelická církev augsburského vyznání v České republice": "christianity.lutheran",
    "Luterská evangelická církev a. v. v České republice": "christianity.lutheran",
    "Slezská církev evangelická augsburského vyznání": "christianity.lutheran",
    "Evangelická církev metodistická": "christianity.methodist",
    "Bratrská jednota baptistů": "christianity.baptist",
    "Společenství baptistických sborů": "christianity.baptist",
    "Jednota bratrská": "christianity.moravian",
    "Křesťanské sbory": "christianity.plymouth",
    "Církev bratrská": "christianity.pietist",
    "Armáda spásy - církev": "christianity.holiness",
    "Anglikánská církev": "christianity.anglican",
    "protestantská/evangelická víra (protestant, evangelík)": "christianity.protestant",

    # ---------------------------------------------------------------- Hussite
    "Církev československá husitská": "christianity.hussite",

    # ---------------------------------------------------------------- Pentecostal
    "Apoštolská církev": "christianity.pentecostal.trinitarian",
    "Církev Křesťanská společenství": "christianity.pentecostal.charismatic",
    "Církev víry": "christianity.pentecostal.charismatic",
    "Církev Slovo života": "christianity.pentecostal.charismatic",
    "Církev živého Boha": "christianity.pentecostal.charismatic",
    "Církev Nová naděje": "christianity.pentecostal.charismatic",
    "Církev Oáza": "christianity.pentecostal.charismatic",
    "Církev Nový Život": "christianity.pentecostal.charismatic",

    # ---------------------------------------------------------------- other Christian
    "Církev adventistů sedmého dne": "christianity.adventist",
    "Církev Ježíše Krista Svatých posledních dnů v České republice": "christianity.latterday",
    "Náboženská společnost Svědkové Jehovovi": "christianity.witnesses",
    "Náboženská společnost českých unitářů": "unitarianuniversalist",
    "Novoapoštolská církev v ČR": "christianity.other",
    "Obec křesťanů v České republice": "christianity.other",
    "Křesťanská církev essejská": "christianity.other",
    "křesťanství": "christianity",

    # ---------------------------------------------------------------- other families
    "islám": "islam",
    "Ústředí muslimských obcí": "islam",
    "judaismus": "judaism",
    "Federace židovských obcí v České republice": "judaism",
    "buddhismus": "buddhism",
    "Buddhismus Diamantové cesty linie Karma Kagjü": "buddhism",
    "Théravádový buddhismus": "buddhism",
    "Společenství buddhismu v České republice": "buddhism",
    "hinduismus": "hinduism",
    "Česká hinduistická náboženská společnost": "hinduism",
    "Mezinárodní společnost pro vědomí Krišny, Hnutí Hare Krišna": "hinduism",
    "Višva Nirmala Dharma": "hinduism",
    "sikhismus": "sikhism",
    "taoismus": "daoism",
    "šintoismus": "shinto",
    "konfucianismus": "chinesefolk",
    "zoroastrismus": "zoroastrianism",
    "Bahá'í víra": "bahai",
    "rastafariánství": "rastafari",
    "Církev sjednocení (moonisté)": "unification",
    "Scientologická církev": "scientology",

    # ---------------------------------------------------------------- pagan / esoteric
    "pohanství": "paganism",
    "druidismus": "paganism",
    "animismus": "paganism",
    "esoterismus": "esoteric",
    "Hnutí Nového věku (New Age)": "esoteric",
    "Hnutí Grálu": "esoteric",
    "Společenství Josefa Zezulky": "esoteric",
    "satanismus": "esoteric",

    # ---------------------------------------------------------------- parody
    "Jedi": "parody",
    "Sith": "parody",
    "pastafariánství": "parody",

    # ---------------------------------------------------------------- residual
    # spec §3.11: residual buckets are per source and never merged with another
    # country's, because they contain different things.
    "Jiné": "other.cz",
    "věřící - hlásící se k církvi - název neuveden": "other.cz",
}


def resolve(category):
    """religiondots branch for a ČSÚ category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
