"""
IBGE Censo 2010 religion classification -> religiondots taxonomy.

**Branch-level mapping, like ca2021.py and cz2021.py.** No leaves are created; the
category's own name travels with the row in `source_category`, so deepening later costs
nothing (spec §2.4).

THE CUT.  IBGE's classification 133 is a NESTED tree, three levels deep, and every level is
published at municipio (sources/br.md §5).  Summing it as delivered triple-counts.  So this
file maps only the LEAVES OF THE SOURCE'S OWN TREE — the deepest category on each path —
and `is_leaf()` computes that from br.py's CATEGORY_PARENT rather than hand-listing it, so
a category IBGE adds later cannot silently be counted twice.

Verified exhaustive before relying on it: on every branch, the children sum to the parent to
within the sample-expansion drift br.md §4 describes.

    Evangelicas de origem pentecostal  25,370,472  children 25,370,485
    Evangelicas de Missao               7,686,827  children  7,686,812
    Evangelicas                        42,275,449  children 42,275,350
    Sem religiao                       15,335,521  children 15,335,517
    Umbanda e Candomble                   588,810  children    588,808
    Novas religioes orientais             155,969  children    155,976

So taking the leaves loses nobody: each parent's "outras" child is a real published
category, not an implied remainder.

WHAT IS NOT HERE.  The 2022 census is nine categories and is not mapped by this file at all
— see countries.py, which draws Brazil on 2010.  spec §3.4's rescale of 2022 totals by 2010
shares is the next step and will need its own correspondence between the two years' lists;
`Outras religiosidades` is the hard part, because it means an 11,307-person leftover in 2010
and a 7,079,101-person catch-all in 2022 (br.md §3).
"""

# Categories that are not religious affiliations and are kept off the tree.
EXCLUDED = {
    "Total":
        "the municipio's own population total, not a category.",
    "Não sabe":
        "196,103 people who did not know. Not a religion and not 'no religion', which is "
        "its own category. Excluded, as Czechia's 'Neuvedeno' is.",
    "Sem declaração":
        "45,841 people who did not answer. Same reasoning. Together with 'Não sabe' this "
        "is 0.13% of Brazil, against 30% in Czechia — the Brazilian question was not "
        "voluntary in the same way.",
}

# Defensible but arguable, recorded so the reasoning is not lost.
REVIEW = {
    "Espírita":
        "-> spiritualism.kardecist. 3,848,897 people, Brazil's fifth largest religion. "
        "Kardecism is the same 19th-century movement as Anglo-American Spiritualism and "
        "institutionally quite separate from it, hence a child node rather than the same "
        "one. Some would file it as its own family.",
    "Evangélicas de origem pentecostal - Igreja Universal do Reino de Deus":
        "-> pentecostal.charismatic. IURD is neo-Pentecostal (1977, prosperity theology, "
        "no classical Pentecostal lineage), so it sits with the later movements rather "
        "than with Assembleia de Deus. Same call for Casa da Benção, Nova Vida and "
        "'Evangélica renovada não determinada'.",
    "Evangélica não determinada":
        "-> christianity.protestant. NINE MILLION people who said 'evangélica' and no "
        "more — 21.8% of all Brazilian evangelicals and the single largest unresolved "
        "category on the Brazilian map. Not pentecostal and not mission: IBGE puts it "
        "beside both, so this file will not guess which.",
    "Católica Apostólica Brasileira":
        "-> catholic.independent. ICAB, founded 1945 by a bishop excommunicated for "
        "rejecting papal authority; uses Catholic orders and rites, not in communion. "
        "That is exactly what the independent node holds.",
    "Outras religiosidades cristãs":
        "-> christianity.other. 1,461,502 and genuinely miscellaneous.",
    "Declaração de múltipla religiosidade":
        "-> other.br. spec §3.3 says syncretism gets a node rather than a split, and this "
        "is 15,387 people declaring exactly that. It is not one syncretic tradition "
        "though — it is the declaration itself — so it goes to the residual rather than "
        "getting a named node of its own.",
    "Tradições indígenas":
        "-> indigenous. 63,083. IBGE does not name individual peoples.",
}

MAP = {
    # ---------------------------------------------------------------- Catholic
    "Católica Apostólica Romana": "christianity.catholic.latin",
    "Católica Apostólica Brasileira": "christianity.catholic.independent",
    "Católica Ortodoxa": "christianity.orthodox.canonical",

    # ---------------------------------------------------------------- mission Protestant
    "Evangélicas de Missão - Igreja Evangélica Luterana": "christianity.lutheran",
    "Evangélicas de Missão - Igreja Evangélica Presbiteriana":
        "christianity.reformed.presbyterian",
    "Evangélicas de Missão - Igreja Evangélica Metodista": "christianity.methodist",
    "Evangélicas de Missão - Igreja Evangélica Batista": "christianity.baptist",
    "Evangélicas de Missão - Igreja Evangélica Congregacional":
        "christianity.reformed.congregational",
    "Evangélicas de Missão - Igreja Evangélica Adventista": "christianity.adventist",
    "Evangélicas de Missão - outras": "christianity.protestant",

    # ---------------------------------------------------------------- Pentecostal
    "Evangélicas de origem pentecostal - Igreja Assembléia de Deus":
        "christianity.pentecostal.trinitarian",
    "Evangélicas de origem pentecostal - Igreja Congregação Cristã do Brasil":
        "christianity.pentecostal.trinitarian",
    "Evangélicas de origem pentecostal - Igreja o Brasil para Cristo":
        "christianity.pentecostal.trinitarian",
    "Evangélicas de origem pentecostal - Igreja Evangelho Quadrangular":
        "christianity.pentecostal.trinitarian",
    "Evangélicas de origem pentecostal - Igreja Deus é Amor":
        "christianity.pentecostal.trinitarian",
    "Evangélicas de origem pentecostal - Igreja Maranata":
        "christianity.pentecostal.trinitarian",
    "Evangélicas de origem pentecostal - Igreja Universal do Reino de Deus":
        "christianity.pentecostal.charismatic",
    "Evangélicas de origem pentecostal - Igreja Casa da Benção":
        "christianity.pentecostal.charismatic",
    "Evangélicas de origem pentecostal - Igreja Nova Vida":
        "christianity.pentecostal.charismatic",
    "Evangélicas de origem pentecostal - Evangélica renovada não determinada":
        "christianity.pentecostal.charismatic",
    "Evangélicas de origem pentecostal - Comunidade Evangélica":
        "christianity.pentecostal.charismatic",
    "Evangélicas de origem pentecostal - outras": "christianity.pentecostal",

    # ---------------------------------------------------------------- evangelical, n.o.s.
    "Evangélica não determinada": "christianity.protestant",

    # ---------------------------------------------------------------- other Christian
    "Outras religiosidades cristãs": "christianity.other",
    "Igreja de Jesus Cristo dos Santos dos Últimos Dias": "christianity.latterday",
    "Testemunhas de Jeová": "christianity.witnesses",

    # ---------------------------------------------------------------- Spiritist
    "Espírita": "spiritualism.kardecist",
    "Espiritualista": "spiritualism",

    # ---------------------------------------------------------------- Afro-Brazilian
    "Umbanda": "afrodiasporic.umbanda",
    "Candomblé": "afrodiasporic.candomble",
    "Outras declarações de religiosidades afrobrasileira": "afrodiasporic",

    # ---------------------------------------------------------------- other families
    "Judaísmo": "judaism",
    "Islamismo": "islam",
    "Hinduísmo": "hinduism",
    "Budismo": "buddhism",
    "Novas religiões orientais - Igreja Messiânica Mundial": "japanesenew",
    "Novas religiões orientais - Outras novas religiões orientais": "japanesenew",
    "Outras religiões orientais": "other.br",
    "Tradições esotéricas": "esoteric",
    "Tradições indígenas": "indigenous",
    "Outras religiosidades": "other.br",

    # ---------------------------------------------------------------- no religion
    "Sem religião - Sem religião": "unaffiliated",
    "Sem religião - Ateu": "secular",
    "Sem religião - Agnóstico": "secular",

    # ---------------------------------------------------------------- residual
    "Não determinada e multiplo pertencimento - Religiosidade não determinada ou mal definida":
        "other.br",
    "Não determinada e multiplo pertencimento - Declaração de múltipla religiosidade":
        "other.br",
}


def leaf_categories(category_parent, names_by_code):
    """The codes with no children — the only ones that may be counted (see THE CUT)."""
    has_child = {p for p in category_parent.values() if p}
    return {names_by_code[c] for c in category_parent if c not in has_child
            and c in names_by_code}


def resolve(category):
    """religiondots branch for an IBGE category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
