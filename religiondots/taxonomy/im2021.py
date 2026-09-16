"""
2021 Isle of Man Census (Statistics Isle of Man), Report Part I, Table 2.12, total column ->
religiondots taxonomy.

Six answers with a count, summing to the 74,487 residents who answered a voluntary question.
One geography, the island; see `sources/im.py` and `sources/im.md`.

    Christianity    40,725  54.67%  -> christianity
    No Religion     32,603  43.77%  -> unaffiliated
    Islam              393   0.53%  -> islam
    Buddhism           390   0.52%  -> buddhism
    Hinduism           263   0.35%  -> hinduism
    Judaism            113   0.15%  -> judaism

Table 2.12 also prints `Other` at 0, which `sources/im.py` does not emit. The form (Part II p.43,
Q8) offered Sikhism and a write-in as well; neither reaches the table, and the report does not
say where those answers went (sources/im.md §3).

Christianity is one box on the form, so no denomination can be drawn; it sits on the parent and
is not arguable.
"""

EXCLUDED = {
    "Resident population (Table 2.1 total)":
        "the universe, 84,069 residents (Tables 1.1, 2.1, 2.4), not a religion category. Kept so "
        "tools/gap_share.py can compute the non-answer from the file.",
    "Did not answer (resident population less Table 2.12 total)":
        "9,582 residents, 11.40%, not in Table 2.12 because the question was voluntary (footnote "
        "14), and possibly Sikh and write-in answers too (sources/im.md §3). Not a printed cell: "
        "the difference of two printed totals. Where it leans: 19.2% of residents aged 85 and "
        "over did not answer, against 9.6% to 12.1% in every other age band (Table 2.1 less "
        "Table 2.12); the report gives no reason.",
}

REVIEW = {}

COLUMNS = {
    # Every category is measured at the island, which is the unit drawn, so no row is derived
    # and nothing rolls up. Recorded per COMMANDS.txt's check_rollup note.
}

MAP = {
    "Christianity": "christianity",
    "No Religion": "unaffiliated",
    "Islam": "islam",
    "Buddhism": "buddhism",
    "Hinduism": "hinduism",
    "Judaism": "judaism",
}


def resolve(category):
    """Source category -> node, or None for a category deliberately not on the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
