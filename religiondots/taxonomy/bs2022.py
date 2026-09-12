"""BNSI 2022 census religion classification -> religiondots taxonomy.

Twenty-five categories plus a per-island residual, on 18 islands. The shares below are of
the whole census population, 398,165, and are the ISLAND sums — which is what gets drawn,
and which falls short of the first release's national table by exactly the 372 people the
small islands folded into `OTHER RELIGION` (`sources/bs.py`).

    34.13%  BAPTIST                      -> christianity.baptist
    11.92%  ANGLICAN                     -> christianity.anglican
     8.86%  OTHER CHRISTIAN DENOMINATION -> christianity              REVIEW
              (INCLUDING NON-DENOMINATIONAL GROUPS)
     8.72%  ROMAN CATHOLIC               -> christianity.catholic
     8.02%  PENTECOSTAL                  -> christianity.pentecostal
     6.20%  NONE                         -> unaffiliated
     4.94%  CHURCH OF GOD AND CHURCH     -> christianity.pentecostal  REVIEW
              OF GOD OF PROPHECY
     4.79%  NOT STATED                   -> EXCLUDED
     4.39%  SEVENTH DAY ADVENTIST        -> christianity.adventist
     2.76%  METHODIST                    -> christianity.methodist
     1.46%  BRETHREN                     -> christianity.plymouth     REVIEW
     1.05%  JEHOVAH'S WITNESS            -> christianity.witnesses
     0.93%  ASSEMBLIES OF GOD            -> christianity.pentecostal.trinitarian
     0.45%  OTHER NON-CHRISTIAN RELIGION -> other.bs   (a NEW node)
     0.28%  RASTAFARIAN                  -> rastafari
     0.26%  AFRICAN METHODIST EPISCOPAL  -> christianity.methodist.african
              (AME)
     0.22%  PRESBYTERIAN                 -> christianity.reformed.presbyterian
     0.12%  LUTHERAN                     -> christianity.lutheran
     0.09%  OTHER RELIGION               -> other.bs                  REVIEW
     0.08%  GREEK ORTHODOX               -> christianity.orthodox.canonical
     0.07%  ISLAM (MUSLIM)               -> islam
     0.07%  HINDU                        -> hinduism
     0.07%  ATHEIST                      -> secular
     0.04%  MORMON                       -> christianity.latterday
     0.04%  JUDAISM (JEWISH)             -> judaism
     0.01%  BAHAI FAITH                  -> bahai

**THE BAHAMAS IS THE MOST BAPTIST COUNTRY ON THIS MAP AND IT IS NOT CLOSE.** One person in
three, 135,875 of them. The next-highest Baptist share among the countries drawn here is
Saint Vincent's 8.86% — a quarter of it — and Jamaica, 500 km south and with a Baptist
history at least as old, is 6.74%. Nothing about being Caribbean or Anglophone predicts
this; it is a fact about the Bahamas.

**AND IT IS THE FIRST CENSUS OUTSIDE THE UNITED STATES TO COUNT AFRICAN METHODISTS UNDER
THEIR OWN NAME.** `christianity.methodist.african` has existed since the U.S. Religion
Census arrived and no other source here had reached it. 1,028 Bahamians, against 10,983 in
the ordinary Methodist cell, and the census keeps them apart.

**IT ALSO SEPARATES `NONE` FROM `ATHEIST`, WHICH ALMOST NOTHING HERE DOES.** 24,668 against
281. That is exactly the distinction `branches.py` draws between `unaffiliated` — the
absence of an answer — and `secular` — a stated position — and most censuses collapse it,
so most of this map's countries can only reach the first. Georgia, Portugal, Macedonia,
Croatia, Ghana and Kenya each had to be mapped to `unaffiliated` alone for want of the
second cell. The Bahamas offers both boxes and 0.07% of the country picks the second one.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "TOTAL":
        "the island's own population total, not a category. Carried in bs.csv because "
        "sources/bs.py checks the categories against it.",
    "NOT STATED":
        "19,074 people, **4.79%**. Spec §3.5: marked, not filled, and never redistributed, "
        "so every share drawn for the Bahamas is a share of the whole population rather "
        "than of the people who answered. **Its geography is mild by the standards of this "
        "map** — 5.57% on New Providence against 2.08% on Grand Bahama and under 1% on "
        "most Family Islands — which is the ordinary big-city pattern and not the "
        "unexplained spread Trinidad's 11.10% has (tt2011.py). It is the only thing "
        "keeping Bahamian coverage below 96%.",
}

REVIEW = {
    "OTHER CHRISTIAN DENOMINATION (INCLUDING NON-DENOMINATIONAL GROUPS)":
        "-> `christianity`, the branch root, and NOT `christianity.other`. 35,278 people, "
        "**8.86% — the third largest Christian answer in the country, ahead of the Roman "
        "Catholics at 8.72%**, so this is not a tail. "
        "**The cell holds two different things and its own label says so**: Christian "
        "bodies with no box of their own, AND non-denominational groups. `christianity."
        "nondenominational` is a real node on this tree, and `christianity.other` is "
        "explicitly *'bodies with no branch to belong to, NOT a residual'* — so filing the "
        "cell there would place the non-denominational half in a node defined to exclude "
        "it. The branch root is the only node that contains everything the cell contains, "
        "and it is the call `uk2021.py`, `fr2024.py`, `gr2024.py`, `li2015.py` and "
        "`lk2024.py` already make for the same shape of cell. "
        "**BNSI's own commentary reads this cell as the non-denominational answer** — the "
        "first release's text is about `Non-Denominational` overtaking Roman Catholic to "
        "become one of the country's three main denominations, and this is the cell that "
        "did it. That is a good reason to record what is probably inside it and a bad "
        "reason to assert a split the table does not publish (§14.4).",
    "CHURCH OF GOD AND CHURCH OF GOD OF PROPHECY":
        "-> christianity.pentecostal. 19,654 people, 4.94%. Two bodies in one cell, and "
        "**the pairing is what identifies them**: the Church of God of Prophecy separated "
        "from the Church of God (Cleveland, Tennessee) in the 1923 split, so a form that "
        "offers the two together is naming the Cleveland family and not the Church of God "
        "(Anderson, Indiana), which is Holiness and explicitly not Pentecostal. Both "
        "Cleveland bodies are Pentecostal on either side of their split. "
        "**This follows `jm2011.py` exactly**, where STATIN splits the same family four "
        "ways and the two Cleveland-lineage cells go to `christianity.pentecostal` while "
        "the Anderson-lineage one goes to `christianity.holiness`. The Bahamian form does "
        "not offer an Anderson cell at all, which is itself consistent — §12's warning is "
        "against mapping a category on its string, and here the string is disambiguated by "
        "what it is printed next to.",
    "BRETHREN":
        "-> christianity.plymouth, NOT christianity.anabaptist.brethren. 5,811 people, "
        "1.46%, and the same collision `jm2011.py` documents: the Brethren assemblies of "
        "the Anglophone Caribbean are the **Plymouth / Christian Brethren**, out of 1820s "
        "Dublin and arriving through nineteenth-century British missions. The Schwarzenau "
        "(German Baptist) Brethren are an unrelated body with no Bahamian presence. §12: "
        "never map a category on its string alone.",
    "OTHER RELIGION":
        "-> other.bs, together with `OTHER NON-CHRISTIAN RELIGION`, and this is the one "
        "call in the file that is a compromise rather than a reading. "
        "**It is not a religion category. It is a disclosure artefact with a footnote.** "
        "Only New Providence prints all 24 named bodies; every other island folds its "
        "smallest answers into a single `OTHER RELIGION*` cell, and a starred footnote "
        "under each table names exactly which ones. So this cell's contents are DIFFERENT "
        "ON EVERY ISLAND — Roman Catholics and Pentecostals on Ragged Island, Greek "
        "Orthodox and Jews on Long Island, Hindus and Muslims on Cat Island — and it holds "
        "no single thing the tree could point at. "
        "**It is 372 people, 0.093% of the country**, which is why folding it here costs "
        "almost nothing. But two islands are not almost nothing: **Ragged Island's "
        "residual is 30.4% of its 56 people and Mayaguana's is 11.8% of its 203**, so on "
        "those two the map shows a genuinely wrong composition for a hundred-odd people. "
        "**And on exactly those two islands the footnote says `None` was folded in**, so "
        "Mayaguana and Ragged Island report no irreligion at all and their non-religious "
        "people — 41 between the two residuals, at most — are drawn as a religion. "
        "**The alternative was to drop the cell, and that is worse**: it would delete a "
        "third of Ragged Island and an eighth of Mayaguana from the map rather than "
        "misfiling them, and §3.5's rule about not redistributing a non-answer does not "
        "apply, because this is an answer — BNSI suppressed the label, not the person. "
        "Recorded, not corrected (§14.4). `sources/bs.py` prints the per-island residual "
        "and its footnote on every run, and `bs.csv` carries the footnote text in the "
        "`note` column of each residual row.",
    "OTHER NON-CHRISTIAN RELIGION":
        "-> other.bs. 1,799 people, 0.45%, and a NARROW residual rather than a wide one: "
        "the form already names Bahá'í, Hindu, Muslim, Jewish and Rastafari separately, so "
        "this is the tail after five non-Christian boxes rather than a bucket standing in "
        "for them. Its likeliest contents are named in branches.py.",
    "ATHEIST":
        "-> secular, and `NONE` -> unaffiliated, which is the distinction branches.py draws "
        "and which most censuses here cannot support. 281 people against 24,668. A stated "
        "atheism is a position; `NONE` is the absence of one.",
}

MAP = {
    "ANGLICAN": "christianity.anglican",
    "ASSEMBLIES OF GOD": "christianity.pentecostal.trinitarian",
    "BAPTIST": "christianity.baptist",
    "BRETHREN": "christianity.plymouth",
    "CHURCH OF GOD AND CHURCH OF GOD OF PROPHECY": "christianity.pentecostal",
    "GREEK ORTHODOX": "christianity.orthodox.canonical",
    "JEHOVAH'S WITNESS": "christianity.witnesses",
    "LUTHERAN": "christianity.lutheran",
    "METHODIST": "christianity.methodist",
    "PENTECOSTAL": "christianity.pentecostal",
    "PRESBYTERIAN": "christianity.reformed.presbyterian",
    "ROMAN CATHOLIC": "christianity.catholic",
    "SEVENTH DAY ADVENTIST": "christianity.adventist",
    "MORMON": "christianity.latterday",
    "OTHER CHRISTIAN DENOMINATION (INCLUDING NON-DENOMINATIONAL GROUPS)": "christianity",
    "BAHAI FAITH": "bahai",
    "HINDU": "hinduism",
    "ISLAM (MUSLIM)": "islam",
    "JUDAISM (JEWISH)": "judaism",
    "RASTAFARIAN": "rastafari",
    "OTHER NON-CHRISTIAN RELIGION": "other.bs",
    "NONE": "unaffiliated",
    "AFRICAN METHODIST EPISCOPAL (AME)": "christianity.methodist.african",
    "ATHEIST": "secular",
    "OTHER RELIGION": "other.bs",
}


def _key(cat):
    """Uppercase alphanumerics only.

    The source spells one category two ways — `JEHOVAH’S WITNESS` with a curly apostrophe
    on five islands and a straight one on three — and prints the residual as `OTHER
    RELIGION*` on sixteen of the seventeen islands that have one. `sources/bs.py` already
    normalises both before writing bs.csv, and this repeats the fold so the module is
    correct on its own terms rather than only in the pipeline (§12).
    """
    return "".join(c for c in str(cat).upper() if c.isalnum())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    **`NONE` IS A CATEGORY NAME HERE, NOT A MISSING VALUE** — 24,668 people, 6.20% of the
    country. Any caller reading bs.csv with a bare `pandas.read_csv` will have turned those
    sixteen rows into NaN before this function sees them; see `_bs_counts` in countries.py,
    which passes `keep_default_na=False`.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
