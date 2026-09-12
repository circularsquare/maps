"""Geostat 2014 census religion classification -> religiondots taxonomy.

Twelve answers plus the universe total, at region. A short list on a coarse geography — but
an unusually well-chosen list, and the geography is the point rather than the grain.

    83.41%  Orthodox                  -> christianity.orthodox.canonical
    10.74%  Muslim                    -> islam
     2.94%  Armenian apostolic        -> christianity.oriental.armenian
     0.92%  Not stated                -> EXCLUDED (non-response)
     0.52%  Catholic                  -> christianity.catholic
     0.51%  None                      -> unaffiliated
     0.33%  Jehovah's Witnesses       -> christianity.witnesses
     0.26%  Refusal                   -> EXCLUDED (non-response)
     0.23%  Yazidis                   -> yazidism
     0.07%  Protestant                -> christianity.protestant
     0.04%  Other                     -> other.ge   (a NEW node)
     0.04%  Judaism                   -> judaism

**Four of the twelve are here because Georgia is where they are**, which is why a 12-answer
form on 11 units is worth drawing: Armenian Apostolic, Yazidi, and two entirely separate
Muslim populations that share one cell.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Refusal":
        "9,635 people, 0.26% — an explicit refusal to answer (spec §3.5), kept apart from "
        "`Not stated`. Geostat publishing both is unusual and useful: most sources have one "
        "cell doing both jobs and there is no way to tell a refusal from a blank.",
    "Not stated":
        "34,251 people, 0.92% — item non-response, three and a half times the refusals. "
        "Together the two are 1.18%, which is small by the standards of any voluntary "
        "religion question on this map (Czechia 30%, Hungary 40%).",
}

REVIEW = {
    "Orthodox":
        "-> christianity.orthodox.canonical. The Georgian Apostolic Autocephalous Orthodox "
        "Church, in communion throughout and one of the oldest autocephalies anywhere — "
        "Georgia adopted Christianity in the 320s-330s. 3,097,573 people, 83.4%. The cell "
        "also carries the Russian, Greek and Armenian-Chalcedonian Orthodox minorities, "
        "which the census does not separate.",
    "Muslim":
        "-> islam, the parent, with no branch — and **this is the most consequential "
        "collapse in the file**, because Georgia's Muslims are two unrelated populations "
        "that the map will show sitting in different places without being able to name "
        "them. In **Adjara** (132,852, 40% of the region) they are **Georgian-speaking "
        "Sunnis**, converted under three centuries of Ottoman rule and Georgian in every "
        "other respect. In **Kvemo Kartli** (182,216, 43%) and lowland **Kakheti** (38,683) "
        "they are **Azerbaijanis, largely Shia**, on the border with Azerbaijan. One cell "
        "of 398,677 for a Sunni/Shia and Georgian/Azeri split that runs the length of the "
        "country. Nothing in the table permits the split and doing it from ethnicity would "
        "be §14.5's derivation on a religiously mixed group.",
    "Armenian apostolic":
        "-> christianity.oriental.armenian. **109,041 people**, 40% of Samtskhe-Javakheti, "
        "the Armenian-majority south where Akhalkalaki and Ninotsminda are. This file "
        "argued for the bare parent from 2026-09-08 back, on the grounds that Australia and "
        "Estonia filed Armenian bodies there too and moving one country alone would assert "
        "a distinction the others do not make. Armenia's 2,793,041 reopened it as "
        "`ask/004-am` and Anita's ruling was **record whatever the source actually says**: "
        "Geostat's category names the Armenian church, so it files at an Armenian node, and "
        "so do Australia's and Estonia's, because they name it too. Georgia and Armenia now "
        "read as one church across the border, which they are.",
    "Yazidis":
        "-> yazidism, a root and not a branch of Islam or Zoroastrianism, which is how the "
        "node was already written for Australia. **8,591 people, and 8,124 of them — 95% — "
        "are in Tbilisi**, the most concentrated religious population in this source. "
        "Georgia's Yazidis are Kurmanji-speaking, arrived largely as refugees from the "
        "Ottoman persecutions of the 1910s-20s, and the Sultan Ezid temple in Tbilisi "
        "(2015) is one of very few purpose-built Yazidi temples outside Iraq. This more "
        "than doubles the node's population on the map.",
    "None":
        "-> unaffiliated and NOT `secular`. One no-religion answer with no atheist/agnostic "
        "split. 19,080 people, 0.51% — **the second-lowest irreligious share on this map "
        "after Kosovo's**, and a striking figure for a country that spent seventy years in "
        "the Soviet Union.",
    "Other":
        "-> other.ge, a per-source residual (§3.11). See branches.py.",
}

MAP = {
    "Orthodox": "christianity.orthodox.canonical",
    "Muslim": "islam",
    "Armenian apostolic": "christianity.oriental.armenian",
    "Catholic": "christianity.catholic",
    "Jehovah’s Witnesses": "christianity.witnesses",
    "Yazidis": "yazidism",
    "Protestant": "christianity.protestant",
    "Judaism": "judaism",
    "Other": "other.ge",
    "None": "unaffiliated",
}


def _key(cat):
    # Geostat writes a CURLY apostrophe in `Jehovah’s Witnesses` and pads some labels; both
    # are normalised here so the table above can be typed either way.
    return " ".join(str(cat).replace("’", "'").split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
