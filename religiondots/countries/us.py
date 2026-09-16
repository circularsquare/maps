# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403

import numpy as np


def _us_census(level):
    """sources/us_dhca.py's drawn rows at one level: the 2020 census's Sikh and Yazidi write-ins.

    The `alone` iterations are witnesses with no node, and a withheld cell has no count; both
    are dropped. `na_values=[""]` is what turns a witness row's empty node into NaN, so filter on
    `notna()`, never on `!= ""` (a scratch script did, and read 45,867 witness people as drawn).
    """
    d = pd.read_csv(HERE / "data" / "normalized" / "us_dhca.csv",
                    dtype={"geoid": str, "iterid": str}, keep_default_na=False, na_values=[""])
    return d[(d["level"] == level) & d["node"].notna() & d["count"].notna()]


def _us_census_counts():
    """County Sikhs and Yazidis from the 2020 census race write-in, `measured`, rings allowed.

    A count read at the county the census printed it for, so `measured`, like an ASARB roll read
    in its county. It is not a roll: it counts the people who wrote the word in the race question
    (sources/us_dhca.md §2) and is drawn as counted, never scaled. The Sikhs it misses stay in
    `other.us`, because these rows are subtracted from Pew's other-world-religions line the way
    the Bahá'í roll is (taxonomy/us_pew2024.py, REVIEW).
    """
    d = _us_census("COUNTY")
    return pd.DataFrame({"unit": d["geoid"].to_numpy(), "node": d["node"].to_numpy(),
                         "count": d["count"].astype(float).to_numpy(), "congregations": 0,
                         "may_ring": True, "tier": "measured"})


def _us_measured():
    """Everything the US draws as `measured`: ASARB's county rolls plus the census's county counts.

    §3.5a's residual is taken against this frame (us_rebase.py), because a residual has to be
    measured against what is drawn. It is NOT what us_weights.py fits on; that stays `_us_counts`.
    """
    roll = _us_counts()
    roll["may_ring"] = True
    roll["tier"] = "measured"
    return pd.concat([roll, _us_census_counts()], ignore_index=True)


class _UsCensusTracts:
    """§8.4's weighter with the census's printed tracts in front, for `sikhism` and `yazidism`.

    Inside a county, the dots go first to the tracts the census printed a count for, in
    proportion to it; whatever the county holds beyond them goes over its other tracts on the
    weight us_weights.py gives the node (for Sikhs the authored tie: Punjabi at home, the ACS
    Sikh answer, a little South Asian ancestry). Noise is added to every cell separately, so the
    printed tracts can hold more than their county; the county count still sets the dots (§4.1)
    and the tracts only share them out. Every other node goes straight to the inner weighter.
    """

    def __init__(self, inner, place):
        self.inner = inner
        key = pd.Series(place["GEOID"].astype(str).to_numpy())
        rows = key.map(key.value_counts()).to_numpy(float)   # a tract split into parts shares its count
        t = _us_census("TRACT")
        self.tracts = {}
        for node, g in t.groupby("node"):
            per = dict(zip(g["geoid"], g["count"].astype(float)))
            self.tracts[node] = key.map(per).fillna(0.0).to_numpy(float) / rows
        self.n_census = 0

    def weights(self, node, idx, count, plain=False):
        base = self.inner.weights(node, idx, count, plain=plain) if self.inner is not None else None
        pub = self.tracts.get(node)
        if pub is None or plain:
            return base
        pub = pub[idx]
        if pub.sum() <= 0:
            return base
        self.n_census += 1
        rest = count - pub.sum()
        if rest <= 0:
            return pub
        other = np.ones(len(idx)) if base is None else np.asarray(base, dtype=float)
        other = np.where(pub > 0, 0.0, other)
        if other.sum() <= 0:
            return pub
        return pub + rest * other / other.sum()

    def summary(self):
        w = self.inner
        head = (f"{w.n_weighted:,} (unit, node) rows placed on fitted demographic weights, "
                f"{w.n_authored:,} on an authored ethnic tie, {w.n_residual:,} on the CES-fitted "
                f"residual model, {w.n_uniform:,} on population alone (§8.4, §8.4a)"
                if w is not None else "no §8.4 weighter; equal shares per tract")
        return (head + f"; {self.n_census:,} Sikh or Yazidi county rows placed on the census's "
                       f"printed tracts first")


def _us_counts():
    """ASARB 2020: county x 372 bodies, already mapped to the taxonomy by hand."""
    paths = pd.read_csv(HERE / "taxonomy" / "usrc_groups.csv", dtype={"Group Code": str})
    path_of = dict(zip(paths["Group Code"], paths["path"]))
    df = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Group_Detail.xlsx",
                       sheet_name="2020 Group by County",
                       dtype={"FIPS": str, "Group Code": str})
    df["unit"] = df["FIPS"].str.strip().str.zfill(5)
    df["node"] = df["Group Code"].str.strip().str.zfill(3).map(path_of)
    df = df[df["node"].notna() & (df["node"] != "UNMAPPED")]
    return df.rename(columns={"Adherents": "count", "Congregations": "congregations"})[
        ["unit", "node", "count", "congregations"]]


def _us_counts_rebased():
    """spec §3.5a: ASARB's rolls, plus the self-identification residual on top.

    ASARB's numbers are untouched — every one of its 372 bodies keeps the county figure it
    always had, tagged `measured`. What is added is one row per (county, root) for the people
    the survey finds and no roll holds, tagged `derived` — recorded per §7, drawn identically
    to everything else since the desaturation was removed on 2026-09-04.

    `_us_counts` stays ASARB-only ON PURPOSE and is not merely an implementation detail:
    us_weights.py fits its §8.4 demographic model against it, and fitting a model of where
    ASARB's adherents live against rows that are a survey residual would be training on the
    output. The two functions must not be merged.

    Since 2026-09-15 the measured half also holds the 2020 census's county Sikhs and Yazidis
    (`_us_census_counts`), and the residual is taken against both.
    """
    from us_rebase import residual_counts

    measured = _us_measured()
    res = residual_counts(measured)  # measured against the frame as drawn, not ASARB's state sheet
    return pd.concat([measured, res], ignore_index=True)


def _us_place_weight(place):
    """spec §8.4, with the census's tracts in front for Sikhs and Yazidis (`_UsCensusTracts`).

    us_weights is imported lazily so a country that does not use it never pays for it."""
    from us_weights import load_weighter

    return _UsCensusTracts(load_weighter(place), place)


ENTRY = {
    "us": dict(
        name="United States",
        name_in="the United States",
        source="U.S. Religion Census 2020 (ASARB), Pew Religious Landscape Study 2023-24 and the "
               "2020 Census",
        basis="self-identification, with membership rolls inside it",
        # §3.5a's "the declaration stays quiet": one sentence here, the numbers in the build
        # log and in counts.json for anyone who looks, and nothing on the map itself. It
        # assumed §7's desaturation would carry "this is modelled" on screen; that was
        # removed 2026-09-04, so THIS NOTE IS NOW THE ONLY PLACE A READER LEARNS IT, and the
        # sentence about the paler dots below has to keep doing that work alone.
        note_public=(
            "The United States asks no religion question, so the totals here are Pew's "
            "survey and the detail inside them is the U.S. Religion Census — 372 bodies "
            "reporting who is on their books, which is 48.4% of the country. A little over "
            "half of the dots here are the difference between the two: people the survey "
            "finds and no membership roll holds, placed among each county's residents who "
            "are on nobody's roll. Those are an estimate and are drawn the same as the "
            "counted ones — nothing on the map marks which is which. It is also the only "
            "reason an American non-religious population can be drawn at all: a roll's "
            "residual means “on no roll”, which is not “no religion”. "
            "Two things it rests on: the survey counts adults and this applies their answers "
            "to children too, and 1.4% of people answered nothing and are not drawn at all. "
            "Adherents are attributed to "
            "the congregation's county rather than the member's home. Counties are the "
            "finest thing anyone counts — the study does not record congregation addresses "
            "— so where a dot sits INSIDE a county is an estimate from the neighbourhood's "
            "ancestry and birthplace, not a measurement, and only for bodies where that "
            "could be checked against the county figures. The rest are spread across the "
            "county's population. Judaism is the known bad case: nothing in the census marks "
            "it, so Jewish neighbourhoods are not drawn as Jewish and the bodies that can be "
            "placed take the space instead. "
            "Sikhs and Yazidis come from the 2020 census instead, which has no religion "
            "question either but codes what people write in its race question. It counted "
            "**68,483** Sikhs in the 213 counties where the figure is large enough to print, "
            "placed inside each county on the tracts where it printed one, and 551 Yazidis, 476 "
            "of them in Lancaster County, Nebraska, which is too few for a dot and is drawn as a "
            "ring. Only people who wrote the word are counted. The Sikh Coalition estimates more "
            "than 500,000 Sikhs in the United States, and the ones the census missed are in "
            "“other religion” with the Daoists, Jains and Zoroastrians the survey cannot "
            "tell apart."),
        view=[-125.0, 24.0, -66.5, 49.8],
        how="church membership rolls, topped up by a national survey",
        grain="counties, 104,000 people on average",
        counts=_us_counts_rebased,
        units=None,              # counts are on counties; tracts carry both ids
        unit_key=None,
        place=HERE / "data" / "geo" / "tracts2020" / "cb_2020_us_tract_500k.shp",
        place_unit=lambda g: g["STATEFP"] + g["COUNTYFP"],
        # spec §8.4. Real ACS tract populations for every node instead of §8.2's equal-share
        # approximation, plus a demographic redistribution inside the county for the 26 nodes
        # whose held-out-metro correlation earned one. Falls back to §8.2 if unbuilt.
        place_weight=_us_place_weight,
        note="re-based on self-identification (spec §3.5a): Pew supplies the root totals, "
             "ASARB's rolls are the structure inside them, and the residual is drawn "
             "`modelled` — half the American dots, and nobody counted them at any level "
             "(us_rebase.py, 2026-09-05). Adherents are attributed to the congregation's county, not the "
             "member's (§3.6). Within a county, placement is a demographic estimate for "
             "some bodies and population-weighted for the rest (§8.4). Sikhs and Yazidis are the "
             "2020 census race write-in by county, `measured`, subtracted from Pew's "
             "other-world-religions line like the Bahá'í roll and placed on the census's printed "
             "tracts first (sources/us_dhca.md, 2026-09-15).",
    ),
}
