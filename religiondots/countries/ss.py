# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ss_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer, calibrated to the 2025 county estimates.

    Six states over 360,000 km2, and the people are on the rivers and in a handful of towns:
    sources/ss_geo.py scales every Kontur hex to its county's 2025 total (raw Kontur reads 2.21x
    the estimate in Eastern Equatoria and 0.51x in Western Bahr el Ghazal).
    """
    return _kontur_place_weight(place, "ss_hexes.gpkg", "sources/ss_geo.py")


def _ss_counts():
    """The High Frequency Survey's household heads at former state: 6 of 10 states, EVERY ROW `modelled`.

    Each sampled state's shares are wave 1's (2015) person-weighted household heads, laid on the
    state's 2025 county-based estimate. Jonglei, Unity, Upper Nile and Warrap have no rows and draw
    nothing (Ecuador's Galapagos construction): they keep their polygons and hexes. sources/ss.md.
    """
    from ss2015 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ss.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "ss" / "ss_lookup.csv", dtype=str)
    missing = sorted(set(df["geo_id"]) - set(lut["unit"]))
    if missing:
        raise SystemExit(f"ss.csv states with no polygon: {missing}; re-run sources/ss_geo.py")
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"{df['geo_id'].nunique()} states in ss.csv, expected the 6 sampled")
    if set(df["geo_id"]) & {"SS03", "SS06", "SS07", "SS08"}:
        raise SystemExit("an unsampled state is in ss.csv and must not be; re-run sources/ss.py")

    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(EXCLUDED))
    if unmapped:
        raise SystemExit(f"ss.csv answers with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    # EVERY row: a survey share on an estimate (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ss": dict(
        name="South Sudan",
        source="High Frequency South Sudan Survey, wave 1, 2015 (World Bank and National Bureau "
               "of Statistics), household heads, on the 2025 county population estimates adopted "
               "by OCHA and the National Bureau of Statistics",
        basis="self-identification, religion of the household head",
        view=[23.4, 3.4, 36.0, 12.3],
        note_public=(
            "**South Sudan has never asked about religion in a census.** The 2008 census, taken "
            "while the south was still part of Sudan, had the question removed. This map is "
            "drawn from a household survey instead: the World Bank and the National Bureau of "
            "Statistics' High Frequency South Sudan Survey, which in 2015 asked **3,550 household "
            "heads** in six states what their religion was. Each state's shares are laid on its "
            "2025 population estimate, and the dots disappear when inferred dots are turned off "
            "because they are survey shares and not a count. "
            "**Everyone in a household is drawn in the head's column.** 88.8% Christian means "
            "that share of people in the six states live in a household whose head is Christian. "
            "**Four states are blank.** The survey never went to Jonglei, Unity or Upper Nile, "
            "where the war that began in December 2013 was fought, and it went to Warrap's towns "
            "only, a year later. Towns and countryside answer differently here: in Northern Bahr "
            "el Ghazal, next door, 7.2% of town households followed traditional religion against "
            "**24.8%** in the countryside, so Warrap's towns cannot stand in for Warrap. The four "
            "hold **48.2%** of the country's people, and rather than paint them in the colours of "
            "the states that were asked, they are left empty. "
            "**Traditional religion is a quarter of Northern Bahr el Ghazal** (23.4%) and a tenth "
            "of Eastern Equatoria, and was not recorded at all in Western or Central Equatoria. "
            "Across the six states it is 6.7%, close to the 7% a 2013 national poll found and far "
            "below the 32.8% \"other religions\" the Pew Research Center estimates for the country. A box offered beside "
            "Christianity is a floor for traditional practice, as everywhere on this map. "
            "**Islam is 10.5% of Western Bahr el Ghazal**, a fifth of its towns, Wau and Raja "
            "among them, and under 2% in every other state. "
            "**In Eastern Equatoria 12.8% of people are in households whose head's religion was "
            "not recorded**, almost all of them in the countryside; they are not drawn."),
        how="household survey, one round in 2015",
        grain="former states, 1.1 million people on average",
        gap=("51.5% of residents, mostly the 6.4 million in Jonglei, Unity, Upper Nile and Warrap, "
             "which the survey missed or sampled only in towns, and about 535,000 refugees from "
             "abroad"),
        # (6,411,144 unsampled + 171,087 not recorded + 535,471 refugees) over (13,297,196 + 535,471):
        # refugees taken as outside the 2025 estimate, which adjusts for internal displacement and
        # returns; sources/ss.md §5.
        gap_share=0.5146,
        counts=_ss_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ss" / "ss_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ss_place_weight,
        note="SURVEY: World Bank Microdata Library catalog 2778 (SSD_2015_HFS-W1_v02_M), hhq C.9, "
             "multi-select, head's religion; Anita's download under ask 042, files stay in "
             "data/raw/ss. Person weight = weight x hhsize (hhsize equals the hhm roster in all "
             "3,550 households). 32 two-answer heads split equally. WAVE 2 (catalog 2777) is towns "
             "only and is a witness, not pooled: its towns against wave 1's, Islam r +0.992, "
             "traditional r +0.973 over the six states. SPLIT-HALF: lits.stability on 300 EAs "
             "(median of 400 halves, 400-draw EA regrouping null): Christianity, traditional, Not "
             "recorded and Islam p 0.0025, Atheism 0.005, Buddhism 0.027 carry; Judaism and "
             "Agnostic flat. NOT DRAWN: Jonglei, Unity, Upper Nile (never sampled) and Warrap "
             "(towns only; wave 1's town/country gap on traditional is 7.2 vs 24.8 in Northern "
             "Bahr el Ghazal). POPULATION: OCHA's 2025 county estimates (13,297,196 without Abyei), "
             "not COD-PS 2022; Kontur calibrated per county (sources/ss_geo.py).",
    ),
}
