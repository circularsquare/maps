# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _MuHexWeighter(_KeHexWeighter):
    """Split a unit's dots across Kontur 400 m hexagons by hex POPULATION.

    Mauritius needs this least of the four countries that use it — 182 units averaging
    11 km² and 6,800 people, so an equal share per polygon is already close. Two things
    still go wrong without it: the coastal VCAs own their LAGOON, because the boundaries run
    out to the reef, and the few big rural units (Grande Rivière Noire 43.5 km², Tamarin
    48.0 km²) are mostly gorge and cane with their people along one road.

    Kontur is coarse relative to this country — 2,072 hexes for 182 units — and three
    cross-district slivers of 0.09 to 3.6 km² contain no hex centroid at all. They hold 890
    people between them, 0.07%, and fall back to an equal share inside their own polygon,
    which at that size is not an approximation of anything.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares in the three sliver units that "
                f"contain no hex (sources/mu_grid.py)")


def _mu_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! mu_hexes.gpkg has no `pop` column — run sources/mu_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _MuHexWeighter(place)


def _mu_counts():
    """Statistics Mauritius 2022 HPC Table D6: 13 drawn categories on 182 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    182 UNITS FOR 1,233,097 PEOPLE IS ~6,800 EACH, the finest counting geography on this map
    after Sri Lanka's GN divisions, the German grid and the UK's output areas, and finer than
    all of them relative to the size of the country. The units are Municipal Council Wards
    and Village Council Areas, which is the tier Statistics Mauritius calls `R`.

    183 CENSUS ROWS BECOME 182 DRAWN UNITS. OpenStreetMap has no polygon for Vacoas-Phoenix
    Ward 5 or Ward 6-West, so those two rows share one unit built from the remainder of the
    town — 35,664 people, 2.89%, and the cost is one internal boundary inside one town.
    `mu_lookup.csv` maps both geo_ids to it and the rows are summed here.

    ONE CATEGORY RESOLVES TO NOTHING and it is the universe row, so the drawn population is
    the whole table: **1,233,097, the entire resident population the census enumerated.**
    `Other & Not stated` is drawn rather than dropped — Mauritius is the only source here
    that pools a non-answer into a residual with no split at any geography, so §3.5's usual
    move is unavailable and the cell is marked instead. See taxonomy/mu2022.py.
    """
    from mu2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mu.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "unit"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mu" / "mu_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mu.csv units with no polygon: {missing} -- re-run "
                         "sources/mu_geo.py, the lookup is stale")
    if df["unit"].nunique() != 182:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 182")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # Two census rows share the Vacoas remainder unit, so (unit, node) is not unique until
    # they are summed. Every other country's rows are already one per pair.
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "mu": dict(
        name="Mauritius",
        source="2022 Housing and Population Census, Volume II Table D6 (Statistics Mauritius)",
        basis="self-identification, religion as reported by the respondent",
        # THE VIEW IS THE MAIN ISLAND ONLY, AND RODRIGUES IS DELIBERATELY OUT OF IT.
        # Rodrigues is 600 km east and the main island is 45 km across, so a box holding both
        # is 13:1 mostly-ocean: it shrinks Mauritius to a speck in the corner and puts
        # Rodrigues behind the legend panel. Rodrigues is drawn and is one pan away, and
        # note_public says what is on it so nobody has to find it by accident.
        view=[57.25, -20.55, 57.85, -19.95],
        note_public=(
            "**Mauritius is the only place on this map where Hinduism is counted as more "
            "than one thing.** Statistics Mauritius asks about religion and takes five "
            "different Hindu answers — Marathi, Tamil, Telugu, Vedic/Arya Samaj, and "
            "everyone else — then publishes them at village level in a country that is "
            "47.9% Hindu. India's census does not do this. Guyana's, which is a quarter "
            "Hindu, does not. Four nodes on the religion tree exist because of this one "
            "table. "
            "**They are not the same kind of category, and the map is worth reading twice "
            "for it.** Marathi, Tamil and Telugu Hindus descend from indentured labourers "
            "out of three different parts of India and have kept separate temples, "
            "priesthoods and festival calendars for a century and a half — these are "
            "communities. **Arya Samaj is a movement**: Dayananda Saraswati's 1875 reform, "
            "Vedas alone and no image worship, which arrived in 1910 and split Mauritian "
            "Hinduism hard enough to shape its politics for decades. Anyone can join it and "
            "7,422 people have. "
            "**Each community has its own map and none of them is the one you would guess.** "
            "Marathi Hindus are the southwest coast and almost nowhere else — La Gaulette "
            "27.7%, Baie du Cap 27.0%, against 1.5% nationally. Tamil Hindus are southern "
            "and central, strongest in Savanne at 8.2%, and **thinner in Port Louis (3.7%) "
            "than in the country as a whole.** The Bhojpuri-descended majority is the cane "
            "belt: Camp Thorel is 95.4%. "
            "**Port Louis Ward 5 is 96.8% Muslim** — 17,058 people, and one of the most "
            "nearly total single-religion units drawn anywhere here. The city as a whole is "
            "40.9% against 18.2% nationally. "
            "**And Rodrigues is a different country — pan 600 km east to see it.** The six "
            "regions of that island run 84.9% to 91.9% Roman Catholic and 0.5% Hindu, "
            "against 24.9% and 38.5% on the main island: a Creole Catholic population inside "
            "a Hindu-majority republic, and the sharpest internal contrast any country on "
            "this map holds. It is outside the opening view because a box holding both "
            "islands is thirteen parts ocean and shows neither."),
        how="census, 2022",
        grain="wards and village councils, 6,800 people on average",
        counts=_mu_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mu" / "mu_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mu_place_weight,
        note="THE WHOLE RESIDENT POPULATION IS DRAWN — 1,233,097, every person the 2022 "
             "census enumerated. Only the universe row resolves to nothing. "
             "THE GEOGRAPHY IS THE FINEST PER HEAD ON THIS MAP AFTER SRI LANKA AND THE "
             "GERMAN GRID: 182 units over 1.23M people, about 6,800 each, on Municipal "
             "Council Wards and Village Council Areas. "
             "THE CATEGORY LIST IS THE PRICE. Table D5 of the same report names sixty-odd "
             "individual bodies — La Voix de la Delivrance, Peniel Tabernacle, Full Gospel "
             "Church, Christian Tamil, Church of England, Presbyterian, Methodist — but "
             "only at ISLAND level, three units. D6 has the geography and pools them into "
             "thirteen groups, of which `Other Christian` is 6.2%. §3.9's trade, made by "
             "the office, inside one publication. "
             "NO HUMANITARIAN SOURCE HAS THESE BOUNDARIES AND OPENSTREETMAP DOES. COD-AB "
             "Mauritius stops at 12 districts and geoBoundaries at ADM1; the drawn tier "
             "exists only in OSM, as 164 relations at admin_level=8 and 35 at 9. Every "
             "pairing was then checked SPATIALLY against the district the census printed it "
             "under — 182 of 182 — because D6 has no code column and a name join on 183 "
             "French place names is where a confident wrong pairing would live. It caught "
             "one: OSM and the census both split Rivière du Poste into East and West and "
             "they are not the same split, so that VCA is rebuilt and re-cut on the "
             "district line. "
             "TWO UNITS SHARE ONE POLYGON. OSM has no boundary for Vacoas-Phoenix Ward 5 or "
             "Ward 6-West, so both are drawn on the remainder of the town after its five "
             "mapped wards are removed — 35,664 people, 2.89%, and one internal boundary "
             "lost inside one town. "
             "`Other & Not stated` (6,931, 0.56%) IS DRAWN AND IS NOT A CLEAN CATEGORY. "
             "Mauritius is the only source here that pools a non-answer into a residual and "
             "publishes no split, so §3.5's usual move — take the non-answer off the tree — "
             "is unavailable. Read it as a ceiling on Mauritius's other religions rather "
             "than a count of them. "
             "AND `Buddhist/Chinese` (5,053) IS ONE CELL FOR TWO THINGS the tree keeps "
             "apart. D5 splits it nationally into Buddhist 2,178, Chinese 2,434 and Other "
             "Chinese 441; D6 does not, so it is drawn on `chinesefolk` as a syncretic "
             "whole per §3.3, and taxonomy/mu2022.py records what that costs.",
    ),
}
