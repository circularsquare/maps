# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ke_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ke_hexes.gpkg has no `pop` column — run sources/ke_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _KeHexWeighter(place)


def _ke_counts():
    """KNBS 2019 KPHC Volume IV Table 2.30 at county: 11 drawn categories on 47 counties.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    47 UNITS FOR 47.2M PEOPLE IS THE COARSEST COUNTING GEOGRAPHY ON THIS MAP, coarser than
    the Philippines' 117, and it is KNBS's ceiling rather than a choice made here: every
    other table in Volume IV is published "by County and Sub-County" and religion is the
    one that stops at county. sources/ke.md §2. The dots are then spread across 230,139
    Kontur hexagons by population, so they land where Kenyans live — but nothing measures
    which part of a county a given church's members are in.

    THREE CATEGORIES RESOLVE TO NOTHING: the universe total, `Don't Know` (73,253) and
    `Not Stated` (6,909). The last two are non-answers per §3.5, so the drawn population is
    47,133,120 — 99.83% of the table's universe, itself 99.26% of the census count. The
    people the census never asked are in hotels, hospitals, prisons, children's homes,
    travelling or sleeping outdoors, and the table says so in its own footnote.
    """
    from ke2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ke.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "county"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ke" / "ke_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ke.csv counties with no polygon: {missing} -- re-run "
                         "sources/ke_geo.py, the lookup is stale")
    if df["unit"].nunique() != 47:
        raise SystemExit(f"{df['unit'].nunique()} counties, expected 47")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ke": dict(
        name="Kenya",
        source="2019 Kenya Population and Housing Census, Volume IV (KNBS)",
        basis="self-identification, conventional household population",
        view=[33.8, -4.8, 42.1, 5.6],
        gap=("0.2%, mostly a respondent who did not know the religion of the person they were "
             "answering for"),
        gap_share=0.001698,
        note_public=(
            "**The deepest religion question in Africa, on the coarsest geography this map "
            "draws.** KNBS offers thirteen answers where Ghana offers nine and most of the "
            "continent offers none, and two of them — *Evangelical Churches* and *African "
            "Instituted Churches* — are counted by no other census anywhere on this map. "
            "It also gives Hindus, the Orthodox and traditional religion cells of their own "
            "instead of burying them in a residual. The price is 47 counties for 47.2 "
            "million people, about a million each. "
            "**The Kenyan split of Christianity is not the usual one, and reading it as the "
            "usual one will mislead you.** *Protestant* here means the mainline mission "
            "inheritance — the Anglican Church of Kenya, the Presbyterian Church of East "
            "Africa, the Methodists — while *Evangelical Churches* is a peer category, not "
            "a subset, holding the Africa Inland Church, the Baptists and the Pentecostal "
            "Assemblies of God. An Anglican here is a Protestant; a Baptist here is an "
            "Evangelical. Together they are 53.9% of the country. "
            "**African Instituted Churches are 3.29 million people and they have a "
            "homeland.** These are the churches founded in Africa by Africans outside the "
            "missions — in Kenya the Legio Maria, the Nomiya Luo Church, the African Israel "
            "Nineveh Church, the Akorino — and they are overwhelmingly a Luo and western "
            "Kenyan phenomenon: Siaya is 23.9%, Kisumu 18.2%, Homa Bay 17.8%, against a "
            "national 7.0%. Almost nowhere else on earth counts these churches at all, so "
            "this is one of the few places the map can show them. "
            "**The Muslim north-east is as near-total as anything drawn here.** Mandera is "
            "99.4% Muslim, Wajir 99.0%, Garissa 97.6% — the sharpest such block outside "
            "Sulu — and the coast runs high behind it. Against that, Kenya's Hindus are "
            "60,287 people and effectively two cities: Nairobi holds 38,141 of them. "
            "**Two smaller things worth finding.** Traditional religion survives among the "
            "northern pastoralists and almost nowhere else — Marsabit 15.5%, Samburu 9.9% "
            "against a national 0.68% — and **Kilifi is 10.2% no-religion**, two and a half "
            "times the next county and a genuine outlier on the Mijikenda coast."),
        how="census, 2019",
        grain="counties, 1.0m people on average",
        counts=_ke_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ke" / "ke_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ke_place_weight,
        note="THE COUNTS ARE COARSE AND THE PLACEMENT IS FINE, and the two must not be "
             "confused. KNBS publishes religion at county and nowhere below it — 47 units "
             "for 47.2M people, the coarsest counting geography on this map. That is the "
             "office's ceiling and not a choice made here: every other table in Volume IV "
             "is published 'by County and Sub-County' and religion is the one that stops. "
             "The census collects to the enumeration area, so the finer data exists and is "
             "not released; IPUMS's 2019 sample reaches division and is the upgrade path "
             "(sources.md §10a). "
             "The dots are spread across 230,139 Kontur 400m hexagons weighted by hex "
             "population (sources/ke_grid.py) — the first country on this map to use "
             "Kontur, and Kenya needs it: Turkana is 68,680 km² and Marsabit 70,961 km², "
             "so an equal share per polygon would have washed the empty north in dots of "
             "one colour. Read a cluster as 'this county, drawn where Kenyans live'. "
             "The universe is the conventional household population: 47,213,282 against a "
             "census 47,564,296, the difference being people in hotels, hospitals, prisons "
             "and children's homes, travellers and outdoor sleepers, who were never asked "
             "(the table's own footnote). `Don't Know` (73,253) and `Not Stated` (6,909) "
             "are non-answers and off the tree per §3.5, so 99.83% of the table is drawn.",
    ),
}
