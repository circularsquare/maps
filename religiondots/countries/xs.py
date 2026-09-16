# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
#
# NOT A COUNTRY. `territory=False`: people the map draws as part of no country, with no outline or
# wash, never chosen by the viewer's Auto (countries.py field docstring). Anita, 2026-09-15.
from countries._shared import *  # noqa: F401,F403


def _xs_place_weight(place):
    """countries.py hook. `place` is sources/xs_geo.py's layer: each CBS unit cut by the Kontur
    hexes it overlaps, weighted by the Kontur population in the overlap, which is the same weight
    sources/ps_geo.py took off Palestine's placement for these people."""
    return _kontur_place_weight(place, "xs_places.gpkg", "sources/xs_geo.py")


def _xs_counts():
    """Israeli settlements in the West Bank: CBS 2022 Jews and Others on 267 units beyond the Green
    Line, 723,899 people.

    **NOBODY HERE IS ON ISRAEL'S OR PALESTINE'S ENTRY, AND THAT IS ASSERTED.** Every unit is one
    `data/geo/il/dropped_units.json` lists, which `_il_counts` excludes; the groups are Jews and
    the register's Others only, and Palestine's census counts Palestinians only. The Muslims and
    Christians in the same units, all but 128 of them in East Jerusalem, stay on Palestine's entry
    (sources/xs.py, taxonomy/xs2022.py).

    Two geo_levels, as for Israel: statistical areas and the localities CBS publishes whole. The
    observance rows are `derived`, as on Israel's entry, and cannot ring.
    """
    from xs2022 import EXCLUDED, _key, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "xs.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(("statarea", "locality"))].copy()
    df["count"] = df["count"].astype(float)

    dropped_path = HERE / "data" / "geo" / "il" / "dropped_units.json"
    if not dropped_path.exists():
        raise SystemExit("xs: data/geo/il/dropped_units.json is missing -- run sources/il_geo.py")
    with open(dropped_path, encoding="utf-8") as fh:
        dropped = set(json.load(fh))
    inside = sorted(set(df["geo_id"]) - dropped)
    if inside:
        raise SystemExit(f"xs.csv carries {len(inside)} units Israel's entry draws: {inside[:6]} "
                         "-- re-run sources/xs.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(c for c in set(df.loc[df["node"].isna(), "source_category"])
                      if _key(c) not in EXCLUDED)
    if unmapped:
        raise SystemExit(f"xs.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = df["tier"] == "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "xs": dict(
        name="Israeli settlements in the West Bank",
        source="2022 Census of Population and Housing (CBS), units beyond the 1949 armistice line",
        basis="population register",
        territory=False,
        note_public=(
            "**These are the Israelis living beyond the 1949 armistice line, in the West Bank "
            "settlements and in East Jerusalem.** Israel's 2022 census counted **723,899** people "
            "there on the population register as Jews or with no religion recorded, 237,240 of "
            "them in East Jerusalem. Israel's entry on this map stops at the armistice line and "
            "Palestine's census does not count them, so they are drawn here, as part of neither "
            "country. The entry has no territory of its own, so the map does not switch to it "
            "from where you are looking; it is chosen from the list. "
            "**The religion is read off Israel's population register, not asked.** The register "
            "records what the state has on file and has no box for having no religion, as on "
            "Israel's entry. The 28,251 people it holds with no religious classification are "
            "drawn grey. "
            "**Haredi households are the largest group of Jews here.** Where an area is at least "
            "85% Jewish, CBS's household-lifestyle table splits the Jewish count: **34.5%** of "
            "the Jews live in ultra-religious households, nine in ten of them in Modi'in Illit, "
            "Beitar Illit and East Jerusalem, against 29.9% religious, 12.6% secular and 10.6% "
            "traditional. "
            "**Muslims and Christians in the same areas are not drawn here.** CBS counts 373,257 "
            "of them, all but 128 in East Jerusalem, and Palestine's census counts East "
            "Jerusalem's Palestinians, so they are on Palestine's entry. The register does not "
            "say which Christians are Arab, so the non-Arab Christians who live in these areas "
            "are on neither. "
            "**The dots follow where people live inside each area.** They are weighted by "
            "Kontur's 2023 population grid, the same weight taken out of Palestine's entry so "
            "that Palestinian dots mostly do not land in the settlements. CBS maps 119 small "
            "localities as placeholder points, and those are drawn as discs sized to their "
            "population."),
        how="population register, not a census question",
        fill="from each area's own household-lifestyle table",
        grain="statistical areas and localities, about 2,700 people each",
        gap="373,257 Muslims and Christians in the same areas, nearly all in East Jerusalem and "
            "left to Palestine's entry, except the non-Arab Christians among them (at most "
            "6,275), who are on neither entry",
        counts=_xs_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "xs" / "xs_places.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_xs_place_weight,
        note="ITS OWN ENTRY BY ANITA'S RULING ON ASK 028 (2026-09-15). Israel's entry stops at the "
             "Green Line (sources/il.md §7) and PCBS does not count the settlers, so they are part "
             "of neither country, with territory=False: no shape in country_shapes.py and never "
             "Auto's pick in the viewer. COUNTS are il.csv's rows for the 267 units in "
             "data/geo/il/dropped_units.json, Jews (with the observance rows) and Others only, "
             "723,899 people, pinned in sources/xs.py; the 373,257 Muslims and Christians in the "
             "same units stay on Palestine's entry, whose census counts East Jerusalem's "
             "Palestinians. MAPPING is il2022 restricted to those groups (taxonomy/xs2022.py). "
             "PLACEMENT is sources/xs_geo.py: ps_geo.settlement_units() (119 placeholders as "
             "discs) cut by the Kontur hexes ps_geo took these people off, weighted by Kontur "
             "population in the overlap; it reproduces ps_lookup.csv's removal per governorate. "
             "Needs the Israel and Palestine builds on disk. sources/xs.md has the record.",
    ),
}
