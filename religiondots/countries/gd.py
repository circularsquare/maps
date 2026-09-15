# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gd_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "gd_hexes.gpkg", "sources/gd_grid.py")


def _gd_counts():
    """CSO 2021 census at parish: 25 nodes on 7 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`geo_level == "parish"` IS LOAD-BEARING, because gd.csv carries two tiers.** The
    census publishes 8 units — it reports the **Town of St. George** apart from the rest of
    the parish — and no boundary set anywhere publishes the town, so `sources/gd.py` writes
    the census's own 8 at `census_unit` and the 7 drawable ones at `parish`. Reading the
    wrong one would double-count St. George. Guarded below.

    **92.89% of the drawn universe is on the map** — 100,581 of 108,279 — the missing part
    being `NOT STATED`, 7,698 people, excluded by taxonomy/gd2021.py per §3.5. That is one
    of the largest non-answers here, and **its geography is the thing the fold hides**: the
    Town of St. George refused at 15.8% against 9.9% for the rest of its parish and 0.91%
    in St. Mark.

    **AND THE UNIVERSE IS ALMOST THE WHOLE COUNTRY**, which is unusual for this region:
    108,279 of a census 109,021, so 99.3% of Grenada is inside the table before the refusals
    come out. Barbados draws on 81.4% of its own estimate and Cayman on 96.3%.
    """
    from gd2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gd.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "parish"].copy()
    if df["geo_id"].nunique() != 7:
        raise SystemExit(f"{df['geo_id'].nunique()} parishes, expected 7 -- gd.csv also "
                         "holds the census's own 8-unit tier at geo_level=census_unit, "
                         "which must NOT be drawn; re-run sources/gd.py")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"NOT STATED", "TOTAL"})
    if unmapped:
        raise SystemExit(f"gd.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "gd": dict(
        name="Grenada",
        source="2021 Housing and Population Census, preliminary results "
               "(Central Statistical Office)",
        basis="self-identification",
        view=[-61.82, 11.97, -61.36, 12.55],
        gap=("7.1% who did not state a religion, running from 0.9% in St. Mark to 15.8% in the "
             "town of St. George"),
        gap_share=0.07109,
        note_public=(
            "**The most evenly divided country in the Caribbean, on this map.** Grenada's "
            "largest religion is **31.5%** — against Saint Lucia's 50.6% Catholic 150 km "
            "north, the Bahamas' 34.9% Baptist and Barbados's 23.9% Anglican — and four "
            "bodies hold more than 7% each, no two of them the same tradition: Roman "
            "Catholic 31.5%, Pentecostal 19.9%, Seventh Day Adventist 12.3%, Anglican "
            "7.3%. The French and the British each left a church behind, and the "
            "twentieth-century missions landed on top of the pair. "
            "**The Adventists have the north and the Pentecostals the south, and they "
            "barely overlap.** Seventh Day Adventists are **24.1% of St. Andrew and 22.6% "
            "of St. Mark** against **7.3% of St. George**; Pentecostals are **25.9% of "
            "St. David and 23.0% of St. Andrew** against **7.2% of Carriacou**. Both are "
            "national churches with regional hearts. "
            "**Carriacou is a different island in this sense too.** **22.1% Anglican** — "
            "seven times St. David's 3.0% — and 40.6% Roman Catholic, with Pentecostals at "
            "a third of their national share. Three centuries of Scottish and English "
            "settlement in the Grenadines, still legible, and the sharpest single-unit "
            "signal in the country. "
            "**This census names twenty-five answers**, one of the deepest lists in the "
            "region: `Spiritual Baptist`, `Mennonite`, `Lutheran`, `Moravian`, "
            "`Presbyterian`, `Independent Baptiste`, `Evangelical`, `Church of God`, "
            "`Buddhist`, `Bahai`, `Hindu`, `Muslim` and `Rastafarian` all have cells of "
            "their own. Several are one parish each: **Presbyterians are 3.2% of St. Mark** "
            "and 0.02% of Carriacou, the old Scottish mission on the west coast; **Church "
            "of God is 8.5% of St. Andrew** and 0.7% of St. Mark; **Spiritual Baptists are "
            "4.8% of St. Mark**, the Afro-Caribbean tradition that Trinidad banned by "
            "ordinance from 1917 to 1951. "
            "**Grenada also splits disbelief from non-affiliation** — 5.95% report no "
            "religious affiliation and **0.05% report atheism**, a 130-fold gap, the widest "
            "on this map from a census that offered both boxes. Saint Lucia's form asks the "
            "same pair and gets the same answer. "
            "**7.1% did not answer**, and where they are is the most striking thing the map "
            "cannot show: the capital. See the note below."),
        how="census, 2021",
        grain="parishes, 15,500 people on average",
        counts=_gd_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gd" / "gd_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gd_place_weight,
        note="**THE COUNTING TIER IS 8 UNITS AND THE MAP DRAWS 7, BECAUSE NOBODY PUBLISHES "
             "A BOUNDARY FOR THE CAPITAL.** The census reports the **Town of St. George** "
             "— 2,681 people — apart from the rest of the parish, and no boundary set has "
             "it: COD-AB's ADM1 is the six parishes plus Carriacou and Petite Martinique, "
             "and OpenStreetMap has the six parishes at `admin_level=6` and, for the town, "
             "only a `place=town` **node**. So the two halves are folded back into one "
             "St. George. **What that hides is this country's sharpest number**: the town "
             "declined the religion question at **15.8%** against 9.9% for the rest of the "
             "parish and 0.91% in St. Mark. `gd.csv` carries BOTH tiers — the census's own "
             "8 units and the 7 drawn ones — so nothing is lost from the record and the "
             "split is already there if a town boundary ever appears. "
             "**CARRIACOU AND PETITE MARTINIQUE ARE ONE UNIT BECAUSE THE CENSUS MAKES THEM "
             "ONE.** COD-AB gives them separate polygons and the census publishes a single "
             "figure, so the polygons are dissolved; splitting one published number between "
             "two islands would be inventing a magnitude (§14.4). After both moves the "
             "tiers agree exactly, 7 on 7, which is also what OSM independently has. "
             "**THE TABLE RECONCILES TO THE PERSON**, in both directions and on every row — "
             "which Saint Lucia's does not and Cayman's does not. **And the universe is "
             "almost the whole country**: 108,279 of a census 109,021, the difference being "
             "690 people in institutions and 52 homeless. 99.3% of Grenada is inside the "
             "table before the refusals come out, against 96.3% in Cayman and 81.4% in "
             "Barbados. "
             "**`NOT STATED` IS 7.1% AND IS NOT AN UNDERCOUNT.** These 7,698 people were "
             "counted and declined the question; they are marked, not filled (§3.5). "
             "**THE REPORT'S TEXT LAYER SUBSTITUTES `Ǫ` FOR `Q`** — `MARTINIǪUE` — which "
             "is invisible on the page and breaks any comparison against a typed name. "
             "`sources/gd.py` folds it back. `MORMOM` and `BAPTISTE` are CSO's own "
             "spellings and are left alone in the data. "
             "**`CHURCH OF GOD` IS FILED AT THE HOLINESS PARENT**, 3.6% of the country, "
             "because the name cannot decide between the Cleveland (Pentecostal) and "
             "Anderson (Holiness) lines and — unlike Cayman — no external evidence names "
             "which body Grenada means. **`SPIRITUAL BAPTIST` IS NOT FILED WITH THE "
             "BAPTISTS**: CSO offers it and `Independent Baptiste` as two separate answers, "
             "so merging them would undo a distinction the source drew. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 509 hexes, and it does two different "
             "jobs here — St. George holds 41% of Grenada on 65.8 km², nearly all of it "
             "between St. George's town and Point Salines; and Carriacou and Petite "
             "Martinique are one unit on two islands 2.4 km apart, where scattering by area "
             "would put far too many people on the smaller one.",
    ),
}
