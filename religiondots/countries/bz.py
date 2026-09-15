# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bz_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "bz_hexes.gpkg", "sources/bz_grid.py")


def _bz_counts():
    """SIB 2022 census at district: 11 nodes on 6 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`keep_default_na=False` IS LOAD-BEARING AND IS NOT A STYLE CHOICE.** Belize's second
    largest religion answer is the literal string `None`, 123,373 people and 31.04% of the
    country. A bare `pd.read_csv` turns those six rows into NaN, `resolve()` never sees them,
    and the country silently loses **the largest non-Catholic category on its map** — with
    every reconciliation in `sources/bz.py` still passing, because that file checks the
    workbook and not this read. Measured: the default read drops exactly 6 rows and
    123,372.67 people. Guarded below by asserting the category survives the parse.

    **THE COUNTS ARE FLOATS.** SIB publishes undercount-adjusted census figures, so the
    national total is 397,483.456 rather than an integer (`sources/bz.py`). Nothing rounds
    them here; the dot allocator takes fractional counts already.

    **98.96% of Belize is drawn** — 393,348.7 of 397,483.5. What is not is `Don't Know/Not
    Stated`, 4,135 people, which taxonomy/bz2022.py excludes per §3.5.
    """
    from bz2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "district"].copy()
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 6 -- re-run "
                         "sources/bz.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("bz.csv has no `None` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here; without it 31% of "
                         "Belize disappears and every other check still passes.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Don't Know/Not Stated", "Total"})
    if unmapped:
        raise SystemExit(f"bz.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "bz": dict(
        name="Belize",
        source="Census 2022 (Statistical Institute of Belize)",
        basis="self-identification",
        view=[-89.30, 15.80, -87.35, 18.55],
        gap="1.0%, whom SIB pools into a single do not know or not stated cell",
        gap_share=0.0104,
        note_public=(
            "**The only census on this map that counts Mennonites.** 15,440 people, 3.9% of "
            "Belize, and they are where the history puts them: **9.9% of Orange Walk and "
            "8.9% of Corozal** against 0.5% of Belize District. The Kleine Gemeinde and Old "
            "Colony communities arrived from Mexico and Canada in 1958 on an agreement that "
            "granted exemption from military service and control of their own schools, and "
            "they farm the north. Nothing else here has ever filled that colour outside the "
            "United States. "
            "**And 31.0% of Belize reports no religion — the highest share this map draws "
            "anywhere in the Americas**, above Jamaica's 21.4%. It is very unevenly spread: "
            "**46.6% in Stann Creek** against 21.7% in Toledo. That is a large rise on 2010 "
            "and the census offers no explanation for it; it is drawn as published. "
            "**Belize is Catholic and Pentecostal along a north-south line.** Catholicism is "
            "37.9% in Corozal and 37.3% in Orange Walk — the Spanish-speaking Mestizo north "
            "— and falls to 26.1% in Stann Creek. Pentecostalism runs the other way, 14.9% "
            "in Cayo and 13.3% in Toledo. Baptists are **12.0% of Toledo**, the Maya south, "
            "against 0.9% of Orange Walk. "
            "**Six districts is coarse and the country is small, so read this as six "
            "readings rather than a map of Belize.** The census publishes religion at "
            "district and nowhere else, though it publishes population down to village. "
            "**What the question does not ask is as important as what it does.** SIB names "
            "nine Christian bodies and pools everything else into `Other` (6.3%). There is "
            "no cell for Hinduism, none for Islam, and none for any indigenous or "
            "Afro-Caribbean tradition, in a country that has all of them. Belize's Hindus "
            "and Muslims, its Bahá'ís, its Rastafari, Maya traditional practice in Toledo "
            "and the Garifuna *dugu* of the Stann Creek coast are either inside that grey "
            "residual or invisible inside a Christian colour, and this source cannot say "
            "which."),
        how="census, 2022",
        grain="districts, 66,000 people on average",
        counts=_bz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bz" / "bz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bz_place_weight,
        note="**THE `None` TRAP, AND IT IS THE MOST DANGEROUS THING IN THIS COUNTRY.** "
             "Belize's second largest answer is the literal string `None` — 123,373 people, "
             "31.04%. `pd.read_csv` converts it to NaN by default, which drops exactly six "
             "rows and 123,372.67 people, and **every other check in the pipeline still "
             "passes** because `sources/bz.py` reconciles the workbook rather than this "
             "read. `_bz_counts` passes `keep_default_na=False` and then asserts the "
             "category survived, because a silent 31% loss is not something a reviewer would "
             "see on the map — Belize would simply look devout. "
             "**DISTRICT ORDER IS THE SECOND TRAP.** SIB prints its six districts north to "
             "south (Corozal first) and COD-AB codes them alphabetically (`BZ01` is Belize "
             "District), so numbering the table by position — the way `sources/mw.py` "
             "legitimately does — would mismatch five of six units while every total still "
             "reconciled. `sources/bz.py` carries the pcode against the name and "
             "`sources/bz_geo.py` asserts the same pairing from the boundary side. The join "
             "is then 6/6 both ways with no name variants at all, which is a first here. "
             "**THE FIGURES ARE FRACTIONAL AND THAT IS THE SOURCE.** SIB publishes "
             "undercount-adjusted counts throughout — the national total is 397,483.456 — "
             "so every identity in `sources/bz.py` is asserted to a 1e-6 relative tolerance "
             "rather than to zero. The 2010 column in the same workbook is fractional too, "
             "so this is SIB's standing practice and not a one-off. "
             "**MALE + FEMALE == TOTAL IS THE CHECK ON THE READ.** The sheet lays every "
             "group out as three columns and only Total is drawn, but all three are read: "
             "the sex columns are the only check that would catch a district's block landing "
             "one column-group left or right, since every other identity reconciles inside "
             "one group whichever columns were taken. 91 cells, zero failures. Zimbabwe's "
             "panel rule (§9aj) applied to a workbook. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 5,095 hexes. Six districts over 22,966 "
             "km² averages 3,828 km² — five times Jamaica's — and Cayo holds the Chiquibul "
             "and the Maya Mountains while Toledo is mostly rainforest, so uniform scatter "
             "would draw a large part of Belize into empty bush. The per-district "
             "Kontur/census ratio runs **0.80x to 1.10x**, tight for an 18-month vintage "
             "gap. 6.8% of the grid's people sit outside every district — the Mexican and "
             "Guatemalan border overrun — and are dropped; Ambergris Caye is inside, which "
             "the 1.02x ratio for Belize District confirms.",
    ),
}
