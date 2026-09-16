# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403

_TK_ATOLLS = 3
_TK_PRESENT = 1_197


def _tk_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Each atoll's people live in one village (two on Fakaofo) on small islets; the populated hexes
    sit on the villages, and the rest of each atoll is lagoon and reef (sources/tk_geo.py).
    """
    return _kontur_place_weight(place, "tk_hexes.gpkg", "sources/tk_geo.py")


def _tk_counts():
    """Tokelau 2016 census at atoll: Table 5.8 in counts, every row `measured`.

    The universe is the 1,197 usual residents present on census night. The 9 not stated are read
    and not drawn, and with the 302 usual residents who were overseas they are `gap`
    (sources/tk.py).
    """
    from tk2016 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tk.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != _TK_ATOLLS or int(df["count"].sum()) != _TK_PRESENT:
        raise SystemExit(f"tk.csv has {df['geo_id'].nunique()} atolls and {int(df['count'].sum()):,} "
                         f"people, expected {_TK_ATOLLS} and {_TK_PRESENT:,} -- re-run sources/tk.py")
    df = df[~df["source_category"].isin(EXCLUDED)].copy()
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"tk.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "tk": dict(
        name="Tokelau",
        source="2016 Tokelau Census of Population and Dwellings, Table 5.8 (Tokelau National "
               "Statistics Office and Stats NZ)",
        basis="self-identification, usual residents present on census night",
        note_public=(
            "**Tokelau's 2016 census counts religion on each of its three atolls.** The form named "
            "three churches, Congregational Christian, Roman Catholic and Presbyterian, and took any "
            "other answer in writing. The Congregational Christian Church is **77.0%** of Atafu and "
            "62.7% of Fakaofo, while Nukunonu is **81.8%** Catholic. Atafu counted 54 Presbyterians, "
            "against 5 in 2011. "
            "**The census asked only the people who were in Tokelau on census night.** That was "
            "1,197 of its 1,499 usual residents. The other 302 were overseas, 48 of them Tokelau "
            "Public Service staff and their families based in Apia, and the census recorded no "
            "religion for them."),
        how="census, 2016",
        grain="atolls, 400 people on average",
        gap="302 usual residents overseas on census night, who were not asked, and 9 not stated; "
            "20.7% of the 1,499 usual residents",
        gap_share=0.2075,
        counts=_tk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tk" / "tk_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tk_place_weight,
        view=[-172.75, -9.60, -171.00, -8.35],
        note="2016 TOKELAU CENSUS TABLE 5.8 (social profile workbook, pinned) is religion x the 3 "
             "atolls in persons for the 1,197 usual residents present on census night. Checked: "
             "rows and atolls close in 2011 and 2016; de jure minus absentees (demography 1.3.1, "
             "1.3.2) equals each atoll total; the profile report's Table 4.1, p.15 and p.28 (the "
             "form names three churches and an other write-in); UNSD's 2011 row equal, its 2016 "
             "row 4 people off in three cells (pinned). Gap is the 302 usual residents overseas "
             "(254 absentees, 48 TPS in Apia) plus 9 not stated, of 1,499 de jure. Units are OSM's "
             "atoll areas (Kontur Boundaries TK), witnessed by the report's printed distances; "
             "placed on Kontur hexes. Congregational Christian is a new node beside Samoa's cccs "
             "(independent since the mid-1990s). sources/tk.md has the record.",
    ),
}
