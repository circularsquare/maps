# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ps_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Two Kontur extracts (the PS one has no hexes in the part of Jerusalem Israel annexed), with
    the Israeli settlements' population taken off the hexes they overlap, from the Israel build's
    own CBS 2022 units beyond the Green Line (sources/ps_geo.py). The West Bank governorates
    are large and mostly thinly settled: Jericho & Al-Aghwar is 593 km2 and 47,325 people, and
    Hebron's people are in the city and the villages along the ridge.
    """
    return _kontur_place_weight(place, "ps_hexes.gpkg", "sources/ps_geo.py")


def _ps_counts():
    """Palestine PCBS census 2017 at governorate: 3 nodes on 16 units, every row `measured`.

    Table 3 of the Preliminary Results in counts, which close on the West Bank, Gaza Strip and
    Palestine rows in every column (sources/ps.py). No rescale: the table is counts.
    """
    from ps2017 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ps.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 16:
        raise SystemExit(f"{df['geo_id'].nunique()} governorates in ps.csv, expected 16 -- "
                         "re-run sources/ps.py")
    df["node"] = df["source_category"].map(resolve)
    known = {"Not Stated", "Total"}
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - known)
    if unmapped:
        raise SystemExit(f"ps.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "ps": dict(
        name="Palestine",
        source="Population, Housing and Establishments Census 2017, Preliminary Results "
               "(Palestinian Central Bureau of Statistics), Table 3",
        basis="self-identification, Palestinians counted",
        note_public=(
            "**Palestine's 2017 census published religion for its 16 governorates, in counts.** "
            "The form asked it of Palestinians only, with three answers: Muslim, Christian and "
            "other. Islam is **98.9%** of the 4,665,426 Palestinians counted and Christianity "
            "1.0%. "
            "**Half of the Christians are in Bethlehem governorate.** It has 23,165, **10.9%** "
            "of its people, followed by Ramallah and Al-Bireh with 10,255 and Jerusalem with "
            "8,558. The Gaza Strip has 1,138. The census does not separate the churches, so "
            "Christianity is one colour. "
            "**Jerusalem governorate includes East Jerusalem.** The census counts the part of "
            "the governorate Israel annexed in 1967 together with the rest, and Israel's entry "
            "on this map leaves out East Jerusalem and the West Bank, so each place is drawn "
            "once. "
            "**Israelis living in the settlements are not drawn.** The Palestinian census does "
            "not count them and Israel's entry stops at the 1949 armistice line, so the "
            "settlements in the West Bank and East Jerusalem, where Israel's 2022 census counted "
            "about **720,000** people, are on neither. "
            "**These are 2017 figures.** Since October 2023 the war has displaced most of the "
            "Gaza Strip's population, so the dots there show where people lived when the census "
            "was taken, not where they live now."),
        how="census, 2017",
        grain="governorates, 292,000 people on average",
        gap="0.89%: the 40,175 people counted who are not Palestinian, whom the form did not "
            "ask about religion, and 1,509 who gave no answer; and the Israeli settlements, "
            "which this census does not count",
        gap_share=0.0089,
        counts=_ps_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ps" / "ps_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ps_place_weight,
        note="TABLE 3 OF PCBS's 2017 PRELIMINARY RESULTS (book2364-1.pdf, p.35) is governorate x "
             "Islam, Christian, Other, Not Stated in COUNTS over Palestinians counted "
             "(4,665,426), closing on the West Bank, Gaza Strip and Palestine rows in all five "
             "columns. The form (25 PHC) asks religion in its `For Palestinians only` block with "
             "three codes and no non-response code. THREE NESTED TOTALS: Table 3 Palestinians "
             "counted <= Table 2 everyone counted (4,705,601) <= Table 25 counted plus the "
             "post-enumeration estimate (4,780,978); Jerusalem 392,835 <= 414,786 <= 435,483, "
             "and J2 alone is 154,320 in Table 25, so Table 3's Jerusalem includes J1. "
             "BOUNDARIES are COD-AB cod-ab-pse v01 admin 2, the same dataset whose admin 0 "
             "sources/il_geo.py cut Israel on, so the two entries meet on one line. PLACEMENT is "
             "Kontur PS plus IL (the PS extract has no hexes in J1) with CBS 2022 Jews and "
             "Others beyond the Green Line (723,899, from il.csv) taken off the hexes they "
             "overlap; 477,068 came off, and Ramallah & Al-Bireh still reads 1.57x Table 2 and "
             "Jericho 1.78x. Needs the Israel build on disk. sources/ps.md has the record.",
    ),
}
