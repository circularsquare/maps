# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _PkHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Pakistan's districts. Balochistan is why: 44% of the
    country's land, 6% of its people, 99.3% Muslim, and Chagai district alone is 44,748 km²
    holding 226,508 people. An equal share per polygon would wash the Makran and the Kharan
    desert — nearly half the map — in evenly spaced dots of one colour.

    A POPULATION weight, not a religion one. Nothing measures where Tharparkar's Hindus sit
    inside Tharparkar, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a district's hexes sum to zero "
                f"(sources/pk_geo.py)")


def _pk_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! pk_hexes.gpkg has no `pop` column — run sources/pk_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _PkHexWeighter(place)


def _pk_counts():
    """Pakistan 2017 census at district: 5 drawn nodes on 135 districts.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **SIX SOURCE CATEGORIES BECOME FIVE NODES**, because taxonomy/pk2017.py sends both
    `Hinduism` and `Scheduled Castes` to `hinduism`: the second is a caste category, not a
    religion, and PBS itself says the 2017 split between the two was poorly differentiated
    and was fixed in 2023. Together they are Pakistan's 4,444,870 Hindus.

    **THE TIER IS DISTRICT AND NOT TEHSIL, AND THAT IS §14.4.** pk.csv also carries the 585
    tehsil-level rows and they are deliberately not read. PBS publishes religion by
    district; its tehsil release has no religion table at all. sources/pk.md §3.

    TWENTY DISTRICTS HAVE NO DATA — Azad Kashmir's ten and Gilgit-Baltistan's ten, which
    PBS did not publish. They draw blank, and note_public says so.
    """
    from pk2017 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pk2017.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 135:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 135 -- re-run "
                         "sources/pk.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    # Hinduism and Scheduled Castes both land on `hinduism`, so the two rows for a district
    # must be added rather than left as duplicates -- scatter.py allocates per (unit, node).
    df = df.groupby(["unit", "node"], as_index=False).agg(
        {"count": "sum", "congregations": "sum"})
    return df[["unit", "node", "count", "congregations"]]


def _pk2023_counts():
    """Pakistan 2023 census at district: 7 drawn nodes on 136 districts (sources/pk.md §9).

    PBS's own Table 9, read by sources/pk_2023.py. ONE level, nothing allocated, nothing
    modelled: every row is `measured` and may ring.

    EIGHT SOURCE CATEGORIES BECOME SEVEN NODES. taxonomy/pk2023.py keeps pk2017.py's merge of
    `Scheduled Castes` into `hinduism` (a caste is not a religion), and adds `Sikh` and `Parsi`,
    the two cells the 2017 form did not have.

    THE TIER IS STILL DISTRICT. The 2023 state prints religion by tehsil too, and pk.csv
    carries those rows; Anita chose district on 2026-09-14 and they are not read here.

    Units are the census's own district ids, and sources/pk_2023_geo.py builds the hexes on the
    same ids: COD-AB tehsils for 129 districts, Lehri's two tehsils moved to Sibi and Kachhi,
    and OSM for Karachi's seven.
    """
    from pk2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pk.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 136:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 136 -- re-run "
                         "sources/pk_2023.py")
    df["node"] = df["source_category"].map(resolve)
    if df["node"].isna().any():
        raise SystemExit(f"pk2023: unmapped categories "
                         f"{sorted(df.loc[df['node'].isna(), 'source_category'].unique())}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    # Hindu Jati and Scheduled Castes both land on `hinduism`; add them, since scatter.py
    # allocates per (unit, node).
    df = df.groupby(["unit", "node"], as_index=False).agg(
        {"count": "sum", "congregations": "sum"})
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "pk": dict(
        name="Pakistan",
        source="Digital Census 2023 (PBS), Table 9",
        basis="self-identification",
        view=[60.8, 23.6, 77.9, 37.1],
        note_public=(
            "**Pakistan is 96.4% Muslim, and 78 of its 136 districts are over 99% Muslim.** "
            "Almost the whole map is one colour. The other 3.6% is not spread thin; it sits in a "
            "few places. "
            "**The Hindu population is southeastern Sindh.** Umer Kot is **54.7%** Hindu, the "
            "only district in Pakistan without a Muslim majority, with Tharparkar at 45.6%, "
            "Mirpur Khas 41.5% and Tando Allahyar 36.6%. Sindh as a whole is 8.8% Hindu, Punjab "
            "0.2% and Khyber Pakhtunkhwa 0.02%. The census counts *Hindu Jati* (3.87m) and "
            "*Scheduled Castes* (1.35m) as separate answers and this map draws both as Hinduism, "
            "because Scheduled Castes is a caste category, Pakistan's Dalit communities such as "
            "the Meghwar, Bheel and Kolhi, and not a separate religion. "
            "**The Christian population is central Punjab and the capital.** Lahore is 4.6% "
            "Christian and Islamabad 4.3%, with Sheikhupura, Gujranwala, Sialkot, Kasur and "
            "Faisalabad all near 3.5%. The census does not divide it by church. "
            "**Ahmadis are counted, and the count is a floor.** It is 162,684 people, down from "
            "191,737 in 2017, and 67,223 of them live in Chiniot district (4.3%), which contains "
            "Rabwah. Ahmadis identify as Muslim and this map files them under Islam; Pakistan's "
            "constitution declares them non-Muslim, which is why the census lists "
            "*Qadiani/Ahmadi* beside *Muslim*. Registering as Ahmadi puts a person on a separate "
            "electoral roll, and the community has boycotted the census over it since 1974, so "
            "independent estimates are several times this figure. "
            "**Sikhs and Parsis have their own colours for the first time.** The 2017 census had "
            "no box for either. There are 15,998 Sikhs, the most in Nankana Sahib (1,887), Guru "
            "Nanak's birthplace, then Peshawar and Buner; and 2,348 Parsis, 952 of them in "
            "Karachi South. The remaining *Others*, 72,346 people, has a geography of its own: "
            "Lower Chitral is 1.5% Others, and that is the Kalasha. "
            "**The blank in the north is missing data, not empty land.** The published census "
            "covers the four provinces and Islamabad, and Azad Kashmir and Gilgit-Baltistan are "
            "in none of its tables. Gilgit-Baltistan is also where Pakistan's Shia population is "
            "most concentrated; the census does not ask about Sunni and Shia anywhere. A further "
            "1,041,342 people in restricted areas were counted by head only, with no religion "
            "recorded."),
        how="census, 2023",
        grain="districts, 1.8m people on average",
        gap=("0.43%, in restricted areas counted by head only with no religion recorded; and "
             "Azad Kashmir and Gilgit-Baltistan, which are in no published table"),
        gap_share=0.004312,
        counts=_pk2023_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pk2023" / "pk_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_pk_place_weight,
        note="**2023 CENSUS, PBS'S OWN TABLE 9, SINCE 2026-09-14** (sources/pk.md §9). The 2017 "
             "USCB build it replaced is kept whole: `_pk_counts`, `sources/pk.py` (now writing "
             "pk2017.csv), `sources/pk_geo.py`, `taxonomy/pk2017.py`. Five PDFs read by page "
             "geometry, and every check closes: each block adds across religions, sexes and "
             "rural/urban; 590 tehsils sum to their districts and 136 districts to their "
             "provinces on all nine columns; provinces equal NCR 2023 Table 4.13, whose Punjab "
             "Muslim cell is misprinted 24,462,897 for 124,462,897; Lahore and Islamabad equal the "
             "dead census23 portal's archived records (§7a). "
             "**THE TIER IS DISTRICT BY ANITA'S CHOICE, NOT §14.4's CEILING.** The 2023 state "
             "prints religion by tehsil too (Lalian 13.4% Ahmadi); she kept district on "
             "2026-09-14 and the tehsil rows in pk.csv are not read. "
             "**THE BOUNDARIES ARE THE 2023 DISTRICT SET, AND NO ONE FILE HAS IT** "
             "(sources/pk_2023_geo.py): COD-AB v01 tehsils for 129 districts, joined by name 1:1 "
             "with COD's tehsil names as a second key; COD's Lehri split tehsil by tehsil into "
             "Sibi and Kachhi, as the census prints it; and Karachi's seven 2023 districts, which "
             "COD's 2001 towns cannot rebuild, from OSM admin_level=6 clipped to COD's Karachi. "
             "Nothing is dissolved. Kontur 2023-11 against the census is 0.98 nationally and 0.98 "
             "at the median district; Balochistan runs low (Quetta 0.46, Surab 0.44), which is "
             "the grid and the census disagreeing about Balochistan rather than the join. Hindu "
             "and Christian shares against their 5 nearest districts give r=0.92 and 0.75, where "
             "200 shuffles reach at most 0.49.",
    ),
}
