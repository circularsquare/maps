# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _to_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    A Tongan village district runs from the shore back across its bush allotments and the
    houses are in a band at one end, so weighting by area would put the dots in the
    plantations. On the outer islands it is worse: Ha'atu'a on 'Eua is 43.5 km2 of forested
    plateau with its people on the west coast road (sources/to_grid.py).
    """
    return _kontur_place_weight(place, "to_hexes.gpkg", "sources/to_grid.py")


def _to_counts():
    """TSD 2021 census Table G 20 at village: 21 drawn categories on 156 villages.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    THE TABLE CLOSES FIVE WAYS. Villages sum to their district, districts to their division,
    divisions to the printed national row; tables G 19 and G 18, typeset separately in the
    same workbook, reproduce the district and division figures cell for cell; and UNSD's
    Demographic Yearbook table 28 reproduces the national row again on all twenty-two
    categories, from Tonga's own return rather than from this workbook. sources/to.py
    asserts every one of those before it writes a line.

    99,408 PEOPLE OVER 156 VILLAGES IS 637 EACH, the finest tier on this map after nothing
    at all -- finer per unit than the Solomon Islands' 3,940.

    THE DIVISION AND DISTRICT ROWS ARE NOT UNITS. sources/to.py writes only the 156 village
    rows to to.csv and asserts the hierarchy adds up before it does.
    """
    from to2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "to.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "village"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "to" / "to_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"to.csv villages with no polygon: {missing} -- re-run "
                         "sources/to_geo.py, the lookup is stale")
    if df["unit"].nunique() != 156:
        raise SystemExit(f"{df['unit'].nunique()} villages, expected 156")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "to": dict(
        name="Tonga",
        source="2021 Census of Population and Housing, General Table G 20 (Tonga Statistics "
               "Department)",
        basis="self-identification, whole enumerated population",
        view=[-176.4, -22.3, -173.0, -15.0],
        note_public=(
            "**Half of Tonga is Methodist and the census counts it as four separate "
            "churches.** The Wesleyan mission arrived in 1826 and converted the kingdom; "
            "every division since has been about who governs the church rather than about "
            "doctrine, and each one left a body that is still here. The Free Wesleyan "
            "Church is **34.2%**, the Free Church of Tonga **11.3%**, the Church of Tonga "
            "**6.8%** and the Constitutional Church of Tonga **1.2%**, which is **53.4%** "
            "of the country between them. Add the two later revival breakaways, Tokaikolo "
            "and Mo'ui Fo'ou 'ia Kalaisi, and 55.6% of Tonga descends from that one "
            "mission. No other country on this map divides a single Protestant tradition "
            "this far. "
            "**The state church is the one body here with no geography.** The Free Wesleyan "
            "Church, which is the church of the monarchy, is 34.1% of Tongatapu, 34.6% of "
            "Vava'u, 33.1% of Ha'apai, 36.3% of 'Eua and 30.7% of the Niuas. Six points "
            "across the whole kingdom, while the Church of Tonga runs from 3.8% to "
            "**20.1%** and the Catholics from 5.4% to **36.3%**. Everything else has a "
            "stronghold and the national church does not. "
            "**The Church of Tonga is the outer islands.** It is 6.8% nationally but "
            "**37.1% of Lulunga** and 29.8% of Ha'ano, the small islands scattered between "
            "Tongatapu and Vava'u, and it reaches 44.3% of Ha'afeva. The Catholics are the "
            "far north instead: **42.8% of Niuatoputapu**, 300 km beyond everything else, "
            "and **71.0% of Lapaha**, which was the seat of the Tu'i Tonga. "
            "**And Tonga is the most Latter-day Saint country the UN has a figure for.** "
            "19,534 people, **19.7%**, which is the largest share of any of the 67 censuses "
            "in the Demographic Yearbook's religion table that count Latter Day Saints "
            "separately; Samoa is second at 16.9%. It is 33.7% of Hahake district in Vava'u "
            "and 59.6% of Matahau on Tongatapu. "
            "**Seven village names in this map appear twice, 900 km apart.** Niuafo'ou was "
            "evacuated after the 1946 eruption and most of its people were resettled on "
            "'Eua, where they gave the new villages the names of the ones they had left. "
            "'Esia, Sapa'ata, Fata'ulua, Mata'aho, Mu'a, Tongamama'o and Petani are each "
            "printed twice in the census, once in the Niuas and once on 'Eua, and the twins "
            "are not alike. 'Esia on 'Eua is **71.4%** Catholic; 'Esia on Niuafo'ou is "
            "**62.7%** Free Wesleyan. Petani on 'Eua is 49.4% Free Wesleyan and Petani on "
            "Niuafo'ou is 33.8% Catholic. Matching them by name alone would have swapped "
            "real congregations and every total would still have added up. "
            "The census names the Baha'i community (730), Hindus (78), Muslims (60) and "
            "Buddhists (58) on lines of their own, which few censuses of a country this "
            "size do. Only 119 people refused the question."),
        how="census, 2021, whole enumerated population",
        grain="villages, 640 people on average",
        counts=_to_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "to" / "to_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_to_place_weight,
        note="RELIGION BY VILLAGE, IN A PUBLISHED SPREADSHEET, AT 637 PEOPLE PER UNIT. TSD "
             "posts the 2021 general tables as one workbook per topic and `4-religion.xlsx` "
             "holds three: G 18 by division crossed with sex, G 19 by district, G 20 by "
             "village. 156 villages, all 22 categories printed at every one. That is the "
             "finest tier on this map by population per unit, ahead of the Solomon Islands' "
             "3,940. "
             "THE TABLE CLOSES FIVE WAYS AND NEEDS NO TOLERANCE. Villages sum to their "
             "district, districts to their division, divisions to the printed national row; "
             "G 19 and G 18 are typeset separately in the same workbook and reproduce the "
             "district and division figures cell for cell; and UNSD's Demographic Yearbook "
             "table 28 reproduces the national row again on all 22 categories, to the "
             "person, from Tonga's own return rather than from this workbook. sources/to.py "
             "asserts all five before writing. "
             "G 20 HAS NO TIER MARKER, WHICH IS WHAT MAKES PARSING IT A PROBLEM. Divisions, "
             "districts and villages share one column with no indentation and no code, and "
             "districts are named for their largest village, so `Pangai` is both a Ha'apai "
             "district and a village inside it. G 19 resolves it: it prints the same figures "
             "for the divisions and districts ALONE, so walking it gives the expected tier "
             "of every G 20 row in order, and each district's villages are then read until "
             "they sum to that district's own printed total. "
             "VILLAGE NAMES ARE NOT UNIQUE AND THE DUPLICATES ARE NOT A SPELLING PROBLEM. "
             "Niuafo'ou was evacuated after the 1946 eruption and its people resettled on "
             "'Eua, naming the new villages after the old ones, so 'Esia, Sapa'ata, "
             "Fata'ulua, Mata'aho, Mu'a, Tongamama'o and Petani each occur twice; Kolofo'ou, "
             "Hihifo, Pangai, Houma and Eueiki repeat for ordinary reasons. A name join "
             "would pair some of them across the country and every total would still "
             "balance, which is exactly the failure [[reference_name_join_wrong_neighbour]] "
             "describes. The join is on the district and the village together. "
             "151 OF 156 PAIR ON THE FOLDED NAME AND THE OTHER FIVE ARE WITNESSED BY "
             "OPENSTREETMAP rather than assumed, because four are a rename and one is an "
             "error in COD. COD calls the census's `Nukunukumotu` Nukumotu; it names the "
             "Ha'apai village polygon for its island, `Lifuka`, where the census names it "
             "for the town, `Pangai`; the census prints `Ha'atu'a / Kolomaile` as one row "
             "against COD's single Ha'atu'a polygon; and COD LABELS TWO 'EUA POLYGONS "
             "`Ohonua` and has no Ta'anga at all. In each case an OSM `place` node of the "
             "census's name falls inside the polygon claimed for it, and for the 'Eua pair "
             "both do: Ta'anga's node is in TO4106 and 'Ohonua town's is in TO4101, which "
             "settles which is which. Kolomaile's node falls inside COD's Ha'atu'a, "
             "confirming that polygon already holds both villages of the combined row. "
             "THE WITNESS ON THE OTHER 151 IS THE DIVISION. COD files each village under an "
             "ADM1 independently of the census, and the two organisations agree on all 156. "
             "sources/to_grid.py then correlates census against Kontur at r=0.842 over 148 "
             "villages, which none of 2,000 random pairings comes near (best 0.26). "
             "TEN COD POLYGONS HAVE NO CENSUS ROW and are not units: uninhabited islets, six "
             "of them in the Vava'u lagoon. Whatever the grid puts on them is snapped to the "
             "nearest village. 16% of Kontur's people fall outside every village and are "
             "snapped, not dropped, on Vanuatu's rule (§9bg): Tonga is 171 islands and the "
             "loss would be entirely seaward, which would pull every shore's dots inland. "
             "EIGHT VILLAGES ARE SMALLER THAN ONE 400 M HEX and are given their own polygon "
             "as a single cell (§8.2), most of them the resettled Niuafo'ou villages on 'Eua "
             "at about 0.11 km2 each.",
        gap_share=0.0012,
        gap="Refuse to answer is a §3.5 residual and is not drawn: 119 people, 0.12%. Tonga "
            "is 99.88% drawn.",
    ),
}
