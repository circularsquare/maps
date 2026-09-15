# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sb_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Solomon Islands wards are coasts with empty interiors: Guadalcanal and Malaita are
    mountain and forest inland and settled around the shore. Weighting a ward's dots by
    its area would put them in the bush (sources/sb_grid.py).
    """
    return _kontur_place_weight(place, "sb_hexes.gpkg", "sources/sb_grid.py")


def _sb_counts():
    """SINSO 2019 census Table P8.3 at ward: 16 drawn categories on 183 wards.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    THE TABLE RECONCILES TO THE PERSON. P8.3's 183 ward rows sum to the printed national
    row exactly on all eighteen columns, and so do the ten province rows, and every ward's
    own categories sum to its own total. No tolerance is needed anywhere, which is rare on
    this map -- Vanuatu's equivalent (§9bg) misses its own totals by up to two.

    720,956 PEOPLE OVER 183 WARDS IS 3,940 EACH, the finest tier of any Pacific country
    here and finer than most of the map.

    THE PROVINCE ROWS ARE NOT UNITS. sources/sb.py writes only the 183 ward rows to sb.csv
    and asserts the hierarchy adds up before it does.
    """
    from sb2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sb.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "ward"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "sb" / "sb_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"sb.csv wards with no polygon: {missing} -- re-run "
                         "sources/sb_geo.py, the lookup is stale")
    if df["unit"].nunique() != 183:
        raise SystemExit(f"{df['unit'].nunique()} wards, expected 183")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "sb": dict(
        name="Solomon Islands",
        source="2019 National Population and Housing Census, Report Vol 2 Basic Tables, "
               "Table P8.3 (Solomon Islands National Statistics Office)",
        basis="self-identification, whole enumerated population",
        view=[155.0, -12.6, 170.5, -4.8],
        note_public=(
            "**The missions divided these islands between them in the 1800s and the census "
            "still shows the line.** The Church of Melanesia, the Anglican province, is "
            "**89.1% of Isabel, 85.1% of Temotu and 82.3% of Central**, and 2.0% of "
            "Choiseul. Choiseul and Western went to the Methodists, whose successor the "
            "United Church is 53.5% and 38.8% there and almost nothing anywhere east. The "
            "Catholics hold Guadalcanal at 36.2%. No province looks like its neighbour. "
            "**The South Sea Evangelical Church is the labour trade coming home.** It is "
            "**17.3% of the country**, and it exists because the Queensland Kanaka Mission "
            "evangelised Solomon Islanders working the Queensland cane fields from 1886 and "
            "they brought it back with them. Malaita, which sent most of those labourers, is "
            "**28.1%** South Sea Evangelical; Isabel is 0.4%. Nothing else on this map is a "
            "church that was founded among migrant workers abroad and became a national "
            "church at home. "
            "**And one church here was founded on one island and stayed there.** The "
            "Christian Fellowship Church is Silas Eto's break with the Methodist mission on "
            "New Georgia in 1960; his followers called him the Holy Mama. It is 16,179 "
            "people, of whom 13,629 are in Western Province, and it reaches **77.2% of "
            "Kusaghe ward and 62.6% of Roviana Lagoon** while rounding to nothing in the "
            "rest of the country. "
            "**Custom belief is a printed census category**, at 0.57%, and it is not spread "
            "thinly: it is 21.5% of one Malaita ward and 21.2% of one on Guadalcanal, the "
            "Kwaio interior and the Weather Coast, which are the two places the missions "
            "reached least. "
            "**The census names the Baha'i and Muslim communities separately**, at 3,104 and "
            "1,100, which most censuses on this map do not. Only 133 people in the whole "
            "country refused the question."),
        how="census, 2019, whole enumerated population",
        grain="wards, 3,900 people on average",
        counts=_sb_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sb" / "sb_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sb_place_weight,
        note="THE TABLE RECONCILES TO THE PERSON, WHICH IS RARE HERE. P8.3's 183 ward rows "
             "sum to the printed national row exactly on all eighteen columns; so do the ten "
             "province rows; and every ward's own categories sum to its own total. No "
             "tolerance anywhere, against Vanuatu's published table (§9bg) missing its own "
             "totals by up to two. Volume 1's Table 8.3.1, typeset separately, reproduces "
             "every figure. "
             "183 WARDS FOR 720,956 PEOPLE IS 3,940 EACH, the finest tier of any Pacific "
             "country on this map. "
             "THE JOIN IS ON SINSO'S OWN WARD ID AND NOT ON THE NAME, WHICH MATTERS HERE. "
             "COD-AB's ADM3 is sourced from SINSO's own census geography and carries "
             "SINSO_WID; the census prints a ward number that restarts inside each province, "
             "and the id is the province number followed by it. 183/183, nothing spare. A "
             "NAME join would have matched only 154 of 183: Solomon Islands English writes "
             "prenasalised stops both ways, so Mbilua/Bilua and Ndovele/Dovele are the same "
             "wards. AND ONE PAIR IS NOT A SPELLING AT ALL -- Isabel ward 02 is `Baolo` in "
             "the census and `Havulei` in COD, neither name appearing on the other side, "
             "which is a renamed ward and is exactly what a name join cannot see. The "
             "province is the witness: OCHA's ADM1 against the census's own province blocks, "
             "agreeing on all 183. sources/sb_grid.py then correlates census against Kontur "
             "at r=0.897 over 183 wards, which none of 2,000 random pairings comes near "
             "(best 0.27). "
             "PARSING P8.3 NEEDED THE COLUMN COUNT, NOT A THRESHOLD. Each page sizes its "
             "columns to its own widest figure, so a fixed x map fails and pooling the pages "
             "merges neighbours; and the last page holds only twelve wards, too few for a "
             "gap threshold to separate. What is known is that there are eighteen columns, "
             "so the right edges are cut at the seventeen largest gaps. That last page also "
             "carries the whole of P8.4 (ethnicity, 13 columns) with its own `Province` "
             "header, so the table is bounded by its own title and the next one's. "
             "17% OF KONTUR'S PEOPLE FALL OUTSIDE EVERY WARD AND ARE SNAPPED, NOT DROPPED, "
             "on §9bg's rule: 99.7% are within 500 m and the loss would be entirely seaward, "
             "which would pull every coastal ward's dots inland. TWO WARDS ARE STILL "
             "OUTLIERS AND BOTH ARE THE GRID'S BLIND SPOT rather than a bad join -- "
             "Sulufou/Kwarande is the Lau Lagoon, where people live on ARTIFICIAL ISLANDS, "
             "and Sikaiana is an atoll 210 km out. A building-footprint model under-detects "
             "both. Naha, an 0.08 km2 ward in Honiara, is smaller than one 400 m hex and is "
             "given its own polygon as a single cell (§8.2).",
        gap_share=0.0002,
        gap="Religion Faith/Refuse to Answer is a §3.5 residual and is not drawn: 133 "
            "people, 0.02%, the smallest on this map. The Solomon Islands are 99.98% drawn.",
    ),
}
