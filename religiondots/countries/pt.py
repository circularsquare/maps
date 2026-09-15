# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _pt_counts():
    """INE Censos 2021 at freguesia: 11 nodes on 3,092 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    pt.csv also carries the country, three NUTS1, nine NUTS2, 26 NUTS3 and 308 municípios,
    which are the same people five more times; only `freguesia` is read.

    **A PERFECT PARTITION AT EVERY LEVEL, WITH NO SUPPRESSION ANYWHERE.** The 11 categories
    sum to each unit's own published total on all 3,439 units, and the 3,092 freguesias sum
    to the national figure category by category with a largest discrepancy of zero. There is
    no rounding, no withheld cell and no `not stated` column — Judaism is published down to
    single people, which is how Belmonte's 49 survive to be drawn.

    **THE UNIVERSE IS PEOPLE AGED 15 AND OVER WHO ANSWERED, AND IT IS NOT SCALED UP.**
    8,781,900 of a 10,343,066 population: 1,331,188 children are outside the question and
    229,978 more declined and were removed by INE from the denominator rather than published
    (sources/pt.py). Chile is the other 15+ source here and cl2024.py takes the same line —
    drawing 85% of a country is honest, and inflating it to 100% on the assumption that
    children and refusers look like their neighbours is not.
    """
    from pt2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pt.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "freguesia"].copy()
    if df["geo_id"].nunique() != 3_092:
        raise SystemExit(f"{df['geo_id'].nunique()} freguesias, expected 3,092 -- re-run "
                         "sources/pt.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "pt": dict(
        name="Portugal",
        source="Censos 2021 (INE)",
        basis="self-identification, voluntary question, people aged 15 and over",
        view=[-9.6, 36.9, -6.1, 42.2],
        note_public=(
            "**Portugal is 80.2% Catholic — the highest share of any country on this map "
            "that asks the question directly** — and it is drawn on 3,092 freguesias, "
            "about 2,800 answering people each, which is fine enough that the exceptions "
            "are individual villages rather than regions. "
            "**The country has a gradient and it runs north to south.** The Azores are "
            "91.6% Catholic and the Norte 88.1%; Grande Lisboa is 68.4% and the Península "
            "de Setúbal 65.3%, where a quarter of people report no religion against 8.7% "
            "in the Norte. That is the older split — a rural, clerical north against the "
            "latifundia south, where the Church was weak long before the 20th century — "
            "and it is still the strongest pattern in the data. "
            "**The most striking thing on the map is 30 km of the Alentejo coast.** In São "
            "Teotónio, 17.1% of people are Hindu; in neighbouring Longueira/Almograve, "
            "17.1% are Buddhist, 9.3% Hindu and 6.2% Muslim, and only 43.7% Catholic — the "
            "least Catholic freguesia in Portugal. This is the intensive berry and "
            "greenhouse belt around Odemira and its South and Southeast Asian workforce, "
            "and it appeared within about fifteen years. Nothing else in Western Europe on "
            "this map looks like it. "
            "**Belmonte is the other one, and it is much older.** 49 people in one "
            "freguesia report Judaism — 1.6%, against 0.03% nationally, and the highest "
            "Jewish share in the country by a wide margin. They are the descendants of the "
            "crypto-Jewish community that kept practising in secret for roughly five "
            "centuries after the forced conversion of 1497 and returned openly only in the "
            "1970s. A census that publishes single people at this grain is what makes 49 "
            "of them visible at all. "
            "**Two immigrations show cleanly.** The Orthodox are Ukrainian, Romanian and "
            "Moldovan and their geography is the Algarve rather than Lisbon — 3.2% of "
            "Algarve answers, 8.5% in Almancil — because they came for the tourism labour "
            "market. And Lisbon's Muslims are concentrated: Santa Maria Maior, the old "
            "Mouraria, is 18.2% Muslim against 0.4% nationally. "
            "**What the source cannot show.** One cell holds every Protestant and "
            "Evangelical from Lusitanian Anglicans to Brazilian Pentecostals; one holds all "
            "Muslims, although Portugal's community is substantially Ismaili from "
            "Mozambique and the Imamat's seat is in Lisbon. "
            "**And what is not drawn.** Children were not asked — the question covers "
            "people 15 and over — and 230,000 more, 2.6% of that universe, declined and "
            "were removed by INE rather than published as a category. So every share here "
            "is a share of the adults who answered, and about 15% of Portugal is absent "
            "from this map entirely."),
        how="census, 2021, voluntary, ages 15 and over",
        grain="freguesias, 2,800 people on average",
        counts=_pt_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pt" / "pt_freguesias.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="**THE CLEANEST INGEST IN THE PROJECT, AND IT NEEDED ONE GET AND NO NEW "
             "DOWNLOAD FOR THE BOUNDARIES.** INE indicator `0012311` is 7.4 MB of JSON with "
             "no key and no wall, and GISCO LAU 2021 — on disk since Poland (§9e) — carries "
             "Portugal's LAU as the freguesia with a six-digit `LAU_ID` that is INE's own "
             "`geocod` character for character. **3,092 counted units, 3,092 polygons, zero "
             "unmatched either way, and the names agree on all 3,092** once accents are "
             "folded. Every other country here has paid for its join; this one did not. "
             "**The find was the catalogue, not the table.** `xml_indic.jsp?opc=3` is the "
             "endpoint that looks like INE's catalogue and it is a trap — 326 recently "
             "updated indicators, zero hits for religion. `opc=2` is the real one: 13,098 "
             "indicators, 21 MB, with `geo_lastlevel` per row, so 'which Portuguese census "
             "tables reach the freguesia' is a string search. An earlier probe of this "
             "office concluded Portugal published nothing on religion and was one digit "
             "away from the answer (sources.md §11k). "
             "**A perfect partition with no suppression at any level**, checked per category "
             "at both drawn tiers: largest discrepancy zero. "
             "**The universe is the caveat and INE does not flag it in the table.** The 11 "
             "categories sum to the published total exactly, which reads as a mandatory "
             "question; it is not one. The 15+ population is 9,011,878 and this table holds "
             "8,781,900, so 229,978 people — 2.55% — declined and were removed from the "
             "denominator rather than given a cell. Guyana's §9r footnote problem without "
             "the footnote, found by differencing against indicator 0011609; sources/pt.py "
             "asserts the gap so a future vintage cannot change it silently. "
             "**Placement is uniform within the freguesia (§8.2), which is the one thing "
             "left undone.** The median freguesia is 16.5 km² and that is fine, but the "
             "Alentejo units run to 863 km² at single-digit people per km², so dots there "
             "spread across empty cork forest. Kontur would fix it as it did for Kenya and "
             "Ethiopia; the tier is already fine enough that this is an improvement rather "
             "than a correction. "
             "**The Azores and Madeira are drawn** and sit outside the default view.",
    ),
}
