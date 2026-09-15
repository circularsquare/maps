# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ro_counts():
    """INS RPL 2021 at UAT: 23 recognised cults on 3,181 units, mapped at branch level.

    ONE level; ro.csv also carries judet and country, which are the same 19 million people
    counted again.

    THE KEY IS INDIRECT. The census publishes no SIRUTA code — rows are named only — so
    `geo_id` here is the string "COUNTY|NAME", and sources/ro_geo.py resolves it to a
    SIRUTA code through the Eurostat LAU-NUTS correspondence table and writes the result
    to ro_uat_lookup.csv. That resolution is where the work is (name folding, ş/ș, and
    four places settled by elimination inside their county), and it is done once there
    rather than every time this runs.

    INS SUPPRESSES. `*` marks a confidential cell and sources/ro.py drops those rows
    rather than guessing, so 16,493 people — 0.087% of the country — are in a category
    somewhere and not in any row here. Nothing else is lost: the totals reconcile exactly.
    """
    from ro2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ro.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "uat"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ro" / "ro_uat_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} ro.csv rows have no SIRUTA code -- re-run "
                         "sources/ro_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ro": dict(
        name="Romania",
        source="Recensământul Populaţiei şi Locuinţelor 2021 (INS)",
        basis="self-identification, partly from administrative registers",
        view=[20.2, 43.5, 30.0, 48.4],
        gap="14.0%, for whom religion is in none of the registers the census was built from",
        gap_share=0.1403,
        note_public=(
            "Religion could not be established for 14% of Romania. The 2021 census was "
            "built largely from administrative registers, which do not record religion, "
            "so this is an absent variable rather than a refusal — and those 2.7 million "
            "people are not drawn, leaving 16.4 million of 19.1 million. The 23 "
            "categories are Romania's list of state-recognised cults, so the detail is "
            "set by statute rather than by the question: no denomination outside the "
            "list is named at all. What the list does carry is unusual — the Lipovan Old "
            "Believers of the Danube delta, the largest such population any census "
            "publishes; the Hungarian Unitarians of Transylvania, a church continuous "
            "since 1568; and the Saxon and Hungarian Lutheran churches counted apart."),
        how="census, 2021, built from registers; missing for 14%",
        grain="communes and towns, 5,100 people on average",
        counts=_ro_counts,
        # UATs are the count layer and the placement layer: INS publishes religion at no
        # finer unit. Median UAT is about 3,000 people, the finest count geography on the
        # map after Ireland's Small Areas and the UK's Output Areas.
        #
        # Bucharest is the exception and it is a bad one — one UAT holding 9.8% of the
        # country in 240 km², worse than Warsaw's 4.7% and close to Prague's 12.4%. The
        # six sectors exist as administrative units but INS publishes no religion for
        # them, so subdividing would invent structure the source does not have.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ro" / "ro_uat.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="INS suppresses small cells with '*'; those rows are dropped rather than "
             "estimated, costing 0.087% of the country (sources/ro.md §3).",
    ),
}
