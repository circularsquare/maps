# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _pl_counts():
    """GUS NSP 2021 at gmina: 139 named churches on 2,477 units, mapped at branch level.

    The second country after Czechia that needs no allocate.py step — GUS publishes named
    denominations at its finest geography, so nothing here is derived and every row may
    ring (spec §3.9/§3.10, sources.md §9e).

    ONE level, and the file carries four. pl.csv holds gmina, powiat, voivodeship and
    country, which are the same 38 million people counted four times; reading it as
    delivered would quadruple the country.

    The join key is SIX digits, not the seven GUS prints. TERYT's seventh digit is the
    gmina TYPE (1 urban / 2 rural / 3 mixed), and the GISCO LAU boundaries do not carry
    it, so `pl_gminy.gpkg` is keyed on the first six — which are already unique per gmina.
    sources/pl_geo.py derives that key and checks the join both ways.

    Unlike Czechia there are no explicit zeros to strip: GUS lists only the denominations
    it found in a gmina, so the 21,926 gmina rows are all positive and every one may ring.
    """
    from pl2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pl.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "gmina"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"].str[:6]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "pl": dict(
        name="Poland",
        source="Narodowy Spis Powszechny 2021 (Statistics Poland)",
        basis="self-identification, voluntary question",
        view=[14.0, 48.9, 24.2, 55.0],
        gap="20.7% of the country, who refused a voluntary question",
        gap_share=0.2072,
        note_public=(
            "The religion question was voluntary and 20.5% of the country refused it. "
            "Those people are not drawn, so this map shows 30.2 million of 38.0 million. "
            "What is drawn is unusually detailed: 139 churches named at the level of the "
            "gmina, with no rounding and no suppression, and the tail is individual "
            "congregations rather than denominations — the Betel congregation in Warsaw "
            "is two people and is on the map as itself. Poland is 98% Latin Catholic "
            "among those who named a church, so the interest is entirely in the other "
            "2%: Orthodoxy along the Belarusian border, Lutherans in Cieszyn Silesia, "
            "the Mariavites — a Polish movement of 1906 and the only Old Catholic church "
            "anywhere with a Polish origin — and Old Believers in Masuria."),
        how="census, 2021, voluntary (20.5% refused)",
        grain="gminas, 12,000 people on average",
        counts=_pl_counts,
        # Like Czechia and Ireland: the counts are already ON the finest unit GUS
        # publishes, so there is no separate placement layer and no allocation inside a
        # unit. Median gmina population is about 7,500, twice a US census tract, so an
        # equal share per polygon is a reasonable weighting nearly everywhere (spec §8.2).
        #
        # WHERE IT IS NOT: Warszawa is one gmina holding 1.79M people, 4.7% of the country
        # in a single 517 km² polygon, and Kraków, Łódź, Wrocław and Poznań are each one
        # too. Czechia had a fix for exactly this — ČSÚ publishes 142 city districts — and
        # GUS does not, so it stands. sources/pl_geo.md records it.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pl" / "pl_gminy.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="GUS is self_id on a voluntary question; the 20.5% who refused are excluded "
             "rather than drawn (spec §3.5).",
    ),
}
