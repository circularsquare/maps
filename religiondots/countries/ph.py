# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _PhBarangayWeighter:
    """Split a unit's dots across its barangays by barangay POPULATION.

    Every other placement layer on this map is one a statistical agency designed to a
    population target, so an equal share per polygon is already a population weighting
    (spec §8.2). Philippine barangays are not that: they are the country's political base
    unit, they range from a few hundred people to well over a hundred thousand, and
    Quezon City's 142 hold as many people as several whole provinces. An equal share would
    put as many dots in an empty upland barangay as in a Metro Manila one.

    So the weight is the barangay's own 2020 population. It is a POPULATION weight and not
    a religion one — unlike Germany, nothing here measures where a given church's members
    live inside a province, so a Baptist dot and a Catholic dot are spread the same way.
    That is §8.2's proxy doing its ordinary job, and it is the reason the map should be
    read as "religion by province, drawn where the people are" rather than as a
    measurement at barangay grain.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on barangay population, "
                f"{self.n_uniform:,} on equal shares where a unit's barangays sum to zero "
                f"(sources/ph_geo.py)")


def _ph_place_weight(place):
    """countries.py hook. `place` is the barangay layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ph_barangays.gpkg has no `pop` column — run sources/ph_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _PhBarangayWeighter(place)


def _ph_counts():
    """PSA 2020 CPH: 129 categories on 117 provinces, HUCs and the BARMM interim province.

    ONE level, no allocation, nothing modelled. PSA publishes the whole 129 x 117 matrix
    and it is a true partition — the categories sum to each unit's household population
    exactly, in all 117, with no residual to compute (§3.2 has nothing to do here) — so
    every row is `measured` and may ring.

    THE FINE TIER IS province + city + municipality AND IT PARTITIONS THE COUNTRY. The
    `province` rows already EXCLUDE any highly urbanised city inside them and the `city`
    rows are those 33 HUCs plus the City of Isabela; `municipality` is Pateros, the only
    one in NCR. The `region` and `country` rows are aggregates of these and are dropped —
    adding them would double the country.

    THE UNIVERSE IS THE HOUSEHOLD POPULATION, 108,667,043 of 109,035,343 (spec §3.7). The
    368,300 not in it are the institutional population, and unlike Chile's 15+ gap this
    one is NOT scaled up: at 0.34% it changes no share, and it is the one gap this project
    would most like to see, since seminaries, convents and monasteries are exactly what a
    household table cannot reach. Drawn: 99.66% of the country.
    """
    from ph2020 import resolve

    # keep_default_na=False: PSA's category for no religion is the string "None", which
    # pandas turns into NaN under default parsing. It would then fail to resolve and be
    # dropped four lines down, silently removing 43,931 people who answered the question.
    df = pd.read_csv(HERE / "data" / "normalized" / "ph.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(["province", "city", "municipality"])].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ph": dict(
        name="Philippines",
        name_in="the Philippines",
        source="2020 Census of Population and Housing (Philippine Statistics Authority)",
        basis="self-identification, household population",
        view=[116.5, 4.3, 127.0, 21.4],
        note_public=(
            "The Philippine census asks for a religion and offers **129 named bodies** to "
            "answer with — the longest list on this map after the US Religion Census, and "
            "the longest anywhere that people answered for themselves. It is also the "
            "most lopsided. Of the 129, one hundred and twenty-six are Christian, and the "
            "entire non-Abrahamic world gets three: Islam, Buddhist, and Tribal religion. "
            "**There is no Hindu box, no Jewish box, no Sikh box and no Chinese folk "
            "religion box anywhere on the form**, so everyone in those traditions is "
            "inside 'other religious affiliations' with no way out. The question is four "
            "levels deep on Philippine evangelicalism and zero levels deep on everything "
            "else, which is a fact about what the country argues about rather than about "
            "who lives in it. "
            "**Catholicism is 78.9% and the interesting thing is the shape of where it "
            "is not.** Three edges do almost all the work. Muslim Mindanao and Sulu are "
            "not a gradient but a wall — Sulu is 95% Muslim and 0.1% Catholic, Tawi-Tawi "
            "97%, Lanao del Sur 95% — and the boundary falls between provinces rather "
            "than running through them. The **Cordillera** is the Protestant region, and "
            "it is the sharpest thing on the northern half of the map: Mountain Province "
            "is 49% Protestant against 42% Catholic, and a quarter of the whole province "
            "is Episcopalian, which is the Anglican mission at Sagada still visible a "
            "century later; Ifugao, Benguet, Kalinga and Apayao run 31-39%. And **Ilocos "
            "Norte is Aglipayan** — 21% of the province belongs to the church Gregorio "
            "Aglipay founded in 1902 and was born a few miles from. "
            "**Iglesia ni Cristo is the country's third largest religious body and has no "
            "home province.** 2.8 million people, founded in Manila in 1914, and its "
            "highest share anywhere is 7.5% in Tarlac. Nearly every other body on this "
            "map has a region; INC has a country, which is unusual enough to be worth "
            "looking for as you pan. "
            "**Two things the census does that the map inherits.** It offered *Aglipay* "
            "and *Iglesia Filipina Independiente* as separate answers, and they are the "
            "same church: 818,916 people chose one name and 640,076 the other, and in "
            "Ilocos Norte both are used side by side. They are added back together here, "
            "which makes the Aglipayan church the fourth largest body in the country at "
            "1.46 million. And **43,931 people, four hundredths of one percent, reported "
            "no religion** — a number that measures the question rather than the country. "
            "One household member answered for everyone, and 'none' is a hard answer to "
            "give on a relative's behalf where belonging is assumed."),
        how="census, 2020",
        grain="provinces and cities, 930,000 people on average",
        counts=_ph_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ph" / "ph_barangays.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ph_place_weight,
        note="THE COUNTS ARE COARSE AND THE PLACEMENT IS FINE, and the two must not be "
             "confused. PSA publishes religion at province + highly urbanised city and "
             "nowhere below it — 117 units for 108.7M people, about 929,000 each, which is "
             "the coarsest counting geography on this map. The dots are then spread across "
             "42,042 barangays weighted by barangay population (sources/ph_geo.py), so "
             "they land where Filipinos live rather than evenly across a province, but "
             "nothing measures which barangay a given church's members are in. Read a "
             "cluster as 'this province, drawn where its people are', never as a "
             "neighbourhood. "
             "The 33 HUCs are cut out of their provinces by the census and by the "
             "polygons alike, and the BARMM Interim Province — 63 barangays with no "
             "polygon in any boundary set — is reconstructed from the US Census Bureau's "
             "own tagging of them. The universe is the household population, 99.66% of the "
             "country; the missing 0.34% is the institutional population, which is where "
             "the seminaries and convents are (spec §3.7).",
    ),
}
