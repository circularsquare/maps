# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _EsMuniWeighter:
    """Split a province's dots across its municipios by municipal POPULATION.

    Spain's counting geography is the province — 52 units, 940,000 people each — and nothing
    in the country measures religion below it, so this is a population weight and not a
    religion one, exactly as sources/ru.py's is. A Muslim dot in Almería sits where Almería's
    people are, not where its Muslims are, and the same is true of every other node.

    **What that costs is visible and worth stating.** Almería's Muslims are really the
    greenhouse belt — El Ejido, Níjar, La Mojonera, Roquetas — and this spreads them evenly
    over a province that also contains the Sierra de los Filabres. The fix exists and is not
    a weight: the Observatorio del Pluralismo Religioso publishes 7,756 geocoded non-Catholic
    places of worship over 1,378 municipios, which is a §4.4 location layer. Using it here
    would make the map's dots follow buildings rather than people, which is a different claim
    from the one the counts support.
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
        return (f"{self.n_pop:,} (unit, node) rows placed on municipal population, "
                f"{self.n_uniform:,} on equal shares")


def _es_place_weight(place):
    """Spain's 8,131 municipios, weighted by WHICH population the dots are of.

    §9as's finding, and Spain is the country it matters most in: **CIS does not sample
    foreigners at all**, so the foreign half is not a correction to the survey but the other
    15% of the country — 7.4M people, a larger share than Italy's 8.5% — and every one of
    them was being scattered by where Spaniards live. INE's table 33571 gives Spanish and
    foreign nationals per municipio at the same five-digit code this layer already uses.
    See `_ItWeighter`; placement only, never a magnitude.
    """
    if not all(c in place.columns for c in ("pop", "spanish", "foreign", "unit")):
        print("  !! es_municipios.gpkg lacks the nationality columns — run "
              "sources/es_geo.py --fetch then sources/es_geo.py")
        return _EsMuniWeighter(place) if "pop" in place.columns else None
    from es2026 import resolve
    return _ItWeighter(place, _foreign_share("es", resolve, "province"),
                       citizen_col="spanish", foreign_col="foreign",
                       citizen_label="Spanish", place_label="municipio")


def _es_counts():
    """Spain at province: 41 nodes on 52 units, from two sources that partition the country.

    ONE level and no allocation, but TWO POPULATIONS, and the reason is the whole country:

      * **Spanish citizens, 42.4M.** CIS's monthly barómetro, 101 studies pooled over three
        years, 464,524 respondents, all 52 provinces — about 8,900 each, which is by a wide
        margin the largest survey sample behind any country on this map. Six answers.
      * **Foreign nationals, 7.4M.** INE's count by province x nationality crossed with Pew's
        composition for each origin country (taxonomy/es_origin.py). **CIS does not sample
        them at all** — its `NACIONALIDAD` variable has two values, both Spanish — so this is
        not a correction to the survey, it is the other 15% of the country.

    Every row is tier `modelled`: a survey is not a count of anybody, and neither is a
    nationality model. §7 draws both desaturated and the about panel says which is which.

    **98.36% of Spain is drawn.** The foreign half is drawn whole; 1.9% of citizens refuse
    CIS's question and spec §3.5 marks that rather than filling it.

    **The one place the two halves have to be reconciled is Islam**, because Spanish-citizen
    Muslims are inside CIS's universe and inside its single unnamed "other religion" cell.
    sources/es.py splits them out using UCIDE's province table; in four provinces — Almería,
    Teruel, Ceuta and Melilla — UCIDE's figure exceeds that whole cell and is capped to it,
    so those four are drawn LESS Muslim than UCIDE would have them, not more.
    """
    from es2026 import resolve

    esp = pd.read_csv(HERE / "data" / "normalized" / "es.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    esp = esp[esp["geo_level"] == "province"].copy()
    esp["node"] = esp["source_category"].map(resolve)
    unmapped = sorted(set(esp.loc[esp["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"es.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "es_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "province"]

    df = pd.concat([esp[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # Both halves are estimates — a survey is not a count of anybody and neither is a
    # nationality model — so nothing here is `measured` and §7 draws all of it desaturated.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "es": dict(
        name="Spain",
        source="CIS barómetros 2023–26 (citizens) + INE padrón x Pew 2020 (foreign residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[-18.3, 27.5, 4.5, 43.9],
        note_public=(
            "**Spain has never asked about religion in a census, and it is one of the "
            "best-measured countries here anyway.** The CIS barómetro asks every single "
            "month and publishes the microdata free, so pooling three years gives 464,524 "
            "answers with a province code — about 8,900 per province, and by a wide margin "
            "the largest survey behind any country on this map. "
            "**But the barómetro only interviews Spanish citizens.** Its nationality "
            "variable has two values, 'Spanish' and 'Spanish and another', so the 7.4 "
            "million foreign nationals living in Spain — 15% of the country, and most of "
            "its religious variety — are outside the frame rather than under-sampled in "
            "it. They are drawn separately, from INE's count of who lives in each province "
            "and where they are from, crossed with the religious make-up of each origin "
            "country. **So half this map is what people said and half is where they came "
            "from**, and the second half is an upper bound: it cannot see anyone who "
            "stopped practising after arriving. "
            "**Catholicism is 51% and the country is more irreligious than Catholic if you "
            "count practice.** 18% of citizens call themselves practising Catholics and "
            "37% non-practising, against 13% agnostic, 16% atheist and 12% indifferent — "
            "36% who claim no religion at all, a share exceeded in Europe only by Czechia "
            "and Estonia. The Catholic map runs southwest to northeast: **Jaén 69%, "
            "Badajoz 66%, Ciudad Real 66% against Girona 41% and Barcelona 41%**, with the "
            "Basque provinces the most atheist and agnostic in Spain at about 31%. "
            "**Islam is 5.8% and the two enclaves are a different country.** Melilla is "
            "38% Muslim and Ceuta 32% — the only majority-or-near-majority Muslim places "
            "in the European Union — and after them come **Almería at 17%, Lleida 14%, "
            "Girona 14%, Tarragona 12% and Murcia 10%**, which is the intensive "
            "agriculture belt and its Moroccan workforce rather than the cities. Madrid is "
            "below the national average. "
            "**The second-largest immigration is invisible in every other account of "
            "Spain.** 630,000 Romanians and 120,000 Bulgarians make the Romanian Orthodox "
            "Church the country's third-largest religious body, and its geography is not "
            "the cities either: **Castellón is 7.8% Romanian Orthodox, Cuenca 5.8%, Lleida "
            "5.2%, Guadalajara 5.1%.** Protestantism, at 1.6%, is the opposite — Alicante "
            "3.7% and Málaga 3.4%, which is British and northern European retirement plus "
            "Latin American evangelical churches in the same provinces. "
            "**The biggest hole is one cell on the CIS form.** Everything that is not "
            "Catholicism is offered to Spanish citizens as a single box, 'a believer of "
            "another religion', with no follow-up asking which. Spanish Muslims are split "
            "back out of it using UCIDE's province figures; the rest — Spain's own "
            "evangelicals, its naturalised Orthodox, about 110,000 Jehovah's Witnesses and "
            "45,000 Jews — stays in one unnamed 2%. **And in Almería, Teruel, Ceuta and "
            "Melilla that cell is smaller than UCIDE's count of Spanish Muslims alone**, so "
            "those four provinces are drawn less Muslim than UCIDE would have them. "
            "**Within a province the dots are not scattered blindly**: people counted as "
            "foreign nationals are placed where foreign nationals live, municipio by "
            "municipio, and Spaniards where Spaniards live. It changes less here than it "
            "would elsewhere — 55.3% of Spain's foreign residents live in cities against "
            "53.5% of its citizens, because the foreign population is on the coast, in the "
            "Almerian greenhouses and in the islands as much as in Madrid and Barcelona. "
            "**About 2% of Spain is not drawn**: the citizens who declined the question."),
        how="opinion poll, 464,524 answers; foreign residents by nationality",
        grain="provinces, 940,000 people on average",
        counts=_es_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "es" / "es_municipios.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_es_place_weight,
        note="**Two sources that partition the country rather than compete for it, and "
             "finding that out was the whole ingest.** CIS's `NACIONALIDAD` variable takes "
             "exactly two values, both Spanish, so its 3-4% 'other religion' figure is a "
             "share of citizens and not of residents — which is why it looks irreconcilable "
             "with UCIDE's 5%-of-Spain Muslim estimate and is not: 4.6% of 42.4M citizens "
             "is 1.9M, and UCIDE's Spanish-citizen Muslim figure is 1.09M, inside it. "
             "**101 CIS studies pooled, 464,524 respondents, and two rejected.** CIS reuses "
             "the variable name `RELIGION` for a different question in estudios 3462 and "
             "3506, where code 1 means 'no religion' instead of 'practising Catholic'. "
             "Pooling those would have moved several hundred thousand irreligious Spaniards "
             "into the Catholic column with every total still summing; taxonomy/es2026.py "
             "checks the whole value-label signature rather than the codes, which is the "
             "only reason it was caught. "
             "**The microdata is open and the site is not.** cis.es answers a plain fetch of "
             "its catalogue with a BunkerWeb challenge page, while `/documents/d/guest/MD<n>` "
             "and `MD<n>-zip` — two naming conventions, both live — serve the zips with no "
             "key at all. §9s's KOSIS wall inverted. "
             "**Boundaries cost nothing.** GISCO LAU 2021 carries Spain's 8,131 municipios "
             "with INE's own five-digit code as `LAU_ID`, and the first two digits ARE the "
             "province, so the counting geography is derivable from the placement geography "
             "with no join and none of §8.1's failure modes. "
             "**Three checks that passed and one that is a limit.** INE's 121 nationality "
             "leaves partition its published foreign total to +0.000%; UCIDE's 52-province "
             "table sums to the national figure the report states in its own prose "
             "(1,085,593), which a mis-parsed column could not do; and the foreign-half "
             "Muslim total, 1.82M, sits 30% above UCIDE's implied foreign figure, which is "
             "the expected direction for a Pew national composition applied to migrants and "
             "is reported rather than tuned away. The limit is vintage: INE's detailed "
             "nationality series stops in **2022** and the totals are July 2026, so every "
             "province's nationality mix is rescaled uniformly (§3.4) and the post-2022 "
             "Ukrainian arrivals in particular are understated. "
             "**Placement is municipal population (§8.2), which is a population weight and "
             "not a religion one** — Almería's Muslims spread over the whole province rather "
             "than over the El Ejido greenhouses. The Observatorio del Pluralismo Religioso's "
             "7,756 geocoded non-Catholic places of worship would fix it and are a §4.4 "
             "layer, not a weight; that is the biggest upgrade outstanding here.",
    ),
}
