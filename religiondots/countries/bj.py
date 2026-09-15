# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bj_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Benin wants the grid for the ordinary reason and a coastal one. Karimama and Malanville
    are large northern communes holding most of the W National Park and almost nobody;
    Cotonou is 679,012 people in 80 km². And the lagoons — Lac Nokoué and the Porto-Novo
    lagoon — are INSIDE the communes rather than cut out of them, so an equal share would
    scatter Sô-Ava's dots over open water. A population grid has no hexes on empty water and
    does have them over Ganvié, the stilt town of ~30,000 built on that lake, which is spec
    §8.2c-i's point standing in one place (sources/bj_grid.py).
    """
    return _kontur_place_weight(place, "bj_hexes.gpkg", "sources/bj_grid.py")


def _bj_counts():
    """INStaD RGPH-4 2013, Tableau 8 at commune: 10 drawn categories on 77 communes.

    ONE level, no allocation, nothing modelled — INStaD publishes these categories at this
    geography and the map draws exactly that, so every row is `measured` and may ring.

    THE COUNTS ARE ARITHMETIC ON TWO PUBLISHED FIGURES, WHICH IS NOT THE SAME AS AN
    ESTIMATE. Tableau 8 prints shares to one decimal and Tableau 2 of the same booklet
    prints the commune's population, so `count = pct/100 x total` and nothing is carried,
    fitted or inferred — every person here was counted by INStaD in the commune they are
    drawn in. The cost is precision, not confidence: +/-0.05% of a unit, which is +/-34
    people in a 68,000-person commune. §7a's tiers are about whether anybody was counted,
    so these are `measured`.

    77 UNITS FOR 10.0M PEOPLE IS ~130,000 EACH — finer per head than Malawi's districts and
    eight times finer than Kenya's counties.

    TWO CATEGORIES DO NOT REACH THE MAP AND ONLY ONE IS A LOSS. `Non déclaré (calculé)` is
    the computed complement of the ten published shares, 120,826 people at 1.21%, and §3.5
    marks non-response rather than filling it. There is no other gap: the drawn population
    is 9,887,923, which is 98.79% of the census.

    COTONOU IS ONE POLYGON AND THE CENSUS OFFERED THIRTEEN. bj.csv carries the thirteen
    arrondissement rows and they are not drawn — no boundary layer for them could be
    verified. sources/bj.md §5 has the measurement; this is §3.10's rule taken the
    conservative way for once.
    """
    from bj2013 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bj.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "commune"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "bj" / "bj_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"bj.csv communes with no polygon: {missing} -- re-run "
                         "sources/bj_geo.py, the lookup is stale")
    if df["unit"].nunique() != 77:
        raise SystemExit(f"{df['unit'].nunique()} communes, expected 77")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "bj": dict(
        name="Benin",
        source="RGPH-4 2013, the twelve departmental Principaux indicateurs, Tableau 8 "
               "(INStaD)",
        basis="self-identification, whole census population",
        view=[0.6, 6.1, 4.0, 12.6],
        gap=("1.2%, the difference between each unit's population and the ten shares InstaD "
             "publishes"),
        gap_share=0.01207,
        note_public=(
            "**Benin is the only country on this map whose census counts Vodun by name.** "
            "Everywhere else in Africa a form offers one *Traditionalist* box against a "
            "column of named churches; Benin offers *Vodoun* and *other traditional* as "
            "two separate answers, and they turn out to be two different religions in two "
            "different halves of the country. Vodun was banned under the Marxist government "
            "of the 1970s and recognised outright in 1996, and 10 January is a national "
            "holiday for it — which is why the question can be asked here at all. "
            "**And the Vodun heartland is not where you would look for it.** The five "
            "highest communes are Djakotomey at 69%, Toviklin 66%, Lalo 56%, Aplahoué 55% "
            "and Klouékanmè 51% — the whole of the Couffo, which is Adja country in the "
            "south-west. Abomey, capital of the kingdom of Dahomey and the name in every "
            "history of the religion, is 24%; Ouidah, the other famous name, sits in a "
            "department at 12%. "
            "**The north-west is a different traditional religion entirely.** *Autres "
            "traditionnelles* is 54% in Boukoumbé, 42% in Cobly and 37% in Tanguiéta, all "
            "in the Atacora highlands, where Vodun is 6%. These are the traditions of the "
            "Bètammaribè and their neighbours, the people whose fortified *tata* houses "
            "are the region's landmark. "
            "**Read the traditional figures as floors, and one of them for a specific "
            "reason.** The boxes are exclusive of *Catholique* and *Islam*, and in Benin "
            "the same person is very commonly both. Beyond that, *no religion* is 5.8% "
            "nationally and 45% in Toucountouna, 27% in Kérou and 20% in Cobly — the same "
            "Atacora communes that lead on traditional religion, and not Cotonou, where a "
            "secularising population would show. Some of that answer is very probably "
            "practice with no church and no name on the form. It is drawn as the census "
            "published it. "
            "**The Celestial Church of Christ has a cell of its own and is 6.8% of "
            "Benin** — an African church founded in Porto-Novo in 1947, still centred "
            "exactly where it started: Sô-Ava 30%, Akpro-Missérété 26%, Bonou 24%, the "
            "Ouémé valley and the lagoons. No other source on this map counts a single "
            "African Instituted Church at that size. "
            "**The north is Muslim and the boundary is sharp.** Karimama is 95%, Malanville "
            "94%, Ségbana 92% — the Niger valley — against 0.3% in Djakotomey, a "
            "three-hundred-fold range across 77 communes."),
        how="census, 2013",
        grain="communes, 130,000 people on average",
        counts=_bj_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bj" / "bj_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bj_place_weight,
        note="THE SOURCE PRINTS SHARES AND THE COUNTS ARE ARITHMETIC, NOT AN ESTIMATE. "
             "Tableau 8 gives ten religion percentages per commune to one decimal and "
             "Tableau 2 of the same booklet gives the commune's population, so every count "
             "here is a published share times a published total — no join to a second "
             "document, nothing carried from a coarser level, nothing fitted. The rows are "
             "`measured` and the caveat is precision: one decimal is ±0.05% of a unit, or "
             "±34 people in a 68,000-person commune. "
             "sources.md §11p recorded Benin as needing commune totals from the Résultats "
             "définitifs; it does not, and each booklet is self-contained. "
             "THE TEN SHARES SUM TO 98.81% AND THE REMAINDER IS NON-RESPONSE. 120,826 "
             "people, 1.21%, computed as the complement and not drawn (§3.5). RGPH-4's "
             "religion tabulations carry exactly ten categories wherever they appear, and "
             "ten values rounded to 0.1pp have a standard error of 0.09pp against an "
             "observed 0.6-2.0pp per commune, so the gap is a category rather than the "
             "rounding. The drawn population is 9,887,923. "
             "COTONOU IS ONE POLYGON AND THE CENSUS OFFERED THIRTEEN. The Littoral booklet "
             "publishes religion for Cotonou's 13 arrondissements — 679,012 people, 6.8% "
             "of Benin — and they are parsed, checked against Cotonou's own row, and NOT "
             "drawn: COD ships no ADM3 for Benin, and geoBoundaries' OpenStreetMap "
             "arrondissements could not be verified against the census's own arrondissement "
             "populations (ratio band 0.64-1.98, r=0.81). §3.10's rule taken the "
             "conservative way; sources/bj.md §5 has the measurement. "
             "The dots are spread across 75,951 Kontur 400m hexagons weighted by hex "
             "population (sources/bj_grid.py). Benin needs that for empty northern "
             "communes AND for water: the lagoons are inside the communes rather than cut "
             "out of them, and Sô-Ava contains Ganvié, a town of ~30,000 built on stilts "
             "over Lac Nokoué. The grid finds Ganvié because it has buildings, and finds "
             "nothing on the open lake — which is also why Sô-Ava and Aguégués are the two "
             "communes Kontur models worst (0.32× and 0.30×). "
             "THE FIRST NAMED AFRICAN TRADITION ON THIS MAP. `indigenous.african.vodun` was "
             "added for Benin; `Autres traditionnelles` stays on the parent node because no "
             "source names the Atacora traditions individually yet.",
    ),
}
