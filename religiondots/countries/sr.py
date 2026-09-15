# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "sr_hexes.gpkg", "sources/sr_grid.py")


def _sr_counts():
    """ABS Census 7 (2004) at ressort: 5 nodes on 62 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **THE FINEST GEOGRAPHY IN THE AMERICAS HERE, AND THE SHALLOWEST QUESTION IN THE
    CARIBBEAN.** 62 ressorten at ~7,900 people each, and five religion categories. Both
    facts are the country: Suriname is 13.45% Muslim and 19.93% Hindu — the highest Muslim
    share this map draws in the Americas — and the grain is fine enough to show that neither
    is spread evenly (Hinduism 65% in Jarikaba, Islam 48% in Nieuw Amsterdam).

    **84.33% of Suriname is drawn.** What is not is `Don't know/No answer`, 77,204 people,
    **15.67% — the largest non-answer on this map**, which taxonomy/sr2004.py excludes per
    §3.5. Every share drawn here is a share of everybody, not of the people who answered.
    """
    from sr2004 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sr.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "ressort"].copy()
    if df["geo_id"].nunique() != 62:
        raise SystemExit(f"{df['geo_id'].nunique()} ressorten, expected 62 -- re-run "
                         "sources/sr.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Don't know/No answer", "Total"})
    if unmapped:
        raise SystemExit(f"sr.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "sr": dict(
        name="Suriname",
        source="Census 7 (2004) — Algemeen Bureau voor de Statistiek",
        basis="self-identification",
        view=[-58.15, 1.80, -53.90, 6.10],
        gap="15.7%, whom ABS pools into a single do not know or no answer cell",
        gap_share=0.1567,
        note_public=(
            "**The most Muslim country in the Americas, and it is not close.** 13.5% of "
            "Suriname is Muslim and 19.9% is Hindu — the legacy of Javanese and Indian "
            "indenture under Dutch rule — and at 7,900 people per unit the map is fine "
            "enough to show that neither is spread evenly. **Hinduism reaches 65% in "
            "Jarikaba** and 60% in the Westelijke Polders; **Islam reaches 48% in Nieuw "
            "Amsterdam** and 47% in Lelydorp. Christianity runs the other way, up to 71% in "
            "Para Zuid and 60% in Brownsweg. With Guyana and Trinidad this is the "
            "Indo-Caribbean world the rest of the hemisphere does not have. "
            "**This is 2004, and that is the newest whole-country religion table Suriname "
            "has.** The 2012 census asked religion in far more detail — it names the "
            "Moravians, the Catholics, the Full Gospel churches, Sunni and Ahmadiyya "
            "Islam, Sanatan and Arya Hinduism separately — but published it **nationally "
            "only**, and its district reports reach three of ten districts. A ninth census "
            "was taken in 2024–25 and has published nothing yet. So the map trades twenty "
            "years of currency for the only geography that exists. "
            "**Christianity is one undivided colour here and it should not be.** Suriname's "
            "Christianity is Moravian, Catholic and Pentecostal in different places — the "
            "Evangelische Broedergemeente has been there since 1735 — and none of that can "
            "be drawn from this table. It is not filled in from the 2012 national figures, "
            "because that would invent where each denomination lives. "
            "**And 15.7% answered 'don't know' or nothing at all** — the largest non-answer "
            "on this map. Every share here is a share of everybody, so a religion's share "
            "among people who answered is about a fifth higher than what is drawn. "
            "**What is missing has a name.** *Winti*, the Afro-Surinamese religion of the "
            "Maroon and Creole populations and the sibling of Vodou and Candomblé, has no "
            "cell of its own: it sits inside a combined `Traditional religion and other` "
            "category with indigenous Amerindian religion, Judaism and the Jehovah's "
            "Witnesses. It was **illegal in Suriname until 1971** and is widely practised "
            "alongside a church, so a one-answer census undercounts it twice over."),
        how="census, 2004; the 2012 and 2024 censuses publish religion nationally only",
        grain="ressorten, 7,900 people on average",
        counts=_sr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sr" / "sr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sr_place_weight,
        note="**§11t CALLED THE VINTAGE A FORK AND IT IS NOT ONE.** The choice looked like "
             "fine-geography-2004 against deep-categories-2012; it was checked and 2012 has "
             "no usable geography at all. Volume 1 carries the full denominational list "
             "cut by ethnicity and nationality and by **no geography**; the "
             "Districtsresultaten presentations carry religion for **3 of 10 districts** "
             "(Volume III only) — verified against the text layer, since all three volumes "
             "have one, so the absence is the source's and not a scan artefact; and Census "
             "9, fielded to July 2025, has published nothing across 3,662 media items. 2004 "
             "is the only whole-country sub-national religion table Suriname has. "
             "**THE FILE HAS NO DISTRICT COLUMN AND THE NAMES ARE NOT UNIQUE.** The ressort "
             "workbook is 62 unlabelled columns; there is a `Welgelegen` in both Paramaribo "
             "and Coronie and a `Centrum` in both Paramaribo and Brokopondo, so a name-only "
             "join collides on four units. The sibling `district-profiel-census.xls` "
             "publishes **ressorten per district** — 12, 7, 5, 3, 6, 6, 6, 5, 6, 6 — which "
             "sums to 62 and consumes the columns in order, which is what makes the "
             "district assignment a read rather than a guess. Asserted, not assumed. "
             "**FOUR NAMES DO NOT FOLD ONTO COD'S AND NO ALIAS TABLE IS WRITTEN.** "
             "`Koewarasan`/`Kwarasan`, `Moengo Tapoe`/`Moengo Tapu`, "
             "`Marchallkreeek`/`Marechallkreek` (ABS's typo, three e's, transcribed as "
             "printed) and `Coeroeni`/`Coeroenie`. The exact fold runs first, and then a "
             "district with exactly one unmatched census ressort and exactly one unmatched "
             "polygon has them paired **by elimination** — a derivation that re-runs every "
             "time and stops if a future vintage leaves two of either, which is the "
             "§12-safe form of what a frozen alias list does badly. 58 folded, 4 forced, "
             "62/62 both ways. "
             "**AN EXACT PARTITION IN BOTH DIRECTIONS**, integers, no rounding and no "
             "suppression: the six categories sum to each ressort's own total on all 63 "
             "columns, and the 62 ressorten sum to the national column on all 7 rows. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 5,689 hexes, and Suriname needs it more "
             "than anywhere: 163,820 km² with ~90% of the people on the coastal strip, and "
             "three interior Sipaliwini ressorten together larger than the Netherlands. "
             "The national ratio is **1.274x**, which is expected rather than alarming — a "
             "2023 grid against a **2004** census, the widest vintage gap on this map — so "
             "the band is asserted at [1.00, 1.60] rather than around 1.0. "
             "**NAME WHERE IT IS WORST.** The per-ressort ratio runs 0.06x to 2.86x, which "
             "is wide because nineteen years of Suriname's growth went to Wanica and "
             "Paramaribo. **Galibi is the worst-placed unit on this country** — 4 hexes and "
             "43 modelled people against a real census population — so its dots sit on "
             "essentially no weighting at all. Galibi is the Kalina (Carib) village area at "
             "the Marowijne mouth, and a building-footprint model reads it as empty. The "
             "dots are still inside Galibi and still the right number (§9t); only the "
             "surface under them is bad.",
    ),
}
