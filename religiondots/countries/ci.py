# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _CiHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Côte d'Ivoire's 33 régions. The 10 largest are 47.6% of the
    country's land and 28.9% of its people; Bounkani alone is 21,800 km² holding 427,037,
    against Abidjan's 2,153 km² holding 6.32 million. An equal share per polygon would
    smear the north's dots across the empty Comoé park — and Bounkani is exactly where
    `Animiste` is 24.7%, so a uniform fill would blur the one category this country shows
    most sharply. sources/ci_geo.py has the numbers.

    A POPULATION weight, not a religion one — nothing measures where a région's Harrists
    sit inside it, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a région's hexes sum to zero "
                f"(sources/ci_geo.py)")


def _ci_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ci_hexes.gpkg has no `pop` column — run sources/ci_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _CiHexWeighter(place)


# sources/ci.py §7a emits these two in place of `Autres religions chrétiennes`. Their
# national magnitudes are ANStat's; their per-région split is a uniform national ratio.
_CI_DERIVED = {"Évangélique", "Autres chrétiens, hors évangéliques"}


def _ci_counts():
    """Côte d'Ivoire RGPH 2021 at région: 8 nodes on 33 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **97.79% of the ordinary-household population is drawn.** The missing 2.21% is `ND`
    (`Non déclaré`), a non-answer taken off the tree per §3.5 as Kenya's `Not Stated` is
    (taxonomy/ci2021.py). It is NOT `Sans religion`, which is a published cell 5.7x larger
    and is drawn.

    The counts in ci.csv are already the product of Tableau 4.6's percentages, the annex's
    régional populations and Tableau 4.1's national magnitudes — see sources/ci.py, which
    rescales each category so the 33 units sum to its published national count.
    """
    from ci2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ci.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 33:
        raise SystemExit(f"{df['geo_id'].nunique()} régions, expected 33 -- re-run "
                         "sources/ci.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0

    # §7a's évangélique split is DERIVED: the national magnitude is ANStat's, the per-région
    # distribution is a uniform national ratio and is not established by any source. So the
    # two halves are `tier="derived"` and may never ring (§3.10) — a ring asserts presence
    # in a unit, and nothing here establishes presence unit by unit. Everything else in the
    # file is a published régional percentage and stays `measured`.
    derived = df["source_category"].isin(_CI_DERIVED)
    df["tier"] = derived.map({True: "derived", False: "measured"})
    df["may_ring"] = ~derived
    n_der = int(derived.sum())
    if not n_der:
        raise SystemExit("no derived rows in ci.csv -- sources/ci.py's évangélique "
                         "split did not run; re-run it")
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "ci": dict(
        name="Côte d'Ivoire",
        source="RGPH 2021, tome 1 (ANStat), Tableaux 4.1 and 4.6",
        basis="self-identification, ordinary households",
        view=[-8.7, 4.1, -2.3, 10.9],
        note_public=(
            "**Côte d'Ivoire is the only country on any map here that counts the Harrist "
            "Church.** 140,482 people follow William Wadé Harris, the Liberian preacher who "
            "walked this coast in 1913-15 in a white robe with a bamboo cross and is "
            "usually credited with more conversions than any missionary in African history. "
            "**His route is still on the map**: the church is 2.4% in La Mé, 1.7% in "
            "Grands-Ponts and 1.6% in Agnéby-Tiassa — the southern lagoons — and a printed "
            "**0.0% in seven northern régions**, which he never reached. A century later "
            "the census can still see where he walked. "
            "**The country divides north and south almost perfectly.** Islam is 42.5% "
            "nationally and runs **95.7% in Folon and 92.3% in Kabadougou** on the Malian "
            "and Guinean border, against 13.9% in N'Zi in the centre-east — a sevenfold "
            "range. Christianity is 40.3% and does the opposite. There is no gradient in "
            "the middle so much as a line. "
            "**Traditional religion survives in one place and it is Bounkani.** 24.7% "
            "there against 2.2% nationally — Lobi and Koulango country in the north-east "
            "corner, on the Burkinabè and Ghanaian border. Every other région is under 8%. "
            "Read the figure as a floor everywhere: the box is exclusive of the Christian "
            "and Muslim ones, and in Côte d'Ivoire the same person is very often both. "
            "**And `no religion` is 12.6% but it is not what it looks like.** It is 29.8% "
            "in Tonkpi and 28.9% in Poro against **3.6% in Abidjan** — the exact inverse of "
            "where a secularising urban population would be. Some unknown part of it is "
            "traditional practice with no church and no box on the form. "
            "**Nearly one in five Ivorians is an evangelical, and that is the fastest "
            "change here.** *Autres chrétiens* went from 3.1% of the country in 1998 to "
            "20.0% in 2021 — twelvefold in a generation — while animists fell from 11.9% "
            "to 2.2% and the Harrist church shrank in absolute numbers, from 197,515 "
            "people to 140,482. The evangelical wave is largely where the animists went. "
            "**The evangelical share is the census's own figure, but its map is not.** "
            "ANStat publishes évangéliques at 18.6% nationally and gives them no geography "
            "at all, so the dots here are that national share applied evenly to every "
            "région's *other Christian* total. The size is measured; **the pattern is "
            "not** — real evangelicals are very likely more southern and more urban than "
            "this shows. "
            "**And what is left of that cell still hides a church Benin counts by name.** "
            "After the evangelicals are taken out, 559,000 people remain in *other "
            "Christian*, and the census collected *Céleste*, *Bouddhiste* and *Témoin de "
            "Jéhovah* as answers it never printed. The Celestial Church of Christ is in "
            "there, mapped commune by commune in Benin next door and invisible here."),
        how="census, 2021",
        fill="from the same census's national total",
        grain="régions, 887,000 people on average",
        # `gap` (§6.12): `ND` is 2.21% and is a non-answer, off the tree per §3.5.
        gap_share=0.022,
        gap="2.2% who did not state a religion",
        counts=_ci_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ci" / "ci_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ci_place_weight,
        note="THE OFFICE WAS NEVER THE ROUTE. `ins.ci` is a parked cPanel page that 404s "
             "every document it used to serve; the office is now ANStat and `anstat.ci` "
             "returns 403 to every scripted request, static PDFs included. The file is the "
             "WAYBACK MACHINE's copy of ANStat's own URL — §11f's technique applied to a "
             "live host rather than a dead one. "
             "AND THE ORACLE WAS TWO CENSUSES STALE: UNSD table 28 lists Côte d'Ivoire at "
             "2014 with 22.7M people; this is the RGPH 2021 at 29.4M. "
             "THE ARCHIVE TRUNCATES, AND THE OBVIOUS CAPTURE IS THE BROKEN ONE. The "
             "2021-04 captures deliver exactly 1,048,576 bytes (2^20) with no %%EOF and "
             "PyMuPDF opens them anyway, reporting a plausible page count. sources/ci.py "
             "queries the CDX for every capture, takes the largest, and asserts the "
             "trailer. "
             "THREE TABLES, ONE VOLUME — §3.4's move. Tableau 4.6 is religion by "
             "district/région in PERCENTAGES to one decimal; Tableau 4.1 is the same nine "
             "categories nationally in COUNTS; the annex is population by région in counts. "
             "Shares from 4.6, denominators from the annex, magnitudes from 4.1, then each "
             "category rescaled so the 33 units sum to its published national count — "
             "which removes the one-decimal rounding from every national figure and leaves "
             "it only in the within-country distribution. The rescales run 0.980–1.025. "
             "TWO NESTED TIERS IN ONE COLUMN, which is Serbia's §9p in a third country: "
             "Tableau 4.6 interleaves the 14 districts with their régions and marks "
             "neither, so summing the column double-counts the country. The separator is "
             "not the layout — it is the population annex, which lists régions and the two "
             "autonomous districts and no other district. That lands on exactly 33 and is "
             "self-checking. "
             "AND `Lacs` IS PRINTED TWICE with different figures; the second is Lagunes, "
             "identifiable because its children in the table are Agnéby-Tiassa, "
             "Grands-Ponts and La Mé. Both are districts so neither is drawn, but ci.py "
             "asserts the count is still two. "
             "THE SAME DOCUMENT USES TWO THOUSANDS SEPARATORS: Tableau 4.1 groups digits "
             "with U+2009 THIN SPACE and the annex with an ordinary space, so a regex "
             "written against one silently matches NOTHING on the other and the failure "
             "reads as 'that table is not on this page'. Every line is folded before any "
             "pattern is tried. "
             "Boundaries are geoBoundaries CIV ADM2, 33 polygons against 33 units, joined "
             "by name: 30 fold directly and 3 need an alias (District Autonome d'Abidjan, "
             "District Autonome de Yamoussoukro, and `Me` for `La Mé`), each unambiguous. "
             "Placement is Kontur's H3 grid, 142,653 hexes weighted by hex population. "
             "**And unlike the Central African Republic (§9av), Kontur here IS independent "
             "of the census**: 27.3% of régions sit within ±5% of the median ratio against "
             "CAR's 78.5%, which is what a 2021 census against a 2023 grid should look "
             "like. ci_geo.py measures it on every run. "
             "THE ÉVANGÉLIQUE SPLIT IS DERIVED AND IS ANITA'S CALL (2026-09-07). Tome 1 "
             "never divides `Autres religions chrétiennes` (6,004,781, 20.5%); the "
             "RÉSULTATS GLOBAUX DÉFINITIFS does, in one sentence of prose — \"20% d'autres "
             "chrétiens, composés principalement des évangéliques (18,6%)\" — whose "
             "percentages are of the total population and reproduce that publication's own "
             "% column exactly. So 5,445,459 évangéliques is the SOURCE'S magnitude. The "
             "geography is nobody's: no publication gives them by région, so the national "
             "ratio is applied uniformly and both halves inherit the residual's shape. "
             "Those rows are `tier=derived` and may never ring (§3.10). "
             "AND THE TWO PUBLICATIONS DISAGREE about where `autres chrétiens` ends and "
             "`autres religions` begins — by 159,208 people, while agreeing on their sum "
             "to the person (6,057,832 both ways). Subtracting the évangéliques from tome "
             "1's larger cell leaves the disputed people in the REMAINDER, which is where "
             "they belong if the Résultats Globaux is right that they are not Christian, "
             "so the évangélique figure is unaffected either way.",
    ),
}
