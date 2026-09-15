# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _IdHexWeighter:
    """Split a drawn unit's dots across Kontur 400 m hexagons by hex POPULATION.

    Indonesia's placement problem is not Kenya's and shows up somewhere else. Most of the
    country is drawn at kecamatan, small enough that an equal share is honest. What is not
    honest is (a) the 89 regencies drawn whole, several of them enormous and nearly empty
    Papuan and Kalimantan units, which is Kenya's failure exactly; and (b) the DENSE URBAN
    kecamatan, which is the one you notice — Cengkareng is 513,920 people and Cakung
    503,846, and an even wash across each polygon makes a city read as flat-shaded tiles
    with administrative edges instead of a built-up area with a shape.

    THE WEIGHT IS POPULATION, NEVER RELIGION, and the distinction is the whole of §14.4
    here. BPS publishes religion at kecamatan and nothing below it, so a Muslim dot and a
    Buddhist dot inside one kecamatan are spread identically. The map gets better at saying
    WHERE THE PEOPLE ARE and no better at all at saying who they are: the street-level
    sorting of, say, Kelapa Gading stays invisible, and weighting religions differently
    inside a unit would be inventing a magnitude the source does not publish.
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
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a unit's cells sum to zero "
                f"(sources/id_grid.py)")


def _id_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! id_hexes.gpkg has no `pop` column — run sources/id_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _IdHexWeighter(place)


def _id_counts():
    """BPS Sensus Penduduk 2010: 7 drawn categories on 5,211 units, 237.1M people.

    ONE level in effect and nothing modelled, but **the drawn tier is decided PER UNIT
    rather than by rule**, which is Ghana's shape (§9n) taken one step further. A regency's
    kecamatan REPLACE it where they sum to it exactly in every category — 403 of 492 — and
    the regency is drawn where they do not. So the drawn set is
    `kecamatan` + `regency` + one `province_residual` = 5,122 + 89 + 1 = 5,212, it is
    disjoint, it covers the country once, and **every row is `measured`** and may ring.

    THE ONE `province_residual` IS KALIMANTAN UTARA. Its five regencies became a province in
    2012 and BPS serves them under neither — wid=25 is empty and their regency slots are
    holes — so they were briefly written off as an unrecoverable 0.22% and left a visible
    hole in northern Borneo. They are recoverable without any other source: Kalimantan
    Timur's row is the 2010 province and its listing has only the nine that remain, so the
    per-category residual IS those five. 524,656 people, summing to the published total
    exactly and non-negative in every category. They were carved wholly out of one province
    and are contiguous, so the residual has a shape and is drawn as one unit over it.

    THE TWO LEVELS THAT MUST NOT BE DRAWN ARE IN THE SAME FILE. `regency_covered` is a
    regency its kecamatan already cover, and `kecamatan_partial` are the kecamatan of a
    regency whose listing is short. Either one added to the drawn tier double counts, and
    `kecamatan_partial` on its own would put a whole unit's population into part of it —
    88 regencies have an incomplete sub-district listing and the worst, Kolaka, carries
    82,726 of its true 255,712 people. sources/id.md §7.

    AND THE COMPLETENESS TEST IS PER CATEGORY, NOT ON THE TOTAL. Nduga (9429) publishes
    eight kecamatan carrying a `Total` row and no religion categories at all: the totals
    reconcile exactly, so a total-only test would promote it and draw 79,053 Kristen as
    79,053 people with no religion. It is drawn as a regency instead.

    THREE CATEGORIES RESOLVE TO NOTHING: the universe total, `Tidak Terjawab` (not
    answered, 139,128) and `Tidak Ditanyakan` (not asked, 754,485) — two non-response
    categories that mean opposite things and are kept apart (§3.5, §9p). So the drawn
    population is 236,223,057 of the tier's 237,116,670, itself 99.78% of SP2010's
    237,641,326; the missing 0.22% is the five regencies that became Kalimantan Utara in
    2012, which BPS serves under neither province.
    """
    from id2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "id.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    # `province_residual` is the third drawn level and there is exactly one of it:
    # Kalimantan Utara, recovered as Kalimantan Timur's per-category residual because BPS
    # serves its five regencies under neither province. Leaving it out is a 524,656-person
    # hole in the north of Borneo that nothing errors about.
    df = df[df["geo_level"].isin(["kecamatan", "regency", "province_residual"])].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "id" / "id_drawn_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"{len(missing)} id.csv units have no polygon ({missing[:3]}) -- "
                         "re-run sources/id_geo.py, the lookup is stale")
    if df["unit"].nunique() != 5212:
        raise SystemExit(f"{df['unit'].nunique()} drawn units, expected 5,212")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "id": dict(
        name="Indonesia",
        source="Sensus Penduduk 2010 (Badan Pusat Statistik)",
        basis="self-identification, one of the six recognised religions",
        view=[94.9, -11.1, 141.1, 6.1],
        gap=("0.4%, most of them people the question never reached rather than people who "
             "declined it"),
        gap_share=0.003769,
        note_public=(
            "**The largest Muslim population on earth, and the whole of the interesting "
            "map is in the other 12.8%.** Indonesia is 87.2% Muslim and the Muslim regions "
            "are as near-total as anything drawn here — Aceh Timur is 99.91%, Aceh Utara "
            "99.90%, Lombok Timur 99.88% across 1.1 million people. What makes the country "
            "worth drawing is that its minorities are not scattered: almost every one of "
            "them has a homeland, and at sub-district resolution you can see the edges. "
            "**Four of them, each on its own island group.** The Papuan highlands are "
            "Protestant to a degree almost nothing else on this map matches — Nduga is "
            "100.0% and Lanny Jaya 99.84% — and the Christian belt runs east through "
            "Maluku. **Flores and Timor are Catholic**: Manggarai 94.7%, Ngada 91.4%, "
            "Sikka 88.0%, a Portuguese inheritance that stops at the Flores Sea. **Bali is "
            "Hindu** at 98.6% in Bangli and 95.2% in Gianyar — but only 76.4% in Badung, "
            "which is Kuta and the airport, and that gap is the clearest thing on this map "
            "that migration does to a religious geography. **And the Buddhists are Chinese "
            "Indonesian and coastal**: Singkawang in West Kalimantan is 29.7%, Tanjung "
            "Pinang 12.9%, Pontianak 12.0%, Medan 8.8%. "
            "**Confucianism is 116,916 people and it is one archipelago.** Bangka is "
            "5.67% Khong Hu Chu against a national 0.049% — a hundredfold — and the whole "
            "province of Bangka-Belitung is 3.25%. The number is also a political artefact: "
            "recognition was withdrawn under the New Order in 1979 and restored in 2000, so "
            "**2010 is the first Indonesian census that counts Confucians at all**, and the "
            "community is generally reckoned far larger than this. "
            "**The most interesting cell is the one labelled 'other', and it is a floor.** "
            "`Lainnya` is 0.13% nationally and lands exactly where Indonesia's indigenous "
            "religions are: Katingan 19.9%, Gunung Mas 16.8% and Murung Raya 15.2% in "
            "Central Kalimantan, which is **Kaharingan**, the Dayak religion; and Sumba "
            "Barat 19.4%, Sumba Timur 13.1%, Sabu Raijua 13.7%, which is **Marapu**. Both "
            "are far bigger than those numbers. In 2010 the belief systems collectively "
            "called Aliran Kepercayaan had no standing on the census form — registration "
            "came only with a 2017 Constitutional Court ruling — so adherents recorded one "
            "of the six recognised religions instead, and Kaharingan was administratively "
            "counted as Hinduism outright. That is most of why interior Kalimantan draws "
            "Hindu at all. "
            "**And there is no box for having no religion.** The census asks which of six "
            "religions you belong to, so Indonesia draws with an entirely empty irreligious "
            "population — not measured at zero, never offered. Every share on this map "
            "should be read as an answer to that question and not as a statement of "
            "belief."),
        how="census, 2010, six permitted answers",
        grain="sub-districts, 46,000 people each (regencies for 89 of 492)",
        counts=_id_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "id" / "id_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_id_place_weight,
        note="THE DRAWN TIER IS DECIDED PER UNIT, WHICH NO OTHER COUNTRY HERE DOES. BPS "
             "publishes religion at kecamatan (sub-district) as well as at kabupaten/kota, "
             "but 88 of the 492 regencies have an incomplete sub-district listing — no gap, "
             "no marker, no error, and the worst of them carries 82,726 of its true 255,712 "
             "people. So a regency's kecamatan REPLACE it only where they sum to it exactly "
             "in every category (403 of 492), and the regency is drawn where they do not. "
             "The result is 5,122 kecamatan plus 89 regencies = 5,211 units, disjoint, and "
             "every one of them measured rather than allocated. Ghana's two-geo_level shape "
             "(sources.md §9n), decided by arithmetic instead of by rule. "
             "AND THE COMPLETENESS TEST IS PER CATEGORY. Nduga in Papua publishes eight "
             "kecamatan carrying a population total and no religion categories at all, so "
             "the totals reconcile perfectly while the religion does not; a total-only test "
             "would have drawn 79,053 Kristen as 79,053 people with no religion. "
             "COVERAGE IS 99.78% OF SP2010. The missing 0.22% is the five regencies that "
             "became Kalimantan Utara in 2012: BPS serves a 2010 census through a "
             "post-2012 geography, and those units are under neither province. "
             "PLACEMENT IS UNIFORM WITHIN EACH DRAWN UNIT and no finer layer is used, which "
             "is honest at kecamatan scale and coarse for the 89 regencies — several of "
             "them are large and thinly populated Papuan and Kalimantan units where a "
             "population grid would place the dots much better. That is the obvious next "
             "improvement. "
             "ON §14: this is exactly the resolution BPS itself publishes, so the map makes "
             "no claim finer than the state's own. Note that the shallow category list is a "
             "fact about Indonesian law rather than about Indonesian religion — six "
             "recognised religions, one cell for everything else — and sources/id.md §5 "
             "sets out what that hides.",
    ),
}
