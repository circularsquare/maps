# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _np_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    753 local levels, fine in PEOPLE (38,400 each) and wild in AREA, because Nepal's federal
    map was drawn to equalise population across the Terai, the middle hills and the
    Himalaya at once: Chandragiri is ~90,000 people in ~50 km2 and Namkha in Humla is ~2,500
    in 2,290 km2. §8.2's trick — fine units make a population layer unnecessary — is about
    units that are fine in AREA, and these are not. An equal share would spread the northern
    units' dots evenly over glaciers and ridge lines, and the north is exactly where the Bon
    and the highest Buddhist shares are (sources/np_grid.py).
    """
    return _kontur_place_weight(place, "np_hexes.gpkg", "sources/np_grid.py")


def _np_counts():
    """NSO NPHC 2021 religion Table 1 at local level: 10 categories on 753 local levels.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    AN EXACT PARTITION. The ten categories sum to the row total on all 918 rows of the
    source and there is no `Other`, no `Not stated` and no residual anywhere in the table,
    so every person in a drawn unit is in a named category. The only thing dropped here is
    the universe row.

    THE 0.82% THAT IS NOT DRAWN IS THE INSTITUTIONAL POPULATION AND IT IS A §3.7 CASE.
    NSO tabulates 239,098 people — barracks, prisons, hospitals, hostels, and Nepal's
    monasteries and gompas — at DISTRICT level only, as a row beside the district's local
    levels rather than inside them. There is no finer geography for it in the source, and
    spreading it across a district's local levels would invent one: the institutional
    population is concentrated in specific places by its nature, so a population-weighted
    spread would be actively wrong rather than merely uncertain. Dropped and stated on the
    map instead, per §3.5 — the `gap=` line below is where the reader learns it, and §3.7's
    point that this is exactly the population R3 most wants to see stands.
    """
    from np2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "np.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "local"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "np" / "np_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"np.csv local levels with no polygon: {missing[:8]} -- re-run "
                         "sources/np_geo.py, the lookup is stale")
    if df["unit"].nunique() != 753:
        raise SystemExit(f"{df['unit'].nunique()} local levels, expected 753")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "np": dict(
        name="Nepal",
        source="National Population and Housing Census 2021 (NPHC 2078), religion Table 1 "
               "(National Statistics Office)",
        basis="self-identification, whole census population",
        view=[80.0, 26.3, 88.3, 30.5],
        gap_share=0.008,
        gap="the institutional population, 0.8%, which is published by district only",
        note_public=(
            "**Nepal asks about ten religions and three of them are drawn on no other map "
            "here.** `Kirat`, `Prakriti` and `Bon` are boxes on the census form, not "
            "write-ins recovered from a residual, and between them they are **1.09 million "
            "people**. "
            "**Kirat Mundhum is the sharpest thing on this map's Asian half.** It is the "
            "religion of the Limbu, Rai, Yakkha and Sunuwar of the eastern hills — an oral "
            "scripture, the *Mundhum*, recited by *phedangma* priests — and it is 924,204 "
            "people, 3.17% of Nepal. Koshi province is **16.8%** Kirat and Sudurpashchim, "
            "at the other end of the country, is 0.01%. Panchthar district is **55.7%**, "
            "Taplejung 44.2%, and Mahakulung in Solukhumbu reaches **87.3%**. The edge "
            "against the Hindu middle hills is abrupt rather than gradual, and it is a "
            "border that has been moving: the Kirat count has risen at every census since "
            "1991, as a revival movement asserts a distinct identity against being "
            "recorded as Hindu. "
            "**Bon is not where you would expect it.** 67,223 people, and the obvious "
            "guess is the trans-Himalayan north — Mustang and Dolpa, where the Yungdrung "
            "Bon monasteries are. The census puts most of it in **Gandaki's middle hills**: "
            "Manang 6.1%, Gorkha 5.7%, Lamjung 4.7%, and Dharche in Gorkha at 32.7%. That "
            "is **Gurung (Tamu) country**, and the likeliest reading is that most of this "
            "cell is the Tamu shamanic tradition — the *pye-ta lhu-ta*, with its *pachyu* "
            "and *klepri* priests — which Gurungs describe as Bon and which is related to, "
            "but not the same institution as, the monastic Bon of Dolpa. One box, two "
            "things, and the map cannot separate them. "
            "**`Prakriti` is the Nepali word for nature, offered as a religion box.** "
            "102,048 people, and it is one region rather than a scatter of odd answers: "
            "Rukum East 16.6%, Rolpa 8.2%, Thawang 45.0% — the Kham Magar hills of the "
            "mid-west. NSO publishes no gloss on what it covers, so what the map can say "
            "is that these people answered the question and did not answer it with any of "
            "the world religions on the form. "
            "**The Muslim Terai is continuous with India across the border**, and both "
            "sides are now drawn. Rautahat is 22.6%, Banke 18.7%, Kapilbastu 18.2% — "
            "against 0.00% in Bajhang and under 0.3% across the whole far west. "
            "**Christianity is the fastest-growing answer in Nepal** — under 0.5% in 2001, "
            "1.4% in 2011, 1.76% now — and it is not in the capital or the Terai but the "
            "central hills, among the same Tamang and Magar communities the Kirat and "
            "Prakriti boxes draw from: Dhading 7.6%, Makwanpur 6.1%, Gorkha 6.0%. **Read "
            "that number as a floor.** Nepal's 2017 penal code criminalises conversion and "
            "'hurting religious sentiment', and people have been prosecuted under it; a "
            "census answer given in that setting undercounts rather than over. "
            "**The five world religions arrive undivided.** One Hindu box for 23.7 million "
            "people — the second-largest Hindu population on earth — with no sampradaya, no "
            "caste tradition and no sect; one Buddhist box covering Tibetan Vajrayana, the "
            "Newar Vajrayana of the Kathmandu valley that exists nowhere else, and a "
            "twentieth-century Theravada revival; one Muslim box; one Christian box naming "
            "no church. Nothing here splits them, because the census does not."),
        how="census, 2021",
        grain="753 local levels (palikas), 38,000 people on average",
        counts=_np_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "np" / "np_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_np_place_weight,
        note="THE FINEST COUNTING GEOGRAPHY OF ANY LARGE ASIAN COUNTRY HERE. 753 local "
             "levels at ~38,400 people each, against India's 5,988 sub-districts at "
             "~202,000 and Bangladesh's 544 upazilas at ~306,000 — Nepal counts religion "
             "roughly five times finer than either neighbour, on a table that is one 270 KB "
             "spreadsheet. "
             "AN EXACT PARTITION, WITH NO RESIDUAL AND NO `NOT STATED`. The ten categories "
             "sum to the row total on all 918 rows of the source, and the local levels, the "
             "districts and the provinces each sum to 29,164,578. Nothing is suppressed and "
             "nothing rounded, so 100% of the census population is in a named category — "
             "the fourth source here of which that is true after Zimbabwe, Malawi and "
             "Guyana, and by far the largest. "
             "THE 0.82% NOT DRAWN IS THE INSTITUTIONAL POPULATION, and it is §3.7's case "
             "exactly. NSO gives 239,098 people — barracks, prisons, hospitals, hostels, "
             "and Nepal's monasteries and gompas — one row per district and no finer "
             "geography at all. Spreading them across a district's local levels would "
             "invent a location, and this is a population that is concentrated by its "
             "nature, so the invented spread would be actively wrong rather than merely "
             "uncertain. Dropped, and said on the map in `gap`. "
             "THE JOIN IS ON NAMES AND THE BOUNDARY FILE CARVES OUT 22 NATIONAL PARKS. "
             "COD-AB has 775 ADM3 polygons against the census's 753 local levels, and the "
             "difference is Chitawan, Parsa, Bardiya, Khaptad, Langtang, Shivapuri, "
             "Shuklaphanta, Koshi Tappu and Dhorpatan, several of them split across "
             "districts. The p-code's unit-type digit separates them and reproduces Nepal's "
             "official 6 metropolitan / 11 sub-metropolitan / 276 municipality / 460 rural "
             "municipality composition exactly, which is the evidence that it means what it "
             "looks like — and it has to be done BEFORE the name join, because four parks "
             "share a name with a palika in the same district. Nobody is attributed to a "
             "park and no dot lands in one. "
             "751 OF 753 NAMES MATCH BY A DERIVED FOLD, not by a hard-coded alias list: "
             "strip the unit-type word (`Gaunpalika`, `Nagarpalika`, `Municipality`, and "
             "NSO's own `Metropolitian City`) and a leading district name, then match "
             "inside the district. The last one, `Melanchi` against COD's `Melamchi`, "
             "falls through to a unique edit-distance-1 match among that district's "
             "unclaimed polygons. "
             "PLACEMENT IS KONTUR'S 400 m GRID, 104,129 hexes, and Nepal needs it more than "
             "its unit count suggests: the local levels are fine in people and wild in "
             "area, from ~50 km² in the Kathmandu valley to 2,290 km² in Humla. Ten of the "
             "753 sit outside a factor of three against the census, against a shuffled "
             "median of 224, and log populations correlate at r = 0.9055 against a best "
             "shuffle of 0.1350. The ten are two different things — a town smeared into its "
             "hinterland (Rohini 4.09 beside Siddharthanagar 0.26, and pooling them gives "
             "1.31) and a block of the Parsa Terai that Kontur simply over-models and that "
             "does not pool away. Neither changes a count; the grid is a within-unit weight "
             "(§8.2). "
             "READ A CLUSTER AS COMPOSITION AT PALIKA SCALE. A local level averages 38,400 "
             "people over ~190 km², so a dot says 'this palika, drawn where Nepalis live' "
             "and nothing about which ward or village.",
    ),
}
