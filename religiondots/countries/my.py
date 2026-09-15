# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _MyHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Malaysia's administrative districts.

    Malaysia needs it for the Ethiopia reason rather than the Bangladesh one, and the
    units that need it most are the ones the country is worth drawing for:

      * **interior Sarawak.** Belaga is 16,196 km² with 22,502 people — 1.4 per km²,
        against Malaysia's national 98 — and Bukit Mabong, Kapit and Song are the same
        shape. They are also 86-90% Christian, with everyone living along the Rajang
        and its tributaries and nobody at all on the ridges between. Spread uniformly,
        the most distinctive religious geography in the country washes evenly across
        empty rainforest.
      * **interior Sabah and the peninsular highlands**, which hold the Orang Asli
        districts where `other.my` and the misleading `unaffiliated` cell concentrate
        (my2020.py). Cameron Highlands, Gua Musang, Lipis and Hulu Perak are large,
        mountainous and mostly empty, and what is in them sits in a few valleys.

    A POPULATION weight, not a religion one. Nothing measures where Belaga's Christians
    sit inside Belaga, so every node's dots are spread identically. sources/my_geo.py has
    the numbers, including the eleven districts whose Kontur/census ratio falls outside
    0.6-1.6 and why none of them is a bad polygon.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a district's hexes sum to zero "
                f"(sources/my_geo.py)")


def _my_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! my_hexes.gpkg has no `pop` column — run sources/my_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _MyHexWeighter(place)


def _my_counts():
    """Malaysia 2020 census at administrative district: 7 nodes on 160 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **100% OF THE PUBLISHED TABULATION IS DRAWN, and it reconciles three ways.** Within
    each of the sixteen state volumes the seven categories sum to the state total and the
    districts sum to the state total; across volumes the sixteen states sum, category by
    category, to the separately published national Table 6 — a different publication, so
    the check is external rather than the file agreeing with itself. The grand total is
    32,447,385, DOSM's census population.

    my.csv also carries the country and state rows, which are the same people twice more;
    only `district` is read.
    """
    from my2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "my.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 160:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 160 -- re-run "
                         "sources/my.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"unmapped Malaysian categories: {unmapped}")
    df = df[df["count"] > 0]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "my": dict(
        name="Malaysia",
        source="Banci Penduduk dan Perumahan Malaysia 2020, Jadual 7 (DOSM)",
        basis="self-identification",
        view=[99.3, 0.5, 119.5, 7.6],
        note_public=(
            "**Malaysia holds a wider religious range inside one border than any other "
            "country on this map.** Terengganu is 97.3% Muslim and Kelantan 95.5%; "
            "Sarawak, 700 km away across the South China Sea, is **50.1% Christian**. No "
            "other country here runs from one of those to the other. "
            "**Borneo is the reason to look.** Sarawak's interior districts are among the "
            "most Christian places on this map — **Tebedu 93.1%, Kapit 89.6%, Lubok Antu "
            "88.0%, Belaga 86.2%** — and Sabah adds Tambunan at 79.8% and Tenom at 70.4%. "
            "This is mission ground among the Iban, Bidayuh, Kadazan-Dusun and Murut, "
            "worked from the nineteenth century, and the largest Protestant body in the "
            "interior is the Sidang Injil Borneo, which appears on no other map here. The "
            "census names no denomination, so all of it — Catholic, Anglican, SIB, Basel "
            "— is inside one colour. "
            "**The peninsula is a different country religiously.** Islam is the state "
            "religion and constitutionally tied to Malay identity, and the Muslim share "
            "tracks the Malay one closely. Against it sit the Chinese and Indian "
            "communities the colonial economy brought: **Timur Laut (George Town) is 52.5% "
            "Buddhist**, Kampar 45.8%, Kinta 33.5% — the tin valleys and the Straits ports "
            "— while the Hindu share follows the rubber estates and the railway, reaching "
            "**21.5% in Bagan Datuk**, 17.3% in Port Dickson and 16.9% in Klang. "
            "**Two cells mean something other than what they say, and both matter.** "
            "*Others* is 0.9% and holds six named traditions at once — Sikh, Taoist, "
            "Confucian, Bahá'í, Chinese folk and animist — so **Chinese temple practice, "
            "which this map draws separately for China and Vietnam, cannot be separated "
            "here at all**. And *no religion*, 0.8%, is not a secular geography: it peaks "
            "at **35.9% in Kecil Lojing** and runs 13.4% in Rompin, 12.5% in Selangau and "
            "8.8% in Cameron Highlands — every one an Orang Asli or interior indigenous "
            "district, and none of them a city. Read it as indigenous practice with no box "
            "on the form rather than as irreligion. Kuala Lumpur, for comparison, is 0.9%. "
            "**And a third of the grey is not about religion at all.** *Religion unknown* "
            "is 0.9% and **97% male** — 67,664 men to 25 women in Perak alone. Malaysia "
            "counts its roughly 2.7 million non-citizens, overwhelmingly male labour in "
            "plantations, construction and factories, and the near-certain reading is "
            "workers counted for a headcount without the religion question being put. It "
            "is left undrawn as its own category rather than spread, because spreading it "
            "would invent religion for exactly those people in exactly those districts."),
        how="census, 2020",
        grain="administrative districts, 203,000 people on average",
        counts=_my_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "my" / "my_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_my_place_weight,
        note="SIXTEEN PUBLICATIONS, ONE FILE EACH, AND FINDING WHICH FILE IS THE WHOLE "
             "TRICK. DOSM publishes *Penemuan Utama Banci 2020* per state, and each "
             "state's download list holds about twenty-one files of which twenty are "
             "`MYLOCAL STATS` socioeconomic tables with no religion in them. The one that "
             "matters is `<STATE> JADUAL 1 HINGGA 16`, and in Perak's list it was record 21 "
             "of 21, alone on page 3. Sabah and Sarawak title theirs *State Sabah* and "
             "*State Sarawak* where the peninsular ones say *Negeri*, and each sits among "
             "27 and 40 per-district publications. sources.md §11s has the route; it needs "
             "a free eStatistik registration and a browser. "
             "**THE RECONCILIATION IS EXTERNAL, WHICH IS RARE HERE.** Within each state "
             "volume the seven categories sum to the state total and the districts sum to "
             "the state total; then all sixteen states sum, category by category, to the "
             "separately published national volume's Table 6. That last check is a "
             "different publication rather than the file agreeing with itself, and it "
             "passes to the person on all seven categories and all sixteen states. The "
             "standalone Kampar district volume was downloaded first and agrees with its "
             "row in the Perak state volume on all eight cells. "
             "**SARAWAK SHIPS AN EMPTY DECOY OF ITS OWN RELIGION TABLE**, and it is the "
             "trap worth naming: sheet `7` carries the correct title, headers, footnote and "
             "all forty district names in capitals — with every value cell blank. The real "
             "table is `7 (T)`. A reader taking the first sheet whose title matches gets a "
             "Sarawak with nobody in it while every other check still passes. Three more "
             "traps in sources/my.py's docstring, including `-` as an in-band zero (nine "
             "Sabah districts) and DOSM spacing its own `Sex : Total` marker two different "
             "ways between the state and national volumes. "
             "**RELIGION STOPS AT ADMINISTRATIVE DISTRICT.** The mukim workbook covers all "
             "1,756 sub-districts for the whole country and carries population, ethnicity "
             "and age only; the state volumes' mukim table is population and households. "
             "160 districts is the floor and there is no finer religion tabulation at any "
             "price. "
             "Boundaries are geoBoundaries `MYS ADM2`, vintage 2020 — the census year — "
             "joined by name after three renames (Kulaijaya→Kulai, Ledang→Tangkak, "
             "Nabawan/Persiangan→Nabawan) and **verified spatially**, every polygon given a "
             "state by point-in-polygon against ADM1 that must match the census. "
             "**Putrajaya is missing from ADM2 and lies entirely inside Sepang**, so it is "
             "subtracted from Sepang before being added rather than appended — appending "
             "would double-count 48.7 km². Placement is Kontur, 144,439 hexes, national "
             "ratio 1.050; 125 of 160 districts sit between 0.8 and 1.25, and the eleven "
             "outliers were checked against polygon area and are genuine Kontur/census "
             "differences rather than bad boundaries (sources/my_geo.py).",
    ),
}
