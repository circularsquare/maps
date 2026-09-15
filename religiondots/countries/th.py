# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _th_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    76 provinces for 66.0M people, and wildly uneven: Bangkok is 8.3M in 1,571 km2 while
    Mae Hong Son is 209,200 in 12,681 km2 of forested mountain. **It matters most in the
    deep south, and for the opposite reason to Cambodia's** — Pattani, Yala and Narathiwat
    are dense rather than empty, with their people along the coast and the Pattani river
    and their interiors in the Sankalakhiri range, so an equal share per polygon would put
    this map's sharpest religious boundary in the wrong place inside each province.
    Kontur's 419,176 hexes reproduce the census at 1.085x nationally with a per-province
    median of 0.90 and **not one of the 76 outside a factor of two**, against 38 of 76 for
    a shuffled null. sources/th_grid.py has the evidence.
    """
    return _kontur_place_weight(place, "th_hexes.gpkg", "sources/th_grid.py")


def _th_counts():
    """NSO 2010 census at province: 8 nodes on 76 changwat, via spec §3.10.

    **THE CATEGORIES AND THE GEOGRAPHY COME OUT OF DIFFERENT DOCUMENTS AND ARE REUNITED BY
    `allocate.py`.** No published Thai census table crosses religion with changwat, in
    either census — 2010's Table 4 and 2000's Table 5 both cut religion by
    municipal/non-municipal only, and the per-province files that once existed died with
    `statbbi.nso.go.th`. What survives is the nine categories at five regions, and
    Buddhist and Muslim percentages at all 76 provinces on a two-page provincial indicator
    sheet. So each province's residual — its population minus its Buddhists and Muslims —
    is split by its OWN REGION's composition of the other seven (`--within 1`).

    **98.5% OF THE COUNTRY IS THEREFORE `measured` AND 1.5% IS `derived`**, which is a much
    better split than an allocation usually buys, and the reason is that the two categories
    published per province are the two that hold 98.5% of Thailand. Canada's allocation is
    71.3% derived; this is 1.5%.

    THE RESIDUAL IS SPLIT PER REGION AND NOT NATIONALLY, and that is the whole reason for
    `--within`: Christianity is 3.05% of the North and 0.35% of the Northeast, so a pooled
    national share would take the hill churches of Chiang Mai and Mae Hong Son and scatter
    them evenly across Isan. §3.10c found this with India and Thailand is the second
    customer.

    ONE PROVINCE IN 76 HAS NO SHEET. Kanchanaburi's `<Province>_T.pdf` was never archived
    anywhere in the Wayback Machine, so its Buddhist and Muslim shares are its region's and
    its rows say so in `note`. 848,000 people, 1.3% of the country, and it is the only
    province here whose headline figures are not its own.
    """
    import th2010
    return _allocated_counts("th", "province", th2010)


ENTRY = {
    "th": dict(
        name="Thailand",
        source="2010 Population and Housing Census (NSO), Table 4 and the provincial sheets",
        basis="self-identification",
        view=[97.2, 5.5, 105.8, 20.6],
        note_public=(
            "**Thailand is 93.6% Buddhist and that is the least interesting thing about "
            "this map.** 56 of its 76 provinces are over 95% Buddhist, and everything worth "
            "looking at is in the twenty that are not. "
            "**The Muslim south is a gradient down the peninsula, not a border.** "
            "Narathiwat is 85.9% Muslim, Pattani 84.4%, Yala 76.6% and Satun 67.1% — the "
            "provinces of the old Sultanate of Patani, annexed in 1909, where the everyday "
            "language is Patani Malay rather than Thai and the Islam is Shafi'i Sunni. But "
            "it does not stop there: **Krabi is 34.6%, Songkhla 25.3%, Phangnga 22.1%, "
            "Phuket 16.0% and Phatthalung 11.7%**, thinning steadily northward against 4.9% "
            "nationally. The Andaman coast has been Muslim for as long as the deep south "
            "has, and a map drawn only from the four border provinces misses half of it. "
            "**Christianity is a highland religion here, not an urban one.** The North is "
            "3.05% Christian against 0.50% in the Northeast, because the Karen, Lahu, Lisu "
            "and Akha of the hills were reached by missions from the 1880s while the "
            "lowland Thai were not. Those are the same peoples China draws across the "
            "border in Yunnan, and this is the census that counts them rather than "
            "inferring them from ethnicity. **Mae Hong Son, on the Myanmar border, comes "
            "out the most Christian province in the country** — but see the last paragraph, "
            "because that figure is inferred rather than counted. "
            "**Bangkok is where the small religions are.** It holds most of the country's "
            "Hindus and Sikhs — the Punjabi merchant community of Phahurat and the Tamil "
            "community around Silom — and at 4.6% Muslim it has more Muslims than any "
            "province outside the south. "
            "**What the census cannot show, and it is most of Thai religious practice.** "
            "The form asks which religion you belong to, and in a country where Buddhist "
            "identity is close to civic default that question does not reach the spirit "
            "houses outside every building, the Brahmanical court ritual, the Chinese "
            "temple practice of the Thai Chinese, or the phi that the same household "
            "attends to alongside the wat. **0.07% of Thailand answers `no religion`, the "
            "smallest such share on this map**, and that is a fact about the question "
            "rather than about the country. "
            "**And only two of the nine categories are counted where you see them.** "
            "Buddhist and Muslim are published for every province; Christian, Hindu, "
            "Confucian, Sikh, other and none are published only for five regions, and each "
            "province's share of them here is its own region's. Turn on the `inferred dots` "
            "control to see which is which — 98.5% of Thailand stays. **Mae Hong Son is "
            "where that assumption is doing the most work**: a quarter of the province is "
            "neither Buddhist nor Muslim, and this map calls almost all of that Christian "
            "because almost all of the North's is. Its Karen and Lahu villages hold both "
            "churches and older traditions, and nothing published separates them."),
        how="census, 2010",
        fill="from the same census at region level",
        grain="provinces, 868,000 people on average",
        counts=_th_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "th" / "th_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_th_place_weight,
        note="**EVERY CENSUS FILE COMES OUT OF THE WAYBACK MACHINE, AND THAT IS THE FIND.** "
             "sources.md recorded Thailand as *'data existed and the server is gone'*: "
             "`statbbi.nso.go.th` and `web.nso.go.th` no longer resolve. What that missed "
             "is that **`www.nso.go.th` is alive and answers 418 to a bare curl and 200 to "
             "a browser User-Agent** — the office moved hosts, as Nepal's had — and that "
             "its old `/sites/2014/Documents/` tree, now 404, was archived wholesale. The "
             "live site's new CKAN (`catalog.nso.go.th`, keyless) carries only a 6-region "
             "3-religion survey table and is not used. "
             "**The two halves are §3.10:** nine categories at five regions from the "
             "regional volumes' Table 4, and Buddhist/Muslim percentages at 76 provinces "
             "from the `kpi_stat` indicator sheets, reunited by `allocate.py --within 1`. "
             "100% of the census population is drawn, 98.5% of it `measured`. "
             "**The check is the residual.** Each region's province residuals, summed, "
             "against that region's own Table 4 non-Buddhist non-Muslim total: -1.8% "
             "Bangkok, -0.8% Central, +4.0% North, +2.2% Northeast, -0.3% South. Those are "
             "two documents that never reference each other agreeing to within a few "
             "percent, and they are the only evidence that the percentages and the counts "
             "describe the same population. "
             "**The rounding is real and is stated**: the indicator sheets give shares to "
             "one decimal, so a province's Buddhist figure carries about ±0.05% and the "
             "residual twice that. The denominator does not compound it — province totals "
             "come from Table 1 at full precision. **A share too small to print appears as "
             "`a`**, NSO's *'less than half the last digit shown'*, and reading that as a "
             "number put Lampang's household-registration rate into its Muslim row before "
             "the check caught it. "
             "**Boundaries are geoBoundaries ADM1, 77 polygons, joined on the TIS 1099 "
             "code through OCHA's COD attribute table** — geoBoundaries has English names "
             "only, the census Thai only, and seven provinces begin `Nakhon`. **Bueng Kan "
             "is dissolved back into Nong Khai** (spec §8.1): it was carved out in March "
             "2011, seven months after the census, so the tables have 76 changwat and the "
             "boundary file has 77. "
             "**Kanchanaburi's indicator sheet was never archived** and takes its region's "
             "shares — 848,000 people, the one province whose headline figures are not its "
             "own.",
    ),
}
