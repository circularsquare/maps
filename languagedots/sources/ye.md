# Yemen (ye): record

Drawn 2026-10-05 (session edd42a8c-arab). Arab Barometer III (2013) first language per
governorate (every answer Arabic) applied to the Population Task Force's 2025 estimates;
Socotra on Soqotri: 34,879,018 people, 22 governorates, 2 nodes. Mainland rows `modelled`,
Socotra `derived`. Placed on religiondots' Kontur 400 m hexes. 34,878 dots.

Files: `sources/ye_surveys.py`, `sources/ab_firstlang.py`, `taxonomy/ye2013.py`,
`taxonomy/tree.d/ye.txt`, `countries/ye.py`, `data/normalized/ye.csv`.

## Sources

| source | item | result |
|---|---|---|
| Census 2004 | no language item found (IHSN DDI 229 unlabelled) | - |
| Task Force 2025 (CSO, UNFPA, IOM, OCHA) via religiondots' `ye_lookup.csv` | population | base |
| Arab Barometer II | first language | item empty for Yemen |
| **Arab Barometer III 2013** | first language | 1,200, all Arabic, 20 governorates |
| Arab Barometer V 2018-19 | none | - |
| WVS 6 Yemen 2014 | language at home | not fetched |
| UNHCR api.unhcr.org | refugees by origin, national only | not drawn |

AB III's "Sana'a" (170) is one sampling region for the capital and the governorate; both take
its answers (all Arabic anyway). al-Mahrah's 20 respondents all answered Arabic.

## Calls someone might reverse

- **Socotra 100% Soqotri** (75,725): no survey reached it; Ethnologue's 110,000 (2020) is more
  than the island's population, so the whole governorate goes on Soqotri. Hadibu's
  Arabic-speaking newcomers are not split off.
- **Mehri not drawn.** Rubin's ~130,000 and Ethnologue's 260,000 cover Yemen and Oman together,
  nothing gives Yemen's part, and AB III's al-Mahrah was all Arabic (Arabic-language fieldwork,
  probably Al Ghaydah). Oman's build left Mehri undrawn the same way.
- **Yemeni Arabic one node.** Glottolog splits Sanaani, Ta'izzi-Adeni, Hadrami (and Judeo-Yemeni);
  no source counts or places them, so no split by governorate (Iraq's Kurdish rule).
- **Refugees not drawn**: UNHCR 2025 has Somali 40,335, Ethiopian 4,371 refugees + 12,171 asylum
  seekers, Syrian 2,446, Iraqi 976 nationally only; IOM's Ethiopian migrants not counted either.

## Room for improvement

- WVS 6 Yemen (online tool) as a second round.
- A Mehri count for al-Mahrah (a Yemeni dialect survey or a 1994/2004 census table, if one asked
  "mother tongue" in al-Mahrah).
- UNHCR Yemen's governorate breakdown (Aden, Lahj's Kharaz camp, Sana'a) from its operational
  portal, which would let the Somali refugees be drawn on Somali.

## Terms

Arab Barometer: free, citation requested. Task Force via religiondots. Kontur CC BY 4.0.
