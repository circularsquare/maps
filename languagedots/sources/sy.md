# Syria (sy): record

**Drawn 2026-10-05** (session edd42a8c-mid). Nothing counts Syria's languages; built on the
supervisor's estimate route (Egypt's, `sources/eg.md`): cited minority figures carved out of
each governorate's population, the rest on the governorate's Arabic. 21,377,000 people (CBS
end-2011), 14 governorates, 7 nodes, every row `modelled`. 21,375 dots at 1:1000.

```
python sources/sy_build.py
python taxonomy/build.py
python tools/check_country.py sy
python scatter.py --country sy
```

Files: `sources/sy_build.py`, `taxonomy/sy2011.py`, `taxonomy/tree.d/sy.txt`, `countries/sy.py`,
`data/normalized/sy.csv`, `data/raw/sy/balanche_sectarianism_2018.pdf` (12 MB, the source PDF
from uni-jena.de).

## 1. What exists

- **Census**: none since 2004; no census since 1960 tabulated language or ethnicity (coverage
  sweep; religiondots `sources/sy.md`). The 1962 Hasakah exceptional census was citizenship.
- **World Values Survey**: Syria is in no wave. Checked 2026-10-05 against the online tool's
  country lists for waves 4-7 (the brief's "Syria WVS wave 6" lead was wrong; Libya is there).
- **Arab Barometer**: no released wave reached Syria; its late-2025 Syria fieldwork is not out
  (religiondots `sources/sy.md`). When it is, it should replace most of this.
- **Population base**: religiondots' `sy_lookup.csv` (read-only), CBS Statistical Abstract 2012
  table 3/2, people living in each governorate on 31/12/2011, 21,377,000, via OCHA HDX. Chosen
  because religiondots uses it: the UN's 2025 governorate baseline is marked not for research.
  Displacement since 2011 is not drawn (said in `note_public`).

## 2. Carve-outs (`sources/sy_build.py` docstring has every quote)

| language | governorate | figure | source |
|---|---|---:|---|
| Kurdish | Al-Hasakeh | 55% = 831,600 | Balanche 2018 p51: Jazira and Kobane cantons 55% Kurdish |
| Kurdish | Aleppo | 19.0% = 925,074 | Afrin district 100% ("almost 100%"), Ayn al-Arab 55%, Aleppo city 22.5% ("20-25%", p53), on 2004 census district/subdistrict figures (Wikipedia district pages) |
| Kurdish | Damascus | 409,444 | Balanche p51, "the one million Kurds in Damascus and Aleppo", less Aleppo city's 590,556 |
| Armenian | Aleppo | 150,000 | Balanche p22, pre-war Aleppo Armenians |
| Turkmen | Latakia | 29,819 | Rabia + Qastal Ma'af subdistricts, 2004 census, grown at the national 2004-11 rate |
| Turkmen | Aleppo | 183,951 | Balanche fig. 16, Turkmen 1% of 2011, less Latakia |
| Aramaic (not split) | Al-Hasakeh | 12.5% = 189,000 | Wikipedia, Al-Hasakah Governorate: Christians 12.5% in 2011, mostly Assyrian |
| Western Neo-Aramaic | Rural Damascus | 30,000 | Wikipedia infobox (2023) |
| Mesopotamian Arabic | Hasakah, Deir-ez-Zor, Raqqa | the rest | Glottolog nort3142 / meso1252 |
| Levantine Arabic | the other 11 | the rest | |

National Kurdish 10.1%, against the usual ~10% (Wikipedia, Kurds in Syria: 5-10%, "usually
estimated at 10%") and Balanche's ethnic 14-15%.

**Placement inside a governorate** (`countries/sy.py` BOXES; moves people only inside the
governorate, AGENT_BRIEF 4.4): Aleppo's Kurds into Afrin, Ayn al-Arab and Aleppo city boxes in
the estimate's proportions; Aleppo's Armenians in the city; Turkmen in the Azaz-al-Rai-Jarabulus
countryside and Jabal al-Turkmen; Hasakah's Kurds and Aramaic north of 36.45N (the Arab south
left out); Western Neo-Aramaic in Maaloula/Jubb'adin. Checked on the dots: Kurdish cells at
Aleppo/Afrin 797, Qamishli-Malikiyah 647, Damascus 409, Kobani 108.

## 3. Calls someone might reverse

- **Damascus Kurds at 23%.** Balanche's million is ethnic; many Damascus Kurds (Rukn al-Din)
  have spoken Arabic for generations. No retention source; drawn in full, said as an upper bound.
  Dropping it would take national Kurdish to 8.2%.
- **Hasakah Aramaic = all Christians (12.5%)**, an upper bound (Armenians and Arabic-speaking
  Syriacs inside it); Turoyo and Assyrian Neo-Aramaic not split, so on Canada's leaf
  `afroasiatic.aramaic`.
- **Hasakah Arabic at 32.5%**, low against the usual 40%-ish, because the 55% Kurdish is the
  two cantons' combined figure.
- **Mesopotamian Arabic** as a new leaf for the Euphrates and Jazira, rather than Levantine
  everywhere or Iraq's "Iraqi Arabic" node.
- Turkmen outside Latakia all in Aleppo (Homs, Hama, Golan Turkmen not placed).
- Syrian Turkmen a leaf of its own rather than `turkic.turkish`.

## 4. Not drawn

Circassians (Quneitra, Homs, Damascus; often put near 100,000), Armenians outside Aleppo
(Damascus, Kessab, Qamishli), Domari, Chechens, Iraqi refugees of 2011. No estimate places them.

## 5. Room for improvement

Arab Barometer's 2025 Syria wave, when released, if it has a first-language item and
governorate codes: a real survey would replace every row here. Subdistrict populations for 2011
would let Hasakah's Kurdish and Arab areas be drawn apart properly.

## Terms

Balanche 2018: a public Washington Institute policy paper, cited. Wikipedia CC BY-SA, figures
cited. CBS population via OCHA HDX (religiondots). Glottolog CC BY. Kontur CC BY 4.0.
