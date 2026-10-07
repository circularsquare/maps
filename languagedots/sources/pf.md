# French Polynesia: drawn from RP 2012, five language groups by subdivision

Sessions d9e44929-pf (placement layer, parked) and edd42a8c-pf (table, build), 2026-10-05.

Drawn: 202,825 people aged 15+, 5 subdivisions, 4 nodes. `sources/pf_rp2012.py` ->
`data/normalized/pf.csv`; `taxonomy/pf2012.py` (no new tree nodes, so no `tree.d/pf.txt`);
`countries/pf.py`; placement `sources/pf_geo.py` -> `data/geo/pf/pf_hexes.gpkg`.

## 1. The question, and what is open

Recensement de la population (ISPF with INSEE), "langue la plus couramment parlée en famille", one
answer, everyone aged 15+. Asked in 2002, 2007, 2012, 2017 and 2022.

Searched, 2026-10-05, for anything below the territory:

- **2022.** ISPF's new site (Nuxt + Strapi; `https://www.ispf.pf/content/api/...`, no key; the
  host answered normally this session, paced requests). Publication 1396 "Le recensement de la
  population en Polynésie française en 2022" (API id 3965): 6-page PDF and data workbook, no
  language anywhere. Strapi's `/content/api/upload/files` lists every uploaded file: the only 2022
  language items are two atlas PDFs (uploads 5592, 5593, 2025-03-13), choropleths of the French and
  Polynesian shares by commune associée in five classes, no figures except each map's "Moyenne"
  (2022 territory: French 76.2%, Polynesian 22.8%; Iles du Vent 82.2 / 16.8, Iles Sous-le-Vent
  61.1 / 38.1, Marquises 40.3 / 59.3, Australes 49.0 / 50.5, Tuamotu-Gambier 63.2 / 36.2).
  Collections `datasets` (188) and `donnees` (116): no language. INSEE Première 1990 (RP 2022):
  no language. data.ispf.pf: 403.
- **2017.** ISPF's ASP.NET pivot tool (`/bases/Recensements/2017/Donnees_detaillees/Langues.aspx`)
  had commune rows; it is gone (404 since 2022-09) and its ~70 Wayback captures all hold the
  default view (French ability by subdivision); the pivot used postbacks, which the archive cannot
  replay. Only five-class maps remain (2017 atlas PDFs). Commune "fiches" xlsx (uploads 7-756) have
  no language sheet (and a broken sharedStrings part).
- **2012, 2007, 2002.** Standard tables (Wayback copies of `docs/default-source/rp20xx/`): by
  subdivision only, and only five groups; every language label is national only. Same layout all
  three years.
- **2007 atlas indicators** (`LAN_Atlas_quartier_indic.xls`, Wayback 2021-08-24): the Polynesian
  and French shares of 15+ for 164 quartiers (communes associées, plus lettered sub-districts in
  urban Tahiti). Shares only, no populations.
- Wikipedia's per-place language paragraphs quote the 2017 pivot for a few places (Papeete's urban
  area), not every commune. Not used.
- The arrêté of 10 January 2024 on RP 2022 dissemination (JORF 7 Feb 2024, on ISPF's site) makes
  univariate counts by commune and commune associée releasable to anyone, and lets ISPF produce
  tables on request (art. 3, 5). That is ask 013.

**queue.csv's lead was wrong twice:** "publication/1396" is a 2016 trade note's API id (1396 is
the RP 2022 note's publication number), and the coverage sweep's "2022: French 68.5%, Polynesian
29.9%" are the **2007** figures (131,672 and 57,475 of 192,176). 2022 is 76.2 / 22.8 (above).

## 2. The table drawn

`Tableaux_standards_RP2012_Langues_v2003.xls`, ISPF, Wayback 2014-10-09
(`web.archive.org/web/20141009011050id_/http://www.ispf.pf/docs/default-source/rp2012/...`),
copied to `data/raw/pf/`. Sheet LAN1b: 15+ by subdivision x group x 10-year age.

| subdivision | French | Polynesian | Asian | European | Other | 15+ |
|---|---|---|---|---|---|---|
| Iles du Vent | 117,448 | 32,412 | 1,401 | 542 | 986 | 152,789 |
| Iles Sous-le-Vent | 13,397 | 12,112 | 119 | 121 | 159 | 25,908 |
| Marquises | 2,383 | 4,332 | 3 | 2 | 12 | 6,735 |
| Australes | 1,790 | 3,149 | 3 | 6 | 21 | 4,969 |
| Tuamotu-Gambier | 6,932 | 5,278 | 115 | 20 | 82 | 12,427 |
| territory | 141,950 | 57,283 | 1,641 | 691 | 1,260 | 202,825 |

Checks in `pf_rp2012.py` (all hold): each subdivision's groups and its 8 age bands sum to its
total; the subdivisions sum to the territory in every group; LAN3b (knowledge of French and of a
Polynesian language, same census) gives the same 15+ total in all five; the national "Chiffres
clés" (31 labels) sum to their groups, the groups to 202,825, and each group equals LAN1b's
territory column (LAN1b's "Autres" = regional languages of France 142 + Pacific 306 + other
foreign 455 + "Sourd et muet" 357).

Corroboration against 2022 (map averages above): the Polynesian share fell in every subdivision
from 2012 (Iles du Vent 21.2 -> 16.8, Sous-le-Vent 46.7 -> 38.1, Marquises 64.3 -> 59.3, Australes
63.4 -> 50.5, Tuamotu-Gambier 42.5 -> 36.2), the same ranking both years.

**Why 2012 and not the 2022 rates.** The 2022 maps give French and Polynesian shares per
subdivision, newer by ten years. Turning them into counts needs a 2022 15+ population per
subdivision, which ISPF has not published (the 1396 workbook has total population by commune and
a national pyramid only), so the counts would rest on an estimated base; and the 1% remainder
would lose the Asian / European split. Measured 2012 counts won; note_public gives the 2022
national figures. Reversible in an hour if wanted: same five units, two groups and a remainder.

## 3. Mapping (taxonomy/pf2012.py)

The five labels are groups; each sits on the narrowest node holding what ISPF filed under it:
French -> French; Langue polynésienne -> `austronesian.oceanic`; Langue asiatique -> `other`
(Sinitic, Japonic, more); Langue européenne (sauf français) -> `indoeuropean` (at most 8 of 691
could be non-Indo-European); Autres -> `other` (several families and sign).

**Polynesian on Oceanic.** The tree has no Polynesian node: Tahitian, Maori, Samoan, Tongan, Rapa
Nui sit flat under Oceanic (fi.txt, au.txt, cl.txt). So 28% of the territory draws washed out as
"language not named", which is what the table says. A Polynesian group node would need those
fragments' leaves re-parented (ids other countries' mappings use), not this country's to do.
Splitting the group by the national mix was rejected: it would draw Tahitian on the Marquesas.

## 4. Placement

**Units.** ISPF's commune layer ("Limites géographiques administratives", data.gouv.fr, ODbL,
2022-06-10): 48 communes, 116 communes associées (each carrying `IDSub`), 119 islands, each with
its 2017 census population. The communes associées sum to their commune in all 48.

**Hexes** (`pf_geo.py`). Kontur PF (2023-11, 2,120 hexes, 308,876 people) reads 1.12x the 2017
census; per commune, normalised, p10 0.73, median 0.95, p90 1.44; log r 0.985 against 0.536 for
the best of 500 shuffles. It puts people on atolls nobody lived on in 2017 (Moruroa 159, Pinaki
93, Tikei 88, Nihiru 63; 543 on 14 islands) and misweights atolls inside a commune (Kaukura 3.37x,
Mataiva 0.39x). So each hex goes to the commune associée its centroid falls in (else the nearest,
799 hexes, all within 1.9 km), hexes on islands empty in 2017 get 0, and each commune associée's
hexes are scaled to its 2017 population. 2,094 hexes, densest 3,704 people/km2. Columns `unit`
(commune), `comas`, `sub` (subdivision, the counted unit), `pop`.

**Inside a subdivision** (`countries/pf.py`, AGENT_BRIEF §4.4, no ask: placement only). French
dots weight each hex's people by its commune associée's 2007 French share, Polynesian dots by the
2007 Polynesian share (`data/normalized/pf_place2007.csv`, from the 2007 atlas indicators);
Asian, European and other dots by people. The most specific placement evidence open: same
question, same census series, five years before the counts. Lettered quartiers (Arue 12A-D,
Papeete 35A-K, Punaauia 38A-J, Teva I Uta 52A-C and so on) are averaged unweighted onto their
commune associée; urban Tahiti's quartiers differ little (Papeete 0.09-0.31 Polynesian), so the
missing weights move little. Polynesian shares run 0.09 (Vairaatea, Nukutavake) to 0.99 (Hakamaii, Ua Pou).
Scatter: 10 rows placed on share-weighted people, 1 on people; 200 dots at 1:1000.

## 5. Calls someone might reverse

- 2012 measured counts over 2022 rates on an estimated base (§2).
- Polynesian on `austronesian.oceanic`, drawn as not named (§3).
- European group on `indoeuropean`, not `other` (§3).
- 2007 shares as the placement weight inside a subdivision (§4).

## 6. Wording

`how`, `grain`, `gap`, `note_public` are in `countries/pf.py`. Gap: under-15s, not asked:
268,270 - 202,825 = about 65,000 in 2012.

## 7. Cut from note_public (2026-10-06 text sweep)

- The Polynesian group nationally in 2012 (RP 2012 "Chiffres clés"): 82% Tahitian, 9% Marquesan,
  5% Paumotu, 4% the languages of the Austral Islands, 1% Mangarevan.
- 2017 and 2022 figures by commune are not public (§1).
- Placement detail: per commune associée (116) and 400 m hexagon (§4).
