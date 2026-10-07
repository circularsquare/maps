# Morocco: RGPH 2024, local languages used (several allowed), by commune

Drawn 2026-10-05 (session edd42a8c-ma, under a supervisor). 36,443,697 people in households,
1,530 drawn units from 1,534 leaves with figures (communes and city arrondissements; 11 drawn
together on remainders, 4 merged into a neighbour), 5 nodes, every row
`derived` (multi-answer scaling, AGENT_BRIEF §2). 36,442 dots at 1:1000, no rings.

```
python sources/ma_rgph.py --fetch     # HCP indicator workbook -> data/normalized/ma.csv
python sources/ma_geo.py --fetch      # Kontur Boundaries MA (OSM communes); religiondots' hexes re-keyed
python taxonomy/build.py
python tools/check_country.py ma
python scatter.py --country ma
```

## Source

- **Table.** HCP, *Indicateurs démographiques et socioéconomiques du Royaume du Maroc selon les
  résultats du RGPH 2024*, Excel, https://www.hcp.ma/file/242671/ (7,597,863 bytes, sha256
  pinned in `ma_rgph.py`), no login. Sheet `Population`, columns 48-52: "Langues locales
  utilisées (non exclusives) (%)" — Darija, Tachelhit, Tamazight, Tarifit, Hassania — for the
  nation, 12 régions, 75 provinces, 213 cercles, 1,503 communes, 41 arrondissements, 8
  préfectures d'arrondissement(s) and 164 "dont le centre urbain" rows. Percentages, one decimal.
- **Question.** Daily use, several answers (national sum 117.5%). Not mother tongue: HCP's
  answer to the Amazigh-figure row (Hespress, "Census figures clarified", 2025) is that people
  able to speak Amazigh but not using it daily are not counted. Amazigh associations reject the
  24.8% Amazigh total (some claim 85%). Said in `note_public`; no ask filed (the map draws the
  census as published and holds nothing back; see Calls).
- **Sample.** HCP's methodology notes: the detailed questionnaire (literacy, education,
  activity, and by its placement among those indicators the language item) went to every
  household in communes under 2,000 households and to a 20% sample elsewhere. Not confirmed
  item by item; `note_public` says "appears to".
- **No single-answer table** exists for 2024; IPUMS-I has 2014 microdata (LANGMA, first-listed
  language) but it is gated (IPUMS account blocked). Not used.

## Build (`ma_rgph.py`)

- **Leaves**: the 41 arrondissements of Tanger, Fès, Rabat, Salé, Casablanca, Marrakech and the
  1,462 communes without arrondissements = 1,503 + 41 − 6 = 1,538 leaves. Arrondissement codes are
  `<province>01<nn>`, the commune `<province>010` (Fès broke a naive prefix test); nesting is by
  row order and asserted by sums.
- **Base: population municipale.** Check 4 rebuilt every province's printed shares from its
  leaves: weighted by municipale the mean miss is 0.027 points (worst 0.09, Tan-Tan Hassania),
  by légale 0.195 (worst 16 points, Es-Semara). So the indicator's base is households (plus
  nomads), not the 337,739 "comptée à part" (barracks, boarding schools, prisons), who go in
  `gap`. National check: 91.90 / 14.22 / 7.44 / 3.19 / 0.79 against 91.9 / 14.2 / 7.4 / 3.2 / 0.8.
- **Aousserd** (Oued Ed-Dahab-Aousserd) cannot be rebuilt (misses up to 13 points); it holds three
  of the four "." communes, so its printed figure must include something the communes do not.
  Excluded from check 4's bar, printed.
- **Scaling**: `count = municipale × share / max(sum, 100)`. Shares sum 87.7-197.5; 293 leaves under
  100 (101 under 99, mostly rounding). Where under 100 the shortfall is people using none of the
  five: 41,523, not drawn (`gap`). Scaling those up to 100 instead would have invented answers.
- **Not drawn**: Lagouira, Aghouinite, Zoug, Mijik (starred, figures from the local
  administration for a mobile population, "." in every cell): 5,371.

| drawn | people | share |
|---|---:|---:|
| Darija | 29,595,775 | 81.2% |
| Tachelhit | 3,849,622 | 10.6% |
| Tamazight | 1,967,030 | 5.4% |
| Tarifit | 822,173 | 2.3% |
| Hassania | 209,098 | 0.6% |

## Taxonomy (`taxonomy/ma2024.py`, `tree.d/ma.txt`)

Darija → new `afroasiatic.darija` "Moroccan Arabic (Darija)" (moro1292), beside `arabic` as ml's
`hassaniya` is, because `arabic` is a leaf many countries draw on. Tachelhit → new
`berber.tachelhit` (tach1250). Tamazight → ca.txt's existing `berber.tamazight` (in Morocco's
census it is Central Atlas Tamazight, cent2194). Tarifit → new `berber.tarifit` "Tarifit
(Riffian)" (tari1263). Hassania → ml.txt's `hassaniya`. Colours: Darija 0.82 0.08 175 (Arabic's
mint turned bluer, #8bd6c1), Tachelhit 0.87 0.15 95 (strong yellow, #f3d350); Tamazight keeps its
generated olive #abc369 and Tarifit its generated orange-brown #c28336; Hassaniya ml's dark teal.
No other country's colour was changed.

## Geography (`ma_geo.py`)

religiondots' 73-unit hex layer (Kontur 2023-11, already cut to B19 west of the berm, Ceuta and
Melilla out, discs on Smara, Tan-Tan and Assa) re-keyed to communes, every hex staying inside its
religiondots unit and keeping its population (104,488 hexes, 37,620,058 Kontur people).

- **Commune polygons**: Kontur Boundaries MA 2023-06-28 (HDX `kontur-boundaries-morocco`, OSM,
  ODbL), Kontur level 9 = OSM admin_level 8, 1,531 polygons (asserted), plus six OSM pachaliks
  (level 7) offered only to leaves no commune polygon took; five paired (Moulay Ali Cherif,
  Aklim, Sidi Slimane, Ajdir, Gueznaia). HCP and COD-AB have no commune layer.
- **Join by name inside each religiondots unit** (never across: Oulad Hcine is in two
  neighbouring provinces). Exact folded name unique on both sides 1,387; spelling rules
  (Ouled/Oulad, My/Moulay, rh/gh, ...) and difflib ≥ 0.72, best pair first, 132, all printed and
  read; 3 hand aliases (Assoukhour Assawda = Roches Noires; Tanger's Souani and Médina = Charf
  Souani and Tanger Médina) and one Arabic-only OSM name (Yacoub El Mansour). The three Tarfaya
  strip polygons forced to EH03, where religiondots puts that strip.
- **Witness**: Kontur per drawn unit against census municipale, log r = 0.882 against a best of
  0.422 over 500 shuffles within religiondots units; median ratio 1.09. The 132 fuzzy pairs:
  median 1.08, 8 outside 1/3-3, all unambiguous names (Al Majjatia Oulad Taleb 0.25, Oukaimeden
  0.03, Mechouar Kasba 9.64, ...): Kontur spread, not wrong pairs.
- **Remainders**: 13 leaves (202,708 people) paired with nothing; per religiondots unit they are
  drawn together on the hexes no paired polygon holds (`<unit>-rest`). In most units that is one
  commune on one unpaired polygon (Ribat El Kheir = old Ahermoumou, Sahel = Khemis Sahel, Bitit,
  Azlaf, Tnine Aglou, El Aioun Sidi Mellouk, Had Al Gharbia, Lamssid, Tifariti, Gleibat El Foula).
  604 hexes (195,699 Kontur people) outside paired polygons in fully paired units (Casablanca's
  and Rabat's whole-city polygons, which overlap their arrondissements) go to the nearest paired
  polygon.
- **Merged into a neighbour** (no hex): Sidi Yahia El Gharb's remainder (39,201; its OSM polygon
  sits in COD's Sidi Kacem) into Sidi Slimane; Ben Ahmed's remainder (33,332; no OSM polygon) into
  its neighbour c64610115; two small communes (4,545 and 2,510). 1,534 leaves → 1,530 units.
- Kontur cap: one registered block near Laâyoune (Dcheira, 901 people) lowered by religiondots'
  registry at scatter; nothing new for `kontur_cap.csv`.

## Calls someone might reverse

- Population municipale as the base, not légale (check 4 decides it).
- People in the five's shortfall left undrawn rather than scaled in.
- Darija as its own node, not `arabic`.
- The Amazigh dispute and the sample design go in `note_public`, no ask: the census is drawn as
  published and nothing is held back.
- Commune grain on OSM polygons instead of religiondots' 73 provinces.
