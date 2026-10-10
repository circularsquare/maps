# Senegal: RGPH-5 2023, language spoken most often, by région

Drawn 2026-10-05 (session d9e44929-sn). 15,940,207 residents aged 3 and over, 14 régions, 22
nodes from 23 table categories, every row `measured`. 15,932 dots at 1:1000, no rings.

```
python sources/sn_rgph.py --fetch     # the 14 regional reports + chapter 1 of the national report
python sources/sn_geo.py              # religiondots' hexes re-keyed to the 14 régions (read-only)
python sources/sn_place.py            # + département and CLEAR zone per hex, sn_clear.csv
python taxonomy/build.py
python tools/check_country.py sn
python scatter.py --country sn
```

## Source

- **Tables.** ANSD, RGPH-5 2023, the fourteen regional reports ("RGPH-5 : Rapport de
  présentation des résultats, Région de ...", dated April 2026), each with a table
  "Répartition (%) de la population résidente de 3 ans ou plus par principale langue
  couramment parlée selon le sexe" (numbered II-11, II-12 or II-13; Diourbel's says III-12):
  23 categories x masculin / féminin / ensemble, counts and %. The second table is chapter 1 of
  the national final report, Tableau I-32 (pp57-58), the same table nationally.
- **Route.** All listed at https://www.ansd.sn/rapports/rgph-5-2023, no login. ANSD's TLS chain
  is broken, so `--fetch` uses `curl -k` with a browser User-Agent; every file is pinned by
  size and sha256 instead.
- **Lead not used.** The coverage sweep had only Tableau I-32 (national) and pointed to CLEAR
  Global's département shares from the 2013 IPUMS sample. The regional reports came out in
  April 2026 and give 2023 counts by région from ANSD itself, so the 2013 sample was not
  used for counts. Since 2026-10-07 it places dots inside régions (Geography, below).
- **Question.** B18 "première langue la plus souvent parlée": the language the person speaks
  most often, residents aged 3+, one answer. B19 (second language) is not tabulated by
  région. Not a mother-tongue question, so `how` says "language spoken most often" and
  `note_public` says Wolof's reach is inflated by it (53.5% nationally, 68.7% in Dakar).
- **Grain.** 14 régions. No regional report has a département language table (each was
  searched); département tables exist only for literacy.

## Universe and gap

Each regional table closes on its own Total (Saint-Louis aside, see Traps). The fourteen sum to
15,940,207. Tableau I-32 prints 16,432,959 (its rows sum to 16,432,957). **The regions are
492,752 short (3.0%), and nearly uniformly so**: the ratio régions / national is 0.968-0.981 for
every language with more than 1,000 speakers except sign language (0.939) and Jalunga (0.960).
Nothing in the reports explains it. The résident population by région (Tableau I-21) does sum to
the national 18,126,390, so it is the language tables, not the régions, that differ. A
uniform 3% looks like a weighting of the national table (the post-enumeration survey put
coverage at 96.0%) rather than a missing group, but that is a guess. The regional counts are
drawn as printed and the difference is in `gap`; scaling them up 3% would change no share.

Children under three were not asked: 18,126,390 - 16,432,959 = 1,693,431 by the national
table, "about 1.7 million" in `gap`.

## Checks (`sources/sn_rgph.py`, `sources/sn_geo.py`, all pass)

| check | result |
|---|---|
| files | 15 PDFs pinned (size, sha256, %%EOF) |
| each regional table | the 23 labels once each, 6 cells a row, M + F = ensemble within 1, rows = Total within 2 |
| Tableau I-32 | rows 16,432,957 against printed 16,432,959 |
| régions vs national | short by exactly the pinned 492,752; per-language ratios printed |
| COD-AB admin1 | 14 pcodes and names asserted (religiondots' v02 shapefile) |
| hexes | 63,771 re-keyed, population unchanged; 415 hexes in split units outside both polygons (coast, edges) go to the nearer |
| Kontur / census by région | national 0.987; normalised 0.88 (Kédougou) to 1.19 (Ziguinchor) |
| log correlation | 0.995 against a best of 0.828 over 500 shuffles |

## Geography

religiondots draws Senegal at its 1988 units (nine régions and Diourbel's three départements),
on COD-AB v02 with Kontur 2023-11 hexes. Each 1988 unit is one of today's régions, two of them,
or part of Diourbel, so `sources/sn_geo.py` keeps religiondots' hexes (with its coastal snaps)
and only splits the four merged units (Saint-Louis/Matam, Tambacounda/Kédougou,
Kaolack/Kaffrine, Kolda/Sédhiou) by the COD-AB admin1 polygon of each hex centroid. Output
`data/geo/sn/sn_hexes.gpkg`. No Kontur cap block (religiondots' check found none).

### Placement inside régions (2026-10-07, fix-place)

Placement-only proxy (AGENT_BRIEF §4.4): counts stay the census's per région; inside one, a
language's dots go to hexes by Kontur population x CLEAR Global's share of that language in the
hex's département. `sources/sn_place.py` writes `data/geo/sn/sn_hexes_dept.gpkg` (adds `dept`,
COD-AB admin2 pcode kept inside its own région, and `zone`) and `data/normalized/sn_clear.csv`;
the weighter is `sources/clear_place.py::ClearWeighter`, shared with Mali.

- **Source.** CLEAR Global `senegal-languages` (HDX, CC BY-SA), admin2 CSV: "main language
  spoken in the household", from the IPUMS sample of the 2013 census. Downloaded 2026-10-07 to
  `data/raw/sn/clearglobal_sen_admin{0,1,2}.csv`. A different year and question from the counts
  (2023, language spoken most often); used only for where inside a région.
- **Coverage is partial.** 18 of 46 départements have their own row; six régions have a "level 2
  unknown" row (`SN01XXX` etc.); Kaffrine and Matam have no row at all (IPUMS files them under
  their pre-2008 parents). Zones: own pcode; else the région's unknown row (Dakar's Guédiawaye,
  Pikine, Keur Massar; Kolda's Kolda and Médina Yoro Foulah); else the région's mean (Gossas,
  Guinguinéo, Linguère). Eight régions are split (Dakar, Diourbel, Fatick, Kaolack, Kolda, Louga,
  Thiès, Ziguinchor); Kaffrine, Kédougou, Matam, Saint-Louis, Sédhiou and Tambacounda are one
  zone and stay on population.
- **Label -> CLEAR codes** (`CLEAR_CODES`): Wolof + Lebu Wolof; Pulaar + Bilkire Fulani;
  Màndienka = Mandinka + West Manding + Western Maninkakan + Jahanka. Bayot, Tourka, sign
  language and the remainders go on population. Node ids are passed through `regroup.move`, so
  the moved Mandinka and French nodes match.
- **Effect.** 130 (région, language) rows placed by CLEAR, 45 on population. Share of a
  language's dots moved off a plain population spread (half the absolute difference by
  département): Serer in Diourbel 46% (to Bambey), French in Dakar 41% (to Dakar département),
  Serer in Kaolack 31% and Thiès 31% (Mbour), Wolof in Ziguinchor 31%, Joola in Ziguinchor 21%
  (Bignona, Oussouye). People-weighted over languages of 20,000+ in split régions: 7.7%.
- **Checks** (all pass): shares sum to 1 per location; every CLEAR code is a COD pcode or a
  région unknown row; every hex's département is in its région, all 46 hit (604 hexes on the
  nearest one); population and hex count equal to sn_geo's layer. Counts untouched: 15,932 dots
  before and after.
- **Sanity, CLEAR région shares against the census:** mostly within 10 points. Exceptions:
  Saint-Louis (Pulaar 22 against 53; CLEAR's Saint-Louis is likely a different footprint) and
  Kédougou Manding 33.8% against the census's 5.0% Màndienka, which supports the reading below
  that Kédougou's 31.5% "Autres langues africaines" is mostly Malinké. Neither affects placement
  (both régions are one zone).

## Mapping calls (`taxonomy/sn2023.py`, `taxonomy/tree.d/sn.txt`)

- **New nodes:** Serer, a Jola group (Joola and Bayot as leaves), Balanta, Manjak, Mankanya,
  a Tenda group (Bassari = Oniyan, Bedik = Mënik, Wamey/Konyagi = Womey), Jaad (Kanjad),
  Bainouk-Gunyuño (Guñuun), all under Atlantic; Yalunka (Jalunga) under Mande; `other.tourka`.
- **Pulaar on `nigercongo.atlantic.fulah`**, the Fula node Mali and others draw.
- **Bayot under Jola.** Glottolog makes Bayot its own branch; the usual classification puts it
  in Jola. Reversible by moving one line.
- **Kanjad (Jaad-Badyara) flat under Atlantic**, not in the Tenda group, where some put it.
- **Tourka (Sénégal), 1,484 people, unidentified.** Printed among the national languages,
  mostly in Kaolack (757) and Fatick (279). Not in Glottolog under that name (Glottolog's Turka
  is a Gur language of Burkina Faso and Côte d'Ivoire), not in the decree lists of codified
  languages I know of, and two web searches found nothing but restatements of the census. A
  node of its own under `other`, as et.txt does for its unidentified names.
- **"Autres langues africaines" (201,613) on `africa_other`.** 31.5% of Kédougou (66,991) and
  5.3% of Tambacounda. In Kédougou this is very likely mostly Malinké (Maninka of Kédougou, which
  the table's 5.0% "Màndienka" cannot hold) and the languages of gold-mining migrants from Mali
  and Guinea, but the census does not name it, so it is not guessed. `note_public` says so.
- **"Langues étrangères" and "Autres langues étrangères non africaines" both on `other`.**
  Neither names a language; the first is printed apart from the African remainder.
- **Sign language** on `signlanguage`. Français on French.
- The Cangin languages (Noon, Laalaa, Saafi, Ndut, Paloor), codified national languages, are
  not printed; they sit inside Sereer or the African remainder.

**Colours.** Wolof (bare in bf/fi/us) and Mandinka (bare in au) are coloured here: Wolof golden
yellow, Mandinka mid green. Fula, Soninke and Hassaniya keep ml.txt's. Closest pairs at 0.4%+ in
one région (OKLab): `africa_other`/`other` 0.073 (shared nodes, not mine), French/Serer 0.087,
Jaad/Wolof 0.096 (Kolda, 0.5%), Balanta/Mankanya 0.100 (Sédhiou). Wolof/Fula/Serer/Jola/
Mandinka are all 0.15 or more apart.

## Traps

- **Layouts differ by report**: labels wrapped over two or three lines (Kaffrine), the column
  head repeated after a page break, Ziguinchor's head carries SPSS's `1ER_LANGUE_RECOD` and
  "Nb.colonnes", counts with and without thousands spaces, "81,3 782927" on one line (Diourbel).
- **Saint-Louis prints a second "Total" row** that repeats the sign-language row; the first
  Total closes the table and is used.
- The regional reports' running heads carry a page number that is not the PDF page.

## Terms

ANSD's reports are public downloads. COD-AB Senegal (OCHA/HDX) CC BY-IGO; Kontur CC BY 4.0;
Glottolog CC BY. CLEAR Global's shares CC BY-SA 4.0 (they rest on an IPUMS sample; used as a
placement weight, as in gn and sl).

## Moved from countries/sn.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- The census does not say whether these are languages of Senegal it did not list or languages of neighbouring countries.
