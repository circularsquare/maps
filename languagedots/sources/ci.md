# Côte d'Ivoire: RGPH 2021, Ivorian language spoken most, by région

Drawn 2026-10-05 (session d9e44929-ci). 21,295,162 people (Ivorians aged 3 and over), 33 units,
15 nodes from 15 table categories. 21,288 dots at 1:1000, no rings.

**Redrawn 2026-10-05 (session edd42a8c-ci, ask 011 approved):** the 18% remainder is shared out
into the 82 labels annex 20 names nationally, rows `modelled` (§"The share-out" below). 94 nodes,
21,249 dots, 4 rings. Needs the build tail.

## Source

- **Table.** ANStat (Agence Nationale de la Statistique), RGPH 2021, *Rapport thématique tome 1:
  État et structure de la population*, Tableau 4.20 "Répartition de la population ivoirienne
  par la langue la plus parlée selon les régions administratives" (pp106-107), with annexes 20,
  21 and 25. The same 151-page volume religiondots draws Côte d'Ivoire's religion from.
- **Route.** `anstat.ci` is Cloudflare and 403s every script, including `/assets/*.pdf`; the
  Wayback Machine has `https://www.anstat.ci/assets/publications/files/rgpg_tom1.pdf` (take the
  largest capture; the first-listed is cut at exactly 2^20 bytes and opens anyway, see
  `religiondots/sources/ci.md` §2). `ci_rgph.py --fetch` copied religiondots' verified download
  (34,204,439 bytes, %%EOF) into `data/raw/ci/`, and falls back to the archive itself.
- **Question** (tome 1 §1.2.1.7): the national (Ivorian) language each person uses most to
  communicate, "distinctement du français qui constitue la langue officielle". One answer each.
  French is not an answer; "Aucune langue nationale parlée" is. Asked of everyone aged 2 and
  over, but every published table is for **Ivorian nationals only**, and the age tables start
  at 3: the universe is 21,295,158 of 22,840,168 Ivorians.
- **Vintage.** 2021 census (the latest). **Grain.** 33 units: 31 régions and the autonomous
  districts of Abidjan and Yamoussoukro, 645,000 Ivorians on average. No finer table exists in
  tome 1 or the *Résultats globaux définitifs* (religiondots checked both for religion; the
  language tables in tome 1 are national, by milieu, by age, by ethnic group and by région).
- **Not used: data.gouv.ci.** The coverage sweep's lead, data354's dataset "Répartition de la
  population ivoirienne par la langue la plus parlée selon le milieu de résidence et le groupe
  d'âge" (Licence Ouverte, origin ANStat), is Tableau 4.17 re-keyed badly: columns shifted, a
  row emptied, digits changed (Baoulé in Abidjan 988,959 where the tome prints 688,989). Its
  "Baoulé 16.1%" nationally, which the sweep quoted, is the tome's 20.1%. A trap for anyone
  taking the portal's CSV.

## How the counts are made

Tableau 4.20 is percentages to 1 dp. Each région's count = share x its Ivorians (annex 25, all
ages) x 21,295,158 / 22,840,168 (one national factor for the children under three), then each
category is rescaled so its 33 régions sum to its national count in annex 20. The rescale
factors measure how well shares and denominators agree; all are within 1.6%:

```
Baoulé x1.0005   Dioula x0.9997   Senoufo x0.9964   Malinké x0.9975   Agni x1.0054
Dan x0.9929      Bété x0.9986     Attié x1.0156     Lobi x0.9884      Gouro x1.0048
Abbey x1.0019    Koulango x0.9967 Guéré x1.0007     Aucune x1.0066    autres x0.9993
```

## Checks (`python sources/ci_rgph.py`, all pass)

1. The PDF: 151 pages, %%EOF.
2. Tableau 4.20: 33 régions + "Ensemble CI", each once; every row's 15 shares sum to its Total
   within 0.2 (bound 0.8).
3. Annex 20: 96 labels (94 languages, "autre langue nationale à préciser", "aucune"); Abidjan +
   other towns = urban and urban + rural = total on every row within 2 (41 rows off by 1-2, the
   cells are rounded); labels sum to 21,295,160 against the printed 21,295,158.
4. Annex 21 (same labels by age), the second table: identical label set, 13 totals differ by at
   most 6 people (Baoulé 4,288,532 against 4,288,538).
5. Tableau 4.20's Ensemble CI row equals annex 20's counts over 21,295,158 to 1 dp, all 15.
6. The 82 labels outside the 13 named and "aucune" sum to 3,850,778; Tableau 4.17 prints
   3,850,775.
7. Annex 25: 33 régions, 22,840,169 Ivorians (published 22,840,168); each row's groups sum to
   its total within 3.
8. Tableau 4.20's 33 names = religiondots' 33 `ci_hexes.gpkg` units, one to one, on folded
   names; annex 25 joins by suffix match on its uppercase names, with aliases for "DISTRICT
   AUTONOME D'ABIDJAN" and "DISTRICT AUTO DE YAKRO", each unit matched exactly once.

## Labels (`taxonomy/ci2021.py`, `taxonomy/tree.d/ci.txt`)

| census | node |
|---|---|
| Baoulé, Agni, Akyé ou Attié, Abbey | `nigercongo.kwa.{baoule, anyi, attie, abbey}` (new) |
| Dioula, Yacouba ou Dan | `nigercongo.mande.dyula`, `.dan` (existing, pl.txt and au.txt) |
| Malinké ou Malinka, Gouro | `nigercongo.mande.{maninka, guro}` (new) |
| Senoufo, Lobi, Koulango | `nigercongo.gur.{senufo, lobi, kulango}` (new) |
| Bété, Guéré | `nigercongo.kru.{bete, guere}` (new) |
| Ensemble des autres langues nationales parlées | `nigercongo` |
| Aucune langue nationale parlée | `other` |

Glottolog puts Kru and Mande in families of their own and Senufo and Kulango outside Gur;
they are under Niger-Congo and Gur here, as most readers know them and as earlier fragments
already have Kru and Mande. "Senoufo" is one answer for the Senufo cluster (Tagbana, Djimini,
Niarafolo, Nafana and Palaka are printed apart nationally), so one leaf.

**The remainder.** "Ensemble des autres langues nationales parlées" is 3,850,778 (18.1%). Annex
20 names its 82 labels nationally: every one an Ivorian Niger-Congo language (largest: Koyaka
274,612, Abron 269,660, Tagbana 261,418, Mahou 260,034, Dida 235,373, Worodougouka 114,127 ... )
plus "Naturalisé" (9,408, naturalised citizens, no language given) and "à préciser" (13). So
`nigercongo`, the narrowest node holding them, drawn as "language not named". It is most of
Hambol (66.9%), Bafing (66.7%), Worodougou (49.6%), Grands-Ponts (45.5%) and Béré (43.4%).
Ask 011 offered to share it out by a model; Anita approved it, see the next section.

## The share-out (`sources/ci_model.py` -> `data/normalized/ci_model.csv`, ask 011)

Each région's remainder (the "Ensemble des autres..." rows of `ci.csv`) and each of the 82
labels' national counts (annex 20) are both met exactly by iterative proportional fitting of a
labels x régions table (worst miss 0.02 people). Only the seed is borrowed:

- **local**: the région's Kontur population near the label's anchor, `pop x exp(-d / 10 km)`.
  Anchors: 60 labels on Glottolog points (every code looked up in `languages.csv`; the
  non-obvious matches, e.g. Ehotilé = Beti eot, Guébié = Gabogbo gie, Gagou = Gban, Komono =
  Khisa, Lohron = Téén, Tchebara/Fodonon/Koufoulo = Glottolog's dialects of Cebaara, Baralaka and
  Finanga = Mahou dialects, are listed in the script's GLOTTOCODE comment); Andoh on Iffou
  (fr.wikipedia "Ano (peuple)": Prikro department); Djamala on Djimini's point (an academia.edu
  history of the Djimini and Djamala at Dabakala).
- **background**: annex 27 (language x the census's five ethnic macro-groups, read and checked
  against annex 20 row by row) times annex 25 (régions x the same groups): where that language's
  speakers' group lives. This puts migrants in Abidjan and the cocoa south-west.
- **seed = 0.5 local + 0.5 background**; 21 labels (247,860 people) take the background alone:
  Bambara, Peul, Foula (migrant languages, Glottolog points outside the country), 16 small labels
  no source located (Gandjé, Sokyia, Souaminlin, Gbonzron, Mangoro, Winnin, N'garadougouka,
  Komara, Ouadougou, Ouodougou, Kotrohou, Kouzié, Sia, Conja, Doma, Samogho), Naturalisé and
  "à préciser".

**Choosing K and the mix.** `--calibrate` runs the same model on 12 of the 13 languages Tableau
4.20 does give by région (Dioula left out, a trade language with no home area) and compares with
the census. Mean share of a language's speakers put in the wrong région:

```
             B=0    0.1   0.2   0.3   0.5   0.7   1.0 (background only)
K= 10 km    46.1  31.7  29.2  27.6  27.0  29.1  38.2
K= 40 km    41.9  35.5  33.6  32.6  32.0  32.8  38.2
K=160 km    44.4  43.5  42.7  41.8  40.1  38.9  38.2
```

K = 10 km, B = 0.5 is in use: 27% misplaced on average (24% people-weighted), from Senoufo,
Malinké and Dan at 13-14% to Agni, Abbey and Gouro at 34-40%. Abidjan is not where most of the
error is (8-34% of each language's). A point anchor beats the ethnic groups alone (38%), and the
mix beats either. Baoulé's Glottolog point is on the coast near Dabou, not in its heartland.

**What it gives** (share of the région's remainder): Bafing Mahou 76% and Baralaka 12%; Hambol
Tagbana 47%, Djimini 35%; Béré Koyaka 57%, Koro 15%; Worodougou Worodougouka 49%, Koyaka 19%;
Gontougo Abron 62%, Nafana 15%; Loh-Djiboua Dida 57%; Guémon Wobé 55%; Iffou Andoh 69%; Grands-
Ponts Adjoukrou 41%; Agnéby-Tiassa Abidji 50%; Sud-Comoé N'zima 31%, Abouré 25%; San-Pedro
Kroumen 31%; Tchologo Niarafolo 35%; Bounkani Lorhon 33%, Birifor 18%. Abidjan's 905,470 are
spread thin (Ebrié 9% the most).

**Nodes** (`taxonomy/ci2021.py` SHARED, `tree.d/ci.txt`): one leaf per label, under the
language's own branch (Glottolog, conventional levels), else the census's macro-group of most of
its speakers. Yaouré and Ngain (Beng) are Mande though their people are counted Akan; Aïzi is Kru
(Glottolog aizi1248) though counted Akan; Samogho is a Mande cluster though counted Gur. The
Senufo languages printed apart (Tagbana, Djimini, Niarafolo, Nafana, Palaka, Tchebara, Fodonon,
Koufoulo) sit flat under Gur beside the "Senoufo" leaf. Gagou and Gban, two names of one language,
share `mande.gban`. Conja and Doma have no branch any source gives: leaves directly under
`nigercongo`. Peul and Foula are cf.txt's `atlantic.peulh` and `atlantic.fulah`. Naturalisé
(9,408, no language) goes on `other` beside "Aucune"; "à préciser" (13) on `nigercongo`.

**Colours.** New leaves generated, 20 hand-picked where a pair at 2%+ of a shared région sat
under 0.05 OKLab; Nzema (pl.txt's bare node) recoloured away from Attié. Closest remaining pairs
0.052-0.059 (Birifor/Lorhon in Bounkani, Dyula/Koyaka in Béré). Logged in `COLOURS.md`.

**"Aucune langue nationale parlée"** (722,167, 3.4%; Abidjan 7.5%, 60.7% of the "autre ethnie"
group and 14.1% of naturalised citizens in Tableau 4.18): Ivorians who use no Ivorian language.
French is the likeliest for most, since the question leaves French out, but the census names no
language, so it sits on `other` (as CAR's "langues non centrafricaines").

**Colours.** Hand-picked for all 13 (see the fragment's header). Every pair of named languages
at 2% or more in a shared région is at least 0.079 apart in OKLab. Dyula and Dan had no colour
before (they appear only in Poland's and Australia's tiny foreign-language rows).

## What it shows

Baoulé across the centre (N'Zi 85%, Bélier 83%, Yamoussoukro 60%, Gbêkê 55%) and far into the
cocoa west (Nawa 45%, Gboklè 42%, Goh 28%, San-Pedro 24%). Senoufo in Poro (63%), Bagoué (55%)
and Tchologo (38%); Malinké in Folon (69%) and Kabadougou (50%); Lobi in Bounkani (58%) with
Koulango beside it (Gontougo 41%); Agni in Moronou (76%) and Indénié-Djuablin (47%); Attié in
La Mé (66%); Abbey in Agnéby-Tiassa (27%); Dan in Tonkpi (58%); Guéré in Cavally (31%) and
Guémon (24%); Gouro in Marahoué (33%); Bété in Goh (17%) and Haut-Sassandra (13%). Dioula, the
trade language, is in every région and is a fifth or more in Kabadougou, Folon, Tchologo,
Bagoué, Loh-Djiboua and the towns.

## Limits

- **Foreign nationals are not drawn**: 6,460,062 residents (22% of the census) were asked the
  question but are in no published table. Against religiondots' census totals per région
  (`ci_lookup.csv`), they are 44% of Cavally, 40% of San-Pedro, 36% of Nawa and 21% of Abidjan
  but 9% of Poro, so the cocoa south-west looks thinner than it is. In `gap` and `note_public`.
- **Children under three**, about 1.5 million Ivorians, were not asked. One national factor
  removes them; régions with more young children are slightly overstated, which the per-category
  rescale (all within 1.6%) bounds.
- **Placement** inside each région is by Kontur population (religiondots' layer), which includes
  the foreign residents; nothing says where inside a région each language's speakers live. That
  holds for the share-out too: the model only says how many per région.
- **The share-out's régions are modelled**: about a quarter of each language's speakers would be
  in the wrong région, judging by the 12-language test. `note_public` says so.
- **The question is "spoken most"**, so Dioula is drawn where people use it day to day, not
  only where it is a first language. `note_public` says so.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Abron (Bono) is in the Akan group. Baoulé and Anyi stay apart: Glottolog puts both in Bia, a sister of Akanic, so Ghana's Akan stopping at this border is right. Dyula, Maninka, Bambara, Mahou, Koyaka, Worodougouka, Koro and Wojenaka (with Odienneka) are in a Manding group with Mali's, Guinea's and Burkina's. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
