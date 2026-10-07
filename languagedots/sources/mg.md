# Madagascar: RGPH-3 2018, languages spoken, region

**Drawn** 2026-10-05 (agent edd42a8c-mg). 25,674,196 people on 22 regions, one node (Malagasy),
25,674 dots at 1:1000, no rings. Every row `derived` (a multi-answer question, spec §3.6). The
coverage sweep had Madagascar as tier C (no question); the 2026-10-05 scout found this table
(`coverage/scout_2026-10-05.md`).

## Source

- INSTAT, Troisième Recensement Général de la Population et de l'Habitation (RGPH-3, 2018),
  thematic report *Compétences linguistiques et scolarisation à Madagascar* (file dated February
  2021, pages October 2021), 155 pages:
  https://www.instat.mg/documents/upload/main/INSTAT-RGPH3_CompetencesLinguistiquesetScolarisation_Fev-2021.pdf
  Fetched with a browser User-Agent, 11,101,312 bytes, sha256 pinned in `sources/mg_rgph.py`.
  No login; no licence stated (an official statistics office's published report, as elsewhere).
- The question: four yes/no items, "savoir parler" Malagasy, French, English, other language,
  for everyone aged 3 and over (non-response 642-685 of 23.5M, Tableau 1.1). Ability, several
  allowed. The report calls the fourth "autres langues étrangères" (p.17), and its summary table
  (PDF p.30) counts "at least one foreign language" as French, English or other.
- **Tableau 2.2** (PDF p.48): per region, the population aged 3+ and the % (one decimal) able to
  speak each. Percentages only. 22 regions of 2018 (Vatovavy Fitovinany one region).
- Population, all ages: RGPH-3 Tome 1, Tableau 6, read from religiondots'
  `data/geo/mg/mg_lookup.csv` (read only; its record says EXACT, sum 25,674,196).
- The report says Madagascar's people are eighteen ethnic groups "each speaking its dialect"
  (p.38, p.40); the census does not ask which. No dialect figure is drawn.

## Checks (`sources/mg_rgph.py`, all asserted, all pass)

1. Pinned digest.
2. 22 regions join one-to-one to religiondots' lookup by name (one apostrophe variant,
   Amoron'i Mania).
3. Regions' population 3+ adds to the printed national 23,507,970 exactly; Tableau 2.1's
   urban + rural (4,608,045 + 18,899,925) and male + female (11,580,496 + 11,927,474) equal it.
4. National shares from the regions weighted by population 3+: Malagasy 99.907 (printed 99.9),
   French 23.566 (23.6), English 8.212 (8.2), other 0.638 (0.6).
5. Population 3+ / Tableau 6 population per region 0.893 (Androy) to 0.928 (Analamanga),
   national 0.916: a wrong-twin guard on the join, since the two populations come from two tables.

## The split and the fold (`countries/mg.py`)

Spec §3.6 with AGENT_BRIEF §2's learned-second-language rule. Within each region the people 3+
are shared across the languages they can speak (`count = mentions * pop3 / sum(mentions)`).
That split gives, before folding:

| answer | able to speak (3+) | share the split gave | drawn as |
|---|---|---|---|
| Malagasy | 23,486,002 | 18,284,340 | Malagasy |
| Français | 5,539,855 | 3,832,062 | Malagasy |
| Anglais | 1,930,582 | 1,294,847 | Malagasy |
| Autres langues | 149,951 | 96,721 | Malagasy |

- **French, English and other languages fold into Malagasy.** The census itself calls all three
  foreign; ability rises steeply with schooling (French 4.8% at ages 3-5, 31.1% at 18-25,
  Tableau 2.1); about half of "other" speakers have tertiary education (52.5% of men, 48.8% of women) and
  about 60% are migrants (Tableau 2.6).
  French as a first language exists in Madagascar (some urban families in Antananarivo, the
  French community) but no source counts it, and it is nowhere near a sizeable group against
  5.5M French speakers; so not flagged further.
- **"Autres langues" folds too.** It may hold some first-language speakers of Comorian
  (Mahajanga, Diana), Gujarati or Chinese, but the census files it as foreign and does not
  separate them; 96,721 people's shares at most.
- **People who speak no Malagasy** (0.1% as printed, so roughly 10,000-35,000; the table's one
  decimal cannot say) are drawn as Malagasy, said in `note_public`.
- **Under-3s and collective households** (Tableau 6 less Tableau 2.2's base, 2,166,226, 8.4%) are
  drawn as Malagasy, as El Salvador's under-3s are as Spanish (`sources/sv.md`).
- `countries/mg.py` asserts the rows add back to each region's population.

## Geography

religiondots' `mg_hexes.gpkg` (Kontur MG 2023-11-01, 400 m hexes, 22 census regions; MG27 and MG34
of COD-AB 2026 dissolved back into MG26 and MG32, see `religiondots/sources/mg.md`). Same units and
ids as this table, so no new join. Its Kontur cap rows (religiondots' `kontur_cap.csv`) applied as
religiondots has them: three blocks capped, two real, two unreviewed in Antananarivo (under 3% of
Analamanga each), drawn as Kontur has them. Water clip reused religiondots' cache.

## Mapping (`taxonomy/mg2018.py`, `taxonomy/tree.d/mg.txt`)

- Malagasy → `austronesian.malagasy`; Français → French; Anglais → English; Autres langues →
  `other`. No new nodes; the fragment repeats the ones used.
- **Known problem, not fixable in this country's files:** `austronesian.malagasy` has a child,
  ca.txt's `austronesian.malagasy.merina` (Canada's census prints "Merina" beside "Malagasy,
  n.o.s."), so build.py treats it as a group and draws its own people washed out (`#99baaa`
  instead of `#4dbc92`). That is the named-label-on-a-group bug of AGENT_BRIEF §3, here for the
  whole country. Two fixes, neither mine to make: add `austronesian.malagasy` to `UNWASHED` in
  `taxonomy/build.py` (Anita's file; the Chinese precedent); or, as ru.txt does for Mari and
  Mordvin, make Canada's Merina a sibling leaf (`austronesian.merina`) in ca.txt and ca2021.py and
  re-scatter Canada. Flagged to the supervisor.

## Calls someone might reverse

- All of French, English and other drawn as Malagasy (the rule; Comorian inside "other" is the
  only real first language plausibly lost, unmeasurable).
- Under-3s and collective households drawn as Malagasy rather than put in `gap`.

## Not done

- Tableau 2.15 (literacy by language, 11+) exists; literacy is not a buildable question.
- No finer grain: the report has nothing below region. The RGPH-3 microdata (district or commune)
  were not looked for; with every answer folding to Malagasy a finer grain would only move dots
  within regions that Kontur already places.

## Cut from note_public (2026-10-06 text sweep)

- French is spoken by 55.7% of those asked in Analamanga (Tableau 2.2), against 23.6% nationally.
- Dots inside each region follow Kontur's 2023 population.
