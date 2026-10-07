# Chad: RGPH2 2009, first national language named, national, placed by région

Drawn 2026-10-05 (session edd42a8c-td). 8,088,816 people aged 6 and over who named a national
language, one national unit, 34 nodes (one per printed row). 8,070 dots at 1:1000, no rings.
715,710 people (8.8%) are `derived` (the Arabic move, §3); the rest `measured`.

```
python sources/td_rgph.py --fetch     # copies religiondots' PDF (SHA-1 pinned); all checks
python taxonomy/build.py
python tools/check_country.py td
python scatter.py --country td
```

## 1. What exists (searched 2026-10-05)

- **RGPH2 2009, *État et structures de la population*** (INSEED, Nov 2013, 210 pp), the volume
  religiondots draws Chad's religion from, already on disk (religiondots/sources/td.md has its
  Wayback route). Its chapter 5 holds what the coverage sweep missed:
  - **Tableau 5.10** (pp134-135): first national language spoken, people aged 6+, **national
    only**, urban/rural and sex, % to one decimal: 33 named rows and "Autres" (8.0%). Universe
    8,088,816 (1,833,267 urban, 6,255,476 rural). **Drawn.**
  - Tableau 5.09: 2.4% of the 6+ named no national language. Tableaux 5.11-5.13 and A5.01:
    second language and "at least one language" (Arabic spoken by 48.0% of the 6+), 13 groups.
    Not drawn (several answers; the single first answer is preferred).
  - **Tableau 5.02**: Chadian nationals by 21 ethnic groups (grand groupe), counts, national
    only (5.03-5.04 urban/rural). Read for §3.
  - Annexes 2-3 (pp206-210): which ethnicities and languages each row holds.
  - No table crosses language or ethnicity with région. The question is B18A/B18B, "langues
    nationales parlées" (first and second coded). French is official, not "national", so
    it is never an answer.
- **Other RGPH2 volumes**: IREDA (`ireda.ceped.org/inventaire/format_liste_operation.php?onglet=0&Chp6=tcd-2009-rec`)
  lists 16 volumes on the dead `inseedtchad.com/IMG/pdf/`. The *Résultats globaux définitifs*
  report (Wayback 2014-09-14, 44 pp, text layer) has no ethnic or language table. Its annex
  tables (`tableaux_annexes_rapport_resultats_definitifs_imp_27_fin.pdf`, Wayback 2014-09-14,
  3.5 MB) and the nomads and refugees reports were **not read**: Wayback returned 429/504
  all session. Worth one look; the global results' annex is most likely population by age.
- **RGPH 1993** (ODSEF's Fonds Gregory-Piché, linked from IREDA, free download, cite the
  original): *Rapport de synthèse* (c-doc_215) has ethnicity nationally (13 groups). Tome 2
  *État de la population* (cdoc_4004, scanned, 47 sheets) has **Tableau 30, the three largest
  ethnic groups and religions per préfecture** (p117); the scan skips pp118-119, so Moyen
  Chari, Ouaddaï, Salamat, Tandjilé and N'Djaména are missing. Volume II (results by
  préfecture, 15 tomes) is not online. Used for Arabic's seed (§4).
- **Microdata**: NADA catalog 26 is a data enclave (on site in N'Djaména, form, supervision).
  IPUMS blocked. Not used.
- **US Census Bureau** HDX geodatabases: 34 countries, Chad not among them.
- **Surveys**: Chad is in no Afrobarometer round (R4-R9 country lists checked) and not in the
  WVS. DHS (registration off); MICS 2019 not downloadable from INSEED's NADA; the World Bank's
  Chad entries are licensed/remote or unrelated. Nothing usable.

## 2. The counts (`sources/td_rgph.py`, all checks pass)

| check | result |
|---|---|
| file | 210 pp, 5,462,787 bytes, SHA-1 pinned, %%EOF |
| Tableau 5.10 parse = transcription | 34 rows x urban/rural/total |
| column sums | urban 99.9, rural 100.0, total 99.8 |
| urban/rural mix vs printed total column | worst 0.09 pp |
| counts (urban % x urban + rural % x rural, scaled to 8,088,816) vs total % x universe | worst 0.09 pp |
| Tableau 5.02 parse = transcription | 21 rows, sum 10,666,836 vs printed 10,666,833 |
| Tableau 5.07 régions | 22, sum 10,941,682, = religiondots' td.csv |

GPLANAT's code 28, Mouloui/Mousgoum, has no printed row in 5.10; the columns still sum to
~100, so it is inside "Autres" or tiny.

## 3. The Arabic move (ask 018's trap; switch `ARABIC` in `countries/td.py`)

21.8% named Chadian Arabic first; 12.9% of nationals are Arab by ethnic group. The southern
groups' languages match their ethnic shares (Sara group 26.6% both ways; Moundang 2.5/2.5;
Toupouri-Kéra 2.0/2.0; Marba-Lélé-Mesmé 3.0/3.0), so "first language named" reads as own
language there. The northern and central groups fall short (Kanembu 8.5 ethnic vs 5.8
language, Ouaddaï-Maba-Massalit-Mimi 7.2 vs 5.5, Hadjaraï 3.7 vs Moubi's 0.6 plus part of
Autres), and the shortfall net of "Autres" is the Arabic excess: they named the lingua franca.

**Drawn (`first_language`)**: Arabic at the Arab ethnic share x the universe (1,045,547). The
excess, 715,710, goes back by ethnic group: groups whose languages are all printed rows get
their full shortfall (162,895); groups with members filed in "Autres" share the rest, 47.9% of
their shortfall (the share that is not explained by "Autres"), split between their printed
languages and "Autres" by size. Result: Kanembu 5.8 -> 6.8%, Maba 3.6 -> 4.8%, Autres 7.9 ->
11.2%, Boulala 1.8 -> 2.4%, Massalit 1.5 -> 2.0%, Dadjo, Moubi, Tama up; the southern
languages barely move. `as_printed` draws Tableau 5.10 unchanged. Assumes the ethnic shares of
nationals hold for the 6+ universe, which includes 275,000 foreigners (mostly Sudanese
refugees in the east).

## 4. Placement (`countries/td.py`, a weight only; every count is national)

One unit, `TD`, on religiondots' Kontur 400 m hexes for the 22 régions of 2009 (read-only;
religiondots/sources/td.md §6 for the Sila/Ouaddaï boundary repair). `_place_unit` keeps the
région as `reg`. Inside a région every language follows Kontur.

Between régions:
- **N'Djaména** is set to Tableau 5.10's urban column (it holds 40% of urban Chad), with the
  Arabic move applied in national proportions: Sara 24%, Arabic 19%, Autres 11%, Gorane 8%.
- **The other 21** are split into Muslims and everyone else (Tableau 5.07 through
  religiondots' td.csv) and raked (IPF) to those populations (scaled to the 6+ universe) and to
  the national counts less N'Djaména. Seed: northern and central languages sit with Muslims,
  southern ones with everyone else, each leaking 0.15 into the other half (the census's
  southern-language total, 43.6%, exceeds its non-Muslims, 41.6%); times nearness to the
  language's Glottolog points (exp(-d/50 km), each point normalised alone, 95:5 with even;
  Sara uses ten points, Autres 27 points of languages Annexe 3 files there); Arabic instead
  takes RGPH 1993 Tableau 30's Arab share by préfecture (missing ones: the printed northern
  mean x Muslim share; Salamat, which the 1993 text calls Arab-predominant, Batha's 33.6);
  Fula an even seed, a fifth of it in Borkou, Ennedi and Tibesti.

Result (% of a région's 6+, top languages): Kanem Gorane 63, Kanembu 33 (1993: Gorane 50.9,
Kanem-Bornou 41.3); Lac Kanembu 72, Gorane 20 (1993: Kanem-Bornou 86.2); Batha Arabic 31,
Boulala 26; Guéra Autres 46 (the Hadjaraï languages), Arabic 16; Ouaddaï Maba 39, Massalit 19,
Arabic 15; Wadi Fira Tama 20, Autres 19, Arabic 18, Zaghawa 12; Salamat Arabic 43; Logone
Oriental Sara 85 (1993: 95.2); Logone Occidental Sara 67 (1993: 93.2); Mayo Kebbi Est Mousseye
20, Massa 12, Sara 10; Mayo Kebbi Ouest Moundang 26, Sara 19; Moyen Chari Sara 49, Sara Kaba 24.
`_TdWeighter(place).table()` prints the full table.

Known weaknesses: Sara overflows into Mayo Kebbi (10-19%, against 7.5% in 1993) because the
Sara régions cannot hold the national total at their non-Muslim size; Ennedi comes out 94%
Zaghawa (Glottolog has one Dazaga point, in Kanem, so Ennedi Ouest's Gorane are under-drawn);
Tibesti is 20% Arabic (4,000 people). All are placement, not counts.

## 5. Mapping and tree (`taxonomy/td2009.py`, `taxonomy/tree.d/td.txt`)

Every printed row is a node; Glottolog checks in the fragment's header. Calls:
- Arabe local on ne.txt's `afroasiatic.arabic.shuwa` (Chadian Arabic, chad1249).
- Gorane ("Gorane Daza") on ne.txt's Tubu leaf; Toubou (Teda) is inside "Autres".
- Kanembu a new leaf beside Kanuri; Maba, Mimi new under Maban; Tama and Daju (Dadjo) new
  leaves directly under Nilo-Saharan; Bagirmi, Bilala (Glottolog's Naba), Sara Kaba new under
  Central Sudanic; Day and Kim new under Adamawa; eleven Chadic leaves.
- Karo/Kado is not in Glottolog under either name; Chadic on the census's grouping with Zimé
  and Pévé (Masa branch). Least certain call.
- "Autres 1ere langues nationales parlées" on `africa_other`. Colours: five hand-picks in the
  fragment, for neighbours on the ground.

## 6. Calls someone might reverse

- Moving Arabic to the ethnic share (`ARABIC`, one line). `as_printed` gives Arabic 21.8%.
- The whole placement is an estimate; 1993 Tableau 30 is the only regional language-adjacent
  figure. `grain` and `note_public` say so.
- Drawing the 6+ universe (8.1M) rather than scaling to the 10.9M censused.

## 7. Room for improvement

A région x language or région x ethnicity table: RGPH2 microdata (enclave) or RGPH 1993's
Volume II préfecture tomes. The missing pp118-119 of the 1993 Tome 2 scan would complete
Tableau 30. The 2009 annex tables (Wayback, §1) are unread.

## Terms

INSEED volumes, public downloads, no licence text, cited. ODSEF Fonds Gregory-Piché: free,
"cite the original". Glottolog CC BY; Kontur CC BY 4.0.
