# Chad: MICS6 2019, mother tongue of the household head, by région

Drawn 2026-10-09 (session 32a047f0) from MICS6 microdata: 10,941,682 people (the 2009 census
population), 22 régions, 39 nodes, every row `modelled`. 10,923 dots at 1:1000, no rings. §0
is this build; §1-§7 are the national census build it replaced (2026-10-05 to 2026-10-09),
kept because its tables still supply the weights for the "other" answers.

```
python sources/td_mics.py             # reads data/raw/td/mics_2019/ and td_rgph.py's tables
python taxonomy/build.py
python tools/check_country.py td
python scatter.py --country td
```

## 0. MICS6 2019 (`sources/td_mics.py`, `taxonomy/td2019.py`, `countries/td.py`)

**Why.** The census build drew one national table and placed it by an estimate (Glottolog
points times religion, §4), so every région's mix was a guess. Anita made a UNICEF MICS account
(mics.unicef.org) and downloaded Chad's MICS6 2019; its SPSS files are in
`data/raw/td/mics_2019/` (gitignored: research use, no redistribution; the readme asks that
copies of publications go to INSEED and UNICEF Chad). 19,217 households sampled, 18,967
interviewed, 112,604 members; 769 clusters, 27-36 per province (N'Djaména 48, Ennedi Est and
Ouest together 57). No households in Tibesti.

**Item.** HC1B, "Langue maternelle du chef de ménage", read as every member's (hl.sav members x
hhweight). Its list: French, Chadian Arabic, Sar, Gorane, Kanembou, Maba/Ouaddaï, Moundang,
Massa, Peul, Lélé, Toupouri, Ngambaye, Zaghawa, other. Compared household by household:
- HH16 (the respondent's mother tongue, same list) agrees in 91.7% of households and, unlike
  Iraq's and Afghanistan's, does not slide to the interview language: Arabic is 13.82% of
  persons by HC1B and 13.79% by HH16, though 62% of interviews (HH15) were in Chadian Arabic
  (14% in French). Per région only Chari Baguirmi (Arabic 34.2 HC1B vs 28.9 HH16) and Wadi Fira
  (8.5 vs 16.6) differ by 3 points or more. HC1B is drawn, as in the other MICS countries.
- WM14 (women 15-49, own mother tongue): other 39.1%, Arabic 12.1%, Ngambaye 11.3, Sar 8.0,
  Kanembou 7.0, Gorane 5.5, Maba 5.2. Close to HC1B.

**"Other" (37.5% of persons).** Split by HC2, the head's ethnic group (the census's 19
grands groupes and "other", Annexe 2), towards the languages of that group the MICS list does
not name:
- one language: Gorane (Téda) -> Tubu; Zaghawa (Bideyat, Kobé) -> Zaghawa; Peul (Bodoré) ->
  Fula; Boulala/Médégo -> Boulala (Glottolog's Naba is the one Bilala-Kuka-Medogo language);
  Toupouri/Kéra -> Kéra.
- several census rows: Ouaddaï/Mimi -> Massalit, Mimi; Marba/Lélé -> Marba, Mesmé; Karo/Zimé ->
  Karo, Pévé; Sara -> the census's Sara row less Ngambay and Sar (Mbay, Gulay, Gor, Laka...),
  Sara Kaba, Daye, Mboum.
- census rows plus the group's unprinted languages: Massa -> Mousseye and Mousgoum (the group's
  only other member; census code 28, no printed row) on cm.txt's Musgum; Tama -> Tama and
  Assongori/Mararit (both Tamaic, on the `nilosaharan` root); Baguirmi, Dadjo, Bidio (the
  Hadjaraï), Gabri and the other ethnic groups -> their printed rows and an unnamed remainder
  on `africa_other`.
- nothing MICS does not already name: Kanembou/Bornou (Kanouri, Boudouma), Mesmédjé/Massalat,
  Moundang, no answer -> `africa_other`.

A candidate's weight in a région is its census count (Tableau 5.10 after §3's Arabic move; a
group's unprinted share is the part of its Tableau 5.02 count that neither its rows nor the move
account for: Bidio 212,170, other ethnic groups 149,429, Kanembou 133,252, Boulala 108,958,
Mesmédjé 81,915, Dadjo 77,493, Tama 30,555, Massa 23,694...) times its nearness to the région
(§4's Glottolog kernel). For Sara, the census row's weight is cut to 0.21 of itself, the part
left after MICS's Ngambay and Sar (18.8% of persons against the census row's 23.8%). So the
census decides between a group's languages, MICS how many of the group live where.

**Arabs who answered "other".** 676 of 2,880 Arab-headed households, concentrated in Salamat,
Sila, Ouaddaï, Wadi Fira and Batha. It is an interviewer habit, not another language:
interviewers in the same cluster disagree (within-cluster interviewer x answer, p = 2e-63); in
29 clusters one interviewer coded all their Arab households "other" and another none (246 of
the 676); in Salamat interviewer 63 coded 41 of 44, interviewer 84 none of 17. 95% of these
interviews were in Chadian Arabic. Drawn as Chadian Arabic (336,322 people). Most likely the
person named a tribe or a local name for their Arabic (Salamat, Missirié, Hémat), which the
questionnaire's "Arabe tchadien" box did not catch.

**Chadian Arabic as a second language.** MICS asks the mother tongue, and the census's
lingua-franca excess (§3) does not appear: Arabic is 17.5% (HC1B 14.4% plus the Arab "other"
3.1%), against 21.7% naming it first in 2009 and Arab ethnic shares of 12.9% (census) and 14.3%
(MICS). The 3 points above MICS's Arab share are 593 non-Arab heads who gave Arabic as their
mother tongue (Ouaddaï/Mimi 76, Kanembou 61, Tama 60, Boulala 56, Bidio 55, Gorane 54), most in
N'Djaména, Wadi Fira and Sila. They are drawn as they answered. §3's move (Arabic held at the
Arab share) is no longer needed; MICS mostly confirms it.

**Units and people.** MICS's 23 provinces are the 2009 régions with Ennedi split (Ennedi Est and
Ouest pooled here, with MICS's weights). Shares x each région's 2009 population (Tableau 5.07,
religiondots' td.csv), largest remainder: 10,941,682. The census build drew the 8,088,816 aged
6+ of Tableau 5.10; MICS's shares cover everyone, so the whole census population is used (the
map's people for Chad go from 40% to 54% of today's population). Tibesti (21,303) has no MICS
households and is drawn on Borkou's shares (Gorane 94%). Inside a région dots follow Kontur.

**Before and after** (% of a région; before = the census build's raked placement, §4):

| région | before | after (MICS) |
|---|---|---|
| N'Djaména | Sara 24, Arabic 19, other 11, Gorane 8 | Arabic 30, other 16, Ngambay 8, Maba 6, Gorane 6 |
| Logone Occidental | Sara 67, Marba 4, Lélé 4 | Ngambay 92, other Sara 2, Fula 2 |
| Logone Oriental | Sara 85 | other Sara 47, Ngambay 32, Sar 8, Mboum 4, Daye 4 |
| Mandoul | Sara 81, Daye 6 | Sar 74, other Sara 15, Daye 4 |
| Moyen Chari | Sara 49, Sara Kaba 24, other 16 | Sar 49, Sara Kaba 19, Arabic 11, other Sara 8 |
| Tandjilé | Sara 35, Marba 12, Mousseye 8 | Lélé 23, other Sara 16, Marba 14, Nangtchéré 8 |
| Mayo Kebbi Est | Mousseye 20, Massa 12, Sara 10, Moundang 10 | Mousseye 39, Massa 26, Toupouri 8, Marba 8 |
| Mayo Kebbi Ouest | Moundang 26, Sara 19, Mousseye 8 | Moundang 52, Ngambay 17, other 5 |
| Ouaddaï | Maba 39, Massalit 19, Arabic 15 | Maba 54, Arabic 20, Massalit 7, Tama 5 |
| Sila | Maba 35, Dadjo 19, Arabic 17 | Dadjo 32, Arabic 20, other 18, Moubi 12, Maba 10 |
| Wadi Fira | Tama 20, other 19, Arabic 18, Zaghawa 12 | Zaghawa 33, Tama 21, Arabic 17, Maba 15 |
| Salamat | Arabic 43, other 29, Rounga 9 | Arabic 76, other 7, Sara Kaba 6 |
| Batha | Arabic 31, Boulala 26, other 18 | Arabic 49, Boulala 41 |
| Guéra | other 46, Arabic 16, Dadjo 11, Boulala 11 | other 52, Arabic 19, Moubi 12, Dadjo 11 |
| Kanem | Gorane 63, Kanembou 33 | Gorane 54, Kanembou 40 |
| Lac | Kanembou 72, Gorane 20 | Kanembou 80, other 10, Gorane 7 |
| Ennedi | Zaghawa 94 | Gorane 64, Zaghawa 31 |
| Borkou / Tibesti | Gorane 38, Arabic 21 / other 48, Arabic 20 | Gorane 94, Arabic 5 (both) |

National, % (before of 8.09M aged 6+, after of 10.94M): Arabic 12.9 -> 17.5; Sara row 23.8 ->
Ngambay 10.3, Sar 8.6, other Sara 6.3 (the Sara group with Sara Kaba 25.5 -> 26.8);
unnamed (`africa_other`) 11.2 -> 7.4; Kanembou 6.8 -> 6.7; Gorane 6.9 -> 6.7; Maba 4.8 -> 5.5;
Mousseye 2.9 -> 3.2; Boulala 2.4 -> 3.1; Moundang 2.5 -> 2.9; Zaghawa 2.5 -> 2.5; Massa 1.7 ->
2.2; Fula 2.1 -> 1.2; Massalit 2.0 -> 0.8; Toupouri 1.4 -> 0.8; Karo 0.9 -> 0.3. The old
build's known weaknesses (Sara overflowing into Mayo Kebbi, Ennedi 94% Zaghawa, Tibesti 20%
Arabic) are gone.

**Still weak.**
- The survey's frame is households in census enumeration areas; nomads and refugee camps are
  probably thinner in it than in the 2009 census, which counted both (275,000 foreigners, mostly
  Sudanese refugees in the east). Fula (2.1 -> 1.2%) and Massalit (2.0 -> 0.8%) fall most; the
  camps' people are drawn on their région's mix. Not checked against the MICS report, which is
  not on disk.
- A cluster sample misses small concentrated groups: with 36 clusters, a group living in its own
  villages and holding 5% of a région is missed entirely about one time in six (0.95^36), 2%
  about half the time. The big languages are far above that; Rounga (Salamat), Kim, Mesmé and
  Buduma (Lac islands) are not measured by MICS at all and come only from the census weights.
- The "other" splits inside an ethnic group use national census proportions and a distance
  kernel, so Moubi reaches Sila (12%) on distance alone; the group's size per région is measured.
- Sara is a group since 2026-10-09 (`taxonomy/regroup.txt`, Anita's preference for grouping by
  family): Ngambay, Sar, Sara Kaba and "Sara (language not given)" (in Chad the other Sara
  languages, Mbay, Gulay, Gor, Laka...; CAR's and the diaspora's plain Sara too).
- Inside a région, dots follow population: Abéché's or Sarh's town mixes are spread over the
  région. MICS has urban and rural strata (HH6) that could be placed apart, as af does.

**Files.** `sources/td_mics.py` (new), `taxonomy/td2019.py` (new), `taxonomy/tree.d/td.txt`
(Ngambay, Sar new; Musgum and French repeated), `countries/td.py` (rewritten: pop_weight on the
22 régions; the old weighter and Arabic switch are gone), `sources/td_rgph.py` (now writes
`td_rgph.csv`; its tables feed td_mics.py), `data/normalized/td.csv`.

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
  WVS. DHS (registration off); MICS 2019 not downloadable from INSEED's NADA (but it is on
  mics.unicef.org with a free account: drawn since 2026-10-09, §0); the World Bank's Chad
  entries are licensed/remote or unrelated.

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

## 3. The Arabic move (ask 018's trap; census build only, superseded by §0; its group shares now weight §0's "other" split)

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

## 4. Placement (census build only, superseded by §0; its Glottolog kernel now weights §0's "other" split)

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
`_TdWeighter(place).table()` printed the full table (countries/td.py before 2026-10-09, in git).

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

(Census build; MICS6 closed the région x language gap, §0.) A région x language or région x
ethnicity table: RGPH2 microdata (enclave) or RGPH 1993's Volume II préfecture tomes. The missing pp118-119 of the 1993 Tome 2 scan would complete
Tableau 30. The 2009 annex tables (Wayback, §1) are unread.

## Terms

MICS6 2019 microdata: research use, no redistribution, copies of publications to INSEED and
UNICEF Chad (readme); only per-région shares are written to data/normalized/. INSEED volumes,
public downloads, no licence text, cited. ODSEF Fonds Gregory-Piché: free,
"cite the original". Glottolog CC BY; Kontur CC BY 4.0.
