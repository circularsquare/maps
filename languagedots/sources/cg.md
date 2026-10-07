# Republic of the Congo (cg): record

**Drawn 2026-10-05** (session edd42a8c-mid). Afrobarometer R9 (2022-23) home language, French /
Kituba / Lingala answers read through the respondent's ethnic group, by département, on the
RGPH-5 2023 preliminary populations. 6,142,180 people, 12 départements, 20 nodes, every row
`modelled`. 6,133 dots at 1:1000.

```
python sources/cg_afro.py
python taxonomy/build.py
python tools/check_country.py cg
python scatter.py --country cg
```

Files: `sources/cg_afro.py`, `taxonomy/cg2023.py`, `taxonomy/tree.d/cg.txt`, `countries/cg.py`,
`data/normalized/cg.csv`.

## 1. What exists

- **Census**: RGPH 2007 and RGPH-5 2023 have no language question; 2023 has preliminary totals
  only. No language table in any year (coverage sweep).
- **Afrobarometer**: only R9 covers Congo-Brazzaville (checked R4-R9 country lists). 1,200
  respondents, all 12 départements (Brazzaville 464, Pointe-Noire 240, the rest 24-88). Q2
  "Language spoken in home": French 609, Kituba 275, Lingala 248, Teke 28, Lari 15, Other 25
  (verbatims: Laali 7, Mbosi 5, BaYaka 4, Likouba 2, Makoua 2, Bembe 2, Soundi, Koyo,
  Bomitaba). No R7-style mother-tongue item in R9.
- **WVS**: Congo not covered. **DHS 2011-12**: registration off. CLEAR Global's congo-languages
  (Afrobarometer-based, per the coverage sweep) not used: it is this same survey.
- **Population**: RGPH-5 preliminary results, 17 May 2023, by département (sum 6,142,180 exactly;
  press release 29 Dec 2023, carried by Les Echos du Congo-Brazzaville and ADIAC). Religiondots
  draws on RGPH 2007 (3,697,490) because its religion table is 2007's; nothing ties language to
  2007, so the newer count is used.

## 2. The lingua franca reading (ask 018, still open)

R9's "language spoken in home" records use: 51% French. The map draws first languages, so per
respondent (weighted, withinwt_hh):
- a named local language (Teke, Lari, a verbatim) is drawn as answered (68 respondents);
- French / Kituba / Lingala from someone whose ethnic group (Q84A) has its own language is drawn
  as that language (966): Kongo -> Kongo, Teke, Mbosi, Mbede -> Mbere, Echira -> Sira, Kota,
  Makaa, Fang, Autochtones -> Aka;
- French / Kituba / Lingala from someone with no such group is kept (166): national identity only
  79, Oubanguiens 33, Sangha 28, refused 19, don't know 7.

Result: Kongo 39.4%, Teke 20.6%, Mbosi 13.5%, French 7.8%, Lingala 7.1%, Mbere 3.2%, Makaa 1.6%,
Laari 1.5%, Kituba 1.4%, Kota 1.1%, Sira 1.0%, Aka 0.8%, the rest under 0.3%. The ethnic shares
match the usual national estimates (Kongo ~40%, Teke ~17%, Mbosi ~13%; CIA World Factbook).
Lingala, Kituba and French as FIRST languages are under-drawn (urban youth), said in note_public.

**2026-10-05, ask 018 closed** (Anita: lingua francas at Afrobarometer R7's mother-tongue
question, Q2A). Congo-Brazzaville is not in R7 (checked: R7's 34 countries), so there is no Q2A
to draw from. The reading above stands unchanged: French 7.8%, Lingala 7.1%, Kituba 1.4%.

## 3. Calls someone might reverse

- The ethnic reading itself (one switch: draw Q2 as answered and French is 51%).
- Lingala / Kituba / French kept for "refused" and "don't know" on ethnicity.
- Ethnic "Kongo" on one `kongo` leaf (Laari, Vili, Yombe, Beembe... not split).
- Oubanguiens and Sangha (multi-people categories) keep their lingua franca answer, mostly
  Lingala: Likouala 61% Lingala.
- "LAALI" verbatim read as Teke-Laali (all 7 in Lékoumou, Mayéyé and Sibiti), not Laari.
- "Autochtones" and "BaYaka" on Aka.

## 4. Room for improvement

A second Afrobarometer round, or any source with a mother-tongue item, would replace the
ethnic reading. Département samples of 24-88 outside the two cities are thin. The religiondots
hex layer's Kontur population disagrees with the census by department (Likouala 0.45x,
Cuvette-Ouest 5x in its own lookup); dots only use it inside a département.

## Terms

Afrobarometer: free download, citation requested. RGPH-5 preliminary figures as published by the
Ministry/INS. Glottolog CC BY. Kontur CC BY 4.0.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Kongo draws on 'Kongo (variety not given)' in a Kongo group that also holds the DRC's varieties, Laari and Suundi. Beembe, Vili and Kituba stay beside it. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
