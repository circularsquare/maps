# Djibouti (`dj`)

Session `fafd1067-dj`, 2026-10-03, under a supervisor. Taken from `queue.md`'s row in *Old
negatives re-checked under the survey and compiler rulings, 2026-10-03* (free, one national mix on
Pew 2020). Code: `sources/dj.py`, `sources/dj_geo.py`, `taxonomy/dj2024.py`, `countries/dj.py`;
node `other.dj` in `taxonomy/branches.py`. `sources.md` §dj-2026-10-03. No ask.

## 0. Outcome

**Drawn from the census, not a compiler.** The 2024 census (RGPH-3) asked every resident's religion
and INSTAD published it by région in counts on 18 November 2025. Six régions, 1,003,800 people
(residents of ordinary and nomadic households), every row `measured`: Islam 998,273 (99.45%),
Christianity 4,455, no religion 868, other 204. The 63,009 homeless and people in collective
households are in `gap` (`gap_share` 0.0591, hand-written). Nothing modelled, no foreigner layer.

## 1. How the closed record was wrong

Every earlier check looked for the 2024 *form* and found none (§11aq; §scout-2026-09-15-negatives;
§scout-2026-10-03-negatives), and read `instad.dj` as empty. It is a Nuxt app: the page HTML carries
`baseURL: https://instad-dj-6abc7b0eb612.herokuapp.com/`; the RGPH page's chunk (`/_nuxt/CV8M61dF.js`)
lists documents of category `RGPH` through `Dg7bgJRf.js`'s `getAll`, which is
`GET <baseURL>/fichiers/<category>/<year>`. `GET .../fichiers/RGPH/undefined` returns all 22
documents (Tomes 0-21 and the final report, uploaded 2025-11-25 to 2026-01-18) with Firebase
Storage links; no login. The listing is saved as `data/raw/dj/rgph3/_listing.json`. Other
categories (`Annuaire statistique`, `Démographie`, `Sociale`, ...) are served the same way.

The lead that found it: UNSD's Global Forum on Gender Statistics, October 2025, INSTAD's
presentation (`unstats.un.org/unsd/demographic-social/meetings/2025/genderstat-forum-10/docs/
Session6.3-Djibouti-Moktar-final.pdf`), which names three 2025 thematic reports.

## 2. The table

*Tome 4: Caractéristiques socioculturelles de la population*, chapter 4, **Tableau 42** (PDF
p.130): région and urban/rural by four answers, counts and column shares. Checked in `dj.py`:

- the text layer equals the transcription exactly; rows, columns and urban + rural close;
- every printed % is the column share (the prose misreads them as row shares; see the docstring);
- Tableau 41 (national), Tableau 43 (sex and nationality blocks) close on the same totals;
- Tableau 2: `P12_RELIGION` 1,003,800 expected, 1,003,800 valid, none missing;
- final report Tableau 14: each région's ordinary and nomadic household population equals
  Tableau 42's, households sum to 172,097;
- final report Tableau 7: 1,003,800 + 30,351 homeless + 32,658 collective = 1,066,809;
- final report Tableau 12: de jure by région; de jure minus the table is the gap, per région.

**The question** (Tome 4 p.124): eight codes, musulmane, catholique, protestante, orthodoxe,
animiste, athée, sans religion, autres religions; four kept for the tables. Which code went where
is not printed. The form itself is not among the 22 documents. Asked of every resident; the
chapter states no rule for young children.

| région | table | Christian | % | none | other | outside table | de jure |
|---|---:|---:|---:|---:|---:|---:|---:|
| Djibouti-Ville | 728,010 | 3,860 | 0.53 | 841 | 182 | 39,240 | 767,250 |
| Ali-Sabieh | 73,402 | 350 | 0.48 | 9 | 17 | 3,012 | 76,414 |
| Dikhil | 63,529 | 74 | 0.12 | 1 | 0 | 2,667 | 66,196 |
| Tadjourah | 57,338 | 94 | 0.16 | 10 | 4 | 3,307 | 60,645 |
| Obock | 35,648 | 23 | 0.06 | 0 | 0 | 11,734 | 47,382 |
| Arta | 45,873 | 54 | 0.12 | 7 | 1 | 3,049 | 48,922 |

By nationality (Tableau 43, national only): Djiboutians 925,503 (1,934 Christian), Ethiopians
51,481 (1,807), Somalis 12,537 (33), Yemenis 3,781 (10), Eritreans 1,679 (132), other foreigners
2,645 (475), stateless 6,174 (64). So foreigners are inside the measured table; no foreigner layer
(the Mauritania or Gulf construction) is needed.

**Witnesses outside the census, not used:** Pew 2020 Muslim 97.7%, Christian 1.1%, unaffiliated
1.1% (everyone, including the bases and the homeless); the census gives 99.45 / 0.44 / 0.09 for
ordinary households. Tome 14 (refugees and asylum seekers, read 2026-10-03, then deleted) puts them
at 96.9% Muslim. The provisional count (May 2024) had Djibouti-Ville 776,966 and Obock 37,666; the
final report moves 9,716 from the capital to Obock (767,250 and 47,382). The table is on the final
figures.

## 3. Surveys, as the brief asked

- **Arab Barometer** waves I-VIII (`data/raw/arabbarometer/`): Djibouti is a value label in wave
  II's shared codebook (`4. djibouti`) and has no rows in any wave. Wave IX unreleased.
- **Afrobarometer** R4-R9 on disk: not in any round. Afrobarometer's survey-resources page
  (read 2026-10-03, rounds 1-10) does not list Djibouti; it does list **Comoros and Chad**, which
  were in no round to R9, so R10 may cover them (not opened; a lead for `km`).

## 4. Calls someone might reverse

1. `Christianisme` on bare `christianity`: the form split Catholic, Protestant and Orthodox, the
   tables do not. The report's "mainly Catholic and Orthodox (Ethiopian)" has no figure.
2. `Autre religion` on a new `other.dj` (204 people), not `indigenous.african`: the report says it
   holds animism and some Asian religions, and does not say how many of each.
3. `Sans religion` on `unaffiliated`; atheist presumably merged in, nothing on `secular`.
4. The homeless and collective households (5.9%) in `gap`, not drawn on any mix. They include
   barracks; whether the foreign bases were enumerated at all is not stated anywhere read.
5. COD-AB (GADM 2022) régions; no finer tier, since nothing crosses religion below région.

## 5. Geography (`sources/dj_geo.py`)

- COD-AB `cod-ab-dji` is GADM's (ITOS-vetted, 2022). ADM1 six, names `Djiboutii` (sic) and
  `Tadjoura` pinned. ADM2 has 20 units that do not match the census's sub-prefectures (Tableau
  14's 30 rows), so no sub-région calibration.
- Area: COD's national 22,372 km2 against 1,066,809 / 46 (22,942-23,446). Tadjourah 6,659 against
  6,384-7,135 from its printed density. **Djibouti-Ville 195.6 km2 against 172.9** (1.17 over the
  national ratio, pinned); the densest 173 km2 hold 100.0% of its Kontur people.
- **Kontur DJ's région shares are 2009's.** Half-L1 against the 2009 census 0.071, against 2024
  0.192 (2009: DISED *Annuaire statistique 2012* Tableau 2.1.2). Per région over the national
  ratio: Djibouti-Ville 0.73, Arta 1.03, Obock 1.30, Dikhil 1.55, Ali-Sabieh 1.84, Tadjourah 2.47
  (pinned). The rural régions shrank between the counts (Tadjourah 86,704 to 60,645) while the
  capital grew 1.6x. Placement is inside each région, so no count moves. Seat test: each région's
  chief town holds Kontur people within 5 km of 1.4-3.5x its town count (capital 0.46 within 5 km of
  one point); the town's share of its région in Kontur is close to Tableau 14's (Tadjourah 38%
  against 31%, Ali-Sabieh 60% against 59%, Dikhil 49% against 41%, Obock 73% against 52%; Obock's
  homeless and collective quarter is not in the table).
- Hexes outside COD: 45,727 people snapped (coastal, max 1.71 km, 42,285 of them Djibouti-Ville);
  dropped 118 in `so_hexes`, 84 in Natural Earth's Somaliland, 42 Ethiopia, 7 Eritrea. No Kontur
  cap block (`kontur_cap.py dj`).

## 6. Reopen if

- INSTAD publishes the RGPH-3 microdata or a table by sub-prefecture or arrondissement with
  religion, or the Christian split (the form's three codes).
- The `Rapport thématique` or Tome 14 gives religion for the homeless or collective households.
- Eritrea is drawn: re-run `dj_geo.py` with `er` in `NEIGHBOUR_LAYERS` (7 people, no effect).

## 7. Review, 2026-10-03 (`fafd1067-rev12`, light pass)

Nothing to change. `dj.csv` sums to 1,003,800 and the note's shares, Christian counts and gap
(63,009 of 1,066,809, 5.91%) check against it; `other.dj` follows the other.gm/other.ne pattern.
Screenshot: dots at Djibouti-Ville, Ali-Sabieh, Dikhil, Tadjourah and Obock, none at sea. Fixed the
stale `queue.csv` row, which still read "one national mix", Pew 2020 and "RGPH-3 2024 form unread".
Eritrea is now drawn, so the reopen line above (`er` in `NEIGHBOUR_LAYERS`) is live; 7 people, left
as a note.
