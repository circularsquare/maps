# Algeria: Arab Barometer VI-VII ethnic group read as language, by wilaya

Drawn 2026-10-05 (session edd42a8c-dz). 34,080,030 people (RGPH 2008), 48 wilayas, 7 nodes,
every row `modelled`. 34,076 dots at 1:1000, no rings. Inside each wilaya, Berber dots lean to
where the 1966 census found Berber speakers.

```
python sources/dz_survey.py --fetch       # Algeria's rows from religiondots' .sav files (read-only)
python sources/dz_survey.py --check1966   # the comparison table in section 2
python taxonomy/build.py
python tools/check_country.py dz
python scatter.py --country dz
```

## 1. What exists

- **Census.** Only 1966 asked: mother tongue ("the language spoken in childhood"), published by
  wilaya and daira (C.N.R.P. 1970, 15 vols, not online). 2,287,997 Berber speakers, 19% of 12
  million. Claude Nesson, "La répartition des berbérophones algériens (au recensement de 1966)",
  Travaux de l'Institut de Géographie de Reims 85-86 (1994), pp. 93-107, quotes it daira by
  daira; its OCR text is in `data/raw/dz/nesson1994/` (persee.fr page fragments; the PDF sits
  behind a captcha). 1977, 1987, 1998, 2008 and 2022 ask no language.
- **Surveys asking a first or home language**: Arab Barometer II (2011, `q10191`), III (2013,
  `q1019_1`), IV (2016, `q1019a`), all "first language"; Afrobarometer R5 (2013) and R6 (2015),
  `Q2` "language of respondent". About 6,000 adults. Algeria is not in Afrobarometer R4, R7,
  R8 or R9.
- **Surveys asking ethnic group**: Arab Barometer VI part 3 (2021, `Q1012B`, 1,204) and VII
  (2021-22, `Q1012B`, 2,162): Arab / Amazigh / Tuareg / other. VIII has no Algerian rows in the
  file religiondots holds. Afrobarometer R6 also asks (`Q87`: Kabyle, Chaoui, Mozabite, Tergui).
- WVS (Algeria waves 4 and 6) not opened: the two survey families above already disagree in a
  way a fourth source would not settle, and the census settles it (section 2).

## 2. Which survey question, and why (the call most worth reversing)

The first-language surveys put Berber at 5-11% nationally (pooled 6.6%), a third of the 1966
census, and they fail in the Kabyle heartland: Béjaïa 26% Berber (1966: over 85% in all its
dairas), Khenchela 1 of 59 (1966: 72%), Oum El Bouaghi 0 of 62 (1966: about a third). Every
Afrobarometer interview in Algeria, Tizi Ouzou included, was in Arabic, by interviewers whose
home language was Arabic. The ethnic-group question puts Amazigh at 24% (24.2% in VI and in VII
separately) and follows the 1966 geography: r = +0.886 against it across the 48 wilayas, against
+0.674 for the first-language pool. So the ethnic group is drawn, read as language under the
brief's ethnicity ruling, and the first-language surveys are used only for French. Malawi's
precedent: rounds that disagree with the census are dropped.

Selected wilayas, % Berber (`--check1966` prints all 48):

| wilaya | 1966 census surface | ethnic group, VI-VII (drawn) | first language, 4 waves |
|---|---|---|---|
| Tizi Ouzou | 78.5 | 91.0 (119) | 69.2 (243) |
| Béjaïa | 71.6 | 91.0 (81) | 25.9 (93) |
| Khenchela | 51.9 | 65.5 (40) | 1.3 (59) |
| Batna | 45.4 | 28.5 (123) | 14.5 (209) |
| Bouira | 50.8 | 37.9 (62) | 8.2 (92) |
| Algiers | 25.4 | 39.9 (310) | 4.6 (550) |
| Ghardaïa | 29.5 | 28.7 (11) | 22.4 (72) |
| Oran | 5.0 | 14.6 (151) | 4.4 (260) |
| national, on 2008 populations | 19.3 | 24.3 | 6.6 |

**Retention is not applied.** The brief asks for the share of an ethnic group that speaks its
language. The only cross-table is Afrobarometer R6 (Kabyle 38 of 127 Berber at home, Chaoui 5 of
107, Mozabite 10 of 22), and it fails its own check: in Tizi Ouzou 24 of 32 named Berber as
their home language but 14 called themselves Arab. Applying it would draw Tizi Ouzou at a
quarter Berber. So Berber is probably overdrawn where people of Berber descent speak Arabic:
Algiers (40% against 30% in 1966), Oran, Laghouat, Ouargla, the West. Said in `note_public`.

**Population** is RGPH 2008 by wilaya, unscaled, as religiondots: ONS has published no wilaya
table since (2022's wilaya results have not appeared).

## 3. How the counts are made (`sources/dz_survey.py`)

Per wilaya: weighted answers of VI and VII pooled (weights normalised within wave; "don't
know" and refusals, 98, dropped), shrunk towards a prior worth K = 8 respondents. The prior is
the wilaya's own 1966 Berber share (from the surface in section 4) for Amazigh, the rest split
as the nation's non-Amazigh answers are; it matters only where the pool is thin (Tindouf 2,
Illizi 2, Tamanrasset 3, El Bayadh 4, Béchar 7, Adrar 9 respondents), where a national prior
made mostly of Kabyles would put a quarter of Tindouf on Berber. French: the first-language
waves' French share in the wilaya (ArB II-IV and AfB R6; 55 answers, 1.08%), shrunk to the
national share with K = 8, taken proportionally out of the other answers. Shares times RGPH
2008, largest-remainder rounding; every wilaya sums to its census population.

- **Afrobarometer R5's wilaya column is not used.** Its 36 REGION labels are the official
  wilayas 1-36 in code order, but 128 respondents sit under "Tamanghasset" (176,637 people), 130
  under "Tiaret", 30 under "Algiers", and the Berber answers under "Bouira" with none under
  "Tizi Ouzou". It counts in the national figures only. A trap for anyone using that column.
- **Wave VII's 22 "Tourag" answers** came from Djelfa (8), Laghouat (3) and ten other northern
  wilayas, none from Tamanrasset or Illizi (which VII did not sample). They go with "Other"
  (89 answers) on Algerian Arabic: neither names a language.
- Wave check, Amazigh share by wilaya, VI against VII: r = +0.605 across 32 wilayas with 15+
  respondents in each (Tébessa 4% in VI and 73% in VII is the worst pair; three clusters each).

## 4. Mapping, tree and placement

`taxonomy/dz2022.py`, `taxonomy/tree.d/dz.txt`. Arab -> Algerian Arabic (alge1239), a node
beside `arabic` as Morocco's Darija; colour one shade off Darija's mint. Amazigh goes on the
wilaya's Berber language where only one is spoken (spec section 3, a label whose meaning depends
on place): Kabyle in Tizi Ouzou, Béjaïa, Bouira, Boumerdès, Bordj Bou Arréridj and Sétif;
Chaoui (tach1249) in Batna, Khenchela, Oum El Bouaghi, Tébessa and Biskra; Tumzabt (tumz1238)
in Ghardaïa; Tamahaq (taha1241, not Mali's Tamasheq) in Tamanrasset and Illizi. Elsewhere,
Algiers above all (1.2M), it stays on the Berber group node, 3.4M people drawn as "language
not named". Sétif and Bordj Bou Arréridj as Kabyle is the shakiest of these: their Berber
speakers are mostly Kabyles of the Babors and Bibans, with some Chaoui in Sétif's south-east.

**Placement** (`countries/dz.py`): a 1966 surface, the inverse-square-distance mean of 73
daira seats' 1966 Berber share (`ANCHORS_1966`, located in religiondots' GeoNames file) with
a 2% background worth an anchor 40 km away. 29 seats have the article's own percentage (Arris
92.3, Khenchela 71.7, Batna 52, Bougaa 70.2, Lakhdaria 21.4, Algiers 30, Cherchell 70.4,
Tamanrasset 62.9, Ghardaïa 50...; the Kabyle dairas "over 85%" at 90). The rest are the article's
"null or very weak" dairas (and ports with a few thousand) at 2%, and six with a count but no share at a rough guess
(Constantine and Oran 5, Ouargla 10, Tébessa 10, El Aouinet 20, Timimoun 30). On 2008
populations the surface gives 19.3% nationally against the census's 19%. Inside a wilaya,
Berber dots weigh population x p and Algerian Arabic population x (1 - p), p the surface
scaled to the wilaya's drawn Berber share and capped at 1. Checked: Chaoui dots in Batna 109
of 241 in the Aurès box, 10 of 218 around Barika. French on population.

## 5. Calls someone might reverse

- Ethnic group over first language (section 2): `sources/dz_survey.py` would need a branch
  building from `LANG_X` instead; the comparison is printed by `--check1966`.
- No retention adjustment, so Berber outside its heartlands is overdrawn.
- Sétif and Bordj Bou Arréridj's Amazigh answers drawn as Kabyle; Biskra and Tébessa's as
  Chaoui.
- The 1966 prior for thin wilayas.

## 6. Room for improvement

A modern language question would replace this. Short of that: the 1966 volumes (C.N.R.P.
1970) would replace the surface's guessed anchors with the real daira table; MICS 2019
(language of household head, if asked) needs a UNICEF registration and was not tried; Arab
Barometer VIII's Algeria file, if it exists, would add a third ethnic wave.

## Terms

Arab Barometer: free download, citation requested. Afrobarometer: free download, citation
requested. Nesson 1994: read on persee.fr (open access); figures quoted, not republished as a
table. GeoNames CC BY 4.0; Kontur CC BY 4.0; Glottolog CC BY.
