# Algeria: Arab Barometer VI-VII ethnic group read as language, by wilaya

Drawn 2026-10-05 (session edd42a8c-dz). 34,080,030 people (RGPH 2008), 48 wilayas, every row
`modelled`. Inside each wilaya, Berber dots lean to where the 1966 census found Berber speakers.

**Reworked 2026-10-19** (session eca0185b-ber, Anita: "berber in africa"): (1) the Amazigh answers
that sat on the Berber group node (3.45M, drawn "language not named") are split over Algeria's
Berber varieties by place, 13 nodes now, section 4; (2) the Sahrawi camps near Tindouf (173,600) are
added as Hassaniya, section 7; (3) stage 2, the other African Berber countries, section 8.
34,253,630 people drawn, 34,247 dots at 1:1000, 1 ring (Tachelhit, 765 people).
**2026-10-21**: a retention cut outside the Berber homelands (section 9) brings Berber from 23.1%
to 16.6%; the figures in sections 4 and 8 below are from 2026-10-19, before the cut, and section 9
has the current ones.

```
python sources/dz_survey.py --fetch       # Algeria's rows from religiondots' .sav files (read-only)
python sources/dz_survey.py --check1966   # the comparison table in section 2
python sources/dz_camps.py                # placement layer with the Tindouf camps (section 7)
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

**Retention was not applied in the first build** (2026-10-05); it is, outside the homelands, since
2026-10-21 (section 9). The brief asks for the share of an ethnic group that speaks its
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
beside `arabic` as Morocco's Darija; colour one shade off Darija's mint.

**The Berber split (2026-10-19).** The survey asks Amazigh or not, never which language, so each
wilaya's Amazigh count is cut over Algeria's Berber varieties by place, in `SPLIT` in
`sources/dz_survey.py`. The varieties are Glottolog's languages of Algeria: Kabyle (kaby1243),
Chaouia of the Aures (tach1249), Tumzabt (tumz1238), Western Algerian Berber (chen1266, drawn as
Chenoua; Beni Snous is its dialect in Glottolog), Ouargli (taga1278), Oued Righ (tema1243),
Sud Oranais-Gourara (tazn1238, drawn as Taznatit; Gourara and Touat are its dialects),
Tidikelt-Tuat Tamazight (tidi1241), Tahaggart Tamahaq (taha1241), and Tachelhit (tach1250, which
Glottolog lists for Algeria). Every call is by place from Glottolog's points and general knowledge
of who lives where. **No wilaya-level source with a Berber variety turned up** (searched: the
1966 volumes are not online, Nesson 1994 gives speaker counts by daira without naming varieties,
no survey asks the variety, ONS has published nothing since), and the published speaker estimates
for the varieties are Ethnologue's, which disagree widely (Kabyle 2.5M to 6.4M in Algeria). So
the shares inside a wilaya are judgement, said so in `how`. Rows stay `modelled`.

Totals at the 2026-10-19 split, before the retention cut of section 9 (people, share of the 7,888,896 drawn Amazigh): Kabyle 5,421,509 (69%), Chaoui
1,555,003 (20%), Chenoua 401,952 (5%), Tumzabt 150,085, Ouargli 102,488, Taznatit 97,128,
Tamahaq 76,778, Oued Righ 71,179, Tidikelt 12,009, Tachelhit 765. The old build had 3,448,672 on
the group node (Algiers 1,150,898 above all), now none.

| kind of wilaya | wilayas | rule |
|---|---|---|
| one Berber language, kept from the first build | Bejaia, Bouira, Tizi Ouzou, Setif, Bordj Bou Arreridj, Boumerdes: Kabyle. Oum El Bouaghi, Batna, Biskra, Tebessa, Khenchela: Chaoui. Ghardaia: Tumzabt. Illizi: Tamahaq | unchanged (Setif and BBA as Kabyle remains the shakiest) |
| two varieties spoken in the wilaya | Tamanrasset: Tamahaq, with Tidikelt around In Salah. Adrar: Taznatit (Gourara, Touat), Tidikelt at Aoulef. Ouargla (2008: includes Touggourt): Ouargli and Oued Righ | cut by the wilaya's Berber mass (hex population x the 1966 surface) on each side of the line half way between the two Glottolog points (Tamanrasset 58,845 Tamahaq, 2,700 Tidikelt; Adrar 34,715 and 9,309; Ouargla 102,488 and 33,929) |
| a local variety only | Chlef (Dahra, Ouarsenis Zenati): Chenoua. Bechar, El Bayadh, Naama (Sud Oranais, Figuig side): Taznatit. El Oued (the lower Oued Righ): Oued Righ. Tindouf: Tachelhit (Soussi traders; 765 people) | all of the wilaya's Amazigh |
| the capital and the centre: the Kabyle diaspora | Algiers (the task's own call: Algiers' Kabyle community; the 1966 Alger-ville Berbers were Kabyle migrants), Medea, Djelfa, Jijel: Kabyle. Blida: Kabyle 80%, Chenoua 20% (the Blida Atlas). M'Sila: Kabyle 70%, Chaoui 30%. Laghouat: Kabyle 50%, Tumzabt 50% (the Mozabite trading community) | round shares |
| the west | Tipaza (Chenoua's home): Chenoua 85%, Kabyle 15%. Tissemsilt (Ouarsenis): 80/20. Tlemcen (Beni Snous), Ain Defla (Zaccar, Miliana), Relizane: Chenoua 50%, Kabyle 50%. Oran, Mostaganem, Mascara, Sidi Bel Abbes, Ain Temouchent, Saida, Tiaret: Chenoua 25%, Kabyle 75% (no Berber of their own to speak of; Kabyle settlers) | round shares |
| the east | Souk Ahras: Chaoui. Annaba, Guelma, El Tarf: Chaoui 80%, Kabyle 20%. Skikda, Constantine, Mila: Chaoui 50%, Kabyle 50% | round shares |

Why these and not a nearest-Glottolog-point rule everywhere: worked through on paper first. It
sends Algiers and Blida to Chenoua (Cherchell is nearer than Tizi Ouzou), Djelfa to Tumzabt, and
puts Oran wholly on Chenoua. The city Berbers of Algeria are mostly Kabyle and Chaoui settlers,
which a point rule cannot know. The point rule is kept only for the three wilayas that hold two
local varieties.

**Overdrawn before the 2026-10-21 cut (Chenoua is 193,044 after it):** Chenoua at 401,952 is five times Ethnologue's 76,000;
the extra is the west's Amazigh answers (the survey counts identity, not speech), a quarter to a
half of which are put on the local Zenati variety instead of all on Kabyle. Western Algerian
Berber is the honest name for the group (Beni Snous, Ouarsenis, Dahra, Chenoua), so the node reads
"Chenoua (Western Algerian Berber)". Ouargli (102,488) is several times Ethnologue's figure for
the same reason. Taznatit (97,128), Tumzabt (150,085) and Tamahaq (76,778) are in line with the
published estimates (Gourara about 100,000; Mzab 150,000 to 200,000; Algeria's Tuareg 50,000 to
100,000), none of which I could source better than Ethnologue and Wikipedia.

**Colours** (hand-set in `tree.d/dz.txt`, OKLab distance): Chenoua `0.64 0.15 125` olive green
beside Kabyle's orange (0.196) and Algerian Arabic's mint (0.188); Oued Righ `0.78 0.17 38`
salmon, between Chaoui (0.216) and Ouargli (0.211); Ouargli `0.60 0.13 78` dark gold-brown beside
Tumzabt (0.103); Taznatit `0.82 0.17 78` amber (Tidikelt 0.211, Tumzabt 0.127); Tidikelt
`0.62 0.15 55` dark orange (Tamahaq 0.298, Taznatit 0.211). Kabyle, Chaoui, Tumzabt, Tamahaq and
Tachelhit unchanged (build.py HAND). Closest pair that meets on the ground: Ouargli and Tumzabt
0.103. Not judged on the map.

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
of 241 in the Aurès box, 10 of 218 around Barika. French on population. Where a wilaya has
several varieties in set shares they all use the same surface x population weight; where it has
two local ones (Adrar, Tamanrasset, Ouargla) each is kept to the hexes nearer its own Glottolog
point (`countries/dz.py` `_Dz1966Weighter.var`). Checked on the dots: Ouargla box 97 Ouargli,
40 Oued Righ; Adrar box 35 Taznatit, 9 Tidikelt; Algiers box 1,057 Kabyle, 2 Chenoua.

## 5. Calls someone might reverse

- Ethnic group over first language (section 2): `sources/dz_survey.py` would need a branch
  building from `LANG_X` instead; the comparison is printed by `--check1966`.
- No retention adjustment, so Berber outside its heartlands is overdrawn.
- Sétif and Bordj Bou Arréridj's Amazigh answers drawn as Kabyle; Biskra and Tébessa's as
  Chaoui.
- The 1966 prior for thin wilayas.
- The 2026-10-19 Berber split (section 4): every share inside a wilaya, above all the 50/50 and
  25/75 western and eastern ones, Algiers wholly Kabyle, Laghouat half Mozabite, and the Chenoua
  total. Reversing is editing `SPLIT` in `sources/dz_survey.py` and re-running it.
- The camps' figure and equal shares (section 7).

## 6. Room for improvement

A modern language question would replace this. Short of that: the 1966 volumes (C.N.R.P.
1970) would replace the surface's guessed anchors with the real daira table; MICS 2019
(language of household head, if asked) needs a UNICEF registration and was not tried; Arab
Barometer VIII's Algeria file, if it exists, would add a third ethnic wave.

## 7. The Tindouf camps (2026-10-19)

The five Sahrawi refugee camps (Awserd, Boujdour, Dakhla, Laayoune, Smara) are in no census: the
2008 count of Tindouf wilaya, 49,149, is Tindouf town and the Algerian villages. Drawn now as
**173,600 Hassaniya** (`afroasiatic.hassaniya`, Morocco's and Mali's node), row `modelled`, unit
DZ37, added in `countries/dz.py` `_counts()`.

- **Figure**: the UN agencies' figure in the Sahrawi Refugee Response Plan, 173,600 for the five
  camps (UNHCR mission report of March 2018, population on 31 December 2017). UNHCR's own
  planning figure is 90,000 (its 2005 satellite-imagery estimate, which the EU also uses) and the
  Polisario and Algeria say more; no registration or census has settled it. The larger figure was
  chosen because it is the one in the current response plan and the one the first build already
  named in `gap`; the gap text now says the 90,000 exists. No reason found not to draw it (173
  dots against 90). Sources as read in search results: reporting.unhcr.org/node/5310 (Algeria
  2020 plan; the page itself refused a fetch) and allafrica.com/stories/202505080057.
- **Language**: Hassaniya, the Sahrawi language (Glottolog hass1238 lists DZ and EH).
- **Split between camps**: equal fifths, 34,720 each. UNHCR's 2003 figures (Dakhla 38,180, Smara
  about 39,000, Laayoune 36,675, Awserd 32,624, via Wikipedia) are within 15% of equal; no 2018
  per-camp split found.
- **Placement** (`sources/dz_camps.py`, layer `data/geo/dz/dz_hexes_camps.gpkg`): Kontur's hexes
  hold only the Laayoune camp (about 9,000 people, which until now carried Tindouf's census
  residents) and are empty at the others. The layer is religiondots' hexes plus 19 hexagons of
  0.8 km2 (a 2.5 km disc) around each camp centre: GeoNames' CMPRF points for Awserd, Boujdour,
  Laayoune, Smara and Wikipedia's coordinates for Dakhla (26.7464 N, 7.2631 W; GeoNames lacks it).
  Hassaniya goes only on those hexagons; every other language is kept off any hex within 3.5 km of
  a camp centre. The layer's name does not end `_hexes.gpkg`, so the Kontur cap check skips it
  (Algeria has no cap rows; it passed before).
- Rabouni (the administrative camp, 27.476 N 8.088 W) is not drawn as a camp of its own.

## 8. The rest of Africa's Berber (stage 2, 2026-10-19)

Checked how every country with Berber on the map draws it. Only Libya had a lumped label; the
rest are named Glottolog languages already.

| country | how Berber is drawn | verdict |
|---|---|---|
| ma | Tachelhit, Central Atlas Tamazight, Tarifit, from the HCP 2024 census, three nodes (ma2024.py) | correct, unchanged |
| ly | one leaf `nafusi` held Nafusi, Zuwara, Ghadames and Awjila speakers | split, `sources/ly.md` section 6 |
| tn | one leaf `tunisian_berber` = Glottolog's Tunisian-Zuwara Berber (tuni1262), Gabsi 2011's 45,000 to 50,000 over Gabes, Medenine, Tataouine | left: one Glottolog language, not a group node; Djerba and Chenini-Douiret are its dialects (doui1234 is listed under it). Splitting dialects would draw borders between villages a few km apart |
| eg | Siwi (siwi1239) on Siwa and Qara | correct |
| ml | census label "Tamasheq" on `tamasheq`, which holds Tamasheq (tama1365) and Tawallammat Tamajaq (tawa1286, Menaka) | left: a named census label, already in the Tuareg group. Menaka's Tawallammat could go on Niger's `tamajaq` leaf (one line in ml2022.py by region) if wanted |
| ne | census "Tamasheq" on `tamajaq`, Tawallammat and Tayart (Air) together | left, same reason; Tetserret (tets1235, a few thousand) not named |
| bf | "Tamacheq" on `tamasheq` | correct |
| mr | no Berber language drawn; Zenaga (zena1248) has a few hundred speakers, under one dot | left; Malian Tuareg refugees at Mbera are Tamasheq via the origin mix |

No African country draws a bare `afroasiatic.berber` node for a named answer. The three that draw it
at all (ca, es, uk) print "Berber, n.i.e." in their own census, a true remainder.

**Downstream.** `sources/origin_mix.py` builds diaspora mixes from `home("DZ")`, which reads
Algeria's counts. Chenoua is 1.17% of Algeria's drawn people, over the 1% cut, so the next rebuild
of any country with Algerian immigrants (the fragments that carry `afroasiatic.algerian_arabic`:
fr, be, nl, dk, no, se, it, pt, gr, ar, br, mx and others) will pull in
`afroasiatic.berber.chenoua` and drop the unnamed `afroasiatic.berber` row. Before rebuilding such
a country run `python sources/origin_mix.py --fragment <cc>` so its fragment lists the node.

## 9. Retention outside the homelands (2026-10-21, session eca0185b-ber)

Anita, looking at the map: "algiers looks maybe overly kabyle? are we sure about this". Section 2
already said the ethnic-group question draws Algiers at 39.9% Amazigh against 30% Berber-speaking in
1966 and 4.6% in the first-language surveys, and that Berber is "probably overdrawn" where people of
Berber descent speak Arabic. The map is of the language spoken at home, so the cut is made now.

**Rule** (`sources/dz_survey.py` `shares()`, `HEARTLAND`). In every wilaya except the homelands,
the drawn Amazigh share is the midpoint of the ethnic-group share (Arab Barometer VI-VII, as before)
and the pooled first-language Berber share (ArB II-IV, AfB R6; shrunk to its national 6.6% with the
same K = 8 respondents), but no more than the ethnic share and no more than the wilaya's 1966
Berber-speaking share (the surface of section 4). The difference goes to Arab, so Algerian Arabic.
The homelands keep the ethnic share whole: Tizi Ouzou, Bejaia, Bouira, Boumerdes, Batna,
Khenchela, Ghardaia, Tamanrasset, Illizi (a list, not a "1966 surface under 50%" threshold, which
would also have caught Batna, Boumerdes and Ghardaia). The variety split of section 4 is unchanged:
each wilaya's smaller Amazigh count is cut in the same shares. Other, French and Tourag answers are
not touched.

**Why this rule and not another.** It was the suggested one and I found nothing better. The first-
language surveys alone are biased low (every interview in Arabic; 1 of 59 Khenchela respondents
Berber against 72% in 1966), so they are used only as half of a midpoint, not as the answer; the
1966 surface is the one independent anchor for each wilaya and is used as a ceiling, which is what
it can bear, since 60 years of Arabisation make it a plausible upper bound and not a target. The
midpoint with a cap needs no new parameter. After it the correlation with the 1966 surface across
the 48 wilayas goes from +0.886 to +0.933.

**Published source for Algiers: none found.** English and French searches (Kabyle in Algiers,
"enquete sociolinguistique Alger berberophones", Insaniyat, Multilinguales) turned up the
Bachir Bessai survey of Kabylie maritime (insecurity, not Algiers), the Bouira and Aures practice
studies, a 2004 estimate of Kabyle at 9.4% of Algeria and the census 17.8% Berber-speaking quoted
by the UMMTO thesis, none with an Algiers figure. The check is therefore internal: Algiers lands at
21.5% Amazigh, under the 1966 census's 30% for Alger-ville and 25.4% for the wilaya on today's
population, which is the direction Arabisation and an Arabic-speaking in-migration would push.
Nothing contradicts it; nothing confirms it either, and a real Algiers survey would settle it.

**Before and after** (Amazigh share of the wilaya's people, French taken out, so all Berber
varieties together):

| | before (ethnic only) | after | 1966 surface |
|---|---|---|---|
| Algiers | 38.5% (1,150,898) | 21.5% (643,750) | 25.4% |
| Oran | 14.5% (210,226) | 5.0% (72,343) | 5.0% |
| Blida | 7.8% (78,627) | 5.7% (57,307) | 16.0% |
| Boumerdes (homeland) | 48.6% (389,707) | 48.6% (389,707) | 37.9% |
| Setif | 32.4% (482,525) | 18.4% (274,773) | 21.5% |
| **Algeria, 2008 populations** | **23.1% (7,888,896)** | **16.6% (5,665,985)** | 19.3% |

Where the cap binds (the 1966 surface is the lower of the two numbers) Oran, Laghouat, Ouargla,
Tebessa, Guelma and most of the West are drawn at exactly the 1966 level; the table of all 48 is
`python sources/dz_survey.py --check1966` (column `drawn_after_retention`). Bordj Bou Arreridj falls from 23.8% to 13.0% and Tipaza (Chenoua home) from 18.8% to 9.6%.

**Second change, same day (Chaoui wilayas with a zero first-language read).** The first-language
surveys found 0 Berber in Oum El Bouaghi (0 of 62) and Tebessa (0 of 65) and 1.8% in Biskra, which
is the Arabic-interview bias of section 2, so halving the ethnic share by it was not sound. In those
three (`CAP_ONLY` in `sources/dz_survey.py`) the midpoint is dropped and the share is the ethnic
share capped at the 1966 surface: Oum El Bouaghi 46.8% -> 29.3% (first cut gave 23.8%; the cap is
the 1966 surface), Tebessa 47.9% -> 15.9% (cap again, same as the first cut), Biskra 5.9% -> 5.8%
after French (the ethnic share is already under the surface; the first cut gave 4.1%). Chaoui
1,042,356 -> 1,088,657; national Berber 16.5% -> 16.6%. `note_public` now says 17% nationally.

**Varieties now** (people): Kabyle 4,060,262 (was 5,421,509); Chaoui 1,088,657 (1,555,003); Chenoua
193,044 (401,952); Tumzabt 118,189 (150,085); Tamahaq 76,778 (76,778, homeland); Taznatit 52,882
(97,128); Oued Righ 37,456 (71,179); Ouargli 29,250 (102,488); Tidikelt 8,702 (12,009); Tachelhit
765. Chenoua is now 0.56% of Algeria, under origin_mix's 1% cut, so the downstream warning of
section 8 no longer applies (Kabyle and Chaoui were already in the diaspora fragments). Chenoua is
still 2.5 times Ethnologue's 76,000.

## Terms

Arab Barometer: free download, citation requested. Afrobarometer: free download, citation
requested. Nesson 1994: read on persee.fr (open access); figures quoted, not republished as a
table. GeoNames CC BY 4.0; Kontur CC BY 4.0; Glottolog CC BY.
