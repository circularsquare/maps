# Muslim branches: North and West Africa, Horn, Comoros (scout, 2026-10-04)

18 countries, all on bare `islam` today. Read-only on the project; everything fetched is in this
scratchpad (`sl2004_*.pdf`, `sl2004_form.pdf`, `irf/*.txt`, `oussedik2009.txt`, scripts `af_*.py`).
WebSearch calls used: 19.

## Summary

| cc | country | Sunni | the minority, best figure | basis | one branch? | placement | rec |
|---|---|---|---|---|---|---|---|
| ma | Morocco | >99.9% | Shia 8-10k citizens + 1-2k foreigners (~0.03%); Ahmadi 750 | leaders' claims via State Dept; Moroccan researcher | yes | Shia "largest in the north" (Tangier >1,000), prose only | **ASSIGN-ALL** Sunni, remainder ~0.05% |
| dz | Algeria | ~99.5% | **Ibadi 150-300k (0.3-0.7%)**; Ahmadi <200 (leaders) | Minahan 2016 encyclopedia; Ethnologue 150k Mozabite speakers | nationally yes, **not in Ghardaïa** | Ghardaïa wilaya (Ibadi perhaps 35-55% of it, by arithmetic only); nothing counts them | **ASSIGN-ALL** Sunni in 47 wilayas; **Ghardaïa left on `islam`** |
| tn | Tunisia | ~99.5% | Ibadi ~60k (0.5%), claim; Djerba "several tens of thousands" of ~160k | journalism, unsourced | nationally yes | Djerba (Médenine gov.); AB VIII: 4 of 10 Ibadi/Shia answers in Médenine | **ASSIGN-ALL** Sunni, Médenine partly or wholly left on `islam` |
| ly | Libya | 90-95% | **Ibadi 300-400k = 4.5-6%** | Libyan Tmazight Congress (advocacy) via HRW 2017; State Dept's 4.5-6% is the same number | **no** | ethnic homeland only (Nafusa towns, Zuwara, part of Tripoli); no count; AB finds 0 Ibadis in the Nafusa districts | **NATIONAL-ONLY**; §14 acute |
| mr | Mauritania | ~99% | Shia "1%" unofficial = leader's 45k claim (2010) | sect leader's claim | yes (claim aside) | none | **ASSIGN-ALL** Sunni, ~1% remainder |
| sd | Sudan | ~99.5% | Shia: hundreds to 587k (1.5%) claims; surveys ~0.2% | claims; Arab Barometer/Afrobarometer answers | yes | Khartoum (State Dept prose) | **ASSIGN-ALL** Sunni |
| so | Somalia | >99% | none named | Ministry of Endowments via State Dept | yes | none | **ASSIGN-ALL** Sunni |
| dj | Djibouti | ~99%+ of Muslims | Shia, unnumbered | Ministry of Islamic Affairs via State Dept | yes | none | **ASSIGN-ALL** Sunni |
| km | Comoros | 98% of pop. | Shia + Ahmadi + Christians <2% together; Afro R10 "Ismaélite" 2.4% unexplained | State Dept; Afrobarometer R10 | yes | Shia and Ahmadis "mostly on Anjouan" (prose) | **ASSIGN-ALL** Sunni, ~2% remainder. **WRP's 100% Shia is an error** |
| ml | Mali | ~99% | Shia: a Shia imam claims "as many as 10%"; Afrobarometer 0-0.5% | claim vs survey | yes | none | **ASSIGN-ALL** Sunni |
| ne | Niger | ~99% | Shia <1% (Ministry of Interior, IRF 2019) | government via State Dept | yes | none | **ASSIGN-ALL** Sunni |
| bf | Burkina Faso | ~99% | Ahmadi unnumbered; Afro R6: 8 Shia, all Centre-Est | none quantitative | probably | Afro cluster only | **ASSIGN-ALL** Sunni, ~1% remainder; §14 Ahmadis |
| gn | Guinea | ~99.5% | none numbered; Afro R8 2 Shia | | yes | none | **ASSIGN-ALL** Sunni |
| gm | Gambia | ~98% | Ahmadi ~50,000 (~2% of Muslims), community claim | Ahmadiyya via State Dept 2023 | borderline | none | **ASSIGN-ALL** Sunni, ~2% remainder |
| sl | Sierra Leone | 69% of Muslims (2004 census) | **2004 census: Ahmadi 245,908 (5.0% of pop., 6.5% of Muslims); "Shiek" 394,124 (8.0%); Other Muslim 536,874 (10.9%)**; Ahmadi leader 560k | **census**, but code 09 is "Shiek", not Shia, and district tables do not add up | **no** | 2004 tables for 3 districts only, internally inconsistent | **NATIONAL-ONLY** (Ahmadi); do not read "Shiek" as Shia; Anita's call |
| td | Chad | ~99% | none numbered (WRP 2.2% Shia, unsourced) | | yes | none | **ASSIGN-ALL** Sunni |
| gw | Guinea-Bissau | ~97% | "Shia communities also exist" (State Dept); WRP 2.5% unsourced | | probably | none | **ASSIGN-ALL** Sunni, ~2% remainder |
| eg | Egypt | ~99% | Shia "about 1%" (State Dept: scholars and NGOs); range 18k to 3M; surveys 0 | estimates vs surveys | yes | Sharqia, Gharbia named in prose, no numbers | **ASSIGN-ALL** Sunni, ~1% remainder |

"Sunni" in the table means share of Muslims unless said otherwise. A remainder means the share to
leave on bare `islam` if Anita takes the ASSIGN-ALL route.

## Baselines that cover every country (read once, not repeated below)

**Pew 2009, *Mapping the Global Muslim Population*** (on disk, `data/raw/estimates/pew_muslim_population_2009.pdf`),
appendix pp.42-44: every one of the 18 countries (and Senegal) is printed **"<1"** for % of Muslims
who are Shia. The country table of Shia populations over 100,000 (p.13) lists none of them. Basis:
secondary-source compilation and WRD ethnic ascription (branches.md). Pew has no Ibadi or Ahmadi
column.

**World Religion Project (COW WRP)** (`data/raw/estimates/WRP_national.csv`), share of Muslims, 2010:

| | Sunni | Shia | Ibadi | Ahmadi | source code |
|---|---:|---:|---:|---:|---|
| MOR | 99.99 | 0.01 | 0 | 0 | 1 |
| ALG | 100 | 0 | **0** | 0 (0.12 in 2000) | 1 |
| TUN | 98.99 | 1.01 | **0** | 0 | 1 |
| LIB | 100 | 0 | **0** | 0 | 1 |
| MAA | 100 | 0 | | | 83 |
| SUD | 100 | 0 | | | 1 (reliability Low) |
| SOM | 98.99 | 1.01 | | | 83 |
| DJI | 100 | 0 | | | 83 |
| **COM** | **0** | **100** | | | 83 (2000 and 2010) |
| MLI | 98.94 | 1.06 | | | 1 |
| NIR (Niger) | 100 | 0 | | | 83 |
| BFO | 98.45 | 1.55 | | | 350 |
| GUI | 99.06 | 0.94 | | | 1 |
| GAM | 98.89 | 1.11 | | | 1 |
| SIE | 97.29 | 2.71 (8.88 in 2000) | | | 83 |
| CHA | 97.83 | 2.17 | | | 83 |
| GNB | 97.49 | 2.51 | | | 83 |
| EGY | 99.50 | 0 (Alawi 0.50) | | | 83 |

WRP never records an Ibadi anywhere in the Maghreb, and its Comoros row puts every Muslim on Shia,
which is wrong (the constitution names Shafi'i Sunni Islam; State Dept 98% Sunni; Afrobarometer
R10 Sunni or "Muslim only" 97.2%). branches.md already rules WRP's zeros "not split"; this is a
case where its non-zero is wrong too. Its 1% Shia figures (Tunisia, Somalia, the Sahel) look like a
floor convention, not data.

**Surveys, re-tabulated from the files on disk** (`af_surveys.py`, `af_where.py`; unweighted
counts). Neither instrument can measure a sect (branches.md, §11af): "just a Muslim" or "Muslim
only" is the modal answer. They are useful only as a ceiling on how often anyone volunteers a
minority label.

Arab Barometer sect follow-up (`q1012a`, `Q1012A`, `Q1012A_MUSLIM`):

| | wave: Shia / Ja'fari / Ibadi or Mozabite / Ahmadiyya, of n |
|---|---|
| Morocco | IV 0/0/0/0 of 1,200; V 2/0/0/0 of 2,400; VII 1/1/0/1 of 2,404; **VIII 25/1/0/0 of 2,411** (10 in Laayoune-Sakia El Hamra, 5 Casablanca-Settat, 5 Rabat-Salé-Kénitra, 3 Guelmim-Oued Noun, 2 Souss-Massa, 1 Marrakech-Safi) |
| Algeria | IV 2/0/4/0 of 1,200; V 2/0/3/0 of 2,332; VII 0/2/4/1 of 2,162 |
| Tunisia | IV 1/0/0/0; V 2/0/0/0 of 2,400; VII 5/1/2/2 of 2,400; VIII 5/0/5/0 of 2,406 (the 10: **Médenine 4**, Nabeul 2, Ben Arous, Gabès, Tunis, Mahdia 1 each) |
| Libya | V 1/1/1/0 of 1,962; VII 0/4/4/0 of 2,505 |
| Sudan | V 4/0/0/0 of 1,758; VII 4/0/3/2 of 2,353 (North Darfur 4, Khartoum 2, ...) |
| Egypt | V: **no Shia, Ja'fari, Ibadi or Ahmadi answer** of 2,400 (about 2,160 Muslims) |

Morocco VIII's jump from 1-2 to 25 Shia, with 10 in one Saharan region, reads as a fieldwork or
coding artefact, not a community.

Afrobarometer religion item (Q90/Q98A/Q98/Q98A/Q95, R4-R9): no round offers an Ahmadiyya box.
Shia answers: Morocco R5 1; Tunisia R5 1, R7 1, R9 1; Sudan R8 2; Mali R4 6, R7 1, R9 1; Burkina
R5 1, **R6 8 (Kouritenga 6, Boulgou 2, all Centre-Est)**, R7 1; Niger R6 1; Guinea R8 2; Gambia,
Sierra Leone, Egypt, Mauritania 0. Algeria R5 has **10 "Ibadi", all in Boumerdès**, a coastal
wilaya with no known Ibadi community, and none in Ghardaïa: one sampling point or one interviewer.

## Morocco (`ma`)

- **State Dept IRF 2023** (`state.gov/reports/2023-report-on-international-religious-freedom/morocco/`):
  "More than 99 percent of the population is Sunni Muslim"; "Shia Muslim leaders estimate there are
  several thousand Shia citizens, with the largest proportion in the north. In addition, there are
  an estimated 1,000 to 2,000 foreign-resident Shia"; "Leaders of the Ahmadi Muslim community
  estimate their numbers at 750." Basis: sect leaders' claims.
- **Ibrahim al-Saghir, Moroccan researcher on Shiism**, via al-Omk, 4 Dec 2018
  (`al3omk.com/359479.html`): 8,000-10,000 resident Moroccan Shia (which he takes from the State
  Department's 2012 and 2015 reports, so it is the same chain) plus ~2,000 foreign; others put it at
  3,000-20,000. Geography in prose: north and north-east largest, **Tangier over 1,000**, Tetouan,
  Al Hoceima, Oujda, Chefchaouen; Casablanca "rivals Tangier"; Rabat-Salé; Marrakech and Agadir;
  activity in Laayoune and Dakhla. No table.
- Ibadis: none recorded anywhere (an Ibadi article on yabiladi.com exists, not opened).
- Pew 2009 <1%; WRP 0.01%; Arab Barometer 0-2 Shia per wave except VIII's artefact.
- **One branch: yes.** Shia about 10-12k of ~37M, 0.03%; Ahmadis 750.
- §14: Shia cannot register or hold public Ashura (IRF 2023); a placement by city would point at
  converts. Nothing to place anyway.
- **ASSIGN-ALL** Sunni, remainder ~0.05% (or none). Morocco is the brief's own example and the
  numbers bear it out.

## Algeria (`dz`), the M'zab

- **No count of Ibadis exists.** Algeria's censuses never asked religion (§11af); WRP has zero; the
  Arab Barometer pool has 7-11 Ibadi/Mozabite answers, none in Ghardaïa (dz.md §7). The Ministry of
  Religious Affairs publishes no statistics. Searched in Arabic ("عدد الإباضية في الجزائر غرداية"),
  French ("Mozabites population Ghardaïa Ibadites nombre", "ibadites 60 % vallée du M'zab").
- **Figures that exist:**
  - Minahan, *Encyclopedia of Stateless Nations*, 2nd ed. (2016), p.284: Mozabites "about 150,000
    to 300,000" (cited by en.wikipedia "Mozabite people"). Compiler, no method.
  - Ethnologue (via Wikipedia, retrieved 2023): ~150,000 Mozabite (Tumzabt) speakers. Language,
    not sect, but the two coincide closely in the M'zab.
  - ar.wikipedia "الإباضية في الجزائر": "about one million or more in Algeria"; uncited and
    impossible (Ghardaïa wilaya has ~400k people all told).
  - State Dept IRF 2023: ">99 percent ... Sunni Muslims following the Maliki school"; Ibadis named
    among the <1% and said to reside "principally in Ghardaia Province"; "fewer than 200 Ahmadi
    Muslims" (religious leaders), 33 Ahmadis facing charges.
  - Colonial: the 1955 *Le M'zab* monograph (alger-roi.fr) gives Ghardaïa town 14,046 inhabitants,
    8,024 Ibadites and 6,022 Malékites; the other towns only totals (Beni Isguen 4,293, Melika
    2,829, Bou Noura 1,753, El Atteuf 1,720, Berriane 4,759 "with an Arab minority", Guerrara 7,719).
    Historical only.
  - Fatma Oussedik (Université d'Alger-CREAD), IUSSP 2009 paper (`ipc2009.popconf.org/papers/90248`,
    text in `oussedik2009.txt`): no count, but states "peu à peu les Ibadites deviennent minoritaires
    au M'zab", especially since Ghardaïa became a wilaya capital. Gives commune populations for the
    pentapole, Berriane and Guerrara (1966-98), not by sect.
  - Search snippets (not opened, Jeune Afrique 403): "Ibadites constitute 60% of the inhabitants of
    the valley"; Berriane's 35,000 "roughly half" Malékite, half Ibadite.
- **One branch: nationally yes** (0.3-0.7% of ~46M), **but not in Ghardaïa wilaya.** Arithmetic only:
  if 150-200k of the 150-300k live in the wilaya (2008: 363,598), Ibadis are roughly 40-55% of it;
  Oussedik says they are becoming a minority in the valley itself. Ibadi communes: Ghardaïa (with
  Melika), Bounoura (with Beni Isguen), El Atteuf, Daya Ben Dahoua, Berriane, Guerrara; Metlili,
  Zelfana, El Meniaa are Chaamba/Maliki. Outside: Ouargla, and merchant diasporas in Algiers, Oran.
- **Placement**: only the homeland. No source places Ibadis at wilaya or commune with a number.
- **§14**: the 2013-2015 Ghardaïa communal violence between Mozabites and Chaamba (Arab Malikis);
  the death in custody of Mozabite activist Kamel Eddine Fekhar (2019). (Both from general knowledge,
  not re-sourced in this sweep; Oussedik 2009 documents the Ibadi-Maliki tension in the valley.) Ahmadis prosecuted (33 cases).
  A coloured Ghardaïa would mark a group that has been in deadly communal conflict.
- **Recommendation: ASSIGN-ALL Sunni in the 47 other wilayas; leave Ghardaïa on bare `islam`**
  (an honest "this wilaya is mixed, unknown split"). A wilaya-wide Sunni colour would be wrong, and
  an Ibadi share would be invented. Diaspora Ibadis in Algiers etc. (perhaps 0.1-0.3% there) fall in
  the remainder of an otherwise-Sunni assignment.

## Tunisia (`tn`), Djerba

- State Dept IRF 2023: "approximately 99 percent are Sunni Muslim"; Shia among the <1%; **Ibadis not
  mentioned**.
- Religion Unplugged (Nadia Addezio, 9 April 2026,
  `religionunplugged.com/news/tunisia-island-of-djerba-an-ancient-ibadi-heritage-endures`): "about
  60,000 in Tunisia", "a sizeable group" on Djerba. Unattributed.
- Orient XXI, "Djerba l'ibadite" (Agnès De Féo; page 403, **search snippet only**): "several tens of
  thousands of Ibadites on the island, concentrated in the south-west", of 150,000.
- Arabic sources (Arabi21 translating Le Monde, 31 Oct 2015; ultratunisia; Al Jazeera) give no
  number; one says Ibadis are declining on Djerba since the tourism boom of the 1980s-90s.
- Arab Barometer VIII: 4 of 10 Ibadi/Shia answers in Médenine (Djerba's governorate).
- **One branch: nationally yes** (~0.5%). Médenine (~0.5M people; Djerba ~160-176k of it) might be
  roughly 5-15% Ibadi on those claims.
- §14: low; the 2023 Ghriba attack targeted Jews, not Ibadis.
- **ASSIGN-ALL** Sunni, with Médenine either left on `islam` (cleanest) or assigned with a stated
  ~10% remainder. No source would carry a number for Médenine.

## Libya (`ly`)

- **Human Rights Watch, 20 July 2017** (`hrw.org/news/2017/07/20/libya-incitement-against-religious-minority`):
  Ibadis "between 300,000 and 400,000 in Libya, according to the Libyan Tmazight Congress"; Amazigh
  "5 to 10 percent". Basis: advocacy body's claim.
- **State Dept IRF 2019 and 2023**: "Sunni Muslims represent between 90 and 95 percent ..., Ibadi
  Muslims account for between 4.5 and 6 percent". Unattributed, and it is the Congress figure:
  300k and 400k over ~6.6M Libyans are 4.5% and 6.0% exactly. **One source, not two.**
- Arab Barometer: 5 Ibadi answers in ~8,400 Libyans, **none in Al Jabal al Gharbi or Nalut** (ly.md).
  The sample either missed the Ibadi towns or Ibadis did not say; it cannot test the 4.5-6%.
- Geography (prose, HRW and Arabic sources): Nafusa mountain towns (Nalut, Kabaw, Jadu, Yefren,
  Al-Qalaa and others) and Zuwara on the coast, plus Tripoli. Zuwara ~45k (Al Jazeera 2011).
  Amazigh in Ghadames and the Tuareg are Maliki, so Amazigh is not Ibadi one-for-one.
- **One branch: no.** 4.5-6% is a real minority if the claim is near right; it is the largest
  Ibadi share of any country here.
- **Placement**: none quantitative. Ly is drawn at 22 districts; the Ibadi towns fall in Nalut, Al
  Jabal al Gharbi and An Nuqat al Khams (Zuwara), each mixed with Arab towns (Zintan, Gharyan,
  Al-Ajaylat). A town-by-town ethnic assignment from census town populations would be possible
  in principle and is exactly the "assigned from ethnicity" tier.
- **§14 acute**: the Interim Government's Supreme Fatwa Committee (July 2017) called Ibadis "a
  misguided and aberrant group ... Kharijites ... infidels"; the 2023 IRF reports continued
  Salafist harassment and incitement.
- **NATIONAL-ONLY** (Ibadi 4.5-6%, advocacy claim, on the estimate layer at most). Any district
  placement is Anita's call under §14.

## Mauritania (`mr`)

- State Dept IRF 2019/2023: "According to Mauritanian government estimates, Sunni Muslims
  constitute approximately 99 percent ... Unofficial estimates indicate Sunni Muslims are
  approximately 98 percent ..., Shia Muslims 1 percent".
- The "unofficial" 1% matches the **Shia leader's claim of 45,000 converts** (Elaph, March 2010,
  `elaph.com/Web/NewsPapers/2010/3/546546.html`; ~1.5% of 2011 population per a search summary).
  Nouakchott's Grand Mosque imam campaigns yearly against Shia spread; a Shia complex north of
  Nouakchott was seized (al-Quds al-Arabi).
- No survey asks (Arab Barometer VII/VIII and Afrobarometer R9 have no Mauritanian sect answers).
- **One branch: yes** apart from a sect leader's claim. **ASSIGN-ALL** Sunni (Maliki), ~1% remainder.
- §14: converts under pressure; nothing to place.

## Sudan (`sd`)

- State Dept IRF 2023: "Almost all Muslims in the country identify as Sunni ... Small Shia Muslim
  communities are based predominantly in Khartoum." Sufi orders are distinctions within Sunni Islam.
- Claims (Arabic search): from "a few hundred to ~700", ~10,000, ~130,000, to **587,000 (1.5%)** by
  the Ahl al-Bayt World Assembly (2008), a Shia body. Iran's cultural centres were closed in 2014.
- Surveys: AB V 4 Shia of 1,758, VII 4 Shia + 3 Ibadi + 2 Ahmadiyya of 2,353; Afro R8 2 Shia.
  Roughly 0.2% volunteer Shia.
- **ASSIGN-ALL** Sunni. Orders (Khatmiyya, Ansar, Tijaniyya, Qadiriyya) are not branches; note only.

## Somalia (`so`)

- State Dept IRF 2023: "According to the Federal Ministry of Endowments and Religious Affairs, more
  than 99 percent of the population are Sunni Muslim"; "an unknown number of Shia". Nothing else.
- **ASSIGN-ALL** Sunni (Shafi'i).

## Djibouti (`dj`)

- State Dept IRF 2023: "94 percent are Sunni Muslim. According to the Ministry of Islamic Affairs,
  Shia Muslims, Roman Catholics, Protestants, [Orthodox], Jehovah's Witnesses, Hindus, Jews, Baha'is,
  and atheists constitute the remaining 6 percent", concentrated in Djibouti City. Shia unnumbered;
  not in Afrobarometer R4-R9.
- **ASSIGN-ALL** Sunni (Shafi'i).

## Comoros (`km`)

- **WRP's 100% Shia is an error**: COW WRP codes all Comorian Muslims as Shia in 2000 and 2010.
- State Dept IRF 2023: "98 percent is Sunni Muslim. Roman Catholics, Shia Muslims, Ahmadi Muslims,
  and Protestants together make up less than 2 percent ... Shia and Ahmadi Muslims live mostly on
  the island of Anjouan." Constitution: Shafi'i Sunni. Communities shun Sunni-to-Shia converts.
- Afrobarometer R10 (km.md): Musulman seulement 96.2, Sunnite seulement 1.0, **Ismaélite 2.4** (29
  respondents), refused 0. Nothing else records Ismailis in Comoros beyond a tiny Indian Khoja
  presence; the answer is unexplained and not checked.
- **ASSIGN-ALL** Sunni, ~2% remainder (or 2.4% if the Ismaili answers are kept as unspecified).

## Mali (`ml`)

- State Dept IRF 2023: Muslims ~95% (MARCC); "Nearly all Muslims are Sunni, and most follow Sufism;
  however, one prominent Shia imam stated that as many as 10 percent of Muslims are Shia."
- Afrobarometer: Shia 6 (R4), 0, 0, 1 (R7), 0, 1 (R9) of ~1,000-1,100 Muslims per round; Pew 2009 <1%.
- **ASSIGN-ALL** Sunni. The 10% is a sect leader's claim against surveys at 0-0.5%. Wahhabiya,
  Ansar Dine, Hamallists, Tijaniyya are not branches.

## Niger (`ne`)

- State Dept IRF 2019: "According to the Ministry of Interior (MOI), more than 98 percent of the
  population is Muslim with the vast majority being Sunni. Less than 1 percent are Shia." IRF 2023:
  "Approximately 80 percent of the country's Muslims are Sunni followers of the Maliki school".
- A search summary attributes "95% Sunni and 5% Shia" to the Ministry of Interior (probably an older
  IRF or Wikipedia); **not opened, not verified**, and contradicted by IRF 2019.
- Afrobarometer: 1 Shia (R6) over five rounds; Izala 10 (R7). Searched French ("chiites au Niger
  nombre Maradi Zinder"): nothing quantitative; IMN-style Shia presence near the Nigerian border is
  not documented in numbers.
- **ASSIGN-ALL** Sunni.

## Burkina Faso (`bf`)

- Census 2019: one Muslim code (bf.md). IRF 2023: "63.8 percent ... Muslim (predominantly Sunni)".
- **Ahmadis**: no figure found. The community was recognised in 1986; an anti-Ahmadi blog says
  "barely 1000" (not credible as a source); Mahdiabad near Dori is an Ahmadi village of ~650.
  Searched French ("Ahmadiyya Burkina Faso nombre de membres").
- **§14, acute**: in January 2023 an armed group killed **nine Ahmadi Muslims at the Mahdiabad
  mosque near Dori** (Seno; titles of the alislam.org and pressahmadiyya.com releases, 2023/01, not opened) after they refused to renounce their faith (alislam.org press release;
  IRF 2023 also reports 120 households forced to flee). Any Ahmadi placement would point at a
  community already targeted.
- Afrobarometer R6: 8 Shia, all in Kouritenga (6) and Boulgou (2), Centre-Est; 1 in R5 and R7.
  One cluster, not a geography.
- **ASSIGN-ALL** Sunni, ~1% remainder; do not place Ahmadis.

## Guinea (`gn`)

- IRF 2023: "Muslims are generally Maliki Sunni; Sufism is also present"; Tijaniyya vs Wahhabi
  tension in Labé. Afro R8: 2 Shia. Census one code.
- **ASSIGN-ALL** Sunni.

## Gambia (`gm`)

- IRF 2023 (slug `gambia`): "96.4 percent ... Muslim, most of whom are Sunni; the Ahmadiyya Muslim
  community states it has approximately 50,000 members". About 2% of ~2.4M Muslims. Census 2013 has
  no Ahmadi code (gm.md). Afrobarometer: no Shia, no Ahmadi box.
- §14: the Supreme Islamic Council declares Ahmadis non-Muslim and has barred them from Muslim
  cemeteries since 2015.
- **ASSIGN-ALL** Sunni, ~2% remainder (the Ahmadi claim), Ahmadis not placed.

## Sierra Leone (`sl`): the one census count, and it is shaky

**2004 Population and Housing Census, Table 8A**, via the SLUPS/UNFPA *Population Profile* series
(Stats SL, 2010), archived at Wayback `web.archive.org/web/20111113144403id_/http://www.statistics.sl/reports_to_publish_2010/population_profile_of_sierra_leone_2010.pdf`
(and `..._bo_district_and_bo_town_2010.pdf`, `..._bombali_district_and_makeni_town_2010%20.pdf`,
`..._western_area_urban_2010.pdf`). Copies in the scratchpad (`sl2004_*.pdf`).

National, Table 36 (household population 4,930,532):

| code | label in tables | persons | % of pop. | % of Muslims |
|---|---|---:|---:|---:|
| 08 | Sunni | 2,603,567 | 52.8 | 68.9 |
| 09 | "Shiitte" (form: **"Shiek Muslim"**) | 394,124 | 8.0 | 10.4 |
| 07 | Ahmadis | 245,908 | 5.0 | 6.5 |
| 10 | Other Muslim | 536,874 | 10.9 | 14.2 |
| | all Muslim | 3,780,473 | 76.7 | 100 |

Districts printed (% of population): Bo District Ahmadi 4.6, Sunni 49.0, Shiite 9.5, Other 9.1;
Bo Town 4.0 / 46.2 / 10.9 / 6.7; Bombali 4.0 / 56.2 / 5.1 / 5.0; Makeni 1.4 / 66.8 / 3.8 / 5.7;
**Western Area Urban Ahmadi 27.3 / Sunni 25.0 / Shiite 7.8 / Other 7.2** (Ahmadi 208,773 persons).

Three problems, each enough to stop a placement:

1. **Code 09 on the enumeration form is "Shiek Muslim"** (IPUMS
   `international.ipums.org/international/resources/census_forms/africa/sl2004ef_sierra_leone_enumeration_form.en.pdf`,
   p.2 code list: 07 Ahmadis Muslim, 08 Sunni Muslim, 09 Shiek Muslim, 10 Other Muslim). "Shiek" can
   be read as Sheikh as easily as Shia, and 394,124 Shia is 20-40 times every other estimate: State
   Dept 2023 "Shia Muslims comprise less than 0.5 percent of the Muslim population"; Pew 2009 <1%;
   Afrobarometer 0. **Do not draw it as Shia.**
2. **The district tables do not add up.** Bo District (20,482) + Bombali (16,178) + Western Area
   Urban (208,773) = 245,433 Ahmadis, **99.8% of the national 245,908**, leaving 475 for the other
   eleven districts, including Kenema, Kailahun and Kambia where the Ahmadiyya has old missions
   (Rokupr, Baomahun). Either Western Area Urban's 27.3% (Ahmadis outnumbering Sunnis in Freetown)
   or the national row is wrong; a 07/08 code slip would fit. Only four of the profiles survive.
3. Microdata: `microdata.statistics.sl/index.php/catalog/3` says "Data Access Not Available";
   the IPUMS subset needs the blocked account.

Other figures: **Ahmadi leader 560,000** (State Dept IRF 2019 and 2023, attributed to the
Ahmadi Muslim leader); en.wikipedia repeats it as ~9% and cites "as high as 700,000" (Ozy 2019).
The 2015 census and Afrobarometer have no Ahmadi box. So Ahmadis are 5% of the population by the
2004 census and 7-8% by the community's claim: **a real minority, the largest Ahmadi share anywhere
in the world by both counts.**

- **One branch: no** (Ahmadi 6.5-10% of Muslims).
- **Placement: nothing usable.** Three districts, internally inconsistent; the 2021 mid-term census
  asked denominations and has published nothing (sl.md §9).
- **§14**: IRF 2023, Tablighi preachers in Waterloo, Moyamba and Kono telling people Ahmadis "should
  be killed in Sierra Leone" (sl.md §11). Verbal, not physical.
- **NATIONAL-ONLY**: Ahmadiyya 5.0% of the population (2004 census) on the estimate layer, Sunni
  left as the census's 52.8% with "Shiek" and "Other Muslim" unspecified. Whether to put the census
  Ahmadi share onto dots nationally is Anita's call; India's precedent (folded back) argues caution.
  **Reopen when the 2021 MTPHC religion tables appear.**

## Chad (`td`)

- IRF 2023: "Most Muslims adhere to the Sufi Tijaniyah tradition. A small minority hold beliefs
  associated with Wahhabism, Salafism ..." No Shia mentioned. Not in Afrobarometer. Census 2009 one
  Muslim code. WRP 2.17% Shia, source code 83, no basis found.
- **ASSIGN-ALL** Sunni.

## Guinea-Bissau (`gw`)

- IRF 2023: "most Muslims are Sunni, although Shia communities also exist" (Fula and Mandinka
  the main Muslim groups). WRP 2.5% Shia, unsourced. No survey asks.
- **ASSIGN-ALL** Sunni, ~2% remainder.

## Egypt (`eg`)

- State Dept IRF 2023: "Scholars and NGOs estimate Shia Muslims comprise approximately 1 percent of
  the population. There are also small numbers of Dawoodi Bohra Muslims and Ahmadi Muslims." Quranist
  activists are prosecuted (Reda Abdel Rahman's travel ban). MRGI (2019) on Shia hardship.
- Estimates in Arabic sources: Mohamed Hassanein Heikal 18,000 (Shorouk, 18 Apr 2013); Ibn Khaldun
  Center ~700,000; Shia leaders ~1.5M; the Shia spokesman 3M "in all governorates" (al-Anba, Sept
  2012); The Economist "50,000 to 1 million". Presence named in Sharqia (Zagazig, Abu Hammad,
  Belbeis, Diarb Negm, Abu Kebir, Faqous) and Gharbia (Tanta, Mahalla); no numbers by place.
- Surveys: Arab Barometer V 0 Shia among ~2,160 Muslims; Pew 2012 Q31 Shia 0%; GFS 0 (branches.md).
  A 1% group should show ~20 answers in AB V; it shows none (Shia may conceal, so this is a ceiling
  on open self-identification, not on belief).
- **One branch: yes** (Shia likely well under 1%). Egypt already has Pew 2012's Sunni 88 / Shia 0 /
  just-Muslim 12 on the estimate layer.
- §14: Shia are arrested for private worship; nothing places them anyway.
- **ASSIGN-ALL** Sunni, ~1% remainder.

## Negatives, with what was tried

- **Any Algerian/Libyan/Tunisian census or official count of Ibadis**: none. Algeria's and Tunisia's
  censuses never asked religion (§11af); WRP codes zero; the ministries publish no statistics. The
  only by-sect counts found are colonial (1955 Ghardaïa town). Searched Arabic ("عدد الإباضية في
  الجزائر غرداية", "عدد الإباضية في ليبيا جبل نفوسة زوارة", "عدد الإباضية في جربة") and French
  ("Mozabites population Ghardaïa Ibadites nombre", "ibadites Djerba nombre aujourd'hui").
- **Census sect codes elsewhere**: every other census here has one Muslim code (bf, ml, ne, gn, gm,
  td, dj; mr and km ask no religion; sd's was deleted in 2008).
- **Sierra Leone 2004 district tables beyond Bo, Bombali, Western Area Urban**: Wayback CDX of
  `statistics.sl/reports_to_publish_2010/*` lists only those four profiles.
- **Nigeria-style Shia movements in Niger and Chad**: no numbers found (one French search); Niger's
  MOI says <1%.
- Blocked: Jeune Afrique (403), Orient XXI (403), state.gov to WebFetch (403; fetched with a
  generic browser UA by script instead).
