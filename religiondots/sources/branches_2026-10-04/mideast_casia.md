# Muslim branches: Middle East and Central Asia (13 countries), 2026-10-04

Scope: jo, ps, ae, qa, sa, om, az, af, tj, uz, kz, kg, tm. Syria not touched. Read-only on the
project; downloads in this scratchpad (`irf_*_2023.pdf`, `pew2012_worlds_muslims.pdf`,
`yunusov_islam_az.pdf`, `yunusov_islam_factor.pdf`). About 25 WebSearch calls used.

## Summary

| cc | best national figure (of Muslims unless said) | real minority? | placement below national | recommendation |
|---|---|---|---|---|
| jo | Pew 2012: Sunni 93, Shia 0, just Muslim 7; State Dept 2023 "virtually all Sunni" | no (Druze, counted Muslim by the state, and a few Iraqi Shia refugees) | none needed | **ASSIGN-ALL** Sunni, remainder ~0.5% unspecified |
| ps | Pew 2012: Sunni 85, Shia 0, just Muslim 15 | no | none needed | **ASSIGN-ALL** Sunni, remainder <0.5% |
| ae | State Dept 2023: citizens >85% Sunni ("media reports"), most of the rest Shia, in Dubai and Sharjah | yes, up to ~15% of Emiratis | words only (Dubai, Sharjah); no per-emirate figure | **NATIONAL-ONLY** (Emiratis stay on `islam`) |
| qa | State Dept 2023: "most citizens Sunni, almost all others Shia"; 10-15% Shia of citizens (US embassy 2007 via Arabic press) | yes, among citizens | none; and the map draws 2004 census Muslims of every nationality together | **LEAVE** |
| sa | State Dept 2023: citizens 85-90% Sunni, Shia 10-12%, and 25-30% of the Eastern Province's population | yes | **Eastern Province has a share** (State Dept; ICG 2005 "one-third", snippet); Najran "majority Ismaili" (no usable count) | **NATIONAL-ONLY now; an ask is justified** (see the section) |
| om | State Dept 2023: 45 Sunni / 45 Ibadi / 5 Shia (of everyone); other claims 21-75% Ibadi | yes (three branches) | descriptive only (Dhofar "entirely Sunni", Peterson) | **NATIONAL-ONLY** (ask 043 stands) |
| az | Pew 2012: Sunni 16, Shia 37, just Muslim 45; SCWRA/Sheikh-ul-Islam 65 Shia / 35 Sunni | yes | **historical counts only**: 1913 and 1916 sect counts by uyezd (Kavkazsky Kalendar, in Yunusov 2004); modern ethnic ascription | **NATIONAL-ONLY** |
| af | Pew 2012: Sunni 90, Shia 7, just Muslim 3; WRD 89/11; Izady 70/29; Shia writers 25-30 | yes | Shia writers' per-province prose (Bamyan, Daykundi majority; Herat city ~50%), no method | **NATIONAL-ONLY** |
| tj | Pew 2012: Sunni 87, Shia 3, just Muslim 7; State Dept: Ismaili 3-4%, most in GBAO | small, but concentrated | GBAO (one of the five drawn regions) | **ASSIGN Sunni outside GBAO, leave GBAO on `islam`** (fits ask 051) |
| uz | Uzbek government via State Dept 2023: 35M Sunni, 122,000 Shia (0.35%) | no | Shia in Bukhara and Samarkand (words, older reports) | **ASSIGN-ALL** Sunni, 0.35% unspecified |
| kz | Pew 2012: Sunni 16, Shia 1, just Muslim 74; no Shia figure anywhere | no (Shia ~1%) | census counts Azerbaijanis (145,615 in 2021) by oblast and religion | **ASSIGN-ALL** Sunni; optional: Azerbaijanis' Muslims left on `islam` |
| kg | Pew 2012: Sunni 23, just Muslim 64; government: Shia <1% of Muslims; Ahmadis ~1,000 | no | none needed | **ASSIGN-ALL** Sunni, <1% unspecified |
| tm | none surveyed; State Dept: "mostly Sunni", Shia pockets of Iranians, Azeris, Kurds | no (~0.6%) | **2022 census nationality by velayat** gives those three groups: 40,312 people | **ASSIGN-ALL** Sunni, those three nationalities left on `islam` per velayat |

## Most important findings

1. **Azerbaijan has real sect counts by district, but from 1913 and 1916.** Yunusov, *Islam v
   Azerbaidzhane* (Baku 2004, open PDF from his institute) prints Tables 4 and 5 from the Russian
   *Kavkazsky kalendar* for 1914 and 1917: Muslims, Shia and Sunni by uyezd (17 units). Zakatal
   okrug 99% Sunni, Nukha (Sheki) 90-91%, Quba 70-89%, Areş 79-85%, Shamakhi 63-66%; Nakhchivan
   99% Shia, Shusha 94%, Lankaran 94-96%, Baku 79-84%; all Azerbaijan 62/38. Yunusov calls these
   "the first and so far the only real data" on the split. Like the British India sect tables, they
   are history. Anita ruled 1950 too old for Iran, so they are a lead only.
2. **Saudi Arabia has a regional figure for one region, and Najran's numbers are impossible.** The
   State Department says 25-30% of the Eastern Province is Shia. On the 2022 census (5,125,254 people,
   2,949,854 Saudis) that is 1.28-1.54M people, which is 43-52% of the province's Saudis. The
   Najran Ismaili counts in circulation (500,000 older IRF; 700,000 IRF 2016 via Wikipedia) are
   more than the 394,976 Saudis the census counts in Najran.
3. **Turkmenistan's census places the only Shia groups the State Department names.** Census 2022
   tables 4.1-4.8 (already in `tm_geo.py`) give Azerbaijanis 26,576, Persians 10,997 and Kurds 2,739.
   That is 0.57% of the country, mostly in Ashgabat (13,119), Mary (11,431), Balkan (7,448) and
   Ahal (6,727), which is where the State Department puts them ("Ashgabat ... the border with Iran
   ... Turkmenbashi").
4. **The governments give small Shia figures for Uzbekistan and Kyrgyzstan.** Uzbekistan's
   government says 122,000 Shia against 35 million Sunni (0.35%). Kyrgyzstan's says under 1% of
   Muslims. Those are numbers for an ASSIGN-ALL Sunni. Pew's "just a Muslim" majorities (54% and 64%)
   are the usual non-answer in former Soviet countries, not a sect.
5. **Tajikistan can be split by region without drawing Ismailis.** The State Department (from
   local academics) says Ismailis are 3-4% of Muslims, most in GBAO. GBAO is one of the five drawn
   regions (227,916 people, 2.4% of the country). Sunni on the other four, with GBAO left on
   `islam`, keeps to ask 051 (no Ismaili node). The open question is Dushanbe's Pamiri migrants.

---

## Jordan (`jo`)

- **Pew 2012** (*The World's Muslims*, Q31, printed p. 30, re-read 2026-10-04 from
  `pewresearch.org/wp-content/uploads/sites/20/2012/08/the-worlds-muslims-full-report.pdf`):
  Sunni 93, Shia 0, something else 0, just a Muslim 7, nothing/DK 0. Self-ID survey, 2011.
- **US State Department, 2023 IRF report, Jordan** (`state.gov/wp-content/uploads/2024/04/547499-JORDAN-2023-...pdf`,
  opened): "Muslims, virtually all of whom are Sunni, make up 97.1 percent". Druze are "considered
  Muslims by the government"; Shia "account for a small number of Syrian refugees and less than
  one-third of the Iraqi refugee population" (57,000 Iraqis registered with UNHCR).
- **Verdict:** one branch. The non-Sunni Muslims are Druze (if the build's "Muslim" holds them) and a
  few thousand Iraqi Shia, with no figure for either. Not concentrated in any way the map can use.
- **§14:** none.
- **Recommendation: ASSIGN-ALL** `islam.sunni`, with a small unspecified remainder (say 0.5%) for
  Druze and Iraqi Shia. Check whether `jo`'s Arab Barometer "Muslim" answers include Druze before
  writing the note.

## Palestine (`ps`)

- **Pew 2012** Q31, p. 30, "Palestinian terr.": Sunni 85, Shia 0, something else 0, just a Muslim 15,
  nothing/DK 0. Self-ID survey.
- PCBS census 2017 has one `Islam` box (`sources/ps.md` §3).
- **Not searched further:** Palestine has no Shia or Ahmadi community of any size anyone describes
  (the Ahmadi centre at Kababir is in Haifa, inside Israel).
- **Verdict:** one branch; Pew's 15% "just a Muslim" is the non-answer, not a minority.
- **Recommendation: ASSIGN-ALL** `islam.sunni`, remainder under 0.5% unspecified.

## United Arab Emirates (`ae`)

- **State Department 2023 IRF, UAE** (opened): citizens about 11% of residents, "of whom more than 85
  percent are Sunni Muslims, according to media reports. Most of the remainder of the citizens are
  Shia Muslims, who are concentrated in the Emirates of Dubai and Sharjah." Non-citizen Muslims: "media
  estimates suggest that less than 20 percent ... is Shia". Ahmadi, Ismaili and Bohra together under
  5% of everyone, "almost entirely noncitizens". Basis: unnamed media reports.
- Arabic search (`شيعة الإمارات نسبتهم من المواطنين ...`): only 15% of all residents, "17-18%"
  attributed to the State Department, and "2% of Emiratis", none with a source.
- **Placement:** Dubai and Sharjah, in words only. `ae` draws Emiratis per emirate (grown estimates),
  so a share per emirate is what would be needed; none exists.
- **Recommendation: NATIONAL-ONLY.** Emiratis stay on `islam` (asks 040/043). Foreigners' branches are
  folded to `islam` in `ae.py`; `origin_religion.py`'s sect shares have no source
  (`branches.md`), so do not unfold them on this evidence.

## Qatar (`qa`)

- **State Department 2023 IRF, Qatar** (opened): "Most citizens are Sunni Muslims, and almost all
  others are Shia Muslims." No figure.
- Arabic press (search summary, not opened): 10-15% of Qatari citizens Shia, or "10% according to a
  2007 US embassy report". Origins: Baharna and 'Ajam (Iranian and Baloch descent).
- **The map's build cannot take it:** `qa` is the 2004 census's Muslims of every nationality by
  municipality (`sources/qa.md`), with no citizen column, so a citizens' share has nothing to sit on.
- **Recommendation: LEAVE.**

## Saudi Arabia (`sa`)

National:
- **State Department 2023 IRF, Saudi Arabia** (opened): "between 85 and 90 percent of the country's
  citizens are Sunni Muslims. Shia Muslims constitute 10 to 12 percent of the citizen population and
  an estimated 25 to 30 percent of the Eastern Province's population." No source given.
- HRW 2009: 10-15% (`sources/sa.md`). ICG, *The Shiite Question in Saudi Arabia* (2005): 10-15% of
  nationals and "one-third" of the Eastern Province (**search summary only**; crisisgroup.org returned
  a Cloudflare block and refworld 403).
- Matthiesen, *The Other Saudis* (Cambridge UP 2015), introduction: "government consultants" put the
  Eastern Province's Shia at about 1 million and native Shia at about 1.5 million, including about
  250,000 Ismailis in Najran (**search summary only**; the Cambridge excerpt PDF timed out twice).
- Arabic: WikiShia Arabic (*al-Mamlaka al-Arabiyya*), opened: Qatif governorate about 750,000 people;
  al-Ahsa about 2.1 million, "70-75% Shia", so about 1.47 million; Eastern Province Shia "about 15.5% of
  all citizens". The extract showed no source for these. Other Arabic pages give 5-15% nationally and
  "about five million" (no source). The 70-75% for al-Ahsa conflicts with the "almost equal" Sunni/Shia
  split in English sources (search summary of State Dept excerpts: al-Ahsa council seats 50% Shia).
- Najran Ismailis: 150,000 in all Saudi Arabia (Arabic press, search summary); 250,000 (Matthiesen,
  snippet); 500,000 (an older IRF, snippet); 700,000 "inhabit the region of Najran" (IRF 2016, via
  English Wikipedia, not opened). HRW's *The Ismailis of Najran* (2008) gives only Najran's 2004
  population (408,000) and "a large majority".
- Zaydis about 20,000 near Yemen (IRF 2016 via Wikipedia). Medina's Nakhawila: no figure opened.

Checked against the census (`sa_geo.REGION_2022`):
- Eastern Province: 2,949,854 Saudis + 2,175,400 non-Saudis = 5,125,254. **25-30% of that is
  1.28-1.54M**, which would be 43-52% of its Saudis if every Shia there is a citizen.
- National 10-12% of 18,792,262 Saudis = **1.88-2.26M**. That leaves 0.34-0.97M for everything outside
  the Eastern Province (Najran, Medina, Riyadh, Jeddah).
- **Najran has 394,976 Saudis** (592,300 people). The 500,000 and 700,000 Ismaili figures are
  impossible; 250,000 would be 63% of Najran's Saudis, which matches "majority".

Verdict and placement: a real minority, about 10-12% of citizens. **One region (Eastern Province)
has a published share on a stated geography**, though the base (everyone or citizens) is unstated
and the source is the State Department with no origin given. Najran has "majority" in words and
no usable count. Medina has nothing.

§14: the 2015 IS bombings of Shia mosques in Qatif, Dammam and an Ismaili mosque in Najran
(`sa.md` §6); Qatif's 2011-13 unrest and the 2017 Awamiyah demolition. A region-level layer at 13
units (2.5M people each) shows nothing finer than the State Department already prints.

**Recommendation: NATIONAL-ONLY for now, and an ask is justified.** The om/sa ruling said "update if
better regional estimates turn up". What turned up is one region's share from a compiler, plus
arithmetic showing Najran's popular figures are wrong. A possible build: Eastern Province Saudis
at the State Dept's 25-30% (as a share of everyone, so 43-52% of Saudis), Najran's Saudis left on
`islam`, everyone else Sunni at 85-90% of citizens less what the Eastern Province takes. That is
Masaili-grade or weaker, and it is her call.

## Oman (`om`)

- **State Department 2023 IRF, Oman** (opened): "The government does not publish statistics on the
  percentages of citizens who practice Ibadhi, Sunni, and Shia Islam. The U.S. government estimates
  the population to be 95 percent Muslim: 45 percent Sunni, 45 percent Ibadhi, and 5 percent Shia."
  The base is everyone (citizens 59%), which makes the Sunni share of citizens lower than 45/95.
- Already in `sources/om.md` §3 and re-checked: Peterson 2004 (~45 Ibadi, 50 Sunni, <5 Shia and
  Hindu, of Omanis); AEI 2013 three-quarters Ibadi; Badr al-Abri (opened 2026-10-04,
  `baderalabri.com/?p=821`): "Sunnis are 77 percent", so Ibadis 21 and Shia 2, with no source and
  the author doubting it himself; nothing by governorate or wilaya.
- Arabic search for shares by governorate or wilaya (`نسبة الإباضية والسنة في سلطنة عمان حسب
  الولايات ...`): only national figures.
- **Placement:** Peterson's descriptions only (Dhofar entirely Sunni; Ibadi interior; Sunni east and
  Batinah Baluch; Lawatiya Shia in Matrah). No number below national.
- **§14:** the July 2024 IS attack on the Imam Ali mosque, Wadi al-Kabir (`om.md`).
- **Recommendation: NATIONAL-ONLY.** Ask 043's option b (Dhofar Sunni) is still the only placement
  anyone could defend.

## Azerbaijan (`az`)

National:
- **Pew 2012** Q31, p. 30: Sunni 16, Shia 37, something else 0, just a Muslim 45, nothing/DK 2.
- **State Department 2023 IRF, Azerbaijan** (opened): "According to SCWRA data, 96 percent of the
  population is Muslim, of which approximately 65 percent is Shia and 35 percent Sunni." (SCWRA =
  the State Committee for Work with Religious Associations.) The Sheikh-ul-Islam said the same
  65/35 (report.az, haqqin.az; search summary). The Caucasian Muslims Board in 2009 said 70-80%
  Shia (ru.wikipedia, opened, its footnote 2 not traced). Yunusov 2004 and 2013 also say "about
  65/35". These all share one figure, with no stated method.
- CRRC 2012: Shia 10, Sunni 4, Islam 85 (`az.md` §6).
- **Yunusov's own surveys** (Institute of Peace and Democracy; quota fieldwork, not probability):
  2003, 981 usable questionnaires in five regions, Muslim believers: just Muslim 58, Shia 30, Sunni 9.
  Baku 69 / 27 / 4; Absheron 47 / 47 / 6; elsewhere 62 / 24 / 14. In the west and north (Ganja,
  Gazakh, Sheki, Khachmaz, Gusar) over 70% chose "just Muslim". The 2013 repeat (*Islamsky faktor v
  Azerbaidzhane*, Baku 2013, p. 203): just Muslim 61, Shia 25, Sunni 10.5 (plus Salafi 2,
  Nurcu 0.4). Same failure as every sect item: the north, where the Sunnis are, answers "just Muslim".

Placement:
- **Historical counts by uyezd.** Yunusov 2004 (`ipd-az.org/wp-content/uploads/2020/06/Islam-az-rus.pdf`,
  216 pp.), Table 4 (1913, from *Kavkazsky kalendar na 1914 god*, pp. 110-133) and Table 5 (1916, from
  *Kavkazsky kalendar na 1917 god*, pp. 178-221). Shia and Sunni counts per uyezd, Sunni share of
  Muslims in 1916: Zakatal okrug 99, Nukha 91, Quba 89, Areş 79, Shamakhi 66, Goychay 57,
  Yelizavetpol 37, Kazakh 32, Karyagino 32, Baku 21, Javanshir 19, Zangezur 10, Javad 6, Lankaran 6,
  Shusha 6, Nakhchivan 1, Sharur-Daralagez 1. Total 1,167,863 Shia and 728,392 Sunni (62/38). He also
  prints 1886 family-list figures (Table 3, *Svod statisticheskikh dannykh ... 1886*). These are
  colonial counts; Javad's swing from 52% to 6% Sunni in three years shows they are not stable either.
- Modern: the Sunni ethnic minorities (Lezgins 167,570, Avars 48,636, Tsakhurs 13,361 in 2019 per
  `az.md`) are 229,567, about 2.3% of Muslims, against a claimed 35% Sunni. Most Sunnis are
  Azerbaijanis, whom no ethnicity table separates. `az.md` §6 already refused the ethnic-only layer
  for that reason.
- **EVS 2017 Azerbaijan** (`v52_cs`) has only a "Muslim" code, 3101 (search summary of the GESIS
  page; GESIS answered 403). **WVS 6:** the pooled V144 has a Shia code with 51 cases across all 59
  countries (IHSN catalogue, opened), so Azerbaijan's ~1,000 Muslims cannot be mostly on Shia/Sunni
  codes. Not a source either way.
- Azerbaijani search (`Azərbaycanda sünnilərin sayı faizi rayonlar üzrə ...`): nothing.
- **§14:** most of Yunusov's 2013 list of 256 convicted believers are Sunni (Salafi), 70%. Shia
  activists (Muslim Unity Movement, Nardaran) are also prosecuted. Both sides are watched.
- **Recommendation: NATIONAL-ONLY.** A §15 national row is possible, but the 65/35 and Pew's 37/16/45
  disagree in kind. The 1916 uyezd table is the best placement anyone has and is a century old.

## Afghanistan (`af`)

- **Pew 2012** Q31, p. 30: Sunni 90, Shia 7, something else 0, just a Muslim 3, nothing/DK 0 (1,509
  interviews, all 34 provinces; `af.md` §3).
- **State Department 2023 IRF, Afghanistan** (opened): WRD 2022 Sunni ~89, Shia ~11; Izady's Gulf/2000
  Shia "as high as 29"; about 90% of Shia are Hazaras; "approximately 25 percent or more of Hazaras are
  Sunni"; Ismailis "mainly in Kabul and in the central and northern provinces"; Ahmadis "in the
  hundreds", mostly Kabul.
- Persian: WikiShia 25-30% citing Bakhtyari 2006; Khwati 2003 Ismailis 3% (`af.md`). New this round:
  hawzah.net, *Joghrafiya-ye Shi'e dar Afghanistan* (opened, no author or date, credits abna.ir):
  Herat city about 50% Shia and 30% around it; Kabul about half; Ghazni "more than half"; Bamyan,
  Daykundi and Wardak majority. No method and no source.
- **Verdict:** a real minority (7-30% depending on who counts). Nothing places it with a number on a
  method. Ethnicity is not sect here (`af.md` §3).
- **§14:** Hazara and Shia targets of ISKP bombings (Dasht-e Barchi, Kabul; Kunduz and Kandahar
  mosques 2021) and of Taliban land evictions in Daykundi. A province layer marks them.
- **Recommendation: NATIONAL-ONLY.**

## Tajikistan (`tj`)

- **Pew 2012** Q31, p. 30: Sunni 87, Shia 3, something else 0, just a Muslim 7, nothing/DK 2. Whether
  Pew's sample covered GBAO was not checked (the methodology pages were not searched).
- **State Department 2023 IRF, Tajikistan** (opened): "According to local academics, the country is
  more than 90 percent Muslim, of whom the majority adhere to the Hanafi school of Sunni Islam.
  Approximately 3 to 4 percent of Muslims are Ismaili Shia, a majority of whom reside in the GBAO
  region." Russian sources (search summary): about 135,000 Ismailis in GBAO; ru.wikipedia says 5%
  Shia, mostly Ismaili.
- From `tj.md` §8: Pamiri districts 154,623 people (2020); Vanj, Darvoz and Murghob 73,293, mostly
  Sunni. GBAO 227,916, 2.4% of 9,657,005. 3-4% of ~9.62M Muslims is 290,000-385,000, so on the State
  Department's figure 60,000-160,000 Ismailis live outside the Pamiri districts (Dushanbe, and Pamiri
  villages resettled in the Vakhsh valley in Soviet times; no figure for either).
- **Recommendation:** Sunni on the four regions outside GBAO, with about 1-2% left unspecified for
  Ismaili migrants. GBAO stays on `islam` with no Ismaili node, as ask 051 ruled. If Anita prefers no
  split anywhere in `tj`, LEAVE.

## Uzbekistan (`uz`)

- **Pew 2012** Q31, p. 30: Sunni 18, Shia 1, something else 0, just a Muslim 54, nothing/DK 26.
- **State Department 2023 IRF, Uzbekistan** (opened): "According to the Uzbek government, there are
  35 million Sunni Muslims, 122,000 Shiite Muslims ..." That is 0.35% of Muslims. The report says
  four Shia mosques operate and Bukhara's Hoji Bahrom mosque reopened. The government forbids
  training Shia imams in the country. Older reports and a search summary put the Shia (Irani/Ironi,
  Persian-speaking) in Bukhara and Samarkand provinces; not re-opened.
- The 2026 census's eight nationality groups have no Iranian or Azerbaijani row (`uz.md` §9).
- **Recommendation: ASSIGN-ALL** `islam.sunni`, 0.35% unspecified; or put that 0.35% on `islam` in
  Bukhara and Samarkand only, which is ascription by words.

## Kazakhstan (`kz`)

- **Pew 2012** Q31, p. 30: Sunni 16, Shia 1, something else 0, just a Muslim 74, nothing/DK 10.
- State Department 2023 IRF (search summary; the PDF URL pattern used for the others did not resolve):
  other Muslim groups "include Shafi'i Sunni, Shia, Sufi, and Ahmadi"; only Hanafi Sunni groups are
  registered. No figure.
- The census counts **Azerbaijanis 145,615 (2021)**, 67% of them Muslim (Wikipedia, citing the
  census; `kz.md` gives their 27.3% refusal). The census engine crosses nationality, religion and
  oblast, so their Muslims can be put on `islam` per oblast. They are usually described as Shia,
  but no source was opened. Some Caucasus deportee groups classed as Azerbaijani were Sunni
  (Karapapakh, Terekeme), so this is weak ascription. Kurds and Turks (Meskhetian) are Sunni.
- **§14:** Ahmadis refused registration (`kz.md`).
- **Recommendation: ASSIGN-ALL** `islam.sunni` for census `Ислам`, remainder ~1% unspecified;
  optionally leave Azerbaijanis' Muslims on `islam` per oblast.

## Kyrgyzstan (`kg`)

- **Pew 2012** Q31, p. 30: Sunni 23, Shia 0, something else 0, just a Muslim 64, nothing/DK 12.
- **State Department 2023 IRF, Kyrgyz Republic** (opened): "The government estimates the Shia
  community makes up less than 1 percent of the Muslim population." Ahmadis about 1,000 (an
  international organisation, 2020). Tengrists claim 50,000; they are not Muslim.
- **Recommendation: ASSIGN-ALL** `islam.sunni`, under 1% unspecified.

## Turkmenistan (`tm`)

- No survey asks sect (`branches.md`). **State Department 2023 IRF, Turkmenistan** (opened): "93
  percent Muslim (mostly Sunni) ... There are small pockets of Shia Muslims, consisting largely of
  ethnic Iranians, Azeris, and Kurds, some located in Ashgabat, with others along the border with
  Iran and in the western city of Turkmenbashi."
- **The 2022 census nationality table** (`tm_geo.NATIONALITY`, read from
  `stat.gov.tm/population-census-pdfs/results/en/4.pdf`) gives those three groups by velayat:

  | velayat | Azerbaijanis | Persians | Kurds | sum |
  |---|---:|---:|---:|---:|
  | Ashgabat | 10,376 | 584 | 2,159 | 13,119 |
  | Ahal | 1,135 | 5,479 | 113 | 6,727 |
  | Balkan | 7,389 | 38 | 21 | 7,448 |
  | Dashoguz | 402 | 47 | 57 | 506 |
  | Lebap | 938 | 131 | 12 | 1,081 |
  | Mary | 6,336 | 4,718 | 377 | 11,431 |
  | **Turkmenistan** | 26,576 | 10,997 | 2,739 | **40,312 (0.57%)** |

  The geography matches the State Department's three places. The Kurds here are Khorasan Kurmanji,
  whom the State Department lists among the Shia. That they are Shia is ascription; no source was
  opened for it.
- **Recommendation: ASSIGN-ALL** `islam.sunni`, with these three nationalities' Muslims left on
  `islam` per velayat. Drawing them as `islam.shia` would be ethnic assignment, the tier Anita
  described for that.

## Search record (2026-10-04)

- Arabic: Shia share in the Eastern Province, Qatif and al-Ahsa; Najran Ismailis; Saudi citizens'
  Shia share and the 2022 census; Ibadi/Sunni by Omani governorate; Qatar citizens' Shia; UAE
  citizens' Shia (Dubai, Sharjah). Pages opened: ar.wikishia (Saudi Arabia), ar.wikipedia (Shi'a of
  Saudi Arabia), HRW Arabic *Ismailis of Najran* ch. 2, baderalabri.com. Blocked: hawamer.com (403),
  fanack.com (403), cairn.info Hérodote 2009 on al-Ahsa (403), crisisgroup.org (Cloudflare),
  refworld ICG copy (403), Cambridge excerpt of Matthiesen (timeouts).
- Russian: Yunusov, Sunni share by district; Ismailis in Tajikistan. Opened: both Yunusov PDFs (2004,
  2013), ru.wikipedia *Islam v Azerbaidzhane*.
- Azerbaijani: Sunni share by rayon. Nothing.
- Persian: Shia share by Afghan province. Opened: hawzah.net.
- English: State Department 2023 IRF PDFs for af, az, jo, kg, om, qa, sa, tj, tm, ae, uz (opened, in
  scratchpad); Pew 2012 p. 30 (opened); EVS 2017 and WVS 6 denomination codes; Wikipedia *Shia Islam in
  Saudi Arabia* and *Azerbaijanis in Kazakhstan*.
