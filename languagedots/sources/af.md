# Afghanistan (af): record

**Drawn, 2026-10-05**, session `edd42a8c-af2`, from the Ministry of Rural Rehabilitation and
Development (MRRD) provincial profiles' village-majority language figures, c. 2006-07, on NSIA's
1404 settled population. A proxy, `tier="derived"` on every row, allowed by Anita in ask 017
("very inaccurate but a vague picture of what is where is still useful", partly as context for
asia1m). 31,692,339 people drawn on 32 of 34 provinces; 10 nodes. **2026-10-06**: minority
languages the profiles never name (Pamiri, Kyrgyz, Parachi, Gawar-Bati, Brahui) added from
speaker estimates, §3b: 31,751,339 drawn, 19 nodes. Still to run (the session's
`taxonomy/build.py` call was refused by the permission check): `python taxonomy/build.py`,
`python tools/check_country.py af`, `python scatter.py --country af`, `claim.py done af`.

Files: `sources/af_mrrd.py` (transcription + checks), `data/raw/af/call_11-16_appa.htm`,
`data/normalized/af.csv`, `taxonomy/af2007.py`, `taxonomy/tree.d/af.txt`, `countries/af.py`.

## 1. The microdata search Anita asked for first (2026-10-05, second pass)

The ruling said safety need not rule the Asia Foundation's *Survey of the Afghan People* out, so
copies other than the withdrawn Australian Data Archive one (doi:10.26193/VDDO0X) were looked for.
None is open.

| where | what was found |
|---|---|
| asiafoundation.org, via Wayback (live site 403s) | The data page `where-we-work/afghanistan/survey/download-data-form/` (2016-2020) was a MachForm (`/forms/embed.php?id=11096`, captured 2016-02-09) requiring first and last name, email, organisation and purpose before the Stata/SPSS/SAS files were shown, under terms that the data "may not be reproduced or distributed without written consent from the Asia Foundation". So it was never open in the brief's sense (no form, no identity), and any third-party copy is a redistribution against those terms. A CDX sweep of asiafoundation.org for .sav/.zip/.csv/.dta/.por finds no data file captured (the files sat behind the form). |
| Harvard Dataverse (search API) | "Afghan People": no copy; the nearest is Sexton's replication data for northern-Afghanistan aid projects (doi:10.7910/DVN/88XL3K), his own survey |
| GitHub (code + repo search) | mentions only; `fabianreutzel/code_OOEC` uses the 2004-2019 file and lists it under `auxiliary_non_public` with an ADA "access request" |
| ICPSR / openICPSR | 403 to scripts; no study found by web search |
| OSF, Kaggle | nothing |
| Princeton DSS catalogue (resource2583) | a licensed library holding, campus only |
| The survey's own reports (2018 full report checked) | language is national only (D-11 "Which languages do you speak?", multi-answer: Dari 77%, Pashto 48%, Uzbeki 11%...); no province table |

The earlier pass's other leads stand (table below): DHS 2015 (registration off), REACH WoAA
(HDX Connect, by request), ALCS/NRVA (no language variable), SDES (literacy only), CLEAR Global
(national only). MICS 2010-11 uses language of household head as a background variable, but its
report tables are by region or by language, never crossed, and its microdata needs an account.

## 2. The source drawn

**MRRD / NABDP provincial profiles**, written for the Provincial Development Plans c. 2006-07. The
copies on nps.edu (Balkh, Herat checked) read word for word as the reprint used here: US Army
Center for Army Lessons Learned, *Handbook 11-16: Afghanistan Provincial Reconstruction Team
Handbook* (2011), Annex A "National and Provincial Data", one HTML page at
https://www.globalsecurity.org/military/library/report/call/call_11-16_appa.htm. Each province has
a sentence like "Dari is spoken by 77 percent of the population and 80 percent of the villages."
The profiles' figures are village counts weighted by village population: everyone in a village is
counted under its majority language.

**Base:** NSIA *Estimated Population of Afghanistan 1404* (2025-26), settled population by province
with its rural/urban split, as religiondots joined it to COD-AB's 34 provinces
(`religiondots/data/geo/af/af_lookup.csv`, read only); 34,935,197 settled people. 1.5 million
Kuchis have no province (gap, as religiondots).

**Checks** (`sources/af_mrrd.py`, every run): each province's transcribed sentence(s) must occur
verbatim in the downloaded page (all 34 do); 34 provinces both sides; NSIA total pinned; each
province's rural + urban = total; rows sum back to 34,935,197. There is no second table to check
the shares against; the national line in the same annex ("Dari 50%, Pashto 35%") is unsourced
and is not used.

## 3. How each province's sentence became shares

Kinds of figure: a percentage (as printed); a head count, over the profile's own 2008 province
population (Kapisa, Bamyan, Ghazni, Paktika, Paktya, Khost, Ghor, Uruzgan); a village count over
the profile's village total (Kunar 771, Badghis 964; Nangarhar's 8% split 60:36 by its Pashaie
and Dari village counts); Parwan's 5:2 ratio. Shares summing past 100 are scaled to 100 (Laghman
100.3, Paktya 101.5, Paktika 102.9, Daykundi 104, Nimroz 108). Shares short of 100 leave a `Not
described` row, not drawn (gap): Kapisa 11%, Baghlan 8%, Badakhshan 11%, Samangan 5%, Sar-e Pol
25%, Zabul 20%, Faryab 6.5%, Nuristan 7%, smaller elsewhere.

| province | drawn as |
|---|---|
| Kabul | rural 812,161: Pashto 60, Dari 40. Urban 5,361,333 (Kabul city): no figure; Dari 93.0, Pashto 7.0 by §3a |
| Herat | Dari and Pashto one figure 98 -> Dari 93.0 / Pashto 7.0 of it by §3a; Turkmen and Uzbek 2 -> Turkic |
| Kandahar, Helmand | Pashto 98 / 92; "Balochi and Dari" 2 / 8 -> Iranian |
| Takhar | not drawn: the profile names ethnic groups only (Uzbek, Tajik, Pashtun, Hazara) |
| Kunduz | not drawn: "Pashtu, Dari, and Uzbeki are spoken by 90 percent" is one figure across two families; Turkmen 8% alone would show Kunduz as Turkmen |
| Panjshir | Dari 100, from "the major ethnic group ... are the Tajiks" (no language sentence) |
| Kapisa | Dari 176,000 and Pashto 107,000 people, Pashai 17%: the head counts used, not the "30 percent" printed beside Dari (107,000 matches the 27% printed for Pashto) |

Named with no figure and left out: Kabul's Pashai (five villages), Badakhshan's Pashto, Turkmen
and Nuristani ("less than 1 percent each"), Daykundi's Turkmen (2 villages) and Balochi (1),
Zabul's Dari (second, no figure), Nangarhar's "other unspecified languages".

## 3a. Kabul city and Herat: the Dari/Pashto split (2026-10-06, session `5d7dac7e-af`)

**The fault.** Anita saw about 90% of the dots in Afghan cities drawn as "Iranian, not named". No
named label was on a group node: the cause was two rows that name no single language, Kabul
city (5,361,333, no figure in the profile) and Herat's "Dari and Pashtu are spoken by 98
percent" (2,335,538), both put on `indoeuropean.iranian`, 7,696,871 people, 24.3% of the map
and nearly all of its two largest cities.

**What else was looked for, to split them by a measured figure** (none found):
- the Kabul profile itself: one province-wide 60/40 Pashto/Dari sentence, the villages' (§5);
- the Herat profile: only the joint sentence; the nps.edu copy reads the same;
- CSO/UNFPA Kabul SDES 2013 (a near-census of Kabul province): no language question (IHSN
  catalogue 6771 lists the topics);
- the Asia Foundation reports, 2006 and 2011 read whole: language is national (and urban/rural
  in 2011, multi-answer "languages spoken": urban Dari 91, Pashto 44), never by region;
- Sustainability 15(8):6589 (2023), "urban ethnic segmentation in Kabul city": one district's
  case study, ethnicity, no shares.

**The split used.** One Dari share d for both rows, the one that makes the whole drawn map's
Dari:Pashto equal the Asia Foundation's *Afghanistan in 2006*, Appendix 3, Q-45 "in which
language did you learn to speak, first?" (single response, 6,226 adults, all provinces): Dari
49%, Pashto 40%, Uzbeki 9%, Turkmeni 2%, "Hazara" 0%. That is the only open single-answer
first-language figure for the country. Over the drawn named rows (Dari 8,717,952 incl.
Panjshir; Pashto 12,425,602): d = (49(P+U) - 40D) / (89U) = **0.930**. After the split the map
draws Dari 15,878,436 and Pashto 12,961,989 (1.2250 = 49/40, asserted). `sources/af_mrrd.py`
reads Q-45 from the downloaded PDF (`data/raw/af/taf_survey2006.pdf`, Wayback copy of
asiafoundation.org/pdf/AG-survey06.pdf) every run and stops if it changes. Rows are labelled
"Dari/Pashtu, by the national first-language residual", so they stay traceable.

**Sensitivity.** The residual is computed over the drawn map only, so it assumes the undrawn
Takhar, Kunduz and "not described" parts (3.2M) share the drawn map's mix. If instead those
were, say, 45% Dari and 20% Pashto, d falls to about 0.90. Either way it is a national figure
put on two places, not a measure of either; Herat at ~93% Persian-speaking is in line with
every description, Kabul city may have more Pashto first-speakers than 7%.

**Hazaragi, Aimaq, Tajik.** Not merged by a mapping: the profiles never name them (only
"Dari"), and Q-45's "Hazara" 0% shows Hazaras answer Dari when asked. Panjshir is the only
"Tajik" (an ethnic sentence, drawn as Dari, §5). Drawing Hazaragi would need a source that names
it; `au.txt`'s `indoeuropean.iranian.hazaragi` node exists if one turns up.

**What still draws as "not named":** Kandahar's and Helmand's "Balochi and Dari" (157,788, on
Iranian) and Herat's "Turkmeni and Uzbeki" (47,664, on Turkic): 205,452 people, 0.6% of the
map. Q-45's Balochi is 0% after rounding, so it gives no ratio worth using.

**The hatched areas (Takhar 1,196,656, Kunduz 1,258,535, plus 0.8M "not described" across
other provinces).** Not drawable from this source. Takhar's profile names ethnic groups in order
with no figure at all. Kunduz's gives one figure (90%) for Pashto, Dari and Uzbek together, plus
Turkmen 8%; the same residual trick across three languages would put Kunduz at about two-thirds
Dari, against every account of it as Pashtun-plurality, so it is not used. The "not described"
remainders are parts the profiles leave unaccounted for.

## 3b. Minority languages from speaker estimates (2026-10-06, session `5d7dac7e-pam`)

**The fault.** Anita: the map showed essentially no Pamiri languages. The profiles never name
them: Badakhshan's sentence is Dari 77, Uzbek 12, Pashto, Turkmen and Nuristani "less than 1
percent each", leaving 11% undescribed. Nor do they name Kyrgyz, Parachi, Gawar-Bati or Brahui.

**The route.** Ask 019's: cited speaker estimates, rows `modelled` (the rest stay `derived`),
each carved out of its own province so NSIA's province totals do not move
(`sources/af_mrrd.py` `MINORITIES`, `BRAHUI`). Under the profiles' village-majority premise a
Pamiri or Parachi village sits in none of the named figures, so these come out of the province's
`Not described` remainder, not out of Dari (a call; it also shrinks the gap). Figures are as
published, not grown to 2025: most are 1990-2012 estimates, so they are likely low.

| language | node | people | province | source | placed (countries/af.py `ZONES`) |
|---|---|---:|---|---|---|
| Shughni (Rushani in it) | `iranian.shughni` (by.txt) | 20,000 | Badakhshan | Endangered Language Alliance, Shughni page: "approximately 20,000" in Afghan Badakhshan | Afghan Shighnan on the Panj, 37.2-38.0N 71.2-71.8E (Kontur 46,780 people) |
| Wakhi | `iranian.wakhi` (by, pk) | 17,500 | Badakhshan | Wikipedia, Wakhi people infobox, Afghanistan 17,500 (2018). NSIA puts Wakhan district at 17,213 (Wikipedia, Wakhan District) | the corridor 71.7-73.6E, 36.4-37.3N (20,609) |
| Munji | `iranian.munji` (new) | 5,300 | Badakhshan | Ethnologue 18th ed. via Wikipedia, 5,300 (2008); Munjan valley, Kuran wa Munjan | 25 km of Glottolog munj1244 (4,436) |
| Sanglechi | `iranian.sanglechi` (new) | 2,200 | Badakhshan | Ethnologue 25th ed. via Wikipedia, 2,200 (2009); six villages of Zebak district | 15 km of Glottolog sang1344 (7,658) |
| Ishkashimi | `iranian.ishkashimi` (by) | 1,500 | Badakhshan | Wikipedia, Ishkashimi language: 1,500 in Ishkashim and Wakhan districts | 12 km of Ishkashim town (10,226) |
| Kyrgyz | `turkic.kyrgyz` (kg) | 1,500 | Badakhshan | T. Callahan, *The Kyrgyz of the Afghan Pamir Ride On* (2007; files.ethz.ch ISN 145126): 1,500, 600 Big Pamir, 900 Little Pamir | east of 73.6E, or east of 73.2E north of 37.2N (762) |
| Parachi | `iranian.parachi` (new) | 3,500 | Kapisa | Ethnologue via Wikipedia, 3,500 (2009), "mainly in the upper part of Nijrab District". Kieffer (Iranica, 1983) says ca. 5,000 in three valleys incl. Shutul | 15 km of 35.05N 69.65E (Glottolog's para1299 point is in Badakhshan, wrong) |
| Gawar-Bati | `dardic.gawarbati` (new) | 7,500 | Kunar | Ethnologue 25th ed. via Wikipedia: Afghanistan 7,500 | 25 km of Glottolog gawa1247 (Arandu), the Afghan valley opposite (75,516) |
| Brahui | `dravidian.northern.brahui` (tree.txt) | 200,000 | Nimroz 26,375, Helmand 156,851, Kandahar 16,774 | E. Bashir, SALRC workshop notes on Brahui (2003), "200,000 in Afghanistan" citing Ethnologue; Ethnologue 2016 via Joshua Project: "Helmand and Kandahar provinces: Chakhansoor to Shorawak" | the southern belt: Nimroz south of 31.0N outside 15 km of Zaranj (41,001), Helmand south of 31.3N (243,832), Kandahar south of 31.0N west of 66.3E (26,076); the provincial split is by those Kontur populations |

Brahui comes out of each province's Balochi-type row first (Nimroz `Balochi`, Helmand and
Kandahar `Balochi and Dari, one figure`), at most half of it, since Brahui are Baloch by identity
and bilingual in Balochi; the rest out of Pashto (Helmand 93,637, Kandahar 1,094). Brahui is
the weakest figure here: 200,000 is an old Ethnologue figure, Joshua Project's ethnic figure is
373,000, and it puts Brahui at 64% of southern Helmand's Kontur population. Reversing: drop
`BRAHUI` in `sources/af_mrrd.py`.

**Kontur under-counts the high valleys**: the Munji and Kyrgyz zones hold fewer people than the
estimate drawn into them (4,436 and 762), so their dots sit denser than Kontur's grid there.

**Considered, not drawn:** Ormuri (Kieffer: "fewer than a hundred", Baraki Barak, Logar); Gujari
(18,850 on Wikipedia, sourced only to dbs.org, and dispersed across the east); Central Asian
Arabic (6,000, 2003, scattered villages in the north); Mogholi (200, near extinct); Hazaragi and
Aimaq (large, but the profiles file them as Dari and §3a says why they stay so). **Pashai and
Nuristani were checked and are not under-recorded**: drawn about 415,000 and 160,000 against
Ethnologue's Pashai 400,000 (2011) and the Nuristani languages' roughly 100,000-140,000 (Kamkata-viri
40,000, Ashkun 40,000, Tregami 3,500 and smaller ones). Tajikistan draws its Pamiris as Tajik
(ask 010 ruling), so Badakhshan's Pamiri dots stop at the Panj; that is the census there, not a
fault here.

**Colours** (`tree.d/af.txt`): Shughni and Ishkashimi hand-picked (their generated colours were a
brown salmon near Wakhi's and Dari's; elsewhere they are a few migrants in by and ca); the
Pamiri languages are greens and teals against Dari's orange, Uzbek's pink, Kyrgyz red and
Wakhi's salmon (left as pk.txt tuned it). Parachi a darker green beside Pashai's cyan; Gawar-Bati
a pale aqua.

## 4. Nodes

Dari `indoeuropean.iranian.dari` (us.txt's node), Pashto, Balochi, Uzbek, Turkmen, `other` for
"some other language" (Ghazni, Paktika, Paktya). New: **Pashai** under Dardic (Glottolog pash1270
is four Pashayi languages; one leaf, conventional Dardic placement) and **Nuristani languages**
directly under Indo-European (Glottolog nuri1243, a branch; the profiles name it as one).
**Hazaragi does not appear**: the profiles never separate it, so Hazara villages are Dari.
Colours: Dari given 0.74 0.15 140 (was generated pale green #c0e5a8, too near Pashto's #ccc882);
Pashai Dardic cyan 0.72 0.11 215; Nuristani violet 0.66 0.13 290. The Dari colour also applies in
the US and Australia, where it is distinct from Persian #8ac191 and Pashto.

Result (2026-10-06, after §3a): Dari 50.1%, Pashto 40.9%, Uzbek 4.9%, Pashai 1.3%, Turkmen
1.2%, Nuristani 0.5%, Iranian not split 0.5% (Kandahar/Helmand), Balochi 0.4%, Turkic not split
0.2%, other 0.1%. Before §3a: Pashto 39.2%, Dari 27.5%, Iranian not split 24.8%.

## 5. Calls someone might reverse

- **Kabul city not on the profile's 60/40 Pashto/Dari.** The figure is the villages' (the
  profile says 81% of Kabul is urban and mentions "five villages" of Pashai); 60% Pashto
  contradicts every account of the city. Reversing: put AF01's urban row through the 60/40
  split in `sources/af_mrrd.py`.
- **Kabul city and Herat split 93/7 Dari/Pashto by the national residual (§3a)** rather than
  left on Iranian. Reversing: drop `split_unsplit()` from `main()` and restore the two labels
  to Iranian in `taxonomy/af2007.py`.
- **Takhar and Kunduz empty** rather than drawn on a guessed split.
- **Panjshir drawn as Dari from ethnicity**, a second proxy on top of the first (185,000 people).
- **Unnamed remainders not drawn** rather than scaled up into the named languages.
- Nangarhar, Kunar and Badghis use village counts as population shares, the proxy's own premise.

## 6. What would replace this

Any open province table of home or first language. The Asia Foundation microdata, if it is ever
republished openly, at province grain (about 300-500 respondents per province per wave) would
replace this outright; so would REACH WoAA household data if HDX opens it and its questionnaire
asks household language (not checked).

## Moved from countries/af.py text (2026-10-06 sweep)

From `note_public`: "The shares are applied to each province's settled population in the
statistics authority's estimate for 2025-26, and within a province the dots follow Kontur's
population grid."
