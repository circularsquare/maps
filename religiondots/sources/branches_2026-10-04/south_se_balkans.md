# Muslim branches: Indonesia, Bangladesh, Malaysia, Brunei, Maldives, Bosnia, Kosovo, Albania (+ Pakistan check)

Scout run 2026-10-04. Read first: `sources/branches.md` (whole), `sources/{al,ba,bd,bn,mv,my,xk,id}.md`
(grepped), `ask/RULINGS.md` (grepped). About 28 WebSearch calls. Nothing written in the project.

## Summary

Shares are % of the country's Muslims. "Remainder" is what I'd leave on bare `islam` if the rest
goes to `islam.sunni`.

| cc | best national minority figures | one branch? | placement below national | recommendation |
|---|---|---|---|---|
| id | Shia ~200k (Kemenag Balitbang 2016 framing) to 2.5M (IJABI claim); Ahmadis 80k (Kemenag) to 200-500k (independent), JAI claims 1.1M | yes: Shia 0.1-1%, Ahmadi 0.03-0.5% | none with numbers (city lists only; JAI's own claim puts 770k of 1.1M in West Java) | **ASSIGN-ALL** Sunni, ~1.5% left on `islam` |
| bd | Shia 2% (Pew 2012 survey; Pew 2009 ascription <1%); Ahmadis ~100k (community claim, 0.07%) | borderline yes | none | **ASSIGN-ALL** Sunni, ~2.5% left (Pew's Sunni/Shia row is already on the estimate layer) |
| my | Shia: JAKIM 1,000-1,500 (2013) vs community 50k-300k; Pew 2009 <2%; Ahmadis ~1,500-2,000 | yes | JAKIM names Selangor, Perak, Johor as "most active", no counts | **ASSIGN-ALL** Sunni, ~1.5% left |
| bn | Pew 2009 <1%; Shia and Ahmadiyya banned by fatwa | yes | none | **ASSIGN-ALL** Sunni, ~1% left |
| mv | citizenship is Sunni-only by law; foreign Muslims (mostly Bangladeshi) carry their own countries' shares | yes | n/a | **ASSIGN-ALL** Sunni (Maldivians 0% remainder; foreigners at bd's ~2.5% if wanted) |
| ba | Pew 2012 Shia 0%; Pew 2009 <1%; Shia institutions and a registered Ahmadi association exist, no count | yes | none | **ASSIGN-ALL** Sunni, ~1% left |
| xk | Pew 2012 Shia 1%, "something else" 2%; Pew 2009 no estimate; tarikat umbrella claims ~60k members (mostly Sunni Sufi orders) | yes | none | **ASSIGN-ALL** Sunni, ~3% left |
| al | census `Mysliman` cell is non-Bektashi Muslims; KAS 2024 survey: Sunni 36.0% of adults vs "other Shia tariqas" 0.5% (~1.4% of non-Bektashi Muslims, ~4 respondents) | yes, once Bektashi is out | Bektashi already drawn by qark from census | **ASSIGN-ALL** `Mysliman` to Sunni, ~1.5% left |
| pk | nothing new (Urdu search found only the CIA/Pew chain, and a false "1998 census 25% Shia") | — | — | no change |

**Common caveat, for whoever decides.** For every country here the census asks only "Muslim" (or
Albania's `Mysliman` beside `Mysliman - Bektashi`). ASSIGN-ALL means filing a census cell on
`islam.sunni` from outside the source, which spec §2.6 forbids as written and which the
"assigned from ethnicity"/ascription tier (§2.6a, branches.md "Direction") would have to carry.
Pew 2012's Sunni shares cannot do the job instead: Indonesia, Bosnia, Kosovo and Albania have a
majority on "just a Muslim" (56, 54, 58, 65%), so a Pew Sunni row would sit beside a parent that
is really Sunni (branches.md already says this). The case for ASSIGN-ALL is that every credible
minority figure is small and unplaceable, not that anyone measured Sunni.

Pew 2009 *Mapping the Global Muslim Population*, "Estimated Percentage Range of Shia by Country"
(`https://www.pewresearch.org/wp-content/uploads/sites/7/2009/10/Shiarange.pdf`, report p. 39,
read at source): Shia as % of Muslims, Albania <5, Bangladesh <1, Bosnia-Herzegovina <1, Brunei
<1, Indonesia <1, Malaysia <2, Maldives <1, Kosovo "--" (no estimate). These are compiler
estimates (consultants plus WRD ethnic ascription, branches.md "Decided"), and their bins are
ceilings, not measurements.

---

## Indonesia (id)

**National figures**

| group | figure | who / basis | year | where read |
|---|---|---|---|---|
| Shia | ~200,000 "tersebar di seluruh wilayah" | stated in the write-up of Balitbang Diklat Kemenag's 2016 qualitative study of Shia in 22 locations (published as *Dinamika Syiah di Indonesia*, 2017); the figure's own origin is not given, and the same article says "belum ada jumlah yang valid" | 2016 | `nu.or.id/balitbang-kemenag/syiah-di-indonesa-seperti-apa-BCObY` (read) |
| Shia | 6-7 million | BIN and Mabes Polri, per the same article; no document | ~2016 | same |
| Shia | 2.5 million in 84 branches and 145 sub-branches in 33 provinces | IJABI (Jalaluddin Rakhmat), sect leader's claim | 2008 | `ahmadbinhanbal.com/sekilas-data-syiah-di-indonesia/` (read; anti-Shia site quoting it) |
| Shia | 20,000 (Ahmad Baragbah, 1995) and 3 million (Dimitri Mahayana, 2000) | quoted by Zulkifli, *The Struggle of the Shi'is in Indonesia* (ANU Press 2013, OAPEN open access), who calls all estimates "without basis" and says an ICC Jakarta questionnaire census of Shia teachers and followers (2000) failed because Shia did not return it | 1995/2000 | search snippet of the OAPEN PDF; the PDF download failed (ECONNRESET), page not verified |
| Shia | "one million" | imamreza.net (id.wikipedia's only cited figure) | undated | id.wikipedia "Islam Syiah di Indonesia" (read) |
| Ahmadi | 80,000 | Ministry of Religion, via Burhani | — | id.wikipedia "Ahmadiyyah di Indonesia" (read; footnote names Burhani only) |
| Ahmadi | 200,000-500,000 in 542 branches | Vaughn, CRS report, Nov 2010 | 2010 | same |
| Ahmadi | 1,100,386, of whom 770,270 in West Java | "menurut perkiraan, pada tahun 2005", citing Sofianto 2014 (Kunto Sofianto, a Padjadjaran historian of Ahmadiyya in West Java); almost certainly JAI's own tajnid claim | 2005 | Kuswanto et al. 2024, *Al-Alamiyah* (`miftahul-ulum.or.id/ojs/index.php/alamiyah/article/download/190/101`, read, p. ~3) |
| Ahmadi | ~600,000 in 192 kabupaten/kota in 38 provinces | news summary (Tempo), community figure | recent | search snippet only |

Against ~237M Muslims (SP2020-ish): Shia 0.08% (200k) to 1.05% (2.5M); BIN's 6-7M (2.8%) has no
source and is the outlier. Ahmadis 0.03% (80k) to 0.2% (500k), 0.46% on JAI's 1.1M.

**One branch?** Yes. Every figure with any standing is ≤1% Shia and ≤0.5% Ahmadi; the high Shia
claims are from IJABI (an interested party) or unsourced security services. Nahdlatul Ulama and
Muhammadiyah are both Sunni (Shafi'i in fiqh for NU), so the ormas split in branches.md does not
touch Sunni/Shia. Pew 2012 (Sunni 26, something else 5, just Muslim 56) and GFS (Sunni 6.0, Shi'a
0.4, just Muslim 92.3) show Indonesians do not use the Sunni label, not that they are not Sunni.

**Placement.** Nothing with numbers below national. Named concentrations only: Shia in Bandung
(Tempo: "kantong Syiah terbesar"), Jakarta, Makassar, Pekalongan, Tegal, Jepara, Semarang, Garut,
Bondowoso, Pasuruan, Madura (Sampang community ~700 before the 2012 expulsion); the 2016 Balitbang
study sites (Jakarta, Kota Tangerang, Cirebon, Bogor, Garut, Tasikmalaya, Surabaya, Malang,
Bondowoso, Jember, Semarang, ...). Ahmadis: Manislor village (Kuningan) 3,026 of 4,300 (Saefullah
2016, via Kuswanto); Tenjowaringin (Tasikmalaya); Parung (Bogor, the national centre); the Transito
displaced in Lombok. JAI's 2005 claim of 70% in West Java is the only province figure and is the
community's own. Nothing to draw.

**§14.** Ahmadis (Cikeusik killings 2011, the 2008 SKB decree, Kuningan's Jalsah ban Dec 2024) and
Shia (Sampang 2012) are both persecuted; the village and city names above are exactly the
placement §14 worries about. ASSIGN-ALL with a remainder on `islam` draws neither.

**Recommendation: ASSIGN-ALL** to `islam.sunni`, ~1.5% left on `islam` (covers IJABI's 2.5M plus
the independent 500k Ahmadi high end, 1.2%).

## Bangladesh (bd)

**National figures**

| group | figure | who / basis | year | where read |
|---|---|---|---|---|
| Shia | 2% (Sunni 92, just Muslim 4) | Pew 2012 Q31, face-to-face survey, n≈1,918 Muslims, so ~38 Shia answers | 2011-12 | branches.md; already on the estimate layer (`estimates.md` row: Sunni 140m, Shia 3m) |
| Shia | <1% | Pew 2009 compiler estimate | 2009 | Shiarange.pdf (read) |
| Shia | ~2.97 million "(2011 statistics)" | bn.wikipedia infobox, which is Pew 2012's 2% times the Muslim total | — | bn.wikipedia "Bangladeshe Shia Islam" (read) |
| Shia | "prai 10 lakh" (~1M) | VOA Bangla, 28 Nov 2015, after the Hussaini Dalan bombing; unattributed | 2015 | `voabangla.com/a/shia-plight-in-bd/3078740.html` (read) |
| Shia | 40,000-50,000 (2009) | "another source" in a Bengali search summary | 2009 | search snippet only, origin not found |
| Ahmadi | ~100,000; 103 branches, 425 places | Ahmadiyya Muslim Jamaat Bangladesh, community claim | — | search summary of bn.wikipedia "Ahmadiyya Muslim Jamaat, Bangladesh" (not opened) |

bn.wikipedia also says most Bangladeshi Shia are Urdu-speaking ("Bihari"), whose whole community
is a few hundred thousand, which sits badly with 3M and better with Pew 2009's <1% or VOA's ~1M.
So the credible Shia range is about 0.3-2% of 150M Muslims. Ahmadis 0.07%.

**One branch?** Borderline yes. Pew 2012's 2% is the only survey figure, rests on ~38 respondents,
and is double Pew's own 2009 compiled ceiling. No Bengali source with a method was found
(searched Shia number, Hussaini Dalan, Urdu-speaking Shia, Ahmadi districts in Bengali).

**Placement.** None with numbers. Named: Shia in Old Dhaka (Hussaini Dalan, Bakshibazar), Mohammadpur
and Mirpur camps (Geneva Camp), Saidpur (Nilphamari), Khalishpur (Khulna), Manikganj, Kishoreganj,
Thakurgaon; Dawoodi Bohra in Chattogram. Ahmadis: Panchagarh (Ahmadnagar-Shalshiri), Brahmanbaria,
Sundarban area, Rajshahi, Cumilla, Jamalpur-Mymensingh (community list).

**§14.** Ahmadis: Ahmadnagar, Panchagarh, Jalsa attacked March 2023 (from memory, not
re-checked); Shia: Hussaini Dalan bombing Oct 2015 and an attack at Shibganj (Bogura) (both named
in the VOA piece). No placement is proposed.

**Recommendation: ASSIGN-ALL** to `islam.sunni`, ~2.5% left on `islam` (Pew 2012's 2% Shia plus
Ahmadis and rounding). The Pew row already on the estimate layer stays as the national statement.

## Malaysia (my)

**National figures.** Census 2020 Islam is one cell (`my.md`), and it includes non-citizen Muslims.

| group | figure | who / basis | year | where read |
|---|---|---|---|---|
| Shia | ~1,500 practitioners, "paling aktif" in Selangor, Perak, Johor | JAKIM, Mohd Aizam Mas'od (research division), enforcement knowledge | 14 Dec 2013 | `mstar.com.my/lokal/semasa/2013/12/14/jakim-kenal-pasti-1500-pengamal-syiah-di-seluruh-negara` (read) |
| Shia | 1,000-1,500, of whom 300-500 active | JAKIM statistics, per a search summary | — | search snippet only |
| Shia | 50,000-200,000 | a Shia follower, per news | — | search snippet (themalaysianinsight / benarnews), not opened |
| Shia | 250,000-300,000 | *Afkar*, sourced to shianumbers.com | — | branches.md |
| Shia | <2% | Pew 2009 | 2009 | Shiarange.pdf (read) |
| Ahmadi | ~1,500-2,000, centred on Kampung Nakhoda, Batu Caves (Selangor) | news summaries | — | search snippet only |

Pew 2012: Sunni 75, Shia 0, just Muslim 18 (and Shia 0 is under a ban, branches.md).

**One branch?** Yes. Even the community's 300k is 1.5% of ~20.6M; JAKIM's 1,500 is 0.007%. Shafi'i
Sunni is the legal norm in every state; Shia teaching banned by National Fatwa Council ruling
(1996, from memory). Non-citizen Muslims (Indonesian, Bangladeshi, Pakistani) bring small Shia shares of their own.

**Placement.** JAKIM's three states, unquantified; the Ahmadi village at Batu Caves. Nothing to draw.

**§14.** Both groups legally suppressed and watched (Selangor's religious department raided the
Ahmadi centre in 2014, 38 detained, per search summary; BenarNews 2019 reports Shia arrests,
headline only). No placement.

**Recommendation: ASSIGN-ALL** Sunni, ~1.5% left on `islam` (the community high end).

## Brunei (bn)

**National figures.** Only Pew 2009 (<1% Shia, compiler). Census has one Islam code (`bn.md`).
Shia Islam, Ahmadiyya (Qadiyaniah), Al-Arqam and others are on MORA's list of banned "deviant"
teachings by fatwa of the State Mufti / Majlis Ugama Islam (search summary of the State Dept IRF
2019-2023 Brunei reports and Humanists International; state.gov returns 403, not read here, as
in `bn.md`). Shafi'i Sunni is the state's legal madhhab.

**One branch?** Yes; no minority figure exists at all, and practice outside Shafi'i Sunni is an
offence. Foreign workers inside the Muslim cell (Bangladeshi, Indonesian) bring small shares.

**Placement / §14.** None. `bn.md` already flags non-Shafi'i Muslims as the people the Sharia Penal
Code bears on; ASSIGN-ALL does not place them.

**Recommendation: ASSIGN-ALL** Sunni, ~1% left on `islam` (Pew 2009's ceiling).

## Maldives (mv)

**National figures.** Maldivians are coded Islam by the tabulation, never asked (`mv.md`). By law
only Muslims may be citizens and only Sunni Islam may be practised by citizens (State Dept IRF
reports, via search summary; state.gov PDFs 403). Pew 2009: <1% Shia. No Shia or Ahmadi figure
for Maldivians anywhere (searched English only; Dhivehi not attempted, as no Dhivehi-language
statistical literature turned up in English results).

**One branch?** Yes, as a legal fact for citizens. Foreign Muslim residents (mostly Bangladeshi,
some Indian and Pakistani) are drawn separately in `mv` and carry their origin countries' shares.

**§14.** Any non-Sunni practice by a citizen is criminalised; nothing to place.

**Recommendation: ASSIGN-ALL** Sunni. Maldivians with no remainder (their Muslim status is
already an assignment); foreign Muslims at bd's ~2.5% remainder if the build wants consistency.

## Bosnia and Herzegovina (ba)

**National figures.** 2013 census `Islam` 1,790,454 (50.7%), one cell. Pew 2012: Sunni 38, Shia 0,
just Muslim 54. Pew 2009: Shia <1%. The Islamic Community (IZ BiH) is Sunni Hanafi-Maturidi and the
Sufi tekkes sit inside it: 56 zikr sites, 47 Naqshbandi, 5 Qadiri, 3 Rifa'i, 1 Shadhili (search
summary of `mizbijeljina.ba/tarikati`), all Sunni Sufi.
Shia: Iranian-funded institutions (Ibn Sina institute, Mulla Sadra foundation, Bastina duhovnosti,
Persian-Bosnian College, Sahar TV) are documented in *Vaninstitucionalna tumačenja islama u BiH:
djelovanje NVO i medija* (Institut za islamsku tradiciju Bošnjaka, 2018; author not noted, `iitb.ba/wp-content/uploads/2018/11/VANINSTITUCIONALNA-TUMA%C4%8CENJA.pdf`,
read): no headcount. Vukelić, "The Spread of Shia Islam in BiH", *Kultura Polisa* 2017: no
figure (abstract read). Ahmadis: "Ahmadija muslimanski džemat BiH" registered as an association in
Sarajevo, no count. Searched in Bosnian ("broj šiita u BiH", "broj derviša ... Ahmadija BiH").

**One branch?** Yes. Shia and Ahmadi presence is institutional and small; no figure anywhere.

**Placement / §14.** None. Low sensitivity, though Shia converts are politically contested
(Iran-influence framing in Serbian and Croatian press).

**Recommendation: ASSIGN-ALL** Sunni, ~1% left on `islam`.

## Kosovo (xk)

**National figures.** 2024 census `Islam` 93.5%, one cell (`xk.md`). Pew 2012: Sunni 24, Shia 1,
something else 2, just Muslim 58, nothing/DK 15. Pew 2009: "--" (no Kosovo estimate). Shia: "a very
small, almost unnoticeable minority ... whose number is not precisely known", no institution
(Balkanweb/Reporteri, 27 Aug 2021, read via Balkanweb English). Sufi orders: the Bashkësia e
Tarikateve të Kosovës, recognised as the sixth religious community, is an umbrella of 9 orders
(Kaderi, Rifai, Saadi, Shazeli, Nakshibendi, Sinani, Bektashi, Halveti, Melami), 80+ teqes,
claimed ~60,000 members (search summary of kallxo.com / evropaelire.org, not opened). Only the
Bektashi of those orders is a separate node here; the rest are Sunni Sufi.

**One branch?** Yes. Pew's Shia 1% (~10 respondents) and "something else" 2% are the ceiling; the
tarikat claim of 60k is 4% of Muslims and mostly Sunni Sufi.

**Placement.** Teqes in Gjakovë, Prizren, Rahovec (`xk.md` §5), no counts. Nothing to draw.

**§14.** Low.

**Recommendation: ASSIGN-ALL** Sunni, ~3% left on `islam` (Pew's Shia 1 + something else 2,
which also holds the few Bektashi).

## Albania (al)

**What the Muslim cell is.** The 2023 census form (INSTAT, `instat.gov.al/media/12433/pyetesori-cens-2023-shqip.pdf`,
p. 15, read) asks "Cila është feja ose besimi që ndjek ose i përket <Emri>?" with the
instruction "MOS lexoni alternativat" (do not read the options); codes are `Mysliman`, `Mysliman -
Bektashi`, Catholic, Orthodox, Evangelical, other (specify), prefer not to answer. So `Mysliman`
is every Muslim who did not say Bektashi: no Sunni word on the form.

**National figures**

| group | figure | who / basis | year | where read |
|---|---|---|---|---|
| Sunni ("Main Muslim Branch") | 36.0% of adults | KAS survey, Shehu and Zaloshnja, face-to-face, n=820, "based on personal beliefs" | late 2024 | `kas.de/documents/271859/0/Statistical+Study+on+Religious+Belief+in+Albania+%281%29.pdf`, p. 4 (read; kas.de's firewall blocks curl, WebFetch got through) |
| Bektashi tariqa | 5.2% | same | 2024 | same |
| "Other Shia tariqas" | 0.5% (±0.0 printed, i.e. ~4 respondents) | same; the category name is the authors' (Albanian usage often calls the Halveti, Rifai, Sadi and Kadiri orders "Shia tarikats" for their Alid devotion) | 2024 | same |
| census reading | "45.9% ... declared themselves as Sunni Muslims" | the same authors' gloss of the census `Mysliman` cell | 2023 | same, p. 2 |
| Pew 2012 | Sunni 10, something else 13, just Muslim 65 | survey; "something else" is probably mostly Bektashi | 2011-12 | branches.md |
| Pew 2009 | Shia <5% | compiler | 2009 | Shiarange.pdf |

KAS gives "other Shia tariqas" at 0.5/(36.0+0.5) = **1.4% of non-Bektashi Muslims**. Its age table
(p. 5) has them at 0.8 / 0.0 / 0.9 across age bands, which is noise at n=820. Ahmadis: a small
mission in Tirana, no figure found. The Iranian MEK camp at Manëz (Durrës), a few thousand Iranian
residents, would sit in `Mysliman` if enumerated; nothing confirms it.

**One branch?** Yes, once the census's own Bektashi split is drawn (it is). The remainder in
`Mysliman` is Sunni Hanafi plus Sunni-adjacent Sufi orders at about 1-1.5%.

**Placement.** The Bektashi are placed by qark from the census (`al.md` §6). Nothing places the
other tariqas; their teqes are known places but no counts.

**§14.** Low.

**Recommendation: ASSIGN-ALL** `Mysliman` to `islam.sunni`, ~1.5% left on `islam` (KAS's
"other Shia tariqas"). KAS itself reads the cell as Sunni, which is a published reading, not ours,
though still not the respondent's word.

## Pakistan (pk), brief check

Searched in Urdu for a district Shia share ("Pakistan mein Shia abadi ka tanasub zila-war"). Found
only: Urdu wikishia's Pakistan page (20% unsourced, Pew 2009's 17-26M, 19% infobox, a map of "Shia
towns" and a madrassa count by province, no population table, no method), and a search summary
claiming "the 1998 census found 25% Shia", which is false (no Pakistani census asks sect,
branches.md). **Nothing new.**

## Negatives, with what was tried

- **Indonesia per-province Shia/Ahmadi counts**: searched "jumlah penganut Syiah di Indonesia
  Kemenag", "sebaran komunitas Syiah per provinsi", "Puslitbang ... 2016 ... 22 daerah", "jumlah
  anggota Ahmadiyah per provinsi cabang JAI", the 770,270 figure. Kemenag's 2016 study is
  qualitative case studies; no per-site counts in what was reachable. Tempo pages answer 403.
  *Dinamika Syiah di Indonesia* (2017) itself was not found online.
- **Bangladesh**: two Bengali searches (Shia number; Ahmadi districts) plus one on Hussaini Dalan
  and Urdu-speaking Shia. Banglapedia pages return a placeholder ("Hello World") to WebFetch.
- **Malaysia**: Selangor state assembly answer "Gerakan Syiah" (`dewan.selangor.gov.my/question/gerakan-syiah/`)
  returned HTTP 500; may hold a state figure.
- **Maldives / Brunei**: state.gov IRF reports 403 on both the 2021-2025 archive and PDFs.
- **Bosnia, Kosovo, Albania**: Bosnian and Albanian searches for Shia and Ahmadi numbers; only the
  KAS Albania survey has a figure.
- **Zulkifli 2013** (OAPEN PDF) failed to download twice; its 20,000 / 3 million quotes are from a
  search snippet.
