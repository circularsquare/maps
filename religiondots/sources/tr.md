# Türkiye — Diyanet İşleri Başkanlığı with TÜİK, *Türkiye'de Dinî Hayat Araştırması* (Ankara 2014)

Wired 2026-09-08. 85,279,553 people, 12 İBBS-1 regions, 8 drawn nodes, **100% of the register
population drawn and every row `modelled`**.

| | |
|---|---|
| source | **Diyanet İşleri Başkanlığı**, *Türkiye'de Dinî Hayat Araştırması*, Ankara 2014, 293 pp. Fieldwork, sample design and weighting by **TÜİK** |
| instrument | Q10 *Hangi dine mensupsunuz?* and **Q11 *Kendinizi hangi mezhebe ait hissediyorsunuz?*** |
| fieldwork | 15 May – 20 September 2013, CAPI on tablets, face to face |
| respondents | **21,632 adults**, 18+, one per sampled household |
| design | three-stage stratified cluster; 2,019 clusters (1,330 urban / 689 rural); 20 households per urban cluster, 16 per rural; frame = Ulusal Adres Veri Tabanı, February 2013 |
| estimation level | **"Türkiye total, Türkiye urban/rural, and İBBS-1 region totals"** — the report's own words |
| geography | **12 İBBS Düzey 1 regions**, 7.1 million people each — the coarsest country on this map |
| magnitude | OCHA **COD-PS 2022** at province, 85,279,553, summed to İBBS-1 |
| boundaries | Eurostat **GISCO NUTS 2024** level 1 — codes and names identical to the report's row labels |
| placement | **Kontur** 400 m hexes, 454,587 of them |
| licence | a state publication, open PDF, no account and no terms attached |

**The finding is not what the data says, it is where it lives.** `sources.md` §11r closed
Türkiye in September 2026 on the sentence *"Türkiye's own publication of religion is nothing
since 1965"*, having asked TÜİK and asked the census. Both of those were correct and the
conclusion was not: **religion in Türkiye is published by the religious affairs directorate,
and TÜİK ran the fieldwork for it.** Neither `tuik.gov.tr` nor the UNSD oracle will ever show
that. The general rule, now in §11ac's *Retired here*: before closing a country on *the office
does not publish it*, ask **which ministry would**.

---

## 1. What is parsed, and from where

Two tables, both read off the PDF by word position rather than retyped.

**Table 4, page 42 — `Ameli mezhep mensubiyetine göre kişi oranı (İBBS, 1. Düzey)`.** Nine
columns × the national row and all twelve regions. Shares **of those who answered Islam at
Q10**, which is 99.2% of the sample. Every row sums to 100 within the report's own rounding
note, asserted in `sources/tr.py`.

| İBBS-1 | Hanefi | Şafi | Maliki | Hanbeli | Caferi | Diğer | Hiçbiri | Bilmiyorum | Cevap vermeyen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TR Türkiye | 77.5 | 11.1 | 0.0 | 0.1 | 1.0 | 0.8 | 6.3 | 2.4 | 0.9 |
| TR1 İstanbul | 81.1 | 10.1 | 0.0 | 0.1 | 2.1 | 0.7 | 2.9 | 1.8 | 1.2 |
| TR2 Batı Marmara | 71.8 | 1.9 | – | – | 0.4 | 0.5 | **20.8** | 3.7 | 0.9 |
| TR3 Ege | 74.2 | 5.9 | 0.1 | 0.1 | 0.7 | 1.1 | 11.9 | 5.0 | 1.1 |
| TR4 Doğu Marmara | 86.9 | 3.5 | – | 0.0 | 0.9 | 0.3 | 5.2 | 2.7 | 0.6 |
| TR5 Batı Anadolu | 88.5 | 2.7 | 0.1 | – | 0.7 | 1.0 | 3.2 | 2.4 | 1.5 |
| TR6 Akdeniz | 81.0 | 7.6 | – | 0.0 | 0.9 | **2.0** | 6.1 | 2.0 | 0.4 |
| TR7 Orta Anadolu | 91.1 | 1.2 | 0.0 | – | 0.5 | 0.2 | 6.5 | 0.3 | 0.2 |
| TR8 Batı Karadeniz | 82.3 | 1.0 | 0.0 | 0.1 | 0.7 | 0.4 | 10.8 | 3.1 | 1.5 |
| TR9 Doğu Karadeniz | 92.1 | 0.2 | – | – | 0.1 | 0.3 | 6.4 | 0.6 | 0.5 |
| TRA Kuzeydoğu Anadolu | 55.7 | 35.2 | – | 0.2 | **4.6** | 0.1 | 2.9 | 1.4 | 0.1 |
| TRB Ortadoğu Anadolu | **45.7** | **48.7** | – | – | 0.3 | 1.3 | 1.5 | 2.1 | 0.4 |
| TRC Güneydoğu Anadolu | 53.6 | 42.0 | 0.0 | 0.1 | 0.3 | 0.3 | 2.2 | 1.1 | 0.4 |

**Table 1 / Grafik 1, page 38 — religion.** İslamiyet **99.2%**, *Diğer* **0.4%**, *Cevap
vermeyen* **0.5%**. **National only.** There is no regional religion table anywhere in the 293
pages, so those two residual shares sit at the same rate in every region and carry no
geography at all.

So a region's people are `population × 99.2% × its own madhhab shares`, plus 0.4% and 0.5% at
the national rate, apportioned by largest remainder so each region sums exactly.

## 2. What it draws that nothing else does

`branches.py`'s note on `islam.sunni` said the school of law is *"almost never enumerated: a
census that asks about religion at all normally stops at 'Muslim'"*, and that stayed true for
eighty-one countries. **Türkiye is the first source here that asks.** Five nodes are new:
`islam.sunni.hanafi`, `islam.sunni.shafii`, `islam.sunni.maliki`, `islam.sunni.hanbali` and
`islam.shia.jaafari`.

The Shafi'i band is the content. **48.7% in Ortadoğu Anadolu — the only region where it leads
Hanafi — 42.0% in the southeast, 35.2% in the northeast, against 0.2% on the eastern Black
Sea.** That is the Kurdish geography, drawn from the state's own survey rather than inferred
from who lives there, which is exactly the distinction §14.5 turns on. Ja'fari does the same
at a tenth of the scale: 1.0% nationally, **4.6% in Kuzeydoğu Anadolu**, which is Iğdır and
Kars on the Azerbaijani border.

Two internal checks pass. `Diğer` peaks at **2.0% in Akdeniz** — Hatay and Adana, where the
Nusayris are, and Nusayri is option 6 on the card with no column in the published table, so it
has been folded there. And Şafi is near zero across the whole Black Sea and Aegean, which is
what every account of Turkish Islam describes.

## 3. The hole: there is no Alevi box, and the questionnaire proves it

The report reprints its own instrument. Question 11, in full:

> **11. Kendinizi hangi mezhebe ait hissediyorsunuz?** — Hanefi (1), Şafi (2), Maliki (3),
> Hanbeli (4), Caferi (5), **Nusayri (6)**, Bilmiyorum (7), Diğer (belirtiniz…) (98), Hiçbiri
> (90), Cevap vermek istemiyorum (99).

Four Sunni schools, Ja'fari, Nusayri, and nothing else. **The word `Alevi` does not appear once
in 293 pages** — searched, zero hits, against fifteen pages carrying `Hanefi`.

So an Alevi respondent's honest answers are *Hiçbiri*, *Diğer*, *Bilmiyorum* or a refusal, and
all four are drawn here on the parent `islam` — 10.4% of Turkish Muslims, Russia's precedent
where *"Muslim, but neither Sunni nor Shia"* stays on the parent.

**`Hiçbiri` IS NOT A PROXY AND MUST NOT BE TREATED AS ONE.** It is 20.8% in Batı Marmara and
11.9% in the Aegean against **6.5% in Orta Anadolu**, which holds Sivas and Yozgat and is the
Alevi heartland by every settlement count there is. Whatever that column measures, its
geography is the opposite of Alevism's; §14.12 already found which way a fractional split over
a bucket like this fails.

### What was searched for an Alevi layer, and found not to exist

Anita, 2026-09-08: *"we can try to find alevi from other source and mix it in."* The hunt was
run and it came back empty. **No source published anywhere gives an Alevi share by region.**

- **KONDA** — the whole archived PDF library swept through the Wayback CDX API, 146 files.
  `2006_09_KONDA_Toplumsal_Yapi.pdf` (*Biz Kimiz?*) has the best national partition anybody
  publishes — **Sünni Hanefi 81.96, Sünni Şafii 9.06, Sünni Diğer 0.4, Alevi 5.02, Nusayri
  0.1, Şii 0.71, Diğer Müslüman 2.1, Ortodoks 0.06, Katolik 0.01, Protestan ve diğerleri
  0.057, Yahudi-Musevi 0.013, Diğer Din 0.04, Dini Yok 0.47** — and **no regional table for
  any of it.** Page 29 says in prose that a third of Alevis live in Istanbul and the next
  densest regions are Ortadoğu Anadolu and Akdeniz, and prints no numbers. The identity
  studies (`Kürtler ve Kürt Sorunu` 2008, `Kürt Meselesini Yeniden Düşünmek` 2010, `Kürt
  Meselesinde Algı ve Beklentiler` 2011, `Siyasal Kimlikler` 2010, `Vatandaşlık Araştırması`
  2016) were all opened and scanned: not one page puts an Alevi figure beside a region name.
- **World Values Survey** — wave 7's country-specific denomination list for **Turkey 2018 is
  `0 no denomination / 2200 Orthodox / 4011 Islam / 9999 Other`**. No Alevi, no Sunni/Shia
  split. Read at source in the WVS variables report. Dead end, and worth recording because it
  looks like a live option from the outside.
- **European Social Survey** — Türkiye is in rounds 2 (2004) and 4 (2008) only, and `rlgdnm`
  is a coarse world-religions list.
- **Peter Alford Andrews and Rüdiger Benninghaus, *Ethnic Groups in the Republic of Turkey***
  (Reichert 1989; vol. 2 with a village index, 2002) — the academic compilation under most of
  this field, Kurdish and Alevi villages specifically in vol. 2. A printed book from a
  commercial academic press, 1989, and no magnitude. Recorded so nobody has to find it twice.
- **Nişanyan's *Index Anatolicus*** — has the geography and is **out on Anita's call**
  (§11ac): its terms of service prohibit systematic retrieval outright, and its sect coverage
  runs inverse to the variable anyway (87.6% of Şanlıurfa's settlements labelled against 0.2%
  of Kastamonu's).

The national figure is stated in the country's `note_public` without a geography, which is
§14 rule 1 applied rather than worked around.

## 4. The other hole: 0.4% is one cell holding three different things

Q10's `Diğer` is described by the report in its own words as *belongs to a religion other than
Islam or belongs to no religion*. One cell, about **341,000 people**, holding Türkiye's
Christians, its Jews and its irreligious together, published nationally and nowhere finer. It
is drawn as **`other.tr`**, whose label is *Other religion or none (Türkiye)* — the only node
in that family whose name admits it is not purely a religion cell.

It carries no geography. Almost all of those people are in Istanbul in reality; here they sit
at 0.4% in all twelve regions. The alternative was not drawing them, which would have rendered
Türkiye as a country with no non-Muslims at all.

## 5. Access notes, and two hosts that do not work

- **The report.** `diyanet.gov.tr` no longer serves it. The 293-page file used here is
  `ceidizleme.org/ekutuphaneresim/dosya/914_1.pdf` — complete, valid `%%EOF`, and
  `sources/tr.py --fetch` checks the trailer rather than the Content-Length
  ([[reference_pdf_truncated_at_source]]). **The Ankara University open-courseware copy that
  also turns up in search is a 23-page summary deck and does NOT contain Table 4**; `tr.py`
  asserts `page_count == 293` for exactly that reason. A state-hosted copy is worth finding
  before citing this as `source` long-term.
- **TÜİK's data portal is unusable for tables.** `data.tuik.gov.tr` and
  `veriportali.tuik.gov.tr` both serve a single-page-app shell on every content path and real
  404s under `/api/`, so the router exists and its paths were not found
  ([[reference_spa_hidden_apis]] applied, no answer). `nip.tuik.gov.tr` works but drills only
  to province. **OCHA COD-PS is the reachable form of the same ADNKS numbers** and is what the
  magnitude uses.
- **Eurostat's dissemination API failed on TLS** for this session (`curl` exit 60), and the
  GISCO static GeoJSON on the same domain was fine. If NUTS population is ever wanted, retry
  the API rather than assuming it is blocked.
- The TÜİK hosts also returned an SSL connect error for several minutes mid-session and then
  recovered. Retry before concluding anything about them.

## 6. Checks

- **Table 4 rows sum to 100** in all thirteen, within the report's own printed rounding note.
- **Two anchors asserted by value** — national Hanefi 77.5 and TRB Şafi 48.7 — so a silent
  column shift fails the build instead of drawing a wrong map.
- **The İBBS-1 crosswalk is asserted exhaustive both ways** against COD-PS's 81 provinces, and
  for a duplicate. A partial map cannot pass ([[reference_name_join_wrong_neighbour]]).
- **The boundary join is the identity function** — GISCO's TR1..TRC are the Diyanet's own row
  labels, names included, and `tr_geo.py` asserts both sets and every name.
- **Region-weighted shares against the report's own national row**: Hanefi 76.72 vs 77.50,
  Şafi 12.15 vs 11.10, Caferi 1.00 vs 1.00, Hiçbiri 6.08 vs 6.30. They are separately
  weighted, so this is a relationship rather than an identity (§9i), and it holds.
- **Kontur against the register**: 0.997x nationally, per-region 0.82x (TR9) to 1.15x (TR1),
  median 0.96, **0 of 12 outside a factor of two — against 7 of 12 when the region labels are
  shuffled.** Twelve units make that weak evidence rather than strong, and it is reported as
  such.

## 7. Open

- **An Alevi layer at any geography**, which is the one thing that would change this country.
  KONDA holds the data and publishes only the national number; its Barometer records mezhep as
  a standard demographic every month. That is a question for a person, not a scraper.
- **TÜİK's microdata service** would give İBBS-2 (26 regions) and the `Diğer (belirtiniz)`
  write-ins, which is where Nusayri and any Alevi write-in would be recoverable. It is an
  application, and [[feedback_gated_data_last_resort]] applies.
- **A newer wave.** Nothing found for 2023 or 2024; the Diyanet's Sayıştay audit reports
  mention no successor survey.
