# cityhistory — gaps, misplacements and false abandonments in the sparse eras

Findings from a manual pass over **pre-AD 1**, then over the **early Americas and Africa**
(Parts 1-3, checked against the build of 2026-08-21 16:48 - the one with `DISAPPEARED`,
`FADE_GAP_REAL = None` and provenance codes), then **Part 4**, seven cases raised from the map on
2026-08-22 and checked against that day's build. `spec.md` says how the pipeline works; this file says what is currently
wrong in the data it produces. Items are grouped by which existing table fixes them.

Every claim below was verified against the built `data/cities.json`, not assumed. Where the
history is contested I say so rather than asserting a date.

---

## Status

Applied on 2026-08-21, in `build.py` unless noted. Everything else in this file is still open.

| | what | where |
|---|---|---|
| ✅ | Sparta, Pyongyang and Benin City had their **entire** pre-modern record in a parenthetical entry | `MERGE_INTO` + a donor exemption from the marker drops |
| ✅ | Pyongyang's Bronze Age figures are not supportable; the Lelang-era city is | `CLIP_BEFORE` −194 |
| ✅ | Carthage drawn at 100–175k straight through 146–29 BC | `DISAPPEARED` + `DROP_YEARS` |
| ✅ | Seven African cities blanked by false abandonment (Gao, Timbuktu, Djenné, Mbanza Kongo, Dongola, Sinnar, Shaki) | `DISAPPEARED` `[]` |
| ✅ | **The strip deletes real benchmarks** — Gao lost 3, Timbuktu 1, Sparta 2 | new **validate check G**; `CF_KEEP` / forward `CF_END` |
| ⚠️ | ~~Baghdad lost 2~~ **wrong — see Part 4.1.** Check G never reported that run; the `CF_KEEP` entry written for it was withdrawn on 2026-08-22 | `CENSUS` + `DROP_YEARS` |
| ✅ | Anyi in Jiangxi, Caracol on Chichén Itzá, Djenné in Senegal | new `SITE_COORDS` table |
| ✅ | 11 ancient capitals filed under the modern town | `RENAME` |
| ✅ | **Cairo had no medieval history** — plain entry started 1859 | `MERGE_INTO` |
| ✅ | Tula drawn 800 years early and absent during the Toltec period | `CENSUS` (recovered figure) + `RENAME` + `DISAPPEARED` |
| ✅ | Cahokia in `chandlerV2.csv`, absent from the build | new `SYNTHETIC` + `NA_CLIP_KEEP` |
| ✅ | Yiyang: 200k held from −300 to 1982 on a county founded in 221 BC | `CLIP_BEFORE` 1983 |
| ✅ | Trujillo carrying **Chan Chan**'s 1400 figure 134 years before Trujillo was founded | `SYNTHETIC` + `CLIP_BEFORE` |
| ✅ | **Mayapán** had no entry; its Chandler row was geocoded onto Mérida | `SYNTHETIC` + `DROP_YEARS` |
| ✅ | **29 pre-1800 cities drawn in a single frame** and invisible in every other | `bracket_lone_anchor()` |
| ✅ | **Dali** drawn as a 517-year slide through the Dali Kingdom, up to 65% low | `CF_KEEP` + `RENAME` |
| ✅ | **Lhasa** blanked 840–1830, erasing the whole Ganden Phodrang era | `CENSUS` (recovered) + `DISAPPEARED` |
| ✅ | **Thanjavur** blanked 975–1807 across 800 years of court cities | `CENSUS` (recovered) + `DISAPPEARED` `[]` |
| ✅ | **The New World cap was deleting the only two BC benchmarks in the Americas** | `chandler_benchmark()` exemption in `nw_cap` |
| ✅ | **Teotihuacán lost its AD 500 peak** — stadester overwrote Chandler's 125,000 | `CENSUS` (recovered) |
| ✅ | **El Tajín drawn in Myanmar** — longitude sign flip, never repaired | `SITE_COORDS` |
| ✅ | **Eight pre-Columbian cities absent from both sources** (Cuicuilco, El Mirador, Kaminaljuyú, Moche, Wari, Calakmul, Copán, Chichén Itzá) | new `ARCHAEOLOGICAL` tier |
| ✅ | **Monte Albán started 1,300 years late** — Chandler has only its AD 800 row | `CENSUS` (Blanton) |
| ✅ | Richmond VA and Hamilton ON also drawn in China — same sign flip as El Tajín | `DROP_KEYS` + `SITE_COORDS` |
| ✅ | **Caracol, Tikal and Tiahuanaco drawn at peak size 4–6 centuries early** — the last unrevised Modelski stamps | `CENSUS` (re-dated) |
| ❌ | ~170 further check-G runs, 63 with a cliff | worklist |
| ❌ | 639 modern single-point entries (city districts, English boroughs) | needs dedup, not bracketing |

**Correction to an earlier claim in this file: Mérida is NOT a conflation.** Chandler files it as
`Merida / Tiho`, one continuous city, because the Spanish built Mérida in 1542 directly on top of
the Maya T'hó using its stones — so its `AD_900 = 40,000` is the same site and the entry is right
as it stands. The only contamination was a stray `1194` row belonging to Chandler's *Mayapán*
benchmark, which Chandler geocoded onto Mérida; that value (25,000) is identical to Chandler's own
Mérida figures at 1500 and 1528, so removing it changes no pixel. Mayapán itself was the real gap.

### What changed on the map

- **1575 sub-Saharan Africa: 9 cities → 14**, and Gao is now correctly the largest at 75,000.
- ~~**Baghdad reads 1.1M in the year 1000 instead of 760,000** — the strip had deleted its 1000 and
  1100 benchmarks, so the Abbasid peak was drawn sliding from 932.~~
  **This claim was wrong and the change has been reversed.** Chandler states 125,000 at 1000 and
  150,000 at 1100, not 1.1M; the plateau the exemption protected was populstat's carry-forward.
  Baghdad now reads **125,000 in the year 1000**, which is Chandler's own figure. Part 4.1.
- **Cairo exists from 700 instead of 1859** — including its 1348 peak of 500,000 and the drop to
  350,000 in 1350, which is the Black Death visible in the data. It is now correctly the largest
  city in the world at 1348.
- **North America has a city before 1700 for the first time**: Cahokia, 1100: 40,000.
- **Tula** is 900: 50,000 collapsing at 1180, not 100,000 held from 300 BC.
- Classical **Sparta** (−430: 40,000) and Roman Sparta (AD 200: 30,000) exist at all.
- **Chan Chan** and **Mayapán** exist, and 27 other lone-benchmark cities became visible for the
  first time — Miletus, Troy, Ani, Khajuraho, Prambanan, Istakhr, Loango among them.
- **Carthage** is a trough from 146 to 29 BC instead of a plateau.

### The one that turned out to be general

Check G reports **170** runs where the carry-forward strip deletes real Chandler benchmarks, 63 of
them followed by a ≥2x cliff. That is a defect class the project did not know it had, and it is
not regional — Baghdad, Kyiv, Kalyan, Quilon, Nara and Veliko Tarnovo are all in it. Five were
verified and fixed (Baghdad, Gao, Timbuktu, Sparta, Dali); the rest is a worklist, and `spec.md`
§3.4 records the table shapes that fix them.

**The list needs per-city judgement, not a batch fix**, because one signature covers three
situations and only the first is a defect:

1. the source asserts a real plateau → `CF_KEEP` (Baghdad, Dali)
2. the source is wrong and the strip accidentally saves us → **leave it** (Nara at 200,000 in
   1400 is not credible; Rajmahal at 100,000 in 1800 is not either)
3. the city did die, just not on that date → `DISAPPEARED` / `CF_END` (Patna's blank should end
   at Sher Shah's 1541 refounding, not 1608)

Two failure modes, and **erasure is worse than understatement**: 45 of the cases leave the city at
the fade floor — invisible — while the source asserts ≥20k. Lhasa and Thanjavur, the two worst,
are now fixed; the rest of that set is where to continue.

Both were instructive about *which* end of the run matters. Neither turned out to be a plateau
worth keeping: Chandler goes silent on Lhasa for nine centuries after 800, and has no pre-1750
benchmark for Thanjavur at all, so in both cases the long flat run really was fabrication and the
strip was right to delete it. What was lost sat at the **far** end — Stadestér had overwritten
Chandler's genuine 1700–1840 Lhasa figures and 1750/1800 Thanjavur figures with the fabricated
value, so deleting the run deleted those too, and each city vanished for centuries it was
demonstrably a capital. The fix is `CENSUS` restoring the compiler's own numbers, then
`DISAPPEARED` to date what remains.

**Guangzhou is the cautionary one.** It looked like the worst case in the list — 73% low at year
1000, and it would have been 4th in the world if restored — but Chandler's row starts at
`AD_1000 = 40,000` and has no earlier benchmark, so the 200,000 was populstat contradicting our
own benchmark source. It is now excluded by `GBM_VALUE_TOL`. Check the Chandler row before acting
on any entry in this list.

---

# Part 1 — before AD 1

## 1.1 Ancient history deleted by the variant drop

Two cities whose **entire pre-modern record** sits in a `(agglomeration)` entry, with the plain
entry starting in the 19th century. Same trap as Athens/Gazi and Copenhagen/Frederiksberg,
and the same fix — `MERGE_INTO`, whose outside-only rule is exactly right here.

| source (dropped) | span | plain entry drawn | ancient values |
|---|---|---|---|
| `Sparti (agglomeration)-Greece` | −430..1991 | `Spartí` starts **1861: 2,000** | −430: 40,000 · −200: 30,000 · 0: 30,000 · 100: 30,000 |
| `P'yõngyang (agglomeration)-North Korea` | −1000..2002 | `P'yõngyang` starts **1890: 40,000** | −1000: 25,000 · −800: 25,000 · −650: 30,000 · 0: 42,858 |

Sparta is the more clear-cut of the two: Chandler lists it at 40,000 in 430 BC and 30,000 in
200 BC, and a classical-antiquity map without Sparta is conspicuous.

Pyongyang is Chandler's Gojoseon capital claim and is genuinely contested — the location of
Wanggeom-seong is disputed and 25,000 in 1000 BC is on nobody's consensus. But it is currently
being dropped for a *mechanical* reason (a parenthesis in the name), not a judged one. If it
should be out, it should be out on purpose.

**Why the existing splice can't rescue them.** `prefer_agglomeration` returns early on
`if not V or not S or not G` — no WUP centre, no splice. Sparta's modern self is 19,100, below
WUP's 50k floor, so `G` is empty and the variant is dropped unconditionally. The splice is
priced *entirely* as a modern-tail problem (the switch step it saves vs the break it opens), so
it structurally cannot value a variant whose worth is 2,300 years earlier. Worth a line in
§3.3 — this is a real limit of the cost model, not a tuning issue.

A sweep of all 144 variant entries that start >50 y before their base found only six pre-1700,
so this is a closed set, not a class needing a general rule:

- `P'yõngyang` (−1000 vs 1890) and `Sparti` (−430 vs none) — above
- `Iskandarîyah, Al-` (−300 vs 1090) — **already spliced correctly**, Alexandria has its antiquity
- `Qâhirah, Al-` (700 vs 1859) — **not spliced: Cairo is missing its entire medieval history.**
  Out of scope for this pass but much the largest item on the list; Cairo was among the largest
  cities on earth for most of 1000–1500 and the map starts it at 1859.
- `Kazan'` (932 vs 1000), `Benin City` (1600 vs 1901) — see §3.4 for Benin

## 1.2 Drawn straight through a documented destruction

`DISAPPEARED` has already fixed **Corinth** (−146, −44) and **Nineveh** (−600, 640). The
matching case is still open:

**Carthage** — `-300:500,000  -200:173,000  -100:100,000  0:175,000  100:237,000`

Razed by Scipio in 146 BC. The Gracchan colony of 122 BC failed; Caesar refounded the site in
44 BC and Augustus settled it in 29 BC. Between those dates it was not a city. The map draws it
at 100,000–175,000 continuously, and the 100,000 at −100 is *the same mis-filed AD 100 benchmark*
`CHANDLER_AD100` documents but deliberately leaves alone because the entry holds a real anchor
at 0. So the wrong figure is anchoring the era.

Suggested: `"Al Marsâ-Tunisia": [(-146, -35)]`. The −100 value should probably also go into
`DROP_YEARS` — the §3.1 note already says "deleting it instead is a separate call and Carthage
in particular wants one".

Two more of the same family, both lower confidence:

- **Istanbul** `-100:36,000 → 300:327,000` — one straight line across 400 years. Byzantium was a
  middling town until Constantine refounded it in 330; the growth was abrupt, not a ramp. This
  doesn't want a fade (the city existed), it wants an anchor near AD 300 so the climb starts
  where it belongs. Chandler's row is misspelled `Instanbul-Turkey`, which is why `CHANDLER_AD100`
  skips it.
- **Kaifeng** `-500:100,000  -400:200,000  -250:1,000`. `CF_END` already declares Kaifeng finished
  at −225 (Qin drowned Daliang). But Daliang only *became* Wei's capital in 361 BC, so 100,000
  at 500 BC and 200,000 at 400 BC both predate the city's importance. The right shape is a short
  spike 361–225 BC, not a 250-year plateau.

## 1.3 Coordinates in the wrong place

Both are geocoder fallbacks onto a same-named modern Chinese county, and both put a
six-figure Warring States city in a region that had no such thing.

**`Anyi`** — drawn at **(28.839, 115.599)**, which is Anyi County, Jiangxi, near Nanchang.
Ancient Anyi 安邑 was the capital of the state of Wei for 223 years (562–339 BC) and is at
**Xia County, Shanxi ≈ (35.14, 111.22)**, ~800 km north-west. Chandler's 100,000 at 500/400/300 BC
is defensible for the Wei capital and impossible for the Gan valley, which was Chu/Yue frontier.
([location](https://baike.baidu.com/en/item/Anyi/31625))

**`Yiyang`** — drawn at **(28.554, 112.355)** = Yiyang, Hunan, at 200,000 in 300 BC. Yiyang county
was founded in **221 BC**, after Qin conquered Chu — it did not exist at the date it is drawn,
let alone at that size. ([founding date](https://en.wikipedia.org/wiki/Yiyang)) The intended city
is most likely **Yiyang 宜陽 in Henan ≈ (34.51, 112.17)**, the Han state's great iron-working
stronghold that Qin besieged in 307 BC — but that identification is my inference from the date
and size, not something the source states. Worth confirming before moving it; the safe minimum is
that the current placement is wrong.

Neither is in `coord_fixes.json` because `propose_coords.py` can only judge modern plausibility,
and both coordinates are perfectly good *modern* towns.

## 1.4 Filed under the modern town — `RENAME` candidates

`RENAME` already covers Memphis, Babylon, Thebes, Nineveh, Carthage, Ephesus, Pergamon, Knossos
and others. These carry an ancient city's series under a modern name and are not yet in it.
Each was confirmed by matching the built series against the `chandlerV2.csv` row at those
coordinates, so the identification is the source's, not mine:

| drawn as | is | evidence |
|---|---|---|
| `Bahtîm` (30.14, 31.27) | **Heliopolis (On)** | Chandler `Heliopolis / Egypt (30.1295, 31.2889)` BC_1360 = 30,000 |
| `Zefat` (32.96, 35.50) | **Hazor** | Chandler `Hazor / Israel (33.017, 35.569)` BC_1600 = BC_1360 = 24,000 |
| `Tûlkarm` (32.31, 35.10) | **Samaria** | Chandler `Samaria / Palestine (32.276, 35.190)` BC_800 = 27,000 |
| `Basyûn` (30.94, 30.82) | **Sais** | Chandler `Sais / Egypt (30.965, 30.769)` BC_650 = 48,000 — the 26th-dynasty capital |
| `Amaliada` (37.80, 21.35) | **Elis** | Chandler `Elis / Greece (37.892, 21.373)` BC_430 = 30,000 |
| `Marv Dasht` (29.88, 52.81) | **Persepolis** | Chandler `Perspolis / Iran` BC_430 = 50,000 |
| `Zhengzhou` (34.75, 113.62) | **Zhengzhou (Ao)** | Chandler `Ao / China (34.767, 113.65)` BC_1360 = 32,000 — the Shang capital |
| `Mallawî` (27.74, 30.84) | **Amarna (Akhetaten)** | Chandler `Amarna / Egypt (27.645, 30.896)` BC_1360 = 30,000 |

Zefat/Hazor and Tûlkarm/Samaria are the two that actively mislead: as drawn, the map asserts a
24,000-person Bronze Age Safed and a 27,000-person Iron Age Tulkarm, neither of which was ever
a city of any size. Bahtîm is a Cairo suburb standing in for On.

## 1.5 Dead cities: two different endings, chosen by accident

35 pre-AD 1 cities end their series with no fade — the bubble simply blinks out. Ten do it at
≥30,000:

```
60,000 at -1200  Sapinuwa      40,000 at -1200  Isin
53,700 at -1200  Hattusa       40,000 at -1800  Larsa
50,000 at  -650  Miletus       40,000 at -1800  Umma
40,000 at  -650  Nimrud        40,000 at -2000  Girsu
30,000 at -1300  Uruk          30,000 at -2400  Lagash
```

Which ending a city gets is not a judgement about the city — it is whether the source happens to
have a later anchor for the fade to ramp toward. Anyang, Pi-Ramesses and Knossos fade because
their entries continue; Hattusa and Girsu blink because theirs stop.

For most of these the blink is *right* (the Hittite collapse really was abrupt; Nimrud fell in
612 BC with Nineveh). The clear exception is **Miletus**, which has a single point at −650 and
then nothing, ever. Miletus was the greatest Ionian city, founded dozens of Black Sea colonies,
was destroyed by Persia in 494 and rebuilt on Hippodamus's grid as a major Hellenistic and Roman
port. One 50,000 point at 650 BC followed by permanent absence is the worst single-city
misrepresentation in the pre-AD 1 layer after Carthage.

This is a per-city list, not a rule — I'd suggest handling only Miletus now and leaving the rest.

## 1.6 Genuine holes — nothing in any source

Checked against the Chandler benchmark columns and the Stadester corpus, so these are absences
of *data*, not pipeline losses.

**Nubia and the Sudan are completely empty for all of antiquity.** No Kerma, no Napata, no Meroë.
Kerma was the earliest urbanised state in sub-Saharan Africa and reached at least 10,000 by
1700 BC; Meroë was the capital of a kingdom of ~1.15 million from c. 590 BC to AD 350 and is
usually put at 20,000–25,000. Neither is in Chandler or Stadester. On a map that shows 10,000-person
Sumerian towns this is the largest regional blank in the pre-AD 1 world.
([Kerma](https://en.wikipedia.org/wiki/Kerma), [Meroë](https://www.worldhistory.org/Meroe/))

**In `chandlerV2.csv` but lost in the Stadester fusion:**

- **Mycenae** — BC_1360 = 30,000. (Chandler's figure is very high for Mycenae; the citadel and
  town are usually put at 6,000–10,000. But its absence from a Bronze Age Aegean map is odd.)
- **Vaishali** — BC_430 = 45,000. Would be the second Indian city at that date beside Rajagriha.

**Not in any shipped source, but standard in the literature:**

- **Hao / Fenghao**, the Western Zhou capital near Xi'an, 1046–771 BC. Some editions of Chandler
  put it first in the world at 1000 BC (~125,000); `chandlerV2.csv`'s BC_1000 column has only three
  cities and Hao is not among them. As drawn, the Zhou heartland is empty and Luoyang (the
  *secondary* capital) stands alone in China at 50,000.
- **Ma'rib** is present but fades at −500. The Sabaean capital and its dam ran until the 6th
  century AD; the map has Arabia empty from 500 BC onward. The city's own size is genuinely
  uncertain, so this is a "the fade is too early" note rather than a number.

**Correctly absent, checked and no action needed:** the Indus is *not* running too long —
Mohenjo-daro fades at −1950, Harappa ends −2000, Dholavira and Rakhigarhi −2000, all consistent
with the Mature Harappan ending c. 1900 BC. If anything the peak sizes are conservative
(Mohenjo-daro at 40,000 against a 30–60k literature range) and Rakhigarhi, the largest
Harappan site by area, is drawn at 10,000.

## 1.7 Data points that could be added

The `CENSUS` bar is four conditions and antiquity almost never meets #1 ("a named census, not a
recollection or a round number"). Two cases do:

- **Chang'an, AD 2 — 246,200 in 80,800 households.** This is the *Hanshu* census, an actual
  enumeration of the city, not an estimate. It lands just outside this pass but fixes the AD-1
  frame, where Xi'an is currently drawn at 419,603 — a straight-line fill, not a datum.
- **Luoyang, AD 140 — c. 195,000**, from the *Hou Hanshu* household registers.

Everything else in antiquity (Rome's grain dole, Diodorus on Alexandria, Josephus) is a
recollection or a derived estimate and fails condition 1. I would not add them.

---

# Part 2 — the early Americas

## 2.1 Quito and Riobamba at 1300 — real, but only as Chandler's assertion

Both come straight from `chandlerV2.csv`, both at certainty 1:

```
Quito     AD_1300=30000  AD_1400=20000  AD_1500=30000  AD_1538=23000  AD_1574=15000
Riobamba  AD_1300=30000  AD_1400=50000  AD_1450=60000  AD_1487=60000  AD_1740=18000
```

So this is not a pipeline artifact — it is Chandler's number, faithfully carried. Two caveats:

1. **Certainty 1 means "the modern city was confidently identified", not "the figure is
   reliable"** — the field is 1165/222/210 across the file and the 1s are overwhelmingly ordinary
   modern cities. Every archaeological entry (Ao, Avaris, Mayapan, Tenayuca, Great Zimbabwe) is 2
   or 3. So the 1 on Quito is telling you about the geocode, not the estimate.
2. A 30,000-person Quito in 1300 has no archaeological support I can find. The Quitu-Cara polity
   is thinly attested, the Inca only took the region in the 1470s, and Riobamba (Liribamba, the
   Puruhá centre) at 60,000 in 1450 would make it larger than Cusco at the same date — which the
   same file puts at 24,000 in 1438.

They are internally inconsistent with Chandler's own Andean numbers, and they are currently the
**only** two cities in South America at 1300. I would not delete them, but they should not be
carrying a continent.

## 2.2 Mesoamerica at 900 — one city, and it is the wrong one

At 900 the whole of Mesoamerica is **1 city drawn, 4 blanked**:

```
drawn:    Mérida 40,000
blanked:  Teotihuacán (103k) · Tula de Allende (100k) · Tapachula (35k) · Xoxocotlán (30k)
```

Three of those four blanks are correct — Teotihuacan was abandoned c. 750, Monte Albán
(= `Santa Cruz Xoxocotlán`) c. 900, Izapa (= `Tapachula`) long before. The problems are the
other two entries.

**`Tula de Allende` is the worst New World defect in the dataset.** Its source series is
`-300:100,000 · 0:100,000 · 100:100,000 · 200:100,000`, and it is then blanked from 175 to 1994.
Toltec Tula (Tollan-Xicocotitlan) flourished **c. AD 900–1150**. So the city is drawn at 100,000
for five centuries when the site was a village, and is invisible for the entire period it was
actually the capital of central Mexico. Worse, `chandlerV2.csv` has the right answer and it never
arrives: `Tollan / Tula / Mexico (20.0637, −99.3410) AD_900 = 50,000` — the same coordinates as the
drawn entry, 50,000, at 900. The Chandler row is lost and a bad populstat series is drawn instead.

Fixing Tula alone changes the 900 frame from one city to two, and puts the right one on the map.

**`Mérida` is carrying Mayapán's series.** Chandler has
`Mayapan / Mexico (20.9785, −89.5934) AD_1194 = 25,000, AD_1441 = 25,000` — and the drawn Mérida
sits at (20.97, −89.59), i.e. *exactly* Chandler's Mayapan coordinates, with `1194:25,000`.
(Chandler's own coordinate for Mayapán is wrong — the site is at 20.63, −89.46 — but the series is
Mayapán's.) The `900:40,000` on the same entry is a different city again, T'hó/Dzibilchaltún
territory. So the one city Mesoamerica has at 900 is a conflation of two or three.

**Chichén Itzá is absent**, which is what should be filling the 900–1050 frames.

## 2.3 `Caracol` is drawn at Chichén Itzá

`Caracol` is drawn at **(20.679, −88.571)**. The real Caracol is in Belize at (16.76, −89.12),
440 km away. (20.679, −88.571) is **Chichén Itzá**, 0.1 km off.

The *series* looks like Caracol's — `500:120,000 · 800:100,000` matches Caracol's Late Classic
peak, which is genuinely estimated above 100,000. So this is a coordinate error, not a naming
one: Caracol's population is being drawn on Chichén Itzá's site. It also means the map appears to
show Chichén Itzá while actually showing it 400 years too early and then blanking it at 800,
just before its real floruit.

## 2.4 Cahokia — in the shipped source, absent from the map

```
Cahokia / Missouri / United States of America  (38.57088, -90.19011)  certainty=1
    AD_1100 = 40,000   AD_1400 = 4,000
```

It is in `data/chandlerV2.csv` and it is in nothing else — not Stadester, not the build.
`Saint Louis` starts at 1840. Chandler's coordinate is downtown St. Louis rather than the mounds
(38.66, −90.06), which is probably why the fusion lost it, but the row exists.

Consequence: **North America has 0 cities drawn and 0 blanked at every year before the colonial
period.** The continent is not sparse on the map, it is empty, and the one pre-Columbian city
large enough to qualify is sitting in a CSV the project already ships.

## 2.5 Other Americas items

- **Chan Chan is drawn as `Trujillo`.** `Trujillo-Peru` runs from 1400 at 25,000; the Spanish
  city was founded in 1534. Chan Chan is 5 km away. So the pre-conquest figure is Chan Chan's —
  at its ~1300 peak the largest city in South America and the largest adobe city anywhere,
  usually put at 30,000–60,000. A `RENAME` would make it findable; the single 1400 point
  understates a 900–1470 occupation.
- **`Tlalnepantla de Baz` is Tenayuca** — Chandler `Tenayuca / Mexico (19.5322, −99.1685)
  AD_1200 = 50,000, AD_1250 = 54,000, AD_1565 = 4,100`, matching the drawn series exactly.
  Tenayuca is inside modern Tlalnepantla. `RENAME`.
- **`Pachacamac` is blanked 875–1955** at 2,000. It was a major pilgrimage centre from c. AD 200
  to the conquest. The 2,000 is Chandler's `AD_800 = 2000` further reduced by the New World cap,
  so this one is genuinely thin data rather than a lost figure — but the blank asserts absence
  where there was continuous occupation.
- **`Chosica` 1400:40,000 → blank 1424** — Chosica is a modern Lima suburb in the Rímac valley.
  40,000 in 1400 there is almost certainly Cajamarquilla or another valley site. Worth a look.
- **Wari** (c. AD 600–1000, 30,000–70,000) is absent entirely; `Ayacucho` starts 1574.

## 2.6 The New World cap makes four cities move in lockstep

`nw_cap()` clips every American city to the same ceiling, so at −200 Caracol, Tula, Tiahuanaco
and Tikal are all drawn at **exactly 800**, and at year 0 all four at **exactly 43,100**. Four
cities from Yucatán to Bolivia, identical to the digit, rising together.

The cap is doing its job — the underlying Modelski figures really are stamped back into deep
antiquity and really should be suppressed. But the *visual* result is a synchronised pulse that
reads as data. Since the values are below the display floor for much of the ramp this may not be
visible in practice; worth a look at the −200..100 frames before deciding it matters. (Related to
§6.10, though the mechanism is the cap rather than a copied source curve.)

Also worth noting: **Tiahuanaco is drawn at 44,700 in AD 800 against Chandler's own
`AD_800 = 20,000`**, and at 100,000 in AD 100 when Tiwanaku's urban phase is c. AD 500–1000. The
larger, earlier numbers are Modelski's, and the cap limits them without correcting their dates.

---

## 2.7 The pre-500 audit (2026-08-22) — three fixes, and a source hole that no fix reaches

Re-checked §2.1–2.6 against the current build and then went further back, because the map was
**empty in the Americas at every year before AD 1** — not sparse, empty. The §2.1–2.5 items had
all landed (Tula, Cahokia, Chan Chan, Mayapán, Tenayuca, Caracol's coordinate), so the emptiness
was something else. It turned out to be three separate defects plus one thing that is not a
defect at all.

### The corpus is nearly empty here, and that is the real finding

The whole of the pre-AD-600 Americas, in both sources the repo ships:

* **`chandlerV2.csv` has THREE rows with any value before AD 600.** Izapa (−200 = 35,000),
  Teotihuacan (−100 = 45,000 · 361 = 90,000 · 500 = 125,000) and Tres Zapotes (−200 = 30,000).
  That is Chandler's entire pre-Classic New World.
* **Stadestér has SEVEN entries with any pre-600 value**, and four of them are a single round
  number stamped across every century: Tikal, Caracol and Tiahuanaco at a flat 100,000 from
  900/500 BC, Tula at a flat 100,000 from 300 BC. Those are Modelski's placeholders and they are
  what `nw_cap` exists to suppress.

So after the fixes below the Americas at 200 BC is two cities, and at AD 100–500 it is four.
That is not a pipeline failure any more — it is the honest extent of the evidence in the
corpus, and getting past it needs **new data**, not a better rule (see §2.8).

### 2.7a The cap was eating the only real benchmarks it had

`nw_cap(NW_RAMP_START)` is `PEAK_FLOOR` — 2,000 — by construction, since the ramp is defined to
start at the floor. And 200 BC is exactly where Chandler's only two BC benchmarks for the
Americas sit. So the ramp was flattening Izapa's 35,000 and Tres Zapotes' 30,000 to 2,000,
i.e. below the display floor, i.e. invisible. **100% of the pre-AD-1 New World evidence in the
corpus was being deleted by the rule written to protect it from Modelski.**

Fixed with `chandler_benchmark()`, a verbatim value-match against `chandlerV2.csv` through
provenance.py's join, exempting a point from the cap when Chandler actually asserts it.

The test has to be a **value** match, not `provenance.classify()`'s label: classify() calls
Tikal's 900 BC row `('chandler', 'default')` too — that is the entry's *type* talking, not a
benchmark — and Chandler has no Tikal row before AD 751. Measured: **2 points exempted, 5 still
capped**, and all 5 of those are Modelski year-0 stamps that `YEAR_ZERO` deletes anyway. Nothing
the ramp was suppressing came back; everything before `NW_RAMP_START` is still dropped outright.

A side effect worth knowing: Izapa's whole flat 35,000 run now begins at its benchmark instead
of at year 100, so `strip_carry_forward` keeps the −200 point and drops the rest. Izapa moved
from "35,000 at AD 100, dead at 175" to "35,000 at 200 BC, bracketed out by 50 BC". That is
better — Chandler's benchmark *is* 200 BC and Izapa's floruit is c. 300 BC–AD 100, so the old
drawing was three centuries late — but it is a bracket, not a claim about when Izapa ended.

### 2.7b Teotihuacán had lost its peak — the Lhasa shape, in the New World

`chandlerV2` reads `AD 100 = 45,000 · 361 = 90,000 · 500 = 125,000 · 622 = 60,000`: a rise to a
Classic maximum, then the collapse. Stadestér holds `400 = 103,168 · 500 = 103,168 ·
600 = 103,168 · 622 = 60,000` — an interpolated 400 value carried forward over the two benchmark
years after it. **Chandler's 125,000 is simply gone.** provenance.py sees it directly: 361 is
`('chandler','exact')`, 400 is `('populstat','default')`, 500 falls through to `('fill','fill')`.

This is exactly the §3.4 / check-G shape that Lhasa and Thanjavur turned out to be — the damage
is at the *far* end of a run, where the compiler's own later benchmarks were overwritten — and
it is the worst-drawn New World city after the Tula case, because it is a **ranking** error:

* the map drew Teotihuacan **peaking at AD 400 and declining**, when 400 is the middle of its
  growth and the peak is a century later;
* it drew it **below Caracol at every year of the Classic** — and Caracol's 120,000 is
  Modelski's, with no Chandler row behind it at all (§2.3, §2.6).

Fixed with `CENSUS["Teotihuacán-Mexico"] = (400, 621, {500: 125000})`, which clears stadester's
three fabricated points and leaves Chandler's own four. **Teotihuacán is now 6th in the world at
AD 500 and the largest city in the Americas**, which is what every account of the period says.

### 2.7c El Tajín is drawn in Myanmar

`El Tajin-Mexico` is filed at longitude **+97.3782**. The site is at −97.3778. The minus sign is
missing and the entry is drawn in the hills east of Lashio, ~14,000 km away. This is noted in
passing in a build.py comment (as an argument for `country` over point-in-polygon) but was never
actually repaired, and nothing else could catch it: the digits are correct so no name-match
repair fires, and `country` says Mexico so `in_americas()` is right about it — **only the dot is
on the wrong continent.** build.py's own region-conflict counter cannot see it either, because it
tests `lon < -30 and not new_world`, which catches an Old World entry in the Americas but not an
American entry in Asia.

The cost was the whole Epiclassic Gulf coast: El Tajín is 40,000–50,000 at AD 622–1000, the
largest Mesoamerican entry the map has between Teotihuacan's collapse and Tula. Fixed in
`SITE_COORDS`.

**Two more entries have the identical sign flip and are still open** (both modern, so out of
scope for this pass, and both need a dedup decision rather than a coordinate):

| entry | drawn at | should be | note |
|---|---|---|---|
| `Richmond-United States of America` | (37.5333, **+77.4667**) — Taklamakan | (37.53, −77.47) | a *third* Richmond VA entry; `Richmond-Virginia` is already drawn correctly and reaches 1.37M. Probably `DROP_KEYS`, not a coord fix. |
| `Hamilton-Canada` | (43.2500, **+79.8661**) — Xinjiang | (43.25, −79.87) | single point, 1975 = 515,000. No other Hamilton ON entry exists, so this one needs moving. |

A cheap general check for the class: flag any entry whose `country` is in `AMERICAS` but whose
longitude is not in [−170, −30]. It returns exactly these three.

### What changed on the map

* **The Americas have cities before AD 1 for the first time**: Izapa 35,000 and Tres Zapotes
  30,000 at 200 BC.
* **Teotihuacán** peaks at AD 500 at 125,000 instead of AD 400 at 103,000, and is the largest
  city in the Americas rather than second to Caracol.
* **El Tajín** (40–50k, AD 622–1000) is in Veracruz instead of Myanmar, and is the second city of
  Mesoamerica at AD 700–1000.
* validate.py unchanged on every check (A 11, B 75, C 11, D 3, E 5, F 11, G 170, H 79).

---

## 2.8 What is still missing before AD 500, and where the data would come from

Everything below is **absent from both sources**, verified by name search across
`stadester_cities.json` and `chandlerV2.csv` and by a 25 km radius search of the built map. This
is the worklist if the pre-Classic Americas is ever to be more than four dots.

| site | floruit | usual estimate | in corpus? |
|---|---|---|---|
| Caral / Norte Chico, Peru | 2600–1800 BC | ~3,000 (site); ~20,000 (complex) | no |
| San Lorenzo Tenochtitlán (Olmec) | 1400–1000 BC | ~5,500 | no |
| La Venta (Olmec) | 900–400 BC | 3,000–8,000 | no |
| Chavín de Huántar | 900–200 BC | 2,000–3,000 | no |
| Cuicuilco | 800 BC–AD 100 | ~20,000 | **no** |
| Monte Albán | 500 BC–AD 800 | 15,000–25,000 at peak | only AD 800 |
| El Mirador | 300 BC–AD 150 | 10,000 core, basin contested | **no** |
| Kaminaljuyú | 400 BC–AD 900 | ~10,000+ | no |
| Moche (Huacas de Moche) | AD 100–800 | 10,000–15,000 | no |
| Cahuachi (Nazca) | AD 1–500 | ceremonial, low resident | no |
| Calakmul | AD 250–900 | ~50,000 | no |
| Copán | AD 400–800 | ~20,000 | no |
| Palenque, Uxmal, Cobá, Chichén Itzá | AD 600–1100 | 10,000–50,000 | no |
| Wari / Huari | AD 600–1000 | 10,000–40,000 | no |

**Two of these are structural, not cosmetic.** *Cuicuilco* is the reason Teotihuacan rose — it
was the rival centre in the Basin of Mexico until Xitle buried it c. AD 100 — so the map shows
the effect and not the cause. *Monte Albán* exists but starts at AD 800, its single Chandler row,
so 1,300 years of the Zapotec capital are missing and it appears only as it is being abandoned.

**Set expectations on the floor, though.** Caral, Chavín, San Lorenzo and La Venta are all at or
below `MINPOP = 5000` on any mainstream estimate. Even with perfect data the Preclassic Americas
is a handful of dots, not a filled continent — the honest map here is thin.

### Where the numbers could come from

1. **Modelski, *World Cities: −3000 to 2000* (2003)** — the other half of the corpus's own
   `chandler_modelski` type; the repo vendors only Chandler. Low expected yield: Modelski's New
   World figures are precisely the flat 100,000s `nw_cap` exists to suppress.
2. **Reba, Reitsma & Seto (2016), "Spatializing 6,000 years of global urbanization"**,
   *Scientific Data* 3:160034 — the cleaned, geocoded Chandler+Modelski release. Same coverage
   hole (same two compilations), but it is a good **coordinate** cross-check and would have
   caught El Tajín.
3. **The Social Reactors Project / settlement-scaling datasets** (Ortman, Cabaniss, Sturm,
   Bettencourt — Colorado / SFI) — open, tabular, per-phase area *and* population estimates for
   the Basin of Mexico and the Maya lowlands, derived from the Sanders/Parsons/Santley *Basin of
   Mexico* survey. **This is the best single fit**: it would supply Cuicuilco, Monte Albán's full
   range, and phase-dated Teotihuacan and Tula in one table.
4. **Site literature for the rest** — Canuto et al. 2018 (*Science*) for the PACUNAM LiDAR
   lowland Maya figures; Chase et al. for Caracol (which would also let us re-date its peak away
   from Modelski's); Shady Solís for Caral; Janusek/Kolata for Tiwanaku (which would fix the
   100,000-from-AD-100 problem in §2.6); Isbell & Schreiber for Wari.

**DECIDED 2026-08-22: the bar was deliberately loosened for this case.** `SYNTHETIC`'s rule 1
is "every figure is a verbatim row from a source already in the repo (chandlerV2.csv), not a
number of ours", and nothing above satisfies it. Rather than stretch that table, a second one was
opened alongside it — **`ARCHAEOLOGICAL`** (spec.md §3.1) — whose rule 1 is *a published estimate
from the archaeological literature, with the source named and the range given where the field
disagrees*. Rules 2 and 3 are unchanged. See §2.9 for what went in and what deliberately did not.

### Two open items from §2.6 that this pass did not touch

* **Caracol has no Chandler row at all.** Its `500 = 120,000 · 800 = 100,000` is Modelski's
  alone, and it is currently the largest city in the Americas for six straight centuries and
  top-12 in the world at AD 700 and 800. Its real peak is c. AD 650–700, not 500. It is the last
  big unsupported New World number on the map.
* **Tikal** is drawn at a flat 100,000 from AD 100 against its own only benchmarks
  (751 = 63,000, 800 = 40,000) — peak four centuries early, declining through its actual
  floruit. **Tiahuanaco** likewise: flat 100,000 from AD 100 when Tiwanaku's urban phase is
  c. AD 500–1000, and drawn 44,700 at AD 800 against Chandler's own `800 = 20,000`.

---

## 2.9 The archaeological tier, as built (2026-08-22)

§2.8 ended on a decision; this is what was decided and done. Eight cities were added through the
new `ARCHAEOLOGICAL` table, Monte Albán was extended through `CENSUS`, and the two remaining
sign-flipped coordinates were repaired.

### Why not just un-suppress Stadestér before AD 1

Asked and answered: **there is nothing there to un-suppress.** Stadestér's entire pre-AD-1
Americas content is the four Modelski stamps plus Teotihuacan's fabricated Bronze Age rows —

```
Tikal        -900..500  flat 100,000        Caracol   -900..500  flat 100,000 (140,000 at -700)
Tiahuanaco   -500..500  flat 100,000        Tula      -300..500  flat 100,000
Teotihuacán  -900:100,000 · -800:150,000 · -700:100,000 · -600:100,000
```

Turning the ramp off would put a 100,000-person Tikal on the map in 900 BC — before Maya
urbanism exists at all — and a 150,000-person Teotihuacán at 800 BC, on a site whose first
monumental construction is around 150 BC. Every one of those numbers is the thing `nw_cap` was
written for. **The new sources are not merely cleaner, they are the only actual data**, so the
ramp stays exactly as it is and the additions bypass it by citation instead.

### What went in

Round figures, phase-dated, no interpolated filler; the build.py comments carry the source and
the range for each. Peak values as drawn:

| city | drawn span | peak | basis |
|---|---|---|---|
| Cuicuilco | 400 BC – AD 100 | 20,000 at 150 BC | Sanders, Parsons & Santley, *The Basin of Mexico* (1979) |
| El Mirador | 200 BC – AD 150 | 15,000 at AD 1 | site core; Hansen's basin figures are a region, not a city |
| Kaminaljuyú | 100 BC – AD 900 | 10,000 at AD 200 | Miraflores-phase peak |
| Moche | AD 200 – 800 | 15,000 at AD 500 | Chapdelaine, urban zone at Huacas de Moche |
| Wari (Huari) | AD 600 – 1000 | 30,000 at AD 800 | Isbell; range 10,000–40,000 |
| Calakmul | AD 250 – 900 | 50,000 at AD 650 | PACUNAM LiDAR, Canuto et al. 2018 |
| Copán | AD 400 – 950 | 20,000 at AD 780 | Copán valley survey (Webster, Freter & Gonlin) |
| Chichén Itzá | AD 800 – 1200 | 40,000 at AD 1000 | northern lowland capital, c. 900–1050 |

Plus **Monte Albán** via `CENSUS["Santa Cruz Xoxocotlán-Mexico"]`, which had Chandler's single
AD 800 row and so appeared only as it was being abandoned. Blanton's settlement sequence takes it
back to 500 BC: `−300: 5,000 · −100: 15,000 · 600: 25,000`, meeting Chandler's 800: 30,000. Done
as a `CENSUS` and not a new entry on purpose — the city already had one, and a second dot on the
same hill would be the Danapur/Pataliputra error rather than a fix.

**Deliberately left out, on the display floor rather than on doubt**: Caral/Norte Chico (~3,000),
Chavín de Huántar (2,000–3,000), San Lorenzo (~5,500), La Venta (3,000–8,000) and Palenque
(6,000–8,000 against a 10,000 floor at AD 700). Adding entries that can never draw would imply
coverage the map does not have. The Preclassic Americas really is a handful of dots.

### What changed on the map

| year | Americas cities drawn, before → after |
|---|---|
| 300 BC | 0 → 1 |
| 200 BC | 0 → 5 |
| AD 1 | 5 → 8 |
| AD 500 | 4 → 9 |
| AD 600 | 4 → 10 |
| AD 900 | 2 → 5 |

The 900–1050 frames that §2.2 called out as "1 city drawn, and it is the wrong one" now carry
Chichén Itzá, El Tajín, Tula and Mérida. `validate.py` is unchanged on C/D/E/F/G/H and **improved**
on A (11 → 10) and B (75 → 74), both from the Richmond drop.

### The two remaining sign flips

Repaired, and by different routes because the situations differ. **Hamilton, Ontario** was moved
in `SITE_COORDS` — there is no other Hamilton-Ontario entry, so dropping it would remove Canada's
ninth-largest city from the map. **Richmond, Virginia** went to `DROP_KEYS` instead: it is a third
duplicate, `Richmond-Virginia` already draws at the real coordinate with a longer record
(1790–2000, 1.37M once the metro graft lands), and its 1900–1975 span is a strict subset. Moving
it would only have handed dedup a fourth co-located entry to arbitrate.

Note Hamilton now shows a 1975 → 1976 step of 515k → 262k. That is the ordinary graft seam —
populstat's terminal figure is the metro, the WUP urban centre is the dense core — and it sits
between the median (1.40x) and p75 (2.24x) of `SWITCH_STEPS`. It is not new behaviour, it is
simply visible now that the city is on the right continent.

## 2.10 The three Modelski stamps, re-dated (2026-08-22)

Done in the same session, and kept in its own section because it is a **different act**: these
override a figure the map already draws rather than fill a hole, which is what the Guangzhou
lesson says to be slowest about. The test applied to each was *does the entry's own benchmark
source contradict it, or is it unsupported by any source* — not *does newer literature prefer a
different number*. All three passed it.

All three are one defect. Modelski's round 100,000 stamped on every century of a New World site
fixes the magnitude at a plausible peak and attaches it to the wrong 500 years. `nw_cap`
suppresses the deep-antiquity end of that stamp and always has; what it cannot do is correct the
**date**, so each was drawn at peak size four to six centuries early and then declining through
its actual floruit.

| | before | after | what settled it |
|---|---|---|---|
| **Caracol** | 100,000 from AD 100, peak 500, dead 800 | 5,000 at AD 1 → **100,000 at 700** → 15,000 at 900 | **No chandlerV2 row at all** — the last New World figure with no benchmark source of any kind. Magnitude kept (Chase & Chase's LiDAR settlement figure, ~200 km²); only the date was wrong. Peak is c. AD 650–700, after the defeat of Tikal in 562; abandoned c. 900. |
| **Tikal** | 100,000 from AD 100 | 10,000 at AD 1 → 30,000 at 400 → **63,000 at 751** → 10,000 at 900 | Its **own source** dissents: chandlerV2 has `751 = 63,000` and `800 = 40,000`, both already correct on the map. Tikal was drawn 60% above its own benchmark four centuries before that benchmark's date. Chandler agrees with Culbert et al. (1990) on the central 120 km²; only the stamp disagreed. |
| **Tiahuanaco** | 100,000 from AD 100, 44,700 at 800 | 3,000 at AD 200 → 8,000 at 500 → **20,000 at 800** → 22,000 at 1000 → 4,000 at 1150 | Barely an override — the **Lhasa shape**. chandlerV2 says `800 = 20,000` and stadester held 44,721 there, so restoring it is a CENSUS *recovery*. Tiwanaku's urban phase is c. AD 500–1000; the flat 100,000 from AD 100 predates the city by four centuries. |

Two details worth keeping. **Tikal's window stops at 750** so both Chandler rows survive untouched
and keep their `c` code — the inserted points are only the rise into them, and its `s` string
reads `iaacba`, which is the whole argument in six characters. And **Tiahuanaco's window clears
the −200 emergence seed** where Caracol's and Tikal's keep it: those two have Preclassic
occupation from c. 600 BC to emerge from, Tiwanaku does not — the site is occupied from c. AD 110.

Caracol is carried at Chase & Chase's figure rather than silently re-estimated, the same way
Cahokia is carried at Chandler's 40,000; conservative readings put the densely settled core
nearer 50,000–65,000, and the comment says so.

### What changed on the map

* **Caracol is top-12 in the world at AD 700 and out of it by 800** — i.e. at its actual peak
  rather than three centuries before it. It is no longer the largest American city for six
  straight centuries.
* AD 700 in the Americas now reads Caracol 100,000 · Tikal 56,600 · El Tajín 40,000 ·
  Calakmul 34,700 · Monte Albán 27,400 · Wari 17,300 · Copán 16,500 · Tiahuanaco 14,700 — a
  hierarchy with a shape, where before it was four cities moving in lockstep.
* `validate.py` unchanged on every check.

One pre-existing oddity noticed in passing, **not** introduced here: Tikal's AD 800 point reads
`b` (Buringh) though it is a Chandler row. Tikal's entry has no `chandler_modelski_key`, so
provenance.py's tier-1 value join never fires for it and the year-grid tier claims 800. It is a
one-character label on one point and nothing depends on it, but it is the same join weakness that
`chandler_benchmark()` had to work around in §2.7a.

---

## 2.11 History notes, after the data moved (2026-08-22)

`data/events.json` was written against the old figures, so the last step was checking whether any
note had been broken, orphaned or newly earned. `tools/check_events.py` reports **0 errors** before
and after.

**Nothing had to be removed, and nothing had to be shifted.** Every existing Americas note still
fires on a year its data supports, and two of them are now better supported than when they were
written:

* **Teotihuacán burns (550)** — its curator note said "the data does NOT give a sharp break …
  the bubble is already sagging when this fires", and floated moving it to 622. With Chandler's
  AD 500 = 125,000 restored, the note now fires 50 years past a sharp peak, on the way down. The
  suggestion is withdrawn in the note itself; 550 is now the better year of the two.
* **Maya collapse (800)** — its note said "both Maya curves simply END at 800". They no longer do;
  Tikal runs to 900 and Caracol to 900, so the Terminal Classic the note wanted to describe is
  drawable, and it is now five bubbles declining rather than two vanishing.

### Five notes the previous pass said it could not write

The curator notes name their own blockers, and the rebuild removed them. Each new note's `p`
quotes the sentence that was blocking it.

| year | note | the blocker it clears |
|---|---|---|
| −300 | Monte Albán | "starts at 800:30k, roughly 1,300 years late … the data catches only its death" |
| 90 | Cuicuilco buried by lava | "Cuicuilco … the standard explanation for where Teotihuacan's first population came from" — listed as missing |
| 650 | Tikal and Calakmul at war | "Calakmul in particular means the Tikal-Calakmul rivalry in the brief cannot be shown at all" |
| 750 | Wari and Tiwanaku | "absent from the dataset for this region: Wari, Moche … the Andes before 1438 is Tiahuanaco (with a fake curve) … and nothing else" |
| 1000 | Chichén Itzá at its height | "Copan, Chichen Itza, Uxmal and Calakmul are all absent from the dataset" |

The four stale curator notes (Teotihuacán ×2, Maya collapse, Inca expansion) each carry a dated
`UPDATE 2026-08-22` block rather than a rewrite — the original reasoning is the record of why the
note is where it is, and it stays.

Three of the five needed their year moved, all for placement rather than history, and all three
reasons are in the `p` notes: **Cuicuilco 100 → 90** (at 100 it fired in the same instant as
`Rome reaches a million`, pri 1, and never won a slot — the block is exactly y=100..120, the
length of Rome's hold, and it has to move *early* because Cuicuilco's series ends at 100);
**Wari 800 → 750** and **Tikal/Calakmul 650, anchor moved to Tikal** (both were sitting on their
anchor city's own peak control point, so `im: 2` had nothing to grow into — the Maya one was
resolved by re-anchoring rather than re-dating, because moving it to 600 made it invisible behind
the Plague of Justinian).

### The one thing that is a judgement call, not a fix

**Two of the five cost an existing note its slot**, and no year avoids it — the AD 600–800 band is
saturated. Measured individually against a 24-note baseline:

| new note | displaces |
|---|---|
| Monte Albán (−300) · Cuicuilco (90) · Chichén Itzá (1000) | **nothing** |
| Tikal and Calakmul at war (650) | `Arab garrison cities founded` (671) |
| Wari and Tiwanaku (750) | `An Lushan rebellion` (756) |

Swept 562–700 and 600–1000 respectively: the Maya note has **no** zero-cost year anywhere in its
defensible range, and Wari's only zero-cost years are 975–1000, which is past the point the note
describes (Wari is collapsing by then, so it would have to become a Tiwanaku note).

Both were kept, on the grounds that the displaced notes are the weaker halves of the trade:
`Maya collapse` at 800 **never rendered either**, so the Maya had no visible note at all before
this, and South America had **none anywhere before 1438** — while East Asia already has three in
the 500–1000 era. Both displaced notes also independently trip the data check (`An Lushan` says
decline but Xi'an never dips; `Arab garrison cities` says growth but Basrah never rises).

**That reasoning is a preference, not a finding.** Either note reverts by deleting one entry, and
the displaced note comes straight back.

---

# Part 3 — sub-Saharan Africa

## 3.1 The answer: no, and the cities are in the source

At **1575** the map draws **9** cities in sub-Saharan Africa and **blanks 8**. The blanked ones
are not missing from the data — they are drawn up to a point and then floored to 1,000 by a
planted fade:

```
blank from   to      entering at   city
      1574  1930          75,000   Gao            <- Songhai capital, at its peak
      1567  1964          50,000   M'Banza Congo  <- capital of Kongo
      1534  1946          60,000   Shaki
      1524  1804          21,300   Tombouctou     <- Timbuktu
      1399  1876          20,000   Dienne         <- Djenné
      1375  2014          20,000   Soba
      1624  1916          40,000   Sinnar         <- Funj sultanate capital
      1165  1924          30,000   Dunqulah       <- Old Dongola, capital of Makuria
      1408  1861          40,000   Karima
       -500 1969          45,000   Ma'rib
```

Chandler's own figures for the year you were looking at:

```
Gao          AD_1550=75000  AD_1575=75000  AD_1585=75000  AD_1591=75000
Tombouctou   AD_1500=25000  AD_1600=25000
Dienne       AD_1500=20000
M'banza-Congo AD_1500=40000  AD_1543=50000
Dongola      AD_1500=20000  AD_1800=10000
```

So at 1575 **Gao at 75,000 would be the largest city in sub-Saharan Africa** — larger than Zaria
(65,000), which currently leads. Timbuktu, Djenné and Mbanza Kongo would follow. The Songhai
empire, the kingdom of Kongo and the Funj sultanate are all in the data and none of them reach
the screen.

Of the ~14 African blanks, I make it **9 or 10 false abandonments**. The correct ones are Ma'rib,
Yarîm/Zafar (already fixed via `DISAPPEARED`), Soba (Alodia fell c. 1504) and Karima/Napata.

## 3.2 Why — and it is not an Africa-specific bug

The mechanism is the carry-forward strip plus the `CF_MODERN = 1800` gate, working exactly as
documented. Timbuktu's raw series is:

```
1300:10,000  1400:17,500  1500:21,250  1600:21,249  1700:21,249  ... 1820:21,249  1828:12,000
```

`1600 → 1820` is 220 years of a verbatim repeat, so the strip fires. The run *ends at* the first
modern-era figure (1828), which by the §3.4 rule means "dead", so `plant_fades` blanks it. But
Chandler has **two real benchmarks here** — AD_1500 = 25,000 and AD_1600 = 25,000 — and the 1600
one is consumed as part of the run. Gao is the same shape with the fade landing at 1574, just
before its 1575/1585/1591 peak.

The gate's dead/alive test is "does the run end at the first modern census, or centuries
earlier at another pre-modern estimate?" For Vijayanagara and Kamakura that reads correctly. For
a Sahelian or Nubian city it reads backwards: the run ends at the first modern census **because
that is when European record-keeping starts**, not because the city died. The rule is detecting
the arrival of the observer.

**This is not Africa-only.** Blanked counts at 1575: Europe 24% of drawn+blanked, Africa 47%,
East Asia 49%, South Asia 55%, Middle East 64%. Europe is the outlier, not Africa — it has dense
populstat census coverage early enough that its cities never produce the carry-forward
signature. Many of the Middle Eastern and Mediterranean blanks are *correct* (Babylon, Ephesus,
Pergamon and Samarra genuinely were ruins in 1575), which is why the raw percentages overstate
the problem there. Africa's are largely not correct, which is why it shows.

The `DISAPPEARED` table is already the right instrument, and the `[]` form ("this city was
continuous, take the fade off") is already the commonest correction in it. These are more of the
same. Candidates, in rough order of confidence:

```
"Tombouctou-Mali":  []            # Caillié found ~12,000 in 1828; never abandoned
"Gao-Mali":         []            # declines hard after 1591 but persists
"Dunqulah-Sudan":   []            # Chandler has 20,000 at 1500 and 10,000 at 1800
"Sinnar-Sudan":     []            # founded 1504, Funj capital until 1821
"M'Banza Congo":    []            # or a narrow (1568, 1571) for the Jaga sack
"Dienne-*":         []            # plus a coordinate fix, see below
```

## 3.3 Djenné is also in the wrong country

`Dienne` is drawn at **(15.033, −16.350)** — in Senegal, ~700 km from the real Djenné at
(13.91, −4.55) in Mali. That is Chandler's own coordinate (its row reads
`Dienne / Guinea, Jenne / Senegal`, certainty 3), carried through unchanged. So Djenné is both
blanked and misplaced; the fade is hiding the error.

## 3.4 Benin City has no history before 1901

Same defect as Sparta and Pyongyang: `Benin City (agglomeration?)-Nigeria` runs from 1600 and is
dropped by `DUP_MARKERS`; `Benin City-Nigeria` starts 1901. Chandler has
`Benin / Oedo / Nigeria AD_1600 = 50,000 · AD_1650 = 60,000 · AD_1668 = 65,000 · AD_1750 = 50,000`.
The walled Edo capital the Portuguese described is entirely absent from the map. `MERGE_INTO`.

**Applied - and it introduced a 19.7x cliff at 1900->1901, since fixed. See Part 4.3.**

## 3.5 Two that may be drawn too early

The inverse error, worth knowing about before adding more African cities:

- **`Alkalawa`** is drawn from 1513 at ~33,000. Chandler's row is certainty 3, and Alkalawa only
  became Gobir's capital in the 18th century. Its coordinates (9.28, 8.40) are also ~200 km south
  of the real site (≈13.4, 5.6). Both the date and the place look wrong.
- **`Oyo`** at 60,000 in 1575, drawn at (8.16, 3.61). Old Oyo (Katunga) is at ≈(8.98, 4.30); the
  Nupe sack of c. 1535 sent the court into exile at Igboho until c. 1610, so 1575 is inside a
  period when Oyo-Ile was not occupied.

## 3.6 Still genuinely missing after all of the above

Not in Chandler or Stadester, but well enough attested to be worth knowing you don't have them:
**Harar**, **Mogadishu**, **Kilwa's peer ports** (Malindi, Pate, Lamu, Sofala — mostly 5–15k, so
borderline), **Kumbi Saleh** (Ghana's capital, abandoned by 1300), and for earlier frames
**Great Zimbabwe** — which *is* in Chandler (`AD_1300 = 25,000 · AD_1400 = 35,000 · AD_1450 =
40,000`, certainty 2) but reaches nothing, and would be the largest city in southern Africa for
those three frames. **Aksum** is in Chandler too (`AD_1100 = 125,000`, which is not credible —
Aksum had collapsed by then — but `AD_1400 = 30,000 · AD_1500 = 33,000`, which is arguable) and
`CF_END` already declares it finished at 700. **Re-examined in Part 4.7: `CF_END 700` is right,
restoring the 1400/1500/1770 benchmarks is not recommended, and what is actually missing —
Aksum's floruit and its port Adulis — is in no shipped source.** **Elmina** and **Ouidah**, and
with them the whole Bight of Benin slaving coast, belong on this list too: Part 4.6.

---

# Part 4 — the 2026-08-22 pass

Seven cases, raised from looking at the map rather than from a scan. Checked against the build of
2026-08-22 and against `chandlerV2.csv` and the raw `stadester_cities.json` in every case, because
the headline finding is that a previous pass got a source claim wrong and nothing caught it.

Four were pipeline defects and are **fixed**. Three are genuine source holes and are **not**, and
the useful thing about them is knowing which is which.

| | case | verdict |
|---|---|---|
| ✅ | **Baghdad** 932–1400 — a `CF_KEEP` exemption on a false premise | `CF_KEEP` withdrawn, `CENSUS` + `DROP_YEARS` |
| ✅ | **Soweto** grafted to Lenasia, a different township | `GRAFT_DENY` |
| ✅ | **Benin City** 1900: 296k → 1901: 15k, a 19.7× cliff we created | `DROP_YEARS` |
| ✅ | **Kaohsiung** the largest city in Taiwan from 1880, at 220,000 | `CLIP_BEFORE 1921` |
| ❌ | **Nubia before AD 800** — Kerma, Napata, Meroë | not in any shipped source |
| ❌ | **Elmina and Ouidah** — the Bight of Benin slaving coast | not in any shipped source |
| ❌ | **Aksum's floruit** — the era the city mattered | not in any shipped source; `CF_END 700` is right |

Applied in a second round on the same day, after the first was reviewed:

| | case | verdict |
|---|---|---|
| ✅ | **Abomey** — Dahomey's capital drawn 85km away on a Cotonou suburb | `CLIP_BEFORE` + `CENSUS` |
| ✅ | **Allada** — blanked from 1706; the fade was right, its dates were not | `CF_END` + `DISAPPEARED` |
| ✅ | **Córdoba** — 500 years held at one number, through the fitna and the Reconquista | `CENSUS` |
| ❌ | **Córdoba's level at 1000** — deliberately not touched, see 4.9 |  |

And a third round, once the rules freeze (`spec.md` §0) made clear that the right response to
check G's blind spot was to *work its output by hand* rather than widen it — see 4.10:

| | case | verdict |
|---|---|---|
| ✅ | **Chang'an** drawn as an even slide through the fall of the Tang | `CENSUS` |
| ✅ | **Luoyang** — 300 years of one spline value | `CENSUS` |
| ✅ | **Basra** — 783 years arriving as two flat blocks | `CENSUS` |
| ✅ | **Great Zimbabwe** labelled "Zimbabwe" | `RENAME` |

## 4.1 Baghdad — a hand exemption that invented a collapse

The map drew **932: 1.1M → 1100: 1.1M → 1150: 10,000 → 1250: 100,000**: a 110× collapse in fifty
years with no event behind it, then a tenfold recovery. Two of the three worst medieval jumps in
`analyze_jumps.py`. All of it ours.

`build.py`'s `CF_KEEP` entry said *"Chandler states 1,100,000 at 932, 1000 AND 1100 — three
benchmarks"* and *"found by validate check G"*. **Both are false**, and the evidence was already
in the repo three ways over:

1. `chandlerV2.csv`'s Baghdad row reads `AD_932 = 1,100,000 · AD_1000 = 125,000 · AD_1100 =
   150,000`. One benchmark at that level, not three.
2. `provenance.py` classifies 932 `chandler exact`, 1000 `fill`, 1100 `populstat default` — the
   signature of a hold, not of a repeated assertion. The built `s` string said so on the map.
3. Check G **never reported that run.** Its value test (`GBM_VALUE_TOL = 2.0`, the Guangzhou
   guard) exists to reject exactly this: 1.1M against 125,000 is 8.8× out. Re-running check G with
   the exemption removed returns Baghdad's *other* run, `1250..1400`, losing Chandler's 1350 and
   1400 behind a 1.1× cliff.

So the exemption propped up populstat's carry-forward at the Abbasid peak and handed it to
Chandler's 1150 outlier. **And then hid it**: `CF_KEEP` short-circuits both `strip_carry_forward`
and check G, so once written the entry made itself invisible to the two things that would have
argued with it. Check A reported it every build, marked `*` — *deliberate, not a defect*.

**Fixed** in three tables (full account in `spec.md` §3.4a):

- `CF_KEEP` — the entry withdrawn, reasoning kept in place as a cautionary note.
- `CENSUS` — Chandler's 1000/1100/1200/1250/1300/1350/1400 restored verbatim across 933–1400.
  Stadestér had overwritten six of the seven: two by holding 932 forward, one by interpolating,
  three by holding 1250 forward.
- `DROP_YEARS` — Chandler's `1150 = 10,000` deleted. **This is a judgement call and the third
  editorial deletion of a benchmark on the map** (after Roma 361 and Carthage −100). It sits
  between his own 1100: 150,000 and 1200: 100,000, i.e. his row asserts a 93% loss and a tenfold
  recovery around a year in which nothing happened to Baghdad — al-Muqtafi's restored caliphate
  withstood a Seljuk siege in 1157. Read as a missing digit it is 100,000, which is what he gives
  at 1200, 1250 and 1350. Nothing in the pipeline could reach it: `despike()` is up-only and
  check F needs a return inside 20 years.

**Result.** Baghdad declines from the Abbasid peak through the Buyid and Seljuk periods, is
100,000 at 1250, **40,000 at 1300** — Hulagu, 1258 — recovers to 100,000 under the Jalayirids by
1350, and is 90,000 at 1401, the year of Timur's second sack. Every figure is Chandler's.
Check A loses its Baghdad row, check H loses it, and the dataset's `10x+` jump bucket goes 25 → 22.

**It changes the AD 1000 frame.** Baghdád goes from 1.1M and first in the world to 125,000 and
fifth, behind Istanbul (330k), Kaifeng (321k), Kyôto (155k) and Merv (144k). Chandler's 125,000 is
at the *low* end of the modern literature exactly as his 932: 1,100,000 is at the high end; both
are carried as he states them. `spec.md`'s regression canary for 1000 has been updated.

**Three notes in `data/events.json` referenced the broken curve and have been updated** — 1258
Baghdad, 969 Cairo, 1221 Merv/Nishapur. The 1258 note is the one that matters: its `im` had been
cut to −1 and its detail rewritten to claim the caliphate rather than a population, because a −3
collapse colour was firing over a *rising* bubble. That constraint is gone — `im: −2` is now the
supported value (heavy loss, and the recovery to 100,000 by 1350 is in the data; −3 is not right,
the city comes back). The rendered fields are left as they are; only the curator note is updated.

## 4.2 Soweto — grafted to a different township

`1970: 602k → 1991: 597k → 2000: 57k → 2025: 215k`. The tail is **Lenasia** (wup 3855, 6.9km
away), a separate township, not Soweto — which is ~1.3M. Full account in `spec.md` §3.6b; fixed
with `GRAFT_DENY`, so Soweto ends on its own 1991 census held forward.

Two checks miss it and it is worth recording why, because the reasons are opposite:

- **`TIGHT_MIN_FRAC = 0.2`** compares all-time peaks. Lenasia reaches 36% of Soweto's peak — but
  only by 2025, and the two series barely overlap in time. The diagnostic quantity is the step
  *at the handover*, which is 10.4× down.
- **Check C** scores the historical peak against the post-2000 *maximum*, and Lenasia's own growth
  brings that to peak/2.8, inside the peak/5 gate. Lenasia growing is what hides it.

**Not fixed, and bigger than this entry:** Soweto double-counts against Johannesburg's 10.7M FUA
either way, as Tembisa (1.3M), Evaton (1.1M), Katlehong, Diepmeadow, Benoni and Germiston already
do. The source says so directly — Soweto's entry carries `is_agglomeration_of: johannesburg`,
which is `spec.md` §6.7's open item. `MERGE_INTO "Soweto-South Africa" → "Johannesburg-South
Africa"` would fold in nothing (1970–1991 sits entirely inside Johannesburg's 1889–2025) and
remove the dot, which is the Bhilai/Durg precedent. Left alone because it removes a famous name
from the map on a rule the other five township entries do not yet get.

## 4.3 Benin City — a cliff the Part 3 fix created

§3.4 above put Benin City's pre-1901 history back with `MERGE_INTO`. That worked, and it brought
137 years of straight line with it: the donor variant runs Chandler's benchmarks to `1854: 60,000`
and then a real populstat agglomeration count of `1991: 762,700`, with nothing between them.
The merge folds every donor year outside the base's 1901–1995, so the whole ramp came too, and the
map drew **1900: 295,943 → 1901: 15,000** — a 19.7× cliff in one year, the *sixth largest jump in
the dataset*, larger than any WUP seam. The 1900 point's provenance code was `i`. Fill.

Fixed with `DROP_YEARS[donor] = range(1855, 1991)`. What remains is the two measurements:
Chandler to 1854 and populstat from 1901, interpolating 60,000 → 15,000 across the 47 years that
contain the 1897 Benin Expedition, which burned the city and exiled the Oba.

**Worth knowing for later:** the 1991 figure (762,700, the Nigerian census agglomeration count) is
real and is declined here on definition grounds — it is an agglomeration and the base is city
proper. It never reached the map anyway, being inside the base's range. It is the figure to reach
for if the 1995 → 1996 seam (224,000 → 1.05M, Africapolis) is ever softened.

**The general shape is not Benin City's alone.** Any `MERGE_INTO` donor whose own record is two
distant anchors joined by Stadestér's fill will do this, and the merge is the operation that makes
the fill visible. Worth a scan: *donor years folded in that survive DP but are `i` in provenance,
adjacent to a base anchor more than 3× away.*

## 4.4 Kaohsiung — the largest city in Taiwan, from 1880, at 220,000

Raised from the map: Kaohsiung led Taiwan at 1880 and then fell a long way. It is wrong, and
every point before 1921 is wrong.

```
1880:220,000  1890:220,000  1898:100,000  1900/1910/1920:100,000  1921:35,400  1924:41,000 ...
```

Two flat blocks of a round number, each ending in a cliff, the last landing on the first real
municipal count. That is an administrative area handed to a city series — the same defect
`trim_admin_tail()` removes from the Chinese entries, except at the front. 220,000 is roughly Qing
Fengshan county, whose territory modern Kaohsiung occupies; 100,000 is plausibly the Japanese
Takow district (打狗支廳); 35,400 is Takao town (高雄街), created in 1920, which is what every
later figure continues.

Takow was a treaty port from 1864 and the consular trade reports describe a small harbour
settlement at Kihou and Takow — thousands. No corroboration exists anywhere in the shipped
sources: `chandlerV2.csv`'s Kaohsiung row starts at `1950 = 261,000`, and Chandler carries nothing
for Taiwan before Taipei's 1900.

**What it cost:** Taipei's entry does not open until 1898 and Tainan — the island's capital and
largest city throughout the Qing, at 70,000 — was drawn a third of Kaohsiung's size. Fixed with
`CLIP_BEFORE 1921`, the Qingdao treatment.

**Not fixed, adjacent:** Taipei has nothing before 1898 in any shipped source, so the Qing walled
city (1884) and the Bangka/Dadaocheng settlements it absorbed are absent. Chandler's Taipei row
starts at `1900 = 80,000`.

## 4.5 Nubia before AD 800 — confirmed, and it is a source hole

§1.6 said this and it is still true after re-checking directly. The whole of Sudan, Ethiopia and
Eritrea is **empty at AD 100 and at 300 BC** — zero cities in the box 3–23°N, 21–44°E at either
year. The earliest thing anywhere in the region is `Karima` at 800.

`chandlerV2.csv`'s complete list of rows in Sudan is **Dongola, Khartoum, Omdurman, Sennar, Soba**,
plus a row filed `Kush / Egypt` at (18.53, 31.84) — Jebel Barkal, i.e. Napata's site — which turns
out to carry an `AD 800–1400` series, not Napata's. No Kerma, no Napata, no Meroë; nothing in
Stadestér either. **`SYNTHETIC`'s first condition — every figure a verbatim row from a source in
the repo — cannot be met**, so this is not fixable with the machinery that exists, and adding it
would mean typing populations rather than recovering them.

Worth recording precisely because it looks like a pipeline gap and is not: on a map that draws
10,000-person Sumerian towns, the kingdom of Kush is invisible for its entire 1,800-year run.

## 4.6 Elmina and Ouidah — the same answer, with a repairable neighbour

Both are on the map and both start far too late:

```
Elmina   1880: 6,000   (entry says `found.yr: 1471`, `first important European settlement in Gold Coast`)
Ouidah   1921: 9,600
```

So the Bight of Benin coast is invisible for the whole period it mattered — Elmina Castle from
1482, Ouidah the second-busiest slaving port in Africa through the 18th century. Neither has a
`chandlerV2.csv` row and neither has anything earlier in Stadestér. **Same verdict as 4.5: a
source hole, not a pipeline loss.**

**But the hinterland is in the source and is misplaced.** Chandler has
`Abomey-Calavi / Benin (7.18286, 1.99119) 1750 = 24,000 · 1780 = 24,000 · 1800 = 24,000 ·
1861 = 20,000`. That coordinate is **Abomey** — the Dahomey royal capital, the polity that ran the
Ouidah trade — to within 300m of Stadestér's own `Abomey-Benin` entry. Stadestér fused the row
into the *modern* `Abomey-Calavi` entry at (6.4503, 2.3468), a Cotonou suburb **85km south-east**,
and everything in that entry after 1861 is a 130-year straight line from 20,000 to 21,300. Abomey
itself starts at 1921.

This is the Chan Chan shape exactly — the historical half of one entry belongs to a different
place that has its own modern entry — and it is repairable with tables that exist:
`CLIP_BEFORE["Abomey-Calavi-Benin"]` to take Chandler's figures off the suburb, plus a `CENSUS`
recovered-figure entry putting the same four numbers on `Abomey-Benin`. **Not applied here** —
it is a different city from the two raised, and it wants its own check of what `Abomey-Calavi`'s
modern WUP tail does once the front of its record is gone.

`Allada` (Ardra), the other Chandler row on that coast (`1682 = 40,000`), is on the map and
**faded from 1706 to 1996** — the §3.2 signature, a 319-year verbatim hold ending at the first
modern count.

**Both applied.** Abomey took `CLIP_BEFORE["Abomey-Calavi-Benin"] = 1862` plus a `CENSUS`
recovery of the same four figures onto `Abomey-Benin`, and the Dahomey capital is now on the map
at 24,000 from 1750 — so the Bight of Benin has a city on it for the period it mattered, even
though its two ports cannot be drawn.

**Allada did NOT take a bare `[]`, and the reason is worth keeping.** Gao, Timbuktu and Dongola
all have *later* Chandler benchmarks proving the compiler kept tracking them; Allada has none.
Removing the fade outright draws a 320-year line from 40,000 (1682) to 23,400 (2002) — an Allada
of ~30,000 through the eighteenth and nineteenth centuries, which is false by roughly a factor of
ten. Agaja of Dahomey took and sacked the town in March 1724 and the court moved to Abomey; the
collapse is real. **What was wrong was the dates, not the fade.** So:

- `CF_END[("Allada-Benin", 1682)] = 1700` recovers Chandler's second benchmark (both certainty 1,
  both 40,000), which carries Allada at full size to the eve of the conquest instead of starting
  its decline in 1682, when the kingdom was at its height. Check G *does* report this one —
  `1 lost, run 1682..2001` — but ranks it 1.7×, far below the printed head, which is why it sat
  unnoticed.
- `DISAPPEARED["Allada-Benin"] = [(1724, 1960)]`. 1724 is the conquest and is the claim. **1960 is
  the weakest number in this pass and is flagged as such**: nothing measures Allada between 1700
  and 2002, and 1960 is the 2002 count of 23,400 back-projected at the growth Benin's small towns
  actually had after 1950. If it is wrong it is wrong by decades at a size the viewer can barely
  draw.

## 4.7 Aksum — `CF_END 700` is right, and it is not what is missing

Aksum runs `1838: 1,700 → 2025: 131,000` and nothing before. The cause is known and is a hand
decision: Stadestér holds Chandler's `1100 = 125,000` verbatim from 1100 to 1830 — 730 years — and
`CF_END["Aksum-Ethiopia"] = 700` declares the city finished before the run starts, so the strip
takes all of it.

**That call is correct.** Aksum had lost Adulis and its Red Sea trade by the 8th century and was a
small ceremonial town thereafter; 125,000 in 1100 is not defensible at any reading.

**What it also takes is three later Chandler benchmarks** — `1400 = 30,000 · 1500 = 33,000 ·
1770 = 3,000` — which Stadestér had already overwritten with the 1100 value. Restoring them is
possible and is **not recommended**: 30,000 in 1400 is the same non-credible figure a century
later (the Zagwe and early Solomonic courts were at Lalibela and on the move, not at Aksum), and
1770: 3,000 — which *is* credible, and roughly what James Bruce described in 1770 — is below
`MINPOP = 5,000` and would not draw.

**So the `CF_END` decision is not the defect. The defect is that Chandler's row starts at 1100**,
by which time Aksum was already finished, and nothing in any shipped source covers its floruit
(c. AD 100–700), when it was one of the great powers Mani names alongside Rome and Persia.
Its port **Adulis** is absent from every source too. `SYNTHETIC` cannot reach either.

Filed with 4.5 and 4.6: the Horn and the Nile above Egypt have no pre-medieval coverage at all,
and no amount of pipeline work will produce any.

## 4.9 Córdoba — fix the hole, decline the level

Raised as a question: *is "the largest city in the world in AD 1000" right?* **No, it is not a
consensus** — and the map's actual defect turned out to be somewhere else.

**The claim.** 450,000, first in the world, comes from Chandler (1974/1987) and is repeated in
essentially every popular account. The published range for caliphal Córdoba is enormous — roughly
**90,000 to 1,000,000**, with the million figure explicitly called unlikely; ~100,000 is the
classic built-area estimate, 250,000–450,000 is what popular accounts quote, and
**Bosker–Buringh–van Zanden — a modern peer-reviewed dataset, and the one this entry *is* —
sit at the bottom at ~79,000**. Kaifeng has its own advocates for the top slot, and even
Constantinople's conventional 300,000 for 1000 is contested as beyond Byzantine logistics. There
is no settled answer to defer to.

**So the level is left alone**, on the project's own rule: this would *override a drawn figure*
rather than fill a hole, which GAPS Part 2 already flags as needing its own decision each time and
which the Guangzhou lesson says to be slowest about. It would also mean taking Chandler's 1000
while ignoring his 900, and his row reads `800: 160,000 · 900: 20,000 · 1000: 450,000 ·
1100: 60,000` — a 22× spike between two benchmarks. Not a row to cherry-pick from.

**The real defect was the hole after it, and it is fixed.** Stadestér holds **79,125 verbatim at
1000, 1100, 1200, 1300, 1400 and 1500** — a 500-year carry-forward — so the strip kept 1000 and
the map drew one straight line from there to 1550: 33,000. Five and a half centuries in a single
segment, through the fitna of 1009–1031 and through the Reconquista of 1236. The 1248 Sevilla
note in `events.json` already said so from the other side.

`chandlerV2.csv` has all five years, coherent and monotone, and *post-1100 his row is fine* even
though the earlier part is not: `1100/1200: 60,000 · 1300: 40,000 · 1400: 36,000 · 1500: 30,000`.
They meet the entry's own next figure at 1.1× (30,000 against stadestér's 1550: 33,000) and the
front junction is 1.32×. Recovered via `CENSUS`. Each year takes the one source that has it; no
year gets two.

**Why no check caught it, and this is the part that generalises.** Check G joins entries to
Chandler rows by coordinate at `GBM_JOIN_KM = 5`, and Chandler geocodes Córdoba to (38.046,
−4.894) — **20km north-west of the city**. Measured across the whole corpus by re-running check
G's own logic at wider radii:

| join | runs found | new | new with a ≥2× cliff |
|---|---|---|---|
| 5km *(current)* | 169 | — | — |
| 10km | 185 | 16 | 5 |
| 20km | 189 | 20 | 7 |
| 30km | 193 | 24 | 9 |

So Córdoba is one of **16 runs missed in the 5–10km band alone**. The new hits are a mix: real
ones include Xi'an (`805..1000` at 600,000, `/13.3×`, Chandler's row 8.5km away — Chang'an after
Zhu Wen), Basra (`1123..1500`, `/6.0×`), Luoyang and Kano; junk includes `Lille-Belgium` matched
to Chandler's **Antwerp** 29km away, `Xianyang-China` matched to **Xi'an**, and `Sololá-Guatemala`
matched to **Q'umarkaj**. So the radius cannot simply be widened. Design in `spec.md` §6.13.

## 4.10 Working the blind spot by hand instead of widening the check

The scan in 4.9 found 16 carry-forward runs that check G cannot see at 5km. Under the rules
freeze the response is not a better check — it is to read that scan's output once and write down
what it found. Four cities were worth repairing; the rest were either already hand decisions
(Gallipoli, Sagaing), cosmetic (Huai'an at 1.4×, Kunming at 1.3×, where Chandler's benchmarks are
flat so the redraw is the same line), or false matches the wider radius invented (`Lille-Belgium`
to Chandler's **Antwerp**, `Xianyang` to **Xi'an**, `Sololá` to **Q'umarkaj**).

**Chang'an**, and the best of the four. Stadestér holds 600,000 at 805, 900 *and* 1000, then reads
45,000 at 1077 — so the strip kept 805 and the map drew the Tang capital sliding evenly across 272
years, reading **243,000 in the year 900 against Chandler's own 500,000**. His row is the right
shape and the map had lost it: a ninth century of gentle decline, then a catastrophe — Huang Chao
took the city in 881 and Zhu Wen dismantled it in 904 and moved the capital to Luoyang. One figure
is the difference between a slide and an event. Chang'an is now second in the world at AD 900.

**Luoyang**, the receiving end of that same collapse, and the clearest case of the strip
protecting a non-measurement: stadestér holds **264,157 — a spline value, not anyone's figure** —
at 700, 800, 900 and 1000. Chandler has 800: 300,000 and 1000: 50,000 inside it. Restored.
Not restored: his 100: 420,000, which check G also lists. The map draws populstat's 260,000 there,
so taking Chandler's would override a drawn figure on a 1.6× disagreement — the Guangzhou case,
declined here as everywhere.

**Basra.** Two runs, 100,000 held 717–1100 and 60,000 held 1123–1500, so 783 years arrived as two
flat blocks and the map drew two long lines. Chandler has five benchmarks across them and they are
a real curve: 100,000 through the eighth century, **halved to 50,000 by 1000** (the Zanj revolt of
869–883 and the silting of the canal country), a recovery to 60,000 across the twelfth, 50,000 at
1200, and then the fall to his 1525: 10,000 that spans Hulagu's 1258 sack.

**Great Zimbabwe** is a label, not a data fix, and it is here because GAPS §3.6 above is **wrong
about it**: it says the city "reaches nothing". It is on the map and always was — 1300: 25,000 →
1450: 40,000, the largest city in southern Africa for those frames. It was just filed under the
name `Zimbabwe`, which reads as a country. `RENAME` to the source's own `other_names` value.

## 4.11 Three things the checks could have caught — and are not getting

- **`check_events.py` does not compare `im` against the curve.** The 1258 Baghdad note carried a
  −3 collapse colour over a *rising* bubble and it was found by eye. The tool already loads
  `cities.json` and already finds the nearest drawn city to each note, so the missing piece is
  small: measure the anchor city's log-slope over an adjusted-time window forward of the note year
  and warn when the sign contradicts `im`. It would also catch the inverse — a `+2` over a flat or
  falling curve — which is the commoner mistake once a note file gets long. **Not implemented.**
**Two of the three below are declined under the rules freeze** (`spec.md` §0) and are recorded so
the reasoning survives, not as a plan.

**The `im`-versus-curve check was built** — Anita added it to `check_events.py` the same day, and
it is on the right side of the freeze: `tools/` validates prose written by hand, not a threshold
applied to 22,149 records. First run: **21 notes of 154 whose stated direction the curve does not
support**, including several worth looking at on their own — `An Lushan rebellion` (756) says
decline over a Chang'an that is still rising, `Rome reaches a million` (100) says growth over a
curve already at its peak, and `Plague of Justinian` (542) says decline over a flat Constantinople.
Those are the same class of defect as the 1258 Baghdad note in 4.1, and now they are findable.

- **Check A's `*` line should print Chandler's value beside the run's.** A `CF_KEEP` entry
  short-circuits both the strip and check G, so the only report that still mentions it is the one
  that has already decided it is deliberate. Printing `1.1M held 932..1100 (Chandler: 1.1M · 125k ·
  150k)` would have made 4.1 obvious from the routine output. **Not implemented.**
- ~~**Córdoba is now the AD 1000 frame's biggest shortfall.**~~ **Resolved in 4.9**: the hole is
  fixed, the level is deliberately declined, and the check-G blind spot it exposed is measured
  there.

---

## Method note

Everything here was checked against the built `data/cities.json` rather than inferred from
`build.py`. The scratch scripts used are throwaway, but the two worth rebuilding if this is
revisited are:

- a **blank-stretch scan** — walk each series for consecutive `FADE_FLOOR` points, report the
  span and the value entering it. This is what surfaced the whole of Part 3, and it is close
  enough to a `validate.py` check to be worth adding as one: *"city blanked for N years having
  entered the blank above X"* is a defect signature with a very high hit rate.
- a **region/year census** — cities drawn vs cities blanked inside a lat/lon box at a year. This
  is what makes "9 drawn, 8 blanked" sayable, and it is the number that tells you whether a
  sparse-looking frame is sparse data or a sparse *drawing* of dense-enough data.
