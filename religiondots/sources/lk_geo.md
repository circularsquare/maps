# Sri Lanka — boundaries for 14,003 GN divisions

Wired 2026-09-04. This is the interesting half of Sri Lanka.

| | |
|---|---|
| source | OCHA **COD-AB `cod-ab-lka` v03**, `lka_admin_boundaries.shp.zip`, 118 MB |
| layer | `lka_admin4` — 14,043 Grama Niladhari polygons, EPSG:4326 |
| vintage | **valid_on 2022-08-16**, against a **2024** census |
| output | `data/geo/lk/lk_gnd.gpkg` (14,003 polygons, keyed by CENSUS id), `lk_lookup.csv` |

It is the only public GN-level boundary set that exists. geoBoundaries stops at ADM2, Kontur
is a different geography, WhosOnFirst is settlements, and DCS publishes no digital GN
boundary of its own. So the two-year vintage gap is not avoidable by shopping around.

---

## 1. The pcode join looks right, matches 96%, and is wrong

COD's `adm4_pcode` is **exactly** `LK` + the census's district / DS / GN code triple,
zero-padded. `LK1103005` is district 11, DS 03, GN 005, Sammanthranapura — the first row of
both files. Joining on that string matches **13,472 of 14,003**, and the 531 misses look
like precisely the vintage gap you would predict from a 2022 file and a 2024 census.

**Thirteen DS divisions carry different codes in the two files.**

```
  census 2302 -> COD LK2321   Kothmale West       census 3135 -> COD LK3160   Madampagama
  census 2307 -> COD LK2318   Mathurata           census 3137 -> COD LK3163   Rathgama
  census 2309 -> COD LK2324   Walapane            census 5112 -> COD LK5115   Eravur Pattu
  census 2310 -> COD LK2309   Nildandahinna       census 5115 -> COD LK5139   Eravur Town
  census 2313 -> COD LK2327   Thalawakelle        census 5224 -> COD LK5221   Kalmunai
  census 2314 -> COD LK2330   Norwood             census 9119 -> COD LK9154   Kaltota
  census 3128 -> COD LK3157   Wanduramba
```

The census's `2309` is **Walapane**; COD's `LK2309` is **Nildandahinna**. Eravur Pattu and
Eravur Town are swapped between the two files.

So the pcode join does not merely *miss* those units. It **silently pairs each of them with
a real polygon in the wrong place** — 762,824 people, 3.5% of the country, placed in the
wrong valley — and the only symptom is a match rate that the vintage gap already explains.
Nuwara Eliya is the worst of it: six of its DS divisions shifted, in a district whose
religious composition changes completely over 20 km, from Buddhist Sinhala villages to
Hindu Tamil tea estates.

**This is a new disguise of an old rule and it generalises** — see spec §12. Two files that
share a code *shape* do not share a code, and a partial match is not evidence that the
matched part is right.

## 2. The fix: align by name first, match codes only inside a pair

Names survive renumbering; codes do not. So `lk_geo.py` runs two stages:

1. **DS divisions, by name, within a district.** 340 census against 339 COD.
2. **GN divisions, by code, but only within an already-aligned DS pair**, then by name for
   whatever is left.

That turns the code from a global key into a local one, which is all it is reliable as.

### The names need a transliteration-tolerant comparison

DCS and COD romanise Sinhala and Tamil differently and **disagree on 81 of 340 DS names**:
Mathugama/Matugama, Thumpane/Tumpane, Dickwella/Dikwella, Pitabeddara/Pitabaddara,
Vadamaradchi/Vadamaradchchi, Samanthurai/Sammanthurai. `fold()` drops parenthetical glosses,
collapses the aspirate digraphs (`th`→`t`, `dh`→`d`, …), `w`→`v`, `ee`→`i`, `oo`→`u`,
doubles, and finally `h` entirely.

That is far too aggressive to use as a global key and it is never used as one: it is applied
**within one district** (≤30 DS names) or **within one DS division** (≤120 GN names), and
every match must be 1:1, so a collision is reported rather than silently resolved.

Four census DS divisions still need `DS_ALIAS` by hand. Three are spelling
(`Kandy Four Gravets & Gangawata Korale` → `Gangawata Korale`, `Laggala-Pallegama` →
`Laggala`, `Trincomalee Town and Gravets` → `Town & Gravets`). The fourth is §3 below.

## 3. Kalmunai, where the codes are not comparable at all

COD holds **Kalmunai as one DS division of 58 GN polygons** numbered 005–300. The census
**splits it into two DS divisions** — `Kalmunai` (5224) and `Kalmunai North Sub` (5221), the
Muslim and Tamil divisions — of 29 GN divisions each, **and each restarts its numbering at
005**.

So inside that pool a code means nothing: matching on it gives all 29 low numbers to
whichever half is processed first and orphans the other half entirely, 52,798 people. The
code stage is therefore **skipped wherever two census DS divisions share one COD polygon
set**, and names are used instead.

Names only get half of it, because **COD names the 29 polygons of the Tamil division and
leaves the Muslim division's 29 completely blank** (`adm4_name` is null on 65 polygons
nationally; 29 of them are here). Which is resolved by §4.

## 4. What is left, and where it goes

**13,950 of 14,003 GN divisions matched (99.62%)** — 13,813 by code inside an aligned pair,
137 by name. **53 census GN divisions have no 2022 polygon**: 95,641 people, 0.44%.

Those are placed in **the unmatched remainder of their DS division** — dissolved — rather
than in the whole division. Every polygon that did match belongs to some other GN division,
so whatever is left is where the unmatched people must be, which is a tighter and strictly
more honest area for nothing.

Kalmunai is what forces that rule and shows its size. The 29 unnamed polygons *are* the
Muslim division, so the remainder puts those 52,798 people in the correct half of the town,
where falling back to the whole DS division would have smeared them across both halves —
and Kalmunai is where the Muslim/Tamil boundary is the sharpest religious line in Sri Lanka.
Verified on the rendered map: the Islam dots sit on the coastal strip through Maruthamunai,
Kalmunai, Sainthamaruthu and Karaitivu, with the Hindu dots inland and north.

52 of the 53 orphans get a remainder; **one** falls back to a whole DS division. The 93
unused COD polygons are dropped.

## 5. The checks

- **Every GN division lands on a polygon in its own district — 0 crossings.** This is the
  check that is genuinely independent of a name join: a scrambled join crosses district
  lines and a correct one cannot.
- **Density is sane**: median 580 people/km², p1 14, p99 12,738. A join pairing a Colombo
  ward with a Vavuniya jungle GND shows up here as an absurdity.
- **GN names agree after folding on 13,533 of 13,950 (97.0%)** — reported, not enforced,
  because the residual is romanisation and genuine renaming, which is what stage 1 exists
  to tolerate.
- Structural: 340/340 DS aligned, 14,003 output polygons, no empty geometries.

## 6. The output is keyed by the CENSUS id, deliberately

`lk_gnd.gpkg` carries the census's own seven-digit code (`2224100`) in a column called
`gnd`, **not** COD's `LK2224100`. `sources/lk.py` writes `geo_id` the same way for the same
reason: the two codes are the same shape and disagree, and if both files spelled them
identically somebody would eventually join on one thinking it was the other. `lk_lookup.csv`
holds the correspondence, including which polygon each unit actually got and whether it is
`gnd` or `dsd_rest` level.

## 7. Not done

- A 2024-vintage GN boundary. If DCS ever publishes one, it removes §4 entirely and
  probably §3. Checked 2026-09-04: HDX has nothing newer, and DCS's own site has no GIS
  downloads.
- The 93 unused COD polygons are not reconciled against the 53 orphans. They plainly
  overlap — a GN division that split between 2022 and 2024 appears on both lists — but
  matching them would be guesswork and the remainder rule already puts the people in the
  right area without it.
