# Indonesia — BPS, Sensus Penduduk 2010

`sources/id.py` → `data/normalized/id.csv`.
Boundaries: `sources/id_geo.py`. Taxonomy: `taxonomy/id2010.py`.

**237.1M people, 492 kabupaten/kota, nine categories, and no account of any kind.** The
second largest country on this map after India, and the cheapest large one ever added.

## 1. Where it comes from, and why it was nearly missed

`sources.md` §11d recorded Indonesia as blocked on a free BPS developer key. It is not
blocked on anything. The relevant hosts behave completely differently:

| host | what it is | to a script |
|---|---|---|
| `www.bps.go.id` | the public web UI | **Cloudflare 403** |
| `webapi.bps.go.id` | the documented API | **`{"status":"Error","message":"Parameter Key is Missing."}`** |
| `sensus.bps.go.id` | the census microsite | **open, no key, no session** |

That is §9q's Lithuania pattern for the third time — the wall is real and is not in front of
the data — and it is why two separate passes concluded Indonesia needed an account.

The endpoint is

    https://sensus.bps.go.id/topik/tabular/sp2010/12/{wid}/{format}

`12` is *Penduduk Menurut Wilayah dan Agama yang Dianut*, found from
`/topik/dataset/sp2010/1`. **The trailing segment is a FORMAT, not a geography**: 0, 1 and 5
are HTML, **2 is a PDF** and **3 is JSON**. Nothing says so, and the sizes mislead — the
national table is 133 KB at format 2 and 5.4 MB at format 3, which reads as a finer
geography and is the same numbers serialised differently.

## 2. The wid space, which is the whole difficulty

`wid` selects the unit whose *children* are returned. It is ordered by BPS code and has
three shapes of slot:

* **normal** — a level-2 row for the regency plus its kecamatan at level 3;
* **gap** — kecamatan at level 3 and **no level-2 row**;
* **hole** — an empty response, no rows at all.

Layout: `0`/`1` → the 34 provinces; `2`–`35` → one province each, returning its regencies;
`36` upward → one regency each, returning its kecamatan. `wid=25` is empty and is asserted
to be: Kalimantan Utara, created 2012, absent from a 2010 census.

**The URL's `wid` is not the payload's `id_wilayah`, and the two spaces overlap.** The
national response gives Aceh `id_wilayah: "1675"`; requesting `wid=1675` returns *Kabupaten
Merangin, in Jambi* — 200, well-formed, a real unit, the wrong one — and 1674 and 1676
return it too. Every number is genuine and only the unit is wrong, which is the one error no
downstream reconciliation can see. Identity is therefore always read from `kode_wilayah` in
the response.

**And past the end of the space, wids do not 404 — they return other provinces' data.**
wid 609, 705, 753 and 800 all answer with Aceh codes. A binary search whose upper bound
lands there reads a small code, concludes the target is further right, and walks off the
end; an early version of the gap stage "found" province 73 at wid=800 holding Aceh's 1117.
Blocks are now located by walking forward from the nearest cached anchor instead.

## 3. The province listings are incomplete, and only arithmetic says so

**Ten of the 33 province responses omit some of their own regencies — 16 units, 2,674,311
people — with no gap, no marker and no error.** Sumatera Utara returns 31 consecutive-looking
rows and is missing `1273` Pematangsiantar and `1277` Padangsidimpuan. Every row present is
correct and the province's own row is correct, so nothing but the parent/child sum can see
it.

What proves the data is sound and only the listing is short: **the 33 province rows sum to
SP2010's published 237,641,326 exactly.**

The omitted units sit at their own wid in the *gap* shape — kecamatan present, own row
absent — so each is rebuilt by summing its children, and checked against the province
shortfall to the person:

| province | omitted | recovered from |
|---|---|---|
| 12 Sumatera Utara | 1273 Pematangsiantar (234,698), 1277 Padangsidimpuan (191,531) | wid 86, 90 |
| 19 Kep. Bangka Belitung | 1971 Pangkalpinang | wid 182 |
| 51 Bali | 5107 Karangasem | wid 315 |
| 61 Kalimantan Barat | 6104 Pontianak (now Mempawah) | wid 353 |
| 63 Kalimantan Selatan | 6302 Kotabaru | wid 379 |
| 73 Sulawesi Selatan | 7301 Kepulauan Selayar, 7372 Parepare | wid 434, 456 |
| 74 Sulawesi Tenggara | 7472 Baubau | wid 474 |
| 76 Sulawesi Barat | 7605 Mamuju Utara | wid 485 |
| 81 Maluku | 8101 Maluku Tenggara Barat | wid 487 |
| 64 Kalimantan Timur | **5 units, not recoverable — see §4** | — |

234,698 + 191,531 = 426,229, which is Sumatera Utara's gap exactly.

**But a kecamatan sum has the same disease as a province listing, and two of them do.**
`6104` sums to 193,661 against a true 234,021 and `6302` to 281,162 against 290,142: both
UNDERSTATE, which is worse than missing, because an understated unit still draws and draws
wrong. So where a province is missing exactly one regency the **province residual** is used
instead — the province row minus its listed regencies, which is exact because the province
rows are exact. Kecamatan sums are kept only where several units are missing at once, and
are then checked against the residual so a shortfall stops the run.

## 4. Kalimantan Utara, and why it needed no other source

**Kalimantan Timur's listing is short by 524,656 and those people are not served anywhere on
this host.** Its province row is the *2010* province, which includes the five regencies that
became **Kalimantan Utara** in 2012 — Malinau, Bulungan, Tana Tidung, Nunukan and Kota
Tarakan. Its regency listing has only the nine that remain. The five are not under Kalimantan
Utara either: wid=25 is empty, and the regency slots where their codes sort (wids 397 and
401–405) are **holes**. BPS is publishing a 2010 census through a post-2012 geography and
these units fell between the two.

That was first recorded here as an unrecoverable 0.22%, which left a conspicuous hole in the
north of Borneo. **It is recoverable, and the fix is arithmetic rather than a new source.**
The residual — province minus its nine listed regencies, *per category* — is:

| | | | |
|---|---|---|---|
| Islam | 378,478 | 72.14% | |
| Kristen | 109,358 | 20.84% | |
| Katolik | 29,366 | 5.60% | |
| Hindu / Budha / Khong Hu Chu / Lainnya | 4,367 | 0.83% | |
| non-response | 3,087 | 0.59% | |
| **Total** | **524,656** | | matches Kaltara's documented 2010 population exactly |

Categories sum to the Total to the person, every category is non-negative, and the
composition is right for the place — Muslim coastal Tarakan and Nunukan, Dayak Christian
interior around Malinau.

**Why it can be drawn.** The five regencies were carved wholly out of Kalimantan Timur and
nothing else joined them, so the residual is not a scattering of unrelated places: it is
exactly the territory of the modern province, and `id_geo.py` dissolves their five COD
polygons into one. It is `_merge_gaps`'s residual rule applied one level up — the same
technique that recovered sixteen regencies, where the missing children happen to be a
province's worth rather than a single regency's.

**What it costs.** This unit is far coarser than the rest of the map: one polygon for 524,656
people where Indonesia is otherwise drawn at kecamatan. It carries `geo_level`
`province_residual` for that reason, and it is the only one. **The drawn population is
237,641,326 — 100.00% of SP2010.**

## 5. Categories — a legal list, not a classification

`Islam`, `Kristen`, `Katolik`, `Hindu`, `Budha`, `Khong Hu Chu`, `Lainnya`,
`Tidak Terjawab`, `Tidak Ditanyakan`.

Indonesia recognises six religions and the census asks which one. That is Germany's §3.9a
shape, and three consequences follow (all in `taxonomy/id2010.py`):

* **`Kristen` means Protestant, not Christian** — a peer of `Katolik`, not its parent.
  Reading it the plausible way double counts 6.8M people. Hungary's `RE_CA` again.
* **There is no `no religion` cell**, so Indonesia draws with an empty unaffiliated family.
  §6.12's case: not measured at zero, never offered.
* **`Lainnya` is a floor, badly.** In 2010 Aliran Kepercayaan had no standing on the form
  (registration came with the 2017 Constitutional Court ruling), so adherents recorded one
  of the six instead; Kaharingan was administratively counted as Hinduism, which is most of
  why Central Kalimantan draws Hindu.

`Tidak Terjawab` (not answered, 139,128) and `Tidak Ditanyakan` (not asked, 754,485) are
separate and stay separate — Serbia's pair in §9p. The second is five times the first.

## 6. Boundaries

COD-AB Indonesia, ADM2. **The join is by code and it is exact**: `adm2_pcode` is `ID` plus
BPS's own 4-digit kode wilayah, the same identifier the census payload carries. Not one name
has to be matched.

Two reader traps, both silent (see `id_geo.py`):

* **`engine="fiona"` reads 522 features; `engine="pyogrio"` reads ZERO** from the same
  geodatabase, no exception either way, and pyogrio is geopandas' default when installed.
  This is §12's Chile symptom and it amends that rule: the format was never the problem.
* Reading the same file through `zip://` returns zero under **both** engines. Extract first.

**§8.1 vintage.** COD is `valid_on 2020-04-01` with 522 ADM2 against SP2010's 497. Every
2010 code matches a polygon, so the join looks perfect — but a dozen-odd regencies were
split after 2010, so their 2020 polygons are too small and the territory that left them has
no census row to paint it. `id_geo.py` names every one on each run.

**The exact fix is the kecamatan layer.** Sub-district codes are `regency(4) + kecamatan(3)`
and a new regency is carved out of whole kecamatan, so dissolving COD's ADM3 by the first
four digits of the *2010* kecamatan code reconstructs the 2010 regency from BPS's own
geography rather than from a guess about parentage. That needs the kecamatan pull.

## 7. The kecamatan tier — built, and it does NOT simply replace the regency one

`--fetch-all` walks the whole regency wid space (wid 36..549, 492 regencies, 33 provinces)
and takes the level-3 rows the slots already carry. It ends by **monotonicity, not by a
count**: the first code that sorts below the highest one seen is past the end of the ordered
space, and wid 550 answering with Aceh's 1101 after Papua's 9471 is what stops it.

**6,357 kecamatan, 231,829,784 people, 97.55% of SP2010**, and every one of the 6,357 joins
to a COD ADM3 polygon by code — a second exact join.

**But §3's disease is much worse one level down.** 88 of the 492 regencies — 18% of them —
have an incomplete kecamatan listing, 5,286,886 people in total, and the worst are not
marginal: `7401` Kolaka is missing 172,986 of its 255,712 people, 68% of the unit. Drawing
those kecamatan as counts would understate specific places by two thirds while every
national figure still looked reasonable. **That is the §14.4 line — never estimate a
magnitude a source does not publish — and it is why the finer tier is not automatically the
one to draw.**

The three options, and the one taken:

| | what it draws | cost |
|---|---|---|
| A. kecamatan everywhere | 6,357 units, 97.55% | 88 units understated, one by 68% |
| **B. kecamatan where the listing reconciles, regency where it does not** | **5,122 + 89 = 5,211 units, 99.78%, every row still `measured`** | two `geo_level`s |
| C. regency counts, kecamatan placement | 492 units of counts | no kecamatan religion drawn at all |

**B is built.** `write()` classifies each regency and emits four `geo_level` values, of which
only two are drawn:

    kecamatan          the finest measured unit                     DRAWN
    regency            a regency whose kecamatan are incomplete     DRAWN in their place
    regency_covered    a regency its kecamatan already cover        record only
    kecamatan_partial  kecamatan of an incomplete regency           record only

The drawn tier is `kecamatan` + `regency`: disjoint, covering the country once, **236,223,057
people on 5,211 units**, and nothing allocated. `countries.py` filters on exactly those two
levels and asserts the unit count; `id_geo.py` asserts that no regency is drawn while its own
kecamatan are also drawn, because that would double the people there and nothing downstream
would notice.

**And the completeness test is PER CATEGORY, not on the total.** Nduga (`9429`) in Papua
publishes eight kecamatan carrying a `Total` row and **no religion categories at all**. The
totals reconcile to the person, so a total-only test calls the unit complete, promotes it,
and draws its 79,053 Kristen as 79,053 people with no religion — a unit on the map with
nothing in it. One regency in 492, invisible in every national figure. It is drawn as a
regency instead.

## 8. Placement — Kontur, and what it does not fix

`sources/id_grid.py` keys 874,919 Kontur 400 m hexes to the drawn units: **784,489 cells
over all 5,211**. Dots land where Indonesians live rather than evenly across a polygon.

Two things made this necessary even at kecamatan resolution: the 89 whole regencies (large
and nearly empty Papuan and Kalimantan units — Kenya's case), and the **dense urban
kecamatan**, which is the one you see. Cengkareng is 513,920 people; an even wash made
Jakarta and Surabaya read as flat-shaded tiles with administrative edges.

**It does not fix the segregation, and nothing on this source will.** BPS publishes
religion at kecamatan and nothing below — the wid space's second pass is a repeat of the
regency tier, not a kelurahan tier (§2) — so inside one kecamatan every religion is placed
on the SAME population weight. Kelapa Gading's Chinese-Indonesian character stays averaged
across 154,692 people. Weighting religions differently within a unit would invent a
magnitude the source does not publish (§14.4). The grid refines where the people are, not
who they are.

Four Yahukimo kecamatan have no populated Kontur cell and fall back to their own polygon —
a unit missing from the placement layer draws nothing while nothing errors. 4.03% of
Kontur's population falls outside every drawn unit and is dropped.

## 9. Boundaries at the finer tier, and what is still unpainted

All 6,357 census kecamatan match a COD ADM3 polygon. **712 of COD's 7,069 polygons have no
census row — 10.1% of polygons and 13.8% of the land area.** They are post-2010 sub-districts
plus the ones missing from the 88 incomplete listings, and 53 of them are Kalimantan Utara,
which has no census data on this route at all (§4). Much of the rest is thinly populated —
Papua, Maluku, interior Kalimantan — so the area figure overstates the population effect,
but it is the honest number and it is printed on every run.

Note that **the 2010-regency reconstruction described in §6 turned out not to be available
the easy way**: a regency created after 2010 gets entirely NEW kecamatan codes with its own
prefix (Mahakam Hulu's five are `6411010`..`6411050`, not Kutai Barat's `6402xxx`), and COD's
ADM3 codes agree with their own ADM2 for all 7,069. So dissolving ADM3 by the 2010 code
rebuilds only the part of a 2010 regency that stayed put. Reconstructing the rest needs the
post-2010 units assigned to a 2010 parent by geometry, which is a guess this file has not
made.
