# Bangladesh — 2011 Population and Housing Census, religion by upazila

`sources/bd.py` rebuilds `data/normalized/bd.csv` from `data/raw/bd/`. `data/` is gitignored,
so this file is the version-controlled record.

Wired 2026-09-06. Picked out of `sources.md` §11j, which audited the four countries left
unbuilt in the USCB series that §11h found. **The fourth-largest country on this map and the
last nine-figure source anywhere in `sources.md`.**

---

## 1. The route

Two GETs, no account, no key, no portal.

| | |
|---|---|
| publisher | **U.S. Census Bureau**, not the Bangladesh Bureau of Statistics |
| platform | HDX (`data.humdata.org`), CKAN, open API |
| dataset | `bangladesh-subnational-boundaries-and-tabular-data` |
| workbook | `bangladesh_uscb_202107.xlsx`, 4.8 MB — sheet **`Religion and Ethnicity`** |
| geodatabase | `bangladesh.gdb.zip`, 69 MB — layers `BD_RELIGION_AND_ETHNICITY_2011census_uscb_202107` and `BD_GEOG_ADM3_2011_uscb_202107` |
| original source | BBS, *Population and Housing Census 2011*, via the office's own REDATAM server (`redatam.bbs.gov.bd`, cited in the Metadata sheet, **now dead**) |

The `Metadata` sheet names the origin, the accession date (30 April 2021) and the unit
counts. §11h's rule — **read the Metadata sheet before the data sheet** — earned its place on
Jamaica and is cheap here, but it is what confirms the vintage and the tier.

`python sources/bd.py --fetch` does the download, checks magic bytes (§5a) and unzips.

---

## 2. What is in it

**617 rows over four levels, and every one of them is a complete partition.**

| level | units | note |
|---|---|---|
| 0 country | 1 | `BGD_00` |
| 1 division | 8 | Barishal, Chattogram, Dhaka, Khulna, Mymensingh, Rajshahi, Rangpur, Sylhet |
| 2 zila | 64 | |
| **3 upazila** | **544** | **the drawn tier** — 483 upazilas + 61 thanas |

544 units for 144,043,696 people is **264,786 each**. Finer than Pakistan's districts
(1.54M), Kenya's counties (1.01M) and the Philippines' provinces; coarser than Ethiopia's
woredas (99,900).

**Five categories, and that is the whole question.**

| cell | people | share | node |
|---|---|---|---|
| `RLG_MSL` Muslim | 130,204,817 | 90.39% | `islam` |
| `RLG_HIN` Hindu | 12,299,981 | 8.54% | `hinduism` |
| `RLG_BUD` Buddhist | 889,721 | 0.62% | `buddhism` |
| `RLG_CHR` Christian | 447,010 | 0.31% | `christianity` |
| `RLG_OTH` Other religion | 202,167 | 0.14% | `other.bd` |

`taxonomy/bd2011.py` has the mapping and the reasoning. Every cell resolves to a parent or a
root; nothing needed a new node except the residual.

The same sheet also carries **27 named ethnic groups** (`ETH_*`, 1,586,183 people — Chakma,
Marma, Tripura, Garo, Santal and 22 more), keyed identically and not read. Worth knowing it
is there if the Hill Tracts ever want a second look.

---

## 3. The reconciliation, which is the strongest on this map

`sources/bd.py` asserts all of the following and refuses to write on any failure.

1. **The five categories sum to each row's own published `RLG_TPOP` — all 617 rows, exact.**
   This is the check that replaces Ethiopia's male + female == both-sexes identity, and it is
   a better one: `RLG_TPOP` is published *beside* the categories rather than derived from
   them, so a transcription slip in any single cell breaks it.
2. **The national row equals 144,043,696**, BBS's published 2011 census population — the one
   figure here that comes from outside the USCB file, and so the only genuinely independent
   anchor.
3. **Every level partitions the country**, category by category: 8 divisions, 64 zilas and
   544 upazilas each reproduce all five national figures exactly.
4. **544 upazila rows, 544 distinct `GEO_MATCH`, none with zero population.**
5. **The `.gdb` religion layer agrees with the `.xlsx` on all 3,085 cells.** Not an
   independent check — same publisher, same release — but it catches a misread of either.
6. **Zero negatives and zero nulls** — see §5.
7. **`USCBCMNT` is empty on all 617 rows** — see §4.

So **100% of the tabulation is drawn**, and unusually there is nothing to subtract first.

---

## 4. The vintages match, and the empty column is the evidence

Ethiopia (§9u) is a 2007 census that USCB **re-cut onto 2021 woredas**: 418 of its units
carry a `USCBCMNT` reading *"Formed from part of <census-era unit>"*, 70 census-era woredas
are split across two to four modern ones, and that per-unit split is USCB's own work and is
not independently verifiable.

**Bangladesh has none of that.** The layers are `BD_GEOG_ADM3_2011` against
`BD_RELIGION_AND_ETHNICITY_2011census` — both 2011 — so spec §8.1 is satisfied outright and
there is nothing to reconcile.

**The tell is that `USCBCMNT` is populated on zero rows.** `bd.py` asserts that rather than
skipping the column for being blank, because the general point is worth keeping:

> **An empty lineage column is evidence of a matched vintage.** In this series it is the
> cheapest available check on whether the boundaries you are joining to are the ones the
> counts were published on — and a blank column reads as "nothing here" rather than as the
> finding it is.

---

## 5. Neither null convention appears, and that is the finding about the series

Three countries from one publisher, three different conventions:

| | null convention |
|---|---|
| Ethiopia (§9u) | **`-999`**, which parses as a number. 4 woredas × 6 categories; summed naively it removes 23,976 people, 0.03%, which reads as rounding. |
| Pakistan (§9t) | **real nulls.** 20 districts (Azad Kashmir, Gilgit-Baltistan) that PBS did not publish. |
| **Bangladesh** | **neither. Zero negatives, zero nulls, no missing units.** |

So **the sentinel convention is per FILE, not per publisher**, and the reflex of masking
`< 0` defensively everywhere in the series would be wrong here in a quiet way — it would pass
on a file that had changed under you.

`bd.py` therefore **asserts both counts at zero** instead of masking. That is a stronger
check than Ethiopia's precisely because it can be: where a file is clean, assert that it is
clean, so a future re-release that adopts a sentinel fails loudly rather than silently
subtracting people.

---

## 6. What the country is worth, which is entirely its geography

The question is five cells — shallower than Ethiopia's six, level with Indonesia's official
list, deeper only than Germany's three. §3.9's trade taken hard to the geography end, like
Sri Lanka. What it buys:

### The largest Hindu population outside India

**12,299,981 people, 8.54%** — larger than Nepal's, about the size of Ohio, and the single
strongest reason to draw the country. It is concentrated, and in three separate belts:

| upazila | division | Hindu |
|---|---|---|
| Dacope | Khulna | **56.5%** |
| Kotalipara | Dhaka | 49.9% |
| Sulla | Sylhet | 47.0% |
| Kaharole | Rangpur | 44.1% |
| Agailjhara | Barishal | 42.3% |
| Sreemangal | Sylhet | 40.6% |

The southwest of Khulna around the Sundarbans, the tea districts of Sylhet, and the northwest
around Dinajpur and Thakurgaon. **Read 2011 as a moment in a long decline**: the Hindu share
of this territory was about 22% at partition and roughly 13.5% in 1974.

### The Chittagong Hill Tracts

A Theravada Buddhist and tribal-Christian country inside a 90% Muslim one, and nothing else on
this map looks like it.

| upazila | Buddhist | | upazila | Christian |
|---|---|---|---|---|
| Juraichhari | **94.6%** | | Ruma | **38.2%** |
| Naniarchar | 83.4% | | Thanchi | 36.4% |
| Lakshmichhari | 79.4% | | Rowangchhari | 16.7% |
| Belai Chhari | 77.5% | | Rajasthali | 12.3% |
| Barkal | 75.1% | | Belai Chhari | 9.8% |
| Baghaichhari | 70.0% | | Alikadam | 7.4% |

Six upazilas are majority Buddhist. Against a national Christian share of **0.31%**, Ruma is
38.2% — twentieth-century mission ground among the Bawm, Mru and Khumi.

### And the residual has a geography, which means it is a missing category

`Other religion` is 0.14% nationally and **15.3% in Ruma**, 7.8% in Thanchi, 5.1% in
Rowangchhari — sitting *beside* the Christian and Buddhist peaks rather than instead of them.
A five-box question has nowhere to put Mru, Khyang or Bawm traditional practice.

This is §9u's Bore-woreda pattern in a second country: **a residual with a sharp geography is
a missing category, not a mixture.** Patnitala in Rajshahi at 3.9% is the Santal and Oraon
version of the same thing.

---

## 7. What the census cannot show

- **No Ahmadi cell.** Perhaps 100,000 people, who have had mosques sealed and communities
  attacked. Pakistan's census prints `Qadiani/Ahmadi` as a category and `pk2017.py` has an
  `islam.ahmadiyya` node for it; Bangladesh's does not, so they are inside the 130 million
  and cannot be brought out.
- **No Christian denomination.** Roughly two-thirds of Bangladeshi Christians are Catholic —
  the Portuguese-descended communities of Dhaka and Chattogram, the Holy Cross missions, and
  the Garo and Santal converts of Mymensingh — with Baptists (Carey's Serampore mission worked
  this ground) and a growing Pentecostal share. One undivided cell.
- **No school or branch for Islam.** Overwhelmingly Sunni Hanafi with a deep Sufi layer (the
  Chishti and Qadiri orders, the Maijbhandari tariqa of Chattogram). `islam.sunni` would be an
  inference rather than a reading.
- **No non-response cell.** All five cells are religions and they sum to the census
  population. As in Ethiopia and Pakistan, this does **not** mean nobody refused — it means
  BBS distributed or never published a refusal cell, and nothing here can undo that.
  `note_public` says so rather than letting the map imply full coverage of an answered
  question.

---

## 8. The 2022 census exists and is not the cheap route

BBS ran a Population and Housing Census in **2022** — about 165 million — and it asked
religion again. It is not a quick upgrade:

- `redatam.bbs.gov.bd`, the host USCB itself cites, **times out entirely** (WinError 10060,
  connection timeout, no response).
- `bbs.gov.bd` answers 200 but has been **rebuilt onto a new CMS** whose publication pages are
  opaque ObjectId slugs (`/pages/static-pages/6922db5c933eb65569e09a5b`). No census or
  publication index is reachable from the homepage; `bbs.portal.gov.bd` is retired and
  `microdata.bbs.gov.bd` does not resolve.
- `data.gov.bd` has no CKAN API at the documented path.

**Build 2011 from the USCB file and treat 2022 as a later re-basing.** §3.4 and §9l already
have the machinery for exactly that move, and Brazil proves it is cheaper as a second pass
than as a first one. The shares are unlikely to have moved much; the magnitudes have.

> **And a caution for whoever tries.** §11j's own Vietnam note was wrong for a day because
> `gso.gov.vn` timed out while `nso.gov.vn` served the same files — the office had renamed.
> Before recording BBS as unreachable, enumerate its other names.

---

## 9. §14

Bangladesh's Hindus are a minority that has faced communal violence, so §14 is engaged and was
considered rather than assumed away. It does not block:

- **The tier is BBS's own.** §14.4's *"for a persecuted group, no resolution finer than the
  state's own publication"* is satisfied **by construction** — the upazila table is what BBS
  published, transcribed. Nothing here is finer than the state's own release, and nothing is
  modelled.
- **§14.2's reflect-vs-reveal test passes.** The Hindu geography of Khulna and Sylhet is not a
  secret, is not small or dispersed, and is in every account of the country.
- **It is a milder case than Pakistan**, which was decided on 2026-09-06 (§9t) in favour of
  drawing at the state's published tier, and than India, which is drawn.

Raised here so the reasoning is on the record rather than re-derived. Anita's call stands
above it either way (§14's opening line).
