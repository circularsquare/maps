# Estonia — Statistics Estonia, Rahvaloendus 2021

Ingested and drawn 2026-09-03. Rebuild with `python sources/ee.py --fetch`.

The smallest country on the map (1.33M) and worth having for three specific things rather
than for its size.

---

## 1. The file — a proper PxWeb API, the first in the project

No download page, no scraping, no bot protection. Statistics Estonia runs a standard
PxWeb API and it is the easiest acquisition the project has had:

```
POST https://andmed.stat.ee/api/v1/en/stat/rahvaloendus/rel2021/
     rahvastiku-demograafilised-ja-etno-kultuurilised-naitajad/usk/RL21452.px
{"query":[{"code":"Elukoht","selection":{"filter":"all","values":["*"]}},
          {"code":"Usk","selection":{"filter":"all","values":["*"]}}],
 "response":{"format":"json-stat2"}}
```

Two tables are pulled, and the pair is a textbook §3.9 split:

| table | places | categories | used for |
|---|---|---|---|
| **RL21452** | **111** — country, 5 NUTS regions, 15 counties, **79 municipalities, 8 Tallinn city districts** | **21** | every drawn level |
| RL21451 | 4 — country + city / town / rural | **44** | `settlement_type`, mapped but undrawn |

RL21451 is the extreme case of §3.9: the fine category list has **effectively no
geography at all**. It is where Anglicans, Quakers, Mormons, Hare Krishna, Wiccans,
Satanists, Theosophists and Anthroposophists are named. They are mapped in `ee2021.py` so
the national tallies are right and an allocation could be run later, and they are drawn
nowhere.

The English API (`/en/`) is used rather than `/et/` because the category labels are the
mapping keys and English ones are legible in `taxonomy/ee2021.py`.

### json-stat2 is a flat cube, not a table

The response has one `value` array in row-major order over the dimensions in `id` with
sizes in `size`. Reading it as rows will silently transpose the data. `sources/ee.py`
computes strides and indexes properly.

## 2. THE UNIVERSE IS 15 AND OVER

The religion question is asked of **persons aged at least 15**, so the table's own total
is 1,114,030 rather than the 1.33M population, and **no Estonian child is on the map**.

This is spec §3.7 — the same shape as the Philippines excluding institutional residents —
and it matters more here than usual, because every other country on the map draws its
whole population. An Estonian dot and a Polish dot are not the same denominator.

## 3. Everything is rounded to base 10

Every published figure is a multiple of 10. It is Canada's base-5 random rounding (§3.8)
one size larger, and it means **nothing reconciles exactly and nothing is supposed to**:

```
municipality  79 units  1,114,070   drift +40 of ±395 allowed
county        15 units  1,114,040   drift +10 of ±75  allowed
country        1 unit   1,114,030   drift  +0 of ±5   allowed
```

The bands are the worst case rounding can produce, not a fudge — a unit's total is within
±5 of truth, so a sum of n units is within ±5n, and a unit's k answers sum to within ±5k of
its total. `ee.py` also asserts that **every figure is a multiple of 10**, so if Statistics
Estonia ever stops rounding, the bands stop being applied on a false premise.

Within-unit overshoot is real and small: the worst observed is a Tallinn district whose 19
answers sum to 20 more than its own total.

## 4. What Estonia contributes

**Maausk and Taarausk** — `Earth Believer` (3,860) and `Taara Believer` (1,770). The
Estonian native-faith movement, a reconstruction with a real 1920s lineage, and **no other
census on earth enumerates it**. Both → `paganism`, which is what that node is for.

**Old Believers** (2,290) — the Russian communities on the west shore of Lake Peipus.
Third source in a row to need `christianity.orthodox.oldbeliever`, after Poland and
Romania, which is why that node exists.

**The least religious population on the map.** 650,900 of 1,114,030 aged 15+ — 58.4% —
feel no affiliation to any religion, against Czechia's 47% and Poland's 6.9%. The map was
short of that end of the range and now has an anchor for it.

And a fourth thing that is an artefact rather than a contribution: **Orthodoxy is larger
than Lutheranism** (181,770 vs 86,030), which inverts the country's own history and tracks
the Russian-speaking population of Ida-Viru and Tallinn rather than the Estonian one.

## 5. A typo is part of the data

Statistics Estonia spells the same category **`Taara Beliver`** in RL21452 and
**`Taara Believer`** in RL21451. Both spellings are mapped in `ee2021.py`.

Correcting it in `ee.py` was the alternative and would have been worse: the normalised CSV
carries the source's own strings verbatim (spec §2.4), so a silent repair there would make
the CSV disagree with the source it claims to reproduce.

## 6. The non-response is split three ways, and only one of them is drawn

| | people | share of 15+ |
|---|---|---|
| total 15+ | 1,114,030 | 100% |
| feels an affiliation | 321,340 | 28.8% |
| **feels none** | **650,900** | **58.4%** — drawn, as `unaffiliated` |
| refused to answer | 126,500 | 11.4% — excluded |
| affiliation unknown | 15,280 | 1.4% — excluded |

Inside "feels an affiliation" there is a further `Religion unknown` (1,530) — people who
report an affiliation without naming it. That IS drawn, as `other.ee`, and is a different
answer from `Religious affiliation unknown`, which is a coverage residual. Czechia has the
same pair and the same treatment.

## 7. What is drawn

21 categories → 15 taxonomy nodes → **958 dots and 2 rings** at 1:1,000.

Under a thousand dots is the smallest country on the map by an order of magnitude, and
0.80% of the drawn population falls under one dot nationally — the highest such share
anywhere, simply because the country is small relative to the global dot value.

One new node: `other.ee`.

## 8. Not done

- **The 44-category list is not allocated down.** Estonia's Anglicans, Quakers, Mormons,
  Wiccans and Theosophists exist in `data/normalized/ee.csv` at `settlement_type` level and
  are drawn nowhere. `allocate.py --fine municipality --coarse settlement_type` is not
  obviously valid, because settlement type is not a geographic partition a municipality
  nests inside — a municipality contains both city and rural settlements. Doing it properly
  needs the county-level table (RL0453's 2021 equivalent) as the coarse tier.
- The 2000 and 2011 censuses are in the same PxWeb tree (`RL229`, `RL0451`) and would give
  Estonia a time series, which nothing else on the map has.
