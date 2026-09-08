# Cambodia — NIS, General Population Census 2019, Tables 2.5.1 and 2.1.1

Wired 2026-09-07. 15,552,211 people, 25 provinces, 4 drawn categories, **100% drawn**.

| | |
|---|---|
| source | National Institute of Statistics, **General Population Census of Cambodia 2019, National Report on Final Census Results**, **Table 2.5.1** (religion, p.24) and **Table 2.1.1** (population, p.15) of 304 |
| basis | `self_id`, whole census population |
| geography | **25 provinces** — ~622,000 people each |
| categories | **4** plus the universe total; all 4 drawn |
| drawn | **15,552,211 people, 100%** — no `not stated`, no residual beyond `Other` |
| licence | NIS publication, free to download and cite; one GET from `nis.gov.kh`, no wall |

**Cambodia is 97.1% Buddhist and everything the map has to say is in the other 3%** — the
Cham Muslim belt along the Mekong and the Tonle Sap, and the two north-eastern highland
provinces where a fifth of the population answers none of the three named religions.

---

## 1. Why draw a country that is 97% one thing

Because "this country is boring on its own" is not a reason to leave it out, and Cambodia is
the case that makes the point cleanly. Three arguments, in increasing order of how much they
generalise:

* **The 3% is not boring.** Tbong Khmum is 11.8% Muslim and Ratanak Kiri is 23.2% *other
  religion*; those are large, sharp, and unlike anything else on the map.
* **A map of Southeast Asia with a hole in it says something false.** Vietnam, Malaysia,
  Indonesia, the Philippines and Sri Lanka are all drawn. An undrawn Cambodia between
  Vietnam and Thailand does not read as "not yet done", it reads as an absence of people, and
  §6.12 exists because that ambiguity is real.
* **Uniformity is a finding.** A country being 97% one religion is a fact about the world,
  and a map that only draws plural countries is a map that has quietly selected for its own
  conclusion.

Recorded in spec §3.9c, because it is a general rule and not a fact about Cambodia.

## 2. Twenty-five provinces is NIS's ceiling, not a choice

**~622,000 people per unit.** Coarser than most of this map and finer than Zimbabwe's
provinces or Myanmar's states. Spec §3.9b is what makes it drawable: there is no minimum
unit count, and the rule is to take the finest geography a country publishes, say what it
therefore cannot show, and draw it.

The ceiling was checked rather than assumed:

* **Table 2.5.1 is the only religion table in the 304-page final report.** Nothing else in
  it crosses religion with anything.
* **The 2008 census's own priority-table list** (`A4 Population by Religion, 5-year Age Group
  and Sex`, from the tabulation plan) gives religion an age and a sex dimension **and no
  geography at all** — so §12's "check the previous census before concluding the country
  publishes religion with no geography" was applied and came back negative.
* **`microdata.nis.gov.kh`**, the NIS microdata portal that secondary sources cite, now
  **404s on every path including its own root.** It is not a wall, it is gone.
* **Open Development Cambodia's CKAN** has 95 census datasets and **no religion table at any
  geography**.
* **IPUMS International has Cambodia 1998–2019 with district geography**, which would fix
  all of this — and the account is dead ([[reference_ipums_account]]).

So the map cannot show which district, commune or village anyone is in. **Read Cambodia as
composition, never as location**: a cluster of dots means "this province, drawn where
Cambodians live" and nothing about which town.

## 3. The source publishes percentages, and this is what that costs

**Table 2.5.1 gives one decimal place and no absolute figure anywhere.** The counts are
Table 2.5.1's percentage times Table 2.1.1's province population, apportioned by largest
remainder so a province's four categories sum to its published population exactly.

That is arithmetic on two published figures rather than an estimate — §14 rule 1 is about
inventing a magnitude a source does not publish, and both factors are printed — so the rows
stay `measured`. **What it costs, stated rather than buried:**

* **Every cell carries a rounding band of ±0.0005 × the province population.** That is
  ±1,141 people in Phnom Penh, ±448 in Prey Veng, ±21 in Kep.
* **Fifteen of the hundred cells are printed as `0.0` and are drawn as zero.** All fifteen
  are `Other`. Zero is the published figure; it is *not* evidence that nobody is there,
  because `0.0` is anything under 0.05%. Marked, not filled (§3.5).
* **The largest-remainder step decides where a residue of at most three people per province
  lands.** It is the only thing here that is not NIS's.

The national totals it produces agree with NIS's own published national percentages:
15,101,607 Buddhist (97.10% against a printed 97.1), 317,934 Muslim (2.04 against 2.0),
50,338 Christian (0.32 against 0.3), 82,332 Other (0.53 against 0.5).

## 4. The parse, and the only check that crosses tables

Both tables are a **line read** — the text layer emits the row label and then its figures one
per line in column order — so there is no geometry to do. The header block is walked
explicitly rather than skipped by pattern, because in Table 2.1.1 one of the column cells is
`Total` and so is the first row label.

**Every identity available inside Table 2.5.1 survives a consistent column permutation**,
which is §12's Zimbabwe warning: the four categories summing to 100 would still hold if every
column had been read in the same wrong order. Two checks do not:

* **`Male + Female == Total` on all 32 rows of Table 2.1.1.** Catches a column landing in the
  wrong place there.
* **The province percentages, weighted by the province populations, reproducing the national
  percentages** — and separately for Urban and Rural. This pairs a figure from each table, so
  it fails if either was read in the wrong order. Worst category is off by 6,881 people
  against a rounding band of ±15,552.

Everything else is asserted too: 25 provinces, the national universe of 15,552,211, Urban +
Rural = Total, the four regions = Total, the 25 provinces = Total, and the four categories
summing to 100 ± 0.2 on all 28 rows of both years — where the ±0.2 is computed as 4 × 0.05
from the one decimal place rather than chosen to pass.

**The 2008 panel is parsed and gets no cross-table check, and cannot.** Table 2.1.1 publishes
2019 populations only, and Cambodia grew 16.1% between the censuses and unevenly by province,
so weighting 2008 percentages by 2019 populations is off by up to 27,000 people on a
perfectly correct read. That was a failing assertion before it was understood, and the
failure was the check's and not the data's. 2008 keeps its own sum-to-100 identity and is not
drawn.

## 5. What the map shows

**The Cham Muslim belt follows the water.** Tbong Khmum is 11.8% Muslim — 91,667 people, both
the largest share and the largest absolute number in the country — then Kratie 6.6%, Kampong
Chhnang 5.8%, Stung Treng 4.7%, Koh Kong 4.6%, Kampot 2.8%. That is the Mekong upstream of
Phnom Penh and the shore of the Tonle Sap, which is where the Cham settled after the fall of
Champa and where they still fish and farm. Against **0.1% in Svay Rieng and Kampong Speu**, a
hundred kilometres away.

**Read the Muslim figure as a floor, for a specific historical reason.** The Cham were
singled out for destruction under Democratic Kampuchea and are estimated to have lost between
a third and a half of their people between 1975 and 1979; the community's recorded size is
still shaped by that.

**The north-east is the other half of the map, and the census will not name what is there.**
Ratanak Kiri is 23.2% `Other` and Mondul Kiri 21.2%, while fifteen provinces are printed as
0.0% — **the sharpest residual geography on this map**. NIS says in the paragraph above its
own table what the cell mostly is: *"the local religious system of the highland tribal groups
and a few minority religious groups from other countries."* That is the animist tradition of
the Bunong, Tampuan, Jarai, Kreung, Brao and Kavet. It gets no box of its own, so the map can
show that a fifth of two provinces answers none of the three named religions and cannot show
what they answer instead. See `taxonomy/branches.py`'s `other.kh` for why it is not filed on
`indigenous`.

**Christianity is 0.32% and its highest shares are in those same two provinces** — Mondul
Kiri 4.0%, Ratanak Kiri 2.1%, twelve and six times the national rate — which is evangelical
mission among the same highland peoples. Phnom Penh has more Christians in absolute terms
(11,410) and a much lower share. The two categories are working on the same population and
the map shows both at once.

**What four categories cannot show.** No cell for the Buddhist school, so nothing separates
the Mahanikay from the Thommayut. No cell for the branch of Islam, so nothing separates the
mainstream Sunni majority from the **Kan Imam San** of Udong, which is the one distinction a
Cambodian source could usefully draw. No Christian body is named at all.

## 6. The universe excludes Cambodians working abroad

Both tables carry the footnote *"These figures exclude migrants working abroad."* That is a
large population — several hundred thousand in Thailand alone — and it is why the census
total is 15,552,211 rather than the ~16.5M a projection gives. It is also part of why Kontur
reads 1.087× the census nationally.

The four categories sum to 100% of *that* universe and there is no `not stated` cell, so 100%
of what NIS counted is drawn.

## 7. Ethics (§14)

Nothing here engages §14's harder rules. The magnitude is the host state's own published
table, the resolution drawn is the resolution published, nothing is derived from ethnicity,
and no group on the map is at risk from being located to a province.

The one thing worth naming is the `Other` cell, because it is the case where the map is least
able to say what it means. A fifth of two provinces is recorded as practising something the
census will not name, and those provinces are the indigenous highlands — a population with a
live land-rights conflict in Cambodia. The map's honest position is that it draws NIS's
category and NIS's geography and asserts nothing further; the risk of saying more would be
inventing a magnitude, which §14.4 forbids outright.

## 8. Not done

* **District or commune religion.** Not published — §2 above lists the five places checked.
* **The 2008 panel**, which is parsed and reconciled but not drawn: there is no 2008
  population table in this report to turn its percentages into counts, and §3.4's rescale
  would need one.
* **Splitting `Other`.** It is 85% highland indigenous religion by NIS's own account and the
  remaining 15% is not; splitting on that would be §14.4.
