# Unspecified Christian splits: West Africa (Nigeria, Senegal to Liberia, Côte d'Ivoire)

**BUILT 2026-10-16** for ng, lr, sl, gn, sn, gm and gw on Anita's calls (Nigeria three ways by
zone with Baptists in the Protestant group; Liberia as proposed; Guinea's Protestants on
`.protestant`; Côte d'Ivoire unchanged): `sources/wa_christian.py`, sources.md
`wa-2026-10-16`. What follows is the research as it was written before the build.

Research only, 2026-10-16. Nothing in taxonomy/, sources/, countries.py or index.html was touched.
Scratch scripts: `%TEMP%\claude\...\scratchpad\westafrica_scripts\` (ab_christian.py extracts the
West African rows from the six merged Afrobarometer .sav files already in `data/raw/afrobarometer/`;
gfs_ng.py, ng_compare.py, ng_apply.py, band.py, splithalf.py do the tables below). Downloads (DHS final
reports, Guinea-Bissau 1991 census) are in `scratchpad\westafrica\`.

## Ranking by visual payoff

| rank | country | bucket now | people | best source | proposal | confidence |
|---|---|---|---|---|---|---|
| 1 | Nigeria | `christianity` (after Catholic, Anglican) | 80.25M, 37.0% | GFS 2023 by zone, Afrobarometer as zone witness | Pentecostal / Protestant / other, six zones | medium-high |
| 2 | Liberia | `christianity`, whole of it | 4.46M, 84.9% | Afrobarometer R4-R9 named answers | 7 nodes nationally; Lutheran, Baptist, Catholic by county | medium |
| 3 | Sierra Leone | `christianity` | 1.55M, 21.9% | **the 2015 census's own national split** | 6 nodes, flat national shares | high |
| 4 | Guinea | `christianity` | 0.71M, 6.8% | Afrobarometer + Pew 2010 agree | Catholic 65 / Evangelical 30 / other 5, flat | medium |
| 5 | Côte d'Ivoire | `christianity.evangelical` | 5.45M, 18.6% | none splits it | not a split; a node-choice question for Anita | n/a |
| 6 | Guinea-Bissau | `christianity` | 0.32M, 22% | **1991 census national split** (89% Catholic) | Catholic / other, flat | medium-low (dated) |
| 7 | Senegal | `christianity` | 0.31M (1988), 4.5% | 1988 census Diourbel + Pew + Afrobarometer | Catholic 92 / other 8, flat | high |
| 8 | The Gambia | `christianity` | 70k, 3.75% | Afrobarometer (n=54), Pew, Church claim | Catholic ~75 / Protestant ~25, flat | low, tiny |

Neighbour palettes the splits should land on (from taxonomy/*.py): Ghana catholic / protestant /
pentecostal / other; Côte d'Ivoire catholic / protestant (Méthodiste) / harrist / evangelical / other;
Mali, Burkina Faso, Chad catholic / protestant (/ other); Benin catholic / methodist / protestant /
africaninstituted / other; Togo catholic / pentecostal / AoG / presbyterian / baptist / methodist / ...;
Cameroon catholic / presbyterian / baptist / bare christianity; Niger bare christianity (not in this
region; same problem).

One cross-check runs through the whole band: the Pew-Templeton 2010 country table
(https://www.pewresearch.org/religion/2011/12/19/table-christian-population-as-percentages-of-total-population-by-country/)
gives Catholics as a share of Christians of Senegal 95%, Gambia 66%, Guinea-Bissau 89%, Guinea 68%,
Sierra Leone 34%, Liberia 8%, Nigeria 25%. It is an estimate (World Christian Database lineage), so it is
a witness only, never a level (§3.1). Catholic Church claims for 2023 (gcatholic.org country pages)
are listed per country as a second witness; they count the baptised.

---

## 1. Nigeria: `christianity`, 80,254,467 people (37.02%)

**What is in the bucket now** (`taxonomy/ng2022.py`, `data/normalized/ng.csv` rows `Christian`):
every Christian but the Catholics (10.2%, NDHS-levelled) and Anglicans (4.2%, GFS-levelled). The
Christian/Muslim balance is the Afrobarometer's per state on COD-PS 2022; all rows already `modelled`.

**Sources examined**

- **Global Flourishing Study wave 1 (2023), already on disk** (`data/raw/gfs/`, CC BY, no account).
  Nigeria 6,827 respondents, 3,849 Christians, `REL3_Y1` denomination, `REGION1_Y1` state. **Only 1.1%
  of Christians name no church.** No respondents in Adamawa, Borno, Taraba, Yobe.
- **Afrobarometer R4-R9** (on disk): 1,307 named non-Catholic non-Anglican Christians, but `Christian
  only` runs 19-47% by round and the named Pentecostal share swings 2.9 / 12.1 / 5.8% of respondents
  across R4-R6, so it cannot level anything.
- NDHS 2008-2024 final reports: Catholic / Other Christian only (already used).
- Pew 2010 Tolerance and Tension: not opened (dataset needs a Pew login; report appendix national).

**The finding: at zone level the two surveys agree.** `sources/ng.md` §"Re-tested" found Pentecostal
failing at state level (two-survey rank +0.20, p 0.15). Pooled to the six geopolitical zones, each
zone's share of the bucket (named, non-Catholic, non-Anglican Christians):

| zone | GFS n | GFS Pentecostal | AB Pentecostal | GFS Protestant* | AB Protestant* | GFS other** |
|---|---:|---:|---:|---:|---:|---:|
| North Central | 328 | 51.9 | 31.4 | 45.4 | 67.4 | 2.6 |
| North East | 136 | 12.2 | 12.5 | 87.1 | 82.9 | 0.7 |
| North West | 134 | 36.1 | 23.7 | 61.7 | 68.0 | 2.2 |
| South East | 320 | 71.0 | 55.8 | 23.7 | 37.6 | 5.3 |
| South South | 769 | 66.2 | 55.1 | 23.9 | 35.1 | 9.9 |
| South West | 794 | 70.4 | 40.4 | 24.7 | 46.6 | 4.8 |

\* Protestant = Baptist + Methodist + Presbyterian + Lutheran + GFS `Orthodox` + `Independent/Evangelical`
(AB: Orthodox, Evangelical, Lutheran, Church of Christ, Independent). \** JW, Adventist, LDS, AIC, other.

Zone rank agreement GFS vs Afrobarometer (exact permutation over all 720 orderings):
**Pentecostal +0.943, p 0.008; Protestant group +0.829, p 0.029**; Baptist alone +0.771 (p 0.051);
Methodist +0.600 (p 0.12) and Presbyterian -0.200 fail. So the three-way split is checkable at zone
level and the named mainline churches individually are not.

**About GFS `Orthodox` (5.8% of Christians).** It is 87% of Plateau's and 72% of Bauchi's Christians.
In Nigerian usage "orthodox churches" means the old mission churches (Anglican, Methodist, Baptist,
and in the north ECWA, COCIN, LCCN, EYN) as against Pentecostal and Aladura churches, so these answers
very likely are those bodies, and they belong in the Protestant group, not with Eastern Orthodoxy.

**Proposal.** Split each state's `Christian` row by its zone's GFS shares (levels from GFS, the
2023 survey that names 99% of Christians; the Afrobarometer as the rank witness, the same roles GFS and
AB have in the Anglican build):

| node | people | % of Nigeria | % of bucket |
|---|---:|---:|---:|
| `christianity.pentecostal` | 48.4M | 22.3 | 60.3 |
| `christianity.protestant` | 27.4M | 12.7 | 34.2 |
| `christianity.other` (or left on `christianity`) | 4.4M | 2.1 | 5.5 |

Every derived row `modelled`, `COLUMNS` mapping all three back to `christianity` (the column the build
already measured), so the toggle returns today's picture. Christian totals per state unchanged.

**What would contradict it:** the AB zone ranks (passes now, a build assert like `CHURCH_P_MAX`);
GFS wave 2 when its Nigeria rows are usable (the 1.1% unnamed should stay small); the level gap: AB
named gives Pentecostal 45% of the bucket against GFS 62%, so assert the drawn level sits inside that
range. Pew 2006 *Spirit and Power* is a further national witness if anyone opens it.

**Caveats.**
- Zone shares make a step at zone borders inside Nigeria (e.g. Kwara/Kogi vs Ekiti/Ondo). The steps sit
  inside one colour band (Pentecostal heavy south, Protestant heavy Middle Belt and north), which is the
  real pattern, but the edges are administrative.
- GFS North East is Bauchi and Gombe only; Adamawa and Taraba (Lutheran LCCN, EYN Brethren, large
  Catholic share already drawn) take the zone figure. The AB North East (which does include Adamawa)
  agrees on Protestant ~83-87%.
- Aladura churches (CAC, C&S, Celestial) have no clear box: GFS `AIC` is only 0.7% of Christians, so
  most of them are probably answering Pentecostal. Benin draws the Celestial Church by name, so the
  border reads Celestial-in-Benin vs Pentecostal-in-Nigeria. Unfixable with these sources; say so.
- Optionally split Baptist out (Cameroon draws Baptists): it nearly passes (p 0.051). I would keep it in
  Protestant.

## 2. Liberia: `christianity`, 4,458,286 (84.9%)

**Now:** the census's single Christian box; county geography fitted from the pooled Afrobarometer.
`lr2022.py` refused a denominational split because `Christian only` swings 23-81% between rounds.

**What the refusal missed:** the swing is in *whether* people name a church, not *which*. Composition of
NAMED Christians by round (Afrobarometer, n named 2,711):

| round | unnamed % | Pentecostal | Methodist | Lutheran | Baptist | Catholic | Evangelical | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 | 32 | 29.0 | 20.1 | 9.9 | 14.6 | 9.9 | 6.0 | 10.5 |
| R5 | 51 | 30.2 | 15.5 | 14.6 | 13.2 | 12.4 | 5.2 | 8.9 |
| R6 | 26 | 20.4 | 15.5 | 15.2 | 13.6 | 12.8 | 2.3 | 20.2 |
| R7 | 77 | 24.7 | 18.9 | 14.5 | 14.2 | 9.8 | 2.0 | 15.9 |
| R8 | 81 | 23.4 | 21.3 | 19.0 | 13.2 | 13.8 | 0.6 | 8.7 |
| R9 | 68 | 19.5 | 20.6 | 23.9 | 12.6 | 6.7 | 11.7 | 5.0 |

Pooled: Pentecostal 24.9, Methodist 18.0, Lutheran 15.1, Baptist 13.7, Catholic 11.0, other Christian
(JW, SDA, Church of Christ, Orthodox) 6.9, Evangelical/CMA 4.8, Presbyterian 2.7, AIC/Independent 1.9,
Anglican (Episcopal) 0.9. Across rounds no denomination's share tracks the unnamed share consistently
(r between -0.46 and +0.57 over six rounds, mixed signs), unlike Nigeria, where Catholics name
themselves. **Catholic witnesses:** Pew 2010 8.4% of Christians; Church claim 361,000 (7.3% of 4.93M) =
8.6% of Christians; AB 11.0%. Consistent.

**County geography** (split-half, odd vs even rounds, 13 counties with 15+ named in both halves,
5,000 permutations): **Lutheran +0.724 (p 0.003), Baptist +0.602 (p 0.019), Catholic +0.564 (p 0.026)**
pass; Pentecostal -0.03, Methodist +0.09, Evangelical, other fail. Lutheran is Lofa, Bong, Gbarpolu,
Grand Gedeh (the Muhlenberg mission's interior), Baptist Nimba and Rivercess, Catholic Maryland, Grand
Kru, Grand Gedeh, Nimba. (Afrobarometer's county labels need folding: `Bassa`/`Grand Bassa`, `Cape
Mount`/`Grand Cape Mount`, case.)

**Proposal.** National shares from the pooled named answers, applied to each county's Christians;
Lutheran, Baptist and Catholic at their own county shares (raked to the national figure), the rest
taking the county's remainder in national proportion. Nodes: `christianity.pentecostal`, `.methodist`,
`.lutheran`, `.baptist`, `.catholic`, `.protestant` (Evangelical + Presbyterian, ~7.5%), `.other` (~9%,
AIC to `.africaninstituted` if preferred), Anglican 0.9% folded into `.protestant` or kept. All
`modelled`, rolling back to `christianity`.

**Contradiction:** the six-round stability above (assert each share's round range), the Catholic level
against Pew and the Church claim, the three county split-halves. **Caveat:** it assumes people who
answer only "Christian" belong to churches in the same proportion as those who name one. Nothing open
measures that directly in Liberia; the round stability is indirect evidence.

## 3. Sierra Leone: `christianity`, 1,547,129 (21.86%)

**The census itself split it, nationally.** National Analytical Report Table 3.25 (already cited in
`sl2015.py`): Catholic 7.0, Anglican 1.2, Methodist 3.0, SDA 0.7, Pentecostal 5.3, other Christian 4.7 (%
of the household population; sum 21.9). As shares of Christians: **Catholic 32.0, Pentecostal 24.2,
other 21.5, Methodist 13.7, Anglican 5.5, Adventist 3.2.**

This is Bulgaria's case exactly: the magnitudes are the publisher's, only the distribution across
districts is borrowed. Independent witness: the Afrobarometer's pooled named answers (n 693) give
Catholic 34.4, Pentecostal 21.2, Methodist 20.3, Anglican 6.8, Baptist 5.1, other 12, and Pew 2010 gives
Catholic 34% of Christians. Church claim 349,000 (4.0% of 8.69M) sits under the census's 7.0%, so the
claim is not inflated here.

`sl2015.py` says "a uniform 7.0% Catholic share cannot even fit [Pujehun, 4.8% Christian]". That is a
share of the *population*; the split would apply 32.0% to each district's *Christians*, which always
fits. Worth correcting in the note if this goes ahead.

**Proposal.** Flat national shares of Christians in all 14 districts, six nodes
(`.catholic`, `.pentecostal`, `.methodist`, `.anglican`, `.adventist`, `.other`). Afrobarometer
regions are the 4-5 provinces, so a provincial shape (Methodist east and south, Catholic north,
Pentecostal Western Area) exists but cannot be tested on so few units; leave it flat.
**Contradiction:** the AB national composition above; the 2004 census (if its denomination table is
found) for drift.

## 4. Guinea: `christianity`, 711,350 (6.77%), 62.5% of them in N'Zérékoré

No census or DHS split (2014 form one box; DHS 2018 FR353 Table 3.1: Musulmane / Chrétienne /
Animiste / Sans religion). Afrobarometer R5-R9, 414 named Christians: **Catholic 64.8, Evangelical
24.3** (the CMA-founded Église Protestante Évangélique), other 11 (Baptist, Anglican, Lutheran,
Orthodox, JW). Stable across rounds (Catholic 57-75%). Pew 2010: Catholic 68%, Protestant 31% of
Christians. Church claim 331,000 (2.4% of 13.6M), which is under the AB/Pew level; noted, not used.

**Proposal.** Flat national: Catholic 65%, Protestant/Evangelical 30%, other 5%. The evangelical node
should be whichever Anita picks for Côte d'Ivoire's évangéliques (section 5), since it is the same
movement across the N'Zérékoré / Man border. N'Zérékoré's own named split (57-76% Catholic by round
group) matches the national one, so flat costs little.

## 5. Côte d'Ivoire: `christianity.evangelical`, 5,445,459 (18.60%)

**No open source splits the évangéliques.** ANStat's prose figure is the only level; DHS 2021 (FR385
Table 3.1, women/men 15-49) has the census's own boxes again: Catholique 17.5/15.7, Méthodiste 2.4/2.3,
**Évangélique 24.4/17.4**, Autre chrétienne 4.8/3.5. Afrobarometer respondents also mostly answer
"Evangelical" (786) rather than Pentecostal (55), Assemblies of God (6), CMA (35) or Baptist (69): Ivorians
use *évangélique* as the self-label, so asking again does not divide it.

What the DHS adds: **a second year agreeing on the level** (women 24.4%, men 17.4% of 15-49s against the
census's 18.6% of everyone), which is the kind of witness the Bulgaria tests ask for.

**The real issue is the node, and it is Anita's call.** Burkina Faso's and Mali's census `Protestant`
boxes (→ `christianity.protestant`) hold the same people: in francophone West Africa *protestant* and
*évangélique* both name the Assemblies of God, CMA and Baptist missions. So the Ivorian side of the
Burkina and Mali borders draws one movement in two colours. Options:
- (a) keep `christianity.evangelical` (the census word); or
- (b) map the évangéliques to `christianity.protestant`, merging with the `Méthodiste/Protestant` cell,
  which matches Burkina, Mali, Ghana's Protestant box and Guinea under the proposal above;
- (c) recolour so `.evangelical` and `.protestant` sit close (a palette move, no data change).

Splitting into Pentecostal (to match Ghana) has no source; I would not.

## 6. Guinea-Bissau: `christianity`, 318,021 (22.05% of nationals)

**The 1991 census split it nationally** (RGPH 1991 *Resultados*, Tabela 6.6, PDF p61, Wayback
`stat-guinebissau.com/Menu_principal/IV_RGPH/rgph1/Resultados_rgph_1991.pdf`): Cristã Católica
125,637, Outras Cristãs 15,310, so **Catholic 89.1% of Christians**. Pew 2010: 89% (very likely the
same source). Not in the Afrobarometer, no DHS.

Christians went from 14.4% (1991) to 26.2% of those who answered (2009), and the US State Department
reports name Brazilian and other Protestant missions as the growth. Church claim 222,000 (13.1% of
1.69M) against ~22-26% Christian implies Catholics at roughly 55-65% of Christians today. So the 1991
ratio almost certainly overstates Catholics.

**Proposal.** Catholic / `christianity.other` (or `.protestant`) flat nationally. Anita's call on the
ratio: the census's own 89/11 (publisher's number, 18 years older than the counts) or something
nearer 70/30 between it and the Church claim (not a published figure; I would not).

## 7. Senegal: `christianity`, 307,722 (1988, 4.46%)

1988's form had Catholic (6) and other Christian (7) and printed them only for Diourbel: **Catholic
3,775 of 3,997 = 94.4%** (`data/normalized/sn.csv`). Pew 2010 95%. Afrobarometer named 83.4% (n 165,
noisy). Church claim 924,000 = 5.1% of 18.1M, above any census Christian share, i.e. consistent with
almost all Christians being Catholic. **Proposal:** Catholic 92%, `christianity.other` 8% (code 7's own
label is *autres chrétiens (protestants, luthériens, témoins de Jéhovah)*, Mali's `Autre religion
chrétienne` node), flat; Diourbel keeps its own counts. Low risk.

## 8. The Gambia: `christianity`, 69,638 (3.75%)

No census or DHS split (DHS 2019-20 Islam/Christianity). Afrobarometer R7-R9 named Christians n=54:
Catholic 87.7, Methodist 5.6, Anglican 2.8. Pew 2010: Catholic 66%, Protestant 33%. Church claim
54,000 (2.4% of 2.28M) ≈ 64% of Christians. **Proposal:** Catholic ~75%, `christianity.protestant`
(Methodist, Anglican, others) ~25%, flat. About 70 dots at 1:1,000; low payoff either way.

---

## Decisions only Anita can make

1. **Nigeria grain:** zone-level shares (pass the two-survey check, step at zone borders) vs state-level
   (smooth but failed, +0.20). And whether Baptist gets its own node.
2. **Liberia:** accept the assumption that unnamed Christians split like named ones (round stability is
   the evidence), reversing `lr2022.py`'s refusal.
3. **Côte d'Ivoire évangéliques:** keep `.evangelical`, map to `.protestant`, or recolour; and the same
   node for Guinea's evangelicals.
4. **Guinea-Bissau ratio:** the 1991 census's 89% Catholic, knowing it is probably high for 2009.

## Sources

- Global Flourishing Study wave 1, OSF `c8hbk` (on disk, `data/raw/gfs/gfs_all_countries_wave2.csv`).
- Afrobarometer merged R4-R9 (on disk, `data/raw/afrobarometer/`).
- Pew-Templeton Global Christianity table, 2011: https://www.pewresearch.org/religion/2011/12/19/table-christian-population-as-percentages-of-total-population-by-country/
- DHS final reports (open, no account): Liberia 2019-20 FR362 p81, Sierra Leone 2019 FR365 p80,
  Guinea 2018 FR353 p89, Gambia 2019-20 FR369, Côte d'Ivoire 2021 FR385 p88 —
  `https://dhsprogram.com/pubs/pdf/<code>/<code>.pdf`. Only FR385 splits Christians.
- Guinea-Bissau RGPH 1991 *Resultados*, Tabela 6.6, via Wayback (`web.archive.org/web/2024id_/https://www.stat-guinebissau.com/Menu_principal/IV_RGPH/rgph1/Resultados_rgph_1991.pdf`).
- Sierra Leone 2015 PHC National Analytical Report Table 3.25 (as cited in `taxonomy/sl2015.py`).
- Senegal RGPH 1988 Diourbel report Tableau 1.12 (as in `data/normalized/sn.csv`).
- gcatholic.org country pages GM, SN, GW, GN, LR, SL (2023-12-31 statistics).
