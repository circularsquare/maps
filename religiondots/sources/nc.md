# New Caledonia (`nc`): Pew's levels, placed by Kohler's 1978 church count on the 2019 census

Drawn 2026-10-03 by session `fafd1067-nc`, freed by the gaps scout (`sources.md`
§scout-2026-10-03-gaps) on Anita's rulings of 2026-09-15 (priority holes) and 2026-09-16 (a country
no source asks is drawn on the best compiler figure, method said). `sources.md` §nc-2026-10-03 is the
summary. Files: `sources/nc.py` (counts), `sources/nc_geo.py` (units, placement),
`taxonomy/nc2020.py`, `countries/nc.py`, node `other.nc`. Raw files in `data/raw/nc/`.

## 1. What asks religion: nothing found

| looked at | result |
|---|---|
| Census forms 1996, 2009, 2019 | no religion item (scout 2026-09-14). The 2019 form asks community, tribe, customary status |
| Earlier censuses | Kohler 1979 p.8: "l'obédience religieuse n'ayant pas été relevée par les derniers recensements administratifs". The 1956 and 1963 volumes were not opened (French Polynesia's 1962 census did ask, so 1956/1963 here are worth a look, not likely to change the build) |
| UNSD table 28 | New Caledonia absent (`tools/oracle.py`) |
| Liogier et al., Aix/UNC survey 2007 (*Histoire, monde et cultures religieuses* 2008/2, pp.177-190) | qualitative fieldwork in Nouméa, Bourail and Houaïlou; not a sample of the population |
| Hamelin and Salomon, INSERM violence-against-women survey 2002-03 (1,012 women 18-54 from electoral rolls) | no religion table found in the published articles; the questionnaire was not seen |
| Searches 2026-10-03 for any ISEE, provincial, health, youth or values survey with a religion item | none. Not checked: Baromètre santé adulte questionnaires, ISEE's 2019 census individual file on data.gouv.nc (`recensement-de-la-population-2019-individus-nc`; the form has no religion, so it cannot) |
| Pew 2020 | Christian 85.10, unaffiliated 10.52, Muslim 2.76, other 0.95, Buddhist 0.62, Jewish 0.04. Pew 2012 Appendix B and Pew 2011 Appendix D: "All estimates based on 2010 World Religion Database". Pew 2011's traditions table: 130,000 Catholics, 80,000 Protestants of 210,000 Christians (rounded) |
| WRD via ARDA (`thearda.com/world-religion/national-profiles?u=162c`, 2025, read 2026-10-03) | Christians 85.10 (Catholics 51.04, Protestants 15.32, Independents 9.52, unaffiliated Christians 9.23); agnostics 9.49, atheists 1.03; Baha'is 0.37; Buddhists 0.63; ethnic religionists 0.18; new religionists 0.41; Jews 0.04; Muslims 2.76 |
| J.-M. Kohler, *Religions et dynamique sociale en Nouvelle-Calédonie, Fasc. II* (ORSTOM Nouméa 1979; IRD Horizon `divers18-07/17982.pdf`) | church members by community (Tableau 1) and by commune and community (Tableau 2, Melanesians by commune of ORIGIN), reference date start of 1978. A roll: counted from church leaders, registers and gendarmerie reports (p.8), children included. The same figures are on *Atlas de la Nouvelle-Calédonie*, planche 27 (ORSTOM 1981; `divers16-08/010023263.pdf`) |

## 2. The route: an ethnicity model, Kohler as the coefficients, Pew as the level

A single national mix (the Comoros and North Korea route) would draw Lifou, whose Kanak were 87%
Protestant in 1978, at the territory's Catholic majority; the Catholic/Protestant geography is the
one thing every description of New Caledonia's religion agrees on, and Kohler is the only source
that places it. The 2019 census prints community by commune (ISEE `rp-structure-communautes.xls`,
sheet `commune`). So spec §14.12's model: X = community by commune; religion × X = Kohler's
Tableau 1, and for Kanak his commune-of-origin rows in Tableau 2.

**Bases.** Pew's figures are `estimate`; Kohler's are a `roll`. Spec §3.1 lets another basis split a
category but not add to it. Here every national level is Pew's; Kohler only splits Pew's Christians
into churches and decides where each group's people go (through the fit, §3). The rows carry
`basis=estimate`.

**Construction** (docstring of `sources/nc.py` has the detail):

1. Seeds per commune and community at Kohler's 1978 mixes. The census's commune table folds
   Tahitians, Indonesians, Vietnamese and Ni-Vanuatu into "other and not declared"; each commune's
   cell is split at its province's proportions (province sheet). Several communities, and the rest
   of "other and not declared" (Calédonien, other Asian, undeclared), take the commune's mix of
   everyone else.
2. Kanak away from home. Kohler puts urban Melanesians at their village of origin (p.9), so his
   Nouméa and Dumbéa have none. Each origin commune's count is scaled by Kanak growth (111,856 /
   57,433 = 1.948); a commune with fewer Kanak in 2019 than that is taken to have lost the
   difference at its own mix, and a commune with more takes the extra people at the mix of all who
   left (46,897 people, 58.1% Protestant, mostly from Lifou and Maré). Kouaoua and Poum use their
   parent's row (Canala, Koumac).
3. IPF of the seed groups to Pew's levels on the 2019 count and to each commune's population.
   Christians are one group, split inside each commune at the seed's church shares. No religion
   on Kohler's `Divers` (which held the atheists, Tableau 1 note); Muslims and Bahá'ís on his.
   Buddhists flat on the census's "other and not declared" column; `other.nc` (WRD ethnic and new
   religionists, plus Jews) flat on everyone.

**Result** (`data/normalized/nc.csv`, 33 communes, 271,407): Catholic 160,533 (59.15%),
congregational Protestant 65,232 (24.03%), no religion 28,562 (10.52%), Muslim 7,491, Buddhist
1,696, other.nc 1,689, other Protestant 1,549, Pentecostal 1,062, Bahá'í 998, Witnesses 874,
Adventist 863, Latter Day Saints 858. Loyalty Islands 71.1% Protestant (Lifou 80.3, Maré 77.1,
Ouvéa 36.5); Bélep, Yaté and the Ile des Pins 94-95% Catholic; Nouméa 60.2% Catholic, 19.4%
Protestant, 14.1% no religion, 3.1% Muslim. 264 dots at 1:1,000.

**Witnesses.** Catholics are 69.5% of drawn Christians, against Kohler's 71.2% in 1978 and the
WRD's 60.0% (whose 22% Independents and unaffiliated Christians are unassigned). The seed before
fitting is 95.4% Christian, 2.8% `Divers`, 1.5% Muslim (Kohler's mixes on 2019 communities), so the
fit multiplies no religion by about 3.7 and Muslims by about 1.8. The Loyalty islands' drawn
Catholic share of Catholics and Protestants (Lifou 15.1, Maré 18.8, Ouvéa 61.3) sits next to
Kohler's (12.8, 18.0, 61.1); it is not independent, since it is the model's input.

**No check exists** (§14.12 condition 3): nothing cuts Kohler's coefficients a second way, and no
later source counts religion at all. The note says so and names no religion's location as the
weakest drawn cell.

## 3. Calls someone might reverse

- **Kohler at all.** A 1978 roll is old and not self-identification. The alternative is one
  national mix with no Catholic/Protestant split (Pew gives none) or Pew 2011's rounded
  130,000/80,000, drawn flat; the Loyalty Islands would then be drawn majority Catholic.
- **Pew's level, not Kohler's.** Kohler's 1978 territory is 3.4% `Divers`; Pew's 10.5% no religion is
  the WRD's agnostics and atheists, itself an estimate. Kept Pew because the queue row named it
  and a 1978 church count cannot see secularisation since.
- **The migrant rule** (§2 step 2). Nouméa's Kanak come out about 58% Protestant; at the plain
  national Kanak mix they would be 50%.
- **Protestants on `christianity.reformed.congregational`**, no node for the Église protestante
  de Kanaky Nouvelle-Calédonie or the Église évangélique libre (split only nationally in Kohler).
- **`other.nc` and Buddhists drawn flat**; Kohler's 1978 Bahá'ís (235 of 320 Kanak) place Pew's
  Bahá'ís.

## 4. Units, population, placement

- 33 communes, ISEE 2019 census (`rp-structure-communautes.xls`, sheet `commune`, 2019 block).
  111 people in secret (`ss`) cells in Bélep, Hienghène, Ouvéa, Sarraméa and Yaté, fitted to each
  row's residual and each column's national shortfall. Poya is printed once; the province rows put
  its southern part in Sud (within 400).
- Boundaries: Gouvernement de la Nouvelle-Calédonie (DTSI/Georep), *Communes de la
  Nouvelle-Calédonie (limites communales terrestres simplifiées)*, data.gouv.nc, Licence Ouverte
  2.0; joined on `code_com`, witnessed by name; 18,351 km² against ISEE's 18,576. HDX has no
  COD-AB; geoBoundaries has ADM0 only.
- Kontur NC 2023: 4,001 hexes, 293,005 people, ratio 1.080 to the census; 435 coastal hexes snapped
  within 1 km, none dropped. **Kontur reads the Loyalty Islands at about half** (Maré 0.47, Ouvéa
  0.55, Lifou 0.58 of the national ratio; Kouaoua 0.62), dispersed tribal hamlets a built-up grid
  misses; Farino 1.94. Placement is inside each commune only, so nobody moves between communes.

## 5. §14

New Caledonia's politics run along community lines (the 2018-21 referendums, the 2024 unrest),
and this map derives religion from community. What it draws by commune is ISEE's own published
community table times Catholic/Protestant/none shares; the Catholic-Protestant divide carries no
persecution risk that the project knows of, and Muslims (about 7,500, most of Indonesian descent)
are not a targeted group there. No ask filed; flagged here so a reviewer can disagree.

## 6. Not checked, the next places to look

- The 1956 and 1963 census volumes (INSEE), for a religion table by commune.
- The INSERM 2002-03 survey's questionnaire (Hamelin, Salomon), and any Baromètre santé round.
- WRD's own New Caledonia sources (behind the WRD subscription).
