# Niger — RGP/H 2012, religion by région

**Drawn 2026-09-15** (session `d743fc47-ne`). 8 régions, 5 categories, 17,096,099 of 17,138,707
residents drawn (the 42,608 with no religion recorded are not), every row `measured`. 17,093 dots at
1:1,000, 1,707 at 1:10,000.

- `sources/ne.py` -> `data/normalized/ne.csv` (the PDF is
  `data/raw/ne/ETAT_STRUCTURE_POPULATION.pdf`, pinned)
- `sources/ne_geo.py` -> `data/geo/ne/ne_regions.gpkg`, `ne_hexes.gpkg`, `ne_lookup.csv`
  (COD-AB v02 ADM1, Kontur 400 m)
- `taxonomy/ne2012.py` -> the mapping; `countries/ne.py` -> the entry; `taxonomy/branches.py`
  `other.ne` is the one new node
- sources.md **§ne-2026-09-15** is the summary; **§11aq** was the scout's row.

```
python sources/ne.py     --fetch
python sources/ne_geo.py --fetch
```

## 1. What the Institut National de la Statistique publishes

| release | religion | tier |
|---|---|---|
| **RGP/H 2012, *État et structure de la population du Niger en 2012*** (88 pp, `stat-niger.org/wp-content/uploads/2020/05/ETAT_STRUCTURE_POPULATION.pdf`) | **Tableau A 11 (PDF p.88), région x religion in counts**; Tableau 20 (p.59) the same as shares; section V.1 (pp.57-58) national counts and each région's share of Christians and animists | **8 régions** |
| UNSD Demographic Yearbook table 28, Niger 2012 | the national row, six categories, equal to A 11 | nation |
| *Niamey en chiffres 2015* (INS regional office, 2 pp) | none; used for Niamey's area (§5) | |
| RGP/H 2012 household questionnaire (UNSD's copy, `NER2012frHh.pdf`) | column C07 (§4) | |

**From the scout, not re-opened** (sources.md §11aq, 2026-09-14): the 2012 household characteristics
volume (département tables, no religion), the Dosso regional monograph (122 pp, none), *Résultats
définitifs* (351 pp, no hit), RGP/H 2001 *État et structure* (religion by milieu and ethnic group,
not by région), the stat-niger.org CDX (11,100 lines). **Not opened by anyone, so not negatives**: the
other seven 2012 regional monographs (Wayback, `www.stat-niger.org/statistique/file/RGPH2012/
Monographie_Regionale_*.pdf`), which are the likeliest place for a département table; the 2012
census microdata; the EDSN-MICS 2012 DHS, which asks religion on a regional design. Afrobarometer R5-R9
(5,998 respondents, 8 regions) was measured by the scout and adds nothing (Muslim +0.64 on 92.7-99.5%).
RGPH-5 is in preparation (`rgph5.ne`).

## 2. The construction

No arithmetic. Each région's five religion cells and `ND` are its row of Tableau A 11. The table
covers the resident population, 17,138,707, the census total in Tableau 3. The rendered page was read
for the column order (Sans religion, Musulman, Chrétien, Animiste, Autre à préciser, ND, Total); the
text layer puts `Total` first in the header and is pinned as a string.

## 3. The checks (`sources/ne.py::check`)

| check | result |
|---|---|
| PDF pinned | stat-niger.org 2026-09-15, digest `N5DQIOCM7DAYQGQ3E5HKDNDC4LPJJWKO`, 1,883,934 bytes, 88 pages; p.88 is text, no images |
| A 11 parses | every cell equals the transcription; each row sums to its total; the régions sum to the Total row in all 7 columns |
| UNSD table 28 | the Total row equals all six UNSD categories and the total (`ND` is UNSD's `Unknown`) |
| Tableau 3 (p.23) | the 2012 population column equals A 11's totals for every région and the nation |
| Tableau 20 (p.59) | shares of **those who stated a religion** (ND left out): every cell rounds from A 11 except three, Dosso animist 0.4, Tahoua Christian 0.2 and Niamey Christian 1.5, each in a row whose rounded shares sum to 99.9 and each printed 0.1 above its rounding; against everyone, 12 cells are off |
| section V.1 prose | the national counts, and each région's share of all Christians (Tillabéri 37.4, Niamey 25.2, Maradi 11.3, Agadez 1.8, Diffa 2.1) and all animists (Zinder 26, Dosso 19.7, Tillabéri 16.3) |
| Tableau 7 (p.32) | 2012 density per région, transcribed for `ne_geo.py`'s area witness; the national row is labelled `Ensemble Niger` and Niamey has no 1977 value |
| Tableau 23 (pp.61-62) | foreigners by région, sexes summing to totals, 113,647 in all |

**The prose's "99,3% de la population résidente totale" is a share of those who stated a religion**
(16,978,889 is 99.07% of all residents). The note uses 99.1% of everyone.

## 4. The questionnaire and the codes

Household form dated 17 December 2012 (UNSD's questionnaire archive,
`unstats.un.org/unsd/demographic/sources/census/quest/NER2012frHh.pdf`, 2 pages, text layer read),
column **C07, *Quelle est la religion de [PRENOM] ?***: **0 = Sans religion, 1 = Musulmane,
2 = Chrétienne, 3 = Animiste, 9 = Autre à préciser.** "Pour les enfants en bas âge, prendre le code
correspondant à la religion du père ou de la mère." The volume quotes the same question (p.17).

So animist is offered beside no religion (`tools/check_no_religion.py`: `separate`), and there is **no
code for no answer**: `ND` is a blank or unreadable field, EXCLUDED, and is the `gap`
(`tools/gap_share.py` agrees, 0.25%). One Muslim code and one Christian code, so no school, order or
church is drawn; the State Department's 2023 report cites the former Ministry of Interior for about
80% of Muslims being Maliki Sunni, which the census cannot place.

## 5. Geography

**OCHA COD-AB Niger v02** (`cod-ab-ner`, valid from 2023-07-20) has the 8 régions as ADM1, NE001
Agadez, NE002 Diffa, NE003 Dosso, NE004 Maradi, NE005 Tahoua, NE006 Tillabéri, NE007 Zinder, NE008
Niamey, joined by pcode with the names asserted. **geoBoundaries gbOpen NER ADM1 is not a witness**:
six features, `Tahoua/Agadez` and `Zinder/Diffa` merged and Dosso spelt `Dossa` (read 2026-09-15; the
download was deleted).

**Area witness.** Tableau 7's density gives the office's area per région. COD's national polygon is
1,185,195 km2 against the census's 1,269,534 (0.934), so each ratio is divided by that:

| région | census km2 | COD km2 | / national ratio | COD départements |
|---|---:|---:|---:|---:|
| Agadez | 696,600 | 621,901 | 0.956 | 6 |
| Diffa | 156,269 | 147,715 | 1.013 | 6 |
| Dosso | 33,849 | 31,413 | 0.994 | 8 |
| Maradi | 41,795 | 39,349 | 1.008 | 9 |
| Tahoua | 113,210 | 107,312 | 1.015 | 13 |
| Tillabéri | 97,232 | 90,312 | 0.995 | 13 |
| Zinder | 155,937 | 146,636 | 1.007 | 11 |
| **Niamey** | **255** | **557** | **2.339** | 1 (5 arrondissements) |

(Agadez's density is printed as 0.7, so its census area is good only to about 7%.)

**Niamey is the exception, and it moves almost nobody.** COD's Ville de Niamey is its five
arrondissements communaux (Niamey I-V, 54-171 km2 each), 557 km2. INS gives 255 km2 twice: behind
Tableau 7, and on *Niamey en chiffres 2015* ("Superficie : 255 Km²", beside the 2012 population
1,026,848 and the same five arrondissements). A web search turned up "552.27 km2" for the capital
district as well; not followed. Which outline is right is not settled. Kontur says it does not
matter: **the densest 255 km2 of COD's Niamey hold 96.4% of its 2,076,171 Kontur people**, 74,283
outside, and the polygon touches only Kollo département of Tillabéri. Pinned in `ne_geo.py`
(`AREA_PINNED`, `NIAMEY_CORE_MIN = 0.95`).

**People witness.** Kontur NE 2023-11: 122,693 hexes, 27,324,506 people, 1.594x the 2012 count.
Per région over that ratio: Agadez 1.32, Diffa 1.47, Dosso 1.23, Maradi 0.86, Tahoua 0.94, Tillabéri
1.06, Zinder 0.80, Niamey 1.27, all inside the 0.60-1.60 band. Diffa's excess fits the displacement
from Nigeria since 2013, which the 2012 count predates. 67,303 people snapped within 500 m, 75,463
(0.28%) dropped as outside the country; 122,346 hexes placed. `kontur_cap.py ne`: no stops. No sea to
clip. The smallest unit is 557 km2, so no grid-floor concern.

## 6. What the table shows

99.07% of residents are Muslim, 0.33% Christian, 0.20% animist, 0.13% no religion, 0.015% other.
**Christians:** Niamey 1.40% (14,353) and Tillabéri 0.78% (21,292) hold 62.7% of them; Dosso 0.24%,
Agadez 0.21%, Diffa 0.20%, Maradi 0.19%, Tahoua 0.13%, Zinder 0.10%. The report (p.58) puts
Tillabéri's figure down to refugee camps for people from Mali and early missions, and (p.44) says
Christianity mostly concerns foreigners; Tableau 23 puts 27,585 foreigners in Tillabéri and 30,659 in
Niamey, but no table crosses religion with nationality, so neither claim is checked. **Animists:**
Dosso and Niamey 0.34%, Zinder 0.26% (the largest count, 9,053). **No religion:** Dosso 0.19%,
Tillabéri 0.18%, Agadez and Diffa 0.02%. **Other:** 1,044 of 2,520 are in Niamey. **ND:** Niamey
0.61%, Agadez 0.43%, Tillabéri 0.17%.

## 7. §14 was considered and no ask was filed

The State Department's *2023 Report on International Religious Freedom: Niger* (state.gov PDF
`547499_NIGER-2023-INTERNATIONAL-RELIGIOUS-FREEDOM-REPORT.pdf`, read in full): IS-GS and JNIM
"maintained violent activity and community presence in Tillaberi and Tahoua Regions"; fatalities
"spiked" in Tillabéri after the July 2023 coup; JNIM "maintained administrative control of communities
in western Tillaberi Region". And, from Open Doors, "in mid-August unknown violent extremists stormed
two churches in a village near the border with Burkina Faso. The extremists beat the worshippers with
whips for disobeying the extremists' ban on Christian worship". Niger's border with Burkina Faso is
Tillabéri's. So this is attacks on people for being Christian, in the région where the census puts
37% of Niger's Christians.

**Drawn at the 8 régions anyway, on the reasoning of ask 018**, where Anita cleared Burkina Faso at 45
provinces and Mali at 20 régions despite identity massacres, because the units are big. Niger's units
are larger still: Tillabéri is 2.7 million people over 97,000 km2, and the map places Christians
nowhere more finely than INS itself prints in a public volume and forwarded to the UN. The one
alternative is Niger as a single unit, which the 2026-09-14 ruling keeping `ne` free did not ask for.
No ask filed; a session that finds a finer table (a département annex in the regional monographs, §1)
must reread this before drawing it.

## 8. Gotchas

- **Tableau 20 and the prose are shares of those who stated a religion**, and Tableau 20 forces three
  cells so rows print 100,0. Take counts from A 11.
- **The text layer reorders A 11's header** (`Total` first). The rendered page is the column order.
- **Tableau 7 has no 1977 density for Niamey**, and labels its national row `Ensemble Niger` over two
  tokens; a label-driven parser reads `Ensemble` as a number otherwise.
- **The report spells the région three ways**: TILLABERI (tables), Tillabéry (prose, Tableau 3),
  Tillabéri (COD). The unit id is `Tillaberi`.
- **geoBoundaries NER ADM1 is six merged units**; COD's Niamey is twice INS's area (§5).

## 9. Reopen when

- A 2012 regional monograph turns out to print religion by département (COD's 67 ADM2; §7 first).
- RGPH-5 publishes religion.
- Anything splits Christians by church below the nation.

## 10. Terms

INS's volumes are public PDFs on its own site with no licence text. OCHA COD-AB is CC BY-IGO; Kontur
Population is CC BY 4.0. The UNSD questionnaire and the State Department report were read, not
redistributed.

## 11. Review, 2026-09-15 (session `d743fc47-rev11`)

Full pass. `check_md`, `built_countries --check` and `check_rollup ne` are clean (17,096,099
measured, 0 orphaned). Every figure in `note_public` was re-added from `ne.csv`: Christians 56,856,
Tillabéri 37.4% of them and Niamey 25.2%, ND 42,608, animists 0.34% in Dosso and Niamey. The five
mappings follow precedent (`other.ne` beside some sixty `other.<cc>` nodes).

**§14: the §7 call holds, and no ask is filed.** Ask 018 cleared Burkina Faso at 45 provinces and
Mali at 20 régions against insurgencies that target Christians in the same border zone, and the
ruling's only stated reason was unit size. Niger's régions average 2.1 million people and Tillabéri
is 2.7 million, larger than either. Churches stormed near the Burkina Faso border is the kind of
attack ask 018 already weighed, not a new kind. Inside Tillabéri the dots follow Kontur population,
so the map does not show where in the région Christians live.

**One suggestion for `note_public`, not made here.** Drop the clause that puts Tillabéri's Christians
down to refugee camps for people from Mali. It is the report's own explanation, which no table
checks. The 2012 refugees came mainly from Tuareg, Arab and Fulani communities in northern Mali (not
checked here), so it is doubtful. And it points readers at one kind of site inside the région where
churches were stormed, which the dots do not do. "Says Christianity in Niger mostly concerns
foreigners" can stay.

**Niamey's outline holds.** The census counted the five arrondissements communaux (§5), and COD's
557 km2 is those five, whichever area figure is right. With 96.4% of Kontur's people in the densest
255 km2, under 4% of Niamey's dots can land in the outer part, and that part is still Niamey.

**Map glance** (1400x900, framed on the country): dots on land only, dense along the south and the
river, sparse clusters at Arlit, Agadez and the Aïr. Nothing blank or flat.
