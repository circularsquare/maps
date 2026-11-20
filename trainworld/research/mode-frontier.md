# Mode frontier: what transport research says

No earlier file on this topic. Nearby files: `running-costs.md`, `tool-options-and-modes.md`.
Books (Vuchic, Meyer/Kain/Wohl, Boardman) were not readable here, so their claims come through lecture notes and agency reports. Figures below are only ones seen in a source. "Unsourced" marks gaps.

## 1. Capacity ranges and demand thresholds

Corridor demand is almost always passengers per hour per direction (pphpd), counted at the busiest point.

Theoretical capacity (ITDP BRT Planning Guide, Table 2.7, [source](https://brtguide.itdp.org/branch/master/guide/why-brt/performance)):

| System | pphpd |
|---|---|
| Metro, 8-car, single track | 35,904 |
| LRT, 8-module, 90 s signals | 10,744 |
| LRT, double track | 21,488 |
| BRT, bi-articulated, no passing | 11,220 |
| BRT, passing lanes | 36,000 |
| BRT, passing lanes + limited stops | 45,000 |

Observed (same page, Tables 2.8-2.9): BRT 529 (Las Vegas) to 37,700 (Bogota); LRT 214 (Seattle streetcar) to 13,400 (Tunis); metro 5,100 (Atlanta) to 84,000 (Hong Kong).

Where the modes overlap:
- World Bank/SYSTRA exhibit: BRT, LRT and metro can all reach 30,000 pphpd; capital cost for that is about $16M/km (BRT), $37M/km (LRT), $84M/km (metro). Only above 50,000 pphpd is metro the single option. [World Bank LUTP slides](https://thedocs.worldbank.org/en/doc/67623cb5aedfad0fd1f9302ada3d9930-0090062024/original/C4-M1-Choosing-Rapid-Transit-Alternatives-Combined.pdf)
- Hidalgo (quoted by Hensher & Golob 2008): metro and high-level BRT give similar capacity between 20,000 and 40,000 pphpd; BRT $5-20M/km, metro $30-160M/km. [Hensher & Golob via USyd](https://ses.library.usyd.edu.au/handle/2123/19548?show=full)
- A Canadian master's project puts the BRT/LRT switch at about 5,000 pphpd (single study, assumption-heavy). [UVic](https://dspace.library.uvic.ca/bitstreams/22424186-0b6b-4f5f-a694-e8ab4d10bd67/download)
- Britannica: fixed guideway needs roughly 10,000-20,000 pphpd or more before unit cost falls below bus/light rail. [Britannica](https://www.britannica.com/topic/mass-transit/Mass-transit-finance)
- Tirachini, Hensher & Jara-Diaz (2010): on Australian data a high-standard bus is cheapest in total (operator + user) cost in most scenarios. Rail wins only if it is clearly faster than BRT. A companion paper says adding crowding dislike can make heavy rail best at high demand. No demand number found in abstracts. [RePEc](https://ideas.repec.org/a/eee/retrec/v29y2010i1p231-242.html), [Part E](https://ideas.repec.org/a/eee/transe/v46y2010i6p1148-1168.html)

So there is no single agreed crossover. Quoted values run from about 5,000 to 20,000+ pphpd, depending on capital cost and labour cost assumptions.

Cost overruns (Flyvbjerg, Holm, Buhl 2002): rail averages +44.7% over estimate (n=58); rail passenger forecasts were about 51% too high on average. [World Bank WP 3781](https://ppp.worldbank.org/public-private-partnership/sites/default/files/2022-05/wps3781.pdf)

## 2. Cost per passenger and annualising capital

Operating cost against demand (World Bank/SYSTRA exhibit, described in the slides): rail O&M per passenger falls as peak-direction volume rises (fewer drivers per passenger offset higher fixed maintenance). BRT O&M rises slowly with volume (more buses, more drivers). Rail becomes cheapest to operate around 15,000-20,000 trips/hour at the max load point in developing-city wages; the band is different in high-wage cities. Same World Bank link as above.

Who dominates where:
- US 2013 capital split: infrastructure is 90% of heavy rail and 93% of light rail capital, but only about 48% for bus (vehicles are the other half). [MIT 1.258J lecture 6](https://www.ocw.mit.edu/courses/1-258j-public-transportation-systems-spring-2017/54e681bdddcefd40e0d75573d2274267_MIT1_258JS17_lec06.pdf)
- Same lecture: metro productivity is 3-4 times bus in passenger-miles per vehicle-hour; US 2013 operating cost per passenger mile roughly $0.9 bus, $0.5 heavy rail, $0.7 light rail, $0.5 commuter rail (APTA, partly garbled in the extract, treat as indicative).
- Labour is 35-75% of operating cost in Europe/N. America, about 20% in lower-income countries. [ITDP BRT Guide, costs](https://brtguide.itdp.org/branch/master/guide/why-brt/costs)
- Per passenger-mile, vehicle capital is $0.05-0.10 for all modes; infrastructure capital runs $0.01-1.00 (MIT lecture 6 above).

Annualising capital (what could be found):
- Discount rate: US DOT grant guidance used 7% real for a long time; the 2024-25 update follows revised OMB rates (lower, check the current text). FHWA notes switching 7% to 3% raises annual investment needs about 20%. [USDOT BCA guidance](https://www.transportation.gov/sites/dot.gov/files/2024-11/Benefit%20Cost%20Analysis%20Guidance%202025%20Update%20%28Final%29.pdf), [FHWA Appendix C](https://www.fhwa.dot.gov/policy/23cpr/appendixc.cfm)
- Asset life: ITDP says buses are depreciated over about 10 years, rail vehicles about 20. [ITDP costs](https://brtguide.itdp.org/branch/master/guide/why-brt/costs)
- A 30-year infrastructure life and the exact rates used by Tirachini et al. were not found in any open source. Unsourced; 30 years is a common default but not confirmed here.
- Cheap debt shifts the answer: a 0.2% JICA loan helped Jakarta's metro (ITDP costs page).

## 3. Trip length, stop spacing, speed

Stop spacing theory: total cost per unit length = operating cost (falls with wider spacing) + in-vehicle time (falls) + walk access time (rises). The optimum has the form of a square root: spacing grows with walk speed, stop cost, time lost per stop and value of in-vehicle time, and shrinks with demand density. Originates with Vuchic and Newell (1968) and Wirasinghe and Ghoneim (1981). [MIT 1.258J lecture 11](https://ocw.mit.edu/courses/1-258j-public-transportation-systems-spring-2017/0f318ed94f7099b98e3ee29630ec11f9_MIT1_258JS17_lec11.pdf)

Numbers seen:

| Item | Value | Source |
|---|---|---|
| Bus stops, US practice | about 200 m | MIT lecture 11 |
| Bus stops, European practice | about 320 m | MIT lecture 11 |
| BRT stations | 300-800 m, best near 450 m in urban areas | [ITDP station location](https://brtguide.itdp.org/branch/master/guide/brt-stations/station-location) (via search summary) |
| Commercial speed, BRT | 17 km/h (Eugene) to 54 km/h (Pittsburgh busways) | ITDP performance page |
| Commercial speed, LRT | 8 km/h (Seattle streetcar) to 40 km/h | ITDP performance page |
| Commercial speed, metro | 33 km/h (San Juan) to 48 km/h (Manila MRT-3) | ITDP performance page |

Metro, LRT and commuter-rail spacing in metres and regional rail speed: no source retrieved (Minnesota Met Council Table 3-1 has them but the extract was unreadable). The speed bands in the brief (bus 12-20, regional rail 40-60) are unsourced here.

Hensher/Tirachini key point: rail beats bus on total cost only through speed; so spacing and speed matter more than capacity until demand is high.

## 4. BRT versus LRT

GAO-01-984 (2001, 2000 dollars): [GAO](https://www.gao.gov/products/gao-01-984)

| Type | Capital $M per mile |
|---|---|
| BRT busway | 13.5 |
| BRT on HOV lanes | 9.0 |
| BRT on city streets | 0.68 |
| Light rail (average) | 34.8 (range 12.4-118.8) |

GAO also found BRT speeds generally at or above light rail, buses cheaper per vehicle operated, and no clear pattern in operating cost per trip. Officials argued LRT's permanence drives development.

ITDP (2013 USD per km): BRT about $10-12M; surface LRT $25M (lower-income) to $37M (higher-income); metro $87M to $434M; worst case NYC Second Avenue about $983M/km. [ITDP costs](https://brtguide.itdp.org/branch/master/guide/why-brt/costs)
Kain mid-1990s US averages: heavy rail $80M/km, LRT $25M/km. Busways $5-11M/km (Bogota $5M, Pittsburgh $11M). (MIT lecture 6.)
Hensher & Golob: most of 44 BRT systems cost under $10M/km.

## 5. How this is drawn

- Cost-per-passenger against passengers per hour: each mode is a curve. Bus is lowest at small volume (low fixed cost, high per-passenger cost that climbs as lines need more buses and drivers). Fixed-guideway curves start high (big fixed infrastructure) and fall steeply with volume. They cross bus/BRT at some demand, then metro crosses LRT. The visible shape is a "lower envelope" of falling curves. (Shape stated in Britannica and the World Bank slides; I did not see a printed copy of the textbook version.)
- Vuchic's mode-family chart (course notes from Vuchic, 2003): x-axis is system performance (speed, reliability, capacity, image); y-axis is investment cost per line length. Three right-of-way categories: C street transit (buses, trams), B semirapid (LRT, BRT-like), A rapid transit (metro). First drawn "bipolarized" with only C and A, then B "filling the gap" between them. [Vuchic slides](https://www.jttri.or.jp/members2/kenkyuh/13_vuchic.pdf)
- SYSTRA/World Bank capacity-vs-cost chart: horizontal bars (capacity range per mode, overlapping) against capital $M/km; BRT bars sit far left on cost, metro far right.
- ITDP Fig. 2.12 plots capital cost ranges against throughput (values only in the image).
- Corridor demand from geometry: MIT lecture 6 builds peak flow from a pie-shaped sector (density at the CBD, density decay with distance, corridor length, trips per person, share inbound, transit share, peak-hour share). This is the usual way to get pphpd from a city.
- A true 2D chart of distance against density as a mode-region map was not found in the sources read. Unsourced.

## What this suggests for the game's chart

- X-axis: peak-hour passengers per direction (the classic one), log scale; it spans 500 to 50,000 pphpd across modes. Second axis, if needed: line length or trip length, which the literature mostly uses only through stop spacing and speed.
- Y-axis: total cost per passenger (or per passenger-km), annualised capital plus operating. Draw the lower envelope of each mode's curve; modes win in bands.
- Expect broad overlap, not clean thresholds. Sources quote anything from about 5,000 to 20,000 pphpd for where rail overtakes BRT, and 30,000 pphpd is reachable by BRT, LRT and metro alike. Show bands, not lines.
- Annualise capital with a capital recovery factor: real rate 3-7% (7% was the old US standard; 3% raises costs about 20%), vehicle life 10 years for buses and 20 for rail vehicles (ITDP), infrastructure life about 30 years (unsourced). Let infrastructure and vehicles use different lives.
- The research cost model has a user-time term (wait, access, in-vehicle) and the game's profit does not. Where rail wins in the papers it wins on speed and fewer drivers per passenger, partly through user cost. In the game, rail will look worse than in the papers unless speed raises demand.
- Research takes demand as fixed per corridor; in the game demand responds to service, so frequency, speed and stop spacing change the x-position of a corridor. Chart axes should state the demand actually served.
- Fixed-cost-heavy modes suffer most from the overrun and forecast-miss findings (rail +45% capex, traffic about half of forecast); a player-facing cost multiplier for tunnels and elevation matches the infrastructure-share evidence.
- Keep the speed lever: Tirachini et al. say rail is cheapest only when it is faster than BRT, so the chart should let stop spacing and commercial speed shift each mode's curve.
