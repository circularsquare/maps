# Walk tiers: who won't walk far (T-3888, 2026-11-20)

Sources for notes/T-3888.md: how many people can't or won't walk far, how their travel differs,
and how transport research and other games treat walking ability. Searched 2026-11-20.

## How many people

- **WHO, Global report on health equity for persons with disabilities (2022)**: about 1.3 billion
  people, 16% of the world, have a significant disability (all kinds, not only walking).
  https://www.who.int/publications/i/item/9789240063600
- **US, American Community Survey, ambulatory difficulty** ("serious difficulty walking or
  climbing stairs"), 2025, as summarised by USAFacts: 67 in 1,000 people (6.7%), 28.3% of people over 75.
  The most common disability type. https://usafacts.org/answers/what-is-the-most-common-type-of-disability-in-the-united-states/country/united-states/
- **England, English Longitudinal Study of Ageing wave 7 (2014-15)**, in NatCen for DfT,
  "Capability, health and travel behaviour of older people" (2020), figure 5.4: difficulty
  **walking 100 yards** (about 90 m): 8% of people 50-64, 14% of 65-79, 35% of 80+. Same report,
  Understanding Society wave 7 (2015-17), figure 4.8: a long-term **mobility impairment**
  ("moving around at home and walking") in 13% of 50-64, 21% of 65-79, 39% of 80+; 19% of all 50+.
  https://assets.publishing.service.gov.uk/media/5fb2c8a5d3bf7f37cf4cf1a7/capability-health-and-travel-behaviour-of-older-people.pdf
- **World, people 65 and over**: 10.2% in 2024 (UN World Population Prospects through the World
  Bank, SP.POP.65UP.TO.ZS); Japan about 29%, the US about 18%, Nigeria about 3%. A global source a
  per-country share could use. https://data.worldbank.org/indicator/SP.POP.65UP.TO.ZS
- **Walking speed**: Health Survey for England 2005 (Asher et al., Age and Ageing 2012, "Most
  older pedestrians are unable to cross the road in time"): people 65+ walk 0.9 m/s (men) and
  0.8 m/s (women) on average; 76% of men and 85% of women walk slower than 1.2 m/s.
  https://www.sciencedaily.com/releases/2012/06/120614082712.htm

**Rough share of trips.** About 6-7% of people have serious difficulty walking (ACS 6.7%; ELSA's
100 yards weighted over an English age mix with an assumed 2-3% under 50 gives about 7%). They
make about a fifth to a quarter fewer trips (NTS below), so they make **about 5% of trips**. More
in old countries, less in young ones. Nobody counts people who could walk but won't (heat, bags,
small children); that part would be a game number.

## How their travel differs

- **NTS 2021, table NTS0709** (England, adults, trips a year, with / without a mobility
  difficulty): walking 131 / 243, bus 28 / 27, car passenger 119 / 97, car driver 177 / 389,
  rail 2 / 14. So they walk about half as much, ride the bus as much, and get lifts more.
  https://www.gov.uk/government/statistics/national-travel-survey-2021/national-travel-survey-2021-travel-by-disabled-people-and-people-with-mobility-difficulties
- **NTS 2025, disability**: disabled adults make 713 trips a year against 901 (21% fewer; 31%
  fewer at 60+; 25-30% fewer before COVID). Shares of their trips: walking 27% against 28%, bus 6%
  against 5%, car passenger 18% against 12%, car driver 41% against 47%.
  https://www.gov.uk/government/statistics/national-travel-survey-2025/nts-2025-disability
- **Yang and Diez-Roux, "Walking distance by trip purpose and population subgroups", Am J Prev
  Med 2012** (NHTS 2009, table 2): people walking on a given day 16.5% at 18-64, 11.8% at 65+;
  distance decay exp(-b x miles) with b 1.65 at 18-64 and 1.78 at 65+ (mean walk 0.61 against
  0.56 miles). Median walk 0.5 miles (800 m), mean 0.7; 18% of walks over a mile. So **older
  people walk less often, but the walks they do make are nearly as long**: the difference is who
  walks, not how far. https://pmc.ncbi.nlm.nih.gov/articles/PMC3377942
- **Inclusive Mobility (DfT 2005), section 2.4**, recommended distance without a rest: wheelchair
  users 150 m, visually impaired 150 m, mobility impaired with a stick 50 m, without a walking aid
  100 m ("average measures; there is a lot of variation between individuals").
  https://assets.publishing.service.gov.uk/media/5a79554040f0b63d72fc4e31/inclusive-mobility.pdf
- **Hess, "Access to public transit and its influence on ridership for older adults in two US
  cities", JTLU 2009**: in San Jose each extra 5 perceived minutes of walk to transit lowers older
  non-drivers' riding frequency about 5%; drivers are more sensitive to the walk than non-drivers;
  no effect found in Buffalo. https://jtlu.org/index.php/jtlu/article/view/11

## Walk weights and walk limits in research and models

- **Meta-analyses of walk time** (Abrantes and Wardman 2011, UK; Wardman, Chintakayala and de
  Jong 2016, Europe; a 2026 worldwide update of 2,872 multipliers): a walk minute is worth about
  1.5-2 minutes in a vehicle, varying with mode, purpose, distance and GDP. **None segments by
  age or disability** in what I could read; the 1998 review says socio-economic detail was out of
  its scope. https://eprints.whiterose.ac.uk/104595/ and https://econpapers.repec.org/RePEc:eee:transa:v:45:y:2011:i:1:p:1-17
- **TfL's PTAL**: stops count within 640 m (bus, 8 minutes) and 960 m (rail, 12 minutes) at
  4.8 km/h; one walker for everyone. https://en.wikipedia.org/wiki/Public_transport_accessibility_level
- Strategic models (four-step and the like) segment by car availability and income; walking
  ability is left to the spread of the access curve. I found no regional model with a
  mobility-impaired segment.

## Other games

- **The subway-building game**: one walk weight for everyone (1.39), spread from each commuter's income
  only (research/competitors.md).
- **The newer city-builder**: path cost from time, comfort, behaviour and money, comparing walking,
  driving and transit; age is said to change parking preferences; no walking rule by age found in
  the dev diaries' summaries.
- Searched and not found: a walk limit by age or disability in the subway-building game's docs (its wiki page
  on commuters was not opened), the newer city-builder, the track-drawing rail game.
