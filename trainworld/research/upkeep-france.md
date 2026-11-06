# Upkeep against French rail: track and stations per year, and a set for the game

Research note, 2026-11-06 (T-2618). The game charges upkeep per game day (docs/track-and-service.md
6.4), and a game day counts 100 real days, so the real figure is the shown one / 100 x 365.
**[SRC]** = read in the source. **[DERIVED]** = my arithmetic on sourced numbers (inputs stated).
**[EST]** = my estimate. Market exchange rates of the year: €1 = US$1.08 (2024), 1.14 (2020),
1.12 (2019), ~1.13 (2018-2022 average). Style and the metro running-cost figures:
research/running-costs.md.

**Maintenance** (entretien et surveillance) is the yearly opex. **Renewal** (renouvellement,
régénération) is capital spent replacing worn track every 30-40 years; SNCF Réseau spends about as
much on it as on maintenance. The game has no renewal, so its upkeep stands for both.

## Summary

Per km a year, real money. A double-track route-km is two track-km.

| What | Year | Per track-km | Per route-km | Includes | Tag |
|---|---|---|---|---|---|
| SNCF Réseau, whole network | 2024 | €64k = **US$69k** | ~US$138k (double) | maintenance only, no sidings | SRC |
| same, busiest lines (UIC 2-4) | 2024 | €89k = US$96k | ~US$192k (double) | maintenance only | SRC |
| same, quietest freight-only (UIC 7-9) | 2024 | €23k = US$25k | | maintenance only | SRC |
| SNCF Réseau renewal and upgrades | 2024 | €66k = US$71k | | renewal capex, network average | DERIVED |
| SNCF Réseau, maintenance + renewal | 2024 | €130k = **US$140k** | €260k = **US$281k** (double) | both | DERIVED |
| French LGVs | 2018-22 | €84k = **US$95k** | €168k = **US$190k** | maintenance, renewal and upgrades | DERIVED |
| RER A and B, RATP's sections (mostly underground) | 2024 | >€500k = **US$540k** | >US$1.08M (double) | maintenance only, stations' equipment partly in | SRC |
| RATP metro 1-14 + RER sections, all infrastructure | 2025-29 | | ~€2.7M = US$2.9M | maintenance, renewal, stations | DERIVED, rough |
| Japan subways, track and wires (running-costs.md) | 2005 | | ~US$1.3M | maintenance | DERIVED |
| **Game today, ground** | | | **US$73k** | everything | |
| **Game today, bored tunnel** | | | **US$146k** | everything | |

Per station a year:

| What | Year | Per station | Tag |
|---|---|---|---|
| Gares & Connexions, average, opex of the passenger part | 2019 | €201k = **US$225k** | DERIVED |
| same, including depreciation and return on capital | 2019 | €244k = US$273k | DERIVED |
| same, station charges over all stations | 2024 | €355k = **US$384k** | DERIVED |
| Lyon Part-Dieu, one big hub, cost base of its charges | 2020 | €8.1M = **US$9.3M** | SRC |
| A small halt (no staff, a shelter, no toilets) | | €20-50k = US$22-54k | EST |
| Tokyo Metro station (running-costs.md) | 2005 | ~US$3M | DERIVED |
| **Game today, any station** | | **US$183k** | |

**How realistic the game is.** Ground track is about half of French maintenance alone and a quarter
of maintenance plus renewal: T-867 aimed at "about a quarter of real upkeep" on purpose, as the
running costs sit under real ones. Stations are about half the French average. Tunnels are furthest
off: US$146k against roughly US$1M a route-km for Paris's RER tunnels, about a seventh.

**Gemini's figures check out roughly.** "US$55-90k per track-km a year for an LGV": France's LGVs got
about US$95k per track-km over 2018-22 with renewal and upgrades included, at the top of that range.
No LGV figure for maintenance alone was found. "€250-400k a station": that is the French average
(€244k in 2019 all-in, €355k in 2024), but it averages a skew of a hundred times or more.

## Details

**SNCF Réseau, 2024 (ART, the regulator, "Marché du transport ferroviaire en 2024", pp. 6, 9).**
27,110 km of line, 48,335 km of track (1.78 tracks per line). Maintenance and surveillance
€3.2bn, "près de 64 000 €" per km of track (sidings, ~5% of the cost, left out), "de près de 89 000 €
par kilomètre pour les voies les plus utilisées (catégories 2 à 4) à 23 000 € pour les voies de
catégories 7 à 9 sans voyageurs". Renewal and upgrade investment ("renouvellement et performance")
€3.2bn, the highest since 2021; [DERIVED] €3.2bn / 48,335 = €66k per track-km. LGV renewal fell 17%
in 2024 and has fallen about 4% a year since 2019; LGVs aged 3 years in 5, and the ART expects more
renewal soon, the oldest (Sud-Est, Nord) first. LGVs see about 100 trains a day.
https://www.autorite-transports.fr/wp-content/uploads/2025/12/bilan-ferroviaire-2024-essentiel-2.pdf

**Spread between sections (Frontier Economics for Arafer, 2017, p. 43).** Most track sections cost
under €100k per km of track a year in maintenance, with a long tail above €400k (nodes, switches).
Marginal wear per train-km (p. 50, €2018): a high-speed train on an LGV €0.38 (single unit) or €0.68
(double), on a conventional line €0.66 or €1.17. An LGV wears less per train than a classic line.
https://www.frontier-economics.com/media/ml0b4az2/20170208_estimation-des-couts-marginaux-dentretien-du-reseau-ferre-national_frontier.pdf

**French LGVs, 2018-22.** France spent €463M a year on "maintenance, renewal, and improvement of the
high-speed network" (European Commission, Rail Market Monitoring report 2025, via Ara, 2026-10-30):
https://en.ara.cat/society/spain-dedicates-half-of-the-budget-that-france-and-germany-to-the-maintenance-of-high-speed_1_5894683.html
[DERIVED] over ~2,750 km of LGV (2,700-2,748 km in the parliamentary and ministry counts), €168k per
route-km, €84k per track-km. Renewal was being put off in those years (above), so this is below the
long-run need. A secondary blog citing Réseau Action Climat gives "up to €100,000 per km per year"
for LGV maintenance (unverified):
https://www.hourrail.voyage/en/blog/prix-du-train-alexis-chailloux-reseau-action-climat

**Cour des comptes, 2014 LGV report** (p. 94): LGV running costs (maintenance and operation of the
line and trains) came out double the forecast on LGV Nord, the Île-de-France link, Rhône-Alpes and
Méditerranée, and 5.7 times on LGV Atlantique. No per-km upkeep figure in it.
https://www.ccomptes.fr/sites/default/files/EzPublish/20141023_rapport_grande_vitesse_ferroviaire.pdf

**Older network figure.** "200 000 euros par kilomètre" for "ses 28 000 kilomètres", against
Germany's €270k per km of line (Public Sénat, 2022, on the 2021-2030 performance contract; scope
probably maintenance plus renewal per route-km, which matches the €232k per route-km derived above):
https://www.publicsenat.fr/actualites/politique/sncf-reseau-le-regulateur-s-inquiete-d-un-contrat-de-performance-sans-0

**Stations (Cour des comptes, "Les gares ferroviaires de voyageurs", April 2021).** 3,017 stations;
551 had under 3,650 passengers a year (p. 24). 2019 regulated revenue €737M (p. 42), which by rule
equals the cost of the passenger part of the stations: upkeep and running, depreciation, return on
capital (p. 40); operating margin on it €129.5M, so [DERIVED] opex €607.5M, €201k a station. Whole
company including shops: revenue €1,131M, margin €185.8M, opex €945M, €313k a station. Lyon
Part-Dieu's cost base for its charges, 2020: €8.147M (p. 45). Halts get little: in 2020, 31% of
category C stations had toilets, 48% live information, none permanent security staff (p. 29).
https://www.ccomptes.fr/sites/default/files/2021-04/20210420-rapport-gares-ferroviaires-voyageurs.pdf
2024 (ART, pp. 24-25): station charges €1,041M; 21 very large stations (over 7M passengers a year),
80 other A, 931 B, over 1,900 C; [DERIVED] €355k a station on ~2,930.

**Skew** [EST]: a big hub costs US$5-40M a year (Part-Dieu US$9.3M; the Paris termini more), an
average station US$250-400k, a halt US$20-50k: about 400 times between ends. A game station is one
size, and game stations are mostly urban and busy, so the average is the right anchor.

**Paris metro and RER (RATP).** RER A and B on RATP's sections, 2024: maintenance and surveillance
€190.9M, "plus de 500 000 euros" per km of track, 21% of it track and track equipment; the rest
signalling, power, electromechanical equipment (escalators, ventilation) and civil works (ART 2024,
p. 11, link above). RATP as infrastructure manager of metro 1-14 and the RER sections: about €5bn
over 2025-2029, "both the daily operation of the network and an investment programme of nearly
€1 billion per year" (ART, 2026-02-11):
https://www.autorite-transports.fr/wp-content/uploads/2026/02/pr-ratp-remuneration-2025-2029-02-2026.pdf
[DERIVED, rough] €1bn a year over ~365 route-km (metro ~250 km, RER sections ~115 km [EST]) and ~385
stations: €2.7M per route-km with stations in. If stations take US$1.3M each [EST], the track part
is about €1.5M per route-km.

## Proposal

Real US$ a year per route-km of double track, then the game day's figure (/365 x 100, rounded).

| Piece | Today, game day | Today, real a year | **Set A (realistic), real a year** | **Set A, game day** | Set B (keeps the quarter), game day |
|---|---|---|---|---|---|
| Ground track, per km | US$20k | US$73k | US$280k | **US$77k** | US$20k |
| Viaduct, per km | 25k | 91k | 350k | **96k** | 25k |
| Cut and cover, per km | 35k | 128k | 800k | **219k** | 55k |
| Bored tunnel, per km | 40k | 146k | 1.0M | **274k** | 70k |
| Station, ground or elevated | 50k | 183k | 380k | **104k** | 50k |
| Station, underground (new) | 50k | 183k | 1.5M | **411k** | 100k |

- **Set A** takes the ground from SNCF Réseau's maintenance plus renewal per double-track route-km
  (US$281k, 2024), stations from the French average (US$384k, 2024), tunnels and underground
  stations from the RER and RATP figures with stations taken out (bored US$1.0M, near RER
  maintenance alone at US$1.08M and Japan's US$1.3M), viaduct at the ground x1.25 as today [EST: no
  French figure]. It is real upkeep, about four times today's ground.
- **Set B** keeps T-867's intent (game upkeep about a quarter of real on track, half on stations)
  and fixes only the tunnels, the part furthest off: cut and cover and bored about a quarter of the
  metro figures, underground stations twice a surface one.
- **High speed.** French LGVs cost less per route-km (US$190k) than the network's average maintenance
  plus renewal, and wear less per train. The game's 1.5x for high-speed upkeep is not supported;
  1x would match. The 1.5x also sets construction, so splitting it adds a second factor; I'd leave
  it, since LGV renewal is overdue and will rise.
- **Lie in Set A's tunnels:** part of a metro tunnel's upkeep comes from its traffic (trains every
  2 minutes, night-only work windows), not from being underground. A quiet tunnel in the game pays
  the busy metro's rate.

## What it does to a line's profit

Example: 30 km double track, 15 stations, 9-car trains 12 an hour each way for 18 hours (12,960
train-km a day). Running cost at US$4 + 9 x 2 = US$22 a train-km: **US$28.5M a game day**.
Profit falls by the upkeep change; fares are unchanged.

| Line | Upkeep today | Set A | Set B | Total cost change, A / B |
|---|---|---|---|---|
| On the ground, ground stations | US$1.35M | US$3.87M | US$1.35M | +8% / 0 |
| Bored tunnel, underground stations | US$1.95M | US$14.4M | US$3.60M | +41% / +5% |
| On the ground, quiet (4-car trains, 2 an hour; running US$2.6M) | US$1.35M | US$3.87M | US$1.35M | +64% / 0 |

A busy surface line loses about US$2.5M a game day under Set A, a modest dent. A busy subway loses
about US$12M a game day, and its costs rise by about two fifths, which can turn a thin line
negative. A quiet branch's costs grow by two thirds. On the real NYC save (4,499 route-km: 2,848 in
tunnel, 1,651 elevated; 1,756 stations; T-867's count), upkeep goes from about US$0.24bn to about
US$1.0-1.5bn a game day under Set A [EST: the cut and cover against bored split and the share of
underground stations are guesses], against ~US$0.95bn of running trains: total costs 1.6 to 2
times today's. That save gives
every line its own track, so a player's shared-trunk network would rise less.

## Uncertain

- No LGV maintenance-only figure found, and the RMMS LGV total was read through a newspaper; the
  report itself (European Commission, Rail Market Monitoring, 2025) was not opened.
- The Paris metro split between track and stations rests on my guess of US$1.3M a station; RATP's
  route-km and station counts are from memory.
- No French figure for viaduct upkeep.
- Station costs by category: the ART's 2024 figure 7.5 (charges by station class) was an image and
  could not be read; the hub-to-halt spread above is an estimate around one sourced hub.
