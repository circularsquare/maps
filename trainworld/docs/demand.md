[Back to the spec](../SPEC.md)

# 4. Demand: local trips (index)

The local demand spec was split by area on 2026-10-17 (T-335). Older notes, archive entries and code
comments that cite "docs/demand.md" with a section number or a paragraph name find it here:

| File | Holds (the old section and paragraph names) |
|---|---|
| [demand-sources.md](demand-sources.md) | 4.1 Where trips come from: tiers, cells, res 8 where it is sparse, jobs data, people in institutions, people who work at home, everyday outings, parks, trips below 150 m, school attendance, university attendance, airport ground trips, medical visits, attractions |
| [demand-timing.md](demand-timing.md) | 4.2 Time of day: kinds of day, the average day |
| [demand-mode.md](demand-mode.md) | 4.3 Mode choice and catchment: generalised cost, the ratio curve between rail and the car (T-918), cost damping, walking, driving, parking, routed walking and driving, the interface, lookup, access and egress, walking to a station, riding to a station |
| [demand-kernel.md](demand-kernel.md) | 4.4 Cost of computing it: commuting does not enumerate cell pairs, and steps 1 to 7 (gravity between zones, access per cell, access subzones, station-to-station times, mode choice per subzone pair, loading, crowding) |
| [demand-runtime.md](demand-runtime.md) | 4.4: which cities are open, bare zones, everyday warm-up, the pool and the bubbles' chain, clusters |
| [demand-market.md](demand-market.md) | 4.4: the world market, a composition opens without its zone pairs, the market's distances are on the ground |
| [demand-views.md](demand-views.md) | 4.4: loads by purpose, transfers, the trip explorer |
| [../notes/demand-measurements.md](../notes/demand-measurements.md) | numbers only: 4.4 cost, accuracy of subzones, cost on the real combined network, memory follow-up; the figures behind 4.1's sources, the workers' opening and bare zones, the world market and the trip explorer |

Other demand files: [demand-worldwide.md](demand-worldwide.md) (each rule's worldwide basis, country
by country), [demand-plain.md](demand-plain.md) (one page for Anita), [regional.md](regional.md)
(the regional layer).
