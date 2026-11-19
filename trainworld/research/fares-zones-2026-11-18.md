# Zone and stepped fares: other games and real systems (2026-11-18)

Agent faredesign, for T-3739 (advanced fare panel, curve points) and T-3740 (fare zones). Web pages read
2026-11-18; where a page could not be read, it says so. The plan these feed is notes/T-3739.md.

## What players asked for

- **Board** (notes/board-requests.json, sorted 2026-11-17): one fare group, g001 "Fares by line or service" (6
  reports, matched to T-3310): per-line fares, express and high-speed surcharges. Nothing about zones or steps.
- **Feedback inbox**, last 720 hours, read-only: the one-line list (206 threads) has fare threads only about
  line fares and express surcharges (1794971300 "Line-Specific Fares and Express Surcharges", 1794582145 "Route-Based
  Fares" (answered by Anita 11-13: "more advanced fare controls are planned"), 1794462461, 1794667196 and 1794979873
  for paid limited expresses, 1794755307 for high-speed fares). Per-line multiples (T-3567, T-3578) already answer
  most of these. No zone or staircase request in the list. The `--full` pass stopped partway with HTTP 429, so full
  texts of older threads were not all read; the titles were.
- So the zone requests Anita mentions came from somewhere else (Discord or in person); what those players
  described is unknown here.

## Other games

| Game | Fares | Zones |
|---|---|---|
| NIMBY Rails | Per line: a base ticket price paid once a boarding plus a price per km ([wiki: Line](https://wiki.nimbyrails.com/Line)). Players report riders choose by time only, price barely steering them ([Steam thread](https://steamcommunity.com/app/1134710/discussions/0/3050608970474596332)). | None |
| Cities: Skylines 1 | Per-line ticket prices through mods (Improved Public Transport 2, Ticket Price Customizer: [workshop comments](https://steamcommunity.com/sharedfiles/filedetails/comments/1393820309)); the base game's own control not checked on a page (from memory: one price per transport type). | None found |
| Cities: Skylines II | Ticket price per line; the price enters citizens' pathfinding (per search snippets of a Paradox guide; [PC Gamer](https://pcgamer.com/cities-skylines-2-is-making-an-important-change-to-how-public-transportation-works), [gameplay.tips guide](https://gameplay.tips/guides/cities-skylines-ii-transportation-guide.html); pages not opened). | None found, no zone-fare mod found |
| Cities in Motion 2 | Ticket kinds per vehicle type and monthly passes. | **Painted zones**: a Define Zones tool paints price zones on the map, up to four, each a colour; unpainted, the whole map is one zone; the diary suggests a London-style centre and rings ([dev diary 10](https://forum.paradoxplaza.com/forum/threads/cities-in-motion-2-timetable-zones-and-ticket-pricing-developer-diary-10.676796/), page did not load for me; quoted through search; [manual](https://www.scribd.com/document/431168062/CiM-20II-Manual-EN-1-pdf)). |
| Subway Builder | v1.6.0 (2026-08-06): "complex fare systems", charging by how far riders travel and/or which routes they take. v1.7.0: the flat fare cap raised from $10 to $100, route distances shown when setting distance fares ([changelog](https://www.subwaybuilder.com/changelog)). | v1.7.x: "You can now draw fare zones!" The changelog fetch dated it v1.7.0, 2026-08-30; a search snippet said v1.7.2, 2026-09-29. How zones are drawn and charged: not found. The wiki (wiki.subwaybuilder.com, search and sitemap) returned 502 twice. Closest comparator; worth a look in the game itself. |
| Transport Fever 2 | No player fares: payment follows straight-line distance and vehicle speed, by a formula players have not fully reproduced ([Steam](https://steamcommunity.com/app/1066780/discussions/0/3878218962819792571), [Reddit](https://redlib.cypherpunktools.com/r/TransportFever/comments/ztpldt/update_payment_formula/j1iy1d9/?context=3)). | None |
| Mini Metro | No money at all. | None |
| Workers & Resources | Nothing found on passenger fares in the wiki or Steam threads searched ([hoodedhorse wiki](https://hoodedhorse.com/wiki/Workers_Resources_Soviet_Republic/Vehicles)). | None found |

Takeaway: per-line base plus per km is the common model (NIMBY, CS2 per line). Zones exist in Cities in Motion 2
(painted, up to four) and Subway Builder (drawn, details unknown). No game found does stepped distance bands.

## Real systems

- **London** (zones in rings, 1 in the centre outward): a pay-as-you-go fare depends on the zones the journey passes
  through, not only start and end: from zone 3 through zone 1 to zone 2 pays zones 1-3; a route avoiding zone 1 can
  be priced lower with pink readers. Peak and off-peak differ; caps are by zones. National rail singles are point to
  point, not zonal ([TfL fares](https://tfl.gov.uk/fares/find-fares/tube-and-rail-fares),
  [Londonist on via-zone-1](https://londonist.com/2013/04/oyster-oddities-how-to-travel-through-zone-1-at-no-extra-cost),
  [TfL fares book](https://foi.tfl.gov.uk/FOI-2034-2223/Book%201%20Fares%20and%20Tickets.pdf)).
- **Tokyo Metro** (distance bands, the staircase): 1-6 km 180 yen, 7-11 km 210, 12-19 km 260, 20-27 km 300, 28-40
  km 330; IC fares to the yen (178, 209); km rounded up to the whole km
  ([Tokyo Metro regular tickets](https://www.tokyometro.jp/en/ticket/regular/index.html),
  [Japanese page](https://www.tokyometro.jp/ticket/types/regular/index.html)). Bands widen with distance and the
  fare tops out at 40 km.
- **Multi-operator in Japan**: each company charges its own fare, so a trip across two companies pays two fares;
  Tokyo Metro and Toei, both subways, take 70 yen off a change between them and nothing else is discounted
  ([Tokyo Metro connections](https://www.tokyometro.jp/lang_en/ticket/types/connection/index.html)). That is today's
  transfer fare idea with a minus sign; nothing to copy now.
- **German Verbund, rings** (Hamburg HVV): concentric rings A (core) to H; a single ticket is sold for Greater
  Hamburg (AB), for a number of rings, or the whole network, priced by ring count
  ([bahn.de regional](https://microsites.bahn.de/regional-de/regionen/schleswig-holstein/tickets),
  [bahninfo 2002](https://bahninfo.de/artikel/1095)).
- **Berlin VBB**: three areas sold as AB, BC and ABC, AB the city to its limit and ABC adding the surroundings and
  Potsdam (from a search summary only, no VBB page opened). Hanover sells one (A, B or C), two (AB, BC) or three
  zones (ABC) ([uestra](https://www.uestra.de/en/tickets-prices/ticket-overview/ticket-detail/single-ticket), title
  only).
- **Honeycomb (Waben)**: networks such as KVV and VRR use cells; each cell passed is counted once, however often it
  is crossed (non-official FAQ, [wandernundmehr](https://www.wandernundmehr.at/faq/wie-zahlt-man-waben)).
- **Zurich ZVV**: the price is the number of zones travelled through; the city zones 110 and 120 each count as
  two; at least two zones are always paid ([Fare zones of Zurich](https://en.wikipedia.org/wiki/Fare_zones_of_Zurich),
  [ZVV zones](https://zvv.ch/zvv/en/travelcards-and-tickets/zones.html)).
- **Netherlands** (per km, no zones): a boarding fare plus a per-km fare for regional transport: the 2026 standard
  boarding fare 1.16 euro (0.77 with discount) ([ROV handbook
  2026](https://rover.nl/images/PDFs/ROV-77%20verbeteringen%20Handboek%20OV%201-1-2026.pdf)); per-km rates of about
  0.20 euro in 2026 are set per region (search summaries of regional documents, e.g. a Brabant concession at
  0.202; not opened); a worked OVpay example 0.77 + 9.5 km x 0.145
  ([AVROTROS Radar](https://radar.avrotros.nl/artikel/jose-betaalt-een-cent-meer-per-rit-na-overstap-van-ov-chipkaart-naar-ovpay-62936)).
  That is today's game rule exactly. NS's own taper with distance was not confirmed. All of this section's pages
  were read through search summaries unless a figure says otherwise; Tokyo Metro, TfL and ZVV figures come from
  those operators' own pages as summarised.

## What this means for the game

- The game's curve already is the Dutch rule; Tokyo's bands are a curve with flat stretches and jumps; London
  and HVV are rings counted from the lowest to the highest zone; ZVV and Waben count cells crossed.
- Every zone system above charges by zones **on the trip**, and London explicitly by the route. Charging by the
  start and end zones only is a simplification that is exact for rings when the trip does not dip toward the
  centre and back out.
- Multi-city: real zone systems stop at their region's edge, and long trips go back to distance (DB, JR). No
  real system found prices a trip across two separate zone networks by zones.
