# Air Traffic Visualization — project guide

Notes for Claude sessions working in `maps/flights/`. Read this first.

## What this is

An interactive visualization of global plane traffic. The goal is to estimate
passengers between airport pairs worldwide and **highlight "unexpected" routes**
— flights carrying far more traffic than distance + population would predict
(diaspora corridors, migration, historical ties). Inspiration: a friend seeing
Korean-language signage at Guatemala City's airport.

The owner runs things from `maps/flights/`. The wider repo (`maps/`) holds
several unrelated map projects.

## Passenger model

`passengers per route ≈ flights × seats(aircraft type) × load factor (0.80)`.
Aircraft type is real per-flight; seat count is a typical-per-type lookup; load
factor is a flat assumption. Good enough for a *comparative* oddity-finder, not
for exact passenger totals.

## Data source — the story so far

No free global origin-destination passenger database exists (that is OAG /
Cirium territory, $15k+/yr). Three iterations:

1. **OpenFlights** (`build_routes.py`) — global route topology, but 2014-vintage
   and no flight frequency.
2. **OpenSky** (`build_opensky.py`) — real observed frequency (July 2019), but
   its crowdsourced ADS-B network has huge coverage holes: airport-resolution
   is US 71%, Europe ~66%, but South America 31%, China 12%. The map looked
   empty over the global south. **Abandoned as the primary source.**
3. **Flightradar24 API** (current direction) — much better global coverage
   (ADS-B + MLAT + satellite + schedule fusion). The owner has the FR24
   **Essential plan** ($90/mo). This is the live plan.

## Files

| File | What it is |
|---|---|
| `index.html` | Viewer — MapLibre globe, arcs, airport bubbles, OpenSky/FR24 data-source toggle. |
| `routes.json` | OpenSky-derived routes (the "OpenSky" toggle option). |
| `routes_fr24.json` | FR24-derived routes (the "FR24" toggle option). Rebuilt by `build_fr24.py`. |
| `fr24_pull.py` | The puller — one UTC day's departures per airport → `pull/<DAY>/<ICAO>.json`. |
| `build_fr24.py` | Aggregates `pull/<DAY>/*.json` → `routes_fr24.json`. Holds the airliner seats table. |
| `build_opensky.py` | Aggregates an OpenSky monthly file → `routes.json`. |
| `build_routes.py` | v1 OpenFlights aggregator. Superseded. |
| `fr24_probe.py` | One-off API probe. Its job is done; kept for reference. |
| `pull/<DAY>/` | Raw per-airport pull JSON + `_summary.json` (gitignored). |
| `data/` | OpenFlights `.dat` files + OpenSky monthly file (gitignored). |

## ⚠️ API USAGE RULES — read before touching FR24

**Credits are real money.** Essential = $90 for 333k credits (×2 promo = 666k).
`flight-summary/light` costs **2 credits/flight** for data ≤30 days old, 3 if
older. One careless query can burn the whole plan.

1. **NEVER query without an airport filter.** An unfiltered date-range query
   returns ~120k flights/day ≈ 240k+ credits — the whole plan, in one call.
   Always filter `airports=<dir>:<ICAO>`, one airport at a time.
2. **Test new API code cheaply first.** Run it on the real key but tiny scope —
   a 1-hour window, a few hundred airports — before any full pull. (The sandbox
   returns only static canned data and its path structure differs; not useful.)
3. **Confirm with the owner before any real-key run that spends meaningful
   credits.** Do not spend on their behalf.
4. **The puller must be resumable + credit-metered.** Checkpoint every response
   to disk; on restart, skip what is already saved; track credits against a
   hard cap and abort before overspending. A crash must never re-spend credits.
5. **Raw-then-aggregate.** The puller only fetches and saves raw JSON.
   Aggregation into `routes.json` is a separate script. **Never re-hit the API
   for data already on disk.**
6. **Use `light`, not `full`** (`full` is 6 cr/flight). Use `/count` (cheap)
   when only a flight count is needed.
7. **Pull recent data** (≤30 days old) — 2 cr/flight instead of 3.
8. **Rate limit: 30 req/min on Essential.** Keep `RATE_SLEEP ≥ 2.2s`. Handle
   HTTP 429.
9. **A `User-Agent` header is mandatory** — FR24 sits behind Cloudflare, which
   403-blocks default library user-agents (Cloudflare error 1010).
10. **Long pulls = the owner runs them.** Per the owner's global rule, hand off
    any command expected to take more than ~1 min; give the exact command and a
    rough duration.

## Secrets

Keys live in **`maps/secrets.env`** (repo root, gitignored via `secrets.*`),
loaded by `load_secret()`. **Never** commit a key, hardcode one, or print one
into logs, output, or commit messages. Two keys: `FR24_API_KEY` (real, paid)
and `FR24_SANDBOX_KEY` (free, static sample data).

## Licensing (matters before publishing)

FR24 terms: derivative works are permitted; **raw-data redistribution is
prohibited**; attribution is required. The public viewer must ship only
**aggregated estimates** (passengers/year per route), credit Flightradar24, and
not expose a bulk download of raw flight records. OpenSky data is
non-commercial / research-licensed.

## FR24 API quick reference

- Base: `https://fr24api.flightradar24.com/api` (`+ /sandbox` for sandbox).
- Headers: `Authorization: Bearer <key>`, `Accept-Version: v1`,
  `Accept: application/json`, `User-Agent: <browser UA>`.
- `/flight-summary/light` — params: `flight_datetime_from`/`_to` (ISO 8601),
  `airports`, `aircraft`, `sort`, `limit`. Returns `orig_icao`, `dest_icao`,
  `type`, `datetime_takeoff`, etc., as a list nested under a `data` key.
- `/flight-summary/count` → `{record_count}`.
- Historical data goes back to 2016-05-11. ~120k commercial flights/day global.

## Status & TODO

The pipeline works end to end: `fr24_pull.py` → `pull/<DAY>/` → `build_fr24.py`
→ `routes_fr24.json` → the viewer's FR24 toggle. A first-hour validation pull
rendered correctly on the globe.

**Pulling now:** 3 *contiguous* UTC days — **Sat 2026-05-16, Sun 05-17, Mon
05-18** (contiguous so every time zone yields a clean local day). One
`fr24_pull.py` run per day (change the `DAY` config). Saturday first; measure
its real credit cost before committing to Sunday/Monday.

- [ ] Pull Saturday May 16 (full day). Check `pull/2026-05-16/_summary.json` —
      `flagged` should be empty.
- [ ] From Saturday's real credit cost, confirm Sun + Mon fit the ~666k budget
      (3 full days projected ~700–800k — a small top-up may be needed).
- [ ] Pull Sun May 17 and Mon May 18.
- [ ] Point `build_fr24.py` `PULL_DIRS` at all 3 day folders, `WINDOW_HOURS=24`,
      rebuild `routes_fr24.json`.
- [ ] "Unexpected routes" engine — gravity model (flow ∝ pop·pop / dist^α),
      rank residuals. Needs city population data (GeoNames `cities1000`, free).
- [ ] Pre-publish: confirm FR24 licensing, add attribution, ship no raw data.

**Puller config** (top of `fr24_pull.py`): `DAY`, `HOUR_FROM`/`HOUR_TO` (0/24 =
full day), `MAX_AIRPORTS` (0 = all ~3260, busiest-first), `CREDIT_CAP`,
`STOP_AFTER_FAILS`. Resume is window-aware: an airport file counts as done only
if `ok:true` AND its `from`/`to` match the current window.

## FR24 API — quirks learned (the data-pulling kinks)

- **Cloudflare blocks the default Python user-agent** (error 1010) — every
  request must send a browser `User-Agent` header.
- **`airports` filter:** `outbound:<ICAO>` for departures (ICAO codes).
  Datetime params (`flight_datetime_from`/`_to`) are ISO 8601 `...Z`.
- **`/flight-summary/light` caps at 300 rows per query** (Essential tier) — must
  paginate.
- **`sort=asc` orders by `first_seen`, NOT `datetime_takeoff`.** The pagination
  cursor MUST advance by `first_seen`. (Using takeoff skips records and can push
  `flight_datetime_from` past the window end → HTTP 400 "to must be later than
  from".) This bug cost a real debugging round — do not reintroduce it.
- **Pagination recipe:** `/count` gives the true total; then page `/light` with
  `sort=asc`, advancing `flight_datetime_from` to the last row's `first_seen`,
  dedup by `fr24_id`, until a short (<300) page or `len(recs) >= count`. Stop
  before the cursor reaches `flight_datetime_to`.
- **`aircraft=` filter caps at 15 type codes** — too few to whitelist all
  airliners, so the puller does NOT server-side filter; `build_fr24.py` drops
  non-airliners during aggregation (~3–7% credit overhead on GA, much of which
  self-filters anyway as origin==destination local hops).
- **Credits** are in `x-fr24-credits-remaining` / `x-fr24-credits-consumed`
  headers. Meter spend as the *drop* in the `remaining` balance — robust. Costs:
  `/count` ≈ 0.15 cr/flight, `/light` ≈ 2 cr/flight (recent data).
- **`flight-summary/light` record fields:** `orig_icao`, `dest_icao`,
  `dest_icao_actual` (diversions), `type` (ICAO aircraft code), `datetime_takeoff`
  (null = cancelled), `first_seen`, `operating_as`/`painted_as` (airline ICAO).
  The list is nested under a `data` key.
- **China coverage is good** via the API (schedule-fused) — the original big
  risk, now retired.
- **Sandbox** is not worth using — static canned data, different path structure.
