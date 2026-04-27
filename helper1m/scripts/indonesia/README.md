# Indonesia data notes

## Boundaries

HDX COD-AB (BPS, `validOn` 2020-04-01). Reused from `../../data/asia1m/indonesia/` when present. PCODEs (`ID` + BPS numeric) encode hierarchy: `ID13` (province) → `ID1306` (regency) → `ID1306050` (kecamatan).

## Populations — via BPS WebAPI

The public www.bps.go.id site is Cloudflare-protected, but BPS also exposes an official
WebAPI with a Python wrapper called `stadata`. This is the supported path.

### Setup (one-time)

1. Register at https://webapi.bps.go.id/developer/.
2. Create an application → copy the API token.
3. Put it in `helper1m/.env` (gitignored):
   ```
   BPS_API_KEY=your_token_here
   ```
4. `pip install -r helper1m/requirements.txt`

### Domains

Each BPS "domain" corresponds to a statistical area:
- `0000` — central / national (hosts the province + national-regency tables)
- 4-digit codes — province offices (`1100` Aceh, `1200` Sumatera Utara, …)
- 4-digit regency codes — regency offices (`1306` Padang Pariaman, etc.)

Each kecamatan population table lives under the regency domain that owns those
kecamatan. `list_domain()` returns the full directory — no hardcoded index needed.

### Pipeline

1. `python fetch.py probe` — lists candidate tables at the central domain and
   dumps a sample `view_dynamictable` response. Use this to pin down the exact
   field names (variable ID, year column, value column) before wiring up the
   real fetchers. Rerun with `--domain 1306` to inspect a regency.

2. `python fetch.py fetch --levels 1 2` — adm1 + adm2 from the central domain.
   Fast (two API calls).

3. `python fetch.py fetch --levels 3` — adm3 across all 514 regency domains.
   Rate-limited, ~10–15 min. Run this yourself; cached per-domain so reruns skip
   completed regencies.

### Data freshness caveat

adm3 data freshness varies by regency — some publish through 2024, others stop at
2017. The viewer's extrapolation uses the last two adjacent data points so this
is tolerated, but maps in fast-growing regions with stale data will under-count.
`probe` output flags regencies whose latest year is older than 2020.

## Scripts

- `fetch.py` — everything above.
- `regency_index.py` — deprecated. `stadata.list_domain()` replaces it.
