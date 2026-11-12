# Resource checks

`venv\Scripts\python.exe trainworld/tools/ops/check_resources.py` from the maps root (plain `python` works too;
`--no-r2` skips the minute-long bucket listing). Read-only everywhere: one ssh to the box running a probe from
stdin, `rclone lsjson` on R2, `gh api`, HEAD requests. The orchestrator runs it daily for now and reports every
flagged line (docs/orchestrator.md, Tools). Measurements and how the figures were reached: notes/T-3243.md.

## Reading a line

`FLAG  group / name   use / limit   %   +growth/day   80% in N d   note`

- **OVER** at or past 100%; **HIGH** 50% or more; **SOON** reaches 80% within 30 days at its growth;
  **MISSING** something that should exist or be on is not (a backup timer, a certificate, Google sign-in, the
  service, refusals in the journal).
- **Growth** comes from the run history (`artifacts/trainworld/ops/history.jsonl`, the oldest run 1-7 days back);
  before there is any, from timestamps in the stores (account `created`, play days, share times, the games size
  the server logs at each start). A store growing in bursts gives a rough figure; read the trend over a few days.
- **Limits** in shonei-server's code are read from the local clone (`../shonei-server`), so a raised cap counts
  once it is in that repo. The box runs what was last deployed: if a cap was raised but not deployed, the line is
  optimistic. The outside limits are constants at the top of the script with their source in the table below.

## The lines and what to do

| Line | Limit and where it is set | When flagged |
|---|---|---|
| box / disk, inodes | Hetzner CPX11, 38 GB disk | Grows with the games store. Lower the games cap or add a Hetzner volume; a resize to CPX21 doubles disk. |
| box / memory | 2 GB, no swap | Grows with accounts and opted-in stats, not players. Over 70%: a resize to CPX21 (4 GB), a few minutes with a reboot, her call. |
| box / load | 2 vCPU | Sustained over 1.5: look at the journal for a loop; then a resize. |
| box / traffic out | 1 TB a month included outside the EU (Hetzner price list for US locations; EU is 20 TB). Incoming is free. | Over 50%: check the Cloud Console's traffic graph; overage is billed per TB. |
| box / service | systemd `shonei` | Not active: tell the orchestrator; a restart is a server action, her word. |
| server / trainworld games | `MaxTwGamesBytes`, alert `twGamesWarnAt`, per account `MaxTwAccountBytes` (trainworld_games.go) | Past the cap, saves that grow are refused and games stay in the browser. Proposals in notes/T-3243.md "Games". |
| server / accounts | `MaxNoriAccounts`, `MaxNoriPasswordAccounts` (noritetsu.go) | Past it every sign-up answers 503. Raise the constant (about 8 KB of memory an account) and deploy, before the 80% date. |
| server / sign-ups, busiest hour | the global `newAccounts` bucket (noritetsu.go) | Over 50%: raise it; a busy evening gets "busy" answers past it. |
| server / browsers tracked | `ptMaxBrowsers` (trainworld_playtime.go), `achMaxBrowsers` (achievements) | Past it new browsers are not counted in the stats (plays still are); only the browser figures go wrong. |
| server / plays in a day | `ptMaxPlaysDay` | Past it, that day's extra plays are not counted. |
| server / other stores | ride saves, scores, community, feedback, guestbook, shonei accounts, map-game logs: each a `Max...` constant | Each refuses or trims past its cap; raise the constant if growth is real. |
| server / refusals in the journal | log lines of the last 24 h: sign-up refused, budget reached, over budget, is full, panic, broken | Any of these means players are being refused now: find the line with `ssh root@5.161.112.237 "journalctl -u shonei --since -24h \| grep -E 'refused\|budget\|full\|panic'"`. |
| mail / Resend | Free plan 100 a day, 3,000 a month (resend.com/pricing). The daily line uses the count at which the journal saw Resend's daily-quota error, which overrides the plan figure. | At the cap, recovery-email confirmations and password resets fail until the next UTC day (or month), and the server refuses adding a recovery email meanwhile (T-3105). Pro is $20 a month for 50,000 and no daily cap. Her account: resend.com/settings/billing. |
| backup / off-site | `shonei-backup.timer` and `/etc/shonei-backup.env` (shonei-server deploy/README.md, Off-site backup) | MISSING means no copy outside Hetzner exists. The install is a server action with her R2 token; T-2918. |
| backup / Hetzner images | daily, 7 kept | Not readable from here: Cloud Console, the server, Backups. |
| google / sign-in | the Google client id in the box's `shonei.env`; the OAuth app must be "In production" (no user cap) | MISSING: the last start did not log "Google sign-in on"; look at the env file and the journal. |
| r2 / storage | 10 GB free, then $0.015 per GB-month | Not a stop, a bill. Purge old data versions with `tools/r2_purge.py`. |
| r2 / Class A, Class B | 1 million and 10 million a month free, then $4.50 and $0.36 per million | Real counts need a Cloudflare API token with Account Analytics read in `CF_ANALYTICS_TOKEN` or `~/.cloudflare-analytics-token`; without one, Class A is the objects of this month's data versions and Class B is plays x reads per first load, both lower bounds. Her dashboard: R2, `anitamaps`, Metrics. Class B: a Cache Rule on `data.anita.garden` would turn repeat reads into cache hits, which R2 does not bill. |
| pages / repo, site | GitHub: repos under 1 GB recommended, 5 GB firm; Pages site 1 GB | A publish fails past 1 GB of site. Move big files to R2. |
| pages / bandwidth | GitHub Pages soft 100 GB a month | Estimated from new browsers x bytes Cloudflare passes through to GitHub on a first load. While the game's `.wasm` answers `cf-cache-status: DYNAMIC`, add a Cloudflare Cache Rule "Eligible for cache" on `anita.garden/trainsworld/assets/*` (dashboard, her step; research/launch.md 5.2). |
| certs, domain | Caddy (market) and Cloudflare (site, data) renew themselves; anita.garden's registration | Under 14 days for a cert means renewal is failing; under 60 days for the domain, renew it. |

Two measured constants in the script go stale with the game: `R2_READS_PER_PLAY` and
`PAGES_BYTES_PER_NEW_BROWSER`, from one live first load of `/trainsworld/?new=1`. Re-measure with
`OUT=<scratch> node trainworld/tools/ops/first_load_requests.mjs` after a release that changes what a first load
fetches: R2 reads are the `data.anita.garden` requests, the bytes are `anita.garden DYNAMIC`.
