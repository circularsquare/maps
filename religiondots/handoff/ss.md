# ss — parked 2026-09-15, session `cb8b206e-ss`

*Parked at checkpoint A because the only route needs a download in Anita's name (ask 042), not for context.*
*`AGENT_BRIEF.md` §4: take a parked country before a fresh one, it is cheaper. Do not resume until ask 042
is answered and the files are in `data/raw/ss/`.*

**Update 2026-09-15 (supervisor `cb8b206e`): ready to resume.** Ask 042 is answered: Anita downloaded
`SSD_2015_HFS-W1_v02_M_STATA8.zip` and `SSD_2016_HFS-W2_v02_M_STATA8.zip`, and the supervisor copied both from
`C:\Users\anita\Downloads\` into `data/raw/ss/` (if they are not there, copy them from Downloads). She did not
rule on the grain in words; build the recommendation in ask 042 (the six sampled states, Jonglei, Unity and Upper
Nile empty) unless `ask/RULINGS.md` says otherwise by then. A §14 point is no longer a stop (RULINGS, 2026-09-15).

## Last COMMANDS.txt step completed

Checkpoint A: the source is known, with its catalog pages, file names, variables and access tier. Nothing
downloaded, no COMMANDS.txt build step started.

## What is on disk

Nothing under `data/`, no `sources/ss.py`, no taxonomy module, no `countries/ss.py`. Records only:
`sources/ss.md`, `sources.md` §ss-2026-09-15, ask 042, the `queue.md` and `queue.csv` rows (`held`, ask 042).

## What I was about to do

If Anita downloads them, the four files are:

- World Bank catalog 2778, `SSD_2015_HFS-W1_v02_M` (https://microdata.worldbank.org/index.php/catalog/2778):
  `hhq` (3,550 cases; `state`, `stratum`, `ea`, `urban`, `weight`, `C_9_hhh_religion1`, `C_9_hhh_religion2`,
  `C_9_1_hhh_religion_spec`) and `hhm` (23,004).
- World Bank catalog 2777, `SSD_2016_HFS-W2_v02_M` (https://microdata.worldbank.org/index.php/catalog/2777):
  `hhq` (1,189; `state`, `ea`, `weight_x`, the same `C_9` variables) and `hhm` (8,575).

Then: write `sources/ss.py` (copy `sources/do.py`'s shape for a weighted household-head file), weight heads to
persons, split-half on EAs within states, COD-AB `cod-ab-ssd` v03 admin1 (10 states) and COD-PS 2022 by state,
at the grain ask 042 settles (recommended: sampled states only, Jonglei, Unity and Upper Nile empty).

## The one thing that will bite you

The combined file `hhq_w1_w2` looks like the convenient one and has **no religion variable**; use the two
per-wave `hhq` files. And check whether `weight` ("Population weight based on listing scaled to Census") already
counts persons before multiplying by household size from `hhm`.

## Everything else

All in `sources/ss.md`: the open routes checked and what came back (§2), the level finding against Pew (§2,
last paragraph), the §14 evidence and refugee figures for `gap` (§3), and what to check in the build (§4).
`iri.org` PDFs need curl with a browser user agent; WebFetch gets 403.
