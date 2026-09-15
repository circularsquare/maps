# 027 — mt: Malta's 2021 census PDF needs a browser download

Summary: Malta's 2021 census asked religion, apparently with district tables, but nso.gov.mt returns 403 to every client here. Could you download the volume 1 PDF in a browser and drop it in data/raw/mt/? Nothing is waiting on it.

*Filed 2026-09-15 by session `d743fc47`. Anita's call; nothing is waiting on it.*

## What I did

Left Malta `blocked` in `queue.csv`, as the Europe scout recorded it. Nothing is built.

## What it costs to reverse

Nothing to undo. Once the PDF is on disk, Malta is one builder session.

## Why it is yours rather than mine

A download behind a client-side wall is yours: the standing rule is to hand you the URL rather than
work around a 403 (`CLAUDE.md`, Blocked downloads).

## The detail

- **The file:** https://nso.gov.mt/wp-content/uploads/Census-of-Population-2021-volume1-final.pdf
- **Where to put it:** `C:\Users\anita\projects\maps\religiondots\data\raw\mt\` (create the folder).
- **What was tried:** `nso.gov.mt` returns 403 to curl and to WebFetch, and the Wayback Machine was
  offline during the sweep (`sources.md` §scout-2026-09-15-europe).
- **Why it is worth it:** the scout calls Malta probably the best find of the Europe sweep. The 2021
  census asked religion, and the volume appears to carry district tables.
- If volume 1 turns out not to hold religion, the same site's other census volumes are the next place
  to look.


---

## Ruled 2026-09-15 by Anita: downloaded

*"ok downloaded to downloads"*

Copied from her Downloads folder to `data/raw/mt/`; Malta is `free` in `queue.csv`.
