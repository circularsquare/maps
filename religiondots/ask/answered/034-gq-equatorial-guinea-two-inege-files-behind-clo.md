# 034 — gq: Equatorial Guinea: two INEGE files behind Cloudflare

Summary: Equatorial Guinea's 2015 census asked religion; two preliminary-results files on inege.org sit behind a Cloudflare challenge. Could you download them in a browser into data/raw/gq/? Details in queue.md, 'Old negatives re-checked, 2026-09-15'.

*Filed 2026-09-15 by session `cb8b206e` (supervisor), from scout `cb8b206e-scout1`. Anita's call; nothing is waiting on it.*

## What I did

Equatorial Guinea stays off the map, now marked blocked instead of closed: the 2015 census form asks religion, but no table by province has turned up in anything an agent can reach.

## What it costs to reverse

Nothing is built. If a file holds religion by province, it is an ordinary census-table build, one agent session.

## Why it is yours rather than mine

It is a download only you can make: `inege.org` answers every client with a Cloudflare challenge, and the standing rule is to hand over the address rather than retry.

## The detail

- **Update 2026-09-15 (cb8b206e-gq):** the `RESULTADOS-DEFINITIVOS-...2015.pdf` you downloaded is byte-identical to the 72-page scan already read (same SHA-256): no religion or ethnic table. No need to fetch it again.
- **What to fetch**, into `data/raw/gq/`, opened in a browser:
  - *Síntesis de los resultados preliminares del censo 2015*: `https://inege.org/?wpfd_file=sintesis-de-los-resultados-preliminares-del-censo-2015`
  - *Resultados preliminares del IV censo de población 2015*: `https://inege.org/?wpfd_file=resultados-preliminares-del-iv-censo-de-poblacion-2015`
  - the page `https://inege.org/wp-json/wp/v2/media?search=religi`, saved as it comes (it lists any uploaded file with "religi" in its name).
- **Honest odds:** the preliminary results were presented a few weeks after fieldwork, so they may be headcounts only. A religion table is possible, not likely.
- **The question exists**: 2015 form, Bloque V question 9, *¿Qué religión profesa?* (none, Catholic, Protestant, Islam, other). Read from the UNSD archive copy.
- **Already read, no religion table**: the 72-page *Resultados definitivos* 2015, *Guinea Ecuatorial en Cifras 2021*, INEGE's 2023 first report, and three yearbook files from the Wayback Machine.
- **If neither file has religion by province**, Equatorial Guinea goes back to closed and nothing else is needed from you.
