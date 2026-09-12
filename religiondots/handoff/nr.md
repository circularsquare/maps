# nr — parked 2026-09-09, session `f95259a4-pg`

*Written because the coordinator called a wind-down, not because I ran out of context and not
because anything is wrong. Nauru was taken fresh this session, after `pg` closed.*
*`AGENT_BRIEF.md` §4: take a parked country before a fresh one, it is cheaper.*

## Last COMMANDS.txt step completed

**None. Checkpoint A, and not all of it** — the office is found and two routes are retired, but
**the religion table has not been located, so there is no URL to hand over and no category
list.** Step 1 has not started.

## What is on disk

**Nothing under `data/`.** No `sources/nr.py`, no `taxonomy/nr*.py`, no `countries.py` entry.
Two written records, both complete and neither a stub:

- `religiondots/sources/nr.md` — the full scouting record. Read this first; it is 60 lines.
- `religiondots/sources.md` §11am — the part that generalises to other Pacific countries.
- `queue.md`'s `nr` row carries the short version.

Nothing in the scratchpad is worth keeping; every fetch below is a single GET and re-running it
costs seconds.

## What I was about to do

**Fetch `https://stats.gov.nr/documents/` and read the page body for links to census PDFs.**

That is the one uncrawled place left on the office's own site. `stats.gov.nr` is the Nauru Bureau
of Statistics — `nauru.prism.spc.int`, the hostname in the queue row, **301-redirects to it**.
It is plain WordPress (Divi theme) with a live unauthenticated REST API, and **`wp/v2/media` is
already swept: 48 items on one page, page 2 returns HTTP 400, and exactly one non-image file**
(`Nauru-Bulletin_GDP2026.pdf`). So the documents are linked from page bodies, not held in the
media library — `[[reference_wordpress_media_api]]`'s stated case. **No WP File Download plugin
is present**, so the Fiji (§9bd) and PNG (§11ab) AJAX sweep does not apply here.

Also worth reading on the same site: `/statistics/social-statistics/` and
`/category/statistics/`.

If `/documents/` comes up empty, in order:

1. `https://stats-nr.pacificdata.org/` — a **per-country .Stat instance**, front door is an SPA
   shell. Find its SDMX agency id and list its dataflows. The regional SPC agency has none:
   `https://stats-nsi-stable.pacificdata.org/rest/dataflow/SPC/all/latest` answers with no key,
   returns 127 dataflows, and **not one of them is religion, for any Pacific country** (checked
   2026-09-09, §11ab). A national instance can still carry what the regional one does not.
2. `https://naurufinance.info/` — Ministry of Finance, linked from the office front page, not
   probed.
3. `sdd.spc.int` — answers 200 at its root and 403 on `/search`, so it is half-walled.

## The one thing that will bite you

**`nauru.popgis.spc.int`'s *Unavailable service* means "wrong parameters", not "switched off".**

The endpoints take *different* parameters on this instance, and getting it wrong returns a
426-byte exception page that is byte-for-byte the shape §9bh read as a disabled endpoint on the
Solomons:

```
GC_init.php?lang=en                     -> 69 KB of config      (adding obs=main BREAKS it)
GC_listIndics.php?obs=main&lang=en      -> 3.4 MB, 4,437 indicators
GC_refdata.php                          -> Unavailable service either way
```

**Do not re-derive the PopGIS.** It is fully open and it has no religion: 168 datasets, all 2011
Census, at Country / Enumeration Area 2011 / District, with the census's own question ids `p1`
to `p43` and `h1` to `h51`. The only *religio* string in all 3.4 MB is
`P38 … Activities of religious organizations (15+)`, which is an industry of work. A PopGIS
carries a subset of its census, so this does **not** prove the 2011 census skipped religion —
just that PopGIS is not the route.

**Second thing:** the widely-quoted shares (Protestant 60.4% incl. Nauru Congregational 35.7%,
Assembly of God 13%, Nauru Independent Church 9.5%; Roman Catholic 33%; other 3.7%; none 2.5%;
unspecified 0.4%, all *2011 est.*) are **secondary and untraced to any Nauruan release**. They
are useful as a shape to recognise a table by, and must not reach a mapping or a `note_public`.

**Third:** the 2021 Census of Population and Housing exists and PopGIS is 2011 only. **Settle
which census the religion table comes from before writing `grain`**, and prefer 2021 if both
publish it.

## Everything else

All in `sources/nr.md` and `sources.md` §11am — read those rather than re-reading this.

One sizing note so nobody over-invests in geography: at ~11,700 people this is a **§9bf
microstate-tier country**, about twelve dots, and §11aa's list of one-polygon builds names Nauru
explicitly. The 14 districts hold under a dot each. **The category list is the prize; the
geography is not.** A national partition is enough to build this country.
