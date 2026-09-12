# Nauru — scouting record, 2026-09-09

**State: CHECKPOINT A, parked. Nothing downloaded, nothing on disk.** The country is almost
certainly buildable; what is missing is the census table itself, and the last uncrawled place it
can be is `stats.gov.nr/documents/`.

Claimed and parked by session `f95259a4-pg` on 2026-09-09, on the coordinator's instruction to
wind down, not because anything went wrong.

## Why it is worth finishing

~11,700 people, so it is a **§9bf microstate-tier country**: §11aa's list of "roughly fifteen to
eighteen" one-polygon builds names Nauru explicitly. At 1 dot = 1,000 people it is about twelve
dots, and the 14 districts hold under a dot each, so **the tier is not the interesting question
here — having the category list is.** A national partition is enough.

The widely-cited figures (CIA World Factbook, *2011 est.*) are Protestant 60.4% — Nauru
Congregational 35.7%, Assembly of God 13%, Nauru Independent Church 9.5% — Roman Catholic 33%,
other 3.7%, none 2.5%, unspecified 0.4%. **Those are secondary and none of them has been traced
to a Nauruan release yet.** Do not put them in a mapping; find the census table.

## The office moved, and it is wide open

`nauru.prism.spc.int` **301-redirects to `stats.gov.nr`** — the Nauru Bureau of Statistics has
its own domain now, and the queue row's hostname is a redirect rather than the office.
`[[reference_dead_stats_office]]` in the ordinary direction.

WordPress, REST API live and unauthenticated:

```
https://stats.gov.nr/wp-json/wp/v2/media?per_page=100&page=<n>
```

**Swept 2026-09-09 and it is nearly empty: 48 items, one page (page 2 returns 400), and exactly
one non-image — `Nauru-Bulletin_GDP2026.pdf`.** So this is
`[[reference_wordpress_media_api]]`'s stated case: the media library is not a superset of
`wp-content/uploads`, and the census PDFs are not in it.

**The next step, and it is the whole remaining task at checkpoint A:** the site's own
`https://stats.gov.nr/documents/` page, plus `https://stats.gov.nr/statistics/social-statistics/`
and `https://stats.gov.nr/category/statistics/`. Read the page bodies for links, the way §11ah's
Burundi and §9bd's Fiji were done — page bodies before any API, per
`[[reference_wpfd_sweep]]`. Nauru's site is plain WordPress with a Divi theme; **no WP File
Download plugin was seen**, so the Fiji/PNG AJAX route does not apply here.

Other hosts linked from the office's own front page, both live, both worth a look if
`/documents/` comes up empty:

* `https://stats-nr.pacificdata.org/` — a **per-country .Stat instance**, SPA shell only on the
  front door. The SPC-agency SDMX service (`stats-nsi-stable.pacificdata.org/rest/dataflow/SPC/all/latest`,
  no key, 127 dataflows) has **no religion dataflow for any Pacific country** (checked 2026-09-09,
  §11ab), but a *national* .Stat can carry dataflows the regional agency does not. Not yet probed
  for its own agency id.
* `https://naurufinance.info/` — Ministry of Finance, not probed.

## SPC's PopGIS is live, open, and has no religion

`nauru.popgis.spc.int` is a **GeoClip PopGIS 3**, the same software as Fiji's (§9bd §5), and the
endpoints answer without a key. Two parameter facts, because they are not the same as Fiji's and
cost half an hour:

* **`GC_init.php` needs `lang` and NOT `obs`.** `GC_init.php?lang=en` returns 69 KB of config;
  adding `obs=main` returns a 426-byte *Unavailable service* exception that looks exactly like
  Solomon Islands' disabled endpoint and is not one.
* **`GC_listIndics.php` needs `obs=main`.** `GC_listIndics.php?obs=main&lang=en` returns
  **3.4 MB, 4,437 indicators**. `GC_refdata.php` is *Unavailable service* with either parameter.

What it holds: **168 datasets, all 2011 Census, at three geographic levels — Country,
Enumeration Area (2011) and District.** The indicator ids are census question numbers, `p1` to
`p43` and `h1` to `h51`, and **there is no religion question among them**: `p1_ethnic`,
`p17_tribe`, `p14_nationality`, `p7_country_birth`, marital status, disability, education,
language, labour, and the housing block. The only string matching *religio* in all 3.4 MB is
`P38. Name of the main industry population work - Activities of religious organizations (15+)`,
which is an industry-of-work category and not affiliation.

**So this is a fourth distinct PopGIS outcome**, after Fiji served the data, Solomon Islands had
the dataset with the door shut, and Vanuatu had no religion theme: Nauru's instance is fully open
and the census question simply is not in it. That does **not** settle whether the 2011 census
asked religion — the widely-cited 2011 shares say it did, and PopGIS carries a subset of any
census — but it does mean **PopGIS is not the route and should not be re-derived.**

## Vintage to settle before mapping

The 2021 Census of Population and Housing exists and PopGIS is 2011 only. **Check which census
the religion table comes from before writing `grain`**, and prefer 2021 if both publish it.

## Where a next session should start

1. `https://stats.gov.nr/documents/` — read the page body, list the PDFs, find the census report.
2. If that fails, `stats-nr.pacificdata.org` — find its SDMX agency id and list its dataflows.
3. If both fail, the SPC digital library (`sdd.spc.int` answers 200 at its root, 403 on
   `/search`) and the Pacific walls in §11ab, all of which are browser jobs.

Nothing has been claimed about what Nauru does or does not publish. **This is a scouting note,
not a negative.**
