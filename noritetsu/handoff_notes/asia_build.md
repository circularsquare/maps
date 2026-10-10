# The rest of Asia (kh, la, ph, mm, mn, np) and the Tenerife tram, 2026-10-08: shared-file changes for the managing session

All six countries are built in `dist/data/` (model, tiles, check_model passing; each
`<cc>_sources.md` "Build (2026-10-08)"); their `.osm.pbf` extracts are deleted. Nothing below
has been applied. The builds need none of it except (1) for batch rebuilds; (2)-(4) join the
borders, (5) puts them on the map, (6) is the Canary Islands proposal (not built into dist).

| cc | register lines | km | running | greyed | OSM lines | check_model worst |
|---|---|---|---|---|---|---|
| kh | 3 | 647 | 536 (2 lines) | 111 (1) | none | 0.02 |
| la | 2 | 419 | 419 | 0 | 2 named trains | 0.02 |
| ph | 4 | 421 | 123 (2) | 298 (2) | LRT-1, LRT-2, MRT-3 (58 km) | 0.02 register, 0.06 metro |
| mm | 22 | 4,084 | 1,549 (6) | 2,535 (16) | Yangon Circular + 6 suburban (196 km) | 0.05 |
| mn | 5 | 1,564 | 1,329 (4) | 235 (1) | 3 named trains | 0.02; main line path check 0.995 |
| np | 1 | 49 | 49 | 0 | none | 0.00 |

## 1. tools/rebuild.py: REGISTER entries

```diff
     "do": None, "pr": None,
+    # The rest of Asia: hand lists through rinf.py (asia_register.py on lk_register.py's
+    # engine, lists in asia_lines.py). After an extract: `asia_register.py --clip <cc>`
+    # (also drops asia_lines.DROP_STOPS), then `--join np` / `--join mm` for those two;
+    # `--convert <cc>` once before a first build (build_model needs data/raw/rinf/<cc>).
+    **{cc: f"asia_register:data/raw/rinf/{cc}" for cc in ("kh", "la", "ph", "mm", "mn", "np")},
 }
```

MINUTES: all six build in under a minute (mm 40 s model + 7 s tiles); the default 1 is right.
Gated: no other country maps to asia_register, so no ab.py run is needed.

## 2. borders.py EXTRA: four new crossings

Each is where OSM's track crosses OSM's national boundary (the extract's admin_level=2
relation, read with pyosmium, 2026-10-08). The two Thai crossings with Laos and Cambodia are
already there (th agent).

```diff
     ("eXTZZM1", 32.763516, -9.315615, ["tz", "zm"]),
+    # Laos - China at Boten / Mohan, inside the Friendship Tunnel: OSM way 732797364 over
+    # boundary way 1482078930. The Laos-China Railway's trains (Kunming - Vientiane D887/D888
+    # daily) cross; la's line runs Vientiane - Boten - here (asia agent, 2026-10-08).
+    ("xBotenMohan", 101.687244, 21.179426, ["cn", "la"]),
+    # Mongolia - Russia at Sükhbaatar / Naushki: way 27366656 over boundary way 497684885.
+    # UB - Irkutsk 305/306, Moscow - UB.
+    ("xNaushkiSukhbaatar", 106.096798, 50.334187, ["mn", "ru"]),
+    # Mongolia - China at Zamyn-Üüd / Erenhot: the middle of three tracks (ways 232541719,
+    # 232643093, 757746890) over boundary way 205080589. Beijing - UB K23/K24, Hohhot - UB.
+    ("xZamynUudErenhot", 111.946562, 43.690326, ["cn", "mn"]),
+    # Mongolia - Russia at Ereentsav / Solovyevsk: ways 755589400/1435673744 over boundary way
+    # 129134986. No passenger train found; mn's Choibalsan line is greyed and stops at
+    # Ereentsav station, so nothing reaches the point yet.
+    ("xEreentsavSolovyevsk", 115.742761, 49.885818, ["mn", "ru"]),
+    # India - Nepal at Jaynagar / Inarwa: way 1433667171 over boundary way 534202189.
+    # Nepal Railway Company's Jaynagar - Janakpur - Bhangaha trains (daily).
+    ("xJaynagarInarwa", 86.137098, 26.606175, ["in", "np"]),
 ]
```

Gated on those countries. la, mn and np already end their lines at these ids
(asia_lines.BORDERS); after it lands they need only a rebuild for the points' neutral names
("China – Laos border" and so on; today the points show their ids as names).

## 3. The neighbours' sides of those crossings (their readers, not mine)

The pattern is cn_register's `CN_BORDERS` for Đồng Đăng: the neighbour builds its station -
border piece under the id of OUR line ending at the point, so a ride over the border is one
ride on one line.

- **cn_register.py**, `CN_BORDERS`:
  ```diff
  -CN_BORDERS = [("xFutian", "广深港高速线", None), ("xDongDang", "湘桂线", "vn")]
  +CN_BORDERS = [("xFutian", "广深港高速线", None), ("xDongDang", "湘桂线", "vn"),
  +              # 磨憨 - border, under la's Laos-China Railway (r4749604174)
  +              ("xBotenMohan", "中老昆万线", "la"),
  +              # 二连 - border, under mn's Trans-Mongolian (Ulaanbaatar - Zamyn-Üüd, r113c59c83d)
  +              ("xZamynUudErenhot", "集二线", "mn")]
  ```
  (cn's 磨憨 is on 中老昆万线 `c925f9254e1`, 二连 on 集二线 `c49c6158dbb`, read from
  dist/data/cn 2026-10-08.) Then rebuild cn.
- **Russia, Naushki**: ru's register ends at Наушки (`e300430214`) on "Заудинский — Наушки"
  (`rafed0d2256`); Наушки - xNaushkiSukhbaatar is ~6 km. The ru reader's border mechanism (as
  `ru_register.BORDER` does for the Psou) should run that piece to the point, under mn's
  Trans-Mongolian (Sükhbaatar - Ulaanbaatar) id `r40853c73fe` if it follows cn's pattern. I did
  not look into ru_register's code; the ru owner decides the shape.
- **India, Jaynagar**: in's register ends at Jaynagar station; Jaynagar - xJaynagarInarwa is
  ~3 km. Proposed: a piece under np's line id `r15258d46d2` (in_register), as cn does, so
  Jaynagar -> Janakpur is one ride.
- **Thailand**: nothing. la's Thanaleng line (border - Thanaleng - Khamsavath) already takes
  th's Nong Khai line id `t4a13e49b73` (asia_register `JOIN`, read from dist/data/th), so Nong
  Khai -> Khamsavath is one ride. Cambodia's Battambang - Poipet is greyed and stops at Poipet
  (OSM has no track for the last ~470 m), and th's Eastern line stops at Ban Khlong Luk, so
  nothing meets at `xAranyaprathetPoipet`.

## 4. Rebuild order once (2) and (3) land

`python tools/rebuild.py la mn np cn ru in` (then compare_lines as usual). la/mn/np move only
in their border points' names; cn, ru, in gain their border pieces.

## 5. tools/build_regions.py

Run it to put kh, la, ph, mm, mn, np in regions.json.

## 6. The Tenerife tram (Canary Islands): proposed, not built into dist

Two OSM tram lines, 16 km (canaries_survey.md). Spain's extract is not on disk and es is a big
shared build, so it is not folded into es here. Two routes:

**(a) Its own small region `ic` (least invasive now).** Needs:
- `borders.py`: `NAME["ic"] = "Canary Islands"` and `OUTLINE["ic"] = ROOT / "data" / "raw" /
  "ic" / "outline.geojson"` (written: religiondots' Spain cut to -18.6,27.4,-13.0,29.6, so
  the islands only). build_regions reads both. flag-icons has an `ic` flag.
- `tools/rebuild.py`: `"ic": None` (OSM only, as qa and mu).
- The extract is done (`data/proc/ic`, from canary-islands-latest; the .pbf is deleted).
- **One OSM fix first**: L2's route_master (16267950) carries no tags at all, so the line comes
  out nameless. Either a clip step that copies route 20281470's name/ref/operator/colour onto
  it (nafrica_register.pair_directions writes masters the same way), or a fix in OSM itself.
- Trial build (`build_model.py --region ic --out <scratch>`, 2026-10-08): 2 lines, 25
  stations, 16 route-km: L1 Intercambiador - La Trinidad 12.4 km, 21 stops (en.WP 12.5, 21);
  L2 La Cuesta - Tíncer 3.4 km, 6 stops (3.6, 6); no colour (OSM has blue on L2 only).
- Then `build_tiles.py --region ic`, a KNOWN["ic"] in check_model, build_regions.
- The cost: a rider finds it under "Canary Islands", not Spain, in the Countries tab.

**(b) Into es (the survey's recommendation; better for the rider, more work).** On es's next
rebuild from a fresh extract: `osmium merge spain-latest.osm.pbf canary-islands-latest.osm.pbf
-o es-merged.osm.pbf` and extract es from the merged file (or a repeatable `--pbf` in
extract.py). Adif has no track in the islands, so the RINF register is unaffected; the two
lines come out as OSM trams, with the same L2 naming fix. Needs Spain's ~1.3 GB extract again
and an es rebuild.

My suggestion: (b) when es is next rebuilt anyway; (a) only if the tram is wanted before then.

## Files the agent wrote

New: `asia_register.py`, `asia_lines.py`, `rinf_countries/{kh,la,ph,mm,mn,np}.py`,
`rules/{kh,la,ph,mm,mn,np}.py`, `colours/{kh,la,ph,mm,mn,np}.csv`,
`data/raw/rinf/{kh,la,ph,mm,mn,np}/`, `data/proc/{kh,la,ph,mm,mn,np,ic}/`,
`data/raw/ic/outline.geojson`, `dist/data/{kh,la,ph,mm,mn,np}*`. Edited: `check_model.py`
(REGISTER and KNOWN, "the rest of Asia" blocks), `{kh,la,ph,mm,mn,np}_sources.md` ("Build
(2026-10-08)"). asia_register imports lk_register (engine), and mideast_register.join for
`--join`; a change to lk_register's `register_country`, `L`, `build`, `clip`, `main` or
mideast_register's `join` signatures would break these six.
