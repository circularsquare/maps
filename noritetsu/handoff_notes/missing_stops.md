# Stops missing from the data (2026-10-08)

The open item from 2026-10-07 round 2: lines no picks can finish because their stations were
never marked as stops, and line ids `prune_dead_track` deleted that were real passenger lines.
Working notes of the ex-USSR + Poland agent. Measured with `tools/line_100_probe.js` in
headless Chrome against localhost:8800/noritetsu/ (three batches of countries; a country's
neighbours count only when loaded in the same batch, so border-end failures depend on the
batch).

## The list before (dist/data as of 2026-10-08 morning)

Lines no picks can finish, by country: ru 57 (3,164 km), ua 4 (129), pl 17 (68), kz 2 (25),
uz 3 (25), tm 1 (27), tj 1 (5), by 1 (9), am 1 (10); kg, md, ge, az, xa none. Elsewhere: de
12 (47), ch 6 (18), cz 6 (23), hu 4 (29), lt 2 (21), nl 5 (15), be 2 (2), fr 2 (16), es 5
(48), it 5 (25), ro 1 (8), gr 1 (1), sk 1 (4), fi 1 (3), gb 2 (41), us 5 (62), ca 5 (134), au
16 (310), th 1 (3), dz 1 (14); jp, kr, tw, hk, sg, cn, in, id, my, vn, ir, ma, tn, eg, za, ar,
cl, br, mx, nz, at, si, hr, lv, ee, se, no, dk, lu, pt, bg, rs, ba, me, mk, al, xk, ie, tr
none.

Why, in the ex-USSR countries:

1. **Clone junctions never folded back into their stops (ru, ua; the big one).**
   ru_register and ua_register end a long stop-to-stop stretch at clones of its end stops
   (`<code>@<section>`, joined to the stop by a 0 km link) so that build_model judges it by
   OSM's routes; `rinf.split_pieces` folds each clone back into its stop afterwards. It can
   only fold into a stop that has a station record, and rinf.build makes records only for
   nodes of the traced sections. Where every pair beside a stop is a stretch (all of Томмот —
   Нижний Бестях, Карталы I — Никель, Мариинск — Ачинск I, Светлоград — Элиста, Сыня —
   Усинск, Егоршино — Устье-Аха ...; ua Арциз — Ізмаїл, Вербка — Камінь-Каширський's end), the
   stop had no record, the clone stayed a junction, and the line shipped with 0 or 1 stops.
   ru: 179 register lines carried clone ids, 68 lines (3,634 km) had fewer than 2 stops.
   The fix is in rinf.py (shared): the diff below.
2. **Book 2 lags behind the trains (ru).** A point is a stop when Book 2 gives it a passenger
   operation (П, Б, О) and an OSM train route stops within 400 m. Воткинск, Переславль,
   Волгореченск, Неман-Новый, Готня, Соломбалка, Пяозеро, Великие Луки, Валуйки ... have no
   such letter, yet OSM's trains stop there by name; the line's end was a junction. Fixed in
   ru_register.py (`named_stop`, `OSM_NAMED_STOP_M` 600 m): 122 points became stops.
3. **No OSM station (pl).** Olecko: PKP Intercity's Gryf calls there, OSM has no station
   node, so rinf made it a junction and line 39 Suwałki - Olecko (42.8 km) was deleted.
   Fixed in rinf_countries/pl.py (`stop_name` hook). Line 41 Ełk - Olecko (27.4 km) comes
   with it.
4. **Not missing stops** (left): ends at a border point whose stops lie in the neighbour (kz
   Lokot - Zyryanovsk 20 km, Semiglavy Mar 5; uz Xojeli and Takhiatash to Turkmenistan, Bekobod
   to Tajikistan; tj 4613 km; by Hudahaj - Lithuania 9; pl's border lines 295, 408, 290, 346,
   348, 276, 291, 93, 51, 299; ua Chop - Slovakia/Hungary); tm Ashgabat - Owadan Depe 27 km (a
   junction end towards the Daşoguz line, no train stops at Owadan Depe in railway.gov.tm's
   schedule); am Masis - Karmir-Blur 10 km (Karmir-Blur is a freight station, Book 2 ops
   3,5,8, and the line leads nowhere else: a freight line an OSM route runs over).

## Deleted ids 2026-10-07, checked

| cc | line | verdict |
|---|---|---|
| ua | Миронівка — Богуслав 17.7 km | closed: uk.wikipedia, "station closed in 2020 with the whole 17 km branch from Myronivka"; no train on poizdato |
| ua | Богодухів — Гути 9.7, Верхньодніпровськ — Дніпровська 6.7 | no trains (poizdato has no Huty or Dniprovska pages) |
| kz | Тараз — Жаңатас 66.5, Ақтаутас — Бугунь 13.2 | no passenger train in KTZ's crawl (every station with an Express code) nor found elsewhere; phosphate freight |
| uz | Nókis — Shımbay 56.2 | no passenger train found (UTY's Nukus trains go to Tashkent and Mangistau) |
| pl | 39 Suwałki - Olecko 42.8 | **real**: Gryf; back since this rebuild |
| pl | 200, 303, 973, 259 | handovers to industrial / regional owners, no trains |
| ru | 11 ids (Кашира-Товарная, Ай — Титан, Троицк-ГРЭС ...) | yards and industrial spurs |
| by | Віцебск — Прыдзвінская 18.6 | freight |

## The shared diff (rinf.py), for the managing session

After the station-record loop in `build()` (line ~1921, just before `rel_names = ...`):

```diff
@@ rinf.py, build(), after "station records"
                 if conf.get("station_en") and not stations[n]["name_en"]:
                     stations[n]["name_en"] = conf["station_en"](points.get(op, {}),
                                                                 stations[n]["name"]) or ""
+        # A 0 km link's stop that no traced section of this line reaches (every pair beside it
+        # a stretch ending at its clone: all of Томмот — Нижний Бестях) still needs its record,
+        # or split_pieces cannot fold the clone back into it and the line ships with no stop.
+        for _j, n in links:
+            if n in stations:
+                continue
+            op = op_of_node.get(n)
+            if op in stop_of:
+                o = ost[stop_of[op]]
+                stations[n] = {"id": n, "name": o["name"], "name_en": o["name_en"],
+                               "lon": o["lon"], "lat": o["lat"], "lines": set()}
+                if conf.get("station_en") and not stations[n]["name_en"]:
+                    stations[n]["name_en"] = conf["station_en"](points.get(op, {}),
+                                                                stations[n]["name"]) or ""
         rel_names = [rel_name[r] for lid in lids for r in [rel_of.get(lid) or named_rel.get(lid)]
```

Only registers with 0 km links move (ru, ua; the others' `links` are empty or their stops
already have records), so a run of ab.py on ru and ua is enough. A record left with no lines
(its stretch dropped) is not shipped (checked: 0 stations with no lines in the trial).

**Trial** (scratch wrapper emulating the diff by wrapping rinf.build, build_model --out to a
temp folder; Book 1 conversion as before the named-stop change):
- ru: link junctions folded 1,145 -> 1,578 (405 link stops given a record); register lines
  with clone ids left 179 -> 0; with fewer than 2 stops 68 (3,634 km) -> 42 (841 km); register
  km unchanged (77,786.8); one line more: Арсентьевка — Победино-Сахалинское splits into two
  pieces over 25 km apart (155.5 + 54.9 km, new id r335708fb8a); station ids 447 gone (408
  clone junctions, 39 OSM stations merged into the register stops they stand on), all 1,592
  carried over as aliases, 0 with nothing in reach. Томмот — Нижний Бестях 0 -> 4 stops,
  Карталы I — Никель 8 of 9 ends stops, Мариинск — Ачинск I 6 of 6.
- ua: folded 117 -> 136 (25 link stops given a record); register km and line count
  unchanged (15,662.6 km, 317 lines); 19 clone junction ids gone, all carried over (360 ids
  aliased, 0 with nothing in reach); Арциз — Ізмаїл 1 -> 8 stops (all), Вербка —
  Камінь-Каширський's Kamin-Kashyrskyi end a stop, Одеса-Пересип — Колосівка's Serbka end a
  stop.

## Rebuilt by this agent (own files only, 2026-10-08)

- **pl** (rinf_countries/pl.py `stop_name`): compare_lines 358 -> 360 register lines, the two
  new ones line 39 Suwałki - Olecko 42.8 km and line 41 Ełk - Olecko 27.4 km; nothing else
  moved. check_model: worst register deviation 0.17 (line 38, as before).
- **ru** (ru_register.py named stops, with the veto and placement guards; rebuilt three times,
  the first two had faults the guards now prevent): compare_lines 919 -> 924 register lines,
  22 differ: five new short lines from new stops (Кусково — Перово 2.0, Москва-Рижская —
  Подмосковная 2.0, Сандарово — Михнево 6.7, Аксарайская — Кигаш 5.6, Владивосток — Мыс-Чуркин
  8.2), Ачинск I — Уяр +2.1 km, Люблино-Сортировочное — Столбовая +0.2, the rest under 0.1 km.
  No stop of the morning's build lost its id (the second rebuild checked Салми, Лебедянь,
  Картымская back). check_model: median 0.995 over 912 lines, worst 0.35 (as before).
  Probe: ru 57 lines / 3,164 km -> 52 / 3,087. The rest waits for the rinf.py diff.
- ua, kz, uz, tm, tj, by, am: nothing to rebuild (ua's are the rinf.py fault; the others are
  border ends or not stop problems).

Expected after the diff lands (trial with the old conversion, so a lower bound): ru lines with
fewer than two stops 42 instead of 68; with the named stops, Пяозеро, Волгореченск,
Переславль, Неман-Новый, Соломбалка get their stop too (they are stretch ends whose stop
records the diff creates).

After landing: `compare_lines save ru ua`, `rebuild.py ru ua` (ru takes ~50 min now, of which
along.py 38), `compare_lines diff`, `check_model`, `build_regions.py`.
