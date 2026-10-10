# kp (North Korea) build, 2026-10-08: shared diffs for the managing session

kp is built (`dist/data/kp/`, `dist/data/kp.pmtiles`); kp_sources.md "Build" and "What runs"
have the detail. Files the kp agent owns and changed: kp_register.py (new), rules/kp.py (new),
kp_sources.md, check_model.py `REGISTER["kp"]` and `KNOWN["kp"]`, data/proc/kp, dist/data/kp*.
The .pbf is deleted. Nothing below is applied.

## 1. tools/rebuild.py REGISTER

```diff
     "il": "il_register:data/raw/il",
+    # North Korea: OSM named track + line relations, th_register's recipe (kp_register.py;
+    # `python kp_register.py --clip` after an extract; kp_sources.md).
+    "kp": "kp_register:data/raw/kp",
 }
```
and in MINUTES `"kp": 1`.

## 2. borders.py EXTRA

```diff
     ("xJaynagarInarwa", 86.137098, 26.606175, ["in", "np"]),
+    # North Korea (kp agent, 2026-10-08): where OSM's track crosses OSM's boundary.
+    # Sinŭiju - Dandong on the Friendship Bridge (way 288979790): K27/28 Beijing - P'yŏngyang
+    # four a week, Dandong - P'yŏngyang daily since 12 March 2026.
+    ("xSinuijuDandong", 124.392329, 40.115133, ["cn", "kp"]),
+    # Tumangang - Khasan on the Tumen bridge (ways 42510186 / 1475326893, boundary way
+    # 140244396): 645/646 Khasan - Tumangang three a week, Moscow through cars.
+    ("eXKPRUTUMANGANG", 130.641271, 42.415217, ["kp", "ru"]),
+    # Manp'o - Ji'an (way 839022580): one passenger car on the daily freight (en.wikipedia
+    # Manp'o Line), for DPRK citizens and ethnic Koreans from China.
+    ("xManpoJian", 126.273205, 41.154877, ["cn", "kp"]),
 ]
```
Not offered: Namyang - Tumen (129.849287, 42.949015, way 199109778) and Ch'ŏngsu -
Shanghekou (124.879328, 40.458402, way 838948375), freight only. kp_register.BORDERS carries
the same three points until these land (it uses a borders.load() point within 50 m instead).

## 3. cn_register.py CN_BORDERS (then rebuild cn)

```diff
               ("xBotenMohan", "中老昆万线", "la"),
-              ("xZamynUudErenhot", "集二线", "mn")]
+              ("xZamynUudErenhot", "集二线", "mn"),
+              # 2026-10-08, kp built (handoff_notes/kp_build.md): 丹东 - the Yalu bridge, K27/28
+              # and Dandong - P'yŏngyang; 集安 - the Ji'an bridge, the one car to Manp'o.
+              ("xSinuijuDandong", "沈丹线", "kp"),
+              ("xManpoJian", "梅集线", "kp")]
```
Both line names are in cn's shipped lines.json. kp's lines ending there: 평의선 (ends at
xSinuijuDandong), 만포선 (ends at xManpoJian). Note cn_sources.md: cn's own extract holds
North Korea's northern lines, which `cn_register.py --clip` removes; keep it so.

## 4. ru_register.py BORDER (then rebuild ru); a note for the ru side

```diff
+    # Tumangang (North Korea, kp built 2026-10-08): 96-045 ends at "Хасан (эксп.)" 987106,
+    # 2 tariff km past Хасан, the Tumen bridge. 645/646 Khasan - Tumangang, Moscow cars.
+    ("XKPRUTUMANGANG", "987002", "987106", "at"),   # Хасан (96-045)
```
(The point's id is "e" + "XKPRUTUMANGANG", as ru's BORDER wants.) Check after the rebuild
that 96-045's piece reaches the point (Хасан station is ~1.6 km from the bridge).

## 5. After landing

`python tools/build_regions.py` (kp is not yet in regions.json), then rebuild cn and ru
for their sides. kp itself needs no rebuild for 2-4 (its BORDERS already places the points);
rebuild kp only if borders.load() then moves a point.

## Open, not done

- The Pyongyang tram route_masters 11495026 / 11495033 have no name in OSM, so two tram
  lines are called "2" and "3" (T2, T3). The routes are named "T2 : 문수>토성" etc. A
  build_model fallback (route_master with no name -> its first route's name before " : ")
  would fix it; not proposed as a diff, it touches every country.
- Lines with trains in en.wikipedia but no named OSM track: 명당선, 청년팔원선, 강덕선,
  다사도선, 수풍선 (kp_sources.md). The Sŏho line is in as an OSM line.
- 함북선 Onsŏng - Mulgol (103 km) is greyed for want of a named train; the least certain grey.
