# Vietnam (vn): sources, decisions, what is off

Built 2026-10-03 by the vn agent. Register reader `vn_register.py`, rules `rules/vn.py`.

    python extract.py --region vn --pbf data/raw/vietnam-latest.osm.pbf --station-areas
    python vn_register.py --clip            # after every extract (China's track, lines not open)
    python vn_register.py --report          # way-to-line assignment, km per line
    python build_model.py --region vn --register vn_register:data/raw/vn
    python build_tiles.py --region vn
    python check_model.py --region vn

`--station-areas` matters: nine stations between Biên Hòa and Suối Kiết (Long Khánh, Dầu Giây,
Gia Ray...) and Vinh are mapped only as station buildings; without it Suối Kiết - Biên Hòa was
one 94 km section. Build time: about 30 s for the model, 4 s for the tiles.

## Result (2026-10-03)

10 register lines, 2,456 km (Kép - Cái Lân's 109 km of it greyed), 272 register stations; 20
lines in all (the 10, 3 metro lines, 7 named trains), 306 stations. check_model:

| line | built km | published | source |
|---|---|---|---|
| Đường sắt Bắc Nam (North-South) | 1,722.4 | 1,726 | VI infobox, Hà Nội - Sài Gòn. Also checked section by section against the 175-station chainage table: km_official 1,727.5, ratio 0.997 |
| Đường sắt Hà Nội - Lào Cai | 282.1 | 283 | VI chainage, Yên Viên km 11 - Lào Cai km 294 |
| Đường sắt Hà Nội - Đồng Đăng | 165.3 | 166.5 | VI 162 to Đồng Đăng + 4.5 km to the border |
| Đường sắt Hà Nội - Hải Phòng | 95.9 | 97 | VI 102 less Hà Nội - Gia Lâm (km 5) |
| Đường sắt Hà Nội - Quan Triều | 53.5 | 54 | VI 75 less Hà Nội - Đông Anh (km 21) |
| Đường sắt Kép - Cái Lân (greyed) | 109.1 | 109.4 | en.WP Kép - Hạ Long 106 + Hạ Long - Cái Lân 3.4 |
| Đường sắt Diêu Trì - Quy Nhơn | 9.9 | 10.5 | VI (0.94: measured from Diêu Trì station along the branch's own track) |
| Đường sắt Bình Thuận - Phan Thiết | 9.5 | ~10 | VI "Ga Phan Thiết" |
| Đường sắt Đà Lạt - Trại Mát | 6.5 | 7 | VI / en.WP, a round figure (0.93) |
| Tàu hỏa leo núi Mường Hoa | 1.7 | 2 | Sun World, a round figure; OSM's whole track is 1.69 km (0.84) |

Metros (OSM lines): Hà Nội 2A 12.6 km / 12 stations against 13.05 / 12; Hà Nội 3 7.7 / 8
against 8.5 / 8 (the elevated Nhổn - Cầu Giấy; published figure with the depot tail); HCMC 1
18.8 / 14 against 19.7 / 14.

Sources: VI = the vi.wikipedia line articles (raw wikitext kept in `data/raw/vn/viwiki/`), and
for the North-South line "Danh sách nhà ga thuộc tuyến đường sắt Thống Nhất" (175 stations with
VNR's chainage, read by `vn_register.chainage`); en.WP "Rail transport in Vietnam"; Wikidata
P2043 (`data/raw/vn/wikidata_lines.json`: 1,736 for the North-South line, 296, 168, 102); the
Vietnam Railway Authority's 2025 level-crossing list (`data/raw/vn/drvn_thong_tin_tai_trong_2025.pdf`)
for the official list of lines and their extents.

## Where the register comes from

No open line register with geometry exists. The Vietnam Railway Authority (Cục Đường sắt Việt
Nam, DRVN) lists the national lines in its 2025 publication of level crossings: Hà Nội - Tp.
Hồ Chí Minh, Yên Viên - Lào Cai, Phố Lu - Pom Hán, Đông Anh - Quán Triều, Bắc Hồng - Văn Điển,
Hà Nội - Đồng Đăng, Kép - Hạ Long - Cái Lân, Chí Linh - Phả Lại, Mai Pha - Na Dương, Gia Lâm -
Hải Phòng, Cầu Giát - Nghĩa Đàn, Diêu Trì - Quy Nhơn, Đà Lạt - Trại Mát, Bình Thuận - Phan
Thiết, and four port and works branches. OSM Vietnam names 98% of its main-line track for the
line (`probe_kr_ways.py --region vn`), with exactly DRVN's extents: the trunk Hà Nội - Gia Lâm
- Yên Viên is named for the Đồng Đăng line, the Hải Phòng line starts at Gia Lâm, the Lào Cai
line at Yên Viên, the Quan Triều line at Đông Anh. So kr_register's recipe works as it does in
Thailand (th_register): each way gets a line by its name (`NAME_LINE`; bridge and tunnel names
"Cầu ...", "Hầm ..." read as no name), else by the line relation it is in, else by its
neighbours (`propagate`). Repairs:

- **Welds**: five slips in OSM's track (dead ends within 120 m of the same line's next way),
  the worst 78 m on the Lào Cai line west of Bắc Hồng, which had cut the line in two (`weld`).
- **Stations by proximity**: every OSM rail station within 250 m of a line's track (track
  densified to a point every 25 m; Chợ Sy and Hương Phố stand 260 m from the nearest vertex)
  is on that line; a branch takes the station nearest its junction end (Gia Lâm 432 m, Yên
  Viên 382 m, Diêu Trì 266 m). Station names lose a leading "Ga " (32 records, mostly the
  station areas: "Ga Vinh").
- **Left out of the station records**: the metros' records (338), Thống Nhất park's mini-train
  stop "Sai Gon" 1.1 km south of Hà Nội station (it had become a station of the North-South
  line), and "Trạm đường sắt Thượng Cát", an operating post of Gia Lâm station ("Trạm bổ trợ ga
  Gia Lâm") where the Hải Phòng line would otherwise have begun.
- **Đà Lạt station** is mapped only as a multipolygon (relation 17877171), which extract.py
  does not read: given by hand at its centre (`EXTRA_STATIONS`).
- **Shortcuts** dropped (gb_register.drop_shortcuts): Lệ Trạch - Kim Liên (22 km, the track
  past Đà Nẵng station, where trains reverse) and Cây Cầy - Lương Sơn (23.7 km, past Nha
  Trang); every passenger train calls at both.
- **The North-South chainage**: all 173 sections have a published km, so the line carries
  km_official (1,727.5; it counts the run into Đà Nẵng and out again).

`--clip` takes out what is not Vietnam's or not open: track with half its nodes outside OSM's
boundary of Vietnam (relation 49915, `data/raw/vn/vn_boundary.geojson`): Hekou's metre-gauge
yard and 昆河线, the standard-gauge line towards Pingxiang, the Hồ Kiều bridge; the route
relations of lines not open (Hà Nội lines 2, 1, 8, 10, 14; HCMC line 2; the Kép - Cái Lân
relation; an unnamed four-way route), and the track of the planned and building metro lines,
which OSM tags railway=rail or subway under their future names (`NOT_OPEN_TRACK`).

## Decisions (made here, not put to Anita)

- **Line unit**: DRVN's lines, under the names everyone uses (vi.wikipedia, Wikidata, OSM's
  own line relations): "Đường sắt Hà Nội - Lào Cai" for DRVN's Yên Viên - Lào Cai and so on.
  VNR counts each northern line from Hà Nội station; here the shared trunk out of Hà Nội is the
  Đồng Đăng line's, as DRVN and OSM have it, so no track is two lines'.
- **Named trains**: every VNR train is a numbered single train (SE1-SE22, TN, NA, SNT, SQN,
  SPT, SP, LC, HP, LP, QT, DL, QN, HĐ, MR...), mostly once a day each way. Each is a named train
  (option B), as in Thailand: the line a rider uses is the register line all of them run over.
  OSM maps only the Hải Phòng trains (HP1/2, LP2/3/5/6/7/8), one relation per number; they come
  out as 7 named trains. OSM's route=train relations named for the lines themselves
  ("Đường sắt Hà Nội - Lào Cai", ref ĐSHN-LC) are not trains: the North-South one is dropped as
  the register line's twin (`TWIN_ON_STATIONS`), the others list under two stops and make no
  line.
- **Kép - Cái Lân: built, greyed** (`suspended`). The last passenger train (Yên Viên - Hạ Long)
  ran in 2020-21; Hạ Long station has had no train since (vi.wikipedia "Ga Hạ Long"; dantri,
  18 Jul 2025, "Ga Hạ Long hoang vắng, chờ tàu trở lại sau gần 4 năm"). Ticket-agency pages
  still list 51501/51502; they are stale. Greyed rather than left out, as for any line that
  had trains (Anita's 2026-09-30 rule).
- **Running, kept**: Hà Nội - Hải Phòng (4 pairs a day), Lào Cai (SP3/4, SP7/8), Quan Triều
  (QT trains, vetau247 2025), Đồng Đăng (MR1/MR2 Gia Lâm - Nanning daily since 25 May 2025, plus
  the Beijing through cars), Diêu Trì - Quy Nhơn (SQN trains and the QN1-QN4 tourist shuttles,
  two pairs a day), Bình Thuận - Phan Thiết (SPT1/2, Friday to Sunday until 30 Dec 2026: more
  than once a week), Đà Lạt - Trại Mát (DL1-DL12, 5-6 pairs a day; its track is tagged
  usage=tourism, which this reader does not filter). Timetables: Đường sắt Sài Gòn's post-summer
  2026 sale notice (cophanvantaiduongsat.vn, 3 Jul 2026), vinwonders/gialai.gov.vn for Quy Nhơn.
- **Freight only, no register line**: Bắc Hồng - Văn Điển, Yên Trạch (Mai Pha) - Na Dương,
  Kép - Lưu Xá, Quan Triều - Núi Hồng, Phố Lu - Pom Hán - Tằng Loỏng, Phủ Lý - Thịnh Châu,
  Dung Quất; Chí Linh - Phả Lại (part of Yên Viên - Phả Lại - Hạ Long, never finished);
  Lào Cai - Hekou over the Hồ Kiều bridge (metre-gauge freight; no passenger train since 2002).
- **Mountain railways**: the Mường Hoa railway at Sa Pa (Sun World, a 2 km two-car Garaventa
  funicular from Sa Pa town to the Fansipan cable car, daily since 2018) is a register line:
  it runs from the town's own station on a public route. The short funicular inside the
  Fansipan summit complex and the Bà Nà Hills funicular (resort-internal, no stations in OSM)
  are left out; so are the amusement monorails (Đầm Sen, Hạ Long) and the Thống Nhất park
  mini-train.
- **Metros**: Hà Nội 2A and 3 (Nhổn - Cầu Giấy only; the underground Cầu Giấy - Ga Hà Nội opens
  end 2027) and HCMC 1 are OSM lines. HCMC 2 and Hà Nội 2 are being built; Hà Nội 1, 8, 10, 14
  are plans; all clipped.

## Borders

| id | lon | lat | countries | trains over it |
|---|---|---|---|---|
| xDongDang | 106.715575 | 21.972213 | cn, vn | yes: MR1/MR2 Gia Lâm - Nanning, daily since 25 May 2025 (suspended a few days in July 2026 for rain). Where OSM's track (way 482593999) crosses OSM's boundary of Vietnam, at Hữu Nghị Quan |

Not in `borders.EXTRA` yet: vn_register offers it as a station itself (`BORDERS`), so the Đồng
Đăng line runs 4.5 km to it, kept by `served_sections`. Lào Cai - Hekou has no point: freight
only. China's side (Pingxiang - border, about 11 km) is not built by cn: its register line 湘桂线
ends at 凭祥 and no OSM route in cn's extract crosses into Vietnam.

## Timetables

None. The Mobility Database catalogue (feeds_v2.csv, 2026-10-03) has no Vietnamese feed, rail
or otherwise; VNR publishes none. No `gtfs_served.FEEDS["vn"]`.

## Open

- OSM's Kà Rôm station stands about 7.5 km from where VNR's chainage puts it (Cam Thịnh Đông -
  Kà Rôm 17.6 built against 10.1; Kà Rôm - Phước Nhơn 8.6 against 16.1; the pair sums agree).
- Xuân Sơn Nam (km 1,162) is in VNR's table and not in OSM.
- The greyed Kép - Cái Lân line's station order looks odd (Yên Dưỡng between Đông Triều and
  Uông Bí); not checked further, no train runs.
- The North-South line's strip reads Đà Nẵng - ... - Sài Gòn, then Kim Liên - ... - Hà Nội:
  Đà Nẵng is a stub station where trains reverse, so the line is a Y at Thanh Khê.
- No line colours: VNR publishes none; the metros carry OSM's.
- China's side of Đồng Đăng - Pingxiang (above).
