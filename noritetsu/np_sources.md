# Nepal register sources

## Survey (2026-10-08)

Research only. Samples in `data/raw/np/survey/` (en.wikipedia article text, 33 KB).

### The short answer

- **Nepal has one passenger railway**: Nepal Railway Company's Jaynagar (India) - Janakpurdham
  - Kurtha - Bhangaha line, 5 ft 6 in broad gauge, rebuilt by Ircon with an Indian grant.
  Jaynagar - Kurtha (34.9 km) opened 2 April 2022 (service from 3 April); Kurtha - Bhangaha
  (17 km; Bhangaha is the old Bijalpura) was handed over in July 2023 with one train a day.
  en.wikipedia's infobox: "currently 52 km is operational" of 68.7 km planned to Bardibas.
  The Bhangaha - Bardibas extension is not built.
- Nothing else carries passengers: Raxaul - Birgunj ICD is freight only (an Indian Railways
  siding into Nepal), the Kathmandu - Kerung (China) and Raxaul - Kathmandu lines are plans,
  and there is no urban rail.
- **Recommended recipe: a one-line hand list traced by rinf.py** (the nafrica / za pattern;
  could live in `nafrica_register.py`'s style as its own tiny `np_register.py`, or as one more
  country in an existing hand-list reader). Stations in order: Jaynagar, Inarwa, Khajuri
  (halt), Baidehi, Parbaha, Janakpurdham, Kurtha, Khutta Pipradhi, Loharpatti,
  Singyahi, Bhangaha. The km are traced over OSM (`no_chain`); check against 34.9 km
  Jaynagar - Kurtha and 52 km Jaynagar - Bhangaha.
- **Expected size**: 1 register line, about 52 km (49 in Nepal).
- **Extract**: Geofabrik `asia/nepal-latest.osm.pbf`, 395 MB (big for one line, but it is
  the only cut that holds the Nepali part; India's extract ends at the border).

### Sources

| source | what it gives | licence |
|---|---|---|
| en.wikipedia "Jaynagar–Bardibas railway line" | stations in order, opening dates, 68.7 km planned / 52 km operational | CC BY-SA |
| Wikidata (via qlever.dev) | 5 railway-line items with P17 Nepal ("Jaynagar–Bardibas railway line" 68.7 km, "Janakpur–Jaynagar Railway", "Nepal Government Railway" 47 km (the closed Raxaul - Amlekhganj), "Raxaul–Kathmandu railway line"); only 4 station items, no station adjacency | CC0 |
| OSM (Overpass, 2026-10-08, Nepal's outline) | 232 `railway=rail` ways, 210 of them named (91%), all 182 `usage=main` ways named; 9 station/halt nodes; `route=railway` relation 11488569 "Nepal Janakpur to Jaynagar Railway" (ref NJJR, operator Nepal Railways); no `route=train` relation | ODbL |
| The Rising Nepal (risingnepaldaily.com/news/29535, July 2023), The Annapurna Express, b360nepal.com | Kurtha - Bijalpura (Bhangaha) opening, one train a day; service suspensions and resumptions on Jaynagar - Kurtha | press |

No GTFS anywhere (Mobility Database, Transitous). No operator timetable online found.

### Open questions (for the country agent to decide)

- **Is Kurtha - Bhangaha running in late 2026?** The press has a 2023 handover with one daily
  train and later reports of suspensions on the line; check before calling it running. If in
  doubt, build Jaynagar - Kurtha running and Kurtha - Bhangaha greyed.
- **The border**: India's register (`in_register`, `NOT_REGISTER`) leaves the Nepal chain out
  and ends at Jaynagar. Nepal's line runs from Jaynagar station across the border (about
  3 km). Either the Nepal line starts at the border point (a `borders.EXTRA` point near
  Inarwa) and India's Jaynagar stays India's, or Nepal's line starts at Jaynagar station
  with its first 3 km in India's outline, as other cross-border pieces are drawn.
- Janakpur's OSM station may be mapped as an area: extract with `--station-areas`.

### Downloads Anita must do by hand

None.

## Build (2026-10-08)

    python tools/slot.py 2 -- python extract.py --region np --pbf data/raw/nepal-latest.osm.pbf --station-areas
    python asia_register.py --clip np             # drops India's track
    python asia_register.py --join np             # OSM's line is in pieces that do not share end nodes
    python asia_register.py --convert np
    python build_model.py --region np --register asia_register:data/raw/rinf/np
    python build_tiles.py --region np; python check_model.py --region np

Reader: `asia_register.py`, list `NP` in `asia_lines.py`; settings `rinf_countries/np.py`,
rules `rules/np.py`, colours `colours/np.csv` (picked).

One register line, the Indian border - Inarwa - Janakpur - Kurtha - Bhangaha (OSM's
"Bijalpura"), 48.8 km, running; against 52 km operational less ~3 km in India: 1.00.
Stops: every OSM station on it (Inarwa, Khajuri, Mahinathpur, Bideha, Perbaha, Janakpur,
Kurtha, Loharpatti, Bijalpura).

Decisions:
- **Kurtha - Bhangaha running**: opened July 2023 (himalpress, 16 July 2023). The whole line
  was suspended on 31 July 2026 under a curfew in Dhanusha (ekantipur, onlinekhabar) and ran
  again once it was lifted in August (prabhatkhabar, "Jaynagar-Janakpur rail service resumed").
- **The border**: the line starts at a new point `xJaynagarInarwa` (86.137098, 26.606175),
  where OSM's track (way 1433667171) crosses OSM's boundary (way 534202189); Nepal's extract
  has no track past it. India's register ends at Jaynagar station. Proposed: the point in
  borders.EXTRA, and India's Jaynagar - border (~3 km) as a piece under this line's id, as
  cn_register does for Đồng Đăng, so Jaynagar -> Janakpur is one ride
  (handoff_notes/asia_build.md). Until then the point shows its id as its name.
- `--join` (mideast_register.join) joins 5 track ends 10-35 m apart; without it Kurtha -
  Loharpatti had no trace.
