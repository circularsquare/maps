# India (`in`): census 2011 C-16, placed on a sub-district settlement layer

Table and build: `sources/in_c16.py`, `sources/in_fetch.py`, `taxonomy/in2011.py`,
`countries/in.py`. This file started on 2026-10-19 with the border fix below; the older
decisions are in `countries/in.py`'s `how`, `parts` and `note_public`, and in followups.md.

## Placement layer cut at Pakistan's border (2026-10-19, session eca0185b-bug)

**Was wrong.** religiondots' `in_places.gpkg` (547,718 settlement polygons in 5,988 sub-districts)
reaches over the Line of Control near Poonch and Rajouri, and over the Punjab and Rajasthan border
in places, so India's dots fell on ground Pakistan administers, while pk drew Azad Kashmir and
Gilgit-Baltistan there too. Before the fix 291 India polygons touched Pakistan's outline (COD-AB
`pak_admin0`, the source of pk's districts); 66 of them reach into Azad Kashmir or Gilgit-Baltistan.
A hex-centroid count: 310 AJK/GB hexes (91,018 Kontur people) sat inside an India polygon.

**Changed.** `sources/in_loc.py` cuts every India place polygon with that outline and scales its
`pop` weight by the share left, writing `data/geo/in/in_places.gpkg`; `countries/in.py` places on
it. 71,013 of 1.2 billion weight units removed (Jammu and Kashmir 20,257, Punjab 48,575, Rajasthan
2,181); 7 polygons gone whole, 547,711 kept, no unit left without one. Census counts per
sub-district are unchanged (1,210,854,977 people); only the placement inside a unit moves.
Rescattered: 1,210,730 dots. religiondots is untouched.

**The reference line.** COD's outline is Pakistan's administered extent and the same layer pk's
hexes are keyed to, so the two countries cannot overlap by construction. Natural Earth's India
polygon differs from COD along the LoC by about 1,300 km2, so neither line is exact; COD is used
because it is the one pk is drawn on.

**Checks after the re-scatter.** India dots (313,980 in the box 68-81E, 23.5-37.5N) inside COD's
Pakistan outline: 1, within the 0.002 degree simplification used to test. Reverse, Pakistan's dots:
246,431 of 246,488 are inside COD's outline (the 57 outside are hexes straddling the edge). Against
Natural Earth's India polygon, which sits about 1,300 km2 further into Azad Kashmir and Gilgit-Baltistan
than COD does, 561 Pakistan dots fall inside India, 45 of them more than 3 km from NE's line and none
more than 11 km: the two reference lines disagree, Pakistan's dots are not on India's own layer.
`check_country in` ok.
