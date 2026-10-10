# Latin America survey: countries with no passenger rail of their own (2026-10-08)

**Built 2026-10-08** (latam agent): cr, pa, cu, pe, bo, ec, do, uy, ve, co, pr, all through
`latam_register.py` (ar_register's recipe in one module; do and pr with no register). Each
`<cc>_sources.md` has a "Build (2026-10-08)" section; `handoff_notes/latam_build.md` has the
shared-file changes. Guatemala and Paraguay below stay unbuilt.

The survey of pe, bo, co, ec, ve, pa, cr, uy, py, cu, do, gt, pr. Countries with a service have
their own `<cc>_sources.md`; these two have nothing to build.

- **Guatemala (gt)**: no passenger train. FEGUA runs no scheduled service; the only rides are
  one-off events at the Railway Museum (Día del Ferroviario, June 2026). MetroRiel (Guatemala
  City light rail, 22 stations) is still at the planning and financing stage (FEGUA and the UK
  agreed an "eight-week roadmap" in July 2026; prensalibre.com/tema/fegua). Nothing opens before
  2028 at the soonest. Geofabrik `guatemala-latest.osm.pbf` (125 MB) not needed.
- **Paraguay (py)**: no domestic passenger train. The steam excursion train Asunción - Areguá
  (Ferrocarril Presidente Carlos Antonio López) has not run regularly for over a decade, and the
  Asunción - Ypacaraí commuter train is still a project. The one train is the binational
  **Posadas - Encarnación** (Trenes Argentinos / SOFSE, 23 a day each way Monday - Friday), which
  `ar_register.py` already builds whole to Encarnación (ar_sources.md, "Borders"). Recommendation:
  no Paraguayan region; if one is ever wanted, the line needs the `borders.EXTRA` point on the San
  Roque González bridge that ar_sources.md asks for, and Paraguay then holds ~1.5 km and one
  station. Geofabrik `paraguay-latest.osm.pbf` (147 MB) not needed.
