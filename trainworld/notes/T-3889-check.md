# T-3889 check (tramrailchk, 2026-11-20)
Tree Vite on 6171 (6166 was taken by another tree), master 6167. NYC sandbox games via play.mjs.
1. PASS: master-built save (tram line, rail line crossing the tram flat) loaded in the tree: both lines identical (kind, cars 2/5, dwell, round trip 558/606 s, km 3.00, path). T-007-real-nyc.save: 109 lines, hash of kind/cars/dwell/round trip/km/path/trains identical master vs tree, 0 tram lines in it.
2. PASS: rail A-B with a joined tram stretch shorter than the rail detour: train line took the detour (21.65 km, no tram edges); a train line needing only tram track refused (TramTrack); tram line rail-tram-HSR ran 7.0 km, round trip 978 s (about 63 km/h average incl. stops), 2 cars, dwell 15. Not done: HSR train behind a tram (PlatformNotLevel on my HSR stretch; no second HSR station built).
3. PASS: adding a tram stop to a train line: tram=true, cars 5->2, dwell 30->15, turnaround 180->120; undo restored 5/30/180/21.6 km exactly, redo again; removing the last tram stop refused (TramTrack), line unchanged. Save and reload of the joined network: stats identical (line id renumbered 2->1 only because a refused add used an id).
4. PARTIAL: station at tram/rail meeting nodes (1, 3, 12, 13) refused (StationFamily, StationJunction). Kind-change/rebuild op not exercised: raw `rebuild` returned ok:false with no issue on a blueprint (op shape unclear); left to rebuild_tests.
5. agent_cargo.py test: 697 passed, 0 failed.
Also seen: page errors "<circle> attribute r: NaN" during drawing in the tree; not compared on master.
