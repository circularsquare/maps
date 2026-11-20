# T-3844 check (regionchk), PASS

Method: play.mjs on the tree (Vite 6185) and on master (6186, main tree), identical edit sequences through tw.client.edit,
cash, ledger and per-region summed ledger read after each edit. Scratch scripts: scratchpad/regionchk/seq*.js (not kept in repo).
- Fixtures: new nyc game (2 lines + extension), real save T-007-real-nyc.save (regions nyc and nhv), nhv line built by edits.
- Totals (cash, ledger build/trains, edit charge) equal to master after every edit in 4 sequences (about 70 edits); sum over regions = ledger total each time.
- Stop click (insertStop on a headway-240 line, nhv): +12.5 and +37.5 trains, tree nhv, master "" (unassigned).
- Waiting room on built station: 5 to the station's region (nhv 1358, nyc 5, built nhv pt0), master "". stationTracks 1361: +10 nhv.
- setTrainCount/setHeadway: split by run shares (line 77: nyc 16.25 / nhv 16.25; line 78 setHeadway 162.5 split 92.5/70.5 nyc/nhv); master all "".
- New track (route+constructAll) in nhv and nyc: split identical to master.
- Undo/redo of stop click: charge 0, cash and region split unchanged on both (neither refunds).
- Export, reload: regions nhv [1378.01, 75], cash 4546.99 unchanged.
Not tested: a new track crossing two regions (no coordinates found), edgeLevel/moveNode (my ops refused), undo of a paid line removal, moved-station/rebuild ops.
