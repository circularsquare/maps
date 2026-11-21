# T-4014 check (newerchk, 2026-11-21)

Tree `.tw-trees/newerlevel` (master 0066d6ac + T-4014, level 38), Vite 6192, headless Chrome with fresh profiles.
Scripts (scratchpad newerchk): a.mjs/a2.mjs fixtures and reload, b3.mjs refused slots and autosave, c.mjs account and share.

1. No false refusals: PASS. `checks/t1826-saves.mjs` passes whole (incl. no-level save opens, level/block/field refused,
   save check "other"). All 11 saves open via Open a file with no newer line and no open failure recorded: 4 in
   checks/fixtures (levels 15, 14, none, 38), T-007-real-nyc, T-125-northeast, 3 in notes/T-3444-design/saves
   (34, 5, 29), t1156 and t1507 all-china. Crafted level none/30/37 copies of a level-38 save open too.
   Weak spot: my station-count read after import was stale for the big saves; the slot summaries the imports
   wrote (1758, 607, 167, ...) show they loaded fully.
2. Autosave and backups: PASS. Slots with level+1, level 1000, format 2, extra block, TWT9 magic, unknown field, each
   with `good` and `t0` backups of an older game under it: opened via ?slot=, refused with the newer line, every
   record (slot, backups, any key) byte for byte, no damaged flag, no `unreadable-` key. Then a fresh game ran and
   autosaved (own slot moved): newer slots and backups unchanged. Newer level in the original `autosave` slot with
   an older good copy under it: refused, unchanged, and unchanged after another reload.
3. Account copy (mocked list and PUT): PASS. Copies listed with level 39 (top-level), facts.level 39 only, 1000: no PUT.
   Upload before any list, server 409 newer: refused, not retried (one PUT). Levels 38, 37 and none: uploaded, meta
   and facts carry level 38. Card line text is the newer line.
4. Community share: PASS. copyFromShare: level 37, 38 and no level open; level 39, 1000, format 2, TWT9 refused
   "version"; level 38 with an unknown field opens (field dropped). forShare: own level goes, newer returns null.
5. Level 38 made here: PASS. Fresh game saved at level 38 (SAVE_LEVEL 38), reopened after reload from autosave.

Notes, not fails:
- After a refused slot is opened, a plain `?paused=1` launch goes to the start menu (the refused slot is the
  last-opened one) with "could not be opened" and the newer line on the card; other games open from the menu.
- A "flag bits" newer save (save_unread through the track model) was not crafted; the block case covers the same
  worker check. Server side and the real account were not tested.
- Only my mock's 409 showed as a console error in c.mjs.
