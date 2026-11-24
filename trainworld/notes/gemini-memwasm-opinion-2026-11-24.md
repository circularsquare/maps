**(1) Verdict**

No — trading a 3–8 minute uninterruptible worker blackout (plus 0.11% trip drift) for ~300 MB (~5% of an all-China tab) is a bad trade, and "hold while editing" only delays the trigger rather than fixing the blackout once a restart starts; keep the 384 MB threshold or gate a lower threshold on `document.hidden` / real memory pressure instead.

*(Note on the other alternatives: restarting one worker at a time does not help on all-China, which already runs 1 solving worker under `SOLVING_SHARE`, and even with 2+ workers `allOpen` / `ALL_PERIODS` blocks solves and queries until every restarting worker finishes reopening. Gating on `document.hidden` is better than an idle timer, though Chrome throttles background timers/fetches so the reopen/re-solve may still be running when the player returns.)*

---

**(2) Best design for "hold while editing"**

- **Where**: [demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts) — track a `lastActivity = 0` timestamp on `DemandClient`, check it in a helper (e.g. `editQuietIn()`) called from `memCheck()` ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3350-L3359)) **only** for `overNeed()` restarts (and `grow()`), leaving `capPool()` ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3343-L3349)) on the existing `quietIn()` so over-budget pool shrinks are not delayed.
- **What counts as editing / active interaction**:
  - **Hard holds** (return `MEM_RETRY_MS`, never start a restart while true):
    - `this.pausedSnap !== null` / `demandHeld.peek()` ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L4216-L4231), [demand.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/game/demand.ts#L417)) — edits queued while paused.
    - `editsSettling.peek()` ([demand.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/game/demand.ts#L414), already imported in `demandClient.ts`) — edits still settling for the edit note.
    - `stopsQueued.peek() > 0` ([world.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/game/world.ts#L85)) — stop clicks in flight.
    - `stopEditing.peek()` ([ui.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/game/ui.ts#L331)) — active line stop-editing session.
  - **Activity hooks** (set `this.lastActivity = performance.now()` and clear/defer `this.memTimer`):
    - Subscribe in `DemandClient`'s constructor to `editStamp.value` ([editCap.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/game/editCap.ts#L19)) — fires on click before `clock.worker.ts` debounces and catches blueprint/fare/undo edits that never change `lastKey`.
    - In the existing `selection.value` effect ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L1062)) and `tool.value` effect ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L1079)).
    - In `ask()` ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L4401)) and `corridorRows()` ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L5306)) — `pendingQueries.size` only blocks while a query is in flight (~100 ms), not after.
    - Note: do **not** make `isBuildTool(tool.peek())` an indefinite hard hold, or a player who leaves the track tool selected and walks away never frees memory; treat entering/clicking with it via `tool` + `editStamp` instead.
- **How long to wait after the last edit/activity**:
  - **90–120 s** (`EDIT_QUIET_MS`, separate from `MEM_QUIET_MS = 20_000` for post-solve stillness, with a `?memQuiet=` override for checks). A 20 s gap is normal reading/planning time between edits; 90–120 s avoids firing mid-session while still firing when the player steps away or watches trains run.

---

**(3) Risks and failure cases (most serious first)**

1. **Irreversible blackout once started**: `workerDied()` calls `this.workers[i].terminate()` immediately ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3066)). There is no way to cancel an in-flight restart; if the player pauses for 90 s to plan a corridor and edits 1 s after `workerDied()` runs, edits and station questions still stall through the full reopen + re-solve (171–475 s). (Spawning a shadow worker instead would double WASM memory during reopen.)
2. **Starvation during long editing sessions**: Continuous editing, keeping `stopEditing` open, or editing while paused with `recalcWhilePaused` off holds restarts indefinitely, letting WASM high-water memory accumulate until OOM on a tight device.
3. **Accidentally gating `capPool()`**: If the edit hold is added directly to `quietIn()`, it also blocks `capPool()`'s post-final worker-count shrink ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3343-L3349)), keeping an over-budget pool alive while the player edits.
4. **Fatal failure on transient network error during an optional restart**: `reopen()` calls `this.fail(...)` on download error ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3165)), permanently killing the demand pool for the tab during an optional 300 MB cleanup.
5. **Unexplained figure drift + query stall**: Even when triggered cleanly during an idle gap, `refreshing = true` blocks `viaRound` queries until the new final ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L4480)) and then shifts displayed trips by ~0.11% with no player edit.

---

**(4) Issues in [memwasm.diff](file:///C:/Users/anita/projects/maps/artifacts/trainworld/gemini/memwasm.diff)**

1. **Check race can hang for 30 minutes** ([t1205-memory.mjs](file:///C:/Users/anita/projects/maps/trainworld/checks/t1205-memory.mjs), diff lines 76–88): Removing the forcing edit means the restart can already be in progress—or finished—before `const closed = await state('zones closed')` runs. If the restart's re-solve has already been dispatched, `at = window.twDemand.log.at(-1)?.version` captures the *restart's own* version, and `await final(at)` then waits for `v.version > at` until the 1800 s timeout. It also makes `closedToRestartSeconds` read `0`.
2. **`out.reopenSeconds` does not measure reopen time** (diff lines 50–51, 90): `c.respawn.size` stays `> 0` until `msg.kind === "done"` deletes the entry ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L4012-L4016)); `reopened()` only sets `r.solving = true` ([demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3174)). So `w.e - w.s` measures reopen **plus** the entire re-solve, not reopen alone.
3. **Weakened memory assertion** (diff line 102): Changing `sum(after) < sum(closed) - 256` to `sum(after) < sum(built) - 256` masks whether memory was freed by the restart vs. anything else (and was likely only needed because `closed` raced with the restart).
4. **Undocumented 0.11% drift vs. code/doc claims** (diff line 103): The check was relaxed from `after.rail === undone.rail` to `<= 2e-3` (0.2%), while [demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3335) still states `"the same to the trip"` and [t1205-memory.mjs](file:///C:/Users/anita/projects/maps/trainworld/checks/t1205-memory.mjs#L2) states `"unchanged to the trip"`. The cause of the 45k-trip drift isn't explained.
5. **Halved hysteresis guard** (diff line 14, [demandClient.ts](file:///C:/Users/anita/projects/maps/trainworld/app/src/workers/demandClient.ts#L3306)): Lowering `MEM_FREE` from 384 to 192 MB also halves the repeat-restart guard (`m.wasm < last.wasm + MEM_FREE && need > last.need - MEM_FREE`), making repeated 3–8 minute restarts twice as easy to trigger.
6. **Changelog wording** (diff line 28): `"the game uses 300 MB less memory after leaving zones on a country-sized save"` — minor, but the saving is ~300 MB *per worker* only after a 3–8 minute re-solve, and on 1 worker is ~300 MB (which is accurate for 1 worker, misleading if 2+ workers run).
