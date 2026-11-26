# T-4367 review (review471), commit 6a0cab532, demand.worker.ts

Verdict: pass on all points, no fix needed. Advisor not called. Nothing run (one tsc --noEmit, no errors in demand.worker.ts).

- Equivalence: with c.part set, `market` is always non-null (loop `continue`s when marketOn has no entry, ~l.2970), so
  `c.part && market && ed && (reach>0 || ed.switched)` equals the old `!!ed && (...)` wherever a part existed. The
  old `e.window && !ends ? part.beside()` is `ends ? null : e.window` + `w.part.beside()`: same. Trip figures
  unchanged for intercity-on players (same set_window args, same less() body; `e.market!` -> `w.market`, same array).
- No part + intercity edit: window null, sub null, win undefined. localSolve runs with win undefined (as the
  `ends` city-edit path already did since T-4237), then whole solve. Rest of block is guarded: l.3356 befores
  (`c.part ? ... : undefined`), l.3357 `c.part && e.market`, l.3409 `if (part)`, finally `if (whole)` (whole null iff
  c.part null at loop start), `if (sub)`. l.3453 guarded. l.3002 `c.part && sharedParts.has`. No other reader of e.reach/e.ed/e.window.
- Part changing between editOf fill and use: forget() (l.1201) nulls c.part only from dropCity/dropMarkets/recompose,
  called from close/pack-open handlers that defer while `busy` (l.1247-1248, 1394). Same exposure as before; new code
  holds w.part from fill time, old read c.part live, equal while sync-guarded by busy.
- Depends on master only? No: the file diff 058b348af..6a0cab532 is this commit alone; it applies on 0.4.7.
- Minor: comment at l.2954 ("`ed`: the stations changed since what stays frozen last solved") is accurate. Nothing stale found.
