# Open questions for Anita

Questions that outlive a session. The orchestrator keeps this list; answered ones move to
DECISIONS.md and leave here.

- 2026-11-07, T-007 stage 2: go ahead with splitting coarse cells into their land pieces?
  Fixes fjord and river country being 30-40% too fast zoomed out (Sogndal 1.32 -> 1.01, Coari
  1.39 -> 1.01); ~5 MB, search speed unchanged; a split L10 cell paints one colour until a finer
  window covers it. notes/T-007.md "In short".
- 2026-11-07, T-091: are the straighter 100 m corridor lines (Kathmandu 27% of km in straight
  runs, was 20%) acceptable, or loosen NEAR/SPREAD?
- 2026-11-07, T-011: do mountain walks now look too fast?
- 2026-11-07, T-035: open sea with 8 moves (today; eight-sided bands, corridors in straight 45
  degree runs) or 16 (round bands, curving corridors, canoe clicks 10-25% slower, ~1 s in Chrome
  on a busy machine)? Shots: artifacts/isochrone/t035_blacksea_live.png and _sea16.png.
- 2026-11-07, T-100: the canoe's river heights and lengths in tropical forest now come from
  Hydrography90m, CC BY-NC 4.0 (non-commercial). Fine for publishing? If not, register for MERIT
  Hydro (ODbL option; Google Form, password by email:
  https://global-hydrodynamics.github.io/MERIT_Hydro/) and h90.py reads its `elv` tiles instead.
