# 049 — lb: Lebanon at one flat national mix?

Summary: Only open route left: Lebanese at one national mix (sect geography is the whole story and flat is plainly wrong in the south), plus Syrians and Palestinians by governorate. Build it anyway, or leave Lebanon undrawn? Supervisor leans leave it.

*Filed 2026-10-03 by session `fafd1067` (supervisor). Anita's call; nothing is waiting on it.*

## What I did

Held Lebanon back. The 2026-10-03 negatives scout (`sources.md` §scout-2026-10-03-negatives) set `lb`
free at one national mix for Lebanese plus Syrians and Palestinians by governorate (OCHA), and I did
not give it to a builder.

## What it costs to reverse

Nothing to undo; saying "build it" puts `lb` in the next free slot, about half an hour of a builder.

## Why it is yours rather than mine

Spec §14 and your own 2026-09-15 ruling on ask 035 ("build it and see"), which assumed the WVS file
would give governorate shares. It did not: `cb8b206e-lb` found WVS 7 and the Arab Barometer both
sample by district and sect quota (`sources/lb_wvs.py`), so no survey places sects. A flat mix would
draw the Shia south, the Sunni north and the Christian Mount Lebanon identically, in the one country
where sect geography is the point. Every other flat-mix country so far (Cuba, Equatorial Guinea,
Sudan) is close to uniform, so the flat mix there is a small error; here it would be most of the map.

## The detail

- Route as scouted: Lebanese at a national sect mix (Pew 2020 or a survey's national shares), plus
  Syrian and Palestinian refugees by governorate from OCHA. `sources.md` §scout-2026-10-03-negatives.
- Closed record: `queue.md` Lebanon rows, `sources/lb_wvs.py`, ask 035 and 038 in `ask/answered/`.
- Options: (a) leave Lebanon undrawn (my lean); (b) draw the refugees only, Lebanese left as a gap;
  (c) build the flat mix with a note saying the sect geography is not shown.
