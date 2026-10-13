# Tests to batch next session (written 2026-10-13)

Anita asked to leave play-testing of the last round for the next session. Each line is one thing
to look at on the served build (localhost:8800/trainworld/). Agents: run what you can headlessly
first, then hand Anita the rest in one batch, and delete lines as they pass.

- Station boxes 25% narrower (T-239): plain and traffic modes at zoom 15-17; lines may be wider
  than boxes in traffic mode.
- Traffic widths (T-227): linear, relative to the network's busiest stretch; check zoom 12-15.
- Stop editing (T-233, T-239, T-237): no check on stops; reachable markers at 80% of a stop, others 65%;
  other lines and trains faded; stations past a line's end now count as reachable.
- Platform fill and dot tint (T-219, T-228): Myrtle Av on the Z, Jamaica, morning peak.
- "N stations full" alert (T-231): click through it.
- Line panel "Trains today" (T-230): "100%", trains leaving full, left behind, longest wait;
  does it make the G2 case clear?
- Station "Waiting now" (T-239, T-238): counts in red, dropping the moment a train leaves.
- Demand figures hold during a re-solve (T-232): "Updating demand 2/4…" then one change.
- Train stepper (T-229): quick clicks all count, no stutter.
- Rikers Island with Homes on (T-234): bubbles gone.
- New line colours (T-223): pink sits close to the medical purpose pink.

Batch 17 (2026-10-13, built to 8800):
- Saves cards, now in the saved games panel (floppy disk button, T-241, T-251): network picture big enough to tell games apart?
- Panel and text size in Settings (T-247): pick a size; 85-90% may suit.
- Loading bar (T-243): moves steadily instead of stalling near 60%?
- Selected train halo (T-242): right amount at zoom 15+ over a station box?
- Track drawn with many clicks (T-244): delete one piece; cuts sit midway between clicks. Track
  inspector's Construct now builds one piece (T-248 asks if that is tedious).
- Platforms as periwinkle lines with track over the box (T-236, #8085f0, track capped at 30% of
  the box width).
- Complex view (T-246, T-250): one row per line on top, platform crowds and trains in the complex
  map; busy at Grand Central's Metro-North terminal.
- Linked complex zoomed out (T-225): two dots and a connector; each dot picks its own side.

Batch 19 (2026-10-13, built to 8800):
- Station boxes following S-curves (T-254): 110 St-Malcolm X Plaza, Wall St, Grand Central's
  Lexington Av platforms at zoom 17. Placement ghost not looked at live.
- Quad track and Express copy (T-255): draw Quad, place a station (gets a twin), make a line,
  click Express copy, remove stops. Pairs 8 m apart merge into one line at zoom 15 and below; the
  express keeps the local's colour.
- Capacity tags: build tools and select clicks reach the station under a tag (T-224, T-256).

Batch 20 (2026-10-13, built to 8800):
- "Demand shown for" in Settings: Today (default), Weekdays, Weekends (T-260, T-266): weekend lines thinner, midday busiest; clock shows "Sat 13"; Saturday 8:00 has no full stations, 13:00 has some.
- Stations partway round curves (T-258): place a few, including near a curve end (snaps to the end).
