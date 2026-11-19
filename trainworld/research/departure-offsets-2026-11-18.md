# Departure offsets in other games (T-3742, agent offsetdesign, 2026-11-18)

The question: when a game places a line's departures for you, can the player move them a little, and is that move
measured from the game's own placement (which keeps running) or does the player then own absolute times?

| Game | What the player sets | Relative to an automatic default? |
|---|---|---|
| NIMBY Rails (1.14+) | Per shift (one train): an offset of the schedule's orders; groups add a time to the orders of chosen groups, typed by hand or in "auto" mode, where the schedule spreads the shifts by an interval and the number of shifts in auto | Either/or per shift and group: auto computes the offset, a manual value replaces it. Groups are off by default and enabled step by step: "on a group by group basis, then with manual shifts, and then by auto interval if desired" |
| OpenTTD | A timetable per vehicle's orders and a start date; Ctrl-click on Start date spreads the vehicles sharing the orders evenly over the timetable's length | The spread is a one-off fill the player can then change; timetables are "relative to the time that vehicle's timetable was last reset". Players add slack to wait times |
| Transport Fever 2 (Timetables mod; no base-game timetable) | Per stop: arrival/departure slots in the hour, or Unbunch (a span after the last departure), or AutoUnbunch (spaced by the line's frequency with a "give", 1 min recommended) | Slots are the player's own times; AutoUnbunch is the automatic spacing, with no offset on top |
| A-Train (Hajimaru Kanko Keikaku) | Per train: first departure time, repeat interval, number of departures, at any station; copying a plan and offsetting it adds trains | Absolute, all by hand; the player works out the cycle time |
| Train Fever / Transport Fever 1 | No timetables; vehicles per line only (general knowledge) | No offsets |
| Simutrans | Per line "spacing" (departures a month) with a shift | The shift is the player's offset on the spacing's slots |
| Real operators | A clock-face family of headways plus an offset per line, so branches alternate on the trunk (research/train-flow.md) | Planners set it; nothing automatic |

What it suggests for the game:

- Nobody adds a player offset on top of an automatic placement that keeps re-placing the line. NIMBY's groups are the
  closest: auto or manual, per group, and manual means the player owns it. So a nudged line should stop being placed
  by the game, and the offset is measured from where the game had put it.
- The usual step is minutes, with seconds available (NIMBY orders, TF2 slots to the second, OpenTTD in days or ticks).
  Her 11-12 call for Exact mode (1 min a click, 15 s with Shift, 10 min with Ctrl) fits.
- Every game that lets the player move departures shows the result on a timetable or a chart, and the cost is
  tedium; a single line-wide value is the cheap end of that.

Sources:
- [NIMBY Rails devblog 2024-08 (shifts, offsets, groups, auto group mode)](https://carloscarrasco.com/page/13/)
- [NIMBY Rails wiki, Timetable](https://wiki.nimbyrails.com/Timetable), [How to manual timetable](https://wiki.nimbyrails.com/How_to_manual_timetable)
- [OpenTTD wiki, Timetable](https://wiki.openttd.org/en/Manual/Timetable)
- [TPF2 Timetables mod](https://github.com/Gregory365/TPF2-Timetables)
- [Game*Spark, A-Train Hajimaru Kanko Keikaku timetable guide](https://www.gamespark.jp/article/2021/04/04/107527.html)
- [Simutrans schedule help](https://translator.simutrans.com/data/htmlex/en/schedule.html)
- Earlier surveys: notes/T-3346.md "How others do it", research/train-flow.md
