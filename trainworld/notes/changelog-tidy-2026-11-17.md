# Changelog tidy, 2026-11-17

Merged by reword (`edited`) through changelog.py's own update_item; later items hidden. 6 chains.

- kept c001: added hiding lines on the map from the lines list, and hiding track levels, buses and ferries in settings; a hidden level is drawn faint and can't be clicked, hovered or snapped to; hid c107
- kept c076: added a real world network snap for tram track that follows real tram and light rail, and redrew the snap buttons as one set of icons, with snap to streets a straight street with a dashed center line; hid c097
- kept c032: merged Arnavutkoy into the Istanbul zone, and moved Karamursel to Kocaeli and Altinova to Yalova; hid c079 (c032 had said "named the Arnavutkoy zone as an Istanbul district", now wrong)
- kept c117: added real world stations to the real world network map, as soft purple dots like the real track, with the name on hover: settings, show stations, off by default; c121 was already hidden
- kept c015: rail line panels have a chart at the bottom with blue fare bars and red running cost bars for each of the last 30 days; c119 was already hidden (c015 said 7 days)
- kept c094: added a schedule window for each rail line, with its departures, a timetable and a timetable chart, and weekend train scheduling so a line can run fewer trains on weekends / holidays; hid c095 (same task T-3521; c123 stays on its own, since copy and paste schedules shipped earlier)
- kept c082: added a setting under Game to keep buses to one-way streets; c122 was already hidden (its "train and platform fullness always shown" part is now in no visible item)

Already fine, nothing to do: c031 "added more long hairstyles" covers c116 (hidden); c075 and c077 are duplicates, both hidden; c106 hidden, c111 kept.

Left alone, unsure:
- c094 (weekend train scheduling) and c095 (schedule window per rail line), both T-3521: maybe one feature
- c123 (timetables window default button, one timetable button in place of copy/paste schedule) and c095: copy/paste schedule is in no item here, so c123 may be a change to something players already have
- c004 (one platform stations at end of single track) and c021 (stations within 5 m of the track end go on the end): related, but each stands as its own change
- c092 (track along a shore not counted as water) and c113 (narrow rivers count as water): same water code, opposite fixes, both noticeable
- c066, c068, c070, c071, c072 (memory savings): separate changes with their own numbers, sit under group c086
