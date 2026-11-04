# Anita's second set of 2026-10-19 notes (the bfe93f9 build), filed as T-703 to T-708

Her words in quotes. Liked: Shanghai's walking now, the sounds. Decided in passing (DECISIONS 2026-10-19):
T-658 two tracks from zoom 17 and a platform each side; points keep their size; no road-class or
time-of-day drive speeds (T-701 closed: "we can start with just distance and maybe spatial traffic but
keep the model quite light").

## T-703, defaults and the demand row
- "lets do nunito ... also lets have default by 90% ui size." Nunito becomes the default font (T-698's
  choice stays in Settings); the UI size defaults to 90%.
- "lets remove the 'avg day / weekday / weekend' selector from always being visible on screen, move it
  under settings demand bubbles opacity. instead lets move in track display 4 settings to always display.
  should go right below none/homes/destinations. this is something we wanna switch often."
  The day selector moves into Settings right under the demand bubbles' opacity; the track display's
  four choices (Settings today: Line, Line + traffic thickness and the rest) go on screen right below
  None / Homes / Destinations, in the same style.

## T-704, the trip explorer, again
- "for walking lets only show effective time since its simple": one row.
- Remove "from the selected destinations".
- "57% of the train riders on this trip take this route, and 8% more the same trains from other
  platforms or with other walks. The rest use other stations or trains nearby." becomes "65% of the
  train users on this trip take this route, the rest use different routes." (the two shares added).
- "im not sure what '- other routes and stations N' means": it is the bonus of having several routes
  (the model's figure is a log-sum over all of them, lower than the best route alone). Name it so it
  reads: "− choice of routes 11.8", and the effective time hover adds one plain line under the equation
  only if the term cannot be read alone. Anita may still ask to drop it.

## T-705, line bullets off centre
"sometimes the line icons have their characters not really aligned with the center of the bubble. would
be good to come up with a universal fix for this (prerender the bullet maybe?) cuz theres a lot of
settings the bullet can appear in". One way to draw a bullet everywhere (panels, tags, map, line
diagram), its character centred by the glyph's measured box, not the font's line box, so every font
(T-698) and symbol length centres the same.

## T-706, notes on a scale
"lets restrain to only pick sounds on chromatic scale. first few lines ever built in each city should
also pick a nice determined set of sounds like pentatonic". A line's note is one of the chromatic
scale's 12 notes in the octave used; the first five lines built in a city (in build order, per city)
take a fixed pentatonic set in a fixed order, later lines a note from the line id.

## T-707, the track tool
- "if we're placing a track and its ramping from 0 to 2 we should say in the tooltip like both 0 and 2
  rather than just whatever we're ramping to"
- "if we're placing a track, clicking on the endpoint that we're starting from should terminate placing
  the track / cancel placing it. as if we were double clicking, but the 2 clicks are separated in time"

## T-708, quad track
- "'double up' track option, for adding an express track to track that has already been placed": turn
  existing double track into quad (the express pair added), blueprint or built, paying the difference.
- "sometimes when placing quad track it says 'the curve does not fit between its neighbors' and idk
  what that means": find when it fires; fix the cause if it is a real limitation that can be lifted,
  else say it in words a player can act on.
- "placement ghost of quad track station looks like a double track station": the ghost shows the
  platforms a quad station will have.
