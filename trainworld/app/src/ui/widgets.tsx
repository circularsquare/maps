// Small shared pieces: line chip, stepper, segmented buttons, checkbox, icons (inline SVG).

import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { levelLabel } from "../game/format";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import { IDLE_TIP, isIdleLine } from "../game/lineRunning";
import { PLANE_PATH } from "../map/glyphs";
import { bulletDraw, bulletEpoch, bulletTextSoon, type BulletDraw } from "./bullet";
import { shapeOf } from "../game/lineShapeState";
import { LINE_SHAPES, type LineShape, type PickShape } from "../game/lineShape";
import { LEVELS, type Level, type LineInput } from "../game/types";
import { selection, uiScale, type ListSort } from "../game/ui";
import { knock } from "../game/world";
import type { Signal } from "@preact/signals";
import { t, type MessageKey } from "../i18n";
import { mobileDevice } from "../workers/deviceMemory";

export const Icon = {
  minus: () => <svg viewBox="0 0 16 16"><path d="M4 8h8" /></svg>,
  plus: () => <svg viewBox="0 0 16 16"><path d="M4 8h8M8 4v8" /></svg>,
  pencil: () => <svg viewBox="0 0 16 16"><path d="M10.5 3l2.5 2.5L6 12.5H3.5V10z" /></svg>,
  check: () => <svg viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" /></svg>,
  // the station name's suggestions, round and round
  cycle: () => <svg viewBox="0 0 16 16"><path d="M12.8 6.5A5 5 0 004 5M3.2 9.5A5 5 0 0012 11M4 2v3h3M12 14v-3H9" /></svg>,
  // take the map to a station or line (a sight with its cross hairs)
  locate: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="3.5" /><path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3" /></svg>,
  // keep the map on a train: the sight with a filled middle
  follow: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="3.5" /><circle class="solid" cx="8" cy="8" r="1.4" /><path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3" /></svg>,
  pause: () => <svg viewBox="0 0 16 16"><path class="stroke2" d="M5.5 3.5v9M10.5 3.5v9" /></svg>,
  play1: () => <svg viewBox="0 0 16 16"><path class="solid" d="M5 3.5l7 4.5-7 4.5z" /></svg>,
  play2: () => <svg viewBox="0 0 16 16"><path class="solid" d="M2.5 4l5.5 4-5.5 4zM8 4l5.5 4L8 12z" /></svg>,
  play3: () => <svg viewBox="0 0 16 16"><path class="solid" d="M1.5 5l4 3-4 3zM6 5l4 3-4 3zM10.5 5l4 3-4 3z" /></svg>,
  track: () => <svg viewBox="0 0 16 16"><path d="M2.5 10.5l8-8M5.5 13.5l8-8M3 9l4 4M6 6l4 4M9 3l4 4" /></svg>,
  station: () => <svg viewBox="0 0 16 16"><path d="M1.5 8h3.5M11 8h3.5" /><circle cx="8" cy="8" r="3" /></svg>,
  // an interchange (T-1448): two station boxes crossing, one shape with one outline
  interchange: () => <svg viewBox="0 0 16 16"><path d="M4.5 1.5h3v7h7v3h-7v3h-3v-3h-3v-3h3z" /></svg>,
  // T-1851: the map's track display row and station ridership row icons: the track with a yellow line
  // along its lower right, the interchange under a big blue station circle centred on its crossing (colors in style.css)
  trackDisplay: () => <svg viewBox="0 0 16 16"><path d="M2.5 10.5l8-8M5.5 13.5l8-8M3 9l4 4M6 6l4 4M9 3l4 4" /><path class="yline" d="M8 15l7-7" /></svg>,
  stationDisplay: () => <svg viewBox="0 0 16 16"><path d="M4.5 1.5h3v7h7v3h-7v3h-3v-3h-3v-3h3z" /><circle class="udot" cx="6" cy="10" r="4.6" /></svg>,
  // a selected demand bubble (T-1137): the ramp's middle purple in the map ink's outline (style.css)
  bubble: () => <svg viewBox="0 0 16 16"><circle class="bubble" cx="8" cy="8" r="5.5" /></svg>,
  // a run of segments, each starting where the last ended
  keepPlacing: () => <svg viewBox="0 0 16 16"><path d="M2 12.5l4-6 3 3 5-6.5" /><circle class="solid" cx="6" cy="6.5" r="1.4" /><circle class="solid" cx="9" cy="9.5" r="1.4" /><circle class="solid" cx="14" cy="3" r="1.4" /></svg>,
  // a magnet over a curving line
  snapReal: () => <svg viewBox="0 0 16 16"><path d="M1.5 14.5c4-.5 9-3 13-8" /><path d="M5 2.5v3.5a3 3 0 006 0V2.5M5 4.5h2M9 4.5h2" /></svg>,
  // a road with its centre line, a rail track curving along it (T-3129)
  snapRoads: () => <svg viewBox="0 0 16 16"><path d="M1.5 12.5c4 0 8-2 13-9M4.5 15c4 0 8-2 11-6.5" /><path d="M2.5 9c3.5-.5 7-2 11-7" stroke-dasharray="1.5 1.5" /></svg>,
  // two curving tracks side by side, the new one dashed
  snapBeside: () => <svg viewBox="0 0 16 16"><path d="M1.5 11.5c4.5 0 8-3 10-9.5" /><path d="M4.5 14.5c4.5 0 8-3 10-9.5" stroke-dasharray="2 1.6" /></svg>,
  // two peaks, the track tag's terrain mark (T-1394)
  terrain: () => <svg viewBox="0 0 16 16"><path d="M1.5 13.5L6 6l2.4 3.4L10.6 6.5l3.9 7z" /></svg>,
  // track carried on at its own height: a raised piece, then an arrow along the same level (T-1161)
  autoLevel: () => <svg viewBox="0 0 16 16"><path d="M1.5 13.5h3M4.5 13.5l3-6h7M12 5l2.5 2.5L12 10" /><path d="M7.5 10.5v3M11 10.5v3" /></svg>,
  // track ramping up over another line, seen side on (T-1615)
  autoRamp: () => <svg viewBox="0 0 16 16"><path d="M1.5 12.5h2.5l3-6h2l3 6h2.5" /><circle class="solid" cx="8" cy="12.5" r="1.6" /></svg>,
  // the track tool's two or four tracks, side on to their direction (T-2931)
  tracks1: () => <svg class="tracks" viewBox="0 0 20 20"><path d="M10 3v14" /></svg>,
  tracks2: () => <svg class="tracks" viewBox="0 0 20 20"><path d="M6 3v14M14 3v14" /></svg>,
  tracks4: () => <svg class="tracks" viewBox="0 0 20 20"><path d="M4 3v14M8 3v14M12 3v14M16 3v14" /></svg>,
  undo: () => <svg viewBox="0 0 16 16"><path d="M5.5 4L2.5 7l3 3M3 7h6.5a3.5 3.5 0 010 7H7" /></svg>,
  // T-2679: make a country the reference for fares
  anchor: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="3.2" r="1.6" /><path d="M8 4.8v9.7M5.2 7.2h5.6M2.7 9.6c.4 2.9 2.6 4.9 5.3 4.9s4.9-2 5.3-4.9" /></svg>,
  redo: () => <svg viewBox="0 0 16 16"><path d="M10.5 4l3 3-3 3M13 7H6.5a3.5 3.5 0 000 7H9" /></svg>,
  cross: () => <svg viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" /></svg>,
  // the controls panel
  help: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.2" /><path d="M6.2 6.3a1.9 1.9 0 1 1 2.6 1.8c-.5.2-.8.6-.8 1.1v.6" /><circle class="solid" cx="8" cy="11.6" r=".5" /></svg>,
  // T-2855: Discord's logo (filled, so no outline), for Settings' link to the server
  discord: () => <svg viewBox="0 0 24 24"><path class="solid" style="stroke:none" d="M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z" /></svg>,
  // Ko-fi's cup mark (Simple Icons, CC0 artwork; the mark itself is Ko-fi's), for Settings' donation link, its heart in Ko-fi's red
  kofi: () => <svg viewBox="0 0 24 24"><path class="solid" style="stroke:none" d="M23.881 8.948c-.773-4.085-4.859-4.593-4.859-4.593H.723c-.604 0-.679.798-.679.798s-.082 7.324-.022 11.822c.164 2.424 2.586 2.672 2.586 2.672s8.267-.023 11.966-.049c2.438-.426 2.683-2.566 2.658-3.734 4.352.24 7.422-2.831 6.649-6.916zm-11.062 3.511c-1.246 1.453-4.011 3.976-4.011 3.976s-.121.119-.31.023c-.076-.057-.108-.09-.108-.09-.443-.441-3.368-3.049-4.034-3.954-.709-.965-1.041-2.7-.091-3.71.951-1.01 3.005-1.086 4.363.407 0 0 1.565-1.782 3.468-.963 1.904.82 1.832 3.011.723 4.311zm6.173.478c-.928.116-1.682.028-1.682.028V7.284h1.77s1.971.551 1.971 2.638c0 1.913-.985 2.667-2.059 3.015z" /><path style="fill:#ff5e5b;stroke:none" d="M12.819 12.459c-1.246 1.453-4.011 3.976-4.011 3.976s-.121.119-.31.023c-.076-.057-.108-.09-.108-.09-.443-.441-3.368-3.049-4.034-3.954-.709-.965-1.041-2.7-.091-3.71.951-1.01 3.005-1.086 4.363.407 0 0 1.565-1.782 3.468-.963 1.904.82 1.832 3.011.723 4.311z" /></svg>,
  // trains reverse at this stop
  uturn: () => <svg viewBox="0 0 16 16"><path d="M5 13.5V7a3.25 3.25 0 0 1 6.5 0v4.5M8.8 9.2l2.7 2.8 2.7-2.8" /></svg>,
  // the line panel's related lines (T-1168), as its diagram looks (T-1860, Anita's sketch): two lines side by side, a station pill across both, the right one curving away at the bottom
  allStops: () => <svg viewBox="0 0 16 16"><path d="M8 1.5v13M5 3.5h6M5 6.5h6M5 9.5h6M5 12.5h6" /></svg>,
  related: () => <svg viewBox="0 0 16 16"><path d="M6 1.5v13M10 1.5v8c0 1.8 1 3 3 5" /><rect x="3.5" y="4.5" width="9" height="3.5" rx="1.75" /></svg>,
  // two parallel tracks with an express pair added beside them
  doubleUp: () => <svg viewBox="0 0 16 16"><path d="M1.5 3h13M1.5 6h13M8 9v6M5 12h6" /></svg>,
  // T-2951: track with a square point on it, and a plus
  addPoint: () => <svg viewBox="0 0 16 16"><path d="M1.5 13.5l4.5-3M10 10.5l4.5 3M12 1.5v5M9.5 4h5" /><rect x="6" y="8.5" width="4" height="4" /></svg>,
  bus: () => <svg viewBox="0 0 16 16"><rect x="3" y="2.5" width="10" height="9" rx="2" /><path d="M3 7h10M5.5 11.5v2M10.5 11.5v2" /><circle class="solid" cx="5.5" cy="9.3" r=".7" /><circle class="solid" cx="10.5" cy="9.3" r=".7" /></svg>,
  // T-2194: a boat on waves
  tram: () => <svg viewBox="0 0 16 16"><rect x="3.5" y="4" width="9" height="7.5" rx="1.8" /><path d="M3.5 8h9M6 4V2.5h4V4M5 13.8l1.2-2.3M11 13.8l-1.2-2.3" /><circle class="solid" cx="5.8" cy="9.8" r=".7" /><circle class="solid" cx="10.2" cy="9.8" r=".7" /></svg>,
  ferry: () => <svg viewBox="0 0 16 16"><path d="M2.5 8.5h11l-1.8 3H4.3zM5 8.5V6h5.5l1 2.5M7 6V4" /><path d="M1.5 13.8c1.1-.7 2.1-.7 3.2 0s2.1.7 3.3 0 2.1-.7 3.2 0 2.1.7 3.3 0" /></svg>,
  // T-2565: three rows of waves, the diagram's water switch
  waves: () => <svg viewBox="0 0 16 16"><path d="M1.5 4.8c1.1-.7 2.1-.7 3.2 0s2.1.7 3.3 0 2.1-.7 3.2 0 2.1.7 3.3 0M1.5 8.5c1.1-.7 2.1-.7 3.2 0s2.1.7 3.3 0 2.1-.7 3.2 0 2.1.7 3.3 0M1.5 12.2c1.1-.7 2.1-.7 3.2 0s2.1.7 3.3 0 2.1-.7 3.2 0 2.1.7 3.3 0" /></svg>,
  bin: () => <svg viewBox="0 0 16 16"><path d="M2.5 4.5h11M6.5 4.5V2.5h3v2M4 4.5l.8 9h6.4l.8-9M6.8 7v4M9.2 7v4" /></svg>,
  // the blueprint selection tool (a dashed box), and switching the selection off or on
  marquee: () => <svg viewBox="0 0 16 16"><path d="M2.5 5.5v-3h3M10.5 2.5h3v3M13.5 10.5v3h-3M5.5 13.5h-3v-3M7 7l5 2-2 .8L9 12z" /></svg>,
  switchOff: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5" /><path d="M4.1 11.9l7.8-7.8" /></svg>,
  switchOn: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5" /><path d="M5.3 8.3l1.8 1.8 3.6-3.9" /></svg>,
  // the dock's tabs and the save button
  // Build's wrench (T-2415: a claw hammer is not a train-building tool)
  wrench: () => <svg viewBox="0 0 16 16"><path d="M13.2 4.6a3.3 3.3 0 01-4.4 3.6L3.9 13.1a1.3 1.3 0 01-1.9-1.9l4.9-4.9a3.3 3.3 0 013.6-4.4L8.6 4l.5 2.4 2.4.5z" /></svg>,
  route: () => <svg viewBox="0 0 16 16"><path d="M2.5 12.5l4-6 4 3 3-6.5" /><circle class="solid" cx="2.5" cy="12.5" r="1.5" /><circle class="solid" cx="6.5" cy="6.5" r="1.5" /><circle class="solid" cx="10.5" cy="9.5" r="1.5" /><circle class="solid" cx="13.5" cy="3" r="1.5" /></svg>,
  // a platform under its canopy, the track beside it
  building: () => <svg viewBox="0 0 16 16"><path d="M1.5 5.5h13l-2-3h-9zM4 5.5v5M12 5.5v5M1.5 10.5h13M1.5 13.5h13" /></svg>,
  money: () => <svg viewBox="0 0 16 16"><rect x="1.5" y="3.5" width="10" height="6.5" /><circle cx="6.5" cy="6.75" r="1.4" /><circle class="bg" cx="11.5" cy="11.5" r="3" /><circle cx="11.5" cy="11.5" r="1.2" /></svg>,
  save: () => <svg viewBox="0 0 16 16"><path d="M2.5 2.5h9l2 2v9h-11zM5 2.5v3.5h5v-3.5M4.5 13.5v-4.5h7v4.5" /></svg>,
  // copying a line's schedule and pasting it on another (T-1293)
  copy: () => <svg viewBox="0 0 16 16"><rect x="5.5" y="5.5" width="8" height="8" rx="1" /><path d="M10.5 3.5v-1h-8v8h1" /></svg>,
  // T-2603: the timetable chart, three train runs crossing between two stations
  marey: () => <svg viewBox="0 0 16 16"><path d="M1.5 2.5h13M1.5 13.5h13M2 13.5l4-11M7.5 2.5l4 11M10 13.5l4-11" /></svg>,
  // T-3346: the schedule panel, a station timetable's hour column and rows
  timetable: () => <svg viewBox="0 0 16 16"><rect x="2.5" y="2.5" width="11" height="11" rx="1" /><path d="M2.5 5.5h11M5.5 5.5v8M7.5 8h4M7.5 10.5h3" /></svg>,
  paste: () => <svg viewBox="0 0 16 16"><path d="M5.5 3h-2v11.5h9V3h-2" /><rect x="5.5" y="1.5" width="5" height="3" rx="0.5" /><path d="M6 8h4M6 10.5h4" /></svg>,
  // the Lines and Stations tabs' scope toggle (the viewed place, the whole world)
  calendar: () => <svg viewBox="0 0 16 16"><rect x="2.5" y="3.5" width="11" height="11.5" rx="1" /><path d="M2.5 6.5h11M5.5 2v3M10.5 2v3" /></svg>,
  // a long holiday's going-out and coming-back days (T-1373), the sky's arrows: out to the right
  goOut: () => <svg viewBox="0 0 16 16"><path d="M2.5 8h10M9 4.5L12.5 8 9 11.5" /></svg>,
  chevron: () => <svg viewBox="0 0 16 16"><path d="M6 3.5L10.5 8 6 12.5" /></svg>,
  leaveGroup: () => <svg viewBox="0 0 16 16"><rect x="1.5" y="2.5" width="8" height="11" rx="1" /><path d="M6.5 8h8M12 5.5L14.5 8 12 10.5" /></svg>,
  goBack: () => <svg viewBox="0 0 16 16"><path d="M13.5 8h-10M7 4.5L3.5 8 7 11.5" /></svg>,
  pin: () =><svg viewBox="0 0 16 16"><path d="M8 14.5s4.5-4.2 4.5-7.6a4.5 4.5 0 00-9 0c0 3.4 4.5 7.6 4.5 7.6z" /><circle cx="8" cy="6.8" r="1.6" /></svg>,
  globe: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" /><path d="M2 8h12M8 2c-2.2 2-2.2 10 0 12M8 2c2.2 2 2.2 10 0 12" /></svg>,
  // how commuters reach a station: on foot, or by a ride (car, bus, bike or lift)
  walk: () => <svg viewBox="0 0 16 16"><circle cx="9.5" cy="2.8" r="1.3" /><path d="M9.5 5.2l-1.8 3 2.3 2.3v3.7M7.7 8.2L5.3 9.4M9.6 6.4l2.4 1.8M7.7 8.2L6.2 14" /></svg>,
  ride: () => <svg viewBox="0 0 16 16"><path d="M1.8 11.2V8.4l1.6-3.4h9.2l1.6 3.4v2.8zM1.8 8.4h12.4" /><circle cx="4.8" cy="11.4" r="1.3" class="bg" /><circle cx="11.2" cy="11.4" r="1.3" class="bg" /></svg>,
  // an inter-region trip's modes (with `ride` for the car)
  train: () => <svg viewBox="0 0 16 16"><rect x="3.8" y="1.8" width="8.4" height="9.7" rx="2" /><path d="M3.8 7h8.4M6 14.2l1-2.7M10 14.2l-1-2.7" /></svg>,
  plane: () => <svg viewBox="0 0 16 16"><path d={PLANE_PATH} /></svg>,
  // an alert's mark, a circled exclamation point, coloured by its severity
  alert: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.2" /><path d="M8 4.6v3.2" /><circle class="solid" stroke="none" cx="8" cy="11.1" r="1.4" /></svg>,
  // the Cities tab, a skyline
  cities: () => <svg viewBox="0 0 16 16"><path d="M1.5 14h13M3 14V7h3.5v7M6.5 14V2.5h3.5V14M10 14V8h3v6" /></svg>,
  // T-2395: the game's home, filled where it is this place
  home: () => <svg viewBox="0 0 16 16"><path d="M2 7.5L8 2l6 5.5M3.8 6v7.8h3V9.8h2.4v4h3V6" /></svg>,
  homeOn: () => <svg viewBox="0 0 16 16"><path class="solid" d="M8 2L2 7.5h1.8v6.3h3V9.8h2.4v4h3V7.5H14z" /></svg>,
  // a city's star in the Cities list
  star: () => <svg viewBox="0 0 16 16"><path class="solid" d="M8 1.5l1.9 4.1 4.5.5-3.4 3 1 4.4L8 11.3l-4 2.2 1-4.4-3.4-3 4.5-.5z" /></svg>,
  // the account (head and shoulders) and the leaderboards
  // T-988: a game on the account, and one only in this browser
  cloud: () => <svg viewBox="0 0 16 16"><path d="M4.6 12.5h7a2.6 2.6 0 00.3-5.2 3.9 3.9 0 00-7.5-.9 2.9 2.9 0 00.2 6.1z" /></svg>,
  cloudOff: () => <svg viewBox="0 0 16 16"><path d="M4.6 12.5h7a2.6 2.6 0 00.3-5.2 3.9 3.9 0 00-7.5-.9 2.9 2.9 0 00.2 6.1zM2.5 2.5l11 11" /></svg>,
  download: () => <svg viewBox="0 0 16 16"><path d="M8 2.5v7.5M4.8 7l3.2 3.2L11.2 7M2.5 11v2.5h11V11" /></svg>,
  history: () => <svg viewBox="0 0 16 16"><path d="M2.9 9.2a5.2 5.2 0 1 0 .9-4.6M3 2.5v2.6h2.6M8 5.2V8.3l2.1 1.4" /></svg>,
  person: () => <svg viewBox="0 0 16 16"><circle cx="8" cy="5.2" r="2.9" /><path d="M2.4 14.2c.6-3 2.9-4.6 5.6-4.6s5 1.6 5.6 4.6" /></svg>,
  // T-2124: the achievements, a medal on its ribbon
  medal: () => <svg viewBox="0 0 16 16"><path d="M4.5 1.5h7L9.3 6.6M4.5 1.5l2.2 5.1" /><circle cx="8" cy="10.3" r="3.9" /><circle class="solid" cx="8" cy="10.3" r="1.4" /></svg>,
  trophy: () => <svg viewBox="0 0 16 16"><path d="M4.5 2.5h7v3.5a3.5 3.5 0 01-7 0zM4.5 3.8H2.5c0 2 .8 3.2 2.3 3.4M11.5 3.8h2c0 2-.8 3.2-2.3 3.4M8 9.5v2.5M5.5 13.5h5" /></svg>,
  people: () => <svg viewBox="0 0 16 16"><circle cx="5.8" cy="5.4" r="2.4" /><path d="M1.5 13.5c.4-2.7 2.1-4.2 4.3-4.2s3.9 1.5 4.3 4.2M10.4 3.2a2.4 2.4 0 010 4.4M11.9 9.6c1.4.6 2.3 1.9 2.6 3.9" /></svg>,
  share: () => <svg viewBox="0 0 16 16"><path d="M8 10V2.5M4.8 5.5L8 2.3l3.2 3.2M3 8.5v5h10v-5" /></svg>,
  // T-3115: a share's link, two links of a chain
  link: () => <svg viewBox="0 0 16 16"><path d="M6.4 9.6l3.2-3.2M7.2 4.8l1.3-1.3a2.6 2.6 0 013.7 3.7l-1.3 1.3M8.8 11.2l-1.3 1.3a2.6 2.6 0 01-3.7-3.7l1.3-1.3" /></svg>,
  heart: () => <svg viewBox="0 0 16 16"><path d="M8 13.8S2 10.3 2 6.1A3.1 3.1 0 018 5a3.1 3.1 0 016 1.1c0 4.2-6 7.7-6 7.7z" /></svg>,
  flag: () => <svg viewBox="0 0 16 16"><path d="M3.5 14.5v-12M3.5 2.5h9l-2 3 2 3h-9" /></svg>,
  gear: () => (
    <svg viewBox="0 0 16 16">
      <path d="M6.83 3.14L7.09 1.26L8.91 1.26L9.17 3.14L10.61 3.74L12.13 2.59L13.41 3.87L12.26 5.39L12.86 6.83L14.74 7.09L14.74 8.91L12.86 9.17L12.26 10.61L13.41 12.13L12.13 13.41L10.61 12.26L9.17 12.86L8.91 14.74L7.09 14.74L6.83 12.86L5.39 12.26L3.87 13.41L2.59 12.13L3.74 10.61L3.14 9.17L1.26 8.91L1.26 7.09L3.14 6.83L3.74 5.39L2.59 3.87L3.87 2.59L5.39 3.74z" />
      <circle cx="8" cy="8" r="2.2" />
    </svg>
  ),
};

/** A line's bullet (T-521). Everywhere it is drawn a hover names the line and a click selects it,
 * except `plain` (the selected line's own panel, a search row that is the line). `pick` replaces
 * the selecting. `gone`: hidden but keeping its space (a station row's bullets past its "+", T-530). */
export function Chip({ line, size = "", plain, pick, gone, tip }: { line: LineInput; size?: "" | "sm" | "lg"; plain?: boolean; pick?: (l: LineInput) => void; gone?: boolean; tip?: string }) {
  const click = plain ? undefined : (e: MouseEvent) => {
    e.stopPropagation();
    if (pick) pick(line);
    else selection.value = { kind: "line", line: line.id };
  };
  void bulletEpoch.value;
  const zoom = useCssZoom();
  const shape = shapeOf(line.id);
  const b0 = bulletTextSoon(line.letter, size, zoom.value), b = b0 && bulletDraw(b0, shape);
  const idle = isIdleLine(line.id);
  return (
    <span ref={zoom.ref} class={"chip " + size + (plain ? "" : " link") + (shape === "square" ? " square" : "") + (idle ? " idle" : "")} style={{ "--c": line.colour, "--ct": inkOn(line.colour), visibility: gone ? "hidden" : undefined }}
      data-tip={plain ? undefined : (tip ?? line.name) + (idle ? "\n" + IDLE_TIP : "")} onClick={click}>
      {b ? (
        <svg style={{ width: b.w + "px", height: b.h + "px" }}>
          <BulletOutline b={b} />
          <text x={b.x} y={b.y}>{line.letter}</text>
        </svg>
      ) : <span>{line.letter}</span>}
    </span>
  );
}

/** T-3319: a shape alone, sized as a 12 px bullet of it would be. */
export function ShapeIcon({ shape }: { shape: LineShape }) {
  const b = bulletDraw({ w: 12, h: 12, x: 0, y: 0, top: 0, iw: 0, ih: 0 }, shape);
  return <svg class="shape-icon" style={{ width: b.w + "px", height: b.h + "px" }} aria-hidden="true"><BulletOutline b={b} /></svg>;
}

const SHAPE_NAME: Record<PickShape, MessageKey> = { circle: "widgets.circle", square: "widgets.square", diamond: "widgets.diamond" };

/** T-3319: a row of the bullet shapes to pick from, drawn in `colour` (the text's when absent); `tips` for a hover on some. */
export function ShapePicker({ value, onChange, colour, tips }: { value: LineShape; onChange: (s: PickShape) => void; colour?: string; tips?: (s: PickShape) => string | undefined }) {
  return (
    <div class="group shapes" role="radiogroup" style={colour ? { "--c": colour } : undefined}>
      {LINE_SHAPES.map((s) => (
        <button class={"btn" + (s === value ? " on" : "")} role="radio" aria-checked={s === value} aria-label={t(SHAPE_NAME[s])} data-tip={tips?.(s)} onClick={() => onChange(s)}>
          <ShapeIcon shape={s} />
        </button>
      ))}
    </div>
  );
}

/** T-3319: a bullet's outline in its shape (ui/bullet.ts `bulletDraw`). */
const BulletOutline = ({ b }: { b: BulletDraw }) =>
  b.path ? <path d={b.path.d} stroke-width={b.path.stroke} /> : <rect y={b.rect!.y} width={b.rect!.w} height={b.rect!.h} rx={b.rect!.r} />;

/** The CSS zoom an element is drawn at (the panels are zoomed by the UI scale, T-247): the UI scale
 * until it is in the page, then what the browser reports, so a bullet's baseline lands on a whole
 * device pixel (ui/bullet.ts). */
function useCssZoom() {
  const ref = useRef<HTMLSpanElement>(null);
  const guess = uiScale.value;
  const [seen, setSeen] = useState<number | null>(null);
  const value = seen ?? guess;
  // T-1021: read after the paint, when the page's style is worked out already: in a layout effect the
  // first chip's read worked out the whole new list's style in the same task (174 ms on Japan's Lines tab)
  useEffect(() => {
    const z = (ref.current as (HTMLElement & { currentCSSZoom?: number }) | null)?.currentCSSZoom;
    if (typeof z === "number" && z > 0 && Math.abs(z - value) > 1e-6) setSeen(z);
  });
  return { ref, value };
}

/** Shake a control once more, from the start if it is shaking already (style.css `.shake`; the
 * control clears the class on `animationend`). */
export function shakeEl(el: HTMLElement | null) {
  if (!el) return;
  el.classList.remove("shake");
  void el.offsetWidth;
  el.classList.add("shake");
}

/** `jump`: Shift-click steps by 5 and Ctrl-click by 20 (T-516), the buttons' hover says so, with
 * `jumpUnit` after the numbers ("min", the headway's, T-1888).
 * `type` (T-626): a click on the value opens a box for typing it; Enter or leaving the box applies
 * `set` with the number typed (anything else is dropped), Escape cancels; `set` returning false is a
 * refusal, shaken and knocked like a step's (T-3152). `text` is what the box
 * starts with, `tip` the value's hover. `onStep` returns false for a step refused (T-1844): the
 * stepper shakes and the refusal knock plays; a promise of false is an edit the worker refused. */
export function Stepper(p: { value: ComponentChildren; onStep: (d: number) => boolean | void | Promise<boolean>; wide?: boolean; label: string; jump?: boolean; jumpUnit?: string; type?: { text: string; set: (v: number) => void | boolean; tip?: string }; tipDown?: string; tipUp?: string; tipValue?: string }) {
  const by = (e: MouseEvent) => (p.jump ? (e.ctrlKey ? 20 : e.shiftKey ? 5 : 1) : 1);
  const jump = p.jump && !mobileDevice() ? (p.jumpUnit ? t("widgets.jumpUnit", { unit: p.jumpUnit }) : t("widgets.jump")) : undefined;
  const tip = (own?: string) => (own && jump ? own + "\n" + jump : own ?? jump);
  const box = useRef<HTMLSpanElement>(null);
  // T-1844: a step that returns false was refused (a limit, or no money): the stepper shakes and the
  // refusal knock plays, as for a refused edit
  const shake = () => shakeEl(box.current);
  // false: refused here for a limit, so the knock plays too. A promise of false: an edit the worker
  // refused, whose notice has played the knock already.
  const step = (d: number) => {
    const r = p.onStep(d);
    if (r === false) { knock(); shake(); }
    else if (r instanceof Promise) void r.then((ok) => { if (ok === false) shake(); });
  };
  return (
    <span ref={box} class={"stepper" + (p.wide ? " wide" : "")} onAnimationEnd={() => box.current?.classList.remove("shake")}>
      <button class="btn" aria-label={t("widgets.less", { label: p.label })} data-tip={tip(p.tipDown)} onClick={(e) => step(-by(e))}>
        <Icon.minus />
      </button>
      {p.type ? <TypedValue label={p.label} type={{ ...p.type, set: (v) => { if (p.type!.set(v) === false) { knock(); shake(); } } }}>{p.value}</TypedValue> : <span class="v" data-tip={p.tipValue}>{p.value}</span>}
      <button class="btn" aria-label={t("widgets.more", { label: p.label })} data-tip={tip(p.tipUp)} onClick={(e) => step(by(e))}>
        <Icon.plus />
      </button>
    </span>
  );
}

/** A stepper's value that opens a box on a click (T-626); also alone, a typed value with no steps. */
export function TypedValue(p: { label: string; type: { text: string; set: (v: number) => void | boolean; tip?: string }; children: ComponentChildren }) {
  const [draft, setDraft] = useState<string | null>(null);
  const done = useRef(false);
  const fresh = useRef(false);
  // focused as the box appears, so what is typed next goes into it
  const open = (el: HTMLInputElement | null) => {
    if (!el || !fresh.current) return;
    fresh.current = false;
    el.focus();
    el.select();
  };
  const commit = () => {
    if (done.current) return;
    done.current = true;
    const v = parseFloat((draft ?? "").replace(/[^0-9.\-]/g, ""));
    setDraft(null);
    if (Number.isFinite(v)) p.type.set(v);
  };
  if (draft === null) {
    return <button class="v typed" data-tip={p.type.tip} aria-label={t("widgets.type", { label: p.label })} onClick={() => { done.current = false; fresh.current = true; setDraft(p.type.text); }}>{p.children}</button>;
  }
  return (
    <input class="v typing" ref={open} value={draft} inputMode="decimal" aria-label={p.label}
      onInput={(e) => setDraft(e.currentTarget.value)} onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") { done.current = true; setDraft(null); }
      }} />
  );
}

/** Buttons sharing borders, one of them on. */
/** `tips`: a hover for the options that have one */
export function Segmented<T>(p: { options: [T, ComponentChildren][]; value: T; onChange: (v: T) => void; tips?: (v: T) => string | undefined }) {
  return (
    <div class="group">
      {p.options.map(([v, label]) => (
        <button class={"btn" + (v === p.value ? " on" : "")} data-tip={p.tips?.(v)} onClick={() => p.onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

export function Check(p: { checked: boolean; onChange: (v: boolean) => void; children: ComponentChildren }) {
  return (
    <label class="check">
      <input type="checkbox" checked={p.checked} onChange={(e) => p.onChange((e.target as HTMLInputElement).checked)} />
      {p.children}
    </label>
  );
}

/** T-310: a list's column headers. A click sorts by the column (numbers biggest first), a second
 * click reverses it, and the sorted column carries a small arrow. A column with a `cls` is a
 * right-aligned number column of that width; the arrow sits before its label so the label stays
 * over the figures. */
/** `hand` (T-1813): the list is in the player's own order, so no column is on; a click sorts again
 * and calls it to drop the order. */
export function SortHead<K extends string>({ sort, cols, hand }: { sort: Signal<ListSort<K>>; cols: { key: K; label: string; tip: string; first: 1 | -1; cls?: string; left?: boolean }[]; hand?: () => void }) {
  const s = sort.value;
  return (
    <div class="sort-head">
      {cols.map((c) => {
        const on = !hand && s.key === c.key;
        const arrow = <i class={"sort-arrow" + (on ? (s.dir > 0 ? " up" : " down") : "")} />;
        return (
          <button class={"sort-btn" + (c.cls ? " val " + c.cls : " grow") + (on ? " on" : "")} data-tip={c.tip}
            onClick={() => { hand?.(); sort.value = on ? { key: c.key, dir: s.dir === 1 ? -1 : 1 } : { key: c.key, dir: c.first }; }}>
            {c.cls && !c.left && arrow}{c.label}{(!c.cls || c.left) && arrow}
          </button>
        );
      })}
    </div>
  );
}

/** T-748: one alert in a plain list (`<ul class="alerts">`): a red exclamation point, the text, and
 * under it any detail lines. */
export function Alert({ tip, children }: { tip?: string; children: ComponentChildren }) {
  return (
    <li class="alert bad">
      <span class="alert-mark"><Icon.alert /></span>
      <div class="alert-body" data-tip={tip}>{children}</div>
    </li>
  );
}

export function Row(p: { label: ComponentChildren; children?: ComponentChildren; bold?: boolean }) {
  return (
    <li class={p.bold ? "b" : ""}>
      <span class="grow">{p.label}</span>
      {p.children !== undefined && <span class="val">{p.children}</span>}
    </li>
  );
}

/** T-1134: the buttons of an inspector, in a row right under its header, never after a list of any length. */
export function Actions({ children }: { children: ComponentChildren }) {
  return <div class="inspector-actions">{children}</div>;
}

/** A bar of coloured segments with a hover each: `share` is a segment's part of the bar (0 to 1),
 * `cls` its class or `colour` its fill. `kind` is the bar's look: the trip panels' or the split bars'. */
export function SegBar({ parts, kind, label }: { parts: { share: number; cls?: string; colour?: string; tip: string }[]; kind: "trip-bar" | "split-bar"; label: string }) {
  return (
    <div class={"bar " + kind} role="img" aria-label={label}>
      {parts.map((p) => p.share > 0 && (
        <i class={"seg" + (p.cls ? " " + p.cls : "")} style={{ width: 100 * p.share + "%", ...(p.colour && { background: p.colour }) }} data-tip={p.tip} />
      ))}
    </div>
  );
}

/** The elevation buttons in the level colours, `value` on; `tip` words each hover. */
export function LevelButtons({ value, onPick, tip }: { value: Level | null; onPick: (l: Level) => void; tip: (l: Level) => string }) {
  return (
    <div class="group levels">
      {LEVELS.map((l) => (
        <button class={"btn lv" + (l === value ? " on" : "")} style={{ "--lc": LEVEL_COLOURS[l], "--lk": inkOn(LEVEL_COLOURS[l]) }}
          data-tip={tip(l)} aria-label={t("widgets.level", { level: levelLabel(l) })} aria-pressed={l === value} onClick={() => onPick(l)}>
          {levelLabel(l)}
        </button>
      ))}
    </div>
  );
}

/** The track count's 1-line and 2-line icon buttons, `value` on (null: neither), the track tool's (T-2931, with its
 * quad button as `children`) and the Select tool's bar's (T-3321). `single` false hides the 1-line one. */
export function TrackCountButtons({ value, onPick, single = true, children }: { value: 1 | 2 | null; onPick: (n: 1 | 2) => void; single?: boolean; children?: ComponentChildren }) {
  return (
    <div class="group tb-count" role="group" aria-label={t("toolBar.trackCount")}>
      {single && <button class={"btn icon" + (value === 1 ? " on" : "")} aria-label={t("toolBar.singleTrack")} aria-pressed={value === 1} data-tip={t("toolBar.singleTrack")}
        onClick={() => onPick(1)}><Icon.tracks1 /></button>}
      <button class={"btn icon" + (value === 2 ? " on" : "")} aria-label={t("toolBar.doubleTrack")} aria-pressed={value === 2} data-tip={t("toolBar.doubleTrack")}
        onClick={() => onPick(2)}><Icon.tracks2 /></button>
      {children}
    </div>
  );
}

/** T-2601: a button label that changes with its state, both laid in one cell so the button keeps
 * the wider one's width either way (a changing width wrapped the row, T-2525). */
export function SwapLabel({ on, off, onText }: { on: boolean; off: string; onText: string }) {
  return (
    <span class="swap-label">
      <span class={on ? "off" : undefined} aria-hidden={on}>{off}</span>
      <span class={on ? undefined : "off"} aria-hidden={!on}>{onText}</span>
    </span>
  );
}

/** A "Blueprint" note under an inspector's header until what it shows is constructed. */
export function Built({ built }: { built: boolean }) {
  return built ? null : <p class="blueprint-tag">{t("widgets.blueprint")}</p>;
}

/** Construct (while a blueprint) on the left and Remove on the right of one row. */
export function ConstructRemove({ built, construct, constructLabel, remove, removeLabel, removeTip }: {
  built: boolean; construct: () => void; constructLabel: string; remove: () => void; removeLabel: string; removeTip: string;
}) {
  return (
    <Actions>
      {!built ? <button class="btn text" onClick={construct}>{constructLabel}</button> : <span />}
      <button class="btn text" onClick={remove} data-tip={removeTip}>{removeLabel}</button>
    </Actions>
  );
}
