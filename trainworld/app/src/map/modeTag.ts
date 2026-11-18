// T-3693: the track tool's mark by the pointer: a stroke a track (the track count buttons' icons) in the track
// type's colour.

import { MODE_HALO_COLOURS } from "../game/palette";
import type { HaloMode } from "../game/buildMode";

/** the track count buttons' icons (ui/widgets.tsx `Icon.tracks1/2/4`), 20 x 20 */
const TRACK_PATHS = { 1: "M10 3v14", 2: "M6 3v14M14 3v14", 4: "M4 3v14M8 3v14M12 3v14M16 3v14" } as const;

export function modeMarkSvg(mode: HaloMode, count: 1 | 2 | 4): SVGSVGElement {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 20 20");
  s.setAttribute("class", "mode-mark-svg");
  s.innerHTML = `<path d="${TRACK_PATHS[count]}"/>`;
  s.style.setProperty("--mode", MODE_HALO_COLOURS[mode]);
  return s;
}
