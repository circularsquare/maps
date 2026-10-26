// The building sounds (T-1282, T-1413, T-1474), under the same Sound switch and volume as the station
// sounds. They play at the player's click (T-1474), not when the clock worker answers the edit: "ke-cha"
// (synthesised) when blueprint is constructed, "bung" (synthesised) when an upgrade or a passage is
// built, and three recordings: the fft when track is placed, a finger snap when a station is, and a
// quiet cork pop for a panel click that changes the world. Not tied to the zoom: the player made them
// happen. One sound per click, however many pieces it places. A line opening (T-1434) plays "bung".

import { display } from "../game/ui";
import { onLinesOpened } from "../game/openingDay";
import { setBeforeEdit, world, type Intent } from "../game/world";
import { audioOut } from "./audioOut";
import { bung, keCha, sampleVoice } from "./buildVoices";
import { sample, type SampleName } from "./samples";

/** The sound an edit makes at the click, or null for a silent one. Track and stations placed are
 * played by the tools themselves (map/tools.ts), which know whether the ghost refuses the click. */
function soundOf(i: Intent): BuildSound | null {
  switch (i.op) {
    case "construct": case "constructAll": case "constructLine":
      return "construct";
    case "stationUpgrade": case "link":
      return i.on ? "bung" : "pop";
    case "stationProps": {
      // a new platform length changes the stop; a new name does not
      const s = world.peek()?.save.stations.find((x) => x.num === i.node);
      return s && s.length !== i.platform ? "pop" : null;
    }
    case "setTrainCount": case "setHeadway": case "lineCars": case "lineFast":
    case "setStops": case "insertStop": case "removeLine": case "copyLine":
    case "doubleUp": case "edgeLevel": case "edgesLevel": case "flying":
      return "pop";
    default:
      return null;
  }
}

/** Plays the building sounds as the player's edits are sent. */
export function startBuildSounds() {
  setBeforeEdit((i) => {
    const k = soundOf(i);
    if (k) playBuild(k);
  });
  // T-1434: a line opening, a beat after the construct's own sound
  onLinesOpened(() => {
    const since = performance.now() - lastConstruct;
    setTimeout(() => playBuild("bung"), since < 2000 ? Math.max(0, OPENING_DELAY_MS - since) : OPENING_DELAY_MS);
  });
}

/** The gap from the start of a "ke-cha" to the opening day's "bung", ms. The "ke-cha" is played at the
 * click, so the worker's answer has used part of the gap already. */
const OPENING_DELAY_MS = 250;
let lastConstruct = -1e9;

/** The loudest sample of each recording at volume 1, as a fraction of full scale. The pop sits well
 * under the others. */
const PEAK: Record<SampleName, number> = { track: 0.25, snap: 0.3, pop: 0.09 };

/** The level at volume 1 for the synthesised sounds, as the station sounds' master
 * (stationSound.ts `MASTER`). */
const MASTER = 0.5;

export type BuildSound = "construct" | "bung" | SampleName;

/** The same sound twice within this many seconds plays once: a drag that places track several times. */
const GAP_S: Record<BuildSound, number> = { construct: 0.3, track: 0.1, bung: 0.3, snap: 0.15, pop: 0.05 };
const last: Record<BuildSound, number> = { construct: -1, track: -1, bung: -1, snap: -1, pop: -1 };

export function playBuild(kind: BuildSound) {
  const d = display.peek();
  if (!d.sound || d.soundVolume <= 0) return;
  const o = audioOut();
  if (!o) return;
  const { ctx, out } = o, t = ctx.currentTime;
  if (t - last[kind] < GAP_S[kind]) return;
  if (kind === "track" || kind === "snap" || kind === "pop") {
    // a file still decoding stays silent, without using up the gap
    const s = sample(kind);
    if (!s) return;
    last[kind] = t;
    sampleVoice(ctx, out, t, s, PEAK[kind] * d.soundVolume);
    return;
  }
  last[kind] = t;
  const level = MASTER * d.soundVolume;
  if (kind === "construct") {
    lastConstruct = performance.now();
    keCha(ctx, out, t, level);
  } else bung(ctx, out, t, level);
}
