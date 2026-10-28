// The building sounds (T-1282, T-1413, T-1474, T-1532), under the same Sound switch and volume as the
// station sounds. They play at the player's click (T-1474), not when the clock worker answers the edit:
// "bung" (synthesised) when an upgrade or a passage is built, and three recordings: a finger snap when
// track or a station is placed or blueprint is constructed, a quiet cork pop for a panel click that
// changes the world (the snap and the pop each in four slight variants, T-1658), and a wood knock for a
// refused action (T-1659). A soft two-note chime (synthesised) when a game load that took over 30 s
// finishes (T-1655). Not tied to the zoom: the player made them happen. One sound per click, however
// many pieces it places.

import { display } from "../game/ui";
import { setLoadChimeHook } from "../game/loading";
import { setBeforeEdit, setRefuseHook, world, type Intent } from "../game/world";
import { audioOut } from "./audioOut";
import { bung, loadChime, sampleVoice } from "./buildVoices";
import { sample, type SampleName } from "./samples";

/** The sound an edit makes at the click, or null for a silent one. Track and stations placed are
 * played by the tools themselves (map/tools.ts), which know whether the ghost refuses the click. */
function soundOf(i: Intent): BuildSound | null {
  switch (i.op) {
    case "construct": case "constructAll": case "constructLine":
      return "snap";
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
  setRefuseHook(() => playBuild("knock")); // T-1659
  setLoadChimeHook(() => playBuild("chime")); // T-1655
}

/** The loudest sample of each recording at volume 1, as a fraction of full scale. The pop sits well
 * under the others; the snap is 3 dB under its first level (T-1658); the knock sits by ear between
 * the pop and the snap, its file being a dense 0.16 s (T-1659). */
const PEAK: Record<SampleName, number> = { snap: 0.21, pop: 0.09, knock: 0.12 };

/** T-1658: the snap and the pop each play one of four variants, a little different in pitch (playback
 * rate) and level, never the same one twice in a row. */
export const VARIANTS: readonly { rate: number; gain: number }[] = [
  { rate: 0.94, gain: 0.95 }, { rate: 0.98, gain: 1 }, { rate: 1.02, gain: 0.85 }, { rate: 1.06, gain: 0.9 },
];
const lastVariant: Partial<Record<SampleName, number>> = {};
export function pickVariant(kind: SampleName, r = Math.random()): number {
  const prev = lastVariant[kind] ?? -1;
  let v = Math.floor(r * (prev < 0 ? VARIANTS.length : VARIANTS.length - 1));
  if (prev >= 0 && v >= prev) v++;
  return (lastVariant[kind] = v);
}

/** The level at volume 1 for the synthesised sounds, as the station sounds' master
 * (stationSound.ts `MASTER`). */
const MASTER = 0.5;

export type BuildSound = "bung" | "chime" | SampleName;

/** The same sound twice within this many seconds plays once: a drag that places track several times. */
const GAP_S: Record<BuildSound, number> = { bung: 0.3, chime: 1, snap: 0.15, pop: 0.05, knock: 0.25 };
const last: Record<BuildSound, number> = { bung: -1, chime: -1, snap: -1, pop: -1, knock: -1 };

export function playBuild(kind: BuildSound) {
  const d = display.peek();
  if (!d.sound || d.soundVolume <= 0) return;
  const o = audioOut();
  if (!o) return;
  const { ctx, out } = o, t = ctx.currentTime;
  if (t - last[kind] < GAP_S[kind]) return;
  if (kind === "snap" || kind === "pop" || kind === "knock") {
    // a file still decoding stays silent, without using up the gap
    const s = sample(kind);
    if (!s) return;
    last[kind] = t;
    const v = kind === "knock" ? { rate: 1, gain: 1 } : VARIANTS[pickVariant(kind)];
    sampleVoice(ctx, out, t, s, PEAK[kind] * v.gain * d.soundVolume, v.rate);
    return;
  }
  last[kind] = t;
  if (kind === "chime") loadChime(ctx, out, t, 0.35 * MASTER * d.soundVolume);
  else bung(ctx, out, t, MASTER * d.soundVolume);
}
