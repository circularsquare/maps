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
import { setAfterEdit, setBeforeEdit, setRefuseHook, world, type Intent } from "../game/world";
import type { EditOp } from "../workers/protocol";
import type { EditAnswer } from "../workers/clockClient";
import { audioOut } from "./audioOut";
import { loadChime, sampleVoice } from "./buildVoices";
import { sample, type SampleName } from "./samples";

/** The sound an edit makes at the click, or null for a silent one. Track and stations placed are
 * played by the tools themselves (map/tools.ts), which know whether the ghost refuses the click. */
function soundOf(i: Intent): BuildSound | null {
  switch (i.op) {
    case "stationUpgrade": case "link":
      return i.on ? null : "pop"; // building one sounds when the worker accepts it (afterEditSound)
    case "stationProps": {
      // a new platform length changes the stop; a new name does not
      const s = world.peek()?.save.stations.find((x) => x.num === i.node);
      return s && s.length !== i.platform ? "pop" : null;
    }
    case "setTrainCount": case "setHeadway": case "lineCars": case "lineFast": case "lineOneWay": case "linePairs": case "lineLimit":
    case "setStops": case "insertStop": case "dropStop": case "endLineAt": case "stopPassed": case "orderStops": case "sortStops": case "turnEnd": case "openLoop": case "removeLine": case "copyLine": case "lineGroup":
    case "busSpacing": case "busHand": case "busHandStop": // T-2940: a bus line's stops, as a train line's
    case "doubleUp": case "addPoint": case "edgeLevel": case "edgesLevel": case "edgesKind": case "edgesTracks": case "flying":
      return "pop";
    case "rebuild": return null; // sounds once the worker has accepted it (afterEditSound)
    default:
      return null;
  }
}

/** T-1683: a construct and a built station upgrade or passage sound once the worker has accepted
 * them; one added only to the blueprint (no charge) gets the quiet pop. */
function afterEditSound(op: EditOp, a: EditAnswer) {
  switch (op.op) {
    case "construct": case "constructAll": case "constructLine": case "rebuild":
      playBuild("construct");
      break;
    case "stationUpgrade": case "link":
      if (op.on) playBuild(a.charge > 0 ? "construct" : "pop");
      break;
    case "stationTracks": case "stationWay": case "reverseStation":
      // T-3332: a built station's change is construction, whatever it charges; a blueprint's is a pop
      playBuild(world.peek()?.save.stations.find((x) => x.num === op.node)?.built ? "construct" : "pop");
      break;
  }
}

/** Plays the building sounds as the player's edits are sent. */
export function startBuildSounds() {
  setBeforeEdit((i) => {
    const k = soundOf(i);
    if (k) playBuild(k);
  });
  setAfterEdit(afterEditSound);
  setRefuseHook(() => playBuild("knock")); // T-1659
  setLoadChimeHook(() => playBuild("chime")); // T-1655
}

/** The loudest sample of each recording at volume 1, as a fraction of full scale. The pop sits well
 * under the others; the snap is 3 dB under its first level (T-1658); the knock sits by ear between
 * the pop and the snap, its file being a dense 0.16 s (T-1659). */
const PEAK: Record<SampleName, number> = { snap: 0.21, pop: 0.09, knock: 0.12, construct: 0.15 };

/** T-1658: the snap and the pop each play one of four variants, a little different in pitch (playback
 * rate) and level, never the same one twice in a row. The construct (T-1683) has three. */
export const VARIANTS: readonly { rate: number; gain: number }[] = [
  { rate: 0.94, gain: 0.95 }, { rate: 0.98, gain: 1 }, { rate: 1.02, gain: 0.85 }, { rate: 1.06, gain: 0.9 },
];
export const CONSTRUCT_VARIANTS: readonly { rate: number; gain: number }[] = [
  { rate: 0.96, gain: 0.95 }, { rate: 1, gain: 1 }, { rate: 1.04, gain: 0.9 },
];
const variantsOf = (kind: SampleName) => (kind === "construct" ? CONSTRUCT_VARIANTS : VARIANTS);
const lastVariant: Partial<Record<SampleName, number>> = {};
export function pickVariant(kind: SampleName, r = Math.random()): number {
  const n = variantsOf(kind).length;
  const prev = lastVariant[kind] ?? -1;
  let v = Math.floor(r * (prev < 0 ? n : n - 1));
  if (prev >= 0 && v >= prev) v++;
  return (lastVariant[kind] = v);
}

/** The level at volume 1 for the synthesised sounds, as the station sounds' master
 * (stationSound.ts `MASTER`). */
const MASTER = 0.5;

/** "station" (T-1698) is the snap three semitones lower, so a station placed sounds unlike track. */
export type BuildSound = "chime" | "station" | SampleName;
const STATION_RATE = 2 ** (-3 / 12);

/** The same sound twice within this many seconds plays once: a drag that places track several times. */
const GAP_S: Record<BuildSound, number> = { chime: 1, snap: 0.15, station: 0.15, pop: 0.05, knock: 0.25, construct: 0.3 };
const last: Record<BuildSound, number> = { chime: -1, snap: -1, station: -1, pop: -1, knock: -1, construct: -1 };

/** The last sounds asked for, whether or not they were heard (checks read it as `tw.soundLog`). */
export const soundLog: BuildSound[] = [];

export function playBuild(kind: BuildSound) {
  soundLog.push(kind);
  if (soundLog.length > 100) soundLog.shift();
  const d = display.peek();
  if (!d.sound || d.soundVolume <= 0) return;
  const o = audioOut();
  if (!o) return;
  const { ctx, out } = o, t = ctx.currentTime;
  if (t - last[kind] < GAP_S[kind]) return;
  if (kind !== "chime") {
    const file = kind === "station" ? "snap" : kind;
    // a file still decoding stays silent, without using up the gap
    const s = sample(file);
    if (!s) return;
    last[kind] = t;
    const v = kind === "knock" ? { rate: 1, gain: 1 } : variantsOf(file)[pickVariant(kind === "station" ? "snap" : kind)];
    sampleVoice(ctx, out, t, s, PEAK[file] * v.gain * d.soundVolume, v.rate * (kind === "station" ? STATION_RATE : 1));
    return;
  }
  last[kind] = t;
  loadChime(ctx, out, t, 0.35 * MASTER * d.soundVolume);
}
