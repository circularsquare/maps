// The building sounds (T-1282, T-1413), made with WebAudio as the station sounds are and under the same
// Sound switch and volume: "ke-cha" when blueprint is constructed, "bung" when a station is placed or
// an upgrade added, "tsh" when track is placed (voices in buildVoices.ts). Not tied to the zoom: the
// player made them happen. One sound per edit, however many pieces it places.

import { display } from "../game/ui";
import { setAfterEdit } from "../game/world";
import { audioOut } from "./audioOut";
import { bung, keCha, tsh } from "./buildVoices";

/** Plays the building sounds as the player's edits are taken. */
export function startBuildSounds() {
  setAfterEdit((op, a) => {
    if (!a.ok) return;
    if (op.op === "route") playBuild("track");
    else if (op.op === "addStation" || (op.op === "stationUpgrade" && op.on) || (op.op === "link" && op.on)) playBuild("bung");
    else if (op.op === "construct" || op.op === "constructAll" || op.op === "constructLine") playBuild("construct");
  });
}

/** The level at volume 1, as the station sounds' master (stationSound.ts `MASTER`). */
const MASTER = 0.5;

export type BuildSound = "construct" | "track" | "bung";

/** The same sound twice within this many seconds plays once: a drag that places track several times. */
const GAP_S: Record<BuildSound, number> = { construct: 0.3, track: 0.1, bung: 0.3 };
const last: Record<BuildSound, number> = { construct: -1, track: -1, bung: -1 };

export function playBuild(kind: BuildSound) {
  const d = display.peek();
  if (!d.sound || d.soundVolume <= 0) return;
  const o = audioOut();
  if (!o) return;
  const { ctx, out } = o, t = ctx.currentTime;
  if (t - last[kind] < GAP_S[kind]) return;
  last[kind] = t;
  const level = MASTER * d.soundVolume;
  if (kind === "construct") keCha(ctx, out, t, level);
  else if (kind === "track") tsh(ctx, out, t, level);
  else bung(ctx, out, t, level);
}
