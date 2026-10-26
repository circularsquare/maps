// The building sounds (T-1282), made with WebAudio as the station sounds are and under the same Sound
// switch and volume: a little metallic clink when blueprint is constructed, and two smaller,
// non-physical tones as track or a station is placed as blueprint. Not tied to the zoom: the player
// made them happen.

import { display } from "../game/ui";
import { setAfterEdit } from "../game/world";
import { audioOut } from "./audioOut";

/** Plays the building sounds as the player's edits are taken: placing track or a station as blueprint,
 * and constructing. */
export function startBuildSounds() {
  setAfterEdit((op, a) => {
    if (!a.ok) return;
    if (op.op === "route") playBuild("track");
    else if (op.op === "addStation") playBuild("station");
    else if (op.op === "construct" || op.op === "constructAll" || op.op === "constructLine") playBuild("construct");
  });
}

/** The level at volume 1, as the station sounds' master (stationSound.ts `MASTER`). */
const MASTER = 0.5;

export type BuildSound = "construct" | "track" | "station";

/** Sounds closer together than this play once: a drag that places track several times. */
const GAP_S = 0.07;
const last: Record<BuildSound, number> = { construct: -1, track: -1, station: -1 };

export function playBuild(kind: BuildSound) {
  const d = display.peek();
  if (!d.sound || d.soundVolume <= 0) return;
  const o = audioOut();
  if (!o) return;
  const { ctx, out } = o, t = ctx.currentTime;
  if (t - last[kind] < GAP_S) return;
  last[kind] = t;
  const level = MASTER * d.soundVolume;
  if (kind === "construct") clink(ctx, out, t, level);
  else if (kind === "track") blip(ctx, out, t, level);
  else twoNotes(ctx, out, t, level);
}

/** One decaying partial. */
function tone(ctx: AudioContext, dest: AudioNode, t: number, f: number, peak: number, decay: number, type: OscillatorType = "sine", glideTo?: number) {
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.type = type;
  o.frequency.setValueAtTime(f, t);
  if (glideTo) o.frequency.exponentialRampToValueAtTime(glideTo, t + Math.min(0.05, decay * 0.6));
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(peak, t + 0.003);
  g.gain.exponentialRampToValueAtTime(0.0001, t + decay);
  o.connect(g).connect(dest);
  o.start(t);
  o.stop(t + decay + 0.02);
}

/** A small struck-metal clink: a high note with the inharmonic partials of a bar, the upper ones
 * dying first, its pitch a semitone either way at random so repeated constructs vary. */
function clink(ctx: AudioContext, dest: AudioNode, t: number, level: number) {
  const f = 1760 * 2 ** ((Math.random() * 2 - 1) / 12);
  tone(ctx, dest, t, f, 0.3 * level, 0.32);
  tone(ctx, dest, t, f * 2.76, 0.14 * level, 0.12);
  tone(ctx, dest, t, f * 5.4, 0.06 * level, 0.05);
}

/** Track placed: one soft rising blip, quieter and shorter than the clink. */
function blip(ctx: AudioContext, dest: AudioNode, t: number, level: number) {
  tone(ctx, dest, t, 620, 0.13 * level, 0.09, "sine", 930);
}

/** A station placed: two soft notes a fourth apart, close together. */
function twoNotes(ctx: AudioContext, dest: AudioNode, t: number, level: number) {
  tone(ctx, dest, t, 784, 0.12 * level, 0.09, "triangle");
  tone(ctx, dest, t + 0.065, 1046.5, 0.12 * level, 0.12, "triangle");
}
