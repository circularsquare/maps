// The sound at stations (T-686, T-696): one soft note, the line's own, when a train pulls in and
// small pops as riders get off and on, made with WebAudio (no files). Quiet by design: a low master level, the volume rising
// with the zoom past the one where platforms are drawn, and fewer sounds the more arrivals are
// already sounding. The audio context is the game's one (audioOut.ts).

import { audioOut } from "./audioOut";
import { STATION_BOX_ZOOM } from "./stationGeometry";

/** The master level at volume 1; the pops and the chime sit under it. */
const MASTER = 0.5;
/** Arrivals within this many seconds count as sounding together. */
const RECENT_S = 0.9;
/** More arrivals than this sounding together: the rest stay silent. */
const MAX_RECENT = 3;
/** Pops from one arrival, getting off and getting on, at most each. */
export const MAX_POPS = 4;
/** T-1122: the pops of riders getting off are skipped at this chance, and otherwise shifted as a
 * group by up to this many semitones either way (each pop a semitone more at most), at a level in
 * proportion to the riders getting off: full (today's) at OFF_FULL_RIDERS, never under OFF_FLOOR nor
 * over OFF_CAP times it. */
export const OFF_SKIP = 0.3;
export const OFF_SHIFT_SEMITONES = 4;
const OFF_JITTER_SEMITONES = 1;
export const OFF_FULL_RIDERS = 80;
export const OFF_FLOOR = 0.25;
export const OFF_CAP = 1.25;
/** The level of the getting-off pops for `riders` getting off, as a multiple of the pops' own. */
export function offGain(riders: number): number {
  return Math.max(OFF_FLOOR, Math.min(OFF_CAP, riders / OFF_FULL_RIDERS));
}

/** T-1153: each arrival's sound takes one of three lengths at random. Short (SHORT_CHANCE): a
 * shorter note and one group of pops, getting off or getting on, at most 2, straight after it.
 * Medium (MEDIUM_CHANCE): both groups, at most 2 each. Otherwise full, as many as the riders give. */
export const SHORT_CHANCE = 0.45;
export const MEDIUM_CHANCE = 0.35;
export type SoundLength = "short" | "medium" | "full";
export function soundLength(r: number): SoundLength {
  return r < SHORT_CHANCE ? "short" : r < SHORT_CHANCE + MEDIUM_CHANCE ? "medium" : "full";
}

export interface Arrival {
  /** -1 (left of the view) to 1 */
  pan: number;
  zoom: number;
  volume: number;
  /** the clock's speed index: 0 plays everything, 1 only a chime and a pop or two, 2 nothing */
  speed: number;
  off: number;
  on: number;
  /** riders getting off (T-1122: the getting-off pops' level) */
  offRiders: number;
  /** the line's note, semitones above C5 (0 to 11) */
  note: number;
  /** the platform's riders waited long (the tired face's test): a lower, softer note */
  crowded: boolean;
}

/** The notes are the 12 semitones from C5 (T-706). */
const C5 = 523.25;
/** A minor third down. */
const CROWDED_RATIO = 0.84;
/** The first five lines of a city take these semitones in this order: C, E, G, D, A, a major
 * pentatonic, so a new city starts out sounding pleasant. */
const FIRST_LINES = [0, 4, 7, 2, 9];

/** A line's note, in semitones above C5: by its rank among its city's lines (0 for the first built)
 * for the first five, else from its number, spread over the 12. */
export function lineNote(line: number, rank: number): number {
  if (rank < FIRST_LINES.length) return FIRST_LINES[rank];
  let h = Math.imul(line + 1, 0x9e3779b1);
  h ^= h >>> 15;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  return (h >>> 0) % 12;
}

/** Zoom gain: a quarter at the zoom platforms appear, full three zooms closer. */
export function zoomGain(zoom: number): number {
  if (zoom < STATION_BOX_ZOOM) return 0;
  return 0.25 + 0.75 * Math.min(1, (zoom - STATION_BOX_ZOOM) / 3);
}

/** How many pops a count of riders gets: one for a few, more for many, capped. */
export function popsFor(riders: number): number {
  if (riders < 1) return 0;
  return Math.min(MAX_POPS, Math.round(1 + 0.7 * Math.log2(1 + riders / 8)));
}

export class StationSound {
  private ctx: AudioContext | null = null;
  private recent: number[] = [];

  play(a: Arrival) {
    if (a.speed >= 2 || a.volume <= 0) return;
    const o = audioOut();
    if (!o) return;
    const { ctx, out } = o;
    this.ctx = ctx;
    const now = ctx.currentTime;
    this.recent = this.recent.filter((t) => now - t < RECENT_S);
    if (this.recent.length >= MAX_RECENT) return;
    const level = (MASTER * a.volume * zoomGain(a.zoom)) / Math.sqrt(1 + this.recent.length);
    if (level <= 0.002) return;
    // a second arrival sounding together gets the note only
    const together = this.recent.length >= 1;
    this.recent.push(now);
    const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
    if (pan) {
      pan.pan.value = Math.max(-0.7, Math.min(0.7, a.pan * 0.7));
      pan.connect(out);
    }
    const dest: AudioNode = pan ?? out;
    const length = soundLength(Math.random());
    this.chime(dest, now, level, a.note, a.crowded, length === "short" ? 0.6 : 1);
    const cap = together ? 0 : a.speed === 1 || length !== "full" ? 2 : MAX_POPS;
    // pops follow the chime, getting off first, then getting on a moment later
    let offPops = Math.random() < OFF_SKIP ? 0 : Math.min(cap, a.off);
    let onPops = Math.min(cap, a.on), onAt = 0.75;
    if (length === "short") {
      if (offPops > 0 && (onPops === 0 || Math.random() < 0.5)) onPops = 0;
      else { offPops = 0; onAt = 0.3; }
    }
    const shift = 2 ** (((Math.random() * 2 - 1) * OFF_SHIFT_SEMITONES) / 12), offLevel = level * offGain(a.offRiders);
    for (let i = 0; i < offPops; i++) this.pop(dest, now + 0.28 + i * 0.09 + Math.random() * 0.05, offLevel, 640 * shift * 2 ** (((Math.random() * 2 - 1) * OFF_JITTER_SEMITONES) / 12));
    for (let i = 0; i < onPops; i++) this.pop(dest, now + onAt + i * 0.1 + Math.random() * 0.06, level, 360 + Math.random() * 140);
  }

  /** One soft sine note, the line's own. At a crowded platform it is a minor third lower, softer,
   * and sags a little as it fades. `len` scales how long it rings. */
  private chime(dest: AudioNode, t: number, level: number, note: number, crowded: boolean, len: number) {
    const ctx = this.ctx!;
    const f = C5 * 2 ** (note / 12) * (crowded ? CROWDED_RATIO : 1);
    const o = ctx.createOscillator(), g = ctx.createGain();
    const end = (crowded ? 0.9 : 0.7) * len;
    o.type = "sine";
    o.frequency.setValueAtTime(f, t);
    if (crowded) o.frequency.exponentialRampToValueAtTime(f * 0.96, t + end - 0.1);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime((crowded ? 0.22 : 0.32) * level, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + end);
    o.connect(g).connect(dest);
    o.start(t);
    o.stop(t + end + 0.05);
  }

  /** A short rounded blip whose pitch drops a little, like a bubble. */
  private pop(dest: AudioNode, t: number, level: number, f: number) {
    const ctx = this.ctx!;
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(f * 1.5, t);
    o.frequency.exponentialRampToValueAtTime(f, t + 0.035);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.24 * level, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.09);
    o.connect(g).connect(dest);
    o.start(t);
    o.stop(t + 0.1);
  }
}
