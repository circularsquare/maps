// The sound at stations (T-686, T-696): one soft note, the line's own, when a train pulls in and
// small pops as riders get off and on, made with WebAudio (no files). Quiet by design: a low master level, the volume rising
// with the zoom past the one where platforms are drawn, and fewer sounds the more arrivals are
// already sounding. Browsers keep audio silent until the page has had a click or a key, so the
// context is made on the first one and nothing plays before it.

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
  private out: DynamicsCompressorNode | null = null;
  private recent: number[] = [];

  constructor() {
    const arm = () => {
      window.removeEventListener("pointerdown", arm, true);
      window.removeEventListener("keydown", arm, true);
      try {
        const C = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!C) return;
        this.ctx = new C();
        const comp = this.ctx.createDynamicsCompressor();
        comp.connect(this.ctx.destination);
        this.out = comp;
      } catch {
        this.ctx = null;
      }
    };
    window.addEventListener("pointerdown", arm, true);
    window.addEventListener("keydown", arm, true);
  }

  play(a: Arrival) {
    const ctx = this.ctx, out = this.out;
    if (!ctx || !out || a.speed >= 2 || a.volume <= 0) return;
    if (ctx.state === "suspended") void ctx.resume();
    if (ctx.state !== "running") return;
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
    this.chime(dest, now, level, a.note, a.crowded);
    const cap = together ? 0 : a.speed === 1 ? 2 : MAX_POPS;
    // pops follow the chime, getting off first, then getting on a moment later
    const offPops = Math.random() < OFF_SKIP ? 0 : Math.min(cap, a.off);
    const shift = 2 ** (((Math.random() * 2 - 1) * OFF_SHIFT_SEMITONES) / 12), offLevel = level * offGain(a.offRiders);
    for (let i = 0; i < offPops; i++) this.pop(dest, now + 0.28 + i * 0.09 + Math.random() * 0.05, offLevel, 640 * shift * 2 ** (((Math.random() * 2 - 1) * OFF_JITTER_SEMITONES) / 12));
    for (let i = 0; i < Math.min(cap, a.on); i++) this.pop(dest, now + 0.75 + i * 0.1 + Math.random() * 0.06, level, 360 + Math.random() * 140);
  }

  /** One soft sine note, the line's own. At a crowded platform it is a minor third lower, softer,
   * and sags a little as it fades. */
  private chime(dest: AudioNode, t: number, level: number, note: number, crowded: boolean) {
    const ctx = this.ctx!;
    const f = C5 * 2 ** (note / 12) * (crowded ? CROWDED_RATIO : 1);
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(f, t);
    if (crowded) o.frequency.exponentialRampToValueAtTime(f * 0.96, t + 0.8);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime((crowded ? 0.22 : 0.32) * level, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + (crowded ? 0.9 : 0.7));
    o.connect(g).connect(dest);
    o.start(t);
    o.stop(t + 0.95);
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
