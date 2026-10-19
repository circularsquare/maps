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

export interface Arrival {
  /** -1 (left of the view) to 1 */
  pan: number;
  zoom: number;
  volume: number;
  /** the clock's speed index: 0 plays everything, 1 only a chime and a pop or two, 2 nothing */
  speed: number;
  off: number;
  on: number;
  /** the line's number: its note */
  line: number;
  /** the platform's riders waited long (the red face's test): a lower, softer note */
  crowded: boolean;
}

/** C major pentatonic, C5 to E6: any two lines' notes sound fine together. */
const NOTES = [523.25, 587.33, 659.25, 783.99, 880, 1046.5, 1174.66, 1318.51];
/** A minor third down. */
const CROWDED_RATIO = 0.84;

/** A line's note, fixed by its number. */
export function lineNote(line: number): number {
  let h = Math.imul(line + 1, 0x9e3779b1);
  h ^= h >>> 15;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  return NOTES[(h >>> 0) % NOTES.length];
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

  /** True once a click or key has made the context. */
  get ready() {
    return !!this.ctx;
  }

  play(a: Arrival): boolean {
    const ctx = this.ctx, out = this.out;
    if (!ctx || !out || a.speed >= 2 || a.volume <= 0) return false;
    if (ctx.state === "suspended") void ctx.resume();
    if (ctx.state !== "running") return false;
    const now = ctx.currentTime;
    this.recent = this.recent.filter((t) => now - t < RECENT_S);
    if (this.recent.length >= MAX_RECENT) return false;
    const level = (MASTER * a.volume * zoomGain(a.zoom)) / Math.sqrt(1 + this.recent.length);
    if (level <= 0.002) return false;
    // a second arrival sounding together gets the note only
    const together = this.recent.length >= 1;
    this.recent.push(now);
    const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
    if (pan) {
      pan.pan.value = Math.max(-0.7, Math.min(0.7, a.pan * 0.7));
      pan.connect(out);
    }
    const dest: AudioNode = pan ?? out;
    this.chime(dest, now, level, a.line, a.crowded);
    const cap = together ? 0 : a.speed === 1 ? 2 : MAX_POPS;
    // pops follow the chime, getting off first, then getting on a moment later
    for (let i = 0; i < Math.min(cap, a.off); i++) this.pop(dest, now + 0.28 + i * 0.09 + Math.random() * 0.05, level, 560 + Math.random() * 200);
    for (let i = 0; i < Math.min(cap, a.on); i++) this.pop(dest, now + 0.75 + i * 0.1 + Math.random() * 0.06, level, 360 + Math.random() * 140);
    return true;
  }

  /** One soft sine note, the line's own. At a crowded platform it is a minor third lower, softer,
   * and sags a little as it fades. */
  private chime(dest: AudioNode, t: number, level: number, line: number, crowded: boolean) {
    const ctx = this.ctx!;
    const f = lineNote(line) * (crowded ? CROWDED_RATIO : 1);
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
