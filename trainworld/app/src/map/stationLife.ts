// Life at stations (T-686): from the platforms' zoom, a train pulling in on a platform in view
// makes a chime, pops as riders get off and on (map/stationSound.ts) and small faces rising from
// the platform (map/faces.ts). Below that zoom, or with both switched off, nothing runs: no timer,
// no canvas.
//
// An arrival is a train whose stop offset passed a station's between two samples, taken eight a
// second. Only the lines stopping at a station in view are looked at, so a view over one complex
// costs a few trains' arithmetic. The riders come from the boarding account the map is drawn from:
// who gets off is what the train carried in, plus who got on, less what it carries on
// (game/boarding.ts); who got on is the platform's crowd when the train came, plus who arrive
// until it leaves, less who it leaves behind.
//
// A face is sad when the train before left CROWDED_LEFT or more riders behind, else tired when the
// platform's riders have waited TIRED_WAIT_S or more on average (boardedAt, T-802).

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { pulsesBetween, QUEUE_STRIDE, stoppedAt, tripLoad, waitingAt, type BoardHour } from "../game/boarding";
import { clock } from "../game/clock";
import { display } from "../game/ui";
import type { LineInput, StationInput } from "../game/types";
import { world } from "../game/world";
import { Faces, type Mood } from "./faces";
import { profileS } from "./network";
import type { Overlay } from "./overlay";
import { STATION_BOX_ZOOM } from "./stationGeometry";
import { lineNote, popsFor, StationSound } from "./stationSound";
import { stationZone } from "../game/listScope";
import { moneyState } from "../game/money";
import { zonesLoaded } from "../game/zones";

/** Samples a second. */
const SAMPLE_MS = 125;
/** A longer gap than this between samples (game seconds: a jump in time, a tab that slept) is not
 * read as trains arriving. */
const MAX_GAP_S = 150;
/** Riders who have waited this long or longer on average make tired faces: ten minutes, orange on
 * the crowding dot's scale. */
export const TIRED_WAIT_S = 600;
/** The train before left this many riders or more: a crowded platform, sad faces. */
export const CROWDED_LEFT = 20;
/** One face per this many riders getting off. */
const RIDERS_PER_FACE = 20;
const MAX_FACES_PER_TRAIN = 4;

interface VisLine {
  line: LineInput;
  num: number;
  /** per run, the stops in view in that run's order: its index and station */
  runs: [{ k: number; s: StationInput }[], { k: number; s: StationInput }[]];
}

const queueIndex = new WeakMap<BoardHour, Map<string, number>>();
function queueOf(b: BoardHour, node: number, line: number, run: number): number {
  let m = queueIndex.get(b);
  if (!m) {
    m = new Map();
    for (let q = 0; q < b.queueCount; q++) {
      const o = q * QUEUE_STRIDE;
      m.set(`${b.queues[o]}|${b.queues[o + 1]}|${b.queues[o + 2]}`, q);
    }
    queueIndex.set(b, m);
  }
  return m.get(`${node}|${line}|${run}`) ?? -1;
}

/** Riders who got on at queue `q` when a train arrived at `t` (seconds from the hour's epoch), the
 * average time the riders waiting had waited, s, and how many the train before left behind. The
 * newcomers arrived evenly since that train left (half the gap each); those it left behind add the
 * gap before it, again half on average. Nothing is known of the hour's first arrival: 0. */
export function boardedAt(b: BoardHour, q: number, t: number): { boarded: number; waited: number; left: number } {
  const Q = b.queues, E = b.events, o = q * QUEUE_STRIDE;
  const first = Q[o + 8], end = first + Q[o + 9];
  let e = first;
  // events are departures; the one a train arriving at `t` makes comes a little after it
  while (e < end && E[e * 2] < t - 8) e++;
  // the crowd just before it stopped (T-1123: it boards as the train stops)
  const stop = e < end ? Math.min(t, stoppedAt(b, e)) : t;
  const waiting = waitingAt(b, q, stop - 1e-3);
  const boarded = e < end ? Math.max(0, waiting + Q[o + 4] * Math.max(0, E[e * 2] - stop) + pulsesBetween(b, q, stop, E[e * 2]) - E[e * 2 + 1]) : 0;
  const p = e - 1;
  if (p < first) return { boarded, waited: 0, left: 0 };
  const gap = Math.max(0, stop - E[p * 2]), left = E[p * 2 + 1];
  const before = p - 1 >= first ? E[p * 2] - E[(p - 1) * 2] : gap;
  const fresh = Math.max(0, waiting - left), riders = left + fresh;
  const waited = riders > 0 ? (left * (gap + before / 2) + (fresh * gap) / 2) / riders : 0;
  return { boarded, waited, left };
}

/** Each line's note (T-706): chosen once and kept in the save (the clock worker's `lineNotes`, the
 * header's `notes`). A line's city is where its first stop is (as the Lines tab has it); its rank
 * is how many lines were given a note in that city before it, deleted ones too, so removing a line
 * changes nobody else's note. Lines found without a note (a new line, or an older save) are given
 * theirs here, in line number order, and sent to the worker. A line with no stops yet, or in no
 * city, keeps the note its number gives and none is stored. */
let posted: Record<string, [number, string]> = {};
let postNotes: ((notes: Record<string, [number, string]>) => void) | null = null;
let noted: { save: unknown; stored: unknown; zones: number; by: Map<string, number> } | null = null;
function lineNotes(save: { stations: StationInput[]; lines: LineInput[] }): Map<string, number> {
  const stored = moneyState.peek()?.notes, zones = zonesLoaded.peek();
  if (noted && noted.save === save && noted.stored === stored && noted.zones === zones) return noted.by;
  // a note the worker has sent back is no longer ours to remember
  if (stored) for (const id of Object.keys(posted)) if (stored[id]) delete posted[id];
  const all: Record<string, [number, string]> = { ...stored, ...posted };
  const at = new Map(save.stations.map((s) => [s.id, s]));
  const fresh: Record<string, [number, string]> = {};
  const missing = save.lines.filter((l) => !all[l.id]).sort((a, b) => Number(a.id) - Number(b.id));
  if (missing.length) {
    const rank = new Map<string, number>();
    for (const [, [, city]] of Object.entries(all)) rank.set(city, (rank.get(city) ?? 0) + 1);
    for (const l of missing) {
      const s = at.get(l.stops[0]), city = s ? stationZone(s) : null;
      if (!city) continue;
      const r = rank.get(city) ?? 0;
      rank.set(city, r + 1);
      fresh[l.id] = all[l.id] = [lineNote(Number(l.id), r), city];
    }
    if (Object.keys(fresh).length) {
      posted = { ...posted, ...fresh };
      postNotes?.(fresh);
    }
  }
  const by = new Map<string, number>();
  for (const l of save.lines) by.set(l.id, all[l.id]?.[0] ?? lineNote(Number(l.id), 99));
  noted = { save, stored, zones, by };
  return by;
}

export function startStationLife(map: MlMap, overlay: Overlay, post: (notes: Record<string, [number, string]>) => void) {
  postNotes = post;
  const faces = new Faces(map, overlay);
  const sound = new StationSound();
  let timer = 0;
  let prevT = NaN;
  /** the time spent sampling, for ?debug=1 */
  const stats = { n: 0, ms: 0, max: 0 };

  let sampledAt = 0;
  const sample = () => {
    const t0 = performance.now();
    // T-1207: half as often while the camera moves; arrivals since are still caught, a little later
    if (overlay.moving && t0 - sampledAt < 2 * SAMPLE_MS - 10) return;
    sampledAt = t0;
    read();
    const ms = performance.now() - t0;
    stats.n++;
    stats.ms += ms;
    if (ms > stats.max) stats.max = ms;
  };

  const read = () => {
    const d = display.peek();
    const r = overlay.renderer, net = r.net, w = world.peek();
    // T-826: no faces or sounds for a hidden page, whose clock may run on
    if (!net || !w || !r.tripCount || !clock.ticking || clock.isHidden) { prevT = NaN; return; }
    const now = clock.now(), prev = prevT;
    prevT = now;
    if (!(now - prev > 0) || now - prev > MAX_GAP_S) return;

    const bounds = map.getBounds();
    const west = bounds.getWest(), east = bounds.getEast(), south = bounds.getSouth(), north = bounds.getNorth();
    const inView = new Map<string, StationInput>();
    for (const s of w.save.stations) {
      if (s.built && s.lng >= west && s.lng <= east && s.lat >= south && s.lat <= north) inView.set(s.id, s);
    }
    if (!inView.size) return;
    const lines = new Map<string, VisLine>();
    for (const l of w.save.lines) {
      if (!l.stops.some((id) => inView.has(id))) continue;
      const run0: { k: number; s: StationInput }[] = [], run1: { k: number; s: StationInput }[] = [];
      l.stops.forEach((id, k) => {
        const s = inView.get(id);
        if (!s) return;
        run0.push({ k, s });
        run1.push({ k: l.stops.length - 1 - k, s });
      });
      lines.set(l.id, { line: l, num: Number(l.id), runs: [run0, run1] });
    }
    if (!lines.size) return;

    const trips = r.trips, epoch = r.tripsEpoch, board = r.board;
    const speed = clock.speed.peek();
    for (let i = 0; i < r.tripCount; i++) {
      const profile = trips[i * 2];
      const vl = lines.get(String(Math.floor(profile / 6)));
      if (!vl) continue;
      const dep = trips[i * 2 + 1] + epoch, tau = now - dep;
      if (tau < 0) continue;
      const run = Math.floor(profile / 3) % 2;
      const S = w.lineStats[vl.line.id]?.stopS[run];
      if (!S) continue;
      const sNow = profileS(net, profile, tau), sPrev = profileS(net, profile, Math.max(0, prev - dep));
      if (sNow <= sPrev) continue;
      for (const { k, s } of vl.runs[run]) {
        // the train's offset passes the stop's (the first stop is where the trip starts: not an arrival)
        const at = S[k];
        if (at === undefined || at <= 2 || sPrev >= at - 2 || sNow < at - 2) continue;
        arrive(i, vl, run, k, s, now - epoch, board, d, speed);
      }
    }
  };

  const arrive = (i: number, vl: VisLine, run: number, k: number, s: StationInput, tEpoch: number, board: BoardHour | null, d: ReturnType<typeof display.peek>, speed: number) => {
    let off = 0, on = 0, mood: Mood = "happy";
    if (board) {
      const n = vl.line.stops.length;
      const stretch = (seg: number) => (run === 0 ? seg : n - 2 - seg);
      const before = k > 0 ? tripLoad(board, i, stretch(k - 1)) : 0;
      const after = k < n - 1 ? tripLoad(board, i, stretch(k)) : 0;
      const q = queueOf(board, s.num, vl.num, run);
      if (before !== null && after !== null) {
        const b = q >= 0 ? boardedAt(board, q, tEpoch) : { boarded: 0, waited: 0, left: 0 };
        on = b.boarded;
        off = Math.max(0, before + on - after);
        mood = b.left >= CROWDED_LEFT ? "sad" : b.waited >= TIRED_WAIT_S ? "tired" : "happy";
      }
    }
    if (d.sound) {
      const p = map.project([s.lng, s.lat]);
      sound.play({ pan: (p.x / Math.max(1, overlay.cssW)) * 2 - 1, zoom: map.getZoom(), volume: d.soundVolume, speed, off: popsFor(off), on: popsFor(on), offRiders: off, note: lineNotes(world.peek()!.save).get(vl.line.id) ?? 0, crowded: mood !== "happy" });
    }
    if (d.faces && off >= 1) {
      const count = Math.min(MAX_FACES_PER_TRAIN, Math.floor(off / RIDERS_PER_FACE + Math.random()));
      if (count > 0) faces.spawn(s.id, s.lng, s.lat, s.length, s.heading, count, mood);
    }
  };

  const sync = () => {
    const d = display.peek();
    const on = map.getZoom() >= STATION_BOX_ZOOM && (d.sound || d.faces);
    if (on && !timer) {
      prevT = NaN;
      timer = window.setInterval(sample, SAMPLE_MS);
    } else if (!on && timer) {
      window.clearInterval(timer);
      timer = 0;
      faces.clear();
    }
    if (!d.faces && faces.alive) faces.clear();
  };
  map.on("zoom", sync);
  effect(() => {
    display.value;
    sync();
  });
  return { faces, sound, sample, stats };
}
