// The load card's pure logic: no text, so a Node check can import it (the words are in loadBar.ts).
// T-1195: a load's bar in stages. Each stage has the share of the bar its expected time gives it
// (what this browser measured on the slot's last load, else a model fitted to measured loads,
// notes/T-1195.md), and moves inside its share by what has landed (bytes, packs, solve rounds) or
// its time against that expectation, never past the share until the stage is done. Written by
// workers/demandClient.ts `jobProgress`; read through game/loading.ts `loadProgress`.

export type Stage = "game" | "data" | "demand";
export const STAGES: readonly Stage[] = ["game", "data", "demand"];
/** what the card says while each stage is the one in hand (T-1194) */
export type StageMs = Record<Stage, number>;

/** With nothing remembered, until the game's network reaches the demand pool: each stage's share, and
 * how long the game stage is expected to take (notes/T-1195.md: measured loads) */
export const GUESS_SHARES: StageMs = { game: 0.06, data: 0.24, demand: 0.7 };
export const GAME_MS = 4000;

/** T-2574: a cached result's showing once its network and packs are in (its packs' digests, its figures unpacked) */
export const CACHE_SHOW_MS = 1500;

/** a stage running late creeps toward this share of itself, never done until it is */
export const STEP_CAP = 0.98;
/** a share of `expected` ms gone since `since`: in step to 90%, then slowing toward STEP_CAP */
export function timeShare(since: number | undefined, expected: number, now: number): number {
  if (since === undefined) return 0;
  const x = Math.max(0, now - since) / Math.max(1, expected);
  return x < 0.9 ? x : 0.9 + (STEP_CAP - 0.9) * (1 - Math.exp(-(x - 0.9) / 0.3));
}

/** the bar's fill: each stage's share times how far it is (1 when done) */
export function stagedFraction(shares: StageMs, done: StageMs): number {
  let f = 0;
  for (const s of STAGES) f += shares[s] * Math.min(1, Math.max(0, done[s]));
  return f;
}

/** `ms` as shares summing to 1 (a stage expected to take nothing has none) */
export function sharesOf(ms: StageMs): StageMs {
  const total = STAGES.reduce((n, s) => n + Math.max(0, ms[s]), 0) || 1;
  return { game: Math.max(0, ms.game) / total, data: Math.max(0, ms.data) / total, demand: Math.max(0, ms.demand) / total };
}

/** the first stage not done: what the card names */
export function stageInHand(done: StageMs): Stage {
  return STAGES.find((s) => done[s] < 1) ?? "demand";
}

/** T-884, T-1195: how long each stage of a game's last load took in this browser, by slot (the last
 * 12), with the network's stops: each stage's time while it was the one in hand */
const LAST_WAIT = "anitabuilder-load-wait";
export interface WaitRecord { ms: number; stops: number; stages?: StageMs }
export function lastWait(key: string): WaitRecord | null {
  try {
    const r = JSON.parse(localStorage.getItem(LAST_WAIT) || "null")?.[key];
    return r && r.ms > 0 ? r : null;
  } catch { return null; }
}
export function rememberWait(key: string, rec: WaitRecord) {
  try {
    const all: Record<string, WaitRecord> = JSON.parse(localStorage.getItem(LAST_WAIT) || "null") ?? {};
    delete all[key];
    const keep = Object.entries(all).slice(-11);
    localStorage.setItem(LAST_WAIT, JSON.stringify(Object.fromEntries([...keep, [key, rec]])));
  } catch { /* storage unavailable */ }
}
/** T-1195: how this browser's last game load compared with the model, for a game it has not loaded
 * before: its data stage's time over the modelled one, and its game stage's time over GAME_MS */
const LOAD_PACE = "anitabuilder-load-pace";
/** T-1479: and `solve`, its solve's time over the solve model's; `later`, each crowding round's time
 * over the first round's (notes/T-1479.md: 0.26 on all-China, 0.6 on New York, 0.8 on Japan) */
export interface Pace { data: number; game: number; solve: number; later: number }
export const LATER_GUESS = 0.5;
export function loadPace(): Pace {
  try {
    const p = JSON.parse(localStorage.getItem(LOAD_PACE) || "null");
    if (p && p.data > 0 && p.game > 0) return { data: p.data, game: p.game, solve: p.solve > 0 ? p.solve : 1, later: p.later > 0 ? p.later : LATER_GUESS };
  } catch { /* storage unavailable */ }
  return { data: 1, game: 1, solve: 1, later: LATER_GUESS };
}
export function rememberPace(p: Pace) {
  const clamp = (x: number) => Math.min(10, Math.max(0.3, x));
  try { localStorage.setItem(LOAD_PACE, JSON.stringify({ data: clamp(p.data), game: clamp(p.game), solve: clamp(p.solve), later: Math.min(1.5, Math.max(0.1, p.later)) })); } catch { /* storage unavailable */ }
}

/** T-1479: the time left as the card says it, rough on purpose (T-1654: always a figure, never a count
 * that gives up) */
export const FEW_MS = 5000;

/** T-1654: the time left shown, from `prev` (shown `dt` ms ago) toward the estimate `target`. It counts
 * down with the clock, or drops to a lower estimate at once. It never climbs by degrees: an estimate
 * over JUMP_OVER times the count (and 10 s more) replaces it in one step, and under FEW_MS a higher
 * estimate of FEW_MS or more does too. A smaller excess slows the count to HOLD_RATE of the clock
 * until it meets the estimate. */
export const JUMP_OVER = 1.4;
export const HOLD_RATE = 0.5;
export function nextLeft(prev: number | null, target: number, dt: number): number {
  if (prev === null) return target;
  const counted = Math.max(0, prev - dt);
  if (target <= counted) return target;
  if (counted < FEW_MS && target >= FEW_MS) return target;
  if (target > counted * JUMP_OVER + 10_000) return target;
  return Math.min(target, Math.max(0, prev - HOLD_RATE * dt));
}

/** a round of W waves takes about W minus this times its first period in, until a round of the same
 * solve shows its own ratio (2.6 of 3 on Japan, 2.1 on New York, 2 workers each) */
const WAVE_SLACK = 0.4;
/** a late round with periods still out: each wave left takes this much of the average wave so far (the last wave holds fewer periods: 24% of a round against 38%) */
const LAST_WAVE = 0.7;
/** T-2552: a later round's first period in under this share of a wave's expected time is one that has converged (0.5 s
 * against 10 s on Anita's 81-zone save): it says nothing of the round's pace */
const QUICK_LAND = 0.5;
function waveRatio(roundAt: readonly number[] | undefined, sentAt: number, firstLand: readonly (number | undefined)[] | undefined, d: number, waves: number): number {
  const r = d - 1, end = roundAt?.[r], land = firstLand?.[r];
  // (the first round carries the setup: its ratio is not the later ones')
  if (r >= 1 && end !== undefined && land !== undefined) {
    const start = r > 0 ? roundAt?.[r - 1] ?? sentAt : sentAt;
    const q = (end - start) / (land - start);
    if (q >= 1 && q <= waves + 1) return q;
  }
  return Math.max(1, waves - WAVE_SLACK);
}
/**
 * T-793, T-1479, T-1654: the solve stage of the bar, through the rounds up to `doneRound`. The first round
 * (setup included) and each later one are timed apart, since their ratio runs from 0.26 to 0.9 by game
 * (notes/T-1479.md): before the first lands, `prior` is split by `later` (this browser's last ratio);
 * after it, the later rounds are expected at `later` times it; from the second on, at their own mean.
 * A round running past its expectation stretches the rounds after it as well. The round in hand shows
 * what has landed of it or the time against its expectation (`timeShare`: creeping, never done until
 * published). The periods of a round land in waves, as many as `waves` (a worker takes that many
 * periods in turn): once the first is in (`firstLand`, when each round's first period landed), the round
 * is expected at that time times the waves' ratio (`waveRatio`; the first round's `setup` ms counted once, and a
 * later round's converged first period, in under QUICK_LAND of a wave, not at all), and the rounds after it at the mean of
 * the later rounds so far with it; a round past even that is expected to take as long again per wave as it
 * has so far, and with nothing landed a fifth of its time so far more. `ms` is the stage's expected
 * length, `left` the time still expected, `first` and `rest` the first round's and each later one's ms as
 * measured or expected.
 */
export function solveStage(sentAt: number, roundAt: readonly number[] | undefined, published: number, landed: number, firstLand: readonly (number | undefined)[] | undefined, doneRound: number, prior: number, later: number, now: number, periods: number, waves: number, setup = 0): { ms: number; share: number; left: number; first: number; rest: number } {
  const d = Math.min(published + 1, doneRound + 1);
  const at = (r: number) => roundAt?.[r] ?? sentAt;
  let first = d > 0 && at(0) > sentAt ? at(0) - sentAt : prior / (1 + doneRound * later);
  let rest = d > 1 && at(d - 1) > at(0) ? (at(d - 1) - at(0)) / (d - 1) : later * first;
  const boundary = d > 0 ? at(d - 1) : sentAt;
  if (d > doneRound) return { ms: boundary - sentAt, share: 1, left: 0, first, rest };
  let exp = d === 0 ? first : rest;
  const firstLandAt = firstLand?.[d];
  const waved = firstLandAt !== undefined && firstLandAt > boundary && landed > 0 && landed < periods && (d === 0 || firstLandAt - boundary >= QUICK_LAND * exp / waves);
  // (T-2552: the first round's setup, before any period, is once a round, not once a wave)
  const once = d === 0 && waved && setup < firstLandAt - boundary ? setup : 0;
  if (waved) exp = once + (firstLandAt - boundary - once) * waveRatio(roundAt, sentAt, firstLand, d, waves);
  const el = Math.max(0, now - boundary), part = Math.min(STEP_CAP, landed / periods);
  let curLeft = Math.max(exp - el, 0.2 * el * (1 - part));
  if (el > exp && landed > 0 && landed < periods && waves > 1) {
    const done = Math.min(waves - 1, Math.ceil(landed / Math.ceil(periods / waves)));
    curLeft = Math.max(curLeft, LAST_WAVE * el * (waves - done) / done);
  }
  const curMs = el + curLeft;
  if (d === 0) { first = waved ? curMs : Math.max(first, curMs); rest = waved ? later * curMs : Math.max(rest, later * curMs); } else rest = waved ? (at(d - 1) - at(0) + curMs) / d : Math.max(rest, curMs);
  const after = doneRound - d;
  const ms = boundary - sentAt + curMs + after * rest;
  const cur = Math.max(part, timeShare(boundary, exp, now));
  const doneMs = boundary - sentAt;
  return { ms, share: (doneMs + exp * cur) / Math.max(1, doneMs + exp + after * rest), left: curLeft + after * rest, first, rest };
}

/** The bar shown for the model's `fraction`, never going back: when the fraction falls (the shares change as the
 * network arrives, an estimate grows) the bar keeps its place and the rest of the work is spread over the rest of it.
 * `bar` holds the best shown so far and that place's anchor, and is updated. */
export interface BarPlace { best: number; anchor?: { shown: number; frac: number } }
export function keepPlace(bar: BarPlace, fraction: number): number {
  const a = bar.anchor;
  let shown = a ? a.shown + (1 - a.shown) * Math.max(0, fraction - a.frac) / Math.max(1e-6, 1 - a.frac) : fraction;
  // (T-2552: a fall below the place's own fraction moves the place down to it, so the bar moves again with the
  // next work done rather than when the fraction is back where it was)
  if (shown < bar.best - 1e-6 || (a && fraction < a.frac)) { bar.anchor = { shown: bar.best, frac: fraction }; shown = bar.best; }
  bar.best = Math.max(bar.best, Math.min(0.99, shown));
  return bar.best;
}

/** T-343, T-2552: the bar of a zone loading outside the load in hand: its download, its opening and the solve that
 * covers it (none for a zone opened bare), each by its share done and weighted by its expected ms */
export function zoneShare(p: { dlMs: number; dl: number; openMs: number; open: number; solveMs: number; solve: number }): number {
  const total = p.dlMs + p.openMs + p.solveMs;
  return total > 0 ? (p.dlMs * p.dl + p.openMs * p.open + p.solveMs * p.solve) / total : 1;
}

/** a remembered wait still describes the game: its network has not changed by over a fifth */
export const sameSize = (rec: WaitRecord, stops: number | null) => stops === null || Math.abs(stops - rec.stops) <= 0.2 * Math.max(stops, rec.stops) + 50;
