// T-1195: a load's bar in stages. Each stage has the share of the bar its expected time gives it
// (what this browser measured on the slot's last load, else a model fitted to measured loads,
// notes/T-1195.md), and moves inside its share by what has landed (bytes, packs, solve rounds) or
// its time against that expectation, never past the share until the stage is done. Written by
// workers/demandClient.ts `jobProgress`; read through game/loading.ts `loadProgress`.

export type Stage = "game" | "data" | "demand";
export const STAGES: readonly Stage[] = ["game", "data", "demand"];
/** what the card says while each stage is the one in hand (T-1194) */
export const STAGE_TEXT: Record<Stage, string> = { game: "Loading", data: "Loading city data", demand: "Calculating demand" };
export type StageMs = Record<Stage, number>;

/** With nothing remembered, until the game's network reaches the demand pool: each stage's share, and
 * how long the game stage is expected to take (notes/T-1195.md: measured loads) */
export const GUESS_SHARES: StageMs = { game: 0.06, data: 0.24, demand: 0.7 };
export const GAME_MS = 4000;

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

/** T-1479: the time left as the card says it, rough on purpose */
export function leftText(ms: number): string {
  const s = ms / 1000;
  if (s < 10) return "A few seconds left";
  if (s < 55) return `${Math.round(s / 10) * 10} s left`;
  if (s < 90) return "1 min left";
  return `${Math.round(s / 60)} min left`;
}

/** T-1479: the time left shown, from `prev` (shown `dt` ms ago) toward the estimate `target`: down with
 * the clock or faster, up only at RISE_RATE a ms and only while the estimate is over a fifth above it,
 * so a late stage slows the count instead of throwing it back */
export const RISE_RATE = 0.5;
export function nextLeft(prev: number | null, target: number, dt: number): number {
  if (prev === null) return target;
  const counted = Math.max(0, prev - dt);
  if (target <= counted) return target;
  return target > counted * 1.2 ? counted + Math.min(target - counted, (1 + RISE_RATE) * dt) : counted;
}

/** a remembered wait still describes the game: its network has not changed by over a fifth */
export const sameSize = (rec: WaitRecord, stops: number | null) => stops === null || Math.abs(stops - rec.stops) <= 0.2 * Math.max(stops, rec.stops) + 50;
