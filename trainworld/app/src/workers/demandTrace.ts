// Opt-in diagnostics (?diagnostics=1, game/freezeTrace.ts, notes/T-4206.md): the demand side of an edit's wait.
// A sampler reads the pool's own bookkeeping four times a second and logs boundaries only (a pass sent, its local
// stage ended, shown, its whole solve began, its final, a zone's pack asked, downloaded, open, closed, a level change,
// a change of the "calculating trips" state). Times are epoch ms, the clock every recorder entry uses. The pool's
// code carries two one-line calls (startDemand, workerDied); nothing here changes what it does.
import { demandHeld, demandScope, demandStep, demandUpdating, demandView, editHeld, editsSettling } from "../game/demand";
import { setDemandSnapshot, trace } from "../game/freezeTrace";
import { loadProgress, zoneLoads, zoneSteps } from "../game/loading";

const epoch = (ms: number | undefined) => (ms === undefined ? undefined : Math.round(performance.timeOrigin + ms));
const MARKET = "market:";
let restartSince = false;

/** A one-off boundary the sampler cannot see (a worker restart and why). */
export function demandNote(event: string, detail: Record<string, unknown>) {
  if (event === "worker-restart") restartSince = true;
  trace("dm-" + event, { t: epoch(performance.now()), ...detail });
}

type Solve = Record<string, any>;
const MILESTONES: [string, string][] = [["firstAt", "dm-estimate"], ["earlyAt", "dm-early-lines"], ["localEndAt", "dm-local-end"],
  ["localShownAt", "dm-local-shown"], ["wholeAt", "dm-whole-start"], ["heldFrom", "dm-held"], ["refinedAt", "dm-final"]];

function why(s: Solve, afterRestart: boolean) {
  if (s.rough) return "rough solve of the zone in view while packs open";
  if (afterRestart) return "after a worker restart";
  if (s.barrier && s.newZones) return "zone opened";
  if (s.barrier) return "load or first figures";
  return "edit";
}
const brief = (s: Solve) => ({ version: s.version, kind: s.rough ? "rough" : s.barrier ? "whole" : "edit", zones: s.cities?.length, markets: s.markets?.length,
  arrived: epoch(s.arrivedAt), sent: epoch(s.sentAt), local: epoch(s.localEndAt), shown: epoch(s.localShownAt), whole: epoch(s.wholeAt), final: epoch(s.refinedAt),
  cutAfterMs: s.cutAfter, memMB: (s.memBytes ?? []).map((b: number) => Math.round(b / 1048576)) });

/** a share as whole tens of percent, so a bar's creep is not logged */
const tenths = (pct: number | null | undefined) => (pct === null || pct === undefined || !Number.isFinite(pct) ? null : Math.floor(pct / 10) * 10);

/** T-4359: the load in hand (`job`): its cities, the solve it waits for against the newest, whether its packs are all in,
 * whether it is a game's load, its stage and bar */
function loadOf(client: any) {
  const j = client.job, p = loadProgress.peek();
  if (!j) return null;
  return { cities: j.cities.slice(0, 8), n: j.cities.length, items: j.items.length, waitsFor: j.version ?? null, newest: client.cur?.version ?? null,
    packsIn: j.dataAt !== undefined, game: j.wait !== null, step: p?.step ?? null, pct: tenths(p ? p.fraction * 100 : null) };
}

/** T-4359: the status tag's words and bar, the loading card's, and the zones' bars (zone, bar, stage), as the player sees them */
function shownOf() {
  const tag = document.querySelector(".status-tag"), card = document.querySelector(".loading-card");
  const pct = (el: Element | null | undefined) => tenths(el ? Number(el.querySelector("[role=progressbar]")?.getAttribute("aria-valuenow")) : null);
  const steps = zoneSteps.peek();
  return { tag: tag?.querySelector(".demand-busy")?.textContent ?? null, tagPct: pct(tag), card: card?.querySelector(".big")?.textContent ?? null, cardPct: pct(card),
    zones: [...zoneLoads.peek()].slice(0, 6).map(([id, f]) => `${id} ${tenths(f * 100)} ${steps.get(id)?.step ?? ""}`) };
}

export function startDemandTrace(client: any) {
  const solves = new WeakMap<object, Set<string>>();
  const loads = new WeakMap<object, number>();
  let levels = new Map<string, number>();
  let waiting: unknown = null, state = "", load = "", tag = "";
  const sample = () => {
    const now = performance.now();
    const w = client.waiting;
    if (w && w !== waiting) {
      const need: Set<string> | null = client.solveNeed?.(w.svc) ?? null;
      trace("dm-waiting", { t: epoch(w.arrivedAt), missing: need ? [...need].filter((id) => !client.opened.has(id)).slice(0, 12) : null });
    }
    waiting = w;
    for (const s of client.log as Solve[]) {
      let done = solves.get(s);
      if (!done) {
        solves.set(s, done = new Set());
        trace("dm-pass", { t: epoch(s.sentAt), ...brief(s), why: why(s, restartSince), chainFrom: epoch(s.chainFrom), restartFrom: epoch(s.restartFrom) });
        restartSince = false;
      }
      for (const [key, name] of MILESTONES) if (s[key] !== undefined && !done.has(key)) {
        done.add(key);
        trace(name, { t: epoch(s[key]), version: s.version, ...(key === "refinedAt" ? { memMB: brief(s).memMB } : {}) });
      }
      if (s.cutAfter !== undefined && !done.has("cut")) { done.add("cut"); trace("dm-stopped", { t: epoch(now), version: s.version, ranMs: s.cutAfter }); }
    }
    let marketMask = [0, 0, 0, 0];
    for (const l of client.start.loads as { item: string; requestedAt: number; downloadedAt?: number; openedAt?: number; closedAt?: number; bare?: boolean }[]) {
      const had = loads.get(l) ?? 0;
      const has = 1 | (l.downloadedAt !== undefined ? 2 : 0) | (l.openedAt !== undefined ? 4 : 0) | (l.closedAt !== undefined ? 8 : 0);
      if (has === had) continue;
      loads.set(l, has);
      const at = [l.requestedAt, l.downloadedAt, l.openedAt, l.closedAt];
      for (let b = 0; b < 4; b++) if ((has & ~had) & (1 << b)) {
        if (l.item.startsWith(MARKET)) marketMask[b]++;
        else trace("dm-zone", { t: epoch(at[b]), zone: l.item, ev: ["asked", "downloaded", "open", "closed"][b], bare: l.bare ? 1 : undefined });
      }
    }
    if (marketMask.some(Boolean)) { const [asked, downloaded, open, closed] = marketMask; trace("dm-market-packs", { t: epoch(now), asked, downloaded, open, closed }); }
    const lv = client.levelPosted as Map<string, number>;
    for (const [zone, level] of lv) if (levels.get(zone) !== level) trace("dm-level", { t: epoch(now), zone, level: level === 0 ? "fine" : "coarse", from: levels.has(zone) ? (levels.get(zone) === 0 ? "fine" : "coarse") : undefined });
    levels = new Map(lv);
    const view = demandView.peek();
    const st = { calculating: demandUpdating.peek(), step: demandStep.peek().done + "/" + demandStep.peek().of, scope: demandScope.peek(),
      heldBehindFinal: !!editHeld.peek(), editsSettling: editsSettling.peek(), pausedHold: demandHeld.peek(), networkWaiting: !!w,
      workersReady: client.ready + "/" + client.workers.length, solving: client.computeN, replacing: client.respawn.size,
      shownVersion: view?.version, shownFinal: !!view?.final, newest: client.cur?.version };
    const text = JSON.stringify(st);
    if (text !== state) { state = text; trace("dm-state", { t: epoch(now), ...st }); }
    // T-4359: the load in hand (what the status tag's "Loading data" and the card describe) and what the tag and the zones'
    // bars show, their bars by tenths: a wait shown as a load is told apart from the calculation it waits on
    const l = loadOf(client);
    const lt = JSON.stringify(l);
    if (lt !== load) { load = lt; trace("dm-load", { t: epoch(now), ...(l ?? { none: true }) }); }
    const shown = shownOf();
    const sh = JSON.stringify(shown);
    if (sh !== tag) { tag = sh; trace("dm-tag", { t: epoch(now), ...shown }); }
  };
  // every network the clock sends, and whether it changed anything demand reads (a train count does not: no pass follows)
  const submit = client.submitNetwork?.bind(client);
  if (submit) client.submitNetwork = (snap: unknown) => {
    const key = client.lastKey, at = epoch(performance.now());
    const out = submit(snap);
    trace("dm-network", { t: at, changed: client.lastKey !== key });
    return out;
  };
  setInterval(() => { try { sample(); } catch { /* diagnostics must not affect play */ } }, 250);
  setDemandSnapshot(() => ({ at: epoch(performance.now()), state: JSON.parse(state || "null"), workers: client.memory?.(), solving: client.computeN,
    zonesOpen: [...client.opened].filter((i: string) => !i.startsWith(MARKET)).length, marketPacksOpen: [...client.opened].filter((i: string) => i.startsWith(MARKET)).length,
    levels: Object.fromEntries(client.levelPosted), recentPasses: (client.log as Solve[]).slice(-8).map(brief), load: loadOf(client), shown: shownOf() }));
}
