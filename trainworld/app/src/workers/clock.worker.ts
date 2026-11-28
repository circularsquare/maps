// The clock worker (SPEC 6.5, 7; T-025): owns the network (the track model's `TrackApi`, T-040),
// applies edits, runs profiles and the capacity pass, and sends the main thread what it draws:
// track polylines, keyframes (shared phase tables) and each game hour's trips (the hourly
// re-base, SPEC 7). After a service change it also builds the network snapshot demand runs on,
// debounced 0.5 s (SPEC 6.5). notes/T-025.md.
//
// Every edit answers with an "edited" message and, if it applied, a new "state".

import init, { set_heap_step, TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { initWasm } from "./wasmInit";
import { anchor, anchorShift, makeAnchor, setAnchor, toLocal } from "../map/geo";
import { CITIES, cityWithin, isCity, NEW_GAME_FRAME, WORLD_ID, zonesTouching } from "../game/cities";
import { blocksTouching } from "../game/terrainBlocks";
import { frameKindName, frameKindNumber, M_PER_DEG } from "../game/frame";
import { PERIODS } from "../game/periods";
import { loanFromSave, loanToSave } from "../game/loan";
import { shapesFrom } from "../game/lineShape";
import { SHADES } from "../game/shadeTable";
import { foldLedger } from "./ledgerFold";
import { shiftAt } from "../game/localTime";
import { strengths } from "../game/moneyTiming";
import { setBandCheck, setBandMode } from "../map/lineBands";
import { setNetCheck } from "../map/network";
import { placeOutlines, readOutlinesForPlaces } from "../map/coarseOutlines";
import { canvasReadsTrue } from "../map/canvasRead";
import { UPGRADE, UPGRADE_BITS } from "../game/upgrades";
import { saveNetKey } from "./demandCache";
import { currentZones, decodeSave, encodeSave, encodeSaveParts, refuseSaveBuffers, isThisWorld, isRetiredTrackFormat, isNewerError, SAVE_LEVEL, SAVE_NEWER, trackFrameKind, readStored, failReads, StorageReadError, SAVE_FORMAT, START_CASH, NEW_LINE_CARS, START_CLOCK, writeStored, listSlots, leaseSlot, ORIGINAL_SLOT, SAVE_DAMAGED, NO_CITIES, isDamagedError, unreadableKey, setNetworkCheck, markDamaged, rescueGood, BACKUP_STEPS, newGameName, slotName, backupSlot, takePending, sweepDeleted, WHOLE_PANEL, type NewGameSpec, type SlotSummary, type SaveHeader, cleanNewHeadways } from "./saveFile";
import { lineProfile } from "./lineProfile";
import type { EditOp, RouteAsk, ToClock } from "./protocol";
import { stampAll } from "./cloud";
import { isMemoryFailure, loadFailKey } from "../game/loadText";
import { autosave, bindSlots, CLOUD_RETRY_MS, cloudFlush, cloudKick, cloudUp, heldFrom, heldNew, hold, intoNewSlot, intoView, leave, letGoSlot, manageSlot, NO_MEMORY_TO_SAVE, noteUnsaved, pendingGame, publishSlots, saveConverted, saves } from "./slots";
import { rowsFirst } from "./clockWordLines";
import { bookCharge, buildSnapshot, busQuoteLL, complexOfNodes, editSteps, nameOnly, putPreviewAt, putXferSet, sameFloats, scheduleSnapshot, sendTrips, snapStations, stationsKey, stepBoarding, turnHour, xferSet } from "./clockDemand";
import { apply, buildingQuoteAnswer, designOf, doubleUpAnswer, DROP_REFUSED, freeEnds, INSERT_REFUSED, issues, newLineShape, ORDER_REFUSED, paidFor, pisArr, rebuildQuoteAnswer, routeArgs, savedHidden, savedShapes } from "./clockEdits";
import { goneLineFares, loadFares, newLineFare, pinFareRef, startFares, tidyLineFares, tidyStationFares } from "./clockFares";
import { EARLY_WAIT_MS, followTimer, followUp, goesAhead, guardedFollowUp, heldEdges, HOLD_MS, holdOn, isPreview, isServiceOp, jumpsAhead, keepUntilState, later, passesFollowUp, passesHeld, previewStaleSeq, putPreviewStaleSeq, sendStale, settleOwedNow, stale, standAside, startNow, stepThenRun, takeForced, waitsFollowUp, Waiting, whenStateWent } from "./clockFollowUp";
import { putLoadMaskWait } from "./clockGround";
import { blocksIn, blueprintPolylines, ensureBlocks, ensureMasks, ensureStreets, fetchWater, groundForBlueprint, masksComplete, masksIn, nodePoints, openNetwork, pumpBlocks, putPackBase, putTerrainBase, putViewBlocks, putViewZones, retryGroundNow, roadFeed, roadsUnder, routePoints, terrainIndexIntoModel, tintSent, VIEW_MARGIN_M, VIEW_SPAN_M, viewBlocks, viewZones, wakeStale, WaterRaster, withMasks, worldWaterIntoModel, zonesOfPoints } from "./clockGround";
import { anchorMoves, drawnEdge, drawnLine, edgeDrawn, forget, forgetDrawn, game, hex, previewMarkRows, putAnchorMoves, putConvertedOnLoad, putShiftFrom, putShiftWas, putView, restart, rowsCanGo, sendHome, sendOpened, sendRows, sendState, SHIFT_REACH, shiftFrom, SHIFTS_EXACT } from "./clockMain";
import { bookTrains, gameNow, lineDays, linesPeriods, loadLineDays, loadRegions, mergeLedger, putTodayGen, putTodayLast, regionEntries, regions, rounded, savedLineDays, savedRegions, sendCash, sendMoney, serviceLeadS, settle, today, todayGen, unsettleAfter } from "./clockMoney";
import { afterLoadPanic, downTold, EDIT_UNDONE, kept, putDownTold, isPanic, keep, MODEL_DOWN, MODEL_RESTORED, modelDown, newModel, openFailReason, panicAfterEdit, panicLater, panicNow, panicOnLoad, panicWhy, putModelDown, putPanicAfterEdit, putPanicOnLoad, recover, SAVE_PANIC } from "./clockPanic";
import { wordAfter, wordBefore } from "./clockWord";
import { mapPatch, patchMiss } from "./clockPatch";
import { routeEnded, runEnds } from "./clockRunEnds";
import { nameArrived, nameOwed, nameOwedStations, placedNameless, withName } from "./clockNames";
import { diagramOn, patchMode, postDiagram, posted, postWhole, putDiagramOn, putDiagramWhole, putPatchMode, sender } from "./clockPost";
import { achievements, api, applyRunFactors, AUTOSAVE_MS, city, clk, copyOf, dayPinned, dayShift, demandBuilt, demandFine, diagramCities, emptyTrack, epoch, failInPlace, fares, frameSpec, freezeCheckpoint, freezePort, freezeTimer, goals, guide, guideWanted, heldCharges, home, hourLedger, initIncoming, ledger, ledgerDay, lineIncome, lineNotes, linePoints, lineRides, lineShapes, loadHolidays, loan, loanOp, messages, modded, mode, newLineCars, newLineHeadway, newLineWeekendShare, opened, outlinesGot, outlinesIn, outlinesRead, packLngLat, packLocal, pageModded, peakC, placeLineCountries, pricesOf, putAchievements, putClk, putCopyOf, putDemandBuilt, putDemandFine, putDiagramCities, putDiagramInside, putEmptyTrack, putEpoch, putFares, putFreezeMemory, putFreezePort, putFreezeTimer, putGoals, putGuide, putGuideWanted, putHiddenLines, putHome, putHourLedger, putIncomePerHour, putInitIncoming, putLedger, putLineIncome, putLineNotes, putLinePoints, putLineRides, putLineShapes, putLineShares, putLoan, putMessages, putMidLoad, putModded, putMoneyFrozen, putNewLineCars, putNewLineHeadway, putNewLineWeekendShare, putOpened, putOutlinesRead, putPageModded, putPrices, putRidesPerHour, putRowsSent, putStars, putTiming, putTrackVersion, putUpkeepRegions, send, setMode, setStartZone, stars, startZone, syncHolidays, timing, trackVersion, withModded } from "./clockState";

// T-3734: the save slot (slots.ts) and what it needs of the game in hand
bindSlots({ api: () => api, game: () => game, send, gameNow, saveBytes, slotSummary, roomFor, outOfRoom, loadGame, inQueue: (f) => (putMessages(messages.then(f).catch((err) => send({ kind: "error", message: String(err) }))), messages), isPanic, panicLater, openFailed: failInPlace });

addEventListener("error", (e) => {
  if (!isPanic(e.error)) return;
  e.preventDefault();
  panicLater(e.error, "a timer"); // i18n-ignore
});
addEventListener("unhandledrejection", (e) => {
  if (!isPanic(e.reason)) return;
  e.preventDefault();
  panicLater(e.reason, "background work"); // i18n-ignore
});

// ---------------------------------------------------------------- saves (T-029)

/** `m`: the network and counts kept after the last edit, for a model that has stopped (a download still works) */
function header(m: typeof kept = null): SaveHeader {
  const fleetBus = m ? m.fleetBus : api!.fleet_bus(), fleetFerry = m ? m.fleetFerry : api!.fleet_ferry(), fleetFerrySmall = m ? m.fleetFerrySmall : api!.fleet_ferry_small(), fleetFerryLarge = m ? m.fleetFerryLarge : api!.fleet_ferry_large(), fleetTram = m ? m.fleetTram : api!.fleet_tram();
  return { game: "anitabuilder", format: SAVE_FORMAT, level: TrackApi.save_level(), city, region: WORLD_ID, cities: CITIES.map(c => c.id), clock: gameNow(), cash: m?.cash ?? api!.cash(), ...loanToSave(loan), fleet: m?.fleet ?? api!.fleet(), fleetFast: m?.fleetFast ?? api!.fleet_fast(), ...(fleetBus ? { fleetBus } : {}), ...(fleetFerry ? { fleetFerry } : {}), ...(fleetFerrySmall ? { fleetFerrySmall } : {}), ...(fleetFerryLarge ? { fleetFerryLarge } : {}), ...(fleetTram ? { fleetTram } : {}), fares: { ...fares }, ledger: ledger.map(rounded), hours: hourLedger.map(rounded), regions: savedRegions(rounded), ...(lineDays.size ? { lineDays: savedLineDays(true) } : {}), stars: Object.fromEntries(stars), goals: Object.fromEntries(goals), notes: Object.fromEntries(lineNotes), ...savedShapes(), ...savedHidden(), ...(diagramCities.length ? { schematic: [...diagramCities] } : {}), mode, start: startZone, ...(dayPinned ? { dayShift } : {}), opened: opened && [...opened], ...(demandFine ? { demandFine } : {}), ...(demandBuilt?.length ? { demandBuilt } : {}), newCars: newLineCars, ...(Object.keys(newLineHeadway).length ? { newHeadway: newLineHeadway } : {}), ...(newLineWeekendShare !== null ? { newWeekendShare: newLineWeekendShare } : {}), ...(guide ? { guide } : {}), ...(achievements ? { achievements: Object.fromEntries(achievements) } : {}), ...(home ? { home } : {}), ...(copyOf ? { copy: copyOf } : {}), ...(modded ? { modded } : {}) };
}

/** The game as file bytes. The network and header are taken at once, before anything awaits. */
function saveBytes(): Promise<Uint8Array> {
  settle();
  settleOwedNow();
  return encodeSave(header(), api!.save());
}
/** T-3869: the same in parts of at most 64 KB, never joined (a download) */
function saveParts(): Uint8Array[] {
  if (modelDown) {
    if (!kept) throw new Error("no network was kept");
    return encodeSaveParts(header(kept), kept.net);
  }
  settle();
  settleOwedNow();
  return encodeSaveParts(header(), api!.save());
}

// T-3869: a save that finds no memory. A browser short of memory refuses new buffers while the rest of the game runs
// on, so making the save and writing it are tried again: after a moment (other threads' garbage collected), then just
// after the page has replaced its demand workers, which gives back most of a big game's memory until the
// replacements have reopened what they held.
/** a failure for want of memory, not a panic (which the model's recovery handles) */
function outOfRoom(err: unknown): boolean { return !isPanic(err) && isMemoryFailure(String((err as Error | null)?.message ?? err)); }
/** the page answered "room": its demand workers were replaced (checks' refused buffers are given while it holds) */
let roomMade = false;
/** the saves waiting on the replaced workers, and the page's answer they wait for */
let roomUsers = 0;
let roomAsked: Promise<void> | null = null;
let roomAnswered: (() => void) | null = null;
/** when the demand workers were last replaced for a save (wall ms): an autosave has them replaced at most once in
 * ROOM_GAP_MS, so a game always short of memory is not solved again at every autosave */
let roomAt = -Infinity;
const ROOM_GAP_MS = 10 * 60_000;
/** the wait before the second try, ms */
const ROOM_WAIT_MS = 1000;
/** the waits before each try after the workers were replaced, ms (an old worker's memory goes back as its thread
 * ends; the replacements take it again as they reopen) */
const ROOM_TRIES_MS = [250, 500, 1000, 2000];
const pause = (ms: number) => new Promise<void>((go) => setTimeout(go, ms));
/** `make` run, and tried again as above while it fails for want of memory; `always`: a save the player asked for (or
 * a download), which has the workers replaced however recently they were. Throws the last failure. */
async function roomFor<T>(make: () => Promise<T>, always: boolean): Promise<T> {
  let last: unknown;
  const attempt = async (): Promise<{ v: T } | null> => {
    try { return { v: await make() }; } catch (err) { if (!outOfRoom(err)) throw err; last = err; return null; }
  };
  let got = await attempt();
  if (got) return got.v;
  await pause(ROOM_WAIT_MS);
  if ((got = await attempt())) return got.v;
  if (!always && Date.now() - roomAt < ROOM_GAP_MS) throw last;
  roomAt = Date.now();
  if (roomUsers++ === 0) {
    console.warn("T-3869: a save found no memory; the demand workers are replaced:", String(last)); // i18n-ignore
    roomAsked = new Promise<void>((go) => { roomAnswered = go; setTimeout(go, 15_000); });
    send({ kind: "room" });
  }
  try {
    await roomAsked;
    for (const ms of ROOM_TRIES_MS) {
      await pause(ms);
      if ((got = await attempt())) return got.v;
    }
    throw last;
  } finally {
    if (--roomUsers === 0) {
      roomMade = false;
      roomAsked = roomAnswered = null;
    }
  }
}
function roomIsMade() {
  roomMade = true;
  roomAnswered?.();
}

// T-251: drawn at twice the size the saves list shows it (240 x 160, T-894), for sharp lines on high-density screens
const THUMB_W = 480, THUMB_H = 320;
/** T-894: one panel per city, the most stations first, at most four; fractions of the picture */
const THUMB_LAYOUTS: [number, number, number, number][][] = [
  [[0, 0, 1, 1]],
  [[0, 0, 0.5, 1], [0.5, 0, 0.5, 1]],
  [[0, 0, 0.6, 1], [0.6, 0, 0.4, 0.5], [0.6, 0.5, 0.4, 0.5]],
  [[0, 0, 0.5, 0.5], [0.5, 0, 0.5, 0.5], [0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
];
/** canvas px: between panels; a panel's margin, with room at the top for the HTML name over it */
const THUMB_GAP = 4, THUMB_PAD = 14, THUMB_TOP = 34;
/** the theme the page is in (main.ts sends it): a thumbnail is drawn on that theme's land (T-543) */
let thumbShade: "light" | "dark" = "light";
/** T-3869: the game's last card, for a save whose card found no memory */
let lastSummary: { game: number; summary: SlotSummary } | null = null;
/** T-241: the saves list's picture of the network and its figures. Read at once, before anything
 * awaits (call it straight after saveBytes); a failure leaves the picture out, never the save. T-3869: a card that
 * finds no memory is the game's last one. */
async function slotSummary(): Promise<SlotSummary> {
  const g = game;
  try {
    const summary = await drawSummary();
    lastSummary = { game: g, summary };
    return summary;
  } catch (err) {
    if (!outOfRoom(err) || lastSummary?.game !== g) throw err;
    return lastSummary.summary;
  }
}
async function drawSummary(): Promise<SlotSummary> {
  // a save never waits long on them: without them a station takes the nearest centre's name
  await Promise.race([outlinesRead, new Promise((go) => setTimeout(go, 3000))]);
  const a = api!;
  const ni = a.nodes_info();
  const byCity = new Map<string, number>();
  const stationXY: { city: string | null; x: number; y: number }[] = [];
  for (let i = 0; i < ni.length; i += 10) {
    if (ni[i + 7] <= 0) continue;
    const [lng, lat] = packLngLat(ni[i + 1], ni[i + 2]);
    const c = cityWithin(lng, lat); // T-318: a station over dead space is in no city
    if (c) byCity.set(c, (byCity.get(c) ?? 0) + 1);
    const [x, y] = toLocal(lng, lat);
    stationXY.push({ city: c, x, y });
  }
  const stations = stationXY.length;
  const cities = [...byCity].sort((x, y) => y[1] - x[1]).map(([c]) => c);
  const li = a.lines_info();
  const summary: SlotSummary = { cities, lines: li.length / 22, stations, cash: a.cash(), mode, ...(copyOf ? { copy: true } : {}) };
  try {
    // T-3705: a browser that makes up a canvas's pixels gets no picture rather than a made-up one
    if (typeof OffscreenCanvas === "undefined" || !canvasReadsTrue()) return summary;
    // track (grey) and each line's route (its colour), in local units: what the renderer draws
    const track: ArrayLike<number>[] = [];
    const ei = a.edges_info();
    for (let i = 0; i < ei.length; i += 10) {
      const id = ei[i];
      const d = drawnEdge(a, id);
      const xy = new Float32Array((d.length / 3) * 2);
      for (let k = 0, j = 0; k < d.length; k += 3, j += 2) { xy[j] = d[k]; xy[j + 1] = d[k + 1]; }
      track.push(xy);
    }
    const routes: { colour: string; xy: ArrayLike<number> }[] = [];
    for (let i = 0; i < li.length; i += 22) {
      if (li[i + 2] !== 1) continue;
      const id = li[i];
      const d = drawnLine(a, id);
      routes.push({ colour: hex(li[i + 1]), xy: d });
    }
    type Box = [number, number, number, number];
    const boxOf = (pts: Iterable<[number, number]>): Box => {
      const b: Box = [Infinity, Infinity, -Infinity, -Infinity];
      for (const [x, y] of pts) { b[0] = Math.min(b[0], x); b[1] = Math.min(b[1], y); b[2] = Math.max(b[2], x); b[3] = Math.max(b[3], y); }
      return b;
    };
    function* points(xy: ArrayLike<number>): Generator<[number, number]> { for (let k = 0; k < xy.length; k += 2) yield [xy[k], xy[k + 1]]; }
    const drawn = [...track.map((xy) => ({ xy, colour: "" })), ...routes].map((p) => ({ ...p, box: boxOf(points(p.xy)) })).filter((p) => p.box[2] >= p.box[0]);
    const all = boxOf(drawn.flatMap((p) => [[p.box[0], p.box[1]], [p.box[2], p.box[3]]] as [number, number][]));
    if (!(all[2] >= all[0])) return summary;
    // T-894: the biggest cities by stations, each framed on its own stations; a city with under 8%
    // of the biggest's stations is a fringe of its network, not a panel. A network with no station
    // in a city is one panel over all of it, unnamed. T-2394: with more than 3 cities to show, the
    // last panel is everything built, named, in place of the fourth city.
    const most = cities.length ? byCity.get(cities[0])! : 0;
    const big = cities.filter((c) => byCity.get(c)! >= 0.08 * most);
    const whole = big.length > 3;
    const shown = big.slice(0, THUMB_LAYOUTS.length - (whole ? 1 : 0));
    const frames = shown.length
      ? shown.map((c) => ({ city: c, stations: byCity.get(c)!, box: boxOf(stationXY.filter((p) => p.city === c).map((p) => [p.x, p.y] as [number, number])) }))
      : [{ city: "", stations, box: all }];
    if (whole) frames.push({ city: WHOLE_PANEL, stations, box: all });
    const layout = THUMB_LAYOUTS[frames.length - 1];
    const sh = SHADES[thumbShade];
    const canvas = new OffscreenCanvas(THUMB_W, THUMB_H);
    const ctx = canvas.getContext("2d")!;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.fillStyle = sh.casing;
    ctx.fillRect(0, 0, THUMB_W, THUMB_H);
    summary.shade = thumbShade;
    summary.panels = [];
    frames.forEach((f, n) => {
      const [fx, fy, fw, fh] = layout[n];
      // panel in canvas px, the gap split between neighbours
      const px0 = fx * THUMB_W + (fx > 0 ? THUMB_GAP / 2 : 0), py0 = fy * THUMB_H + (fy > 0 ? THUMB_GAP / 2 : 0);
      const px1 = (fx + fw) * THUMB_W - (fx + fw < 1 ? THUMB_GAP / 2 : 0), py1 = (fy + fh) * THUMB_H - (fy + fh < 1 ? THUMB_GAP / 2 : 0);
      const pw = px1 - px0, ph = py1 - py0;
      summary.panels!.push({ city: f.city, stations: f.stations, x: px0 / THUMB_W, y: py0 / THUMB_H, w: pw / THUMB_W, h: ph / THUMB_H });
      // the frame: the city's stations inside the margins, at least 3 km across so a short stretch is not blown up
      const top = f.city ? THUMB_TOP : THUMB_PAD;
      const iw = pw - 2 * THUMB_PAD, ih = ph - top - THUMB_PAD;
      const [x0, y0, x1, y1] = f.box;
      const s = iw / Math.max(x1 - x0, (y1 - y0) * iw / ih, 3000);
      const ox = px0 + THUMB_PAD + iw / 2 - ((x0 + x1) / 2) * s, oy = py0 + top + ih / 2 - ((y0 + y1) / 2) * s;
      // what lies inside the panel, in local units
      const v: Box = [(px0 - ox) / s, (py0 - oy) / s, (px1 - ox) / s, (py1 - oy) / s];
      const inside = (b: Box) => b[2] >= v[0] && b[0] <= v[2] && b[3] >= v[1] && b[1] <= v[3];
      ctx.save();
      ctx.beginPath();
      ctx.rect(px0, py0, pw, ph);
      ctx.clip();
      ctx.fillStyle = sh.land;
      ctx.fillRect(px0, py0, pw, ph);
      const k = Math.max(0.8, Math.sqrt((pw * ph) / (THUMB_W * THUMB_H)));
      for (const p of drawn) {
        if (!inside(p.box)) continue;
        ctx.strokeStyle = p.colour || sh.thumbTrack;
        ctx.lineWidth = (p.colour ? 6 : 3) * k;
        ctx.beginPath();
        let lx = NaN, ly = NaN;
        for (let j = 0; j < p.xy.length; j += 2) {
          const qx = ox + p.xy[j] * s, qy = oy + p.xy[j + 1] * s;
          if (j > 0 && j < p.xy.length - 2 && Math.hypot(qx - lx, qy - ly) < 0.8) continue;
          if (j === 0) ctx.moveTo(qx, qy); else ctx.lineTo(qx, qy);
          lx = qx; ly = qy;
        }
        ctx.stroke();
      }
      ctx.restore();
    });
    const png = new Uint8Array(await (await canvas.convertToBlob({ type: "image/png" })).arrayBuffer());
    let bin = "";
    for (let k = 0; k < png.length; k += 0x8000) bin += String.fromCharCode(...png.subarray(k, k + 0x8000));
    summary.thumb = "data:image/png;base64," + btoa(bin);
  } catch { /* no picture */ }
  return summary;
}

/** Replace the game with a save's. Throws a player-readable reason, changing nothing. A save whose
 * track is not in the world frame is converted once, through longitude and latitude, and loaded in
 * it (T-308); the return value says so, and the caller then writes the slot again in the new
 * format. Money, fares and the ledger load as saved, but for anything settled after the clock
 * (`unsettleAfter`). With `fresh`, the save is a real network a new game starts from (T-731): only
 * its network and fleet are kept, the rest is a new game's (fares too, T-1906). */
async function loadGame(bytes: Uint8Array, from: "autosave" | "file" | "new", fresh?: NewGameSpec): Promise<boolean> {
  putMidLoad(true);
  try {
    return await openSave(bytes, from, fresh);
  } finally {
    putMidLoad(false);
  }
}
async function openSave(bytes: Uint8Array, from: "autosave" | "file" | "new", fresh?: NewGameSpec): Promise<boolean> {
  putTodayGen(todayGen + 1);
  putTodayLast(null);
  putRowsSent(""); // a load's service starts with no line calendars (`TrackApi.set_line_rows`)
  send({ kind: "loadStep", step: "decoding the save" }); // i18n-ignore (T-4072)
  const decoded = await decodeSave(bytes);
  const { track } = decoded;
  const h: SaveHeader = currentZones(fresh
    ? { ...decoded.header, clock: startClock(fresh), cash: START_CASH, loan: undefined, loanPay: undefined, loanDays: undefined, ledger: [], hours: [], regions: undefined, lineDays: undefined, stars: undefined, goals: undefined, notes: undefined, newCars: undefined, newHeadway: undefined, newWeekendShare: undefined, guide: undefined, achievements: undefined, home: undefined, ...newGameHeader(fresh) }
    : decoded.header);
  if (!isThisWorld(h)) throw new Error("This save is for another world.");
  if (isRetiredTrackFormat(track)) throw new Error("This save is from an older version of the game and no longer opens.");
  // T-4014: a newer build's blocks or flag bits (`decodeSave` has refused its level, header and magic already); before
  // any conversion or write, which would keep only what this build read
  if (TrackApi.save_unread(track) > 0) throw new Error(SAVE_NEWER);
  let net = track, converted = false;
  if (trackFrameKind(track) !== NEW_GAME_FRAME.kind) {
    const c = TrackApi.convert_save_frame(track, frameKindNumber(NEW_GAME_FRAME.kind));
    // an unreadable network converts to nothing: let the load below say so
    if (c.length) { net = c; converted = true; }
  }
  // T-889: the demand pool opens the zones the built stations reach while the network is worked out
  const st = TrackApi.save_stations(net);
  putDemandFine(h.demandFine);
  putDemandBuilt(h.demandBuilt);
  if (st.length > 3) send({ kind: "stationsAhead", frame: { kind: frameKindName(st[0]), lon: st[1], lat: st[2] }, xy: st.slice(3), fine: demandFine ?? null, built: demandBuilt ?? null, home: h.home });
  // T-1539: the demand result kept for this save's network can show while the network is worked out
  // (China's 17-42 s); awaited here, since the load below holds the worker
  if (st.length > 3) {
    send({ kind: "loadStep", step: "kept trips' key" }); // i18n-ignore
    await saveNetKey(net, h).then((key) => send({ kind: "netAhead", key }), () => {});
  }
  // T-3647: a player's game saved before 0.4 keeps the way its single track ran then
  if (import.meta.env.DEV && panicOnLoad) { putPanicOnLoad(false); panicNow(); }
  await openNetwork(net, fresh ? undefined : decoded.header.level ?? 0);
  // T-3297: a real network started on the other side than its save keeps to
  if (fresh?.rightHand !== undefined && fresh.rightHand !== api!.right_hand()) api!.flip_right_hand();
  api!.set_fleet(h.fleet ?? 0);
  api!.set_fleet_fast(h.fleetFast ?? 0);
  api!.set_fleet_bus(h.fleetBus ?? 0);
  api!.set_fleet_ferry(h.fleetFerry ?? 0);
  api!.set_fleet_ferry_small(h.fleetFerrySmall ?? 0);
  api!.set_fleet_ferry_large(h.fleetFerryLarge ?? 0);
  api!.set_fleet_tram(h.fleetTram ?? 0);
  putFares(fresh ? startFares(fresh) : loadFares(h.fares));
  goneLineFares.clear();
  tidyStationFares(api!);
  setStartZone(h.start, h.dayShift); // before the ledgers, whose days follow its local time
  putLedger(mergeLedger(h.ledger, (d) => d.day));
  putHourLedger(mergeLedger(h.hours, (d) => d.hour));
  loadRegions(h.regions);
  // T-3757: a save from before the fold shrinks here and at its next save
  foldLedger(ledger, ledgerDay(h.clock));
  for (const r of regions.values()) foldLedger(r.days, ledgerDay(h.clock));
  loadLineDays(h.lineDays, ledgerDay(h.clock));
  putStars(new Map(Object.entries(h.stars ?? {})));
  putGoals(new Map(Object.entries(h.goals ?? {})));
  putAchievements(h.achievements ? new Map(Object.entries(h.achievements)) : null);
  putLineNotes(new Map(Object.entries(h.notes ?? {})));
  putLineShapes(shapesFrom(h.shapes));
  putHiddenLines(new Set(Array.isArray(h.hidden) ? h.hidden.filter((x) => typeof x === "string") : []));
  putDiagramCities(Array.isArray(h.schematic) ? h.schematic.filter((x) => typeof x === "string" && (x === NO_CITIES || isCity(x))) : []);
  putDiagramInside({ stations: new Set(), stops: new Set() });
  putNewLineCars(h.newCars ?? NEW_LINE_CARS);
  putNewLineWeekendShare(typeof h.newWeekendShare === "number" && Number.isFinite(h.newWeekendShare) ? h.newWeekendShare : null);
  putNewLineHeadway(cleanNewHeadways(h.newHeadway));
  putGuide(h.mode === "sandbox" ? undefined : h.guide);
  putHome(h.home);
  putCopyOf(!fresh && typeof h.copy === "string" && h.copy ? h.copy : undefined);
  putModded(withModded(h.modded));
  api!.set_cash(h.cash + unsettleAfter(h.clock));
  setMode(h.mode === "sandbox" ? "sandbox" : "normal");
  putLoan(mode === "sandbox" ? null : loanFromSave(h));
  putOpened(h.opened);
  putConvertedOnLoad(converted);
  restart(h.clock, from);
  return converted;
}

/** A new game's clock (T-728): the first time from day 1 07:00 on the game clock at which it is
 * 07:00 in the start city, so every start opens on its own morning peak (Tokyo: Tue 2 07:00 there). */
function startClock(spec?: NewGameSpec): number {
  const c = spec?.start ? CITIES.find((x) => x.id === spec.start) : undefined;
  const ahead = c ? shiftAt(c.center[0], c.center[1]) : 0;
  return START_CLOCK + (((-ahead % 86400) + 86400) % 86400);
}

/** A new game's mode, start and opened zones (T-728). */
function newGameHeader(spec?: NewGameSpec): Pick<SaveHeader, "mode" | "start" | "opened" | "demandFine" | "demandBuilt"> {
  return { mode: spec?.mode ?? "normal", start: spec?.start, opened: spec?.start ? [spec.start] : [], demandFine: undefined, demandBuilt: undefined };
}

function setNewGame(spec?: NewGameSpec) {
  const g = newGameHeader(spec);
  setMode(g.mode!);
  setStartZone(g.start);
  putOpened(g.opened);
  putDemandFine(undefined);
  putDemandBuilt(undefined);
  putGuide(guideWanted && g.mode === "normal" && !saves.testPage ? { done: 0, ...(g.start ? {} : { zoom: true as const }) } : undefined);
  putHome(undefined);
  putCopyOf(undefined);
  putModded(withModded(undefined));
  putFares(startFares(spec));
  goneLineFares.clear();
}

/** A new game: empty, or from `spec.real`'s network (T-731). */
async function newGame(spec?: NewGameSpec) {
  if (spec?.real) {
    await loadGame(new Uint8Array(spec.real), "new", spec);
    return;
  }
  setNewGame(spec);
  putLoan(null);
  api!.load(emptyTrack!);
  api!.set_right_hand(spec?.rightHand ?? true);
  api!.set_cash(START_CASH);
  api!.set_fleet(0);
  api!.set_fleet_fast(0);
  api!.set_fleet_bus(0);
  api!.set_fleet_ferry(0);
  api!.set_fleet_tram(0);
  putLedger([]);
  putHourLedger([]);
  loadRegions(undefined);
  loadLineDays(undefined, 0);
  putStars(new Map());
  putGoals(new Map());
  putAchievements(new Map());
  putLineNotes(new Map());
  putLineShapes(new Map());
  putHiddenLines(new Set());
  putDiagramCities([]);
  putDiagramInside({ stations: new Set(), stops: new Set() });
  putNewLineCars(NEW_LINE_CARS);
  putNewLineHeadway({});
  putNewLineWeekendShare(null);
  restart(startClock(spec), "new");
}

// ---------------------------------------------------------------- messages

let ready: Promise<void> | null = null;

function viewMasks(): Promise<void> {
  if (api && viewBlocks.size) void ensureBlocks(viewBlocks);
  return api && viewZones.length ? ensureMasks(viewZones).then(() => {}) : Promise.resolve();
}

/** T-3196: above 0 while the queue waits on a download with the track model at rest (an edit's or a
 * ghost's water masks, Construct's ground): a new hour's trips are sent at once meanwhile, or every
 * train would stop until the download ends */
let parked = 0;
async function parkedOn<T>(p: Promise<T>): Promise<T> {
  parked++;
  try { return await p; } finally { parked--; }
}
/** the new hour's trips, out of the queue's turn */
function tripsNow(e: number) {
  if (!api || modelDown || e === epoch) return;
  putEpoch(e);
  try { turnHour(); } catch (err) {
    if (isPanic(err)) panicLater(err, "the hour's trips"); // i18n-ignore
    else send({ kind: "error", message: String(err) });
  }
}
/** T-4145: an accepted edit's word with nothing to draw (the money-only, link, building and loan edits), so the page's
 * entry for it is armed and goes with the next state, as every other edit's; `noNet` (T-4214): no network follows (a
 * fare or loan edit), so no map click waits for one */
const plainWord = (seq: number, noNet: boolean) => send({ kind: "applied", seq, word: {}, ...(noNet ? { noNet: true as const } : {}) });
/** T-4145: the newest edit refused since the last one accepted, -1 for none: an undo asked while it was unanswered is
 * refused, so the step under it is not taken back instead */
let refusedLast = -1;
/** T-4114: the rows state stop clicks owe, a task later through the queue: clicks sent meanwhile are applied first and
 * share it, and the newest click's own takes its place */
let rowsGen = 0;
function rowsSoon() {
  const gen = ++rowsGen;
  setTimeout(() => putMessages(messages.then(() => { if (gen === rowsGen) rowsOwed(); }).catch((err) => send({ kind: "error", message: String(err) }))), 0);
}
/** the rows for the edits of the follow-up still waiting for a world holding them, then their answers (none can go while
 * the drawn pieces are to move: the full state answers them) */
function rowsOwed() {
  const f = followUp;
  if (!api || modelDown || !f?.waiting.some((w) => w.owes())) return;
  // T-4339: these rows answer edits, so they hold what answered edits owe (as `standAside`'s do)
  settleOwedNow();
  if (!sendRows()) return;
  for (const w of f.waiting) w.worldSent();
}
// T-4272: demand's results kept while a follow-up waited go into the queue, in order, once none waits; one from a game no
// longer in hand is dropped (the message that ran the follow-up may open another, whose restart clears riders)
// (T-4281: a question kept so is answered empty, as one the model cannot answer is)
whenStateWent((msgs) => { for (const k of msgs) putMessages(messages.then(() => k.game === game ? processMessage(k.m) : failAnswer(k.m, [])).then(pumpBlocks).catch((err) => send({ kind: "error", message: String(err) }))); });
onmessage = (e: MessageEvent<ToClock>) => {
  const msg = e.data;
  if (msg.kind === "freezeTrace") {
    freezePort?.close(); putFreezePort(msg.port);
    if (freezeTimer === undefined) putFreezeTimer(setInterval(() => freezeCheckpoint("pulse"), 1000));
    freezeCheckpoint("pulse");
    return;
  }
  if (msg.kind === "epoch" && parked > 0) return tripsNow(msg.epoch);
  if (msg.kind === "retryGround") return retryGroundNow();
  // T-3478: leaving never waits on a download: not on a game still opening (unless it is a picked file,
  // which would be lost), nor on an edit parked for its ground, whose network is at rest
  if (msg.kind === "leave" && (parked > 0 || (ready && !saves.initDone && !initIncoming))) return void leave(msg, !saves.initDone);
  if (msg.kind === "view") {
    putView(msg.w, msg.s, msg.e, msg.n);
    // outside the queue, so the fetch starts at once; a mask is installed between messages
    const midLat = (msg.s + msg.n) / 2, wide = (msg.e - msg.w) * M_PER_DEG * Math.cos(midLat * Math.PI / 180) > VIEW_SPAN_M;
    const cx = (msg.w + msg.e) / 2;
    const skip = new Set(msg.skip);
    putViewZones((wide ? zonesTouching(cx, midLat, cx, midLat, VIEW_MARGIN_M) : zonesTouching(msg.w, msg.s, msg.e, msg.n, VIEW_MARGIN_M)).map((c) => c.id).filter((id) => !skip.has(id)));
    putViewBlocks(new Set(wide ? blocksTouching(cx, midLat, cx, midLat, VIEW_MARGIN_M) : blocksTouching(msg.w, msg.s, msg.e, msg.n, VIEW_MARGIN_M)));
    // the view's blocks are the newest wanted, so the ones the camera left go first
    for (const k of viewBlocks) {
      if (blocksIn.delete(k)) blocksIn.add(k);
      if (tintSent.delete(k)) tintSent.add(k);
    }
    for (const id of viewZones) if (!masksIn.has(id)) void fetchWater(id).catch(() => {});
    void viewMasks();
    return;
  }
  if (msg.kind === "shade") {
    thumbShade = msg.shade;
    return;
  }
  if (msg.kind === "roomMade") return roomIsMade(); // T-3869: a save in the queue waits on it
  if (msg.kind === "roadTiles") {
    roadFeed.setTemplate(msg.template);
    if (api) void roadsUnder(...blueprintPolylines());
    return;
  }
  if (msg.kind === "account") {
    const before = saves.cloudCtx;
    saves.cloudCtx = msg.ctx;
    if (msg.ctx?.on && (msg.settle || !before?.on || before.acct !== msg.ctx.acct)) void cloudKick();
    return;
  }
  // T-2802: at once, not behind init's queue, which a save's summary can wait on
  if (msg.kind === "outlines") return outlinesIn(msg.packed && placeOutlines(msg.packed));
  // T-3707: at once, so a game loading now is marked too; the queue below marks the game in hand
  if (msg.kind === "moneyFrozen") return putMoneyFrozen(true); // T-3851
  if (msg.kind === "modded") putPageModded([...new Set([...pageModded, ...msg.why.filter((x) => typeof x === "string")])].slice(0, 16));
  if (msg.kind === "previewStale") {
    putPreviewStaleSeq(Math.max(previewStaleSeq, msg.seq));
    wakeStale?.();
    return;
  }
  if (isPreview(msg)) putPreviewAt(performance.now());
  // T-4214: a station's name, taken by its edit if that still waits in the queue
  if (msg.kind === "stationName") nameArrived(msg.ticket, msg.name, msg.game);
  putMessages(messages.then(() => processMessage(msg)).then(pumpBlocks).catch(err => send({ kind: "error", message: String(err) })));
};
async function processMessage(msg: ToClock) {
  if (msg.kind === "init") {
    if (msg.patch) putPatchMode(msg.patch);
    setBandCheck(patchMode === "check");
    setNetCheck(patchMode === "check");
    if (msg.bandMode !== undefined) setBandMode(msg.bandMode);
    putPackBase(msg.pack);
    putTerrainBase(msg.terrain);
    putOutlinesRead(outlinesGot.then((p) => {
      if (p) readOutlinesForPlaces(p);
      if (!api) return;
      placeLineCountries();
      pinFareRef();
      if (syncHolidays()) sendTrips();
    }));
    // T-3722: a leave that comes before the pending game is read must not skip it (a picked file would be lost)
    putInitIncoming(!!msg.pending);
    ready = (async () => {
      const t0 = performance.now();
      putFreezeMemory((await initWasm(() => init(), "clock worker")).memory); // i18n-ignore
      if (msg.heapStep) set_heap_step(msg.heapStep);
      // T-4014: a track model built from other sources than this page (a stale WASM build) would write saves the page
      // refuses as newer, or label newer bytes as its own
      if (TrackApi.save_level() !== SAVE_LEVEL) console.error(`T-4014: the track model's save level ${TrackApi.save_level()} is not the page's ${SAVE_LEVEL}; rebuild the WASM`); // i18n-ignore: console text
      // T-3395: the save check here also decodes the network (TWT4-6; anything else is left to the load); T-4014: bytes
      // its decoder leaves over are a newer build's, as the load finds them
      setNetworkCheck((track) => !/^TWT[456]$/.test(new TextDecoder().decode(track.subarray(0, 4))) ? "intact"
        : TrackApi.save_stations(track).length < 3 ? "damaged" : TrackApi.save_unread(track) > 0 ? "other" : "intact");
      const model = newModel();
      loadHolidays();
      putEmptyTrack(model.save());
      putPrices(pricesOf(model.money_params()));
      void viewMasks(); // the view sent before the track model was up
      void worldWaterIntoModel();
      void terrainIndexIntoModel();
      send({ kind: "ready", initMs: performance.now() - t0, frame: frameSpec() });
      // The last game, unless the page asked for a new one (T-029).
      saves.testPage = !!(msg.fresh || msg.testPage);
      putGuideWanted(msg.guide === true);
      if (msg.backupMs && msg.backupMs > 0) saves.backupSteps = BACKUP_STEPS.map((s) => s / BACKUP_STEPS[0] * msg.backupMs!);
      if (msg.readFails) failReads(msg.readFails);
      if (msg.maskWaitMs !== undefined) putLoadMaskWait(msg.maskWaitMs);
      if (import.meta.env.DEV && msg.loadPanic === true) putPanicOnLoad(true);
      if (import.meta.env.DEV && msg.saveRoom) { const all = msg.saveRoom === "all"; refuseSaveBuffers((n) => !roomMade && (all || n >= 1 << 20)); }
      // T-1196: the game picked in the saves panel before the page reloaded; a file or backup goes
      // into a slot of its own below, the slot it was picked from held until then (one that fails goes to the start menu, T-3773)
      send({ kind: "loadStep", step: "saves in the browser" }); // i18n-ignore (T-4072)
      const pending = msg.pending ? await takePending() : null;
      await sweepDeleted(); // T-1827's leftovers, if any
      let incoming: Awaited<ReturnType<typeof pendingGame>> = null;
      if (pending?.kind === "new") {
        msg.newSlot = pending.name || newGameName(await listSlots().catch(() => []), pending.spec);
        msg.game = pending.spec;
      } else if (pending) {
        incoming = await pendingGame(pending);
        msg.slot = pending.from;
      } else if (msg.pending && msg.game) {
        // T-3478: a new game whose spec is gone from storage is still a new game, never the game left for it
        msg.newSlot = newGameName(await listSlots().catch(() => []), msg.game);
      }
      putInitIncoming(!!incoming);
      if (msg.fresh) {
        // T-320: a new game for a check owns no slot, so nothing it does can reach a saved game
        saves.activeSlot = "";
        saves.activeLease = null;
      } else if (msg.newSlot) {
        saves.activeSlot = "slot-" + crypto.randomUUID();
        saves.newSlotName = { key: saves.activeSlot, name: msg.newSlot };
        hold(await leaseSlot(saves.activeSlot));
      } else {
        saves.activeSlot = msg.slot?.startsWith("slot-") ? msg.slot : ORIGINAL_SLOT;
        if (saves.activeSlot !== ORIGINAL_SLOT && !await readStored(saves.activeSlot)) saves.activeSlot = ORIGINAL_SLOT;
        // a stress page reads the slot but does not take it, so it never writes it
        hold(msg.testPage ? null : await leaseSlot(saves.activeSlot));
      }
      await publishSlots();
      if (incoming) {
        const g = incoming;
        try {
          if (g.view) await intoView(g.name, () => loadGame(g.bytes, "file"));
          else await intoNewSlot(g.name, () => loadGame(g.bytes, "file"), true, g.version);
          return;
        } catch (err) {
          // T-3773: to the start menu, the game the player came from left as it is in storage
          const panicked = isPanic(err) ? afterLoadPanic(err) : null;
          letGoSlot();
          const from = g.view ? "view" : pending?.kind === "restore" ? "restore" : "file";
          send({ kind: "openFailed", from, slot: pending?.kind === "restore" ? backupSlot(pending.key) : undefined, name: g.name, ...openFailReason(err, panicked, (err as Error).message) });
          return;
        }
      }
      if (!msg.fresh && !msg.newSlot) await stampAll(null).catch(() => {}); // T-2037
      send({ kind: "loadStep", step: "reading the save" }); // i18n-ignore
      const stored = msg.fresh || msg.newSlot ? null : await readStored(saves.activeSlot);
      if (!stored && saves.activeSlot) saves.unwritten = saves.activeSlot;
      if (stored) heldFrom(stored); else heldNew();
      if (msg.fresh || msg.newSlot) {
        // T-728, T-731: the start menu's choice; a real network is written to its slot at once
        if (msg.game?.real) {
          try {
            await newGame(msg.game);
            saves.dirty = true;
            void autosave();
            return;
          } catch (err) {
            if (msg.newSlot) {
              // T-3788: a start from the menu that fails leaves no empty slot behind: back to the start menu, as an open that failed
              const panicked = isPanic(err) ? afterLoadPanic(err) : null;
              letGoSlot();
              send({ kind: "openFailed", from: "new", name: msg.newSlot, ...openFailReason(err, panicked, loadFailKey(String(err))) });
              if (panicked === null) console.error("real network did not load:", err);
              return;
            }
            if (isPanic(err)) send({ kind: "loaded", from: "new", clock: START_CLOCK, error: SAVE_PANIC, why: afterLoadPanic(err), frame: frameSpec(), game });
            else send({ kind: "loaded", from: "new", clock: START_CLOCK, error: (console.error("real network did not load:", err), loadFailKey(String(err))), frame: frameSpec(), game });
          }
        }
        const empty = msg.game && { ...msg.game, real: undefined };
        // a start on another clock than New York's starts again at its own morning
        if (startClock(empty) !== START_CLOCK) { await newGame(empty); return; }
        setNewGame(empty);
        api!.set_right_hand(empty?.rightHand ?? true);
      }
      if (stored) {
        try {
          if (await loadGame(stored.bytes, "autosave") && saves.activeLease) await saveConverted(saves.activeSlot, stored);
          return;
        } catch (err) {
          // T-2820: first, so a panic's stopped model is out of the way of everything below
          const panicked = isPanic(err) ? afterLoadPanic(err) : null;
          const key = saves.activeSlot;
          // Keep the unreadable save aside (T-743), its slot as it is; T-3948: a newer version's save is not
          // unreadable, only not readable here, so it stays in its slot alone
          if (!isNewerError(err)) await writeStored(unreadableKey(key, stored.at), stored).catch(() => {});
          // T-3395: a damaged slot is marked so (never uploaded) and its last good copy found
          const damaged = isDamagedError(err);
          if (damaged && saves.activeLease) { await markDamaged(key, stored.at).catch(() => {}); await rescueGood(key).catch(() => {}); }
          // T-3773: no game of its own takes over (an empty one read as the save lost): the page goes to the start
          // menu, holding no slot meanwhile
          letGoSlot();
          send({ kind: "openFailed", from: "slot", slot: key, name: slotName(key, stored), ...openFailReason(err, panicked, damaged ? SAVE_DAMAGED : loadFailKey(String(err))) });
          if (panicked === null && !isNewerError(err)) console.error("last game did not load:", err);
          return;
        }
      }
      keep();
      sendOpened();
      sendHome();
      sendState();
      send({ kind: "snapshot", snapshot: buildSnapshot() });
    })().catch((err) => {
      // T-2703: a start that failed (a slot that could not be read among the causes) writes no slot
      letGoSlot();
      const storage = err instanceof StorageReadError;
      // T-2820: after a panic nothing more is asked of the model, and the error is the first one
      if (isPanic(err)) putModelDown(true);
      send({ kind: "failed", error: storage ? err.message : isPanic(err) ? panicWhy(err) : String(err), storage });
    });
    // T-988: the game in hand claimed on the account, the other slots settled
    void ready.then(() => { saves.initDone = true; if (import.meta.env.DEV && msg.loadPanic === "late") putPanicOnLoad(true); void cloudKick(); });
    addEventListener("online", () => { void cloudKick(); if (saves.cloudRetryAt) void cloudUp(saves.activeSlot); });
    setInterval(tick, 1000);
    setInterval(liveMoney, LIVE_MONEY_MS);
    return;
  }
  await ready;
  // T-2404: the track tool asks for the held state now: its pieces start (T-3803; a later early edit still joins it)
  if (msg.kind === "release" && followUp && api && !modelDown) {
    // (its hold over, so a ghost asked next does not hold it again)
    if (followUp.held) followUp.held.until = performance.now();
    if (!followUp.job) { clearTimeout(followTimer); stepThenRun(followUp); }
    return;
  }
  // T-4214: a name its stations took when applied (or for a station refused, or another game's) has nothing left to do
  if (msg.kind === "stationName" && !nameOwed(msg.ticket, game)) return;
  // T-3803: a follow-up's first edit HOLD_MS ago: an edit neither joins it nor goes ahead of it any more, so its state
  // still comes at most then (T-2404, T-2766) however quickly edits keep coming
  const due = !!followUp && msg.kind === "edit" && performance.now() - followUp.ask.at >= HOLD_MS;
  // T-4272: demand's results wait for the state; the page's money figures and the clock pass
  if (followUp && waitsFollowUp(msg, followUp) && !(followUp.held && passesHeld(msg, followUp.held))) return keepUntilState(msg);
  if (followUp && !jumpsAhead(msg) && !passesFollowUp(msg) && !(followUp.held && passesHeld(msg, followUp.held) && !due)) {
    // T-3803: an edit goes ahead of a follow-up's state still to come or between its pieces
    if (!due && api && !modelDown && goesAhead(msg, followUp)) standAside(followUp);
    else await guardedFollowUp(due ? "due" : msg.kind === "query" ? "query:" + msg.q.what : msg.kind);
  }
  if (msg.kind === "release") return;
  if (msg.kind === "leave") return await leave(msg);
  if (!api) return;
  if (modelDown) {
    const asked = msg.kind === "edit" || msg.kind === "saveSlot" || msg.kind === "import" || msg.kind === "export" || msg.kind === "newGame";
    // T-4109: the page's notice says it; a download still works, from the network kept after the last edit
    if (msg.kind === "export" && !msg.share && kept) return await whole(msg);
    if (asked && !(msg.kind === "edit" && downTold)) { putDownTold(true); send({ kind: "refused", notice: MODEL_DOWN }); }
    if (msg.kind === "edit") send({ kind: "edited", seq: msg.seq, ok: false, issues: [], charge: 0 });
    else failAnswer(msg, []);
    return;
  }
  // T-3734: a message made in a game no longer in hand (T-715's clock, a queued stop edit) is dropped here; an edit is
  // answered as refused with nothing to say, so nothing waits on it
  if ((msg.kind === "clock" || msg.kind === "edit") && msg.game !== undefined && msg.game !== game) {
    if (msg.kind === "edit") send({ kind: "edited", seq: msg.seq, ok: false, issues: [], charge: 0 });
    return;
  }
  if (isPreview(msg) && stale(msg.seq)) return sendStale(msg.seq);
  const changes = !isPreview(msg) && msg.kind !== "query";
  if (changes) putTodayGen(todayGen + 1);
  try {
    if (msg.kind === "saveSlot") return await manageSlot(msg);
    if (msg.kind === "import" || msg.kind === "export" || msg.kind === "newGame") return await whole(msg);
    await handle(msg);
    // T-1407: an edit's follow-up after the previews asked so far
    // (one still stepping goes on in its own tasks)
    if (followUp?.held && !followUp.held.stops && isPreview(msg)) holdOn();
    else if (followUp && !followUp.step && !followUp.held && !followUp.job && jumpsAhead(msg)) startNow(followUp);
  } catch (err) {
    if (isPanic(err)) await recover(err, msg.kind, msg.kind === "edit" ? EDIT_UNDONE : MODEL_RESTORED, performance.now(), msg.kind === "edit");
    else send({ kind: "error", message: String(err) });
    failAnswer(msg);
  } finally {
    // T-3803: a follow-up an edit went ahead of and no edit's took in (the edit refused, or one with no state) goes
    // on its own
    if (followUp?.carry) {
      followUp.carry = false;
      startNow(followUp);
    }
    if (changes) putTodayGen(todayGen + 1);
    if (msg.kind === "edit" || (msg.kind === "saveSlot" && msg.action !== "save") || msg.kind === "import" || msg.kind === "newGame" || msg.kind === "flipSide") putTrackVersion(trackVersion + 1);
  }
};

let maskedRoute = "";
const routeKey = (r: RouteAsk) => `${trackVersion}|${JSON.stringify([r.from, r.to, r.pis, !!r.single, !!r.quad, !!r.through, designOf(r), r.flyover ?? null, !!r.hsr, !!r.tram, !!r.street, r.street ? r.streetLine ?? null : null, !!r.ramp, r.beside ?? null])}`;

/** An answer for a message the track model could not work out, so nothing waits on it for ever. */
function failAnswer(msg: ToClock, issues: string[] = ["Internal"]) { // i18n-ignore
  if (msg.kind === "edit") send({ kind: "edited", seq: msg.seq, ok: false, issues, charge: 0 }); // i18n-ignore
  if (msg.kind === "previewStation") send({ kind: "stationCheck", seq: msg.seq, ok: false, issues, cost: 0 }); // i18n-ignore
  if (msg.kind === "previewCrossover") send({ kind: "crossoverCheck", seq: msg.seq, ok: false, issues, cost: 0, parts: [] }); // i18n-ignore
  if (isPreview(msg)) send({ kind: "preview", seq: msg.seq, ok: false, cost: 0, lengthM: 0, mult: 0, issues, verts: new Float64Array(0), pts: new Float32Array(0) });
  if (msg.kind === "query" || msg.kind === "flipSide") send({ kind: "answer", seq: msg.seq, data: new Float64Array(0) });
}

/** Once a wall second: autosave when due. */
function tick() {
  // T-4272: nor while answered edits owe their services, or the cars bought for them are off the ledger (an edit's
  // follow-up works them out and books them within a few tasks)
  if (!api || modelDown || api.owed_left()) return;
  if (saves.dirty && Date.now() - saves.savedAt >= AUTOSAVE_MS) void autosave();
  else if (saves.cloudRetryAt && Date.now() >= saves.cloudRetryAt) { saves.cloudRetryAt = Date.now() + CLOUD_RETRY_MS; void cloudUp(saves.activeSlot); }
  noteUnsaved();
}

/** T-284: how often the cash is settled and sent while the clock runs, wall ms: 15 game seconds at
 * normal speed, 4 game minutes at the fastest. */
const LIVE_MONEY_MS = 250;
function liveMoney() {
  if (!api || modelDown || !(clk.rate > 0)) return;
  if (settle()) sendMoney();
  else sendCash();
}

/** Messages that replace or copy the whole game. */
async function whole(msg: Extract<ToClock, { kind: "import" | "export" | "newGame" }>) {
  if (msg.kind === "export") {
    const day = Math.floor(gameNow() / 86400), share = msg.share ? { share: true as const } : {};
    try {
      if (msg.share) {
        const bytes = await roomFor(saveBytes, true);
        send({ kind: "exported", bytes, day, ...share }, [bytes.buffer]);
      } else {
        // T-3869: a download is made in parts the page puts in the file one by one, so it needs no buffer the size
        // of the file, nor the slot's write: a game that cannot be saved for memory can still be downloaded
        const parts = await roomFor(async () => saveParts(), true);
        send({ kind: "exported", parts, day }, [...new Set(parts.map((p) => p.buffer))]);
      }
    } catch (err) {
      if (isPanic(err)) throw err;
      console.error("T-3869: the game could not be downloaded:", err); // i18n-ignore
      send({ kind: "exported", day, ...share, error: outOfRoom(err) ? NO_MEMORY_TO_SAVE : "savesPanel.downloadFailed" });
    }
  } else {
    try {
      const name = msg.name?.trim();
      // i18n-ignore: a stored name or a developer diagnostic
      if (msg.kind === "import") await intoNewSlot(name || "Loaded game", () => loadGame(new Uint8Array(msg.bytes), "file"), true);
      else {
        const spec = msg.game;
        await intoNewSlot(name || newGameName(await listSlots().catch(() => []), spec ?? { mode: "normal" }), () => newGame(spec), !!spec?.real);
      }
    } catch (err) {
      if (isPanic(err)) {
        if (msg.kind !== "import") throw err;
        // T-2820: rebuilt on the game the player had, then the reason this file did not open
        await recover(err, msg.kind, MODEL_RESTORED, performance.now(), true);
        send({ kind: "loaded", from: "file", clock: gameNow(), frame: frameSpec(), error: SAVE_PANIC, why: panicWhy(err), game });
        return;
      }
      // "file": the main thread keeps the game it has
      send({ kind: "loaded", from: "file", clock: gameNow(), frame: frameSpec(), error: (err as Error).message, game });
    }
  }
}

/** Before an edit that makes new track: the masks its geometry needs (worked out as its preview). */
async function masksFor(op: EditOp) {
  const a = api!;
  if (op.op === "route" && maskedRoute === routeKey(op)) return;
  if (op.op === "route") await withMasks(() => a.preview_route(...routeArgs(a, op)), routePoints);
  else if (op.op === "edgePis") await withMasks(() => (a.set_design_speed(designOf(op)), a.preview_edge(op.edge, pisArr(op.pis))), routePoints);
  else if (op.op === "moveNode") await withMasks(() => (a.set_design_speed(designOf(op)), a.preview_move_node(op.node, op.x, op.y)), nodePoints);
  else if (op.op === "moveNodes") await withMasks(() => (a.set_design_speed(designOf(op)), a.preview_move_nodes(new Uint32Array(op.nodes), op.dx, op.dy)), nodePoints);
  // T-1577: a bus is timed from the street speeds of the zones it crosses
  else if (op.op === "addBusLine" || op.op === "busRoute") await ensureStreets(zonesOfPoints(op.route, 3));
}

async function handle(msg: ToClock) {
  const watched = msg.kind === "edit" || msg.kind === "query";
  const detail = msg.kind === "edit" ? { kind: msg.kind, seq: msg.seq, op: msg.op.op }
    : msg.kind === "query" ? { kind: msg.kind, seq: msg.seq, q: msg.q.what, line: "line" in msg.q ? msg.q.line : undefined } : null;
  if (watched) freezeCheckpoint("handle-start", detail);
  try { return await handleMessage(msg); }
  finally { if (watched) freezeCheckpoint("handle-end", detail); }
}
async function handleMessage(msg: ToClock) {
  if (!api) return;
  if (settle()) sendMoney();
  // an edit's new departures start now; trains already running keep their places (T-206)
  api.set_now(gameNow());
  if (msg.kind === "opened") {
    putOpened([...msg.zones]);
    saves.dirty = true;
  } else if (msg.kind === "demandFine") {
    putDemandFine([...msg.zones]);
    putDemandBuilt(msg.built.map((b) => [...b]));
    saves.dirty = true;
  } else if (msg.kind === "lineShape") {
    if (msg.shape) lineShapes.set(msg.line, msg.shape);
    else lineShapes.delete(msg.line);
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
  } else if (msg.kind === "newHeadway") {
    putNewLineHeadway(msg.seconds);
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
  } else if (msg.kind === "bindNew") {
    api.set_bind_new(msg.on);
  } else if (msg.kind === "hideLines") {
    putHiddenLines(new Set(msg.lines));
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
  } else if (msg.kind === "listOrder") {
    api.set_list_order(msg.list, new Uint32Array(msg.ids));
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
  } else if (msg.kind === "modded") {
    const was = modded?.length ?? 0;
    putModded(withModded(modded));
    if ((modded?.length ?? 0) === was) return;
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
    sendMoney();
    void autosave(); // the mark is in the slot at once
  } else if (msg.kind === "sandbox") {
    if (mode === "sandbox") return;
    setMode("sandbox");
    putGuide(undefined);
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
    sendMoney();
    void autosave(); // the slot's card shows the mark at once
  } else if (msg.kind === "edit") {
    if (msg.op.op === "fares" || msg.op.op === "fareScale" || msg.op.op === "fareCityScale" || msg.op.op === "fareByCity" || msg.op.op === "fareHsr" || msg.op.op === "fareFerry" || msg.op.op === "fareBus" || msg.op.op === "fareTransfer" || msg.op.op === "fareLine" || msg.op.op === "fareZones") {
      // T-3567: a line's own multiple for a line that is gone is refused
      const ok = apply(msg.op).r === 0 || msg.op.op !== "fareLine";
      if (ok) {
        // T-3740: a removed area's stations come out of it, which the map and panels show
        const tidied = msg.op.op === "fareZones" && tidyStationFares(api) > 0;
        plainWord(msg.seq, !tidied);
        saves.dirty = saves.edited = true; saves.editAt = Date.now();
        if (tidied) { forget(); sendState(); }
        sendMoney();
        scheduleSnapshot();
      }
      send({ kind: "edited", seq: msg.seq, ok, issues: [], charge: 0 });
      return;
    }
    if (msg.op.op === "fareRef") {
      // T-2679: no fare changes, so demand is not asked again
      const r = apply(msg.op).r;
      if (r === 0) { plainWord(msg.seq, true); saves.dirty = saves.edited = true; saves.editAt = Date.now(); sendMoney(); }
      send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: [], charge: 0 });
      return;
    }
    if (msg.op.op === "borrow" || msg.op.op === "repay") {
      const r = loanOp(msg.op);
      if (r === 0) {
        plainWord(msg.seq, true);
        saves.dirty = saves.edited = true; saves.editAt = Date.now();
        sendMoney();
        keep();
      }
      send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: [], charge: 0 });
      return;
    }
    if (msg.op.op === "link") {
      // T-197: a setting outside the undo history, like fares; the complexes change, so the
      // drawing (links in the state) and demand (the snapshot's complexes) both follow
      const { r } = apply(msg.op);
      // T-1281: a passage between constructed stations is paid at once
      const charge = r === 0 && msg.op.on ? api.link_charge() : 0;
      if (r === 0) {
        plainWord(msg.seq, false);
        saves.dirty = saves.edited = true; saves.editAt = Date.now();
        if (charge > 0) bookCharge(msg.seq, charge, 0);
        sendState();
        scheduleSnapshot();
        keep();
      }
      send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : msg.op.on ? issues() : [], charge });
      return;
    }
    if (msg.op.op === "addBuilding" || msg.op.op === "reshapeBuilding" || msg.op.op === "renameBuilding" || msg.op.op === "removeBuilding") {
      // T-2218: as a passage: outside the undo history and paid at once; the complexes change with
      // the outline, so the drawing and demand both follow
      const { r } = apply(msg.op);
      const charge = r === 0 && (msg.op.op === "addBuilding" || msg.op.op === "reshapeBuilding") ? api.building_charge() : 0;
      const node = r === 0 && msg.op.op === "addBuilding" ? api.last_building() : undefined;
      if (r === 0) {
        plainWord(msg.seq, false);
        saves.dirty = saves.edited = true; saves.editAt = Date.now();
        if (charge > 0) bookCharge(msg.seq, charge, 0);
        sendState();
        if (msg.op.op !== "renameBuilding") scheduleSnapshot();
        keep();
        if (charge > 0) void autosave();
      }
      send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : issues(), charge, node });
      return;
    }
    // T-4214 stage D: a track run's segment sent on from the one before starts at that one's end node
    if (msg.op.op === "route") {
      const r = runEnds(msg.op);
      if (!r) {
        routeEnded(msg.seq, undefined);
        refusedLast = msg.seq;
        return void send({ kind: "edited", seq: msg.seq, ok: false, issues: [], charge: 0 });
      }
      msg.op = r;
    }
    await parkedOn(worldWaterIntoModel());
    await parkedOn(masksFor(msg.op));
    if (msg.op.op === "construct" || msg.op.op === "constructAll" || msg.op.op === "constructLine" || msg.op.op === "rebuild") {
      const missing = await parkedOn(groundForBlueprint());
      if (missing) return void send({ kind: "edited", seq: msg.seq, ok: false, issues: [missing], charge: 0 });
    }
    const tA = performance.now(), forced = takeForced();
    // T-2763: blueprint track taken out carries no train, as a segment put in (T-1780)
    const bpRemove = msg.op.op === "removeEdge" && posted.edges.get(msg.op.edge)?.built === false;
    // T-1071: a train count takes effect a moment ahead, once its trips are on screen
    // T-1073: so is an undo or redo of one
    const service = isServiceOp(msg.op);
    if (service) {
      api.set_now(gameNow());
      api.set_edit_lead(clk.rate * serviceLeadS());
    }
    const named = nameOnly(msg.op);
    // T-4145: an undo asked while the edit it meant was out, and that edit refused: nothing taken back (no notice)
    if ((msg.op.op === "undo" || msg.op.op === "redo") && msg.op.after !== undefined && refusedLast >= 0 && refusedLast <= msg.op.after) {
      refusedLast = -1;
      api.set_edit_lead(0);
      return void send({ kind: "edited", seq: msg.seq, ok: false, issues: [], charge: 0 });
    }
    msg.op = withName(msg.op, game);
    const before = wordBefore(api, msg.op, posted);
    api.take_service_moved(); // T-4249: what the previews asked meanwhile moved and put back does not count
    const applied = apply(msg.op);
    // T-4272: trains owed services needed, bought as the edit began (their services worked out first)
    bookTrains(api.take_trains_bought());
    const moved = api.take_service_moved();
    const { r, fault, line, node, skipped, charge: charged, trainCharge, why } = applied;
    refusedLast = r === 0 ? -1 : msg.seq;
    if (msg.op.op === "route") routeEnded(msg.seq, r === 0 ? node : undefined);
    api.set_edit_lead(0); // a refused edit leaves it set
    if (import.meta.env.DEV && panicAfterEdit) { putPanicAfterEdit(false); panicNow(); }
    const tB = performance.now();
    putTiming({});
    // an op of two edits (T-1899) says what both charged
    const charge = r === 0 ? charged ?? api.last_charge() : 0;
    const op = msg.op, seq = msg.seq;
    const answer = (early?: boolean, held?: Float32Array[]) => send({ kind: "edited", seq, ok: r === 0, issues: r === 0 ? why ?? [] : r === INSERT_REFUSED ? ["InsertStop"] : r === DROP_REFUSED ? ["DropStop"] : r === ORDER_REFUSED && fault ? [`StopOrder:${fault.join(":")}`] : issues(), charge, line, node, paid: paidFor(op), skipped, ms: { apply: tB - tA, ...timing, ...forced, total: performance.now() - tA }, early, held }); // i18n-ignore
    if (r !== 0) return answer();
    placedNameless(api, msg.op, game);
    // T-4145: what the page can show of the edit, before its rows and network (a stop click's line path, T-4114)
    // T-4170: and whether a network follows (`later`'s `net` below: none for a service edit or a rename sent as rows)
    const noNet = service || (named && !msg.early && !msg.rows && rowsCanGo());
    const word = wordAfter(api, op, applied, posted, before, charge);
    // T-4214: and the records it touched, which the page patches the map its tools click on with (none for an edit whose
    // network is not drawn, a service edit's, a rename's or a stop click's: no click waits for a stop click, and its rows
    // follow at once)
    const tP = performance.now();
    const patch = service || named || msg.rows ? undefined : mapPatch(api);
    timing.patch = performance.now() - tP;
    if (!patch) timing.noPatch = 1;
    if (import.meta.env.DEV && !patch && !service && !named && !msg.rows) console.debug(`T-4214: no patch for ${op.op}: ${patchMiss}`);
    send({ kind: "applied", seq, word, ...(noNet ? { noNet: true as const } : {}), ...(patch ? { patch } : {}) });
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
    if (line !== undefined) { newLineShape(op, line); newLineFare(op, line); }
    // T-3581: a line deleted or brought back by an undo or redo has its fare multiple dropped or given back in the
    // state this edit sends, not only at the next network for demand
    if (op.op === "removeLine" || op.op === "undo" || op.op === "redo") tidyLineFares(api);
    if (op.op === "undo" || op.op === "redo") tidyStationFares(api);
    if (charge > 0) bookCharge(seq, charge, trainCharge ?? api.last_train_charge());
    forget();
    const demand = api.dirty_demand_lines();
    // T-1780: blueprint track changes no train's run, so the trips and boarding account stand
    // (about 60 ms on New York's real save, more on shared track); T-1805: nor does the
    // junction it pays for when it branches off constructed track; T-4249: nor any edit that
    // rebuilt, rescheduled or re-timed no line (a station on track no line runs over)
    const trips = moved && !((op.op === "route" || bpRemove || named) && !demand.length);
    const effectiveAt = service ? api.edit_time() : undefined;
    // T-1407: the track tool's free segment is answered now (its next ghost needs only the model),
    // its state a moment later (a junction it pays for has its money's split read off that state)
    const early = !!msg.early && !trips;
    // T-2404: its state held for the pointer's rest, its edges drawn as stand-ins meanwhile; not a
    // segment that paid (a junction off built track splits built track, and the client reads the
    // charge's split off the next state)
    const hold = early && charge === 0;
    if (early) answer(true, hold ? heldEdges() : undefined);
    // T-2766: a stop click's rows (the world, its money) and its answer after them, so the next click reads the stops
    // this one left; the network, trips and boarding wait for the clicks to stop. T-4114: the rows a task later, after
    // the clicks already queued, which share them (`rowsSoon`); the answers go with them (`Waiting.worldSent`)
    const stops = !!msg.rows && !early && !service && rowsCanGo();
    if (stops) rowsSoon();
    // T-3795: a change of periods redoes every line's boarding before the full state (about a second on New York's
    // real save), which the schedule window does not need: its rows go first. The answer stays after the full state,
    // and the follow-up is not held as a stop click's is.
    if (!stops && !early && op.op === "setPeriods") sendRows();
    // T-3793: a name edit's rows (the world: labels, panels, lists) now and its answer after them; its state follows
    // unheld. T-3803: that state leaves the renderer's network as it is, as a train count's does (nothing drawn in it
    // has a name, and on a big save building it took most of a second, which the next edit waited behind)
    // T-4147: and a line edit's list and panel entries (`rowsFirst`), its network still to come
    const renamed = (named || rowsFirst(op)) && !early && !service && !msg.rows && sendRows();
    if (renamed) answer();
    // T-4214: a map click's patched edit that cost nothing (`click`: a station, a delete) is answered now: the map the tools
    // pick on holds it, so the tool waiting on the answer (a station's tries, a delete) lets the next click go. Any other
    // edit is answered once `world` holds it, as before; a paid one waits for its state, whose world its money's split is
    // read off (`chargeShares`).
    const patched = !!patch && !!msg.click && !(charge > 0) && !early && !renamed && !stops;
    if (patched) answer();
    // T-1407: the state, the trips and the boarding hour (and the answer, so the world is up to
    // date when it arrives) after the previews asked meanwhile
    // T-3803: answered already, or once a world holding it went (a rows state when an edit comes before its own state);
    // an early answer's money split waits for the next state the page takes
    let answered = early || renamed || patched, shares = early && charge > 0;
    const rest = () => {
      // T-3758: a line's fare mode changes no train's run, yet demand reads it; T-3892: nor need a boat size (its riders, fares);
      // T-3915: nor BRT on a route with no road in town (its fares); T-3944: which a Construct turns on
      const brtOn = (op.op === "constructLine" || op.op === "constructAll" || op.op === "construct") && charge > 0;
      if (demand.length || op.op === "undo" || op.op === "redo" || op.op === "separate" || op.op === "lineFareDist" || op.op === "ferryBoat" || op.op === "busBrt" || brtOn || stationsKey() !== snapStations) scheduleSnapshot();
      if (!answered) { answered = true; answer(); }
      keep();
      // Paid for something: keep it (T-029, autosave on construct).
      if (charge > 0) void autosave();
    };
    const waiting: Waiting = { owes: () => !answered || shares, worldSent: () => { shares = false; if (!answered) { answered = true; answer(); } } };
    later({ net: !service && !(renamed && named), name: renamed && named, service, effectiveAt, trips, early, at: tA }, rest, () => { if (!answered) { answered = true; failAnswer(msg); } }, waiting, early ? EARLY_WAIT_MS : 0,
    // the boarding account's lines the edit moved (~1.5 ms a line; a train count on New York's
    // real save moves ~90 of them), a few a task, so a ghost asked meanwhile waits for one step;
    // a service edit's first step only before its state (T-3655)
    trips && api.has_riders() ? editSteps(service) : undefined,
    hold || stops, stops);
  } else if (msg.kind === "stationName") {
    // T-4214: stations placed before their name came take it now, in the state their edit's follow-up sends (it goes ahead
    // of that state, `goesAhead`), or in a rows state of its own once that state went
    if (msg.game !== game || !nameOwedStations(api!, msg.ticket, msg.name, game)) return;
    saves.dirty = saves.edited = true; saves.editAt = Date.now();
    const patch = mapPatch(api!);
    send({ kind: "named", ...(patch ? { patch } : {}) });
    later({ net: false, name: true, service: false, trips: false, early: false, at: performance.now() }, keep, () => {}, null, 0);
  } else if (msg.kind === "clock") {
    putClk({ t: msg.t, rate: msg.rate, at: msg.at });
  } else if (msg.kind === "income") {
    putIncomePerHour(msg.perHour.map((p) => p.slice(0, PERIODS.length)));
    // T-4284: with `order`, only the lines whose entries changed came; the others stand as held
    const got = new Map((msg.lines ?? []).map((l) => [l.line, l]));
    const order = msg.order ?? (msg.lines ?? []).map((l) => l.line);
    putLineIncome(new Map(order.map((id) => [id, got.get(id)?.perHour ?? lineIncome.get(id) ?? []])));
    putRidesPerHour(msg.rides.map((p) => p.slice(0, PERIODS.length)));
    putLineRides(new Map(order.map((id) => [id, got.get(id)?.rides ?? lineRides.get(id) ?? []])));
  } else if (msg.kind === "regionShares") {
    putLineShares(new Map(msg.lines.map((l) => [l.line, { fare: l.fare, ride: l.ride, run: l.run }])));
    putUpkeepRegions(msg.upkeep);
    // the money state carries the running cost an hour, which the factors move
    if (applyRunFactors()) sendMoney();
  } else if (msg.kind === "points") {
    const got = new Map(msg.lines.map((l) => [l.line, l.regions]));
    putLinePoints(msg.order ? new Map(msg.order.map((id) => [id, got.get(id) ?? linePoints.get(id) ?? []])) : got);
  } else if (msg.kind === "star" && msg.share !== undefined) {
    const g = goals.get(msg.zone) ?? {};
    if (g[msg.share] === undefined) {
      goals.set(msg.zone, { ...g, [msg.share]: Math.floor(gameNow()) });
      saves.dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "star") {
    if (!stars.has(msg.zone)) {
      stars.set(msg.zone, Math.floor(gameNow()));
      saves.dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "achieve") {
    // T-2124: a sandbox earns none
    if (mode !== "sandbox") {
      const fresh = !achievements, before = achievements?.size ?? 0;
      putAchievements(achievements ?? (new Map())); const now = achievements!, at = Math.floor(gameNow());
      for (const [k, v] of Object.entries(msg.got)) if (!now.has(k)) now.set(k, { ...v, at });
      if (fresh || now.size !== before) {
        saves.dirty = true;
        sendMoney();
      }
    }
  } else if (msg.kind === "guide") {
    if (mode !== "sandbox") { // T-1835: Settings can start the guide in a game saved without one
      putGuide(msg.guide);
      saves.dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "home") {
    putHome(msg.home ?? undefined);
    saves.dirty = true;
  } else if (msg.kind === "lineNotes") {
    let added = false;
    for (const [id, v] of Object.entries(msg.notes)) if (!lineNotes.has(id)) { lineNotes.set(id, v); added = true; }
    if (added) {
      saves.dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "chargeShares") {
    // T-531: move a paid edit's charge from "" to the regions it was spent in
    const c = heldCharges.get(msg.seq);
    heldCharges.delete(msg.seq);
    if (c) {
      const move = (shares: [string, number][], v: number, k: "build" | "trains") => {
        if (!v || !shares.length) return;
        for (const x of regionEntries("", c.day, c.hour)) x[k] -= v;
        for (const [z, s] of shares) for (const x of regionEntries(z, c.day, c.hour)) x[k] += v * s;
      };
      move(msg.build, c.build, "build");
      move(msg.trains, c.trains, "trains");
      saves.dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "riders") {
    // a new demand result: the boarding account follows it (T-164), a few lines at a time, one
    // kind of day's riders a message (T-260)
    // an empty one clears the kind (an away day's once none is near, T-264)
    if (peakC[msg.day] && msg.packed.length >= PERIODS.length) peakC[msg.day] = strengths(msg.packed.subarray(0, PERIODS.length));
    api.set_riders(msg.packed, msg.day);
    if (msg.arrive) api.set_arrivals(msg.arrive, msg.day);
    stepBoarding();
  } else if (msg.kind === "transfers") {
    // T-554: when riders changing lines reach their next platform; the account is unchanged.
    // T-1810: the same transfers again change nothing drawn, and while new riders are still being
    // worked through, the step that finishes them sends the hour (a whole hour now would work out
    // every line left at once, ~0.5 s on shared New York)
    if (sameFloats(msg.packed, xferSet)) return;
    putXferSet(msg.packed);
    api.set_transfers(msg.packed);
    if (api.boarding_step(0)) stepBoarding();
    else sendTrips(true);
  } else if (msg.kind === "flush") {
    void autosave().then(cloudFlush);
  } else if (import.meta.env.DEV && msg.kind === "debugPanic") {
    if (msg.when === "afterEdit") putPanicAfterEdit(true);
    else if (msg.when === "read") api.edge_polyline(0xffffffff, 0);
    else panicNow();
  } else if (import.meta.env.DEV && msg.kind === "debugCash") {
    api.set_cash(msg.cash);
    if (msg.fleet) { api.set_fleet(api.cars_needed()); api.set_fleet_fast(api.cars_needed_fast()); }
    sendMoney();
  } else if (msg.kind === "flipSide") {
    // T-3297: every train, tram and bus to the other side, in place (ids kept); what changes is every figure, so
    // the game goes on as after a rebuilt model: the hour's accounts afresh, demand asked again
    const data = api.flip_right_hand();
    restart(gameNow(), "autosave", "rebuilt");
    saves.dirty = true;
    send({ kind: "answer", seq: msg.seq, data }, [data.buffer]);
  } else if (msg.kind === "query") {
    const q = msg.q;
    if (q.what === "lineProfile") {
      // T-2734: worked between the other messages, and answered when done
      const a = api, seq = msg.seq;
      const masksUnder = async (xy: number[]) => (await Promise.all(zonesOfPoints(xy).map((id) => fetchWater(id).catch(() => null)))).filter((m): m is WaterRaster => !!m);
      void lineProfile(a, q.line, packLngLat, () => api === a, masksUnder).catch(() => new Float64Array(0))
        .then((data) => send({ kind: "answer", seq, data }, [data.buffer]));
      return;
    }
    const t0 = performance.now();
    const data =
      q.what === "insertStop" ? api.insert_stop(q.line, q.station)
      : q.what === "clickPath" ? api.click_path(q.line, q.station)
      : q.what === "stopTiers" ? api.stop_tiers(q.line)
      : q.what === "stopGap" ? api.stop_gap(q.line, q.station)
      : q.what === "passedStations" ? api.passed_stations(q.line)
      : q.what === "junction" ? api.junction_info(q.node)
      : q.what === "edgePis" ? api.edge_pis(q.edge)
      : q.what === "flyover" ? new Float64Array([api.flyover_cost(q.node)])
      : q.what === "upgradeCosts" ? new Float64Array(q.nodes.map((n) => api!.station_upgrade_cost(n, q.bit)))
      : q.what === "tracksCost" ? new Float64Array([api.station_tracks_cost(q.node)])
      : q.what === "upgradeCost" ?new Float64Array(UPGRADE_BITS.map((b) => api!.station_upgrade_cost(q.node, b)))
      : q.what === "platformCost" ? new Float64Array([api.platform_cost(new Uint32Array(q.nodes), q.platform)])
      : q.what === "doubleUp" ? doubleUpAnswer(q.edge)
      : q.what === "oneWays" ? new Float64Array(api.oneway_effects_many(new Uint32Array(q.edges), new Uint8Array(q.dirs)))
      : q.what === "lineAccount" ? api.line_account(q.line, q.day)
      : q.what === "linePeriods" ? api.line_periods(q.line, q.day)
      : q.what === "linesPeriods" ? linesPeriods(api, q.lines, q.day)
      : q.what === "today" ? today()
      : q.what === "lineHeld" ? (q.rows && api.line_period_count(q.line) > 3 ? api.line_row_held(q.line) : api.line_held(q.line))
      : q.what === "lineNudges" ? api.line_nudges(q.line)
      : q.what === "lineMarks" ? api.line_marks(q.line)
      : q.what === "lineRelated" ? api.line_related(q.line, complexOfNodes())
      // T-2603: the line's timetable chart, four numbers a trip (`TrackApi.line_trips`)
      : q.what === "lineTrips" ? api.line_trips(q.line, q.t0, q.t1)
      // T-3346: a station's departures board, three numbers a departure (`TrackApi.station_departures`)
      : q.what === "stationDeps" ? api.station_departures(q.line, q.node, q.t0, q.t1)
      : q.what === "removal" ? new Float64Array(api.removal_lines(new Uint32Array(q.edges), new Uint32Array(q.stations)))
      : q.what === "busQuote" ? busQuoteLL(api, q.line, q.route)
      : q.what === "rebuildQuote" ? rebuildQuoteAnswer(q.edges, q.stations, q.level, q.kind, q.tracks)
      : q.what === "blueprintQuote" ? Float64Array.from(api.blueprint_cost_items())
      : q.what === "blueprintPieces" ? Float64Array.from(api.blueprint_piece_costs())
      : q.what === "lineCost" ? new Float64Array([api.line_build_cost(q.line), api.line_train_cost(q.line)])
      : q.what === "linePairs" ? new Float64Array([api.line_pairs_ok(q.line) ? 1 : 0, ...api.line_pairs_why(q.line)])
      : q.what === "buildingQuote" ? buildingQuoteAnswer(q.id, q.corners)
      : q.what === "lineWaits" ? (q.period !== undefined ? api.line_waits_in(q.line, q.period) : api.line_waits(q.line, q.level))
      // T-2232: [station, its passing tracks' price] pairs
      : q.what === "passingSpots" ? Float64Array.from(Array.from(api.passing_spots(q.line), (n) => [n, api!.station_upgrade_cost(n, UPGRADE.passing)]).flat())
      : q.period !== undefined ? api.line_delays_in(q.line, q.period) : api.line_delays(q.line, q.level);
    send({ kind: "answer", seq: msg.seq, data, ms: performance.now() - t0 }, [data.buffer]);
  } else if (msg.kind === "previewStation") {
    const at = msg.at;
    api.set_station_one(at.one ? new Float64Array([at.one.x, at.one.y]) : new Float64Array(0));
    const r = at.kind === "node" ? api.preview_station(1, at.node, 0, 0, msg.platform)
      : at.kind === "edge" ? api.preview_station(2, at.edge, at.x, at.y, msg.platform)
      : (api.set_route_kind(at.tram ? 3 : 0), api.set_route_hsr(!!at.hsr), api.preview_free_station(freeEnds(at), at.quad ? 4 : 2, msg.platform));
    send({ kind: "stationCheck", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : issues(), cost: r === 0 ? api.station_cost() : 0 });
  } else if (msg.kind === "previewCrossover") {
    const out = api.preview_crossover(msg.edge, msg.x, msg.y);
    const ok = out[0] === 1;
    const parts: Float32Array[] = [];
    for (let k = 2; k < out.length; ) {
      const n = out[k++], pts = new Float32Array(n * 3);
      for (let j = 0; j < n; j++, k += 3) {
        const [x, y] = packLocal(out[k], out[k + 1]);
        pts.set([x, y, out[k + 2]], j * 3);
      }
      parts.push(pts);
    }
    send({ kind: "crossoverCheck", seq: msg.seq, ok, issues: ok ? [] : issues(), cost: out[1], parts, anchor: anchor().version }, parts.map((p) => p.buffer));
  } else if (msg.kind === "previewNode" || msg.kind === "previewNodes") {
    const a = api;
    const out = await parkedOn(withMasks(() => (a.set_design_speed(designOf(msg)), msg.kind === "previewNode" ? a.preview_move_node(msg.node, msg.x, msg.y) : a.preview_move_nodes(new Uint32Array(msg.nodes), msg.dx, msg.dy)), nodePoints, msg.seq));
    if (out) sendNodePreview(msg.seq, out);
    else sendStale(msg.seq);
  } else if (msg.kind === "preview" || msg.kind === "previewEdge") {
    const a = api;
    // T-4214 stage D: a ghost from a run's end still to be answered, read as that end's node (none from a refused one)
    const m = msg.kind === "preview" ? runEnds(msg) : msg;
    if (!m) return sendStale(msg.seq);
    const out = await parkedOn(withMasks(() => m.kind === "preview"
      ? a.preview_route(...routeArgs(a, m))
      : (a.set_design_speed(designOf(m)), a.preview_edge(m.edge, pisArr(m.pis))), routePoints, msg.seq));
    if (!out) return sendStale(msg.seq);
    void roadsUnder(routePoints(out));
    if (m.kind === "preview") maskedRoute = masksComplete ? routeKey(m) : "";
    let arc: Float64Array | undefined;
    const ra = msg.kind === "preview" ? a.route_arc() : null;
    if (ra?.length === 3) {
      const [x, y] = packLocal(ra[0], ra[1]);
      arc = new Float64Array([x, y, ra[2], 0]);
    }
    const ok = out[0] === 1, nv = out[3];
    const verts = new Float64Array(nv * 4);
    for (let i = 0; i < nv; i++) {
      const [x, y] = packLocal(out[5 + i * 4], out[6 + i * 4]);
      verts.set([x, y, out[7 + i * 4], out[8 + i * 4]], i * 4);
    }
    const p0 = 5 + nv * 4;
    const pts = new Float32Array(out.length - p0);
    for (let k = p0; k < out.length; k += 3) {
      const [x, y] = packLocal(out[k], out[k + 1]);
      pts.set([x, y, out[k + 2]], k - p0);
    }
    const speeds = a.preview_render_speeds();
    if (speeds.length !== Math.max(0, pts.length / 3 - 1)) throw new Error("Preview speed/point length mismatch");
    const lanes = a.preview_render_lanes();
    if (lanes.length !== speeds.length) throw new Error("Preview lane/speed length mismatch");
    const marks = previewMarkRows(a.preview_marks());
    const ground = msg.kind === "preview" ? Array.from(a.preview_ground()) : undefined;
    const flatJunctions = msg.kind === "preview" ? a.route_flyovers_flat() : 0;
    const joins = msg.kind === "preview" ? a.route_joins() : undefined;
    // T-3168: each flyover leg's slowest curve, labelled on the map beside the route's own
    const legRows = msg.kind === "preview" ? a.preview_leg_curves() : new Float64Array(0);
    const legs = new Float64Array(legRows.length);
    for (let k = 0; k + 3 < legRows.length; k += 4) {
      const [x, y] = packLocal(legRows[k], legRows[k + 1]);
      legs.set([x, y, legRows[k + 2], legRows[k + 3]], k);
    }
    // T-3027: each ramp slower than the track's top speed, labelled as a curve is
    const rampRows = a.preview_ramp_labels();
    const ramps = new Float32Array(rampRows.length);
    for (let k = 0; k + 2 < rampRows.length; k += 3) {
      const [x, y] = packLocal(rampRows[k], rampRows[k + 1]);
      ramps.set([x, y, rampRows[k + 2]], k);
    }
    send({ kind: "preview", seq: msg.seq, ok, cost: out[1], lengthM: out[2], mult: out[4], ground, issues: ok ? [] : issues(), verts, pts, speeds, lanes, arc, anchor: anchor().version, marks, ...(joins ? { joins } : {}), ...(flatJunctions > 0 ? { noFlyover: true } : {}), ...(legs.length ? { legs } : {}), ...(ramps.length ? { ramps } : {}) }, [verts.buffer, pts.buffer, speeds.buffer, lanes.buffer, marks.buffer]);
  } else if (msg.kind === "epoch") {
    if (msg.epoch !== epoch) {
      putEpoch(msg.epoch);
      turnHour();
    }
  } else if (msg.kind === "wholeState") {
    // the client could not take a state as changes to the last (T-1029)
    postWhole();
    if (api) sendState();
  } else if (msg.kind === "diagram") {
    putDiagramOn(msg.on);
    putDiagramWhole(!!msg.whole);
    if (diagramOn && api) postDiagram();
  } else if (msg.kind === "diagramCut") {
    const cities = msg.cities.filter((c) => c === NO_CITIES || isCity(c));
    if (cities.join() !== diagramCities.join()) (putDiagramCities(cities)), (saves.dirty = saves.edited = true), (saves.editAt = Date.now());
    putDiagramInside({ stations: new Set(msg.stations), stops: new Set(msg.stops) });
    if (diagramOn && api) postDiagram();
  } else if (msg.kind === "anchor") {
    // the view travelled far (T-307): every drawn position again, relative to the new anchor
    if (msg.version !== anchor().version) {
      // T-1170: the drawn pieces move over, unless the move is too far for float32 to keep them
      // within a few centimetres there; laid exactly again every SHIFTS_EXACT moves
      const to = makeAnchor(msg.lng, msg.lat, msg.version), [dx, dy] = anchorShift(anchor(), to);
      // (`?patch=check` keeps every build exact, as its T-1029 check compares them with fresh ones)
      if (patchMode !== "check" && (putAnchorMoves(anchorMoves + 1), anchorMoves) % SHIFTS_EXACT && Math.max(Math.abs(dx), Math.abs(dy)) < SHIFT_REACH) putShiftFrom(shiftFrom ?? (anchor()));
      else putShiftFrom(null);
      setAnchor(to);
      // T-1170: only positions are in local units; the rows hold none, so they stay as posted and
      // the page keeps its world's rows (labels, complexes and the rest are not rebuilt)
      putShiftWas(shiftFrom ? new Map(edgeDrawn) : new Map());
      forgetDrawn();
      posted.net = null;
      sender.forget();
      sendState();
    }
  }
}

/** A node drag's preview (`TrackApi.preview_move_node`, T-093): every edge at the node, its
 * vertices one after another and its polyline as one part of `pts`. */
function sendNodePreview(seq: number, out: Float64Array) {
  const ok = out[0] === 1, n = out[2];
  const verts: number[] = [], pts: number[] = [], parts: number[] = [0];
  let i = 3;
  for (let e = 0; e < n; e++) {
    const nv = out[i++];
    for (let k = 0; k < nv; k++, i += 4) verts.push(...packLocal(out[i], out[i + 1]), out[i + 2], out[i + 3]);
    const np = out[i++];
    for (let k = 0; k < np; k++, i += 3) pts.push(...packLocal(out[i], out[i + 1]), out[i + 2]);
    parts.push(pts.length / 3);
  }
  const v = new Float64Array(verts), p = new Float32Array(pts), pa = new Uint32Array(parts);
  const marks = previewMarkRows(api!.preview_marks());
  const partLanes = api!.preview_part_lanes();
  send({ kind: "preview", seq, ok, cost: out[1], lengthM: 0, mult: 0, issues: ok ? [] : issues(), verts: v, pts: p, parts: pa, partLanes, anchor: anchor().version, marks }, [v.buffer, p.buffer, pa.buffer, partLanes.buffer, marks.buffer]);
}

// T-2710: the script has loaded (every import with it); the page posts its held messages on this
send({ kind: "booted" });
