// T-1436: one rider of a selected train. Drawn from the train's own riders (game/riderSample.ts,
// on the boarding account's boardings and the line's alighting shares), then asked of the demand
// workers as the trip explorer asks: the perceived minutes by train and by car. A way into the
// explorer for the same two stations.

import { useEffect, useRef, useState } from "preact/hooks";
import { demandView, type LineDemand } from "../../game/demand";
import { exact } from "../../game/format";
import { PERIODS } from "../../game/periods";
import { asFilter, purposeColour, purposeName, type PurposeKey } from "../../game/purposes";
import { pickPurpose, pickStops, purposeWeights } from "../../game/riderSample";
import { stationPoint, tripOpenedFrom, tripPeriod } from "../../game/trip";
import type { LineInput, Selection } from "../../game/types";
import { purposeFilter, selection } from "../../game/ui";
import { stationNames, world } from "../../game/world";
import { riderTrip, type RiderTrip } from "../../map/tripView";
import { Icon } from "../widgets";
import { alightShares } from "./TrainStops";
import { minText } from "./tripText";

interface Rider {
  from: number;
  to: number;
  purpose: PurposeKey;
}

/** Draws that found no answer for the pair before the block says so. */
const TRIES = 6;

const pct = (v: number | null) => (v === null ? "" : `${Math.round(100 * v)}%`);

export function OneRider({ line, dm, run, seg, p }: { line: LineInput; dm: LineDemand; run: number; seg: number; p: number }) {
  const w = world.value;
  const dv = demandView.value;
  const n = line.stops.length;
  const wrap = w?.lineStats[line.id]?.loop === "circle";
  const last = wrap ? n - 2 : n - 1;
  const picked = useRef<Rider | null>(null);
  const tries = useRef(0);
  const [, again] = useState(0);
  const [read, setRead] = useState<{ id: string; trip: RiderTrip | null | undefined } | null>(null);
  const weights = dv ? purposeWeights(dv, line.id, p, run * (n - 1) + seg) : [];
  const shares = alightShares(dm, n, run, wrap)[p] ?? [];

  let r = picked.current;
  // a rider who has got off by now is replaced
  if (!r || r.to <= seg || r.from > seg) {
    const at = dm.board?.[p] ? pickStops({ board: dm.board[p], run, n, seg, last, shares, rand: Math.random }) : null;
    r = picked.current = at ? { ...at, purpose: pickPurpose(weights, Math.random) } : null;
  }
  const id = r ? `${line.id}:${run}:${r.from}:${r.to}:${r.purpose}:${p}` : "";

  useEffect(() => {
    if (!r || !w) return;
    const stops = run === 0 ? line.stops : [...line.stops].reverse();
    const a = stationPoint(stops[r.from]), b = stationPoint(stops[r.to]);
    if (!a || !b) return;
    let live = true;
    riderTrip(a, b, r.purpose, p).then((trip) => {
      if (!live) return;
      // an inter-city purpose between two stations of one zone is read as one of the local purposes
      if (trip && !trip.market && r!.purpose === "longer") {
        picked.current = { ...r!, purpose: pickPurpose(weights.filter((x) => x.key !== "longer"), Math.random) };
        again((x) => x + 1);
      } else if (trip === null && tries.current < TRIES) {
        // no answer for this pair: draw another
        tries.current++;
        picked.current = null;
        again((x) => x + 1);
      } else setRead({ id, trip });
    });
    return () => { live = false; };
  }, [id, dv?.version, dv?.round]);

  if (!r || !w) return null;
  const name = stationNames(w);
  const stops = run === 0 ? line.stops : [...line.stops].reverse();
  const from = name.get(stops[r.from]) ?? "a station", to = name.get(stops[r.to]) ?? "a station";
  const trip = read?.id === id ? read.trip : undefined;
  const total = weights.reduce((s, x) => s + x.w, 0);
  const purposeTip = total > 0
    ? `Drawn from the ${exact(total)} riders on this stretch in the ${PERIODS[p].name.toLowerCase()}:\n` + [...weights].sort((x, y) => y.w - x.w).slice(0, 4).map((x) => `${purposeName(x.key)} ${pct(x.w / total)}`).join("\n")
    : "No split by purpose for this stretch yet";
  const better = trip && trip.train !== null && trip.car !== null ? (trip.train <= trip.car ? "train" : "car") : null;
  const chances = trip && trip.pTrain !== null && trip.pDrive !== null
    ? `\nOf the people making this trip in the ${PERIODS[p].name.toLowerCase()}, ${pct(trip.pTrain)} take the train and ${pct(trip.pDrive)} drive`
    : "";
  const minutes = (m: number | null | undefined, none: string) =>
    m === undefined ? <span class="muted">…</span> : m === null ? <span class="muted" data-tip={none}>none</span> : minText(m);
  const another = () => {
    tries.current = 0;
    picked.current = null;
    again((x) => x + 1);
  };
  const open = () => {
    const a = stationPoint(stops[r!.from]), b = stationPoint(stops[r!.to]);
    if (!a || !b) return;
    const back = selection.peek();
    const trip: Selection = { kind: "trip", from: a, to: b, fromStation: stops[r!.from], toStation: stops[r!.to] };
    tripOpenedFrom.value = { trip, from: back, places: null };
    purposeFilter.value = asFilter(r!.purpose);
    selection.value = trip;
    // the explorer resets the period for a new trip; the rider's is the one riding now
    tripPeriod.value = p;
  };
  return (
    <>
      <div class="h">
        <span data-tip="One rider picked from the riders aboard: where they got on and where they get off from the train's boardings, why they ride from this stretch's riders. Minutes are the trip explorer's effective times.">One rider</span>
        <span class="one-rider-acts">
          <button class="btn icon mini" aria-label="Pick another rider" data-tip="Pick another rider" onClick={another}><Icon.cycle /></button>
          <button class="btn icon mini" aria-label="Open this trip in the trip explorer" data-tip="Open this trip in the trip explorer" onClick={open}><Icon.route /></button>
        </span>
      </div>
      <ul class="rows one-rider">
        <li data-tip={`Starts from the homes and places around ${from}`}>
          <span class="grow">From</span>
          <span class="val name">{from}</span>
        </li>
        <li data-tip={`Gets off at ${to}, and the trip ends at the places around it`}>
          <span class="grow">To</span>
          <span class="val name">{to}</span>
        </li>
        <li data-tip={purposeTip}>
          <span class="grow">Purpose</span>
          <span class="val"><i class="mk" style={{ background: purposeColour(r.purpose) }} /> {purposeName(r.purpose)}</span>
        </li>
        <li class={better === "train" ? "b" : ""} data-tip={`By train, in effective time: waits, rides, walks and the fare counted as minutes${chances}`}>
          <span class="grow">By train</span>
          <span class="val">{minutes(trip === undefined ? undefined : trip?.train ?? null, "No train route for this trip right now")}</span>
        </li>
        <li class={better === "car" ? "b" : ""} data-tip={`By car, in effective time: the drive, parking and what the car costs counted as minutes${chances}`}>
          <span class="grow">By car</span>
          <span class="val">{minutes(trip === undefined ? undefined : trip?.car ?? null, "No answer for the car")}</span>
        </li>
      </ul>
    </>
  );
}
