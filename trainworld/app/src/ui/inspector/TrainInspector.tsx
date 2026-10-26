// The train inspector (SPEC 8): where the train is and who is aboard.

import { clock } from "../../game/clock";
import { demandView, lineDemand } from "../../game/demand";
import { loadFailure } from "../../game/loading";
import { PERIODS, periodAt } from "../../game/periods";
import { distanceOf, exact } from "../../game/format";
import type { LineInput } from "../../game/types";
import { linesById, stationNames, world } from "../../game/world";
import { live } from "../../map/live";
import { tripAt } from "../../map/network";
import { BUS_CARS, CRUSH_PER_CAR, SEATS_PER_CAR, tripLoad } from "../../game/boarding";
import { Chip, Row } from "../widgets";
import { OneRider } from "./OneRider";
import { TrainStops } from "./TrainStops";

export function TrainInspector({ line, profile, dep }: { line: string; profile: number; dep: number }) {
  const w = world.value;
  void clock.minute.value; // re-render each game minute: the train moves
  const l = w ? linesById(w).get(line) : undefined;
  const net = live.overlay?.renderer.net;
  if (!w || !l || !net) return null;
  const st = w.lineStats[l.id];
  // a train standing at a terminal for its next departure is drawn past its trip (T-206)
  const ren = live.overlay!.renderer, ti = ren.tripIndex(profile, dep - ren.tripsEpoch);
  const at = tripAt(net, profile, dep, clock.now(), ti >= 0 && ren.tripEnds ? ren.tripEnds[ti] + ren.tripsEpoch : -Infinity);
  const name = stationNames(w);
  const run = Math.floor(profile / 3) % 2;
  const stops = run === 0 ? l.stops : [...l.stops].reverse();
  const stopS = st.stopS[run] ?? [];
  const next = at ? stopS.findIndex((s) => s > at.s + 1) : -1;
  const len = net.meta[profile * 4 + 3];
  return (
    <>
      <div class="title">
        <Chip line={l} size="lg" />
        <span class="name">{l.name} {st.bus ? "bus" : "train"}</span>
      </div>
      <div class="sub muted">Towards {name.get(stops[stops.length - 1])}</div>
      {at ? (
        <ul class="rows">
          <Row label="Next stop">{next >= 0 ? name.get(stops[next]) : "Terminus"}</Row>
          <Row label="Along the line">
            {distanceOf(at.s, len)}
          </Row>
        </ul>
      ) : (
        <p class="muted">This {st.bus ? "bus" : "train"} finished its trip and left service.</p>
      )}
      {at && <OnBoard line={l} run={run} s={at.s} stopS={stopS} cars={st.bus ? BUS_CARS : st.cars} bus={!!st.bus} profile={profile} dep={dep} />}
    </>
  );
}

/**
 * Who is on board (T-083, T-164, SPEC 8): this train's own riders on the stretch it is running,
 * from the boarding account (game/boarding.ts), against seats and crush load; and the typical
 * train of the period, demand's riders on the stretch shared among the trains then.
 */
function OnBoard({ line, run, s, stopS, cars, bus, profile, dep }: { line: LineInput; run: number; s: number; stopS: number[]; cars: number; bus: boolean; profile: number; dep: number }) {
  const dm = lineDemand(line.id);
  const n = line.stops.length;
  // T-309: the period in the line's own local time
  const p = periodAt(clock.minute.value * 60 + (world.value?.lineStats[line.id]?.shift ?? 0));
  const tph = line.tph[PERIODS[p].demand];
  const trains = tph * (PERIODS[p].to - PERIODS[p].from);
  let seg = 0;
  while (seg + 1 < n - 1 && stopS[seg + 1] <= s + 1) seg++;
  const loads = dm?.loads[p];
  const r = live.overlay?.renderer;
  const i = r?.board ? r.tripIndex(profile, dep - r.tripsEpoch) : -1;
  const own = r?.board && i >= 0 ? tripLoad(r.board, i, run === 0 ? seg : n - 2 - seg) : null;
  if (!demandView.value || own === null || !loads || loads.length !== 2 * (n - 1) || trains <= 0)
    return (
      <ul class="rows">
        <Row label="Riders aboard">
          {/* the train is drawn striped while the riders are being worked out */}
          <span class="muted">{loadFailure.value ? "Not available" : "Calculating"}</span>
        </Row>
      </ul>
    );
  const atEnd = s >= stopS[n - 1] - 1;
  // the typical train this hour on this stretch, from the same account as this one (T-230): its
  // line's trains in this direction drawn this hour
  let sum = 0, many = 0;
  const b = r!.board!, stretch = run === 0 ? seg : n - 2 - seg;
  for (let k = 0; k < r!.tripCount; k++) {
    const pk = r!.trips[k * 2];
    if (Math.floor(pk / 6) !== Math.floor(profile / 6) || Math.floor(pk / 3) % 2 !== run) continue;
    const v = tripLoad(b, k, stretch);
    if (v !== null) { sum += v; many++; }
  }
  const average = atEnd || !many ? 0 : sum / many;
  const aboard = atEnd ? 0 : own;
  const seats = SEATS_PER_CAR * cars, crush = CRUSH_PER_CAR * cars;
  const state = aboard <= 0.5 ? "Empty" : aboard <= seats ? "Seats free" : aboard < crush - 0.5 ? "Standing" : "Full";
  return (
    <>
      <div class="h">
        <span data-tip={`Counted ${bus ? "bus" : "train"} by ${bus ? "bus" : "train"} from the demand forecast: riders board at the forecast's rate until the ${bus ? "bus" : "train"} is full`}>On board</span><span class="right muted small">{PERIODS[p].name.toLowerCase()}{dm?.estimate && <span data-tip="First estimate for this new or changed line, refined when demand finishes updating">, estimate</span>}</span>
      </div>
      <ul class="rows">
        <Row label="Riders aboard" bold>{exact(aboard)}</Row>
        <Row label="Typical train this hour">{exact(average)}</Row>
        <li>
          <span class="grow">
            <span class="board-bar" data-tip={`${exact(aboard)} riders aboard. The bar is full at ${exact(crush)} riders, and the tick marks the ${exact(seats)} seats.`}>
              <i class="fill" style={{ width: Math.min(100, (aboard / crush) * 100) + "%" }} />
              <i class="seats" style={{ left: (seats / crush) * 100 + "%" }} />
            </span>
          </span>
          <span class="val">{state}</span>
        </li>
        <Row label="Seats">{exact(seats)}</Row>
        <Row label="Full at">{exact(crush)}</Row>
      </ul>
      {!atEnd && dm && <TrainStops line={line} dm={dm} board={b} trip={i} run={run} seg={seg} stopS={stopS} profile={profile} dep={dep} />}
      {!atEnd && dm && aboard > 0.5 && <OneRider key={`${profile}:${dep}`} line={line} dm={dm} run={run} seg={seg} p={p} />}
    </>
  );
}
