// Station use (T-482): riders a day boarding at each station, split into those starting their rail
// trip there (origins) and those changing lines there (transfers). From the demand solve shown
// (the day the demand bar picks), as the line widths are: only the solve knows where a boarding
// rider came from. Boardings are `demandView.stations`; transfers are the solve's transfer pairs
// (`complexTransfers` over every station), each counted at the station the rider boards at.
// Origins are the rest. Every purpose together: the transfer pairs are not kept by purpose.

import { effect, signal } from "@preact/signals";
import { complexTransfers, demandView, type DemandView } from "./demand";
import { display, type StationUseMode } from "./ui";

export interface StationUse {
  /** boardings a day, transfers included */
  total: number;
  /** of them, riders changing lines here */
  transfers: number;
}

/** Per station id, for the solve `dv`; the last answer stays while a newer one is asked for. */
export const stationUse = signal<{ dv: DemandView; at: Map<string, StationUse> } | null>(null);

/** Riders a day station `id` stands for under mode `m`, or undefined while not known: the total
 * from `dv` itself, origins and transfers from the last split worked out (an older solve's while
 * the one shown is asked for). */
export function useFigure(dv: DemandView | null, m: StationUseMode, id: string): number | undefined {
  if (m === "none" || !dv) return undefined;
  if (m === "total") return dv.stations.get(id)?.boardings;
  const u = stationUse.peek()?.at.get(id);
  if (!u) return undefined;
  return m === "transfers" ? u.transfers : Math.max(0, u.total - u.transfers);
}

let ask = 0;
effect(() => {
  const dv = demandView.value;
  const mode = display.value.stationUse;
  if (!dv || mode === "none" || mode === "total") return;
  const cur = stationUse.peek();
  if (cur?.dv === dv) return;
  const ids = [...dv.stations.keys()];
  const n = ++ask;
  const done = (xfer: Map<string, number>) => {
    if (n !== ask) return;
    const at = new Map<string, StationUse>();
    for (const [id, s] of dv.stations) at.set(id, { total: s.boardings, transfers: Math.min(s.boardings, xfer.get(id) ?? 0) });
    stationUse.value = { dv, at };
  };
  if (!ids.length) return done(new Map());
  void complexTransfers(ids).then((r) => {
    // null: the workers moved on; the next view asks again
    if (!r) return;
    const xfer = new Map<string, number>();
    for (const p of r.pairs) xfer.set(p.toStation, (xfer.get(p.toStation) ?? 0) + p.riders);
    done(xfer);
  });
});
