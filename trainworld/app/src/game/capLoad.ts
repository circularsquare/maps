// T-3165: a game that has just loaded has any hand-set train or bus count above the panel's cap
// (the whole trains the track runs plus one, `trainCap`) lowered to it. Once per load; a cap that
// falls later, from a track edit, leaves counts alone until the player edits the count.

import { effect } from "@preact/signals";
import { queryClock } from "../workers/clockClient";
import { gameReady } from "./loading";
import { edit, linesByNum, trainCap, world } from "./world";

let seq = 0;
let done = false;

async function lowerOverCap(my: number) {
  const w = world.peek();
  if (!w) return;
  for (const l of linesByNum(w).values()) {
    if (my !== seq) return;
    const st = world.peek()?.lineStats[l.id];
    if (!st || st.trainsNeeded.every((n) => n <= 1)) continue;
    const d = await queryClock({ what: "lineHeld", line: l.num });
    if (!d || my !== seq) continue;
    const had = st.trainsNeeded;
    const to = had.map((n, lev) => {
      const s = d[21 + 5 * lev];
      return s > 0 && !st.autoS?.[lev] ? Math.min(n, trainCap(st, st.roundTripByLevel[lev], s)) : n;
    }) as [number, number, number];
    if (to.some((n, i) => n !== had[i])) await edit({ op: "setTrainCount", line: l.num, trains: to }, true, true);
  }
}

effect(() => {
  if (!gameReady.value) {
    done = false;
    seq++;
    return;
  }
  if (done) return;
  done = true;
  const my = seq;
  void lowerOverCap(my);
});

