// T-1130: fare changes undo and redo with the game's keys and buttons while they are the player's
// last edits. Fares live in the clock worker, outside the sim's edit history (the track model's,
// in Rust), so their steps are kept here and the sim's undo takes over when none applies.

import { computed, signal } from "@preact/signals";
import { tidyNodes, type FareNode } from "./fareCurve";
import { moneyState } from "./money";
import { edit, editsSent, setFares } from "./world";

interface Step { from: FareNode[]; to: FareNode[] }
const undos: Step[] = [], redos: Step[] = [];
/** What each undo since the last other edit took back, newest last, so redo keeps the order. */
const undone: ("fare" | "sim")[] = [];
/** `editsSent` when the steps were taken; any other edit since drops them. */
let at = 0;
const version = signal(0);

/** The curve last sent from here and when: the game's own copy may not have caught up yet. */
let sent: { nodes: FareNode[]; t: number } | null = null;

const same = (a: FareNode[] | undefined, b: FareNode[]) => !!a && JSON.stringify(a) === JSON.stringify(b);
/** Whether the game's curve is `nodes`, or is about to be (sent in the past 3 s); after a load it
 * is neither, so a step from the game before cannot apply. */
const isNow = (nodes: FareNode[]) => same(moneyState.peek()?.fares.nodes, nodes) || (!!sent && performance.now() - sent.t < 3000 && same(sent.nodes, nodes));
const send = (nodes: FareNode[]) => {
  sent = { nodes, t: performance.now() };
  return setFares(nodes);
};
const fresh = () => {
  if (editsSent.peek() === at) return;
  undos.length = redos.length = undone.length = 0;
  at = editsSent.peek();
};
/** The step a fare undo would take back, if the curve is still where it left it. */
const nextUndo = () => { fresh(); const s = undos[undos.length - 1]; return s && isNow(s.to) ? s : undefined; };

/** Whether Ctrl+Z would take back a fare change now (the Money tab's undo button). */
export const canUndoFares = computed(() => {
  void version.value; void editsSent.value; void moneyState.value;
  return !!nextUndo();
});

/** The player's fare change: sent, and kept as a step to undo. */
export function changeFares(nodes: FareNode[]) {
  fresh();
  const from = sent && performance.now() - sent.t < 3000 ? sent.nodes : moneyState.peek()?.fares.nodes;
  const to = tidyNodes(nodes);
  if (from && !same(from, to)) {
    undos.push({ from, to });
    redos.length = 0;
    undone.length = 0;
    version.value++;
  }
  return send(to);
}

export function undo() {
  const s = nextUndo();
  if (!s) {
    undone.push("sim");
    return edit({ op: "undo" }, true);
  }
  undos.pop();
  redos.push(s);
  undone.push("fare");
  version.value++;
  return send(s.from);
}

export function redo() {
  fresh();
  const kind = undone.pop();
  const s = redos[redos.length - 1];
  if (kind !== "fare" || !s || !isNow(s.from)) return edit({ op: "redo" }, true);
  redos.pop();
  undos.push(s);
  version.value++;
  return send(s.to);
}
