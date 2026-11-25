// Linking stations into one complex by hand (T-197, SPEC 8): the station inspector's "Link to
// another station" starts it, the next station clicked on the map is linked, and a linked pair is
// undone from the inspector. The links are saved with the network (the track model's
// `link_stations`); complexes take them in (game/complexes.ts), and so does demand, which lets
// riders change between any two stations of a complex (T-161). While a passage is being placed, each
// join drawn has a remove button (map/joinButtons.ts, T-4048): a passage's removes it, a join the
// distance rule makes keeps the stations on either side of it apart (`removeJoin`).

import { t } from "../i18n";
import { signal } from "@preact/signals";
import { BOX_TOUCH_M, LINK_MAX_M, nearEnough, pairKey, platformOffsets, polylineGap, straightOffsets } from "./complexes";
import { withBuildingLinks } from "./buildings";
import { groundDist, groundScale } from "./coords";
import { complexOf } from "./complexView";
import { distance, money, roughDistance } from "./format";
import { linkFarePlan } from "./fareZonePaint";
import type { Selection, StationId } from "./types";
import { ask } from "./ui";
import { edit, say, world } from "./world";
import { ahead, shown, type Show } from "./pending";
import type { EditOp } from "../workers/protocol";
import { passagePrice } from "./upgrades";

/** T-4146: the player's links as the panels show them, one still on its way to the world included */
const shownLinks = (): readonly [StationId, StationId][] => shown("links", "pairs", world.value?.save.links ?? []);
/** T-4170: a link added or taken away, in the panels' list at the click (game/editGuess.ts) */
export function linkGuess(op: EditOp): Show | null {
  if (op.op !== "link" && op.op !== "separate") return null;
  const st = world.peek()?.save.stations ?? [];
  const a = st.find((s) => s.num === op.a)?.id, b = st.find((s) => s.num === op.b)?.id;
  if (a === undefined || b === undefined) return null;
  const now = shownLinks();
  const pairs = op.op === "link" && op.on ? [...now, [a, b] as [StationId, StationId]] : now.filter(([x, y]) => !((x === a && y === b) || (x === b && y === a)));
  return { fields: new Map([["links", { pairs }]]) };
}

/** The station a link starts from while the player picks the other one; null otherwise. */
export const linking = signal<StationId | null>(null);

export function startLink(id: StationId) {
  linking.value = id;
}

/** Stop picking; true if a link was being made (Esc takes one thing at a time). */
export function cancelLink(): boolean {
  if (linking.peek() === null) return false;
  linking.value = null;
  return true;
}

/** Why `a` and `b` cannot be linked, or null if they can. */
export function linkRefusal(a: StationId, b: StationId): string | null {
  const st = world.peek()?.save.stations ?? [];
  const sa = st.find((s) => s.id === a), sb = st.find((s) => s.id === b);
  if (!sa || !sb) return t("links.pickStation");
  if (a === b) return t("links.pickAnother");
  if (complexOf(a).includes(b)) return t("links.alreadyComplex", { name: sb.name });
  const d = groundDist(sa.x, sa.y, sb.x, sb.y);
  if (d > LINK_MAX_M) return t("links.tooFar", { name: sb.name, away: distance(d), reach: roughDistance(LINK_MAX_M) });
  return null;
}

/** A map click while linking: the station picked is linked with the one the link started from,
 * which stays selected; a click on anything else stops linking. A refused pick (too far, already
 * one complex) says why and keeps picking. */
export async function linkPicked(picked: Selection): Promise<void> {
  const from = linking.peek();
  if (from === null) return;
  // T-4079: a bus stop is no station (its line's stops move with every edit), so it takes no passage
  if (picked?.kind === "busStop") {
    say(t("links.busStop"), true);
    return; // still picking
  }
  if (picked?.kind !== "station") {
    linking.value = null;
    return;
  }
  const why = linkRefusal(from, picked.station);
  if (why) {
    say(why, true);
    return; // still picking
  }
  linking.value = null;
  const st = world.peek()?.save.stations ?? [];
  const sa = st.find((s) => s.id === from)!, sb = st.find((s) => s.id === picked.station)!;
  // a complex is in one fare area: stations in different ones ask first (Cancel leaves them unlinked)
  const fares = linkFarePlan(sa.id, sb.id);
  if (fares?.ask && !(await new Promise<boolean>((yes) => {
    ask.value = { text: t("fareZonePaint.moveAsk"), yes: t("fareZonePaint.move"), no: t("toolNotes.cancel"), run: () => yes(true), stay: () => yes(false) };
  }))) return;
  // T-4146: listed and paid at the click; a passage between constructed stations is paid at once (T-1281)
  const cost = sa.built && sb.built ? passagePrice(groundDist(sa.x, sa.y, sb.x, sb.y)) : 0;
  const a = await ahead({ cost }).send(() => edit({ op: "link", a: sa.num, b: sb.num, on: true }));
  if (a.ok && fares) await fares.apply();
  if (a.ok) say(a.charge > 0 ? t("links.built", { a: sa.name, b: sb.name, cost: money(a.charge * 1e6) }) : t("links.added", { a: sa.name, b: sb.name }));
}

/** The station panel's Remove on a passage: the map's remove button on it (`removeJoin`), so either one means the two
 * stations no longer connect there (T-4048, DECISIONS). */
export function unlink(a: StationId, b: StationId) {
  return removeJoin(a, b);
}

/** T-4048: every join the map draws (map/joinButtons.ts sets it): passages, the distance rule's bridges and BRT
 * stations' dashed joins, hidden ones included */
export const drawnJoins = { list: (): readonly (readonly [StationId, StationId])[] => [] };

/** T-4048: the pairs the distance rule joins between the stations on `a`'s side of the join drawn from `a` to `b`
 * and the rest of their complex. `joins`: every join the map draws in it (passages, the distance rule's
 * bridges, a BRT station's dashed joins); besides them, a passage, a station building and boxes that touch
 * (no bridge drawn) keep stations on one side. Only the pair itself where they stay joined some other way. */
export function cutPairs(a: StationId, b: StationId, joins: readonly (readonly [StationId, StationId])[]): [StationId, StationId][] {
  const w = world.peek();
  if (!w) return [];
  const members = complexOf(a);
  if (!members.includes(b)) return [];
  const byId = new Map(w.save.stations.map((s) => [s.id, s]));
  const geo = members.map((id) => byId.get(id)).filter((s) => s !== undefined);
  const at = new Map(geo.map((s, i) => [s.id, i]));
  const up = geo.map((_, i) => i);
  const top = (i: number): number => (up[i] === i ? i : (up[i] = top(up[i])));
  const join = (x: StationId, y: StationId) => {
    const i = at.get(x), j = at.get(y);
    if (i !== undefined && j !== undefined) up[top(i)] = top(j);
  };
  const clicked = pairKey(a, b);
  for (const [x, y] of joins) if (pairKey(x, y) !== clicked) join(x, y);
  for (const [x, y] of withBuildingLinks(w.save.links, w.save.buildings, w.save.stations, groundScale)) if (pairKey(x, y) !== clicked) join(x, y);
  const k = geo.map((s) => groundScale(s.y)), seg = geo.map((s, i) => straightOffsets(s, k[i])), pts = geo.map((s, i) => platformOffsets(s, k[i], seg[i]));
  for (let i = 0; i < geo.length; i++)
    for (let j = i + 1; j < geo.length; j++)
      if (polylineGap(geo[i].x, geo[i].y, pts[i], geo[j].x, geo[j].y, pts[j]) <= BOX_TOUCH_M * (k[i] + k[j]) / 2) join(geo[i].id, geo[j].id);
  const apart = new Set((w.save.apart ?? []).map(([x, y]) => pairKey(x, y)));
  const near = (i: number, j: number) => !apart.has(pairKey(geo[i].id, geo[j].id)) && nearEnough(geo[i], pts[i], seg[i], geo[j], pts[j], seg[j], (k[i] + k[j]) / 2);
  const ia = at.get(a)!, ib = at.get(b)!;
  if (top(ia) === top(ib)) return near(ia, ib) ? [[a, b]] : [];
  const out: [StationId, StationId][] = [];
  for (let i = 0; i < geo.length; i++)
    for (let j = i + 1; j < geo.length; j++)
      if ((top(i) === top(ia)) !== (top(j) === top(ia)) && near(i, j)) out.push([geo[i].id, geo[j].id]);
  return out;
}

/** T-4048: remove the join drawn between stations `a` and `b` (`cutPairs`), free and undoable: the passage between
 * them if there is one, and the distance rule's join across it. */
export async function removeJoin(a: StationId, b: StationId, joins = drawnJoins.list()) {
  const st = world.peek()?.save.stations ?? [];
  const num = new Map(st.map((s) => [s.id, s.num]));
  const sa = st.find((s) => s.id === a), sb = st.find((s) => s.id === b);
  if (!sa || !sb) return;
  const passage = shownLinks().some(([x, y]) => pairKey(x, y) === pairKey(a, b));
  const apart = cutPairs(a, b, joins).flatMap(([x, y]) => [num.get(x)!, num.get(y)!]);
  if (!passage && !apart.length) return;
  const r = await edit({ op: "separate", a: sa.num, b: sb.num, apart });
  if (r.ok) say(t(passage ? "links.removed" : "links.separated", { a: sa.name, b: sb.name }));
}

/** The player's links with an end in `members` (a station or a complex). */
export function linksAmong(members: readonly StationId[]): [StationId, StationId][] {
  const inside = new Set(members);
  return shownLinks().filter(([a, b]) => inside.has(a) || inside.has(b));
}
