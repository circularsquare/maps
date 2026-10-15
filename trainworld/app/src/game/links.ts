// Linking stations into one complex by hand (T-197, SPEC 8): the station inspector's "Link to
// another station" starts it, the next station clicked on the map is linked, and a linked pair is
// undone from the inspector. The links are saved with the network (the track model's
// `link_stations`); complexes take them in (game/complexes.ts), and so does demand, which lets
// riders change between any two stations of a complex (T-161). Stations the distance rule groups
// cannot be split: only the player's own links can be undone (DECISIONS 2026-10-12).

import { signal } from "@preact/signals";
import { LINK_MAX_M } from "./complexes";
import { groundDist } from "./coords";
import { complexOf } from "./complexView";
import { distance, roughDistance } from "./format";
import type { Selection, StationId } from "./types";
import { edit, say, world } from "./world";

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
  if (!sa || !sb) return "Pick a station to link with.";
  if (a === b) return "Pick another station to link with.";
  if (complexOf(a).includes(b)) return `${sb.name} is already in this complex.`;
  const d = groundDist(sa.x, sa.y, sb.x, sb.y);
  if (d > LINK_MAX_M) return `${sb.name} is ${distance(d)} away. Stations can be linked up to ${roughDistance(LINK_MAX_M)} apart.`;
  return null;
}

/** A map click while linking: the station picked is linked with the one the link started from,
 * which stays selected; a click on anything else stops linking. A refused pick (too far, already
 * one complex) says why and keeps picking. */
export async function linkPicked(picked: Selection): Promise<void> {
  const from = linking.peek();
  if (from === null) return;
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
  const a = await edit({ op: "link", a: sa.num, b: sb.num, on: true });
  if (a.ok) say(`${sa.name} and ${sb.name} are one complex now. Riders change between them on foot.`);
}

/** Undo the player's link between two stations. */
export async function unlink(a: StationId, b: StationId) {
  const st = world.peek()?.save.stations ?? [];
  const sa = st.find((s) => s.id === a), sb = st.find((s) => s.id === b);
  if (!sa || !sb) return;
  const r = await edit({ op: "link", a: sa.num, b: sb.num, on: false });
  if (r.ok) say(`${sa.name} and ${sb.name} are no longer linked.`);
}

/** The player's links with an end in `members` (a station or a complex). */
export function linksAmong(members: readonly StationId[]): [StationId, StationId][] {
  const inside = new Set(members);
  return (world.value?.save.links ?? []).filter(([a, b]) => inside.has(a) || inside.has(b));
}
