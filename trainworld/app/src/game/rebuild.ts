// T-3059: rebuilding constructed track in place. The Select tool's pick of built pieces gets a level, a
// track type and a track count to choose, a Rebuild button and a question; the track model (`TrackApi::rebuild_quote`,
// `rebuild_pieces`) changes the same edges and stations to the new settings, so lines and names stay, and
// charges the full price of what changed (nothing for double track made single, T-3321). The pieces it would refuse are named in the quote and drawn in red
// on the map (map/blueprintBox.ts) before anything is spent. Main thread only, not saved.

import { computed, effect, signal } from "@preact/signals";
import { queryClock } from "../workers/clockClient";
import { distance, levelLabel, money } from "./format";
import { bpPicked, pickedState } from "./blueprintPick";
import { heightRunLength, noteHeight } from "./deleteHint";
import type { Level } from "./types";
import { ask } from "./ui";
import { edit, say, world } from "./world";
import { describeIssues } from "./issues";
import { t, tList } from "../i18n";

export type RebuildKind = "regular" | "hsr" | "tram";

/** The level and track type chosen for the built pick; null keeps what the pieces have. */
export const rebuildLevel = signal<Level | null>(null);
export const rebuildKind = signal<RebuildKind | null>(null);
/** The track count chosen for the built pick: 2 makes single track double (T-3296), 1 double track single (T-3321); null keeps. */
export const rebuildTracks = signal<1 | 2 | null>(null);

export interface RebuildQuote {
  /** what it would cost, US$M: the full price of what changes, built new */
  price: number;
  /** the edges and stations that change */
  changed: number[];
  stations: number;
  /** picked edges the track model would refuse */
  refused: number[];
  /** T-3321: why double track stays double, in words ("" for no such reason) */
  why: string;
  /** of those, the ones refused for their own sake, drawn red; the rest would only change family alone (T-3088) */
  red: number[];
  /** T-3241: track outside the pick that the ramps of the stations at its ends take: the edge, the node the
   * change starts at, and how far along the edge it reaches, m */
  beyond: { edge: number; from: number; m: number }[];
}

/** The quote for the pick and choices now; null with nothing chosen, or while it is being worked out. */
export const rebuildQuote = signal<RebuildQuote | null>(null);

const KIND = { regular: 1, hsr: 2, tram: 3 } as const;
const version = computed(() => world.value?.version);

let seq = 0;
effect(() => {
  const p = bpPicked.value, level = rebuildLevel.value, kind = rebuildKind.value, tracks = rebuildTracks.value;
  version.value;
  const mine = ++seq;
  if (!p.built || (level === null && kind === null && tracks === null)) {
    if (!p.built) { rebuildLevel.value = rebuildKind.value = rebuildTracks.value = null; }
    rebuildQuote.value = null;
    return;
  }
  void queryClock({ what: "rebuildQuote", edges: p.edges, stations: p.stations, level, kind: kind ? KIND[kind] : 0, tracks: tracks ?? 0 }).then((answer) => {
    if (mine !== seq || !answer) return;
    // the quote, then why pieces are refused (clock.worker.ts `rebuildQuoteAnswer`)
    const d = answer.subarray(1, 1 + answer[0]);
    const kinds = String.fromCharCode(...answer.subarray(1 + answer[0])).split("\n").filter((k) => k.startsWith("NoSingle:"));
    const refused = d[3], red = d[4], end = 5 + refused + d[1];
    const beyond: RebuildQuote["beyond"] = [];
    for (let k = end; k + 2 < d.length; k += 3) beyond.push({ edge: d[k], from: d[k + 1], m: d[k + 2] });
    rebuildQuote.value = {
      price: d[0],
      changed: Array.from(d.subarray(5 + refused, end), Number),
      stations: d[2],
      refused: Array.from(d.subarray(5, 5 + refused), Number),
      why: describeIssues(kinds),
      red: Array.from(d.subarray(5, 5 + red), Number),
      beyond,
    };
  });
});

/** What the pick would become, in words: "level +1 and high-speed track". */
export function rebuildTo(level: Level | null, kind: RebuildKind | null, tracks: 1 | 2 | null = null): string {
  return tList([
    level === null ? "" : t("blueprintPick.levelTo", { level: levelLabel(level) }),
    kind === null ? "" : t(kind === "hsr" ? "blueprintPick.toHsr" : kind === "tram" ? "blueprintPick.toTram" : "blueprintPick.toRegular"),
    tracks === null ? "" : t(tracks === 1 ? "blueprintPick.toSingle" : "blueprintPick.toDouble"),
  ].filter(Boolean));
}

/** The length of the edges in `ids`, m. */
export function lengthOf(ids: number[]): number {
  const w = world.peek();
  if (!w) return 0;
  const set = new Set(ids);
  return w.edges.reduce((s, e) => s + (set.has(e.id) ? e.lengthM : 0), 0);
}

/** The track a quote changes, m: the changed edges and the stretches beyond the pick (T-3241). */
export function quoteLength(q: RebuildQuote): number {
  return lengthOf(q.changed) + q.beyond.reduce((s, b) => s + b.m, 0);
}

/** The Rebuild button's hover: the length and what it becomes, the price, and how many pieces cannot change. */
export function rebuildTip(q: RebuildQuote | null): string {
  const s = pickedState.peek(), n = s.edges + s.stations;
  if (!q) return t("rebuild.pickTip");
  if (!q.changed.length && !q.stations) return q.refused.length ? [t("rebuild.cantChange", { all: true, one: n === 1, n: q.refused.length, total: n }), q.why].filter(Boolean).join("\n") : t("rebuild.pickTip");
  return [
    t("rebuild.tip", { length: distance(quoteLength(q)), to: rebuildTo(rebuildLevel.peek(), rebuildKind.peek(), rebuildTracks.peek()) }),
    // T-3321: double track made single, and nothing else, costs nothing
    q.price > 0 ? money(q.price * 1e6) : t("rebuild.free"),
    ...(q.refused.length ? [t("rebuild.cantChange", { all: false, one: false, n: q.refused.length, total: n }), q.why] : []),
  ].filter(Boolean).join("\n");
}

/** Ask, then rebuild the pick. A short purse, or terrain not loaded, is explained by the refusal; so
 * is a pick none of which can change, which is not asked about. */
export function rebuildPicked() {
  const p = bpPicked.peek(), q = rebuildQuote.peek(), level = rebuildLevel.peek(), kind = rebuildKind.peek(), tracks = rebuildTracks.peek();
  if (!p.built || !q || (!q.changed.length && !q.stations && !q.refused.length)) return;
  if (!q.changed.length && !q.stations) {
    void edit({ op: "rebuild", edges: p.edges, stations: p.stations, level, kind: kind ? KIND[kind] : 0, tracks: tracks ?? 0 }).then((a) => {
      if (!a.ok && !a.issues.length) say(t("rebuild.none"), true);
    });
    return;
  }
  const n = p.edges.length + p.stations.length, to = rebuildTo(level, kind, tracks);
  const what = tList([
    q.changed.length ? t("blueprintPick.trackOf", { length: distance(quoteLength(q)), built: true }) : "",
    q.stations ? t("blueprintPick.stationsPicked", { n: q.stations, built: true }) : "",
  ].filter(Boolean));
  const go = async () => {
    ask.value = null;
    const row = heightRunLength();
    const a = await edit({ op: "rebuild", edges: p.edges, stations: p.stations, level, kind: kind ? KIND[kind] : 0, tracks: tracks ?? 0 });
    if (!a.ok) {
      if (!a.issues.length) say(t("rebuild.none"), true);
      return;
    }
    rebuildLevel.value = rebuildKind.value = rebuildTracks.value = null;
    const skipped = a.skipped ?? 0;
    if (level !== null && skipped < n) noteHeight(row, n);
    say(skipped ? t("rebuild.doneSome", { done: n - skipped, n, to, skipped }) : t("rebuild.done", { n, to }));
  };
  ask.value = {
    text: q.price > 0 ? t("rebuild.question", { some: q.refused.length > 0, what, to, price: money(q.price * 1e6), n: q.refused.length, total: n })
      : t("rebuild.questionFree", { some: q.refused.length > 0, what, to, n: q.refused.length, total: n }),
    yes: t("rebuild.yes"),
    no: t("rebuild.no"),
    run: () => void go(),
  };
}
