// What each track type may follow on the real network (T-3069, DECISIONS 2026-11-17): one table
// for the router (workers/realRoute.worker.ts) and the layer drawn while snapping
// (map/realNetwork.ts), so what is drawn is what can be snapped to.

/** The graph's kind codes (pipeline/realgraph.py KIND), by the name the tiles' `k` carries. */
export const REAL_KINDS = ["rail", "light_rail", "subway", "tram", "monorail", "funicular", "narrow_gauge", "heritage", "freight", "abandoned"] as const;

export type SnapMode = "rail" | "tram";

/** Per track type, each kind's family for the router, -1 where it is not followed. A route keeps
 * to the family of the track nearest its first click, the others at three times their length.
 * Train track never follows tram rail; tram track follows tram and light rail, other rail at the
 * other-family cost (tram-trains and street running on rail tracks), never a monorail. Both follow freight
 * and abandoned track, each its own family (T-3495). */
export const FOLLOW: Record<SnapMode, readonly number[]> = {
  //     rail lrt sub tram mono fun ng her frt abn
  rail: [0, 0, 1, -1, 2, 3, 4, 5, 6, 7],
  tram: [1, 0, 2, 0, -1, 3, 4, 5, 6, 7],
};

/** The tile kinds a track type follows. */
export const followedKinds = (mode: SnapMode): string[] => REAL_KINDS.filter((_, k) => FOLLOW[mode][k] >= 0);

/** The same tables for the Passenger setting: freight and abandoned track (kinds 8 and 9, in the graphs' extra files that
 * the router loads only under All) are never followed there. */
export const FOLLOW_PASSENGER: Record<SnapMode, readonly number[]> = {
  rail: FOLLOW.rail.map((f, k) => (k >= 8 ? -1 : f)),
  tram: FOLLOW.tram.map((f, k) => (k >= 8 ? -1 : f)),
};

/** What `mode` follows, with or without the extra kinds. */
export const followFor = (mode: SnapMode, all: boolean): readonly number[] => (all ? FOLLOW : FOLLOW_PASSENGER)[mode];

/** Whether a follow table is a tram one. */
export const isTramFollow = (f: readonly number[]) => f === FOLLOW.tram || f === FOLLOW_PASSENGER.tram;
