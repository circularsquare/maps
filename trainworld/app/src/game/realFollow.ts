// What each track type may follow on the real network (T-3069, DECISIONS 2026-11-17): one table
// for the router (workers/realRoute.worker.ts) and the layer drawn while snapping
// (map/realNetwork.ts), so what is drawn is what can be snapped to.

/** The graph's kind codes (pipeline/realgraph.py KIND), by the name the tiles' `k` carries. */
export const REAL_KINDS = ["rail", "light_rail", "subway", "tram", "monorail", "funicular", "narrow_gauge", "heritage"] as const;

export type SnapMode = "rail" | "tram";

/** Per track type, each kind's family for the router, -1 where it is not followed. A route keeps
 * to the family of the track nearest its first click, the others at three times their length.
 * Train track never follows tram rail; tram track follows tram and light rail, other rail at the
 * other-family cost (tram-trains and street running on rail tracks), never a monorail. */
export const FOLLOW: Record<SnapMode, readonly number[]> = {
  //     rail lrt sub tram mono fun ng her
  rail: [0, 0, 1, -1, 2, 3, 4, 5],
  tram: [1, 0, 2, 0, -1, 3, 4, 5],
};

/** The tile kinds a track type follows. */
export const followedKinds = (mode: SnapMode): string[] => REAL_KINDS.filter((_, k) => FOLLOW[mode][k] >= 0);
