// The lines related to a line (T-1168, notes/T-1168.md): the clock query `lineRelated`
// (`TrackApi.line_related`, sim/src/track/related.rs) decoded. One group per span another line
// runs beside the selected one, or per lone station where it crosses. Offsets are ground metres
// along each line's first run; strays and runs are ground metres.

export interface RelatedMeeting {
  complex: number;
  /** Offsets on the selected line and on the other. */
  sL: number;
  sO: number;
  /** Stop indices on each line, -1 where it passes. */
  stopL: number;
  stopO: number;
}

/** A stretch between two meetings the other line runs on its own track. */
export interface RelatedGap {
  sL0: number;
  sL1: number;
  /** How far its track strays from the selected line's, m. */
  stray: number;
  /** Its own length between the meetings, and the selected line's, m. */
  run: number;
  lStep: number;
}

export interface RelatedGroup {
  line: number;
  kind: "beside" | "crossing";
  /** Some of the span is on the same track. */
  shared: boolean;
  /** 1 where the other line runs the same way as the selected one along the span, -1 the other way. */
  dir: 1 | -1;
  sL0: number;
  sL1: number;
  sO0: number;
  sO1: number;
  /** The other line's stops before and after the span, in its own order. */
  stopsBefore: number;
  stopsAfter: number;
  /** The gap that split this group from the line's previous one: stray NaN when the line turned
   * back along the selected one; null for its first group. */
  split: { stray: number; run: number; lStep: number } | null;
  meetings: RelatedMeeting[];
  /** Shared-track stretches, as offsets on the selected line (from, to). */
  track: [number, number][];
  gaps: RelatedGap[];
  /** The other line's stops inside the span not at a meeting, at the nearest point of the
   * selected line. */
  own: { stop: number; sL: number; complex: number }[];
}

export function decodeRelated(v: ArrayLike<number>): RelatedGroup[] {
  const out: RelatedGroup[] = [];
  let i = 0;
  while (i + 17 <= v.length) {
    const [nm, nt, ng, no] = [v[i + 13], v[i + 14], v[i + 15], v[i + 16]];
    const g: RelatedGroup = {
      line: v[i], kind: v[i + 1] === 1 ? "beside" : "crossing", shared: v[i + 2] === 1, dir: v[i + 3] < 0 ? -1 : 1,
      sL0: v[i + 4], sL1: v[i + 5], sO0: v[i + 6], sO1: v[i + 7], stopsBefore: v[i + 8], stopsAfter: v[i + 9],
      split: Number.isNaN(v[i + 11]) ? null : { stray: v[i + 10], run: v[i + 11], lStep: v[i + 12] },
      meetings: [], track: [], gaps: [], own: [],
    };
    i += 17;
    for (let k = 0; k < nm; k++, i += 5) g.meetings.push({ complex: v[i], sL: v[i + 1], sO: v[i + 2], stopL: v[i + 3], stopO: v[i + 4] });
    for (let k = 0; k < nt; k++, i += 2) g.track.push([v[i], v[i + 1]]);
    for (let k = 0; k < ng; k++, i += 5) g.gaps.push({ sL0: v[i], sL1: v[i + 1], stray: v[i + 2], run: v[i + 3], lStep: v[i + 4] });
    for (let k = 0; k < no; k++, i += 3) g.own.push({ stop: v[i], sL: v[i + 1], complex: v[i + 2] });
    out.push(g);
  }
  return out;
}
