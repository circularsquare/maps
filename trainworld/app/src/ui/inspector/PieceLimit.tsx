// A piece of track's top speed and its speed limit (T-3405), as a line's (LineTopSpeed, T-2987): the toggle
// opens a stepper that holds trains to less. The selection bar sets it on every picked piece at once (T-3525,
// `PickLimit`), one piece included (T-3746); the track panel shows it.

import { useEffect, useState } from "preact/hooks";
import { speed } from "../../game/format";
import { t } from "../../i18n";
import { edit } from "../../game/world";
import { withShown } from "../../game/pieceAhead";
import { TRACK_SPECS } from "../../game/trackKinds";
import type { TrackEdge } from "../../game/types";
import { Row } from "../widgets";

import { LIMIT_MIN as MIN, nextLimit } from "../../game/pieceLimit";

/** The top speed of a piece's kind of track, km/h (the sim's `TrackKind::v_cap`). */
export const kindTop = (e: TrackEdge) => (e.kind ? TRACK_SPECS[e.kind === 4 ? "busway" : e.kind === 2 ? "streetTram" : "tram"].vCapKmh : e.hsr ? 320 : 160);

/** What a piece keeps when `kmh` is asked (the sim's `piece_limit`): a limit at or over its top is none (T-3525). */
const kept = (e: TrackEdge, kmh: number) => (kmh <= 0 || kmh >= kindTop(e) ? 0 : Math.max(MIN, kmh));

/** The limit control's state for pieces of track (no crossovers): the speed every one runs at (its limit,
 * else its kind's top), null when they differ; whether any has a limit or the stepper was opened; the
 * Limit button; and a stepper step (`nextLimit`), which sets every piece to one speed in one edit. */
export function usePieceLimit(edges: readonly TrackEdge[]) {
  // the limit as asked, until the world shows it (an edit takes a moment, and clicks come faster)
  const [asked, setAsked] = useState<number | null>(null);
  // T-3525: the stepper is open with no limit stored at the top speed, since a limit there is none
  const [open, setOpen] = useState(false);
  const ids = edges.map((e) => e.id).join(",");
  useEffect(() => { setOpen(false); setAsked(null); }, [ids]);
  const shown = edges.map((e) => (asked === null ? e.limit ?? 0 : kept(e, asked)));
  const settled = asked !== null && edges.every((e, i) => (e.limit ?? 0) === shown[i]);
  useEffect(() => { if (settled) setAsked(null); }, [settled]);
  const speeds = edges.map((e, i) => shown[i] || kindTop(e));
  const value = speeds.every((v) => v === speeds[0]) ? speeds[0] ?? null : null;
  const top = Math.max(0, ...edges.map(kindTop));
  const limited = shown.some((l) => l > 0);
  const send = (kmh: number) => {
    setAsked(kmh);
    return edit({ op: "pieceLimit", edges: edges.map((e) => e.id), kmh }).then((a) => {
      if (!a.ok) setAsked(null);
      return a.ok;
    });
  };
  const toggle = () => {
    if (limited || open) {
      setOpen(false);
      if (limited) void send(0);
    } else setOpen(true);
  };
  const step = (d: number) => {
    const to = nextLimit(speeds, edges.map(kindTop), d);
    if (to === null) return false;
    // stepping up to the top takes the limit off; the stepper stays until the toggle closes it
    setOpen(true);
    return send(to);
  };
  return { value, top, on: limited || open, toggle, step };
}

/** The track panel's top speed and the piece's limit, if any; T-3746: set in the selection bar. */
export function PieceLimit({ e: edge }: { e: TrackEdge }) {
  const e = withShown([edge])[0];
  // the hover says what the kind of track allows; with a limit on, the row below says otherwise
  const tip = e.kind || e.limit ? undefined : t(e.hsr ? "trackInspector.speedHsr" : "trackInspector.speedNormal");
  return (
    <>
      <li data-tip={tip}>
        <span class="grow">{t("trackInspector.topSpeed")}</span>
        <span class="val">{speed(kindTop(e))}</span>
      </li>
      {!!e.limit && <Row label={t("trackInspector.speedLimit")}>{speed(e.limit)}</Row>}
    </>
  );
}
