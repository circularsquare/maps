// A piece of track's top speed and its speed limit (T-3405), as a line's (LineTopSpeed, T-2987): the row
// shows the top speed of the kind of track; its button adds a row with a stepper that holds trains to less.

import { useEffect, useState } from "preact/hooks";
import { speed } from "../../game/format";
import { t } from "../../i18n";
import { edit } from "../../game/world";
import { TRACK_SPECS } from "../../game/trackKinds";
import type { TrackEdge } from "../../game/types";
import { Row, Stepper } from "../widgets";

const STEP = 10;
const MIN = 40;

export function PieceLimit({ e }: { e: TrackEdge }) {
  // the limit as asked, until the world shows it (an edit takes a moment, and clicks come faster)
  const [asked, setAsked] = useState<number | null>(null);
  const have = e.limit ?? 0;
  useEffect(() => { if (asked !== null && asked === have) setAsked(null); }, [have, asked]);
  const kindTop = e.kind ? TRACK_SPECS[e.kind === 2 ? "streetTram" : "tram"].vCapKmh : e.hsr ? 320 : 160;
  const limit = Math.min(asked ?? have, kindTop);
  const send = (kmh: number) => {
    setAsked(kmh);
    return edit({ op: "pieceLimit", edges: [e.id], kmh }).then((a) => {
      if (!a.ok) setAsked(null);
      return a.ok;
    });
  };
  const toggle = () => void send(limit > 0 ? 0 : kindTop);
  const step = (d: number) => {
    const to = Math.max(MIN, Math.min(kindTop, limit + d * STEP));
    return to === limit ? false : send(to);
  };
  // the hover says what the kind of track allows; with a limit on, the row below says otherwise
  const tip = e.kind || limit > 0 ? undefined : t(e.hsr ? "trackInspector.speedHsr" : "trackInspector.speedNormal");
  return (
    <>
      <li data-tip={tip}>
        <span class="grow">{t("trackInspector.topSpeed")}</span>
        <span class="val">
          {speed(kindTop)}
          <button class={"btn text mini" + (limit > 0 ? " on" : "")} aria-pressed={limit > 0} onClick={toggle}>{t("trackInspector.limit")}</button>
        </span>
      </li>
      {limit > 0 && (
        <Row label={t("trackInspector.speedLimit")}>
          <Stepper wide label={t("trackInspector.speedLimit")} value={speed(limit)} onStep={step} />
        </Row>
      )}
    </>
  );
}
