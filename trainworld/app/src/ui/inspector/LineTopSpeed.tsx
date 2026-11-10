// A rail line's top speed and its speed limit (T-2987): the row shows the fastest its trains reach on
// the route; its button adds a row with a stepper that caps them everywhere on it.

import { useEffect, useState } from "preact/hooks";
import { speed } from "../../game/format";
import { t } from "../../i18n";
import { edit } from "../../game/world";
import type { LineInput, LineStats } from "../../game/types";
import { Row, Stepper } from "../widgets";

const STEP = 10;
const MIN = 40;

export function LineTopSpeed({ l, st }: { l: LineInput; st: LineStats }) {
  // the limit as asked, until the world shows it (an edit takes a moment, and clicks come faster)
  const [asked, setAsked] = useState<number | null>(null);
  useEffect(() => { if (asked !== null && asked === st.limit) setAsked(null); }, [st.limit, asked]);
  const kindTop = st.fast ? 320 : 160;
  const limit = Math.min(asked ?? st.limit, kindTop);
  const reached = Math.max(0, ...st.stretches.map((s) => s.reached));
  if (!(reached > 0)) return null;
  const send = (kmh: number) => {
    setAsked(kmh);
    return edit({ op: "lineLimit", line: l.num, kmh }).then((a) => {
      if (!a.ok) setAsked(null);
      return a.ok;
    });
  };
  const toggle = () => void send(limit > 0 ? 0 : Math.max(MIN, Math.min(kindTop, Math.floor(reached / STEP) * STEP)));
  const step = (d: number) => {
    const to = Math.max(MIN, Math.min(kindTop, limit + d * STEP));
    return to === limit ? false : send(to);
  };
  return (
    <ul class="rows top-speed">
      <Row label={t("lineInspector.topSpeed")}>
        {speed(reached)}
        <button class={"btn text mini" + (limit > 0 ? " on" : "")} aria-pressed={limit > 0} onClick={toggle}>{t("lineInspector.limit")}</button>
      </Row>
      {limit > 0 && (
        <Row label={t("lineInspector.speedLimit")}>
          <Stepper wide label={t("lineInspector.speedLimit")} value={speed(limit)} onStep={step} />
        </Row>
      )}
    </ul>
  );
}
