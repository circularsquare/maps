// T-4243: a running train's speed on a round dial (ui-inspectors.md "Trains"): the rim in the Speed view's colours,
// a needle turning around a centre circle and the figure in it, in the speed's colour. The panel re-renders once a
// game minute; the dial moves with the train from its own frame loop, which writes to the page only when the needle
// or the figure shown changes and stops while the clock does.

import { useLayoutEffect, useRef } from "preact/hooks";
import { clock } from "../../game/clock";
import { speed, speedParts } from "../../game/format";
import { legibleOn, SPEED_STOPS, speedColour } from "../../game/palette";
import { display, shade } from "../../game/ui";
import { t } from "../../i18n";
import { live } from "../../map/live";
import { profileTop, profileV } from "../../map/network";
import { LAG_DECAY } from "../../map/trainGuard";

/** The dial, CSS px: its width, the rim's outer and inner radius, the centre circle's and the needle's reach. */
const D = 68, C = D / 2, RIM_OUT = C, RIM_IN = C - 3, HUB = 16, TIP = RIM_IN - 3;
/** Degrees the rim sweeps, from its slow end at the lower left (degrees clockwise from east), the gap at the bottom. */
const SWEEP = 270, START = 135;
/** The unit's baseline under the centre circle, and the dial's height, down to the unit, px. */
const UNIT_Y = C + HUB + 12, H = UNIT_Y + 3;
const LAST = SPEED_STOPS.length - 1;
/** The speed at the rim's end, km/h: high-speed track's; the rim is linear up to it (DECISIONS, Map and UI). */
const TOP = SPEED_STOPS[LAST][0];
/** Where `kmh` sits along the rim, 0-1. */
const dialAt = (kmh: number): number => Math.min(1, Math.max(0, kmh / TOP));
/** The rim: each speed colour at its own speed, blended between as `speedColour` blends them. */
const RIM = `conic-gradient(from ${START + 90}deg, ${SPEED_STOPS.map(([v, c]) => `${c} ${dialAt(v) * SWEEP}deg`).join(", ")}, transparent 0)`;

/** A point `r` from the centre at `deg` along the rim. */
const rimPoint = (deg: number, r: number): [number, number] => {
  const a = ((START + deg) * Math.PI) / 180;
  return [C + Math.cos(a) * r, C + Math.sin(a) * r];
};

export function Speedometer({ profile, dep }: { profile: number; dep: number }) {
  const needle = useRef<SVGLineElement>(null), fig = useRef<SVGTextElement>(null), redraw = useRef<(() => void) | null>(null);
  const us = display.value.usUnits, dark = shade.value === "dark";
  const n = live.overlay?.renderer.net;
  const top = n ? profileTop(n, profile) * 3.6 : 0;
  useLayoutEffect(() => {
    // the trip's place in the renderer's buffer (found again when the buffer is replaced), and the guard's last two
    // looks at its lag (T-1965): a held train is drawn later and later in its run as it stands, and makes the lag up
    // drawn faster, so the needle follows the train as drawn: its timetable speed times one minus the lag's rate
    let trips: Float32Array | null = null, epoch = NaN, i = -1;
    let seenAt = NaN, seenLag = 0, step = 0, rate = 0;
    let angle = NaN, shown = "", raf = 0;
    const draw = () => {
      const r = live.overlay?.renderer, net = r?.net;
      if (!r || !net) return;
      if (r.trips !== trips || r.tripsEpoch !== epoch) {
        trips = r.trips;
        epoch = r.tripsEpoch;
        i = r.tripIndex(profile, dep - epoch);
        seenAt = NaN;
        rate = 0;
      }
      const now = clock.now(), lag = r.tripLag(i), g = r.guard.lastTime;
      if (g !== seenAt) {
        rate = g > seenAt ? (lag - seenLag) / (g - seenAt) : 0;
        step = g > seenAt ? g - seenAt : 0;
        seenAt = g;
        seenLag = lag;
      } else if (now - seenAt > 4 * step) rate = 0; // the guard has stopped looking (trains not drawn)
      const kmh = profileV(net, profile, now - dep - lag) * 3.6 * Math.min(1 + LAG_DECAY, Math.max(0, 1 - rate));
      const a = Math.round(dialAt(kmh) * SWEEP * 4) / 4;
      if (a !== angle && needle.current) needle.current.setAttribute("transform", `rotate(${START + a} ${C} ${C})`);
      angle = a;
      const [v] = speedParts(kmh);
      if (v !== shown && fig.current) {
        fig.current.textContent = v;
        fig.current.style.fill = legibleOn(speedColour(kmh), dark);
      }
      shown = v;
    };
    const frame = () => {
      raf = 0;
      draw();
      if (clock.ticking) raf = requestAnimationFrame(frame);
    };
    frame();
    redraw.current = draw;
    const off = clock.onChange(() => { if (!raf) frame(); });
    return () => { off(); cancelAnimationFrame(raf); redraw.current = null; };
  }, [profile, dep, us, dark]);
  // the panel's own re-renders (an edit, a stopped clock's minute) redraw it too
  useLayoutEffect(() => redraw.current?.());
  const [x0, y0] = rimPoint(0, (RIM_OUT + RIM_IN) / 2), [x1, y1] = rimPoint(SWEEP, (RIM_OUT + RIM_IN) / 2);
  return (
    <div class="speedo" style={{ width: D + "px", height: H + "px" }} data-tip={top > 0 ? t("trainInspector.topSpeed", { speed: speed(top) }) : undefined}>
      <div class="speedo-rim" style={{ width: D + "px", height: D + "px", background: RIM }} />
      <svg width={D} height={H} viewBox={`0 0 ${D} ${H}`}>
        <circle class="bg" cx={C} cy={C} r={RIM_IN} />
        {/* round ends on the rim */}
        <circle cx={x0} cy={y0} r={(RIM_OUT - RIM_IN) / 2} fill={SPEED_STOPS[0][1]} />
        <circle cx={x1} cy={y1} r={(RIM_OUT - RIM_IN) / 2} fill={SPEED_STOPS[LAST][1]} />
        <line ref={needle} class="speedo-needle" x1={C} y1={C} x2={C + TIP} y2={C} transform={`rotate(${START} ${C} ${C})`} />
        <circle class="speedo-hub" cx={C} cy={C} r={HUB} />
        <text ref={fig} class="speedo-fig" x={C} y={C} />
        <text class="speedo-unit" x={C} y={UNIT_Y}>{speedParts(0)[1]}</text>
      </svg>
    </div>
  );
}
