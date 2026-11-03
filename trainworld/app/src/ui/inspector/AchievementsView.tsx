// T-2124: the achievements view, opened from the medal at the top of the Cities tab (ui/Achievements.tsx).
// Each tiered achievement is a bar of equal parts, one a tier, filling from the tier before to its own
// (the tiers grow 3-10x, so a bar to scale would show nothing below the top), with a tick per tier lit
// once reached. Then the firsts, a box each, ticked once earned, the day at the right.

import { demandFigures, FIRSTS, networkFigures, short, TIERED, tierKey, tierText, when, type Tiered } from "../../game/achievements";
import { dayOf } from "../../game/format";
import { moneyState } from "../../game/money";
import type { Achieved } from "../../workers/saveFile";
import { Icon } from "../widgets";
import "../achievements.css";

/** A tiered achievement's figure now and what has it; null while unknown. */
function figureOf(id: Tiered["id"]): { v: number; of?: string } | null {
  if (id === "cities") { const n = networkFigures.value; return n ? { v: n.cities } : null; }
  const d = demandFigures.value;
  if (!d) return null;
  if (id === "trips") return { v: d.trips };
  const x = id === "station" ? d.station : d.line;
  return x ? { v: x.v ?? 0, of: x.of } : { v: 0 };
}

function TierRow({ t, got }: { t: Tiered; got: Record<string, Achieved> }) {
  const f = figureOf(t.id), n = t.tiers.length;
  const label = (x: number) => (t.id === "cities" ? String(x) : short(x));
  let fill = 0;
  if (f) {
    const k = t.tiers.filter((x) => f.v >= x).length;
    const lo = k === 0 ? 0 : t.tiers[k - 1], hi = t.tiers[k];
    fill = k >= n ? 1 : (k + Math.max(0, Math.min(1, (f.v - lo) / (hi - lo)))) / n;
  }
  const next = f ? t.tiers.find((x) => f.v < x) : undefined;
  const text = !f ? "–" : (t.id === "cities" ? String(f.v) : short(f.v)) + (next !== undefined ? "/" + label(next) : "");
  return (
    <li class="ach-tier">
      <div class="line">
        <span class="grow">{t.name}</span>
        <span class="val num" data-tip={f?.of}>{text}</span>
      </div>
      <div class="ach-bar">
        <i class="ach-fill" style={{ width: fill * 100 + "%" }} />
        {t.tiers.map((x, k) => {
          const a = got[tierKey(t.id, x)];
          return <i class={"ach-tick" + (a ? " reached" : "")} style={{ left: ((k + 1) / n) * 100 + "%" }}
            data-tip={a ? `${tierText(t.id, x, a)}, reached on ${when(a.at)}` : tierText(t.id, x)} />;
        })}
      </div>
      <div class="ach-labels">
        {t.tiers.map((x, k) => <span class={got[tierKey(t.id, x)] ? "" : "muted"} style={{ left: ((k + 1) / n) * 100 + "%" }}>{label(x)}</span>)}
      </div>
    </li>
  );
}

export function AchievementsView() {
  const got = moneyState.value?.achievements ?? {};
  return (
    <>
      <div class="title">
        <span class="ach-mark"><Icon.medal /></span>
        <span class="name">Achievements</span>
      </div>
      <ul class="rows ach-tiers">
        {TIERED.map((t) => <TierRow t={t} got={got} />)}
      </ul>
      <div class="ach-firsts">
        <p class="muted small">Firsts</p>
        <ul class="rows">
          {FIRSTS.map((f) => {
            const a = got[f.id];
            return (
              <li class={a ? "" : "muted"} data-tip={a ? `${f.text(a)}\nReached on ${when(a.at)}` : f.rule()}>
                <span class="ach-box">{a && <Icon.check />}</span>
                <span class="grow">{f.name}</span>
                {a && <span class="val">day {dayOf(Math.floor(a.at / 60))}</span>}
              </li>
            );
          })}
        </ul>
      </div>
    </>
  );
}
