// T-2124, T-2179: the achievements view, opened from the medal at the top of the Cities tab
// (ui/Achievements.tsx). Each tiered achievement is its name, a medal or an empty place for each tier
// under it, and at the right a ring in the achievement's colour filling from zero toward the next tier
// with the figure in the middle. Then the firsts, a gold medal or an empty place each, the day at the right.

import { demandFigures, FIRSTS, networkFigures, short, TIERED, tierKey, tierText, when, type Tiered } from "../../game/achievements";
import { dayOf } from "../../game/format";
import { moneyState } from "../../game/money";
import type { Achieved } from "../../workers/saveFile";
import { Medal, METALS } from "../Achievements";
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

/** The ring: `p` of the way round from the top, clockwise. */
function Ring({ p }: { p: number }) {
  return (
    <svg class="ach-ring" viewBox="0 0 64 64" aria-hidden="true">
      <circle class="track" cx="32" cy="32" r="28" />
      {p > 0 && <circle class="fill" cx="32" cy="32" r="28" pathLength={100} stroke-dasharray={`${Math.min(1, p) * 100} 100`} transform="rotate(-90 32 32)" />}
    </svg>
  );
}

function TierRow({ t, got }: { t: Tiered; got: Record<string, Achieved> }) {
  const f = figureOf(t.id);
  const label = (x: number) => (t.id === "cities" ? String(x) : short(x));
  const next = f ? t.tiers.find((x) => f.v < x) : undefined;
  const p = !f ? 0 : next === undefined ? 1 : f.v / next;
  return (
    <li class={"ach-tier r-" + t.id}>
      <div class="ach-left">
        <span class="ach-name">{t.name}</span>
        <span class="ach-medals">
          {t.tiers.map((x, k) => {
            const a = got[tierKey(t.id, x)];
            return a
              ? <span class="ach-slot" data-tip={`${tierText(t.id, x, a)}, reached on ${when(a.at)}`}><Medal metal={METALS[k]} of={t.id} /></span>
              : <span class="ach-slot empty" data-tip={tierText(t.id, x)} />;
          })}
        </span>
      </div>
      <div class="ach-dial" data-tip={f?.of}>
        <Ring p={p} />
        <span class="ach-fig">
          <b class="num">{f ? label(f.v) : "–"}</b>
          {f && next !== undefined && <span class="muted num">/{label(next)}</span>}
        </span>
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
      <ul class="rows ach-firsts">
        {FIRSTS.map((f) => {
          const a = got[f.id];
          return (
            <li class={a ? "" : "muted"} data-tip={a ? `${f.text(a)}\nReached on ${when(a.at)}` : f.rule()}>
              <span class={"ach-slot" + (a ? "" : " empty")}>{a && <Medal metal="gold" of="first" />}</span>
              <span class="grow">{f.name}</span>
              {a && <span class="val">day {dayOf(Math.floor(a.at / 60))}</span>}
            </li>
          );
        })}
      </ul>
    </>
  );
}
