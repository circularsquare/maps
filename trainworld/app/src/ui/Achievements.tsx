// T-2124, T-2179: the medal at the top of the Cities tab that opens the achievements view (the
// leaderboards' trophy beside it), the medal itself, and the toasts at the map's bottom left as steps
// are earned (in `#tools`, above the tool notes). A toast rises, stays 5 s (longer under the pointer),
// fades; up to three stacked, one on a phone, newest at the bottom; more than three waiting fold into
// one. They wait while a menu, the saved games or a confirmation is up, or the tab is hidden. A click
// opens the view. Steps recorded on load (an older save's) never toast.

import { effect, signal } from "@preact/signals";
import { useEffect, useState } from "preact/hooks";
import { earned, FIRSTS, STEPS, TIERED, tierText, type TieredId } from "../game/achievements";
import { accountOpen, accountState, boardsOpen } from "../game/account";
import { choosing } from "../game/loading";
import { moneyState } from "../game/money";
import { ask, selection, tab } from "../game/ui";
import { gamesLoaded } from "../workers/clockClient";
import type { Achieved } from "../workers/saveFile";
import { LeaderboardButton } from "./Leaderboard";
import { DiagramButton } from "./DiagramButton";
import { savesOpen } from "./SavesPanel";
import { Icon } from "./widgets";
import "./achievements.css";

export function openAchievements() {
  tab.value = "cities";
  selection.value = { kind: "achievements" };
}

export function AchievementsButton() {
  const m = moneyState.value;
  const sandbox = !m || m.mode === "sandbox";
  const on = selection.value?.kind === "achievements";
  // T-2161: the network diagram's button at the row's left, in every mode (Anita, 2026-11-03)
  if (sandbox && accountState.value !== "on") return <div class="ach-head"><DiagramButton /></div>;
  return (
    <div class="ach-head">
      <DiagramButton />
      <LeaderboardButton />
      {!sandbox && <button class={"btn text ach-open" + (on ? " on" : "")} data-tip="Achievements" aria-label="Achievements" aria-pressed={on}
        onClick={() => (on ? (selection.value = null) : openAchievements())}>
        <Icon.medal /><span class="num">{earned.value}/{STEPS}</span>
      </button>}
    </div>
  );
}

/** A tier's metal by its place in the ladder; a first is gold. */
export const METALS = ["bronze", "silver", "gold", "platinum", "emerald"] as const;
export type Metal = (typeof METALS)[number];
/** whose colour the ribbon takes: a tiered achievement's, or the firsts' */
export type RibbonOf = TieredId | "first";

/** T-2179: a medal, its disc the tier's metal and its ribbon the achievement's colour (achievements.css). */
export function Medal({ metal, of }: { metal: Metal; of: RibbonOf }) {
  return (
    <svg class={"ach-medal m-" + metal + " r-" + of} viewBox="0 0 16 22" aria-hidden="true">
      <path class="rib back" d="M14 0h-4L5.6 9.6l3 1.4z" />
      <path class="rib" d="M2 0h4l4.4 9.6-3 1.4z" />
      <circle class="face" cx="8" cy="15" r="6.2" />
      <circle class="inner" cx="8" cy="15" r="3.9" />
    </svg>
  );
}

interface Toast { id: number; title: string; line?: string; medal?: { metal: Metal; of: RibbonOf } }
const HOLD_MS = 5000, FADE_MS = 1200, GAP_MS = 1000, MAX_SHOWN = 3;
const PHONE = matchMedia("(max-width: 600px) and (min-height: 560px)");

const waiting = signal<Toast[]>([]);
let nextId = 1, lastIn = -Infinity;

/** The toasts for steps just earned: the highest tier of each tiered achievement, and each first. */
function toastsFor(keys: string[], got: Record<string, Achieved>): Toast[] {
  const out: Toast[] = [];
  for (const t of TIERED) {
    const top = t.tiers.filter((x) => keys.includes(`${t.id}:${x}`)).at(-1);
    if (top !== undefined) out.push({ id: nextId++, title: t.name, line: tierText(t.id as TieredId, top, got[`${t.id}:${top}`], true),
      medal: { metal: METALS[t.tiers.indexOf(top)], of: t.id } });
  }
  for (const f of FIRSTS) if (keys.includes(f.id)) out.push({ id: nextId++, title: f.name, line: f.text(got[f.id]), medal: { metal: "gold", of: "first" } });
  return out;
}

// what the game had when its first money state came, so only steps earned since then toast
let known: Set<string> | null = null, knownGame = -1;
effect(() => {
  const g = gamesLoaded.value, m = moneyState.value;
  if (g !== knownGame) { knownGame = g; known = null; waiting.value = []; }
  const a = m?.achievements;
  if (!m || !a) { known = null; return; }
  if (!known) { known = new Set(Object.keys(a)); return; }
  const fresh = Object.keys(a).filter((k) => !known!.has(k));
  for (const k of fresh) known.add(k);
  if (fresh.length && m.mode !== "sandbox") waiting.value = [...waiting.peek(), ...toastsFor(fresh, a)];
});

interface Shown extends Toast { held: number; out: boolean }

export function AchievementToasts() {
  const [shown, setShown] = useState<Shown[]>([]);
  const [over, setOver] = useState<number | null>(null);
  const blocked = choosing.value || savesOpen.value || !!ask.value || boardsOpen.value || accountOpen.value;
  const queue = waiting.value;
  useEffect(() => {
    if (!shown.length && !queue.length) return;
    let last = performance.now();
    const step = () => {
      const now = performance.now(), dt = now - last;
      last = now;
      if (blocked || document.hidden) return;
      setShown((list) => {
        let out = list.map((s) => (s.id === over || s.out ? s : { ...s, held: s.held + dt }));
        out = out.map((s) => (!s.out && s.held >= HOLD_MS ? { ...s, out: true, held: 0 } : s.out ? { ...s, held: s.held + dt } : s));
        out = out.filter((s) => !(s.out && s.held >= FADE_MS));
        const room = (PHONE.matches ? 1 : MAX_SHOWN) - out.filter((s) => !s.out).length;
        const q = waiting.peek();
        if (room > 0 && q.length && now - lastIn >= GAP_MS) {
          let t: Toast;
          if (q.length > MAX_SHOWN) {
            t = { id: nextId++, title: `${q.length} more achievements` };
            waiting.value = [];
          } else {
            t = q[0];
            waiting.value = q.slice(1);
          }
          lastIn = now;
          out = [...out, { ...t, held: 0, out: false }];
        }
        return out;
      });
    };
    const timer = setInterval(step, 100);
    step();
    return () => clearInterval(timer);
  }, [shown.length > 0, queue.length > 0, blocked, over]);
  if (blocked || !shown.length) return null;
  return (
    <div class="ach-toasts">
      {shown.map((s) => (
        <button key={s.id} class={"ach-toast" + (s.out ? " out" : "")} onPointerEnter={() => setOver(s.id)} onPointerLeave={() => setOver(null)}
          onClick={() => { openAchievements(); setShown((l) => l.filter((x) => x.id !== s.id)); setOver(null); }}>
          {s.medal ? <Medal metal={s.medal.metal} of={s.medal.of} /> : <span class="ach-mark"><Icon.medal /></span>}
          <span class="ach-words">
            <b>{s.title}</b>
            {s.line && <span class="muted">{s.line}</span>}
          </span>
        </button>
      ))}
    </div>
  );
}
