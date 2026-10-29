// The Getting started card (T-1822, notes/T-1822.md): a post-it at the map's upper right, under the
// search button, with six steps ticked off as the game shows them done. game/guide.ts checks the
// steps and keeps the progress in the save; this draws it. On a phone held upright it shrinks to its
// heading and the step in hand until the heading is tapped.

import { signal } from "@preact/signals";
import { h, render } from "preact";
import { guideShown, guideSteps, startGuide, type GuideStep } from "../game/guide";
import { moneyState } from "../game/money";
import { ask } from "../game/ui";
import { world } from "../game/world";
import { saveGuide } from "../workers/clockClient";
import { savesOpen } from "./SavesPanel";
import { Icon } from "./widgets";

/** the phone layout shows all six steps after a tap on the heading */
const opened = signal(false);

function counter(c: NonNullable<GuideStep["count"]>): string {
  return c.km ? `${c.have.toFixed(1)}/${c.goal.toFixed(1)} km` : `${c.have}/${c.goal}`;
}

function Guide() {
  const m = moneyState.value, w = world.value, g = m?.guide;
  if (!m || !w || !g || !guideShown(g, m.mode === "sandbox") || ask.value || savesOpen.value) return null;
  const steps = guideSteps(w);
  const done = steps.map((s, i) => s.met || (g.done & (1 << i)) !== 0);
  const now = done.indexOf(false);
  const finished = now < 0;
  return (
    <div class={"guide" + (opened.value ? " open" : "")}>
      <div class="guide-head">
        <button class="guide-title" aria-expanded={opened.value} onClick={() => (opened.value = !opened.peek())}>Getting started</button>
        <button class="btn icon guide-x" data-tip="Hide the guide" aria-label="Hide the guide" onClick={() => saveGuide({ ...g, hidden: true })}>
          <Icon.cross />
        </button>
      </div>
      <ol class="guide-steps">
        {steps.map((s, i) => (
          <li key={i} class={done[i] ? "done" : i === now ? "now" : "later"}>
            <span class="guide-box">{done[i] && <Icon.check />}</span>
            <span class="guide-words">{s.words}</span>
            {s.count && <span class="guide-count" data-tip={s.count.tip}>{counter(s.count)}</span>}
          </li>
        ))}
      </ol>
      {finished && (
        <div class="guide-end">
          <span>Your first line is open.</span>
          <button class="btn text" onClick={() => saveGuide({ ...g, closed: true })}>Done</button>
        </div>
      )}
    </div>
  );
}

/** Mount the card in the map's container. */
export function mountGuide(container: HTMLElement) {
  const root = document.createElement("div");
  container.appendChild(root);
  render(h(Guide, null), root);
  startGuide();
}
