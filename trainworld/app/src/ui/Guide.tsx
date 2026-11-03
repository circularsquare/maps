// The tutorial card (T-1822, T-2123; notes/T-1822.md): a post-it at the map's upper right, under the
// search button, with the steps of the card in hand ticked off as the game shows them done.
// game/guide.ts checks the steps and keeps the progress in the save; this draws it. On a phone held
// upright it shrinks to its heading and the step in hand until the heading is tapped.

import { signal } from "@preact/signals";
import { h, render } from "preact";
import { guideShown, guideSteps, guideTicks, startGuide, type GuideStep } from "../game/guide";
import { moneyState } from "../game/money";
import { ask } from "../game/ui";
import { world } from "../game/world";
import { saveGuide } from "../workers/clockClient";
import { ripple, startRipple } from "./ripple";
import { savesOpen } from "./SavesPanel";
import { Icon } from "./widgets";

/** the phone layout shows every step after a tap on the heading */
const opened = signal(false);

function counter(c: NonNullable<GuideStep["count"]>): string {
  return c.unit === "km" ? `${c.have.toFixed(1)}/${c.goal.toFixed(1)} km` : c.unit === "%" ? `${c.have.toFixed(1)}/${c.goal}%` : `${c.have}/${c.goal}`;
}

function Guide() {
  const m = moneyState.value, w = world.value, g = m?.guide;
  if (!m || !w || !g || !guideShown(g, m.mode === "sandbox") || ask.value || savesOpen.value) return null;
  const steps = guideSteps(w, g);
  const { done, now } = guideTicks(steps, g);
  const finished = now < 0;
  const close = () => (ask.value = { text: "You can restart the tutorial in Settings.", yes: "Close tutorial", no: "Keep going", run: () => saveGuide({ ...g, hidden: true }) });
  return (
    <div class={"guide" + (opened.value ? " open" : "")} data-card={g.card ?? 1}>
      <div class="guide-head">
        <button class="guide-title" aria-expanded={opened.value} onClick={() => (opened.value = !opened.peek())}>Tutorial</button>
        <button class="btn icon guide-x" aria-label="Close the tutorial" onClick={close}>
          <Icon.cross />
        </button>
      </div>
      <ol class="guide-steps">
        {steps.map((s, i) => (
          <li key={s.id} data-step={s.id} class={done[i] ? "done" : i === now ? "now" : "later"} onClick={() => ripple(s.id)}>
            <span class="guide-box">{done[i] && <Icon.check />}</span>
            <span class="guide-words">{s.words}</span>
            {s.count && <span class="guide-count">{counter(s.count)}</span>}
          </li>
        ))}
      </ol>
      {finished && (
        <div class="guide-end">
          {g.card === 2
            ? <button class="btn text" onClick={() => saveGuide({ ...g, closed: true })}>Done!</button>
            : <button class="btn text" onClick={() => saveGuide({ ...g, card: 2 })}>Continue</button>}
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
  startRipple();
}
