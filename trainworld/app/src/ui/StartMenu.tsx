// T-677: the start menu. Every page load opens on it, unless a URL switch already says which game
// (`choiceFromUrl`): New game, then the saved games in this browser, the most recent first, each a
// card as in the saves panel. Nothing heavy has started under it: main.ts starts the clock and demand
// workers with the pick. It covers the page and takes every key until then.

import { h, render } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { choosing } from "../game/loading";
import { listSlots, ORIGINAL_SLOT, type SaveSlot } from "../workers/saveFile";
import type { GameChoice } from "../workers/clockClient";
import { SlotFacts, SlotThumb } from "./SavesPanel";

/** The game the URL chooses, or null for the menu: `?new=1` a new game that owns no slot (T-320);
 * `?synth=1`, `?demandTest=1`, `?paused=1` and `?continue=1` the game opened last. */
export function choiceFromUrl(): GameChoice | null {
  const q = new URLSearchParams(location.search);
  const on = (k: string) => q.get(k) === "1";
  if (on("new")) return { fresh: true, newGame: true };
  if (on("synth") || on("demandTest") || on("paused") || on("continue")) return {};
  return null;
}

function Menu({ slots, pick }: { slots: SaveSlot[]; pick: (c: GameChoice) => void }) {
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => first.current?.focus(), []);
  const now = Date.now();
  // with no saved game the new one takes the original slot, as before the menu; beside saved games
  // it gets a slot of its own, so no saved game is replaced
  const fresh: GameChoice = slots.length ? { newGame: true, newSlot: `Game ${slots.length + 1}` } : { newGame: true, slot: ORIGINAL_SLOT };
  return <div class="start-menu" role="dialog" aria-modal="true" aria-labelledby="start-title">
    <div class="start-box">
      <h1 id="start-title" class="big">anitabuilder</h1>
      <button ref={first} class="btn start-new" onClick={() => pick(fresh)}>New game</button>
      {slots.length > 0 && <>
        <h2 class="start-head">Saved games</h2>
        <ul class="start-saves">
          {slots.map((s) => <li key={s.key}>
            <button class="start-save" data-slot={s.key} onClick={() => pick({ slot: s.key })}>
              <SlotThumb slot={s} />
              <span class="save-info"><strong>{s.name}</strong><SlotFacts slot={s} now={now} /></span>
            </button>
          </li>)}
        </ul>
      </>}
    </div>
  </div>;
}

/** Put up the menu; `onPick` gets the player's choice once, after the menu has gone. */
export function showStartMenu(onPick: (c: GameChoice) => void) {
  choosing.value = true;
  document.documentElement.classList.add("choosing");
  const root = document.createElement("div");
  document.body.appendChild(root);
  // the map's keys and the hotkeys wait for the pick; Tab and Enter work inside the menu
  const block = (e: KeyboardEvent) => {
    if (e.target instanceof Node && root.contains(e.target)) return;
    e.stopImmediatePropagation();
    e.preventDefault();
  };
  window.addEventListener("keydown", block, true);
  window.addEventListener("keyup", block, true);
  let done = false;
  const pick = (c: GameChoice) => {
    if (done) return;
    done = true;
    window.removeEventListener("keydown", block, true);
    window.removeEventListener("keyup", block, true);
    render(null, root);
    root.remove();
    document.documentElement.classList.remove("choosing");
    choosing.value = false;
    onPick(c);
  };
  // storage blocked (a private window): no saved games, New game alone
  void listSlots().catch(() => [] as SaveSlot[]).then((slots) => { if (!done) render(h(Menu, { slots, pick }), root); });
}
