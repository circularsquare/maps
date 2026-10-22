// T-677: the start menu. Every page load opens on it, unless a URL switch already says which game
// (`choiceFromUrl`): New game (the new game sheet, T-728), then the saved games in this browser, the
// most recent first, each a card as in the saves panel. Nothing heavy has started under it: main.ts starts the clock and demand
// workers with the pick. It covers the page and takes every key until then.

import { h, render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { CITIES, HOME_CITY } from "../game/cities";
import { choosing } from "../game/loading";
import { listSlots, newGameName, type SaveSlot } from "../workers/saveFile";
import type { GameChoice } from "../workers/clockClient";
import { NewGameSheet } from "./NewGame";
import { SlotFacts, SlotThumb, SlotTitle } from "./SavesPanel";

/** The game the URL chooses, or null for the menu: `?new=1` a new game that owns no slot (T-320),
 * normal in New York unless `&mode=sandbox` or `&city=<zone id>` say otherwise (T-728);
 * `?synth=1`, `?demandTest=1`, `?paused=1` and `?continue=1` the game opened last. */
export function choiceFromUrl(): GameChoice | null {
  const q = new URLSearchParams(location.search);
  const on = (k: string) => q.get(k) === "1";
  if (on("new")) {
    const city = q.get("city");
    return { fresh: true, newGame: true, game: { mode: q.get("mode") === "sandbox" ? "sandbox" : "normal", start: CITIES.some((c) => c.id === city) ? city! : HOME_CITY } };
  }
  if (on("synth") || on("demandTest") || on("paused") || on("continue")) return {};
  return null;
}

/** T-927: the view behind the menu, the band of the world from 60° S to 75° N covering the page:
 * all of it on a wide screen, its centre on a narrow one. Mercator y of 75° N is 2.028, of 60° S
 * -1.317, so the band is 3.345 / 2π of the world's 512 px tile height at zoom 0. */
export function worldCover(el: HTMLElement): { center: [number, number]; zoom: number } {
  const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
  return { center: [10, 20], zoom: Math.max(0, Math.log2(w / 512), Math.log2(h / (512 * 3.345 / (2 * Math.PI)))) };
}

function Menu({ slots, pick }: { slots: SaveSlot[]; pick: (c: GameChoice) => void }) {
  const first = useRef<HTMLButtonElement>(null);
  const [sheet, setSheet] = useState(false);
  useEffect(() => { if (!sheet) first.current?.focus(); }, [sheet]);
  const now = Date.now();
  // T-728: the new game sheet; its game gets a slot of its own named after its start, so no saved game is replaced
  if (sheet) return <div class="start-menu" role="dialog" aria-modal="true" aria-labelledby="start-title">
    <div class="start-box new-game-box">
      <h1 id="start-title" class="big">New game</h1>
      <NewGameSheet heading={false} onBack={() => setSheet(false)} onStart={(game) => pick({ newGame: true, newSlot: newGameName(slots, game), game })} />
    </div>
  </div>;
  return <div class="start-menu" role="dialog" aria-modal="true" aria-labelledby="start-title">
    <div class="start-box">
      <h1 id="start-title" class="big">Welcome to anitabuilder!</h1>
      <button ref={first} class="btn start-new" onClick={() => setSheet(true)}>New game</button>
      {slots.length > 0 && <>
        <h2 class="start-head">Saved games</h2>
        <ul class="start-saves">
          {slots.map((s) => <li key={s.key}>
            <button class="start-save" data-slot={s.key} onClick={() => pick({ slot: s.key })}>
              <SlotThumb slot={s} />
              <span class="save-info"><SlotTitle slot={s} /><SlotFacts slot={s} now={now} /></span>
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
