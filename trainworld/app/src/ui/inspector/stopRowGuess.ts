// T-3232: a stop click's row in the line panel's stop list changes at the click, before the worker's world. A stop
// taken out has its row hidden; a stop added gets a placeholder row where the line passes it (the panel's "passed"
// stations know the gap), and a station off the line's route gets none. The rows are drawn from the world with
// per-stop figures lined up by index, so the guess is done on the DOM and the next render replaces it
// (`applyRowGuesses` in a layout effect of the inspector).
import { world } from "../../game/world";

interface Guess { add: boolean; n: number; num: number }
const guesses = new Map<string, Guess>();

const rowsOf = (station: string) => [...document.querySelectorAll<HTMLElement>(`ul.stops > li[data-station="${station}"]`)];

/** the guesses drawn over the rows the panel has now; they go and come back after each render */
export function applyRowGuesses() {
  for (const e of document.querySelectorAll(".guess-out, li[data-guess]")) {
    if (e.hasAttribute("data-guess")) e.remove();
    else e.classList.remove("guess-out");
  }
  for (const [station, g] of guesses) {
    const rows = rowsOf(station);
    if (!g.add) {
      for (const r of rows) r.classList.add("guess-out");
      continue;
    }
    if (rows.length) continue;
    const w = world.peek();
    const line = document.querySelector<HTMLElement>("ul.stops > li[data-stop]")?.closest("ul.stops");
    const l = w?.save.lines.find((x) => x.num === g.num);
    const st = l && w?.lineStats[l.id];
    if (!w || !l || !st || !line || st.loop) continue;
    const at = st.passed?.find((p) => String(p.node) === station);
    if (!at) continue;
    const flip = (line.firstElementChild as HTMLElement | null)?.dataset.stop === String(l.stops.length - 1) && l.stops.length > 1;
    const next = line.querySelector<HTMLElement>(`:scope > li[data-stop="${flip ? at.gap : at.gap + 1}"]`);
    if (!next) continue;
    const li = document.createElement("li");
    li.className = "adding";
    li.setAttribute("data-guess", "");
    const grow = document.createElement("span"), name = document.createElement("span");
    grow.className = "grow";
    name.className = "stop-name";
    name.textContent = w.save.stations.find((s) => String(s.id) === station)?.name ?? "";
    grow.appendChild(name);
    li.appendChild(grow);
    next.before(li);
  }
}

/** a plain click on `station`, drawn at once; `isStop` is whether the world's line stops there, and a click already
 * out for the station counts as done. */
export function guessRow(station: string, isStop: boolean, num: number) {
  const g = guesses.get(station);
  guesses.set(station, { add: g ? !g.add : !isStop, n: (g?.n ?? 0) + 1, num });
  applyRowGuesses();
}

/** the click's edit is answered. A refused one puts the row back; one the world holds is left to the render that
 * shows the world. */
export function settleRow(station: string, isStop: boolean) {
  const g = guesses.get(station);
  if (!g) return;
  if (--g.n > 0) return;
  guesses.delete(station);
  if (g.add !== isStop) applyRowGuesses();
  else setTimeout(applyRowGuesses, 300);
}
