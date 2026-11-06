// T-1771: anonymous plays and playtime, on Anita's own server (shonei-server's /trainworld/ping, its
// docs/trainworld-api.md "Plays and playtime"). A random id is made per page load and lives in this
// module only. A second random id, kept in localStorage, is the same across visits from one browser,
// so the server can count distinct browsers (Anita, 2026-11-06); it is never tied to an account, and
// the server forgets it after 13 months unseen. While a game is open (not the start menu or the new
// game sheet) and the tab is visible, one ping about every minute carries both ids, the zone the game
// began in, whether it is a sandbox and the build. Signed in, the ping carries the session like every
// other call (game/account.ts), so the server counts minutes per account.
//
// Nothing is sent: with `notrack` in localStorage (the website's owner opt-out; `?notrack` sets it,
// `?track` clears it), on localhost unless `?track` is in the address, or from an automated browser
// (smoke, screenshots) unless `?track` is. Failures are silent.

import { moneyState } from "./money";
import { choosing } from "./loading";
import { gameStartZone } from "./realms";

declare const __TW_BUILD__: string | undefined;
const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "dev";

const PING_MS = 60_000;

export interface Ping { id: string; b?: string; start: string; sandbox: boolean; v: string }

const BROWSER_KEY = "trainworld.browser";

/** Whether this page sends nothing. */
function quiet(): boolean {
  const q = new URLSearchParams(location.search);
  let stored = false;
  try {
    if (q.has("notrack")) localStorage.setItem("notrack", "1");
    if (q.has("track")) localStorage.removeItem("notrack");
    stored = !!localStorage.getItem("notrack");
  } catch { /* storage blocked: only the address can say */ }
  if (q.has("notrack") || stored) return true;
  if (q.has("track")) return false;
  return /^(localhost|127\.0\.0\.1)$/.test(location.hostname) || navigator.webdriver === true;
}

function newId(): string {
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
}

/** This browser's id, made on first use; undefined when storage is blocked. */
function browserId(): string | undefined {
  try {
    let id = localStorage.getItem(BROWSER_KEY);
    if (!id || !/^[0-9a-f]{32}$/.test(id)) {
      id = newId();
      localStorage.setItem(BROWSER_KEY, id);
    }
    return id;
  } catch {
    return undefined;
  }
}

let started = false;
/** Begin pinging through `send`. Idempotent; does nothing on a page that sends nothing. */
export function startPlaytime(send: (ping: Ping) => Promise<unknown>) {
  if (started || quiet()) return;
  started = true;
  const id = newId();
  const b = browserId();
  let last = 0;
  // A once-a-second check, not a once-a-minute timer: the first ping goes the moment a game is open,
  // and a tab that was hidden starts a fresh minute when it is shown again.
  setInterval(() => {
    const now = Date.now();
    if (document.visibilityState !== "visible" || choosing.peek() || !moneyState.peek()) {
      if (last) last = now;
      return;
    }
    if (last && now - last < PING_MS) return;
    last = now;
    void send({ id, b, start: gameStartZone() ?? "", sandbox: moneyState.peek()!.mode === "sandbox", v: BUILD }).catch(() => {});
  }, 1000);
}
