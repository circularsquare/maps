// T-3038: how many players have each achievement (shonei-server's /trainworld/achieved and
// /trainworld/achievements, its docs/trainworld-api.md "Achievement counts"). When this game earns
// achievements this browser has not sent, one post carries their keys and the anonymous browser id the
// plays ping uses (game/playtime.ts); nothing else, and no session. The remembered keys live in
// localStorage, so each is sent once per browser. The page sends nothing where the plays ping does not
// (`?notrack`, localhost, automated browsers). The counts are read when the achievements view opens.

import { effect, signal } from "@preact/signals";
import { moneyState } from "./money";
import { browserId, quiet } from "./playtime";
import { tNumber } from "../i18n";

type Call = <T>(method: string, path: string, body?: unknown) => Promise<T>;

/** The server's answer, or null before it has come (or against a server without it). */
export const achievementCounts = signal<{ players: number; counts: Record<string, number> } | null>(null);

const SENT_KEY = "trainworld.achSent";
/** Under this many players a share says nothing. */
const MIN_PLAYERS = 20;
const SEND_DELAY_MS = 4000, COUNTS_TTL_MS = 10 * 60_000;

let call: Call | null = null;

function sentKeys(): Set<string> {
  try { return new Set<string>(JSON.parse(localStorage.getItem(SENT_KEY) ?? "[]")); } catch { return new Set(); }
}

/** Begin reporting, and make `loadAchievementCounts` work. Idempotent. */
export function startAchievementCounts(send: Call) {
  if (call) return;
  call = send;
  if (quiet()) return;
  const b = browserId();
  if (!b) return;
  let timer: ReturnType<typeof setTimeout> | undefined;
  effect(() => {
    const m = moneyState.value;
    if (!m || m.mode === "sandbox" || m.copy || !m.achievements) return; // T-3115: a copy holds another player's achievements
    const keys = Object.keys(m.achievements);
    clearTimeout(timer);
    timer = setTimeout(() => {
      const sent = sentKeys();
      const fresh = keys.filter((k) => !sent.has(k)).slice(0, 64);
      if (!fresh.length) return;
      send("POST", "trainworld/achieved", { b, ids: fresh }).then(() => {
        for (const k of fresh) sent.add(k);
        try { localStorage.setItem(SENT_KEY, JSON.stringify([...sent])); } catch { /* sent again next time */ }
      }).catch(() => { /* silent; tried again with the next change */ });
    }, SEND_DELAY_MS);
  });
}

let askedAt = -Infinity;
/** Read the counts, at most every ten minutes; a failure leaves what is there. */
export function loadAchievementCounts() {
  if (!call || performance.now() - askedAt < COUNTS_TTL_MS) return;
  askedAt = performance.now();
  call<{ players: number; counts: Record<string, number> }>("GET", "trainworld/achievements")
    .then((r) => { if (r && typeof r.players === "number" && r.counts) achievementCounts.value = r; })
    .catch(() => { askedAt = -Infinity; });
}

/** "12%", or one significant digit under one ("0.03%"); null while the counts are unknown or the players too few to say. */
export function shareOfPlayers(key: string): string | null {
  const c = achievementCounts.value;
  if (!c || c.players < MIN_PLAYERS) return null;
  const f = Math.min(1, (c.counts[key] ?? 0) / c.players);
  return tNumber(f, f < 0.01 ? { style: "percent", maximumSignificantDigits: 1 } : { style: "percent", maximumFractionDigits: 0 });
}
