// T-972, T-973: the optional anita.garden account and the leaderboards (notes/T-972.md). Signing in
// is never needed to play. The account is noritetsu's (shonei-server's `/noritetsu/` sign-in routes,
// its token kept under noritetsu's storage key, so one sign-in serves both pages on anita.garden);
// the leaderboards are shonei-server's `/trainworld/` routes (its docs/trainworld-api.md).
//
// Which server: market.anita.garden on anita.garden. On localhost none unless `?acct=local` (a
// shonei-server run on 127.0.0.1:8083) or `?acct=live` was given once (remembered; `?acct=off`
// forgets it), so a dev page or a check makes no request that could fail. Everything waits for
// `/noritetsu/config` to say `trainworld`: an older server, or none, keeps every icon hidden.
//
// While signed in, each normal game's points (the past 24 game hours, the Cities tab's) are sent at
// most every two wall minutes when they change, joined or not, so a board is current the moment its
// player joins. A game owns a random id per browser slot (`GAMES_KEY`); a sandbox sends only the
// fact, once, which the server keeps for good. A slot deleted here takes its figures off.

import { computed, effect, signal } from "@preact/signals";
import { moneyState } from "./money";
import { pointsPast24h } from "./points";
import { saveSlotsState } from "../workers/clockClient";

export interface AccountView {
  id: string;
  logins: string[];
  username?: string;
  email: string;
  pendingEmail: string;
}
interface Session { token: string; account: AccountView }
export interface AccountConfig { googleClientId: string; dev: boolean; password: boolean; signIn: boolean; trainworld?: boolean }
export interface BoardSetting { on: boolean; name: string }
export interface BoardRow { rank: number; name: string; value: number }
export interface Board { board: string; count: number; rows: BoardRow[]; me?: BoardRow }

/** The most cities a game sends (shonei-server's MaxTwCities): its best ones. */
const MAX_CITIES = 64;
/** Wall ms between two sends of a game's figures. */
const SEND_GAP_MS = 120_000;
const API_KEY = "trainworld.acct-api";
const GAMES_KEY = "trainworld.score-games.v1";

const LOCAL = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);

function apiChoice(): "live" | "local" | null {
  const q = new URLSearchParams(location.search).get("acct");
  try {
    if (q === "off") localStorage.removeItem(API_KEY);
    else if (q === "live" || q === "local") localStorage.setItem(API_KEY, q);
  } catch { /* storage blocked */ }
  if (/(^|\.)anita\.garden$/.test(location.hostname)) return "live";
  if (!LOCAL) return null;
  try {
    const v = localStorage.getItem(API_KEY);
    return v === "live" || v === "local" ? v : null;
  } catch { return null; }
}

const CHOICE = apiChoice();
const BASE = CHOICE === "live" ? "https://market.anita.garden/" : CHOICE === "local" ? "http://127.0.0.1:8083/" : "";
// noritetsu's key where noritetsu talks to the same server (the live site, or local with local), so
// a sign-in is shared; a localhost page on the live server keeps its own, so it never signs a
// local noritetsu out with a token its server refuses.
const AUTH_KEY = (CHOICE === "live") === !LOCAL ? "noritetsu.auth.v1" : "trainworld.auth.live.v1";

const lsGet = <T>(k: string): T | null => { try { return JSON.parse(localStorage.getItem(k) ?? "null") as T; } catch { return null; } };
const lsSet = (k: string, v: unknown) => {
  try { if (v == null) localStorage.removeItem(k); else localStorage.setItem(k, JSON.stringify(v)); } catch { /* this page only */ }
};

/** "off" (no server for this page), "checking", "on", or "unavailable" (no answer, or a server
 * without the leaderboards); the icons show only when "on". */
export const accountState = signal<"off" | "checking" | "on" | "unavailable">(BASE ? "checking" : "off");
export const accountConfig = signal<AccountConfig | null>(null);
const readSession = (): Session | null => { const s = lsGet<Session>(AUTH_KEY); return s?.token ? s : null; };
export const session = signal<Session | null>(readSession());
export const signedIn = computed(() => !!session.value);
/** The leaderboard setting, from `/trainworld/me`; null until it answers. */
export const boardSetting = signal<BoardSetting | null>(null);
export const accountOpen = signal(false);
export const boardsOpen = signal(false);

export class ApiError extends Error { constructor(msg: string, readonly status: number) { super(msg); } }

export async function api<T>(method: string, path: string, body?: unknown, opts: { auth?: boolean; keep401?: boolean } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const s = session.peek();
  if (opts.auth !== false && s) headers.Authorization = "Bearer " + s.token;
  let r: Response;
  try {
    r = await fetch(BASE + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError("The server could not be reached.", 0);
  }
  let j: any = null;
  if (r.status !== 204) { try { j = await r.json(); } catch { /* no body */ } }
  if (!r.ok) {
    // a dead token means signed out; a wrong current password (keep401) does not
    if (r.status === 401 && opts.auth !== false && !opts.keep401 && s) tokenGone();
    const msg = typeof j?.error === "string" ? j.error : `The server answered ${r.status}.`;
    throw new ApiError(msg.charAt(0).toUpperCase() + msg.slice(1).replace(/([^.])$/, "$1."), r.status);
  }
  return j as T;
}

export function storeSession(j: { token: string; account: AccountView }) {
  session.value = { token: j.token, account: j.account };
  lsSet(AUTH_KEY, session.value);
  void loadBoardSetting();
}
function tokenGone() {
  session.value = null;
  boardSetting.value = null;
  lsSet(AUTH_KEY, null);
}
export function signOut() { tokenGone(); }

/** Sign in, register or log in: the answer is a session. */
export async function signIn(path: "signin" | "login" | "register", body: unknown) {
  storeSession(await api<{ token: string; account: AccountView }>("POST", "noritetsu/" + path, body, { auth: false }));
}

export async function deleteAccount() {
  await api("DELETE", "noritetsu/account");
  tokenGone();
  lsSet(GAMES_KEY, null);
}

async function loadBoardSetting() {
  if (!session.peek() || accountState.peek() !== "on") return;
  try { boardSetting.value = (await api<{ board: BoardSetting }>("GET", "trainworld/me")).board; }
  catch { /* shown as not joined; a 401 has signed out */ }
}

export async function setBoard(on: boolean, name = "") {
  boardSetting.value = (await api<{ board: BoardSetting }>("PUT", "trainworld/optin", { on, name })).board;
  if (on) await sendScores(true);
}

// ---------------------------------------------------------------- the boards

export const boards = signal<{ list: Record<string, number>; data: Board | null; key: string; at: number; error: string }>({ list: {}, data: null, key: "points", at: 0, error: "" });
let fetching = false;
export async function fetchBoards(key = boards.peek().key) {
  if (fetching) return;
  fetching = true;
  boards.value = { ...boards.peek(), key };
  try {
    const [list, data] = await Promise.all([
      api<{ boards: Record<string, number> }>("GET", "trainworld/boards", undefined, { auth: false }),
      api<Board>("GET", `trainworld/board?board=${encodeURIComponent(key)}`),
    ]);
    boards.value = { list: list.boards ?? {}, data, key, at: Date.now(), error: "" };
  } catch (e) {
    boards.value = { ...boards.peek(), at: Date.now(), error: (e as Error).message };
  } finally {
    fetching = false;
  }
}

// ---------------------------------------------------------------- scores

/** Each browser slot's game id, the account it last sent to, what it sent, and whether its
 * sandbox notice went. */
interface GameEntry { id: string; acct?: string; sent?: string; at?: number; sandbox?: boolean }
const games = (): Record<string, GameEntry> => lsGet<Record<string, GameEntry>>(GAMES_KEY) ?? {};

function newGameId(): string {
  const a = "abcdefghijkmnpqrstuvwxyz23456789", b = crypto.getRandomValues(new Uint8Array(12));
  return Array.from(b, (v) => a[v % a.length]).join("");
}

/** The playing game's slot, when it is a stored one this tab writes (not a `?new=1` page). */
function playingSlot(): string | null {
  const s = saveSlotsState.peek();
  return s.writable && s.active && s.slots.some((x) => x.key === s.active) ? s.active : null;
}

/** The game's figures now: points in the past 24 game hours in all, and its best cities. */
function figures(): { day: number; points: number; cities: Record<string, number> } | null {
  const m = moneyState.peek();
  const hour = m?.hours.at(-1)?.hour;
  if (!m || hour === undefined) return null;
  const per = [...pointsPast24h.peek()].filter(([id, v]) => id && v > 0).sort((a, b) => b[1] - a[1]);
  const points = Math.round(per.reduce((a, [, v]) => a + v, 0));
  const cities: Record<string, number> = {};
  for (const [id, v] of per.slice(0, MAX_CITIES)) cities[id] = Math.round(v);
  return { day: Math.floor(hour / 24), points, cities };
}

let sending = false;
/** Send the playing game's figures if they changed and the gap has passed (`now` skips the gap). */
export async function sendScores(now = false) {
  const s = session.peek(), slot = playingSlot(), m = moneyState.peek();
  if (sending || !s || accountState.peek() !== "on" || !slot || !m) return;
  const all = games(), e = all[slot] ?? { id: newGameId() };
  const acct = s.account.id;
  let body: unknown;
  if (m.mode === "sandbox") {
    // only a game that may have figures on this account needs the notice
    if (e.sandbox || e.acct !== acct || !e.sent) return;
    body = { mode: "sandbox" };
  } else {
    const f = figures();
    if (!f) return;
    const sig = JSON.stringify([f.points, f.cities]);
    const before = e.acct === acct ? e.sent : undefined;
    if (sig === before || (!before && f.points <= 0)) return;
    if (!now && before && Date.now() - (e.at ?? 0) < SEND_GAP_MS) return;
    body = { mode: "normal", ...f };
    e.sent = sig;
  }
  sending = true;
  try {
    await api("PUT", `trainworld/score/${e.id}`, body);
    if (m.mode === "sandbox") e.sandbox = true;
    e.acct = acct;
    e.at = Date.now();
    lsSet(GAMES_KEY, { ...games(), [slot]: e });
    if (boards.peek().at) boards.value = { ...boards.peek(), at: 0 }; // the boards may have moved
  } catch (err) {
    // a game the server knows as a sandbox: it stays unlisted, so stop asking
    if ((err as ApiError).status === 403) lsSet(GAMES_KEY, { ...games(), [slot]: { ...e, sandbox: true, acct } });
  } finally {
    sending = false;
  }
}

/** Slots gone from this browser take their figures off the account they were sent to. */
async function dropDeleted() {
  const s = session.peek(), st = saveSlotsState.peek();
  if (!st.slots.length) return; // storage unread or blocked: nothing is known to be gone
  const all = games(), keys = new Set(st.slots.map((x) => x.key));
  for (const [slot, e] of Object.entries(all)) {
    if (keys.has(slot) || slot === st.active) continue;
    if (e.sent && !e.sandbox) {
      if (!s || e.acct !== s.account.id || accountState.peek() !== "on") continue; // later, signed in
      try { await api("DELETE", `trainworld/score/${e.id}`); } catch { continue; }
    }
    const now = games();
    delete now[slot];
    lsSet(GAMES_KEY, now);
  }
}

// ---------------------------------------------------------------- start

let started = false;
/** Ask the server once whether accounts are on; refresh the session; start sending. Idempotent. */
export function startAccount() {
  if (started || !BASE) return;
  started = true;
  void (async () => {
    try {
      const cfg = await api<AccountConfig>("GET", "noritetsu/config", undefined, { auth: false });
      accountConfig.value = cfg;
      accountState.value = cfg.signIn && cfg.trainworld ? "on" : "unavailable";
    } catch {
      accountState.value = "unavailable";
    }
    if (accountState.peek() !== "on") return;
    if (session.peek()) {
      try { storeSession(await api("POST", "noritetsu/refresh")); }
      catch { /* a 401 has signed out; offline keeps the token for next time */ }
    }
    setInterval(() => void sendScores(), 15_000);
    effect(() => { void saveSlotsState.value; void session.value; void dropDeleted(); });
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") void sendScores(true); });
  })();
  // another tab on this origin (noritetsu, or this game) signed in or out
  addEventListener("storage", (e) => {
    if (e.key !== AUTH_KEY) return;
    session.value = readSession();
    boardSetting.value = null;
    void loadBoardSetting();
  });
}
