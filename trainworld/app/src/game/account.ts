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
// player joins. A game owns a random id per browser slot (`GAMES_KEY`), or its account game's id
// (T-988); a sandbox sends only the fact, once, which the server keeps for good. A slot deleted here
// takes its figures off.
//
// T-988: the games themselves go on the account too (workers/cloud.ts, game/accountGames.ts); this
// file gives the clock worker the account (`cloudContext`) on every change.

import { effect, signal } from "@preact/signals";
import { moneyState } from "./money";
import { pointsPast24h } from "./points";
import { startPlaytime } from "./playtime";
import { startGameTime } from "./gameTime";
import { startAchievementCounts } from "./achievementCounts";
import { onAccountGone, postAccount, saveSlotsState } from "../workers/clockClient";
import { newGameId, onTokenGone, type CloudContext } from "../workers/cloud";
import { listSlots, writeSync } from "../workers/saveFile";
import { t } from "../i18n";
import { patientFetch, CLOUD_STALL_MS } from "../workers/patientFetch";

export interface AccountView {
  id: string;
  logins: string[];
  username?: string;
  email: string;
  pendingEmail: string;
}
interface Session { token: string; account: AccountView }
export interface AccountConfig { googleClientId: string; dev: boolean; password: boolean; signIn: boolean; trainworld?: boolean; trainworldGames?: boolean; trainworldFeedback?: boolean; trainworldFeedbackThreads?: boolean; trainworldPing?: boolean; trainworldAchievements?: boolean; trainworldCommunity?: boolean; trainworldCommunityPreview?: boolean; trainworldCommunitySearch?: boolean; trainworldFlags?: boolean; /** T-3279: the flag list version the server accepts (game/flags.json `set`) */ trainworldFlagSet?: number }
export interface BoardSetting { on: boolean; name: string; /** set when a moderator hid the display name; absent on a server that does not say */ nameHidden?: boolean; /** T-3269: the flag beside the name, game/flags.ts */ flag?: string }
export interface BoardRow { rank: number; name: string; value: number; flag?: string }
export interface Board { board: string; count: number; rows: BoardRow[]; me?: BoardRow }

/** The most cities a game sends (shonei-server's MaxTwCities): its best ones. */
const MAX_CITIES = 64;
/** Wall ms between two sends of a game's figures. */
const SEND_GAP_MS = 120_000;
/** T-988: wall ms between two asks for the config while the server cannot be reached */
const CONFIG_RETRY_MS = 120_000;
const API_KEY = "trainworld.acct-api";
const GAMES_KEY = "trainworld.score-games.v1";
/** T-988: this browser profile's random device id, nothing about the machine */
const DEVICE_KEY = "trainworld.device.v1";
/** T-988: whether the server last said it keeps games */
const GAMES_ON_KEY = "trainworld.games-on.v1";

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
/** The leaderboard setting, from `/trainworld/me`; null until it answers. */
export const boardSetting = signal<BoardSetting | null>(null);
export const accountOpen = signal(false);
export const boardsOpen = signal(false);

export class ApiError extends Error { constructor(msg: string, readonly status: number, readonly code = "", /** the rest of an error answer, e.g. `played` seconds on a "too-new" */ readonly extra: Record<string, unknown> | null = null) { super(msg); } }

export async function api<T>(method: string, path: string, body?: unknown, opts: { auth?: boolean; keep401?: boolean; headers?: Record<string, string>; form?: FormData } = {}): Promise<T> {
  const headers: Record<string, string> = { ...opts.headers };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const s = session.peek();
  if (opts.auth !== false && s) headers.Authorization = "Bearer " + s.token; // i18n-ignore
  let r: Response;
  try {
    r = await patientFetch(BASE + path, { method, headers, body: opts.form ?? (body === undefined ? undefined : JSON.stringify(body)) }, CLOUD_STALL_MS);
  } catch {
    throw new ApiError(t("account.unreachable"), 0);
  }
  let j: any = null;
  if (r.status !== 204) { try { j = await r.json(); } catch { /* no body */ } }
  if (!r.ok) {
    // a dead token means signed out; a wrong current password (keep401) does not
    if (r.status === 401 && opts.auth !== false && !opts.keep401 && s) tokenGone();
    const msg = typeof j?.error === "string" ? j.error : t("account.answered", { status: r.status });
    throw new ApiError(msg.charAt(0).toUpperCase() + msg.slice(1).replace(/([^.])$/, "$1."), r.status, typeof j?.code === "string" ? j.code : "", j && typeof j === "object" ? j : null);
  }
  return j as T;
}

/** T-3115: a file from the server as bytes, without the session; an ApiError for a failed answer, "too-big" past
 * `max` bytes (read no further than that). */
export async function fetchBytes(path: string, max: number): Promise<Uint8Array> {
  let r: Response;
  try { r = await patientFetch(BASE + path); } catch { throw new ApiError(t("account.unreachable"), 0); }
  if (!r.ok || !r.body) throw new ApiError(t("account.answered", { status: r.status }), r.status);
  const reader = r.body.getReader(), parts: Uint8Array[] = [];
  let n = 0;
  for (;;) {
    let got: ReadableStreamReadResult<Uint8Array>;
    try { got = await reader.read(); } catch { throw new ApiError(t("account.unreachable"), 0); }
    if (got.done) break;
    n += got.value.length;
    if (n > max) { void reader.cancel().catch(() => {}); throw new ApiError("too big", 413, "too-big"); } // i18n-ignore: a code, not shown
    parts.push(got.value);
  }
  const out = new Uint8Array(n);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

/** T-1673: the feedback form's send, to the account server (which mails it on; no address is in
 * the page). Signed in, it carries the session so the mail names the account (T-1719). Throws an ApiError whose message says what to do. */
export async function sendFeedback(f: { name: string; contact: string; message: string; context: Record<string, string>; kind?: string }) {
  if (!BASE) throw new ApiError(t("account.noServer"), 0);
  // a server that does not say it takes feedback answers 404 without CORS headers: do not call it
  const cfg = accountConfig.peek();
  if (cfg && !cfg.trainworldFeedback) throw new ApiError(t("account.feedbackNotYet"), 404);
  // T-2853: only a server that says it keeps threads gets the tag and the browser key (an older one would
  // refuse the key header in its CORS preflight)
  const threads = cfg?.trainworldFeedbackThreads === true;
  if (threads && !session.peek() && !readFeedbackKey()) makeFeedbackKey();
  await api("POST", "trainworld/feedback", { kind: threads && f.kind ? f.kind : "feedback", message: f.message, name: f.name, contact: f.contact, context: f.context },
    { headers: threads ? keyHeaders() : undefined });
}

// ---------------------------------------------------------------- feedback threads (T-2853)

/** The random key a signed-out browser's feedback belongs to (32 bytes, base64url). Made the first time
 * feedback is sent signed out; the server keeps only its hash. Never in a URL; not the device id. */
const FB_KEY = "trainworld.feedback-key.v1";
/** How many of Anita's replies to each thread were already announced, so a reload says nothing twice */
const FB_CHECK_MS = 30 * 60_000;

export interface FeedbackRow { id: string; kind: string; created: number; updated: number; first: string; messages: number; unread: number }
export interface FeedbackMessage { at: number; from: "player" | "anita"; text: string }
export interface FeedbackThread { id: string; kind: string; created: number; updated: number; unread: number; messages: FeedbackMessage[] }

/** Threads with a reply not yet read: the dot on Submit feedback and on the Settings tab. */
export const feedbackUnread = signal(0);

/** Whether the server keeps feedback threads; off, nothing here is called. */
export const feedbackThreadsOn = () => !!BASE && accountConfig.value?.trainworldFeedbackThreads === true;

export function readFeedbackKey(): string {
  try { const k = localStorage.getItem(FB_KEY); return k && /^[A-Za-z0-9_-]{32,128}$/.test(k) ? k : ""; } catch { return ""; }
}
function makeFeedbackKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  const k = btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  try { localStorage.setItem(FB_KEY, k); } catch { return ""; } // blocked storage: sending works, no list shows
  return readFeedbackKey();
}
const keyHeaders = (): Record<string, string> => { const k = readFeedbackKey(); return k ? { "X-Feedback-Key": k } : {}; };
/** Whether this browser can have earlier feedback to list: a session or a key. */
export const canListFeedback = () => feedbackThreadsOn() && !!(session.value || readFeedbackKey());

const fbTrail = (id: string) => `trainworld/feedback/thread/${encodeURIComponent(id)}`;
export async function listFeedback(): Promise<FeedbackRow[]> {
  const r = await api<{ threads?: FeedbackRow[] }>("GET", "trainworld/feedback/mine", undefined, { headers: keyHeaders() });
  const rows = r.threads ?? [];
  feedbackUnread.value = rows.filter((x) => x.unread > 0).length;
  return rows;
}
export const getFeedbackThread = (id: string) => api<FeedbackThread>("GET", fbTrail(id), undefined, { headers: keyHeaders() });
export const replyFeedback = (id: string, message: string) => api<FeedbackThread>("POST", fbTrail(id) + "/reply", { message }, { headers: keyHeaders() });
export const deleteFeedback = (id: string) => api<void>("DELETE", fbTrail(id), undefined, { headers: keyHeaders() });
export async function markFeedbackRead(id: string) {
  await api<void>("POST", fbTrail(id) + "/read", undefined, { headers: keyHeaders() });
}

let claiming = false;
/** Signed in with a key: the key's threads move to the account and the key goes (on a 200 only). */
async function claimFeedbackKey() {
  const k = readFeedbackKey();
  if (!k || !session.peek() || claiming) return;
  claiming = true;
  try {
    await api("POST", "trainworld/feedback/claim", {}, { headers: { "X-Feedback-Key": k } });
    try { localStorage.removeItem(FB_KEY); } catch { /* blocked */ }
  } catch { /* the next check tries again */ } finally { claiming = false; }
}

/** The cheap check, from memory on the server: claims a key if there is one and sets the dot. */
export async function checkFeedback() {
  if (!feedbackThreadsOn()) return;
  await claimFeedbackKey();
  if (!session.peek() && !readFeedbackKey()) { feedbackUnread.value = 0; return; }
  let unread: { id: string; count: number }[];
  try { unread = (await api<{ unread?: { id: string; count: number }[] }>("GET", "trainworld/feedback/unread", undefined, { headers: keyHeaders() })).unread ?? []; }
  catch { return; } // offline or rate limited: the dot stays as it was
  feedbackUnread.value = unread.length;
}

let watching = false;
function startFeedbackWatch() {
  if (watching || !feedbackThreadsOn()) return;
  watching = true;
  effect(() => { void session.value; void checkFeedback(); }); // on load, and when a sign-in or sign-out changes whose threads these are
  setInterval(() => void checkFeedback(), FB_CHECK_MS);
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
  storeSession(await api<{ token: string; account: AccountView }>("POST", "noritetsu/" + path + (path === "register" ? SITE : ""), body, { auth: false }));
}

/** Names this game in the account server's mails (T-3104); the accounts are shared with noritetsu. */
export const SITE = "?site=trainsworld";

/** Deleting asks again: the password, or a fresh Google credential for an account with none (shonei-server's checkReauth). */
export async function deleteAccount(proof: { password: string } | { credential: string } = { password: "" }) {
  const acct = session.peek()?.account.id;
  await api("DELETE", "noritetsu/account", proof);
  tokenGone();
  lsSet(GAMES_KEY, null);
  // T-988: the games stay in this browser, tied to no account
  for (const s of await listSlots().catch(() => [])) if (acct && s.cloud?.acct === acct) await writeSync(s.key, null);
}

// ---------------------------------------------------------------- games on the account (T-988)

const DEVICE = (() => {
  let d = lsGet<string>(DEVICE_KEY);
  if (!d) { d = newGameId() + newGameId(); lsSet(DEVICE_KEY, d); }
  return d;
})();

/** What the account's games need, while signed in (workers/cloud.ts); `on` once the server says it
 * keeps games. */
export function cloudContext(): CloudContext | null {
  const s = session.peek();
  if (!BASE || !s) return null;
  const ids = Object.fromEntries(Object.entries(games()).map(([slot, e]) => [slot, e.id]));
  // a page opened while the server cannot be reached keeps the account's games as last known, so the
  // game plays with "Not on your account yet" and goes up once the server answers again
  const cfg = accountConfig.peek(), st = accountState.peek();
  const on = cfg ? st === "on" && !!cfg.trainworldGames : st === "unavailable" && lsGet<boolean>(GAMES_ON_KEY) === true;
  return { base: BASE, token: s.token, acct: s.account.id, device: DEVICE, on, ids };
}
/** The device id, for "Open on another device". */
export const deviceId = () => DEVICE;

export async function loadBoardSetting() {
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
interface GameEntry { id: string; acct?: string; sent?: string; at?: number; sandbox?: boolean; shared?: boolean }
const games = (): Record<string, GameEntry> => lsGet<Record<string, GameEntry>>(GAMES_KEY) ?? {};

/** The playing game's slot, when it is a stored one this tab writes (not a `?new=1` page). */
function playingSlot(): string | null {
  const s = saveSlotsState.peek();
  return s.writable && s.active && s.slots.some((x) => x.key === s.active) ? s.active : null;
}

/** T-2887: the stored game being played, read as a signal; null for a page with none to share (`?new=1`). */
export function shareableSlot(): string | null {
  void saveSlotsState.value;
  return playingSlot();
}

/** The playing game's id on the account's figures, if it has one yet. */
export function playingGameId(): string | null {
  const slot = playingSlot();
  if (!slot) return null;
  return saveSlotsState.peek().slots.find((x) => x.key === slot)?.cloud?.id ?? games()[slot]?.id ?? null;
}

/** T-3132: the stored game in this browser with this id (the leaderboards' and Community's), if any. */
export function slotOfGame(id: string): string | null {
  const local = games();
  return saveSlotsState.peek().slots.find((x) => x.cloud?.id === id || local[x.key]?.id === id)?.key ?? null;
}

/** T-2887: the id a share of the playing game goes up under, its figures sent first so the server copies the current
 * ones; null with no stored game or no session. A sandbox game's notice goes up even with no figures before it, or the
 * share would not be tagged. Remembered, so a share of a game later deleted here is taken down with it. */
export async function gameForShare(): Promise<string | null> {
  const s = session.peek(), slot = playingSlot(), m = moneyState.peek();
  if (!s || !slot) return null;
  await sendScores(true);
  const cid = saveSlotsState.peek().slots.find((x) => x.key === slot)?.cloud?.id;
  let e = games()[slot] ?? { id: cid ?? newGameId() };
  if (cid && e.id !== cid) e = { id: cid };
  if (m && unranked(m) && !e.sandbox) {
    try {
      await api("PUT", `trainworld/score/${e.id}`, { mode: "sandbox" });
      e.sandbox = true;
    } catch { /* shared untagged */ }
  }
  e.acct = s.account.id;
  e.shared = true;
  lsSet(GAMES_KEY, { ...games(), [slot]: e });
  return e.id;
}

/** The game's figures now: points in the past 24 game hours in all, and its best cities. */
function figures(): { day: number; points: number; cities: Record<string, number> } | null {
  const m = moneyState.peek();
  const hour = m?.hours.at(-1)?.hour;
  if (!m || hour === undefined) return null;
  const per = [...pointsPast24h.peek()].filter(([id, v]) => id && v > 0).sort((a, b) => b[1] - a[1]);
  // the server refuses a figure over 1e12 (twMaxPoints), which would leave a bigger game stuck on its last accepted one
  const cap = (v: number) => Math.min(Math.round(v), 1e12);
  const points = cap(per.reduce((a, [, v]) => a + v, 0));
  const cities: Record<string, number> = {};
  for (const [id, v] of per.slice(0, MAX_CITIES)) cities[id] = cap(v);
  return { day: Math.floor(hour / 24), points, cities };
}

/** A game whose figures never rank: a sandbox, or (T-3115) a copy of a shared network, which the server is told is a
 * sandbox so it never ranks there either, the copy's own play notwithstanding. */
const unranked = (m: { mode: string; copy?: boolean }) => m.mode === "sandbox" || m.copy === true;

let sending = false;
/** Send the playing game's figures if they changed and the gap has passed (`now` skips the gap). */
export async function sendScores(now = false) {
  const s = session.peek(), slot = playingSlot(), m = moneyState.peek();
  if (sending || !s || accountState.peek() !== "on" || !slot || !m) return;
  const all = games(), acct = s.account.id;
  // T-988: the slot's game id on the account names its figures too (a fork or a kept copy has a new one)
  const cid = saveSlotsState.peek().slots.find((x) => x.key === slot)?.cloud?.id;
  let e = all[slot] ?? { id: cid ?? newGameId() };
  if (cid && e.id !== cid) e = { id: cid };
  let body: unknown;
  if (unranked(m)) {
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
    if (unranked(m)) e.sandbox = true;
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

/** T-988: slots removed from this browser at sign-out whose games stay on the account, figures and all:
 * forgotten here, so they are not taken off as deleted. */
export function forgetScoreSlots(keys: string[]) {
  const all = games();
  for (const k of keys) delete all[k];
  lsSet(GAMES_KEY, all);
}

/** Slots gone from this browser take their figures off the account they were sent to. */
async function dropDeleted() {
  const s = session.peek(), st = saveSlotsState.peek();
  if (!st.slots.length) return; // storage unread or blocked: nothing is known to be gone
  const all = games(), keys = new Set(st.slots.map((x) => x.key));
  for (const [slot, e] of Object.entries(all)) {
    if (keys.has(slot) || slot === st.active) continue;
    if ((e.sent && !e.sandbox) || e.shared) {
      if (!s || e.acct !== s.account.id || accountState.peek() !== "on") continue; // later, signed in
      try { await api("DELETE", `trainworld/score/${e.id}`); } catch (err) { if ((err as ApiError).status !== 404) continue; }
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
  startGameTime();
  if (started || !BASE) return;
  started = true;
  // T-988: the clock worker gets the account on every change; a dead token found there signs out
  onTokenGone(tokenGone);
  onAccountGone(tokenGone);
  let posted = "";
  effect(() => {
    void session.value; void accountState.value; void accountConfig.value;
    const c = cloudContext(), k = JSON.stringify(c ? [c.token, c.acct, c.on] : null);
    if (k === posted) return;
    posted = k;
    postAccount(c);
  });
  const askConfig = async () => {
    try {
      const cfg = await api<AccountConfig>("GET", "noritetsu/config", undefined, { auth: false });
      lsSet(GAMES_ON_KEY, !!cfg.trainworldGames);
      accountConfig.value = cfg;
      accountState.value = cfg.signIn && cfg.trainworld ? "on" : "unavailable";
      return true;
    } catch {
      accountState.value = "unavailable";
      return false;
    }
  };
  void (async () => {
    // T-988: a page opened while the server could not be reached asks again, back online or every
    // CONFIG_RETRY_MS, so its account comes back without a reload
    if (!await askConfig()) await new Promise<void>((done) => {
      let busy = false;
      const again = async () => {
        if (busy) return;
        busy = true;
        if (await askConfig()) { clearInterval(t); removeEventListener("online", again); done(); }
        busy = false;
      };
      const t = setInterval(again, CONFIG_RETRY_MS);
      addEventListener("online", again);
    });
    // T-1771: plays and playtime, only against a server that says it takes them
    if (accountConfig.peek()?.trainworldPing) startPlaytime((p) => api("POST", "trainworld/ping", p));
    // T-3038: how many players have each achievement; sent without the session
    if (accountConfig.peek()?.trainworldAchievements) startAchievementCounts((method, path, body) => api(method, path, body, { auth: false }));
    if (accountState.peek() !== "on") { startFeedbackWatch(); return; }
    if (session.peek()) {
      try { storeSession(await api("POST", "noritetsu/refresh")); }
      catch { /* a 401 has signed out; offline keeps the token for next time */ }
    }
    startFeedbackWatch(); // T-2853: after the refresh, so a lapsed session is found out before a claim
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
