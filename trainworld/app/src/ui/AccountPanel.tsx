// T-972: the account panel, a centred modal like the saves panel, opened by the person icon at the
// top of Settings or on the start menu (`AccountButton`). Signed out: Google's button, then a
// username and password with Sign in, Create account and Forgot password, as noritetsu has.
// Signed in: who, the recovery email (password accounts), Change password, Sign out and Delete
// account. The panels mount in a root of their own (`startAccountUi`), so the
// start menu, which hides the dock, can open them too.

import { h, render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { accountConfig, accountOpen, accountState, api, deleteAccount, session, signIn, signOut, startAccount, storeSession, type AccountView } from "../game/account";
import { shade } from "../game/ui";
import { accountList, answerMergeAsk, gamesOn, mergeAsk, refreshAccountGames, signOutRemoving, tiedHere, watchMerge } from "../game/accountGames";
import { plural } from "../game/format";
import { clockRunning } from "../workers/clockClient";
import { Leaderboard } from "./Leaderboard";
import { Modal } from "./Modal";
import { Icon } from "./widgets";
import "./account.css";

declare const google: any;
let gis: Promise<void> | null = null;
/** Google's sign-in script, loaded the first time the panel opens, never with the game. */
function loadGis(): Promise<void> {
  gis ??= new Promise<void>((ok, fail) => {
    const s = document.createElement("script");
    s.src = "https://accounts.google.com/gsi/client";
    s.async = true;
    s.onload = () => ok();
    s.onerror = () => fail(new Error("Google sign-in could not load."));
    document.head.appendChild(s);
  });
  gis.catch(() => { gis = null; });
  return gis;
}
let gisFor = "";

function GoogleButton({ onMsg }: { onMsg: (m: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const id = accountConfig.peek()?.googleClientId;
    if (!id) return;
    let live = true;
    loadGis().then(() => {
      const el = ref.current;
      if (!live || !el || typeof google === "undefined") return;
      if (gisFor !== id) {
        // the credential goes straight to the server: it holds the person's email and name
        google.accounts.id.initialize({ client_id: id, callback: (r: { credential: string }) => signIn("signin", { provider: "google", credential: r.credential }).catch((e) => onMsg(e.message)) });
        gisFor = id;
      }
      google.accounts.id.renderButton(el, { type: "standard", size: "medium", text: "signin_with", theme: shade.peek() === "dark" ? "filled_black" : "outline" });
    }).catch((e) => onMsg(e.message));
    return () => { live = false; };
  }, []);
  return <div class="acct-google" ref={ref} />;
}

const val = (form: HTMLFormElement, name: string) => ((form.elements.namedItem(name) as HTMLInputElement | null)?.value ?? "");

function SignInForms() {
  const cfg = accountConfig.value;
  const [form, setForm] = useState<"login" | "register" | "forgot">("login");
  const [msg, setMsg] = useState("");
  const run = (f: () => Promise<unknown>) => { setMsg(""); f().catch((e) => setMsg(e.message)); };
  if (!cfg?.signIn) return <p>Signing in is not available right now.</p>;
  return <>
    {cfg.googleClientId && <GoogleButton onMsg={setMsg} />}
    {cfg.dev && <form class="acct-row" onSubmit={(e) => { e.preventDefault(); const sub = val(e.currentTarget, "sub").trim(); if (sub) run(() => signIn("signin", { provider: "dev", sub })); }}>
      <input name="sub" placeholder="Made-up id (dev server)" aria-label="Made-up id" />
      <button class="btn text" type="submit">Sign in</button>
    </form>}
    {cfg.password && (form === "forgot"
      ? <form class="acct-form" onSubmit={(e) => { e.preventDefault(); const email = val(e.currentTarget, "email"); run(async () => setMsg((await api<{ message?: string }>("POST", "noritetsu/forgot", { email }, { auth: false })).message ?? "Link sent.")); }}>
        <input name="email" type="email" placeholder="Recovery email" aria-label="Recovery email" autocomplete="email" />
        <div class="acct-btns"><button class="btn text acct-go" type="submit">Send reset link</button><button class="btn text" type="button" onClick={() => { setForm("login"); setMsg(""); }}>Back</button></div>
      </form>
      : <form class="acct-form" onSubmit={(e) => {
        e.preventDefault();
        const f = e.currentTarget, username = val(f, "username"), password = val(f, "password"), email = val(f, "email").trim();
        run(() => form === "register" ? signIn("register", { username, password, ...(email ? { email } : {}) }) : signIn("login", { username, password }));
      }}>
        <input name="username" placeholder="Username" aria-label="Username" autocomplete="username" autocapitalize="none" spellcheck={false} />
        <input name="password" type="password" placeholder="Password" aria-label="Password" autocomplete={form === "register" ? "new-password" : "current-password"} />
        {form === "register" && <input name="email" type="email" placeholder="Recovery email (optional)" aria-label="Recovery email" autocomplete="email" />}
        <div class="acct-btns">
          {form === "register"
            ? <><button class="btn text acct-go" type="submit">Create account</button><button class="btn text" type="button" onClick={() => { setForm("login"); setMsg(""); }}>Sign in instead</button></>
            : <><button class="btn text acct-go" type="submit">Sign in</button><button class="btn text" type="button" onClick={() => { setForm("register"); setMsg(""); }}>Create account</button><button class="btn text" type="button" onClick={() => { setForm("forgot"); setMsg(""); }}>Forgot password</button></>}
        </div>
      </form>)}
    {msg && <p class="acct-msg">{msg}</p>}
    <p class="acct-legal small muted"><a href="privacy.html" target="_blank">Privacy</a> · <a href="terms.html" target="_blank">Terms</a></p>
  </>;
}

function SignedIn({ a }: { a: AccountView }) {
  const [panel, setPanel] = useState<"" | "password" | "delete" | "signout">("");
  const open = accountOpen.value;
  useEffect(() => { if (!open) setPanel(""); }, [open]);
  const [msg, setMsg] = useState("");
  const run = (f: () => Promise<unknown>) => { setMsg(""); f().catch((e) => setMsg(e.message)); };
  const password = a.logins.includes("password");
  const who = a.username ? <>Signed in as <b>{a.username}</b></> : a.logins.includes("google") ? "Signed in with Google" : "Signed in";
  const setAccount = (account: AccountView) => { const s = session.peek(); if (s) storeSession({ token: s.token, account }); };
  if (panel === "password") return <form class="acct-form" onSubmit={(e) => {
    e.preventDefault();
    const f = e.currentTarget, o = val(f, "old"), n = val(f, "new");
    run(async () => { storeSession(await api("PUT", "noritetsu/password", { old: o, new: n }, { keep401: true })); setPanel(""); setMsg("Password changed."); });
  }}>
    <input name="old" type="password" placeholder="Current password" aria-label="Current password" autocomplete="current-password" />
    <input name="new" type="password" placeholder="New password" aria-label="New password" autocomplete="new-password" />
    <div class="acct-btns"><button class="btn text acct-go" type="submit">Save</button><button class="btn text" type="button" onClick={() => { setPanel(""); setMsg(""); }}>Back</button></div>
    {msg && <p class="acct-msg">{msg}</p>}
  </form>;
  return <>
    <p>{who}</p>
    <AccountGames />
    <MergeQuestion />
    {password && <form class="acct-form acct-sec" onSubmit={(e) => { e.preventDefault(); const email = val(e.currentTarget, "email").trim(); run(async () => setAccount((await api<{ account: AccountView }>("PUT", "noritetsu/email", { email })).account)); }}>
      <input name="email" type="email" placeholder="Recovery email" aria-label="Recovery email" autocomplete="email" value={a.pendingEmail || a.email || ""} key={a.pendingEmail || a.email} />
      {a.pendingEmail ? <p class="small muted">Confirm your email at {a.pendingEmail}.</p>
        : !a.email && <p class="small acct-warn">No recovery email</p>}
      <div class="acct-btns">
        <button class="btn text" type="submit">Save email</button>
        {(a.email || a.pendingEmail) && <button class="btn text" type="button" onClick={() => run(async () => setAccount((await api<{ account: AccountView }>("PUT", "noritetsu/email", { email: "" })).account))}>Remove</button>}
        <button class="btn text" type="button" onClick={() => { setPanel("password"); setMsg(""); }}>Change password</button>
      </div>
    </form>}
    <div class="acct-btns acct-sec">
      <button class={"btn text" + (panel === "signout" ? " on" : "")} data-tip="Signs out of anitabuilder and noritetsu in this browser" onClick={async () => {
        // T-988: with this account's games in this browser, whether they stay is asked first
        if (gamesOn() && await tiedHere()) { setPanel("signout"); return; }
        signOut();
        accountOpen.value = false;
      }}>Sign out</button>
      <button class="btn text danger" onClick={() => setPanel("delete")}>Delete account</button>
    </div>
    {panel === "signout" && <div class="callout acct-signout">
      <p>Remove your games from this browser?</p>
      <div class="acct-btns">
        <button class="btn text acct-go" data-tip="Your games stay in this browser and sync again when you sign in" onClick={() => { signOut(); accountOpen.value = false; }}>Keep games</button>
        <button class="btn text" data-tip={`Games saved to your account leave this browser. A game with changes not on your account yet stays.${clockRunning() ? " The game you are playing closes." : ""}`}
          onClick={() => { setPanel(""); accountOpen.value = false; signOutRemoving(); }}>Remove games</button>
      </div>
    </div>}
    {panel === "delete" && <div class="callout bad">
      <p>Delete your account? Its games, its leaderboard entries and its noritetsu rides go too. Games in this browser stay.</p>
      <div class="pane-foot"><button class="btn text danger" onClick={() => run(async () => { await deleteAccount(); accountOpen.value = false; })}>Delete account</button><button class="btn text" onClick={() => setPanel("")}>Keep it</button></div>
    </div>}
    {msg && <p class="acct-msg">{msg}</p>}
  </>;
}

/** "0.2 MB", "12 MB". */
const mb = (bytes: number) => { const v = bytes / 1048576; return `${v >= 10 ? Math.round(v) : Math.max(v, bytes ? 0.1 : 0).toFixed(1).replace(/\.0$/, "")} MB`; };

/** T-988: how many games the account keeps and the room they take. */
function AccountGames() {
  void accountState.value; void accountConfig.value; void session.value;
  const list = accountList.value, open = accountOpen.value;
  useEffect(() => { if (open) void refreshAccountGames(); }, [open]);
  if (!gamesOn() || !list) return null;
  return <p class="small muted num acct-games">
    <span data-tip={`Your account keeps up to ${list.maxGames} games`}>{plural(list.games.length, "game")} on your account</span>
    {" · "}
    <span data-tip="Backups count too">{mb(list.used)} of {mb(list.limit)}</span>
  </p>;
}

/** T-988: signing in with games in this browser asks whether they go on the account. */
function MergeQuestion() {
  const ask = mergeAsk.value;
  const [sure, setSure] = useState(false);
  if (!ask) return null;
  const one = ask.length === 1;
  return <div class="callout acct-merge">
    {!sure ? <>
      <p>{one ? `Add ${ask[0].name} to your account?` : `Add the ${ask.length} games in this browser to your account?`}</p>
      <div class="acct-btns"><button class="btn text acct-go" onClick={() => void answerMergeAsk(true)}>Add</button><button class="btn text" onClick={() => setSure(true)}>Don't add</button></div>
    </> : <>
      <p>Are you sure? Games kept only in this browser are lost if its storage is cleared.</p>
      <div class="acct-btns"><button class="btn text" onClick={() => { setSure(false); void answerMergeAsk(false); }}>{one ? "Keep it here only" : "Keep them here only"}</button><button class="btn text acct-go" onClick={() => { setSure(false); void answerMergeAsk(true); }}>{one ? "Add it" : "Add them"}</button></div>
    </>}
  </div>;
}

export function AccountPanel() {
  const s = session.value;
  return <Modal open={accountOpen.value} onClose={() => (accountOpen.value = false)} title={s ? "Account" : "Sign in"} cls="acct-panel acct-account">
    {s ? <SignedIn a={s.account} /> : <SignInForms />}
  </Modal>;
}

/** The person icon: the start menu's (`start`) or the top of Settings (`settings`); nothing until accounts are on. */
export function AccountButton({ kind }: { kind: "start" | "settings" }) {
  if (accountState.value !== "on") return null;
  const label = session.value ? "Account" : "Sign in";
  return <button class={kind === "settings" ? "btn icon" + (accountOpen.value ? " on" : "") : "btn icon acct-start"} data-tip={label} aria-label={label}
    onClick={() => (accountOpen.value = !accountOpen.peek())}><Icon.person /></button>;
}

/** T-1139: Settings' Account row, in words beside the icon at its top: who is signed in, and the button
 * that opens the same panel as the start menu's (sign in, change password, sign out, delete). */
export function AccountSection() {
  if (accountState.value !== "on") return null;
  const a = session.value?.account;
  const who = !a ? "Not signed in" : a.username ? `Signed in as ${a.username}` : a.logins.includes("google") ? "Signed in with Google" : "Signed in";
  return <>
    <div class="h">Account</div>
    <div class="game-mode">
      <span>{who}</span>
      <button class={"btn text" + (accountOpen.value ? " on" : "")} onClick={() => (accountOpen.value = !accountOpen.peek())}>{a ? "Account settings" : "Sign in"}</button>
    </div>
  </>;
}

let mounted = false;
/** Start the account (`startAccount`) and mount its panels. Idempotent; the start menu and the dock
 * both call it. */
export function startAccountUi() {
  startAccount();
  watchMerge(); // T-988
  if (mounted || accountState.peek() === "off") return;
  mounted = true;
  const root = document.createElement("div");
  root.id = "account-root";
  document.body.appendChild(root);
  render(h(() => <><AccountPanel /><Leaderboard /></>, null), root);
}
