// T-972: the account panel, a centred modal like the saves panel, opened by the person icon at the
// top of Settings or on the start menu (`AccountButton`). Signed out: Google's button, then a
// username and password with Log in, Create an account and Forgot password, as noritetsu has.
// Signed in: who, the recovery email (password accounts), Change password, Sign out everywhere else,
// Sign out and Delete account. The panels mount in a root of their own (`startAccountUi`), so the
// start menu, which hides the dock, can open them too.

import { h, render } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { accountConfig, accountOpen, accountState, api, deleteAccount, session, signIn, signOut, startAccount, storeSession, type AccountView } from "../game/account";
import { shade } from "../game/ui";
import { Leaderboard } from "./Leaderboard";
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
            ? <><button class="btn text acct-go" type="submit">Create account</button><button class="btn text" type="button" onClick={() => { setForm("login"); setMsg(""); }}>Log in instead</button></>
            : <><button class="btn text acct-go" type="submit">Log in</button><button class="btn text" type="button" onClick={() => { setForm("register"); setMsg(""); }}>Create an account</button><button class="btn text" type="button" onClick={() => { setForm("forgot"); setMsg(""); }}>Forgot password</button></>}
        </div>
      </form>)}
    {msg && <p class="acct-msg">{msg}</p>}
    <p class="acct-legal small muted"><a href="privacy.html" target="_blank">Privacy</a> · <a href="terms.html" target="_blank">Terms</a></p>
  </>;
}

function SignedIn({ a }: { a: AccountView }) {
  const [panel, setPanel] = useState<"" | "password" | "delete">("");
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
    {password && <form class="acct-form acct-sec" onSubmit={(e) => { e.preventDefault(); const email = val(e.currentTarget, "email").trim(); run(async () => setAccount((await api<{ account: AccountView }>("PUT", "noritetsu/email", { email })).account)); }}>
      <input name="email" type="email" placeholder="Recovery email" aria-label="Recovery email" autocomplete="email" value={a.pendingEmail || a.email || ""} key={a.pendingEmail || a.email} />
      {a.pendingEmail ? <p class="small muted">Confirm your email at {a.pendingEmail}.</p>
        : !a.email && <p class="small acct-warn">Add a recovery email. Without one, a forgotten password means a lost account.</p>}
      <div class="acct-btns">
        <button class="btn text" type="submit">Save email</button>
        {(a.email || a.pendingEmail) && <button class="btn text" type="button" onClick={() => run(async () => setAccount((await api<{ account: AccountView }>("PUT", "noritetsu/email", { email: "" })).account))}>Remove</button>}
        <button class="btn text" type="button" onClick={() => { setPanel("password"); setMsg(""); }}>Change password</button>
      </div>
    </form>}
    <div class="acct-btns acct-sec">
      <button class="btn text" data-tip="Signs out of anitabuilder and noritetsu in this browser" onClick={() => { signOut(); accountOpen.value = false; }}>Sign out</button>
      <button class="btn text" data-tip="Every other browser signed in to this account signs out" onClick={() => run(async () => { storeSession(await api("POST", "noritetsu/signout-all")); setMsg("Signed out everywhere else."); })}>Sign out everywhere else</button>
      <button class="btn text danger" onClick={() => setPanel("delete")}>Delete account</button>
    </div>
    {panel === "delete" && <div class="callout bad">
      <p>Delete your account? Its leaderboard entries and its noritetsu rides go too. Games in this browser stay.</p>
      <div class="pane-foot"><button class="btn text danger" onClick={() => run(async () => { await deleteAccount(); accountOpen.value = false; })}>Delete account</button><button class="btn text" onClick={() => setPanel("")}>Keep it</button></div>
    </div>}
    {msg && <p class="acct-msg">{msg}</p>}
  </>;
}

/** A centred modal: a native dialog, Esc or a click outside its box closes it. */
export function Modal({ open, onClose, title, cls, children }: { open: boolean; onClose: () => void; title: string; cls: string; children: preact.ComponentChildren }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useLayoutEffect(() => {
    const el = dialog.current;
    if (!open || !el) return;
    const before = document.activeElement;
    if (!el.open) el.showModal();
    return () => {
      if (el.open) el.close();
      if (before instanceof HTMLElement && before.isConnected) before.focus();
    };
  }, [open]);
  if (!open) return null;
  return <dialog class={"saves-panel acct-panel " + cls} ref={dialog} aria-label={title}
    onCancel={(e) => { e.preventDefault(); onClose(); }}
    onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Escape") { e.preventDefault(); onClose(); } }}>
    <div class="saves-inner">
      <div class="saves-head">
        <h2>{title}</h2>
        <button class="btn icon" aria-label="Close" data-tip="Close (Esc)" onClick={onClose}><Icon.cross /></button>
      </div>
      {children}
    </div>
  </dialog>;
}

export function AccountPanel() {
  const s = session.value;
  return <Modal open={accountOpen.value} onClose={() => (accountOpen.value = false)} title={s ? "Account" : "Sign in"} cls="acct-account">
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

let mounted = false;
/** Start the account (`startAccount`) and mount its panels. Idempotent; the start menu and the dock
 * both call it. */
export function startAccountUi() {
  startAccount();
  if (mounted || accountState.peek() === "off") return;
  mounted = true;
  const root = document.createElement("div");
  root.id = "account-root";
  document.body.appendChild(root);
  render(h(() => <><AccountPanel /><Leaderboard /></>, null), root);
}
