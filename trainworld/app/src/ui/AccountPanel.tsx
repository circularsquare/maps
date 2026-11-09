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
import { accountList, answerMergeAsk, gamesOn, mergeAsk, refreshAccountGames, watchMerge } from "../game/accountGames";
import { t, tRich, tNumber } from "../i18n";
import { Leaderboard } from "./Leaderboard";
import { Community, ShareDialog } from "./Community";
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
    s.onerror = () => fail(new Error(t("accountPanel.googleLoadFail")));
    document.head.appendChild(s);
  });
  gis.catch(() => { gis = null; });
  return gis;
}
let gisFor = "";
/** Where the button that is showing sends its credential; initialize is once per page, so it reads this. */
let onGoogle: (credential: string) => void = () => {};

function GoogleButton({ onMsg, onCredential, text = "signin_with" }: { onMsg: (m: string) => void; onCredential: (credential: string) => void; text?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  onGoogle = onCredential;
  useEffect(() => {
    const id = accountConfig.peek()?.googleClientId;
    if (!id) return;
    let live = true;
    loadGis().then(() => {
      const el = ref.current;
      if (!live || !el || typeof google === "undefined") return;
      if (gisFor !== id) {
        // the credential goes straight to the server: it holds the person's email and name
        google.accounts.id.initialize({ client_id: id, callback: (r: { credential: string }) => onGoogle(r.credential) });
        gisFor = id;
      }
      google.accounts.id.renderButton(el, { type: "standard", size: "medium", text, theme: shade.peek() === "dark" ? "filled_black" : "outline" });
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
  if (!cfg?.signIn) return <p>{t("accountPanel.unavailable")}</p>;
  return <>
    {cfg.googleClientId && <GoogleButton onMsg={setMsg} onCredential={(credential) => { signIn("signin", { provider: "google", credential }).catch((e) => setMsg(e.message)); }} />}
    {cfg.dev && <form class="acct-row" onSubmit={(e) => { e.preventDefault(); const sub = val(e.currentTarget, "sub").trim(); if (sub) run(() => signIn("signin", { provider: "dev", sub })); }}>
      <input name="sub" placeholder={t("accountPanel.devPlaceholder")} aria-label={t("accountPanel.devLabel")} />
      <button class="btn text" type="submit">{t("accountPanel.signIn")}</button>
    </form>}
    {cfg.password && (form === "forgot"
      ? <form class="acct-form" onSubmit={(e) => { e.preventDefault(); const email = val(e.currentTarget, "email"); run(async () => setMsg((await api<{ message?: string }>("POST", "noritetsu/forgot", { email }, { auth: false })).message ?? t("accountPanel.linkSent"))); }}>
        <input name="email" type="email" placeholder={t("accountPanel.recoveryEmail")} aria-label={t("accountPanel.recoveryEmail")} autocomplete="email" />
        <div class="acct-btns"><button class="btn text acct-go" type="submit">{t("accountPanel.sendReset")}</button><button class="btn text" type="button" onClick={() => { setForm("login"); setMsg(""); }}>{t("accountPanel.back")}</button></div>
      </form>
      : <form class="acct-form" onSubmit={(e) => {
        e.preventDefault();
        const f = e.currentTarget, username = val(f, "username"), password = val(f, "password"), email = val(f, "email").trim();
        run(() => form === "register" ? signIn("register", { username, password, ...(email ? { email } : {}) }) : signIn("login", { username, password }));
      }}>
        <input name="username" placeholder={t("accountPanel.username")} aria-label={t("accountPanel.username")} autocomplete="username" autocapitalize="none" spellcheck={false} />
        <input name="password" type="password" placeholder={t("accountPanel.password")} aria-label={t("accountPanel.password")} autocomplete={form === "register" ? "new-password" : "current-password"} />
        {form === "register" && <input name="email" type="email" placeholder={t("accountPanel.recoveryEmailOptional")} aria-label={t("accountPanel.recoveryEmail")} autocomplete="email" />}
        <div class="acct-btns">
          {form === "register"
            ? <><button class="btn text acct-go" type="submit">{t("accountPanel.createAccount")}</button><button class="btn text" type="button" onClick={() => { setForm("login"); setMsg(""); }}>{t("accountPanel.signInInstead")}</button></>
            : <><button class="btn text acct-go" type="submit">{t("accountPanel.signIn")}</button><button class="btn text" type="button" onClick={() => { setForm("register"); setMsg(""); }}>{t("accountPanel.createAccount")}</button><button class="btn text" type="button" onClick={() => { setForm("forgot"); setMsg(""); }}>{t("accountPanel.forgotPassword")}</button></>}
        </div>
      </form>)}
    {msg && <p class="acct-msg">{msg}</p>}
    <p class="acct-legal small muted"><a href="privacy.html" target="_blank">{t("accountPanel.privacy")}</a> · <a href="terms.html" target="_blank">{t("accountPanel.terms")}</a></p>
  </>;
}

function SignedIn({ a }: { a: AccountView }) {
  const [panel, setPanel] = useState<"" | "password" | "delete">("");
  const open = accountOpen.value;
  useEffect(() => { if (!open) setPanel(""); }, [open]);
  const [msg, setMsg] = useState("");
  const run = (f: () => Promise<unknown>) => { setMsg(""); f().catch((e) => setMsg(e.message)); };
  const password = a.logins.includes("password");
  const who = a.username ? tRich("accountPanel.signedInAs", { name: <b>{a.username}</b> }) : a.logins.includes("google") ? t("accountPanel.signedInGoogle") : t("accountPanel.signedIn");
  const setAccount = (account: AccountView) => { const s = session.peek(); if (s) storeSession({ token: s.token, account }); };
  const google = a.logins.includes("google");
  const saveEmail = (form: HTMLFormElement, email: string) => run(async () => setAccount((await api<{ account: AccountView }>("PUT", "noritetsu/email", { email, password: val(form, "emailPassword") })).account));
  const remove = (proof: { password: string } | { credential: string }) => run(async () => { await deleteAccount(proof); accountOpen.value = false; });
  if (panel === "password") return <form class="acct-form" onSubmit={(e) => {
    e.preventDefault();
    const f = e.currentTarget, o = val(f, "old"), n = val(f, "new");
    run(async () => { storeSession(await api("PUT", "noritetsu/password", { old: o, new: n }, { keep401: true })); setPanel(""); setMsg(t("accountPanel.passwordChanged")); });
  }}>
    <input name="old" type="password" placeholder={t("accountPanel.currentPassword")} aria-label={t("accountPanel.currentPassword")} autocomplete="current-password" />
    <input name="new" type="password" placeholder={t("accountPanel.newPassword")} aria-label={t("accountPanel.newPassword")} autocomplete="new-password" />
    <div class="acct-btns"><button class="btn text acct-go" type="submit">{t("accountPanel.save")}</button><button class="btn text" type="button" onClick={() => { setPanel(""); setMsg(""); }}>{t("accountPanel.back")}</button></div>
    {msg && <p class="acct-msg">{msg}</p>}
  </form>;
  return <>
    <p>{who}</p>
    <AccountGames />
    <MergeQuestion />
    {password && <form class="acct-form acct-sec" onSubmit={(e) => { e.preventDefault(); saveEmail(e.currentTarget, val(e.currentTarget, "email").trim()); }}>
      <input name="email" type="email" placeholder={t("accountPanel.recoveryEmail")} aria-label={t("accountPanel.recoveryEmail")} autocomplete="email" defaultValue={a.pendingEmail || a.email || ""} key={a.pendingEmail || a.email} />
      <input name="emailPassword" type="password" placeholder={t("accountPanel.currentPassword")} aria-label={t("accountPanel.currentPassword")} autocomplete="current-password" />
      {a.pendingEmail ? <p class="small muted">{t("accountPanel.confirmEmail", { email: a.pendingEmail })}</p>
        : !a.email && <p class="small acct-warn">{t("accountPanel.noRecoveryEmail")}</p>}
      <div class="acct-btns">
        <button class="btn text" type="submit">{t("accountPanel.saveEmail")}</button>
        {(a.email || a.pendingEmail) && <button class="btn text" type="button" onClick={(e) => saveEmail((e.currentTarget as HTMLButtonElement).form!, "")}>{t("accountPanel.remove")}</button>}
        <button class="btn text" type="button" onClick={() => { setPanel("password"); setMsg(""); }}>{t("accountPanel.changePassword")}</button>
      </div>
    </form>}
    <div class="acct-btns acct-sec">
      <button class="btn text" data-tip={t("accountPanel.signOutTip")} onClick={() => {
        // T-1851: the games stay tied to the account and sync again at the next sign-in (the "remove
        // games" choice, `signOutRemoving` in game/accountGames.ts, is no longer offered)
        signOut();
        accountOpen.value = false;
      }}>{t("accountPanel.signOut")}</button>
      <button class="btn text danger" onClick={() => setPanel("delete")}>{t("accountPanel.deleteAccount")}</button>
    </div>
    {panel === "delete" && <form class="callout bad acct-form" onSubmit={(e) => { e.preventDefault(); remove({ password: val(e.currentTarget, "deletePassword") }); }}>
      <p>{t("accountPanel.deleteQuestion")}</p>
      {password && <input name="deletePassword" type="password" placeholder={t("accountPanel.currentPassword")} aria-label={t("accountPanel.currentPassword")} autocomplete="current-password" />}
      {google && <><p>{t("accountPanel.deleteConfirmGoogle")}</p><GoogleButton onMsg={setMsg} onCredential={(credential) => remove({ credential })} text="continue_with" /></>}
      <div class="pane-foot">{!google && <button class="btn text danger" type="submit">{t("accountPanel.deleteAccount")}</button>}<button class="btn text" type="button" onClick={() => { setPanel(""); setMsg(""); }}>{t("accountPanel.keepIt")}</button></div>
    </form>}
    {msg && <p class="acct-msg">{msg}</p>}
  </>;
}

/** "0.2 MB", "12 MB". */
const mb = (bytes: number) => { const v = bytes / 1048576; return t("accountPanel.megabytes", { size: tNumber(v >= 10 ? Math.round(v) : Math.max(v, bytes ? 0.1 : 0), { maximumFractionDigits: 1 }) }); };

/** T-988: how many games the account keeps and the room they take. */
function AccountGames() {
  void accountState.value; void accountConfig.value; void session.value;
  const list = accountList.value, open = accountOpen.value;
  useEffect(() => { if (open) void refreshAccountGames(); }, [open]);
  if (!gamesOn() || !list) return null;
  return <p class="small muted num acct-games">
    <span data-tip={t("accountPanel.gamesTip", { max: list.maxGames })}>{t("accountPanel.gamesCount", { count: list.games.length })}</span>
    {" · "}
    <span data-tip={t("accountPanel.backupsTip")}>{t("accountPanel.storageUsed", { used: mb(list.used), limit: mb(list.limit) })}</span>
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
      <p>{one ? t("accountPanel.mergeOne", { name: ask[0].name }) : t("accountPanel.mergeMany", { count: tNumber(ask.length) })}</p>
      <div class="acct-btns"><button class="btn text acct-go" onClick={() => void answerMergeAsk(true)}>{t("accountPanel.add")}</button><button class="btn text" onClick={() => setSure(true)}>{t("accountPanel.dontAdd")}</button></div>
    </> : <>
      <p>{t("accountPanel.mergeSure")}</p>
      <div class="acct-btns"><button class="btn text" onClick={() => { setSure(false); void answerMergeAsk(false); }}>{one ? t("accountPanel.keepHereOne") : t("accountPanel.keepHereMany")}</button><button class="btn text acct-go" onClick={() => { setSure(false); void answerMergeAsk(true); }}>{one ? t("accountPanel.addIt") : t("accountPanel.addThem")}</button></div>
    </>}
  </div>;
}

export function AccountPanel() {
  const s = session.value;
  return <Modal open={accountOpen.value} onClose={() => (accountOpen.value = false)} title={s ? t("accountPanel.account") : t("accountPanel.signIn")} cls="acct-panel acct-account">
    {s ? <SignedIn a={s.account} /> : <SignInForms />}
  </Modal>;
}

/** The person icon: the start menu's (`start`) or the top of Settings (`settings`); nothing until accounts are on. */
export function AccountButton({ kind }: { kind: "start" | "settings" }) {
  if (accountState.value !== "on") return null;
  const label = session.value ? t("accountPanel.account") : t("accountPanel.signIn");
  return <button class={kind === "settings" ? "btn icon" + (accountOpen.value ? " on" : "") : "btn icon acct-start"} data-tip={label} aria-label={label}
    onClick={() => (accountOpen.value = !accountOpen.peek())}><Icon.person /></button>;
}

/** T-1728: who is logged in, for the text beside Settings' account icon and its Account row. */
export function loginText(): string {
  const a = session.value?.account;
  return !a ? t("accountPanel.notLoggedIn") : a.username ? t("accountPanel.loggedInAs", { name: a.username }) : a.logins.includes("google") ? t("accountPanel.loggedInGoogle") : t("accountPanel.loggedIn");
}

/** T-1139: Settings' Account row, in words beside the icon at its top: who is signed in, and the button
 * that opens the same panel as the start menu's (sign in, change password, sign out, delete). */
export function AccountSection() {
  if (accountState.value !== "on") return null;
  const a = session.value?.account;
  const who = loginText();
  return <>
    <div class="h">{t("accountPanel.account")}</div>
    <div class="game-mode">
      <span>{who}</span>
      <button class={"btn text" + (accountOpen.value ? " on" : "")} onClick={() => (accountOpen.value = !accountOpen.peek())}>{a ? t("accountPanel.accountSettings") : t("accountPanel.signIn")}</button>
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
  render(h(() => <><AccountPanel /><Leaderboard /><Community /><ShareDialog /></>, null), root);
}
