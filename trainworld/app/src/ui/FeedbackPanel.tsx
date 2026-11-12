// T-1673: Settings' "Submit feedback" opens this modal: name and email (both optional) and a message,
// sent to the account server's /trainworld/feedback (game/account.ts sendFeedback), which stores it
// and mails it on. No address is in the page. What was typed stays until it is sent.
//
// T-2853: against a server that keeps feedback threads (`trainworldFeedbackThreads` in its config) the form
// also takes a tag, and below it are the player's earlier feedback with Anita's replies, opening in place,
// with a box to write back and a delete. A dot on the Submit feedback button and on the Settings tab says
// there is a reply (game/account.ts checks on load and every 30 minutes). Against any other server none
// of that shows and the form sends as before.

import { signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import {
  ApiError, canListFeedback, deleteFeedback, feedbackThreadsOn, feedbackUnread, getFeedbackThread, listFeedback, markFeedbackRead,
  replyFeedback, sendFeedback, session, type FeedbackRow, type FeedbackThread,
} from "../game/account";
import { loadFailure } from "../game/loading";
import { stateReport } from "../game/diagnostics";
import { display } from "../game/ui";
import { language, t, tDate } from "../i18n";
import { Modal } from "./Modal";
import { showPassingNote } from "./PassingNote";
import { Icon } from "./widgets";
import "./account.css";

declare const __TW_BUILD__: string | undefined;
const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "dev";

export const feedbackOpen = signal(false);

/** T-2853: the tag buttons: the `kind` the server stores, and the label. */
const TAGS = [["language", "feedbackPanel.tagLanguage"], ["error", "feedbackPanel.tagError"], ["feature", "feedbackPanel.tagFeature"]] as const;

/** T-2826: the browser and version from the user agent ("Firefox 159", "Chrome 131"); Edge and Opera
 * say Chrome too, so they are looked for first. Anything unknown is sent as the agent's own text. */
export function browserOf(ua: string): string {
  const m = /(Edg|OPR|Firefox|FxiOS|CriOS|Chrome|Version)\/(\d+)/.exec(ua);
  if (!m) return ua.slice(0, 120);
  const name = ({ Edg: "Edge", OPR: "Opera", FxiOS: "Firefox", CriOS: "Chrome", Version: /Safari/.test(ua) ? "Safari" : "Browser" } as Record<string, string>)[m[1]] ?? m[1];
  return name + " " + m[2];
}

/** T-2826: what a report carries beside the message: the game version, the browser, whether the device
 * reports its memory (Chrome and Edge do, Firefox and Safari do not) and how much, and why demand last
 * failed to load, if it did. T-2853: and the game's language, so Anita knows what to answer in. */
function reportContext(): Record<string, string> {
  const dm = typeof navigator === "undefined" ? undefined : (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const ctx: Record<string, string> = {
    version: BUILD,
    browser: typeof navigator === "undefined" ? "unknown" : browserOf(navigator.userAgent),
    memory: typeof dm === "number" ? dm + " GB" : "not reported",
    language: language.peek(),
  };
  const failure = loadFailure.peek();
  if (failure) ctx.lastFailure = failure.slice(0, 1000);
  // T-3313: where the clock worker and the demand pool stood, and the money's currency (another device can
  // start on another one)
  ctx.state = stateReport();
  ctx.currency = display.peek().currency;
  return ctx;
}

function failText(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.code === "too-many") return t("feedbackPanel.tooManyThreads");
    if (e.code === "long") return t("feedbackPanel.threadLong");
    if (e.code === "full") return t("feedbackPanel.full");
    if (e.status === 0) return e.message.startsWith("This copy") ? e.message : t("feedbackPanel.unreachable");
    if (e.status === 429) return t("feedbackPanel.tooMany");
    return e.message;
  }
  return t("feedbackPanel.wrong");
}

/** The dot for a reply not yet read; nothing when there is none. Sits at the corner of its parent. */
export function NewReplyDot() {
  if (!feedbackUnread.value) return null;
  return <span class="fb-dot" role="img" aria-label={t("feedbackPanel.newReply")} />;
}

/** The hover line for what carries the dot; undefined when no reply is unread. */
export const replyTip = () => (feedbackUnread.value ? t("feedbackPanel.hasReply") : undefined);

/** The Settings button. */
export function FeedbackButton() {
  return <div class="pane-foot">
    <button class={"btn text fb-open" + (feedbackOpen.value ? " on" : "")} data-tip={replyTip()} onClick={() => (feedbackOpen.value = true)}>{t("feedbackPanel.open")}<NewReplyDot /></button>
    <a class="btn icon discord-link" href="https://discord.gg/RemhwVcZFw" target="_blank" rel="noopener noreferrer" data-tip={t("settingsTab.discord")} aria-label={t("settingsTab.discord")}>
      <Icon.discord />
    </a>
  </div>;
}

export function FeedbackPanel() {
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [message, setMessage] = useState("");
  const [kind, setKind] = useState("");
  const [state, setState] = useState<"" | "sending">("");
  const [msg, setMsg] = useState("");
  const close = () => { feedbackOpen.value = false; };
  const threads = feedbackThreadsOn();
  const send = async () => {
    if (!message.trim()) { setMsg(t("feedbackPanel.empty")); return; }
    setState("sending"); setMsg("");
    try {
      await sendFeedback({ name: name.trim(), contact: contact.trim(), message: message.trim(), context: reportContext(), kind: kind || undefined });
      setMessage(""); setKind(""); setState("");
      close();
      showPassingNote(t("feedbackPanel.thanks"));
    } catch (e) {
      setState("");
      setMsg(t("feedbackPanel.failed", { reason: failText(e) }));
    }
  };
  return <Modal open={feedbackOpen.value} onClose={close} title={t("feedbackPanel.title")} cls="acct-panel fb-panel">
    <form class="acct-form" onSubmit={(e) => { e.preventDefault(); if (state !== "sending") void send(); }}>
      <input name="name" placeholder={t("feedbackPanel.namePlaceholder")} aria-label={t("feedbackPanel.nameLabel")} autocomplete="name" maxLength={100}
        value={name} onInput={(e) => setName((e.target as HTMLInputElement).value)} />
      <input name="email" type="email" placeholder={t("feedbackPanel.emailPlaceholder")} aria-label={t("feedbackPanel.emailLabel")} autocomplete="email" maxLength={200}
        value={contact} onInput={(e) => setContact((e.target as HTMLInputElement).value)} />
      <textarea name="message" placeholder={t("feedbackPanel.messageLabel")} aria-label={t("feedbackPanel.messageLabel")} rows={6} maxLength={4000}
        value={message} onInput={(e) => setMessage((e.target as HTMLTextAreaElement).value)} />
      {threads && <div class="fb-tags" role="group" aria-label={t("feedbackPanel.tagsLabel")}>
        {TAGS.map(([id, key]) => (
          <button type="button" class={"btn text" + (kind === id ? " on" : "")} aria-pressed={kind === id} onClick={() => setKind(kind === id ? "" : id)}>{t(key)}</button>
        ))}
      </div>}
      <div class="acct-btns">
        <button class="btn text acct-go" type="submit" disabled={state === "sending"}>{state === "sending" ? t("feedbackPanel.sending") : t("feedbackPanel.send")}</button>
      </div>
      {msg && <p class="acct-msg">{msg}</p>}
    </form>
    {feedbackOpen.value && <Earlier />}
  </Modal>;
}

/** "Your earlier feedback": this browser's or account's threads, newest first; nothing at all when the
 * server has the threads off (its routes answer 404), when there are none, or when the list cannot be read
 * and there is nothing to say about it. */
function Earlier() {
  const [rows, setRows] = useState<FeedbackRow[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [openId, setOpenId] = useState("");
  const [thread, setThread] = useState<FeedbackThread | null>(null);
  const [reply, setReply] = useState("");
  const [sending, setSending] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [err, setErr] = useState("");
  const current = useRef("");
  const signedIn = !!session.value;

  useEffect(() => {
    if (!canListFeedback()) return;
    let live = true;
    listFeedback().then((r) => { if (live) setRows(r); }).catch((e) => { if (live && !(e instanceof ApiError && e.status === 404)) setFailed(true); });
    return () => { live = false; };
  }, [signedIn]);

  const recount = (rs: FeedbackRow[]) => { feedbackUnread.value = rs.filter((x) => x.unread > 0).length; };
  const patch = (id: string, f: (r: FeedbackRow) => FeedbackRow) => setRows((rs) => {
    const next = (rs ?? []).map((r) => (r.id === id ? f(r) : r));
    recount(next);
    return next;
  });

  const toggle = async (row: FeedbackRow) => {
    if (openId === row.id) { current.current = ""; setOpenId(""); return; }
    current.current = row.id;
    setOpenId(row.id); setThread(null); setReply(""); setConfirm(false); setErr("");
    try {
      const th = await getFeedbackThread(row.id);
      if (current.current !== row.id) return;
      setThread(th);
      if (th.unread > 0) {
        await markFeedbackRead(row.id);
        patch(row.id, (r) => ({ ...r, unread: 0 }));
      }
    } catch (e) { if (current.current === row.id) setErr(failText(e)); }
  };
  const sendReply = async () => {
    if (!thread || !reply.trim() || sending) return;
    setSending(true); setErr("");
    try {
      const th = await replyFeedback(thread.id, reply.trim());
      setThread(th); setReply("");
      patch(th.id, (r) => ({ ...r, updated: th.updated, messages: th.messages.length }));
    } catch (e) { setErr(failText(e)); }
    setSending(false);
  };
  const remove = async () => {
    if (!thread) return;
    const id = thread.id;
    setErr("");
    try {
      await deleteFeedback(id);
      current.current = ""; setOpenId(""); setThread(null); setConfirm(false);
      setRows((rs) => { const next = (rs ?? []).filter((r) => r.id !== id); recount(next); return next; });
    } catch (e) { setErr(failText(e)); }
  };

  if (failed) return <p class="acct-msg">{t("feedbackPanel.loadFailed")}</p>;
  if (!rows || !rows.length) return null;
  const when = (at: number) => tDate(at * 1000, { day: "numeric", month: "short" });
  return <section class="fb-earlier">
    <div class="h">{t("feedbackPanel.earlier")}</div>
    {!signedIn && <p class="fb-note">{t("feedbackPanel.browserOnly")}</p>}
    <ul class="fb-rows">
      {rows.map((r) => {
        const open = openId === r.id;
        return <li key={r.id} class={open ? "open" : ""}>
          <button type="button" class="fb-row" aria-expanded={open} onClick={() => void toggle(r)}>
            <span class="fb-first">{r.first}</span>
            <span class="fb-date">{when(r.updated)}</span>
            {r.unread > 0 && <span class="fb-dot inline" role="img" aria-label={t("feedbackPanel.newReply")} />}
          </button>
          {open && thread && thread.id === r.id && <div class="fb-thread">
            {thread.messages.map((m) => (
              <div class={"fb-msg " + m.from}>
                <div class="fb-who"><span>{m.from === "anita" ? t("feedbackPanel.anita") : t("feedbackPanel.you")}</span><span>{when(m.at)}</span></div>
                <div class="fb-text">{m.text}</div>
              </div>
            ))}
            <textarea rows={3} maxLength={4000} placeholder={t("feedbackPanel.replyLabel")} aria-label={t("feedbackPanel.replyLabel")}
              value={reply} onInput={(e) => setReply((e.target as HTMLTextAreaElement).value)} />
            <div class="acct-btns">
              <button type="button" class="btn text acct-go" disabled={sending || !reply.trim()} onClick={() => void sendReply()}>{t("feedbackPanel.sendReply")}</button>
              {!confirm && <button type="button" class="btn text" onClick={() => setConfirm(true)}>{t("feedbackPanel.delete")}</button>}
            </div>
            {confirm && <div class="fb-confirm">
              <p>{t("feedbackPanel.deleteConfirm")}</p>
              <div class="acct-btns">
                <button type="button" class="btn text fb-del" onClick={() => void remove()}>{t("feedbackPanel.delete")}</button>
                <button type="button" class="btn text" onClick={() => setConfirm(false)}>{t("feedbackPanel.keep")}</button>
              </div>
            </div>}
            {err && <p class="acct-msg">{err}</p>}
          </div>}
          {open && !thread && err && <p class="acct-msg">{err}</p>}
        </li>;
      })}
    </ul>
  </section>;
}
