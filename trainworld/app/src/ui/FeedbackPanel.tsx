// T-1673: Settings' "Submit feedback" opens this modal: name and email (both optional) and a message,
// sent to the account server's /trainworld/feedback (game/account.ts sendFeedback), which stores it
// and mails it on. No address is in the page. What was typed stays until it is sent.

import { signal } from "@preact/signals";
import { useState } from "preact/hooks";
import { ApiError, sendFeedback } from "../game/account";
import { loadFailure } from "../game/loading";
import { t } from "../i18n";
import { Modal } from "./Modal";
import { showPassingNote } from "./PassingNote";
import { Icon } from "./widgets";
import "./account.css";

declare const __TW_BUILD__: string | undefined;
const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "dev";

export const feedbackOpen = signal(false);

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
 * failed to load, if it did. */
function reportContext(): Record<string, string> {
  const dm = typeof navigator === "undefined" ? undefined : (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const ctx: Record<string, string> = {
    version: BUILD,
    browser: typeof navigator === "undefined" ? "unknown" : browserOf(navigator.userAgent),
    memory: typeof dm === "number" ? dm + " GB" : "not reported",
  };
  const failure = loadFailure.peek();
  if (failure) ctx.lastFailure = failure.slice(0, 1000);
  return ctx;
}

function failText(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 0) return e.message.startsWith("This copy") ? e.message : t("feedbackPanel.unreachable");
    if (e.status === 429) return t("feedbackPanel.tooMany");
    return e.message;
  }
  return t("feedbackPanel.wrong");
}

/** The Settings button. */
export function FeedbackButton() {
  return <div class="pane-foot">
    <button class={"btn text" + (feedbackOpen.value ? " on" : "")} onClick={() => (feedbackOpen.value = true)}>{t("feedbackPanel.open")}</button>
    <a class="btn icon discord-link" href="https://discord.gg/RemhwVcZFw" target="_blank" rel="noopener noreferrer" data-tip={t("settingsTab.discord")} aria-label={t("settingsTab.discord")}>
      <Icon.discord />
    </a>
  </div>;
}

export function FeedbackPanel() {
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [message, setMessage] = useState("");
  const [state, setState] = useState<"" | "sending">("");
  const [msg, setMsg] = useState("");
  const close = () => { feedbackOpen.value = false; };
  const send = async () => {
    if (!message.trim()) { setMsg(t("feedbackPanel.empty")); return; }
    setState("sending"); setMsg("");
    try {
      await sendFeedback({ name: name.trim(), contact: contact.trim(), message: message.trim(), context: reportContext() });
      setMessage(""); setState("");
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
      <div class="acct-btns">
        <button class="btn text acct-go" type="submit" disabled={state === "sending"}>{state === "sending" ? t("feedbackPanel.sending") : t("feedbackPanel.send")}</button>
      </div>
      {msg && <p class="acct-msg">{msg}</p>}
    </form>
  </Modal>;
}
