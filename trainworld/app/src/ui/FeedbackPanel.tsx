// T-1673: Settings' "Submit feedback" opens this modal: name and email (both optional) and a message,
// sent to the account server's /trainworld/feedback (game/account.ts sendFeedback), which stores it
// and mails it on. No address is in the page. What was typed stays until it is sent.

import { signal } from "@preact/signals";
import { useState } from "preact/hooks";
import { ApiError, sendFeedback } from "../game/account";
import { t } from "../i18n";
import { Modal } from "./Modal";
import { showPassingNote } from "./PassingNote";
import "./account.css";

declare const __TW_BUILD__: string | undefined;
const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "dev";

export const feedbackOpen = signal(false);

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
      await sendFeedback({ name: name.trim(), contact: contact.trim(), message: message.trim(), version: BUILD });
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
