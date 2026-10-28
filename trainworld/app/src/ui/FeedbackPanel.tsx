// T-1673: Settings' "Submit feedback" opens this modal: name and email (both optional) and a message,
// sent to the account server's /trainworld/feedback (game/account.ts sendFeedback), which stores it
// and mails it on. No address is in the page. What was typed stays until it is sent.

import { signal } from "@preact/signals";
import { useState } from "preact/hooks";
import { ApiError, sendFeedback } from "../game/account";
import { Modal } from "./Modal";
import { showPassingNote } from "./RidesNotes";
import "./account.css";

declare const __TW_BUILD__: string | undefined;
const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "dev";

export const feedbackOpen = signal(false);

function failText(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 0) return e.message.startsWith("This copy") ? e.message : "The server couldn't be reached. Try again in a minute.";
    if (e.status === 429) return "Too many messages from here for now. Try again in an hour.";
    return e.message;
  }
  return "Something went wrong. Try again in a minute.";
}

/** The Settings button. */
export function FeedbackButton() {
  return <div class="pane-foot">
    <button class={"btn text" + (feedbackOpen.value ? " on" : "")} onClick={() => (feedbackOpen.value = true)}>Submit feedback</button>
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
    if (!message.trim()) { setMsg("Write a message first."); return; }
    setState("sending"); setMsg("");
    try {
      await sendFeedback({ name: name.trim(), contact: contact.trim(), message: message.trim(), version: BUILD });
      setMessage(""); setState("");
      close();
      showPassingNote("sent! thank you for feedback");
    } catch (e) {
      setState("");
      setMsg(failText(e) + " Your message is still here.");
    }
  };
  return <Modal open={feedbackOpen.value} onClose={close} title="Feedback" cls="acct-panel fb-panel">
    <form class="acct-form" onSubmit={(e) => { e.preventDefault(); if (state !== "sending") void send(); }}>
      <input name="name" placeholder="Name (optional)" aria-label="Name" autocomplete="name" maxLength={100}
        value={name} onInput={(e) => setName((e.target as HTMLInputElement).value)} />
      <input name="email" type="email" placeholder="Email (optional)" aria-label="Email" autocomplete="email" maxLength={200}
        value={contact} onInput={(e) => setContact((e.target as HTMLInputElement).value)} />
      <textarea name="message" placeholder="Message" aria-label="Message" rows={6} maxLength={4000}
        value={message} onInput={(e) => setMessage((e.target as HTMLTextAreaElement).value)} />
      <div class="acct-btns">
        <button class="btn text acct-go" type="submit" disabled={state === "sending"}>{state === "sending" ? "Sending…" : "Send feedback"}</button>
      </div>
      {msg && <p class="acct-msg">{msg}</p>}
    </form>
  </Modal>;
}
