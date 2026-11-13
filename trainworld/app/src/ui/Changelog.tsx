// T-2810: the list of what changed in each update, a Modal; opened from the start menu's update row and
// from Settings (T-3220).

import { Modal } from "./Modal";
import { UPDATES } from "./updates";
import { mobileDevice } from "../workers/deviceMemory";
import { t, tDate, tRich } from "../i18n";

export function Changelog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return <Modal open={open} onClose={onClose} title={t("startMenu.changelog")} cls="changelog">
    {UPDATES.map((u) => <section key={u.version}>
      <h3 class="changelog-head">{u.version}<span class="muted small">{tDate(Date.parse(u.at), { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" })}</span></h3>
      <ul class="changelog-list">{u.changes.filter((k) => !(mobileDevice() && k === "updates.v01DeleteKeys")).map((k) => { const fig = u.nerfs?.[k]; return <li key={k}>{fig ? tRich(k, { figure: <span class="nerf">{t(fig)}</span> }) : t(k)}{u.sub?.[k] && <ul>{u.sub[k]!.map((s) => <li key={s}>{t(s)}</li>)}</ul>}</li>; })}</ul>
    </section>)}
  </Modal>;
}
