// T-3269: a player's flag beside their name (leaderboard rows, Community cards), and the account panel's picker.

import { useMemo, useRef, useState } from "preact/hooks";
import { boardSetting } from "../game/account";
import { FLAG_COUNTRIES, FLAG_MORE, flagName, flagUrl, flagsOn, isFlag, setFlag } from "../game/flags";
import { normalise } from "../game/search";
import { t } from "../i18n";
import "./flag.css";

/** The flag's picture, its name in the hover; nothing for no flag or a code this page does not know. */
export function Flag({ code }: { code?: string | null }) {
  if (!isFlag(code)) return null;
  const name = flagName(code);
  return <img class="flag" src={flagUrl(code)} alt={name} data-tip={name} width={14} height={14} loading="lazy" decoding="async" />;
}

const collate = new Intl.Collator(undefined, { sensitivity: "base" }).compare;
/** a name matches a typed query at the start of any of its words */
const hit = (name: string, q: string) => ` ${normalise(name)}`.includes(` ${normalise(q)}`);

/** The account panel's row: the flag now shown, and a search over every flag (no flag first, then the game's own few,
 * then the countries by name). */
export function FlagSetting() {
  const b = boardSetting.value;
  const [text, setText] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const [msg, setMsg] = useState("");
  const list = useRef<HTMLUListElement>(null);
  const items = useMemo(() => {
    if (text === null) return [];
    const q = text.trim();
    const countries = [...FLAG_COUNTRIES].sort((a, b) => collate(flagName(a), flagName(b)));
    const all = ["", ...FLAG_MORE, ...countries];
    return q ? all.filter((c) => c && (hit(flagName(c), q) || c === q.toLowerCase())) : all;
  }, [text]);
  if (!flagsOn() || !b) return null;
  const current = isFlag(b.flag) ? b.flag : "";
  const label = (c: string) => (c ? flagName(c) : t("accountPanel.noFlag"));
  const pick = (code: string) => {
    setText(null);
    (document.activeElement as HTMLElement | null)?.blur();
    if (code === current) return;
    setMsg("");
    setFlag(code).catch((e) => setMsg(e.message));
  };
  const move = (to: number) => {
    setCursor(to);
    list.current?.children[to]?.scrollIntoView({ block: "nearest" });
  };
  return <div class="acct-flag">
    <div class="acct-flag-row">
      <span data-tip={t("accountPanel.flagTip")}>{t("accountPanel.flag")}</span>
      <div class="acct-flag-pick">
        <span class="acct-flag-now">{current ? <Flag code={current} /> : <span class="flag flag-none" />}</span>
        <input type="text" class="new-game-find" aria-label={t("accountPanel.flag")} placeholder={t("accountPanel.flagSearch")} maxLength={40}
          value={text ?? label(current)}
          onFocus={(e) => { setText(""); setCursor(0); (e.target as HTMLInputElement).select(); }}
          onBlur={() => setText(null)}
          onInput={(e) => { setText((e.target as HTMLInputElement).value); setCursor(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown" && items.length) { e.preventDefault(); move((cursor + 1) % items.length); }
            else if (e.key === "ArrowUp" && items.length) { e.preventDefault(); move((cursor + items.length - 1) % items.length); }
            else if (e.key === "Enter" && items.length) { e.preventDefault(); pick(items[Math.min(cursor, items.length - 1)]); }
            else if (e.key === "Escape" && text !== null) { e.preventDefault(); e.stopPropagation(); setText(null); (e.target as HTMLInputElement).blur(); }
          }} />
        {items.length > 0 && <ul class="new-game-hits acct-flag-hits" role="listbox" ref={list}>
          {items.map((c, i) => (
            <li role="option" aria-selected={i === cursor} class={(i === cursor ? "on" : "") + (c === current ? " picked" : "")} onMouseDown={(e) => { e.preventDefault(); pick(c); }}>
              <span>{c ? <img class="flag" src={flagUrl(c)} alt="" width={14} height={14} decoding="async" /> : <span class="flag flag-none" />}{label(c)}</span>
            </li>
          ))}
        </ul>}
      </div>
    </div>
    {msg && <p class="acct-msg">{msg}</p>}
  </div>;
}
