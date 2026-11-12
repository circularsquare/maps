// T-3269, T-3279: a player's flag after their name (leaderboard rows, Community cards), and the account panel's flag
// button beside the display name, which opens the picker.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { boardSetting } from "../game/account";
import { flagCountry, flagGroups, flagName, flagUrl, flagsOn, homeCountry, isFlag, setFlag } from "../game/flags";
import { normalise } from "../game/search";
import { t } from "../i18n";
import "./flag.css";

/** The flag's picture, its name in the hover; nothing for no flag or a code this page does not know. `small` beside
 * small grey text. */
export function Flag({ code, small }: { code?: string | null; small?: boolean }) {
  if (!isFlag(code)) return null;
  const name = flagName(code);
  return <img class={"flag" + (small ? " small" : "")} src={flagUrl(code)} alt={name} data-tip={name} loading="lazy" decoding="async" />;
}

/** A name with its flag after it; a long name is cut short, the flag never. */
export function Named({ name, flag, small }: { name: string; flag?: string | null; small?: boolean }) {
  return <span class="named"><span class="named-text">{name}</span><Flag code={flag} small={small} /></span>;
}

const collate = new Intl.Collator(undefined, { sensitivity: "base" }).compare;
/** a name matches a typed query at the start of any of its words */
const hit = (name: string, q: string) => !!name && ` ${normalise(name)}`.includes(` ${normalise(q)}`);
const byName = (codes: readonly string[]) => [...codes].sort((a, b) => collate(flagName(a), flagName(b)));

/** The account panel's row: the flag (an empty outline while there is none) and the display name after it. The flag
 * opens the picker. */
export function FlagSetting() {
  const b = boardSetting.value;
  const [open, setOpen] = useState(false);
  const [msg, setMsg] = useState("");
  const btn = useRef<HTMLButtonElement>(null);
  if (!flagsOn() || !b) return null;
  const current = isFlag(b.flag) ? b.flag : "";
  const pick = (code: string) => {
    setOpen(false);
    if (code === current) return;
    setMsg("");
    setFlag(code).catch((e) => setMsg(e.message));
  };
  return <div class="acct-flag">
    <div class="acct-flag-row">
      <button type="button" class="acct-flag-btn" ref={btn} aria-haspopup="listbox" aria-expanded={open}
        aria-label={current ? flagName(current) : t("accountPanel.flag")}
        data-tip={(current ? flagName(current) + "\n" : "") + t("accountPanel.flagTip")}
        onClick={() => setOpen(!open)}>
        {current ? <img class="flag" src={flagUrl(current)} alt="" decoding="async" /> : <span class="flag flag-none" />}
      </button>
      {b.name && <span class="acct-flag-name">{b.name}</span>}
    </div>
    {open && <FlagPicker anchor={btn} current={current} onPick={pick} onClose={() => setOpen(false)} />}
    {msg && <p class="acct-msg">{msg}</p>}
  </div>;
}

type Item = { code: string; gap?: boolean };

/** The list over the page, under the flag button (above it when there is more room there), free to reach past the
 * panel: no flag (when there is one), the country the player is probably in, every country, other places, the rest. */
function FlagPicker({ anchor, current, onPick, onClose }: { anchor: { current: HTMLElement | null }; current: string; onPick: (code: string) => void; onClose: () => void }) {
  const [text, setText] = useState("");
  const [cursor, setCursor] = useState(0);
  const [home, setHome] = useState("");
  const [at, setAt] = useState<{ left: number; top?: number; bottom?: number; height: number } | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => { void homeCountry().then(setHome); }, []);
  useLayoutEffect(() => {
    const place = () => {
      const a = anchor.current;
      if (!a) return;
      // the panel is zoomed by the UI scale: the box's left and top are in its units
      const z = (a as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1;
      const r = a.getBoundingClientRect(), vw = innerWidth / z, vh = innerHeight / z;
      const left = Math.max(8, Math.min(r.left / z, vw - 300 - 8));
      const below = vh - r.bottom / z - 12, above = r.top / z - 12;
      if (below >= 320 || below >= above) setAt({ left, top: r.bottom / z + 4, height: Math.min(480, below) });
      else setAt({ left, bottom: vh - r.top / z + 4, height: Math.min(480, above) });
    };
    place();
    addEventListener("resize", place);
    return () => removeEventListener("resize", place);
  }, []);
  const placed = at !== null;
  useEffect(() => {
    if (placed && matchMedia("(pointer: fine)").matches) input.current?.focus(); // on a phone the keyboard would cover the list
  }, [placed]);
  useEffect(() => {
    const away = (e: PointerEvent) => {
      const el = e.target as Node;
      if (!box.current?.contains(el) && !anchor.current?.contains(el)) onClose();
    };
    // Esc closes the picker, not the panel, wherever the focus is (the search has none on a phone)
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); onClose(); } };
    document.addEventListener("pointerdown", away, true);
    document.addEventListener("keydown", esc, true);
    return () => { document.removeEventListener("pointerdown", away, true); document.removeEventListener("keydown", esc, true); };
  }, []);

  const items = useMemo<Item[]>(() => {
    const g = flagGroups();
    const q = text.trim();
    const groups = [byName(g.countries), byName(g.places), [...g.more]];
    if (q) {
      const found = groups.map((codes) => codes.filter((c) => hit(flagName(c), q) || hit(flagCountry(c), q) || c === q.toLowerCase()));
      return found.filter((f) => f.length).flatMap((f, i) => f.map((code, j) => ({ code, gap: i > 0 && j === 0 })));
    }
    const top = [...(current ? [""] : []), ...(home ? [home] : [])];
    return [top, ...groups].filter((f) => f.length).flatMap((f, i) => f.map((code, j) => ({ code, gap: i > 0 && j === 0 })));
  }, [text, home, current]);

  const move = (to: number) => {
    setCursor(to);
    list.current?.querySelectorAll("li")[to]?.scrollIntoView({ block: "nearest" });
  };
  if (!at) return null;
  return <div class="flag-pop" ref={box} style={{ left: `${at.left}px`, top: at.top === undefined ? "auto" : `${at.top}px`, bottom: at.bottom === undefined ? "auto" : `${at.bottom}px`, maxHeight: `${at.height}px` }}>
    <input type="text" ref={input} class="new-game-find" aria-label={t("accountPanel.flag")} placeholder={t("accountPanel.flagSearch")} maxLength={40}
      value={text} autocomplete="off" spellcheck={false}
      onInput={(e) => { setText((e.target as HTMLInputElement).value); setCursor(0); if (list.current) list.current.scrollTop = 0; }}
      onKeyDown={(e) => {
        if (e.key === "ArrowDown" && items.length) { e.preventDefault(); move((cursor + 1) % items.length); }
        else if (e.key === "ArrowUp" && items.length) { e.preventDefault(); move((cursor + items.length - 1) % items.length); }
        else if (e.key === "Enter" && items.length) { e.preventDefault(); onPick(items[Math.min(cursor, items.length - 1)].code); }
      }} />
    <ul role="listbox" ref={list}>
      {items.map(({ code, gap }, i) => (
        <li role="option" aria-selected={i === cursor} class={(i === cursor ? "on" : "") + (code === current ? " picked" : "") + (gap ? " gap" : "")}
          onMouseDown={(e) => { e.preventDefault(); onPick(code); }}>
          {code ? <img class="flag" src={flagUrl(code)} alt="" loading="lazy" decoding="async" /> : <span class="flag flag-none" />}
          <span class="flag-pop-name">{code ? flagName(code) : t("accountPanel.noFlag")}</span>
          {code && flagCountry(code) && <span class="flag-pop-where">{flagCountry(code)}</span>}
        </li>
      ))}
    </ul>
  </div>;
}
