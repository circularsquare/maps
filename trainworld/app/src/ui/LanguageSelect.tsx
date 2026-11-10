// T-2636, T-2753: the language picker, a globe and a select of the languages each in its own name; in Settings'
// Language row and on the start menu. A pick shows at once and is remembered on this device (i18n chooseLanguage).

import { useState } from "preact/hooks";
import { chooseLanguage, isLang, language, LANG_NAMES, pickable, t } from "../i18n";
import { noteStaleBuild } from "../game/staleBuild";
import { Icon } from "./widgets";

export function LanguageSelect({ class: cls = "" }: { class?: string }) {
  // as the purpose select (T-169): the focus ring for keyboard focus only
  const [mouse, setMouse] = useState(false);
  if (pickable.length < 2) return null;
  const tip = t("settingsTab.language") + "\n" + t("settingsTab.languageAi");
  return (
    <div class={"lang-row " + cls}>
      <span class="lang-icon" data-tip={tip}><Icon.globe /></span>
      <label class={"purpose-filter" + (mouse ? " mouse" : "")}>
        <select aria-label={t("settingsTab.language")} value={language.value}
          onPointerDown={() => setMouse(true)} onKeyDown={() => setMouse(false)} onBlur={() => setMouse(false)}
          onChange={(e) => { const v = (e.currentTarget as HTMLSelectElement).value; if (isLang(v)) chooseLanguage(v).catch(noteStaleBuild); }}>
          {pickable.map((l) => <option key={l} value={l} lang={l}>{LANG_NAMES[l]}</option>)}
        </select>
      </label>
    </div>
  );
}
