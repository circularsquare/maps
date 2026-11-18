// The one text box every search field uses: a light grey clear button at the right while it holds text (T-3715).

import type { JSX, Ref } from "preact";
import { t } from "../i18n";

type Props = Omit<JSX.IntrinsicElements["input"], "ref"> & {
  /** classes for the wrapper, which takes the box's width and margins */
  wrap?: string;
  inputRef?: Ref<HTMLInputElement>;
  onClear: () => void;
  /** whether the clear button shows; by default when the box holds text */
  showClear?: boolean;
  /** a muted icon inside the box, at the left */
  icon?: JSX.Element;
};

/** The magnifying glass of the find place tool; `size` in px */
export function SearchIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
      <circle cx="5.75" cy="5.75" r="4.25" />
      <path d="M8.9 8.9 L13 13" stroke-linecap="round" />
    </svg>
  );
}

export function FindInput({ wrap, inputRef, onClear, showClear, icon, ...input }: Props) {
  const on = showClear ?? !!input.value;
  return (
    <span class={"find-box" + (icon ? " with-icon" : "") + (wrap ? " " + wrap : "")}>
      {icon && <span class="find-icon">{icon}</span>}
      <input ref={inputRef} {...(input as object)} />
      {on && <button type="button" class="find-clear" aria-label={t("blueprint.clear")} tabIndex={-1}
        onMouseDown={(e) => e.preventDefault()}
        onClick={(e) => { onClear(); (e.currentTarget.previousElementSibling as HTMLInputElement | null)?.focus(); }}>
        <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
      </button>}
    </span>
  );
}
