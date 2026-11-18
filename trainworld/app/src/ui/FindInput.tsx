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
};

export function FindInput({ wrap, inputRef, onClear, showClear, ...input }: Props) {
  const on = showClear ?? !!input.value;
  return (
    <span class={"find-box" + (wrap ? " " + wrap : "")}>
      <input ref={inputRef} {...(input as object)} />
      {on && <button type="button" class="find-clear" aria-label={t("blueprint.clear")} tabIndex={-1}
        onMouseDown={(e) => e.preventDefault()}
        onClick={(e) => { onClear(); (e.currentTarget.previousElementSibling as HTMLInputElement | null)?.focus(); }}>
        <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
      </button>}
    </span>
  );
}
