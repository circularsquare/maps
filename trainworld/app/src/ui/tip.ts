// The game's own hover tag for the page's DOM (T-210; CLAUDE.md: every hover is the game's own
// tag, never a browser `title` tooltip). An element with `data-tip` gets a tag by the pointer the
// moment the pointer is over it, in the map tags' style (map/demandHover.ts), following the pointer
// and kept inside the window; a "\n" in the text starts a new line. The map's canvas has its own
// tags (demand discs, track and network, capacity), which this leaves alone.
//
// Panels, the top bar, the search box and the map's DOM markers write `data-tip` where they once
// wrote `title`; SVG drawings put it on the shape instead of a <title> child. As a net for markup
// the game does not write (MapLibre's attribution button), a `title` met under the pointer is moved
// to `data-tip` at once, before the browser's own tooltip would show.

export function startTips() {
  const tag = document.createElement("div");
  tag.className = "tool-tag ui-tag";
  tag.setAttribute("role", "tooltip");
  tag.style.display = "none";
  document.body.appendChild(tag);
  let el: Element | null = null;
  let text = "";
  let px = 0, py = 0;

  const hide = () => {
    if (el) watch.disconnect();
    el = null;
    text = "";
    tag.style.display = "none";
  };
  const place = () => {
    const w = tag.offsetWidth, h = tag.offsetHeight;
    // below right of the pointer; flipped where it would leave the window
    let x = px + 14, y = py + 16;
    if (x + w > innerWidth - 4) x = Math.max(4, px - 10 - w);
    if (y + h > innerHeight - 4) y = Math.max(4, py - 10 - h);
    tag.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  };
  const fill = () => {
    const t = el?.getAttribute("data-tip") ?? "";
    if (!t || !el?.isConnected) return hide();
    if (t !== text) {
      text = t;
      tag.replaceChildren(...t.split("\n").map((line) => {
        const d = document.createElement("div");
        d.textContent = line;
        return d;
      }));
    }
    tag.style.display = "";
    place();
  };
  // the text can change under a still pointer (the clock's day, a count updating)
  const watch = new MutationObserver(fill);

  const move = (e: PointerEvent) => {
    if (e.pointerType === "touch") return;
    if (e.buttons) return hide();
    px = e.clientX;
    py = e.clientY;
    const target = e.target instanceof Element ? e.target : null;
    const found = target?.closest("[data-tip], [title]") ?? null;
    if (found?.hasAttribute("title")) {
      const t = found.getAttribute("title");
      found.removeAttribute("title");
      if (t && !found.hasAttribute("data-tip")) found.setAttribute("data-tip", t);
    }
    if (found !== el) {
      if (el) watch.disconnect();
      el = found;
      text = "";
      if (!el) return hide();
      watch.observe(el, { attributes: true, attributeFilter: ["data-tip"] });
    }
    fill();
  };
  document.addEventListener("pointermove", move, { capture: true, passive: true });
  document.addEventListener("pointerover", move, { capture: true, passive: true });
  // a press, a scroll, a key or the pointer leaving the page puts it away; the next move brings it back
  document.addEventListener("pointerdown", hide, true);
  document.addEventListener("wheel", hide, { capture: true, passive: true });
  document.addEventListener("scroll", hide, true);
  document.addEventListener("keydown", hide, true);
  document.documentElement.addEventListener("pointerleave", hide);
  window.addEventListener("blur", hide);
}
