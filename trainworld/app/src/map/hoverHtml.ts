// The hover tag's lines are HTML (map/modeBar.ts: bars and dot rows), but some carry text from a save
// (station and line names), so the tag is filled through this allowlist (T-2914): div, span and i, with
// the few classes and style properties the bars use. Any other element is dropped and its text kept;
// every other attribute (on*, href, src) is dropped.

const TAGS = new Set(["DIV", "SPAN", "I"]);
const CLASSES = new Set(["mode-bar", "tag-facts", "tag-fact", "purpose-row", "tag-dot"]);
const STYLE: Record<string, RegExp> = {
  "flex-grow": /^\d+(\.\d+)?(e[+-]?\d+)?$/i,
  background: /^(#[0-9a-f]{3,8}|var\(--[\w-]+\))$/i,
  width: /^\d+(\.\d+)?%$/,
};

function cleanStyle(s: string): string {
  const out: string[] = [];
  for (const decl of s.split(";")) {
    const i = decl.indexOf(":");
    if (i < 0) continue;
    const k = decl.slice(0, i).trim().toLowerCase(), v = decl.slice(i + 1).trim();
    if (STYLE[k]?.test(v)) out.push(`${k}:${v}`);
  }
  return out.join(";");
}

function copy(from: Node, into: Node) {
  for (const n of Array.from(from.childNodes)) {
    if (n.nodeType === Node.TEXT_NODE) into.appendChild(document.createTextNode(n.textContent ?? ""));
    else if (n.nodeType === Node.ELEMENT_NODE) {
      const e = n as Element;
      if (!TAGS.has(e.tagName)) { copy(e, into); continue; }
      const o = document.createElement(e.tagName.toLowerCase());
      const cls = (e.getAttribute("class") ?? "").split(/\s+/).filter((c) => CLASSES.has(c)).join(" ");
      if (cls) o.className = cls;
      const st = cleanStyle(e.getAttribute("style") ?? "");
      if (st) o.setAttribute("style", st);
      copy(e, o);
      into.appendChild(o);
    }
  }
}

/** The tag's lines as a fragment holding only what hovers use. */
export function safeHoverHtml(html: string): DocumentFragment {
  const tpl = document.createElement("template");
  tpl.innerHTML = html; // inert: nothing in a template loads or runs
  const out = document.createDocumentFragment();
  copy(tpl.content, out);
  return out;
}

/** Text for a hover line: a name from a save shows as itself, never as markup. */
export const escapeHtml = (s: string): string => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
