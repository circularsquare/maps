/** `?patch=` of the page (T-1029, T-1380): "0" works out the complexes and station names whole after
 * every change, "check" also compares each patched result with a whole one and reports a difference
 * as a console error. */
export const PATCH_MODE: string = typeof location === "undefined" ? "1" : new URLSearchParams(location.search).get("patch") ?? "1";
