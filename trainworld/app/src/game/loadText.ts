import { t, type MessageKey } from "../i18n";
// T-2545: what a load that fails says. A memory failure says so; any other or unknown one says it
// failed. Both add that the game works best on a computer. Pure, so the clock worker can use it too.

/** The two sentences, read when needed so they follow the language. */
export const outOfMemoryText = () => t("loadText.outOfMemory");
export const loadFailedText = () => t("loadText.failed");

/** Whether an error's text is a failed allocation: Rust's "out of memory" (`last_panic`), V8's "could not
 * allocate" and "Array buffer allocation failed", and WebAssembly.Memory refusing to grow. A dead worker's
 * text always names its WASM size and a panic always aborts as `unreachable`, so those alone do not count (T-1891). */
export const isMemoryFailure = (why: string) => /out of memory|could not allocate|allocation failed|WebAssembly\.Memory|Array buffer allocation/i.test(why);

/** The sentence for a load that failed with `why`. */
export const loadFailKey = (why: string): MessageKey => (isMemoryFailure(why) ? "loadText.outOfMemory" : "loadText.failed");
export const loadFailText = (why: string) => t(loadFailKey(why));
