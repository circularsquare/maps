import { t, type MessageKey } from "../i18n";
import { addressLimited, mobileDevice } from "../workers/deviceMemory";
// T-2545: what a load that fails says. A memory failure says so; any other or unknown one says it
// failed. T-2867: a memory failure's second sentence is what can help on this device: a phone is told
// about computers, a 32-bit browser about a 64-bit one, and a computer to close other tabs. Pure, so the
// clock worker can use it too.

const outOfMemoryKey = (): MessageKey =>
  mobileDevice() ? "loadText.outOfMemory" : addressLimited() ? "loadText.outOfMemory32" : "loadText.outOfMemoryDesktop";

/** The two sentences, read when needed so they follow the language. */
export const outOfMemoryText = () => t(outOfMemoryKey());
export const loadFailedText = () => t("loadText.failed");

/** Whether an error's text is a failed allocation: Rust's "out of memory" (`last_panic`), V8's "could not
 * allocate" and "Array buffer allocation failed", and WebAssembly.Memory refusing to grow. A dead worker's
 * text always names its WASM size and a panic always aborts as `unreachable`, so those alone do not count (T-1891). */
export const isMemoryFailure = (why: string) => /out of memory|could not allocate|allocation failed|WebAssembly\.Memory|Array buffer allocation/i.test(why);

/** The sentence for a load that failed with `why`. */
export const loadFailKey = (why: string): MessageKey => (isMemoryFailure(why) ? outOfMemoryKey() : "loadText.failed");
export const loadFailText = (why: string) => t(loadFailKey(why));
