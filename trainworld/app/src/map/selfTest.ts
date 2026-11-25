/** checks/t4153-net.mjs --self-test=<name>: a patched pass of the drawn network's build with one input left out (named where
 * it is left out), which the check must catch. Off (empty) in the game. */
export let fault = "";
export function setFault(name: string) { fault = name; }
