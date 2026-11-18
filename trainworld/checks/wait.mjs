// T-3666: wait for a line in a file (a build's or a background job's log) instead of polling by hand or sleeping.
//   node trainworld/checks/wait.mjs <file> <pattern> <timeout-s> [--new] [--pid N] [--fail <pattern>]
// Prints the first line matching <pattern> (a regular expression, case-insensitive) and exits 0. A match already in
// the file counts, unless --new (then only lines written after the wait starts). A file not there yet is waited for.
// Exits 1 with the file's last lines on: the timeout, the process --pid ending first, or a line matching --fail
// (e.g. 'error|panicked'). The timeout is at most 170 s: a tool call still running at 180 s is moved to the
// background, so a longer job is waited for in a second call.
import { existsSync, readFileSync, statSync } from 'node:fs';

const args = process.argv.slice(2), pos = [], opt = {};
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--new') opt.new = true;
  else if (args[i].startsWith('--')) opt[args[i].slice(2)] = args[++i];
  else pos.push(args[i]);
}
const [file, pattern, secs] = pos;
const timeout = Number(secs);
if (!file || !pattern || !(timeout > 0)) { console.log('usage: node trainworld/checks/wait.mjs <file> <pattern> <timeout-s> [--new] [--pid N] [--fail <pattern>]'); process.exit(2); }
if (timeout > 170) { console.log(`timeout ${timeout} s is over 170 (a tool call is moved to the background at 180 s): wait again in a second call`); process.exit(2); }
const want = new RegExp(pattern, 'i'), bad = opt.fail ? new RegExp(opt.fail, 'i') : null, pid = opt.pid ? Number(opt.pid) : null;
const alive = (p) => { try { process.kill(p, 0); return true; } catch (e) { return e.code === 'EPERM'; } };
const size = () => { try { return statSync(file).size; } catch { return 0; } };
const tail = (text, n = 8) => text.trimEnd().split(/\r?\n/).slice(-n).join('\n');

const t0 = Date.now();
let from = opt.new ? size() : 0;
for (;;) {
  let text = '';
  if (existsSync(file)) {
    try { text = readFileSync(file, 'utf8'); } catch { /* being written */ }
    if (text.length < from) from = 0; // the file was started again
    for (const line of text.slice(from).split(/\r?\n/)) {
      if (want.test(line)) { console.log(line.trim()); process.exit(0); }
      if (bad?.test(line)) { console.log(`failed: ${line.trim()}\n--- last lines of ${file}\n${tail(text)}`); process.exit(1); }
    }
  }
  const s = Math.round((Date.now() - t0) / 1000);
  if (pid && !alive(pid)) { console.log(`process ${pid} ended after ${s} s with no "${pattern}"\n--- last lines of ${file}\n${tail(text)}`); process.exit(1); }
  if (Date.now() - t0 > timeout * 1000) { console.log(`timed out after ${timeout} s waiting for "${pattern}"${existsSync(file) ? `\n--- last lines of ${file}\n${tail(text)}` : ` (${file} was never written)`}`); process.exit(1); }
  await new Promise((r) => setTimeout(r, 500));
}
