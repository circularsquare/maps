// T-2630: a check's Chrome keeps to its own CPU cores. An unpinned headless Chrome took ~15 of 16 cores
// (2026-11-06). lib.mjs's launchChrome uses this, so every browser check is covered; a check never calls it.
//   TW_CORES=n          use n cores, no ledger claim (the caller owns the budget)
//   TW_CORE_MASK=0xC000 which cores (hex mask); default: the top cores down, skipping those a live check holds
//   TW_CORES_WANT=n     cores to ask the ledger for when TW_CORES is unset (default 3)
//   TW_NOPIN=1          no pinning and no ?demandWorkers (timing runs against a quiet machine)
// With TW_CORES unset it claims from the ledger (`python cores.py claim`, at the maps root) and
// releases at exit. The page gets ?demandWorkers = cores - 1 (at least 1), added by an init script.
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { cpus } from 'node:os';
import { basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const MAPS = fileURLToPath(new URL('../../', import.meta.url));
const ARTIFACTS = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));

const REGISTRY = ARTIFACTS + 'check-cores.json';
let PLAN = null;

const alive = (pid) => { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } };
const readRegistry = () => { try { return JSON.parse(readFileSync(REGISTRY, 'utf8')); } catch { return {}; } };
const writeRegistry = (r) => { try { mkdirSync(ARTIFACTS, { recursive: true }); writeFileSync(REGISTRY, JSON.stringify(r)); } catch { /* best effort */ } };

/** `n` core numbers from the top down, skipping those in `taken` (a bit mask) while enough remain */
function pickCores(n, total, taken) {
  const free = [], used = [];
  for (let c = total - 1; c >= 0; c--) (taken & (1 << c) ? used : free).push(c);
  return [...free, ...used].slice(0, n).sort((a, b) => a - b);
}

/**
 * This process's core plan, made once: `{ n, cores, mask, workers, claimed }`, or null when TW_NOPIN is set
 * or this is not Windows (no pinning there; the page's worker count still follows the grant).
 */
export function corePlan() {
  if (PLAN !== null) return PLAN.off ? null : PLAN;
  if (process.env.TW_NOPIN) { PLAN = { off: true }; return null; }
  const total = cpus().length;
  let n, claimed = null;
  if (process.env.TW_CORES) n = Math.max(1, Math.min(total, Math.floor(Number(process.env.TW_CORES))));
  else {
    const name = `check-${basename(process.argv[1] ?? 'node').replace(/\.mjs$/, '')}-${process.pid}`;
    const want = Number(process.env.TW_CORES_WANT ?? 3);
    const r = spawnSync('python', [MAPS + 'cores.py', 'claim', String(want), name, '--pid', String(process.pid), 'Playwright check'], { encoding: 'utf8' });
    n = Number(r.stdout?.trim().split('\n').at(-1));
    if (!(n >= 1)) n = 1;
    claimed = name;
  }
  const reg = readRegistry();
  let taken = 0;
  for (const [pid, m] of Object.entries(reg)) { if (alive(Number(pid))) taken |= m; else delete reg[pid]; }
  const cores = process.env.TW_CORE_MASK
    ? [...Array(total).keys()].filter((c) => Number(process.env.TW_CORE_MASK) & (1 << c))
    : pickCores(n, total, taken);
  const mask = cores.reduce((m, c) => m | (1 << c), 0);
  reg[process.pid] = mask;
  writeRegistry(reg);
  PLAN = { n: cores.length, cores, mask, workers: Math.max(1, cores.length - 1), claimed };
  process.on('exit', () => {
    const r = readRegistry(); delete r[process.pid]; writeRegistry(r);
    if (claimed) spawnSync('python', [MAPS + 'cores.py', 'release', claimed], { stdio: 'ignore' });
  });
  console.log(`      cores: ${cores.length} (${cores.join(',')}, mask 0x${mask.toString(16).toUpperCase()})${claimed ? ', from the ledger' : ''}, demandWorkers ${PLAN.workers}`);
  return PLAN;
}

// Sets the affinity of every chrome.exe that is the root process, a descendant of it, or has `profile`
// in its command line; with no MASK only reports. Prints "pid mask" per process.
const PS = `
$mask = $env:TW_PIN_MASK; $root = $env:TW_PIN_ROOT; $prof = $env:TW_PIN_PROFILE
$all = @(Get-CimInstance Win32_Process -Filter "Name='chrome.exe'")
$ids = New-Object 'System.Collections.Generic.HashSet[int]'
if ($root) { [void]$ids.Add([int]$root) }
if ($prof) { foreach ($p in $all) { if ($p.CommandLine -and $p.CommandLine.Contains($prof)) { [void]$ids.Add([int]$p.ProcessId) } } }
do { $n = $ids.Count; foreach ($p in $all) { if ($ids.Contains([int]$p.ParentProcessId)) { [void]$ids.Add([int]$p.ProcessId) } } } while ($ids.Count -ne $n)
foreach ($id in $ids) {
  try { $pr = Get-Process -Id $id -ErrorAction Stop; if ($mask) { $pr.ProcessorAffinity = [IntPtr][int64]$mask }; "$id " + [int64]$pr.ProcessorAffinity } catch {}
}`;

/** pin (or, with mask null, read) the Chrome started with `profile` (or process `rootPid`); { count, masks } */
export function chromeAffinity({ profile = '', rootPid = '', mask = null } = {}) {
  if (process.platform !== 'win32') return { count: 0, masks: [] };
  const r = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', PS], {
    encoding: 'utf8',
    env: { ...process.env, TW_PIN_MASK: mask ? String(mask) : '', TW_PIN_ROOT: String(rootPid || ''), TW_PIN_PROFILE: profile.replace(/\//g, '\\') },
  });
  const rows = (r.stdout ?? '').split('\n').map((l) => l.trim().split(' ')).filter((p) => p.length === 2);
  return { count: rows.length, masks: [...new Set(rows.map((p) => '0x' + Number(p[1]).toString(16).toUpperCase()))] };
}

/** the init script that adds ?demandWorkers=n to a page's address before the app reads it, where the check gave none */
export const workersInit = (n) => `(() => { try { if (location.protocol.startsWith('http') && !/[?&]demandWorkers=/.test(location.search)) { const u = new URL(location.href); u.searchParams.set('demandWorkers', ${n}); history.replaceState(history.state, '', u); } } catch {} })();`;
