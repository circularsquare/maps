// Agent worktrees (T-792): a git worktree on its own branch for a code agent whose edits would collide
// with others' in shared files. Run from anywhere; trees live in maps/.tw-trees/<name>, branch tw/<name>.
//   node trainworld/checks/agent-tree.mjs new <name> [commit]   from master's tip unless a commit is given
//   node trainworld/checks/agent-tree.mjs list
//   node trainworld/checks/agent-tree.mjs land <name> "<one sentence>"   orchestrator: one commit on master
//   node trainworld/checks/agent-tree.mjs remove <name> [--force]
// `land` takes everything changed under trainworld/ in the tree (committed on the branch or not), rebases it
// onto master, squashes it into one commit and fast-forwards master to it. Master's own uncommitted edits
// stay; git stops the fast-forward if one of them is in a file the commit touches.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const git = (cwd, ...args) => execFileSync('git', ['-C', cwd, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
// The main tree, also when this copy of the script sits in an agent's tree.
const here = path.resolve(fileURLToPath(new URL('../..', import.meta.url)));
const maps = path.dirname(path.resolve(here, git(here, 'rev-parse', '--git-common-dir')));
const trees = path.join(maps, '.tw-trees');
// Untracked in git, so a new tree gets them as junctions to the main tree's.
const LINKS = ['trainworld/data', 'trainworld/app/node_modules', 'artifacts/trainworld', 'venv'];

const gitLoud = (cwd, ...args) => execFileSync('git', ['-C', cwd, ...args], { stdio: 'inherit' });
const fail = (msg) => { console.error(msg); process.exit(1); };
const isLink = (p) => { try { return fs.lstatSync(p).isSymbolicLink(); } catch { return false; } };

function treeOf(name) {
  if (!/^[a-z0-9][a-z0-9-]{0,30}$/.test(name ?? '')) fail(`a name is lowercase letters, digits and dashes: "${name ?? ''}"`);
  return { dir: path.join(trees, name), branch: `tw/${name}` };
}

function excludeTrees() {
  const file = path.join(maps, '.git', 'info', 'exclude');
  const text = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '';
  if (!/^\.tw-trees\/$/m.test(text)) fs.appendFileSync(file, (text.endsWith('\n') || !text ? '' : '\n') + '.tw-trees/\n');
}

function unlinkJunctions(dir) {
  for (const rel of LINKS) {
    const p = path.join(dir, rel);
    if (isLink(p)) fs.unlinkSync(p);
  }
}

// Changed paths in the tree against master, split into trainworld/ and the rest.
function changes(dir) {
  const base = git(dir, 'merge-base', 'master', 'HEAD');
  const files = new Set([
    ...git(dir, 'diff', '--name-only', base).split('\n'),
    ...git(dir, 'ls-files', '--others', '--exclude-standard').split('\n'),
  ].filter(Boolean));
  const tw = [...files].filter((f) => f.startsWith('trainworld/'));
  return { tw, other: [...files].filter((f) => !f.startsWith('trainworld/')) };
}

const [cmd, name, ...rest] = process.argv.slice(2);

if (cmd === 'new') {
  const { dir, branch } = treeOf(name);
  if (fs.existsSync(dir)) fail(`${dir} exists already`);
  const from = rest[0] ?? 'master';
  excludeTrees();
  fs.mkdirSync(trees, { recursive: true });
  gitLoud(maps, 'worktree', 'add', '-q', '-b', branch, dir, from);
  for (const rel of LINKS) {
    const target = path.join(maps, rel);
    if (!fs.existsSync(target)) continue;
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
    fs.symlinkSync(target, path.join(dir, rel), 'junction');
  }
  // Generated WASM bindings, so the app runs before the first `npm run wasm`; a copy, since they differ by branch.
  const wasm = path.join(maps, 'trainworld', 'app', 'src', 'wasm');
  if (fs.existsSync(wasm)) fs.cpSync(wasm, path.join(dir, 'trainworld', 'app', 'src', 'wasm'), { recursive: true });
  console.log(`\n${dir}\nbranch ${branch} from ${git(dir, 'rev-parse', '--short', 'HEAD')}`);
  console.log(`Rust builds into the tree's own trainworld/sim/target (first wasm build a few minutes).`);
} else if (cmd === 'list') {
  if (!fs.existsSync(trees)) { console.log('no trees'); process.exit(0); }
  for (const n of fs.readdirSync(trees)) {
    const dir = path.join(trees, n);
    if (!fs.existsSync(path.join(dir, '.git'))) continue;
    const { tw, other } = changes(dir);
    const ahead = git(dir, 'rev-list', '--count', 'master..HEAD');
    const behind = git(dir, 'rev-list', '--count', 'HEAD..master');
    console.log(`${n.padEnd(20)} ${git(dir, 'rev-parse', '--short', 'HEAD')}  ${ahead} ahead, ${behind} behind master, ${tw.length} trainworld files changed${other.length ? `, ${other.length} outside` : ''}`);
  }
} else if (cmd === 'land') {
  const { dir, branch } = treeOf(name);
  const message = rest.join(' ').trim();
  if (!message) fail('land needs a commit message');
  if (!fs.existsSync(dir)) fail(`no tree ${dir}`);
  const { tw, other } = changes(dir);
  if (other.length) fail(`changes outside trainworld/, land them by hand:\n  ${other.join('\n  ')}`);
  if (!tw.length) fail('nothing to land');
  // The links are untracked junctions; `add trainworld` must not see them as content.
  gitLoud(dir, 'add', '--all', '--', 'trainworld');
  if (git(dir, 'diff', '--cached', '--name-only')) gitLoud(dir, 'commit', '-q', '-m', 'agent-tree: work in progress');
  try { gitLoud(dir, 'rebase', '-q', 'master'); } catch {
    fail(`the rebase onto master stopped on a conflict in ${dir}: resolve there and \`git rebase --continue\`, or \`git rebase --abort\`; then land again`);
  }
  gitLoud(dir, 'reset', '-q', '--soft', 'master');
  gitLoud(dir, 'commit', '-q', '-m', message);
  try { gitLoud(maps, 'merge', '-q', '--ff-only', branch); } catch {
    fail(`master did not fast-forward to ${branch} (an uncommitted edit in master's tree touches the same file, or master moved): the commit waits on the branch`);
  }
  console.log(`master is now ${git(maps, 'log', '-1', '--format=%h %s')}`);
} else if (cmd === 'remove') {
  const { dir, branch } = treeOf(name);
  const force = rest.includes('--force');
  if (fs.existsSync(dir)) {
    const { tw, other } = changes(dir);
    if (!force && (tw.length || other.length)) fail(`${dir} has ${tw.length + other.length} changed files not on master; land them or pass --force`);
    unlinkJunctions(dir);
    gitLoud(maps, 'worktree', 'remove', '--force', dir);
  }
  try { git(maps, 'branch', '-D', branch); } catch { /* already gone */ }
  console.log(`removed ${name}`);
} else {
  console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(0, 9).join('\n'));
  process.exit(cmd ? 2 : 0);
}
