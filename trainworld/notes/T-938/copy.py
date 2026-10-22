"""T-938: a private copy of the sim crate with the ceiling test added as everyday::t938_ceiling."""
import pathlib
import shutil
import sys

sim = pathlib.Path(sys.argv[1]) / "sim"
note = pathlib.Path(sys.argv[2]).resolve()
dest = pathlib.Path(sys.argv[3]) / "sim"
if (dest / "src").exists():
    shutil.rmtree(dest / "src")
dest.mkdir(parents=True, exist_ok=True)
shutil.copytree(sim / "src", dest / "src")
for f in ["Cargo.toml", "Cargo.lock", "build.rs"]:
    if (sim / f).exists():
        shutil.copy2(sim / f, dest / f)
ev = dest / "src" / "demand" / "everyday.rs"
with ev.open("a", encoding="utf-8") as f:
    f.write(f'\n#[cfg(test)]\n#[path = "{note.as_posix()}"]\npub(crate) mod t938_ceiling;\n')
print(f"copied to {dest}")
