"""T-1003: a private copy of the sim crate with the walk harness added as everyday::t1003."""
import pathlib
import shutil

sim = pathlib.Path("C:/Users/anita/projects/maps/trainworld/sim")
here = pathlib.Path(__file__).resolve().parent
note = here / "t1003.rs"
dest = here / "copy" / "sim"
if (dest / "src").exists():
    shutil.rmtree(dest / "src")
dest.mkdir(parents=True, exist_ok=True)
shutil.copytree(sim / "src", dest / "src")
for f in ["Cargo.toml", "Cargo.lock", "build.rs"]:
    if (sim / f).exists():
        shutil.copy2(sim / f, dest / f)
ev = dest / "src" / "demand" / "everyday.rs"
with ev.open("a", encoding="utf-8") as f:
    f.write(f'\n#[cfg(test)]\n#[path = "{note.as_posix()}"]\npub(crate) mod t1003;\n')
print(f"copied to {dest}")
