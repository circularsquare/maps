"""T-1546: a copy of master's committed sim crate (git archive of HEAD, so other agents' edits in the
main tree stay out) with the dump test `t1546_test.rs` added. No other change to the code.

    python trainworld/notes/T-1546/t1546_copy.py <dest folder>
"""
import io, pathlib, shutil, subprocess, sys, tarfile

HERE = pathlib.Path(__file__).resolve().parent
MAPS = HERE.parents[2]
dest = pathlib.Path(sys.argv[1])
tar = subprocess.run(["git", "-C", str(MAPS), "archive", "--format=tar", "HEAD", "trainworld/sim/src", "trainworld/sim/Cargo.toml", "trainworld/sim/Cargo.lock"],
                     check=True, capture_output=True).stdout
if (dest / "sim" / "src").exists():
    shutil.rmtree(dest / "sim" / "src")
with tarfile.open(fileobj=io.BytesIO(tar)) as t:
    for m in t.getmembers():
        if not m.isfile():
            continue
        rel = pathlib.Path(m.name).relative_to("trainworld")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(t.extractfile(m).read())
build = MAPS / "trainworld" / "sim" / "build.rs"
if build.exists():
    shutil.copy2(build, dest / "sim" / "build.rs")
ic = dest / "sim" / "src" / "demand" / "intercity.rs"
s = ic.read_text(encoding="utf-8")
s += f'\n#[cfg(test)]\n#[path = "{(HERE / "t1546_test.rs").as_posix()}"]\nmod t1546_test;\n'
ic.write_text(s, encoding="utf-8")
head = subprocess.run(["git", "-C", str(MAPS), "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
print("copied", head, "to", dest)
