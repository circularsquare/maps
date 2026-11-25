//! T-4199: a hash of the crate's sources and build settings, `TW_SIM_SRC_HASH`, which the market's
//! corridor file is keyed by (src/demand/intercity/corridor_file.rs): a file made by another build is
//! never read. Carriage returns are left out, so a checkout's line endings do not change it.
use std::fs;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).expect("src") {
        let p = e.expect("src entry").path();
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let mut files = vec![];
    walk(&root.join("src"), &mut files);
    files.sort();
    files.extend(["build.rs", "Cargo.toml", "Cargo.lock", ".cargo/config.toml"].map(|f| root.join(f)));
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    for f in &files {
        let rel = f.strip_prefix(&root).unwrap_or(f).to_string_lossy().replace('\\', "/");
        rel.bytes().chain([0]).for_each(&mut eat);
        fs::read(f).unwrap_or_default().into_iter().filter(|&b| b != b'\r').chain([0]).for_each(&mut eat);
    }
    println!("cargo:rustc-env=TW_SIM_SRC_HASH={h:016x}");
    for f in ["src", "build.rs", "Cargo.toml", "Cargo.lock", ".cargo/config.toml"] {
        println!("cargo:rerun-if-changed={f}");
    }
}
