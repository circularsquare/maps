"""T-1541: a copy of tree t1521's sim crate (read-only source) with the access diagnosis test
(`t1541_test.rs`) and two switches, both in the copy only:
  T1521_AIR    "base,per_km": the plane's fare (default the built $100 + 5c)
  T1541_AIRACC "mult,add": the drive to the two airports counted as mult x the plain minutes + add
               (default 1,0: as built)

    python trainworld/notes/T-1541/t1541_copy.py <t1521 trainworld folder> <dest folder>
"""
import pathlib, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
SIM = pathlib.Path(sys.argv[1]) / "sim"
dest = pathlib.Path(sys.argv[2]) / "sim"
if (dest / "src").exists():
    shutil.rmtree(dest / "src")
dest.mkdir(parents=True, exist_ok=True)
shutil.copytree(SIM / "src", dest / "src")
for f in ["Cargo.toml", "Cargo.lock", "build.rs"]:
    if (SIM / f).exists():
        shutil.copy2(SIM / f, dest / f)
ic = dest / "src" / "demand" / "intercity.rs"
s = ic.read_text(encoding="utf-8")
old = "    Cost { min: min as f32, usd: (AIR_FARE_USD + AIR_USD_PER_KM * km) as f32 }"
assert s.count(old) == 1
s = s.replace(old, "    let (b, k) = t1521_air();\n    let _ = (AIR_FARE_USD, AIR_USD_PER_KM);\n    Cost { min: min as f32, usd: (b + k * km) as f32 }")
old = "                let c = air_cost(km, x + y, direct);"
assert s.count(old) == 1
s = s.replace(old, "                let c = air_cost(km, t1541_acc(x + y), direct);")
s += '''
fn t1521_air() -> (f64, f64) {
    static V: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("T1521_AIR").unwrap_or("100,0.05".into());
        let (b, k) = v.split_once(',').unwrap();
        (b.parse().unwrap(), k.parse().unwrap())
    })
}
fn t1541_acc(m: f64) -> f64 {
    static V: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    let (f, a) = *V.get_or_init(|| {
        let v = std::env::var("T1541_AIRACC").unwrap_or("1,0".into());
        let (b, k) = v.split_once(',').unwrap();
        (b.parse().unwrap(), k.parse().unwrap())
    });
    f * m + a
}
'''
s += f'\n#[cfg(test)]\n#[path = "{(HERE / "t1541_test.rs").as_posix()}"]\nmod t1541_test;\n'
ic.write_text(s, encoding="utf-8")
print("copied to", dest)
