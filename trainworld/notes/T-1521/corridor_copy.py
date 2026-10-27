"""T-1521 round 3: a copy of the tree's sim crate with T-712's corridor test brought up to date
(`corridor_test.rs`) and one switch, the plane's fare from the environment: `T1521_AIR` "base,per_km"
(default the built $100 + 5c) times `T1521_AF`, a JSON {country: factor} for the departing country
(default 1). The tree's own code is untouched.

    python trainworld/notes/T-1521/corridor_copy.py <dest folder>
"""
import pathlib, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE.parents[1] / "sim"
dest = pathlib.Path(sys.argv[1]) / "sim"
if (dest / "src").exists():
    shutil.rmtree(dest / "src")
dest.mkdir(parents=True, exist_ok=True)
shutil.copytree(SIM / "src", dest / "src")
for f in ["Cargo.toml", "Cargo.lock"]:
    if (SIM / f).exists():
        shutil.copy2(SIM / f, dest / f)
for extra in ["build.rs"]:
    if (SIM / extra).exists():
        shutil.copy2(SIM / extra, dest / extra)
ic = dest / "src" / "demand" / "intercity.rs"
s = ic.read_text(encoding="utf-8")
old = "    Cost { min: min as f32, usd: (AIR_FARE_USD + AIR_USD_PER_KM * km) as f32 }"
assert s.count(old) == 1
s = s.replace(old, "    let (b, k) = t1521_air();\n    let _ = (AIR_FARE_USD, AIR_USD_PER_KM);\n    Cost { min: min as f32, usd: (b + k * km) as f32 }")
old = "    (r.rt, r.rk) = (d.rt, d.rk);\n"
assert s.count(old) == 1
s = s.replace(old, old + "    r.fl.usd *= t1521_af(&countries[p.country as usize]);\n")
s += '''
fn t1521_air() -> (f64, f64) {
    static V: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("T1521_AIR").unwrap_or("100,0.05".into());
        let (b, k) = v.split_once(',').unwrap();
        (b.parse().unwrap(), k.parse().unwrap())
    })
}
fn t1521_af(country: &str) -> f32 {
    static V: std::sync::OnceLock<HashMap<String, f32>> = std::sync::OnceLock::new();
    let m = V.get_or_init(|| std::env::var("T1521_AF").ok().map(|v| serde_json::from_str(&v).unwrap()).unwrap_or_default());
    *m.get(country).unwrap_or(&1.)
}
'''
s += f'\n#[cfg(test)]\n#[path = "{(HERE / "corridor_test.rs").as_posix()}"]\nmod t1521_corridors;\n'
ic.write_text(s, encoding="utf-8")
print("copied to", dest)
