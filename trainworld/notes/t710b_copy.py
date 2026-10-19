"""T-710 stage 2: a copy of the sim crate as it now is (the relative rule and crowding from 70% in
the code, today's model by `rel_gamma=0,crowd_start=0`), with the trip test and calib's readouts
of how far each crowding round moves the loads and of the busiest segments.

    python t710b_copy.py <trainworld folder> <test file> <dest folder>
"""
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from t710_copy import patch  # noqa: E402


def main():
    sim = pathlib.Path(sys.argv[1]) / "sim"
    test = pathlib.Path(sys.argv[2]).resolve()
    dest = pathlib.Path(sys.argv[3]) / "sim"
    if (dest / "src").exists():
        shutil.rmtree(dest / "src")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sim / "src", dest / "src")
    for f in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy2(sim / f, dest / f)
    e = dest / "src" / "demand" / "everyday.rs"
    with e.open("a", encoding="utf-8") as f:
        f.write(f'\n#[cfg(test)]\n#[path = "{test.as_posix()}"]\npub(crate) mod t710_trips;\n')
    c = dest / "src" / "bin" / "calib.rs"
    patch(c, "        for _ in 0..rounds {\n",
          "        let mut prev_avg: Vec<f32> = vec![];\n        let mut prev_raw: Vec<f32> = vec![];\n        for rr in 0..=rounds {\n")
    patch(c, "            assert!(api.crowd_with_external(q, &total));\n",
          "            let segt = total[..nr].to_vec();\n"
          "            let base: f64 = segt.iter().map(|&x| x as f64).sum::<f64>().max(1e-9);\n"
          "            if !prev_avg.is_empty() {\n"
          "                let moved: f64 = segt.iter().zip(&prev_avg).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / base;\n"
          "                let raw: Vec<f32> = segt.iter().zip(&prev_avg).map(|(a, b)| b + (a - b) * (rr as f32 + 1.0)).collect();\n"
          "                let flip = if prev_raw.is_empty() { -1.0 } else { raw.iter().zip(&prev_raw).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / base };\n"
          "                eprintln!(\"T710 q {q} before round {rr}: avg moved {:.4}, raw moved {:.4}\", moved, flip);\n"
          "                prev_raw = raw;\n"
          "            } else {\n"
          "                prev_raw = segt.clone();\n"
          "            }\n"
          "            prev_avg = segt;\n"
          "            if rr == rounds { break; }\n"
          "            assert!(api.crowd_with_external(q, &total));\n")
    patch(c, "        let s = api.summary(q);\n",
          "        let s = api.summary(q);\n"
          "        {\n"
          "            let mut segq = api.seg(q);\n"
          "            if let Some(e) = &ev { for (a, b) in segq.iter_mut().zip(e.seg(q)) { *a += b; } }\n"
          "            let mut ix: Vec<usize> = (0..segq.len()).collect();\n"
          "            ix.sort_by(|&a, &b| segq[b].partial_cmp(&segq[a]).unwrap());\n"
          "            let top: Vec<String> = ix.iter().take(12).map(|&r| format!(\"{r}:{:.0}\", segq[r])).collect();\n"
          "            eprintln!(\"T710 q {q} round {} max_crush {:.3} over_seats {} busiest {}\", s[0], s[7], s[6], top.join(\" \"));\n"
          "        }\n")
    print(f"copied to {dest}")


if __name__ == "__main__":
    main()
