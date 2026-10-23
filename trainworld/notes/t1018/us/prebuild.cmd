@echo off
rem Build the gate's rates test in this folder's sim copy once, so the runs' build_everyday finds it fresh
set S=%~dp0simroot\sim
set CARGO_BUILD_JOBS=4
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
cargo test --release --features demand --lib --no-run --manifest-path %S%\Cargo.toml > C:\Users\anita\projects\maps\artifacts\trainworld\us-prebuild.log 2>&1
