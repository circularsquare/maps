@echo off
rem T-3443: the three linters, about 1 minute warm. Stops at the first one that fails.
rem   tsc with the unused and missing-return flags (app/tsconfig.json), clippy on sim/ with -D warnings
rem   (the allowed groups are in sim/Cargo.toml), ruff F and E9 on pipeline/ and tools/ (trainworld/ruff.toml,
rem   ruff 0.14.5 in the maps venv: venv\Scripts\python.exe -m pip install ruff==0.14.5).
rem Rust uses 3 cores (CARGO_BUILD_JOBS): claim them first with python cores.py claim 3 lint "lint.cmd".
setlocal
set TW=%~dp0..
set VENV=%TW%\..\venv\Scripts
echo == tsc
cd /d %TW%\app || exit /b 1
call npx tsc --noEmit || exit /b 1
echo == clippy
cd /d %TW%\sim || exit /b 1
set CARGO_BUILD_JOBS=3
cargo clippy --all-targets -- -D warnings || exit /b 1
cargo clippy --all-targets --features demand -- -D warnings || exit /b 1
echo == ruff
cd /d %TW% || exit /b 1
%VENV%\ruff.exe check pipeline tools || exit /b 1
echo lint ok
