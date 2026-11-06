@echo off
rem T-2642 scratch: run.cmd <log>, with TW_T780_NET, TW_MODES, TW_REPS, TW_CROWD, TW_Q set by the caller
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
set CARGO_BUILD_JOBS=1
set CARGO_TARGET_DIR=C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21\scratchpad\searchcost\target
cd /d C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21\scratchpad\searchcost\sim
cargo test --release --no-default-features --features demand --lib %TW_TEST% -- --ignored --nocapture --test-threads=1 > %1 2>&1
echo EXIT %ERRORLEVEL% >> %1
rem (start runs a .cmd under cmd /K, which stays open: exit)
exit
