@echo off
rem Rebuild 8800's snapshot at the master commit given as %1 (an explicit hash, never HEAD).
rem Builds into dist-new beside the served dist and swaps at the end (T-2483).
rem Run: cmd /c "trainworld\checks\snapbuild.cmd <hash> > <log> 2>&1"   (about 4 minutes)
set MAPS=%~dp0..\..
git -C %MAPS%\.tw-snap checkout --detach %1 || exit /b 1
set CARGO_TARGET_DIR=%MAPS%\.tw-snap-target
set CARGO_BUILD_JOBS=2
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
cd /d %MAPS%\.tw-snap\trainworld\app
if exist dist-new rmdir /s /q dist-new
if exist dist-old rmdir /s /q dist-old
call npm run build -- --outDir dist-new || exit /b 1
if not exist dist-new\index.html exit /b 1
if exist dist move dist dist-old || exit /b 1
move dist-new dist || (move dist-old dist & exit /b 1)
if exist dist-old rmdir /s /q dist-old
rem localhost:8800/twwiki/ (T-2854); a failure here leaves the old pages up and the game swapped
python "%MAPS%\trainworld\tools\wiki_build.py" --html
echo swapped
