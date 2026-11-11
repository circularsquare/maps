@echo off
rem Rebuild 8800's snapshot at the master commit given as %1 (an explicit hash, never HEAD).
rem Builds into dist-new beside the served dist and swaps at the end (T-2483).
rem Run: cmd /c "trainworld\checks\snapbuild.cmd <hash> > <log> 2>&1"   (about 2.5 minutes when sim/ changed)
set MAPS=%~dp0..\..
git -C %MAPS%\.tw-snap checkout --detach %1 || exit /b 1
set CARGO_BUILD_JOBS=2
rem packs in dist as hard links to data/packs (or to the freeze's, snapbuild-frozen.cmd), not 6 GB of copies
set TW_LINK_PACKS=1
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
cd /d %MAPS%\.tw-snap\trainworld\app
if exist dist-new rmdir /s /q dist-new
if exist dist-old rmdir /s /q dist-old
rem the last build's detached delete (see below) is normally done; if it still runs this finds the rest and waits
for /d %%d in (dist-trash-*) do rmdir /s /q "%%d"
rem the two WASM modules at the same time, each with its own Cargo target dir; then what `npm run build` does after wasm
python "%MAPS%\trainworld\tools\snap_wasm.py" "%MAPS%\.tw-snap-target" "%MAPS%\.tw-snap-target-demand" || exit /b 1
call npx tsc --noEmit || exit /b 1
call npx vite build --outDir dist-new || exit /b 1
if not exist dist-new\index.html exit /b 1
rem T-2962: the served build's files stay beside the new ones, so a tab open across this swap can still load its lazy files
if exist dist python "%MAPS%\trainworld\tools\keep_assets.py" carry dist dist-new
if exist dist move dist dist-old || exit /b 1
move dist-new dist || (move dist-old dist & exit /b 1)
rem the old build (7 GB) is renamed out of the way and deleted in a separate window, so the next build cannot trip on it
set TRASH=dist-trash-%RANDOM%
if exist dist-old move dist-old %TRASH% >nul && start "" /min cmd /c "rmdir /s /q %TRASH% >nul 2>&1"
rem localhost:8800/twwiki/ (T-2854); a failure here leaves the old pages up and the game swapped
python "%MAPS%\trainworld\tools\wiki_build.py" --html
echo swapped
