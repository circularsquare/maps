@echo off
rem 8800 rebuild while a country run owns data/packs: packs from the freeze named in %2, drives from the main tree.
rem Run: cmd /c "trainworld\checks\snapbuild-frozen.cmd <hash> <freeze> > <log> 2>&1"
set MAPS=%~dp0..\..
set TW_PACKS=%MAPS%\trainworld\data\frozen\%2\packs
set TW_MARKET_DRIVES=%MAPS%\trainworld\data\packs\market-drives.bin
call %~dp0snapbuild.cmd %1
