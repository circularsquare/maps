@echo off
rem T-3688 second restart from wave 1 (maps-5a, 11-22): the first wave 1 parked haj on Geofabrik's India move (404); the 20 Nov eastern-zone extract was put in data\raw\osm by hand.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3688\kit\redraws.ps1" -Want 6 -From wave1 %*
