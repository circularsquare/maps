@echo off
set PY=C:\Users\anita\projects\maps\venv\Scripts\python.exe
set OUT=C:\Users\anita\projects\maps\artifacts\trainworld\t1140
cd /d C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8a3f6fa5-349e-4cb4-90bd-f8d565510cd9\scratchpad\t1140
if not exist %OUT% mkdir %OUT%
%PY% draw1140.py us us_runs.pkl rbp %OUT%\us-west.png "US West: zones that change (T-1140 rule, full replan)" --isos USA --regions "Washington,Oregon,California,Nevada,Idaho,Montana,Wyoming,Utah,Colorado,Arizona,New Mexico"
%PY% draw1140.py us us_runs.pkl rbp %OUT%\us-midwest.png "US Midwest: zones that change (T-1140 rule, full replan)" --isos USA --regions "North Dakota,South Dakota,Nebraska,Kansas,Minnesota,Iowa,Missouri,Wisconsin,Illinois,Michigan,Indiana,Ohio"
%PY% draw1140.py us us_runs.pkl rbp %OUT%\us-south.png "US South: zones that change (T-1140 rule, full replan)" --isos USA --regions "Texas,Oklahoma,Arkansas,Louisiana,Mississippi,Alabama,Tennessee,Kentucky,Georgia,Florida,South Carolina,North Carolina,Virginia,West Virginia"
%PY% draw1140.py us us_runs.pkl rbp %OUT%\us-northeast.png "US Northeast: zones that change (T-1140 rule, full replan)" --isos USA --regions "Pennsylvania,New York,New Jersey,Delaware,Maryland,Connecticut,Rhode Island,Massachusetts,Vermont,New Hampshire,Maine"
%PY% draw1140.py us us_runs.pkl rbp %OUT%\us-colorado-washington.png "Colorado and Washington: Pueblo, Colorado Springs, Bellingham, Yakima" --isos USA --regions "Colorado,Washington"
