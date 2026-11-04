@echo off
set PY=C:\Users\anita\projects\maps\venv\Scripts\python.exe
set OUT=C:\Users\anita\projects\maps\artifacts\trainworld\t1140
cd /d C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8a3f6fa5-349e-4cb4-90bd-f8d565510cd9\scratchpad\t1140
%PY% draw1140.py europe europe_repair_rb.pkl - %OUT%\italy.png "Italy: the rule with the 0.5M floor" --isos ITA
%PY% draw1140.py europe europe_repair_rb.pkl - %OUT%\france.png "France: the rule with the 0.5M floor" --isos FRA
%PY% draw1140.py europe europe_repair_rb.pkl - %OUT%\greece-portugal-spain.png "Greece, Portugal, Spain: the rule with the 0.5M floor" --isos GRC,PRT,ESP
%PY% draw1140.py europe europe_repair_rb.pkl - %OUT%\germany-benelux.png "Germany, the Netherlands, Belgium" --isos DEU,NLD,BEL
%PY% draw1140.py europe europe_repair_rb.pkl - %OUT%\britain-ireland.png "Britain and Ireland" --isos GBR,IRL
%PY% draw1140.py europe europe_repair_rb.pkl - %OUT%\central-europe.png "Poland, Czechia, Austria, Switzerland, Hungary" --isos POL,CZE,AUT,CHE,HUN
%PY% draw1140.py europe europe_repair_rb.pkl - %OUT%\southeast-north.png "Romania, Bulgaria, Serbia, Albania, the Nordics" --isos ROU,BGR,SRB,ALB,DNK,SWE,NOR,FIN
%PY% draw1140.py nuts nuts_repair_rb.pkl - %OUT%\balkans-baltics.png "Slovenia, Croatia, North Macedonia, Kosovo, the Baltics, Malta" --isos SVN,HRV,MKD,XKX,LVA,LTU,MLT
%PY% draw1140.py europe europe_repair_rb_a10000.pkl - %OUT%\italy-with-area-term.png "Italy with T-903's area term (people/0.5M + km2/10,000)" --isos ITA
%PY% draw1140.py twn twn_repair_rb.pkl - %OUT%\taiwan.png "Taiwan (Zhubei, T-1002)" --isos TWN
%PY% draw1140.py chn chn_repair_rb.pkl - %OUT%\china.png "China: the rule on built zones (gravity, 1M floor)" --isos CHN
