"""sim.py's zone_groups alone, for retrofit.py."""
import os
from shapely import union_all, STRtree
import world_plan as wp

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim.py'), encoding='utf-8').read()
_body = _src[_src.index('def zone_groups'):_src.index('t0 = time.time()')]
exec(_body)
