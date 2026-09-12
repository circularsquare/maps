"""Calendar and corridor regression checks; run python -m unittest discover -s scripts."""
import contextlib
import csv
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from build_finland import build, locate, service_days


class CountsTest(unittest.TestCase):
    def test_calendar_exceptions_and_empty_days(self):
        dates = [dt.date(2026,9,14)+dt.timedelta(days=i) for i in range(7)]
        calendar = [{'service_id':'weekday','start_date':'20260914','end_date':'20260920',**dict.fromkeys(['monday','tuesday','wednesday','thursday','friday'],'1'),**dict.fromkeys(['saturday','sunday'],'0')}]
        exceptions = [{'service_id':'weekday','date':'20260914','exception_type':'2'},{'service_id':'weekday','date':'20260919','exception_type':'1'},{'service_id':'special','date':'20260920','exception_type':'1'}]
        result = service_days(calendar,exceptions,dates)
        self.assertEqual(result['weekday'],{1,2,3,4,5})
        self.assertEqual(result['special'],{6})
        self.assertEqual(result['missing'],set())

    def test_loop_shape_uses_destination_at_end(self):
        points = [[24,60],[24.1,60],[24.1,60.1],[24,60]]
        self.assertEqual(locate(points,[[24,60],[24.1,60.1],[24,60]]),[0,2,3])

    def test_parent_platforms_passbys_and_bidirectional_counts(self):
        # Two trains use different platforms, sharing the same pass-through point.
        # One runs every day, the other weekdays except Mon plus Sat.
        files = {
            'feed_info.txt':[['feed_start_date','feed_end_date','feed_version'],['20260914','20260920','test']],
            'agency.txt':[['agency_id','agency_name'],['10','VR']],
            'routes.txt':[['route_id','agency_id','route_type','route_short_name'],['r','10','102','IC']],
            'trips.txt':[['trip_id','service_id','route_id','shape_id'],['1_x','weekday','r','out'],['2_x','daily','r','back']],
            'stops.txt':[['stop_id','stop_name','parent_station','stop_lon','stop_lat'],['A','Alpha','','24','60'],['A_1','Alpha 1','A','24','60'],['A_2','Alpha 2','A','24','60'],['X','Pass point','','24.1','60'],['B','Beta','','24.2','60']],
            'stop_times.txt':[['trip_id','stop_id','stop_sequence','pickup_type','drop_off_type'],['1_x','A_1','0','0','1'],['1_x','X','1','1','1'],['1_x','B','2','1','0'],['2_x','B','0','0','1'],['2_x','X','1','1','1'],['2_x','A_2','2','1','0']],
            'shapes.txt':[['shape_id','shape_pt_sequence','shape_pt_lon','shape_pt_lat'],['out','0','24','60'],['out','1','24.1','60'],['out','2','24.2','60'],['back','0','24.2','60'],['back','1','24.1','60'],['back','2','24','60']],
            'calendar.txt':[['service_id','start_date','end_date','monday','tuesday','wednesday','thursday','friday','saturday','sunday'],['weekday','20260914','20260920','1','1','1','1','1','0','0'],['daily','20260914','20260920','1','1','1','1','1','1','1']],
            'calendar_dates.txt':[['service_id','date','exception_type'],['weekday','20260914','2'],['weekday','20260919','1']]
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            feed = root/'fixture.zip'
            with zipfile.ZipFile(feed,'w') as z:
                for name, values in files.items():
                    stream=io.StringIO();csv.writer(stream).writerows(values);z.writestr(name,stream.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                build(feed,dt.date(2026,9,14),7,root/'out')
            metadata=json.loads((root/'out/metadata.json').read_text())
            features=json.loads((root/'out/segments.geojson').read_text())['features']
            self.assertEqual(metadata['daily_trains'],[1,2,2,2,2,2,1])
            self.assertEqual(metadata['stations'],2) # X is not a boarding station
            self.assertEqual(len(features),2) # no separate edges for A's platforms
            for f in features:
                p=f['properties']
                self.assertEqual(p['daily'],[1,2,2,2,2,2,1])
                self.assertEqual(p['average'],round(12/7,3))
                self.assertEqual([a+b for a,b in zip(p['forward'],p['reverse'])],p['daily'])


if __name__ == '__main__':
    unittest.main()
