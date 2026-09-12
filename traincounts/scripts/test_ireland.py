import contextlib
import csv
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from build_ireland import build, project_stations


class IrelandTests(unittest.TestCase):
    def test_express_local_exceptions_platforms_direction_and_identity(self):
        """Two boarding patterns share A-B-C; duplicate service records don't count twice."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tables={
                'feed_info.txt':[{'feed_start_date':'20260901','feed_end_date':'20261001','feed_version':'fixture'}],
                'agency.txt':[{'agency_id':'IR','agency_name':'Rail','agency_timezone':'Europe/London'}],
                'routes.txt':[{'route_id':'main','route_short_name':'rail','route_long_name':'Main','route_type':'2','agency_id':'IR'}],
                'stops.txt':[
                    {'stop_id':'A','parent_station':'','stop_lon':'-7','stop_lat':'53','stop_name':'A'},
                    {'stop_id':'AP','parent_station':'A','stop_lon':'-7','stop_lat':'53','stop_name':'A platform'},
                    {'stop_id':'B','parent_station':'','stop_lon':'-6.99','stop_lat':'53','stop_name':'B'},
                    {'stop_id':'C','parent_station':'','stop_lon':'-6.98','stop_lat':'53','stop_name':'C'},
                    {'stop_id':'NEAR','parent_station':'','stop_lon':'-6.99','stop_lat':'53.002','stop_name':'Separate railway'},
                ],
                'calendar.txt':[dict(service_id='daily',start_date='20260901',end_date='20261001',**{x:'1' for x in ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']})],
                'calendar_dates.txt':[{'service_id':'extra','date':'20260914','exception_type':'1'},{'service_id':'daily','date':'20260915','exception_type':'2'},{'service_id':'extra','date':'20260915','exception_type':'1'}],
                'trips.txt':[], 'shapes.txt':[], 'stop_times.txt':[]}
            for sid,points in [('east',[-7,-6.99,-6.98]),('west',[-6.98,-6.99,-7])]:
                for index,longitude in enumerate(points):
                    tables['shapes.txt'].append(dict(shape_id=sid,shape_pt_sequence=index,shape_pt_lon=longitude,shape_pt_lat=53))
            for tid,service,code,shape,stations in [('local','daily','L','east',['AP','B','C']),('duplicate','daily','L','east',['A','B','C']),('express','extra','E','west',['C','A'])]:
                tables['trips.txt'].append(dict(trip_id=tid,service_id=service,trip_short_name=code,shape_id=shape,route_id='main'))
                for i,s in enumerate(stations):
                    tables['stop_times.txt'].append(dict(trip_id=tid,stop_id=s,stop_sequence=i,arrival_time=f'25:0{i}:00',departure_time=f'25:0{i}:00',pickup_type='0',drop_off_type='0'))
            feed=root/'fixture.zip'
            with zipfile.ZipFile(feed,'w') as z:
                for name,records in tables.items():
                    text=io.StringIO()
                    writer=csv.DictWriter(text,fieldnames=list(records[0]))
                    writer.writeheader();writer.writerows(records)
                    z.writestr(name,text.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                build(feed,dt.date(2026,9,14),2,root/'out')
            features=json.loads((root/'out/segments.geojson').read_text())['features']
            self.assertEqual({(f['properties']['from'],f['properties']['to']) for f in features},{('A','B'),('B','C')})
            for f in features:
                self.assertEqual(f['properties']['daily'],[2,1])
                self.assertEqual(f['properties']['forward'],[1,0])
                self.assertEqual(f['properties']['reverse'],[1,1])
            self.assertEqual(json.loads((root/'out/metadata.json').read_text())['daily_trains'],[2,1])

    def test_return_visit_is_retained(self):
        points=[[-7,53],[-6.99,53],[-6.98,53],[-6.99,53],[-7,53]]
        coords={'A':points[0],'B':points[1],'C':points[2]}
        self.assertEqual([s for _,s,_ in project_stations(points,coords)],['A','B','C','B','A'])


if __name__=='__main__':
    unittest.main()
