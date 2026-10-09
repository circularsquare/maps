//! The network part of a save (SPEC 11, notes/T-008.md 4.6): inputs only, ids renumbered densely,
//! varints, positions as millimetre deltas. Compression (deflate, `CompressionStream` in the
//! browser) is the caller's. Derived geometry is rebuilt on load and is bit-identical, because
//! positions and radii are already on the 1 mm grid.

use super::geom::Pi;
use super::net::{EdgeData, LineData, Network, NodeData, StationData};
use super::params::*;

/// TWT3 (T-079): an edge may carry the points it was drawn through (tracks | 8, then count and
/// points like PIs, no radius). Only edges drawn between T-079 and T-092 have them (`EdgeData::
/// thru`); they are kept so such an edge reshapes from its clicks, and are written back as long
/// as it is unchanged. TWT2 (T-055: constructed flags on stations, node flag 4, and edges,
/// tracks | 4) has none and still loads.
const MAGIC: &[u8; 4] = b"TWT3";
const MAGIC_TWT2: &[u8; 4] = b"TWT2";

struct W(Vec<u8>);
impl W {
    fn u(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(b);
                return;
            }
            self.0.push(b | 0x80);
        }
    }
    fn i(&mut self, v: i64) {
        self.u(((v << 1) ^ (v >> 63)) as u64);
    }
    fn mm(&mut self, v: f64) {
        self.i((v / QUANT).round() as i64);
    }
    fn s(&mut self, v: &str) {
        self.u(v.len() as u64);
        self.0.extend_from_slice(v.as_bytes());
    }
    fn tenth(&mut self, v: f32) {
        self.u((v as f64 * 10.0).round().max(0.0) as u64);
    }
}

struct R<'a>(&'a [u8], usize);
impl R<'_> {
    fn u(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        let mut sh = 0;
        loop {
            let b = *self.0.get(self.1).ok_or("truncated")?;
            self.1 += 1;
            v |= ((b & 0x7f) as u64) << sh;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            sh += 7;
            if sh > 63 {
                return Err("bad varint".into());
            }
        }
    }
    fn i(&mut self) -> Result<i64, String> {
        let u = self.u()?;
        Ok((u >> 1) as i64 ^ -((u & 1) as i64))
    }
    fn mm(&mut self) -> Result<f64, String> {
        Ok(self.i()? as f64 * QUANT)
    }
    fn s(&mut self) -> Result<String, String> {
        let n = self.u()? as usize;
        let b = self.0.get(self.1..self.1 + n).ok_or("truncated")?;
        self.1 += n;
        String::from_utf8(b.to_vec()).map_err(|e| e.to_string())
    }
    fn tenth(&mut self) -> Result<f32, String> {
        Ok((self.u()? as f64 / 10.0) as f32)
    }
}

/// Encode the network's inputs.
pub fn encode(net: &Network) -> Vec<u8> {
    let mut w = W(MAGIC.to_vec());
    w.0.extend_from_slice(&net.origin_lon.to_le_bytes());
    w.0.extend_from_slice(&net.origin_lat.to_le_bytes());
    w.u(net.right_hand as u64);
    let nodes: Vec<u32> = (0..net.node_count() as u32).filter(|&n| net.node_ok(n)).collect();
    let edges: Vec<u32> = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e)).collect();
    let lines: Vec<u32> = (0..net.line_count() as u32).filter(|&l| net.line_ok(l)).collect();
    let mut node_id = vec![u32::MAX; net.node_count()];
    for (i, &n) in nodes.iter().enumerate() {
        node_id[n as usize] = i as u32;
    }
    let mut edge_id = vec![u32::MAX; net.edge_count()];
    for (i, &e) in edges.iter().enumerate() {
        edge_id[e as usize] = i as u32;
    }
    w.u(nodes.len() as u64);
    let (mut px, mut py) = (0.0, 0.0);
    for &n in &nodes {
        let i = n as usize;
        w.mm(net.node_x[i] - px);
        w.mm(net.node_y[i] - py);
        (px, py) = (net.node_x[i], net.node_y[i]);
        w.i(net.node_level[i] as i64);
        let plat = net.node_platform[i];
        w.u(net.node_flying[i] as u64 | ((plat > 0) as u64) << 1 | ((plat > 0 && net.node_built[i]) as u64) << 2);
        if plat > 0 {
            w.u(plat as u64);
            w.s(&net.node_name[i]);
        }
    }
    w.u(edges.len() as u64);
    let mut prev_a = 0i64;
    for &e in &edges {
        let d = net.edge_data(e).unwrap();
        let a = node_id[d.a as usize] as i64;
        w.i(a - prev_a);
        w.i(node_id[d.b as usize] as i64 - a);
        prev_a = a;
        w.u(d.tracks as u64 | (d.built as u64) << 2 | (!d.thru.is_empty() as u64) << 3);
        w.u(d.pis.len() as u64);
        let (mut px, mut py) = (net.node_x[d.a as usize], net.node_y[d.a as usize]);
        for p in &d.pis {
            w.mm(p.x - px);
            w.mm(p.y - py);
            (px, py) = (p.x, p.y);
            w.u((p.radius / QUANT).round() as u64);
            w.i(p.level as i64);
        }
        if !d.thru.is_empty() {
            w.u(d.thru.len() as u64);
            let (mut px, mut py) = (net.node_x[d.a as usize], net.node_y[d.a as usize]);
            for p in &d.thru {
                w.mm(p.x - px);
                w.mm(p.y - py);
                (px, py) = (p.x, p.y);
                w.i(p.level as i64);
            }
        }
    }
    w.u(lines.len() as u64);
    for &l in &lines {
        let d = net.line_data(l).unwrap();
        w.s(&d.name);
        w.u(d.colour as u64);
        for t in d.tph {
            w.tenth(t);
        }
        w.tenth(d.dwell_s);
        w.tenth(d.turnaround_s);
        w.u(d.cars as u64);
        w.u(d.stops.len() as u64);
        let mut prev = 0i64;
        for &s in &d.stops {
            let v = node_id[s as usize] as i64;
            w.i(v - prev);
            prev = v;
        }
        w.u(d.path.len() as u64);
        let mut prev = 0i64;
        for &p in &d.path {
            let v = edge_id[(p >> 1) as usize] as i64;
            w.i(((v - prev) << 1) | (p & 1) as i64);
            prev = v;
        }
    }
    w.0
}

/// Decode into a fresh network (dense ids). Derive with `TrackWorld::new`.
pub fn decode(bytes: &[u8]) -> Result<Network, String> {
    if bytes.len() < 20 || (&bytes[..4] != MAGIC && &bytes[..4] != MAGIC_TWT2) {
        return Err("not a trainworld track save".into());
    }
    let f = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let mut net = Network::new(f(4), f(12), false);
    let mut r = R(bytes, 20);
    net.right_hand = r.u()? == 1;
    let n_nodes = r.u()? as usize;
    let (mut px, mut py) = (0.0, 0.0);
    for _ in 0..n_nodes {
        let id = net.alloc_node();
        let (x, y) = (quant(px + r.mm()?), quant(py + r.mm()?));
        (px, py) = (x, y);
        let level = r.i()? as i8;
        let flags = r.u()?;
        net.set_node(id, Some(NodeData { x, y, level, flying: flags & 1 == 1 })).map_err(|e| format!("{e:?}"))?;
        if flags & 2 != 0 {
            let platform = r.u()? as u16;
            let name = r.s()?;
            net.set_station(id, Some(StationData { platform, name, built: flags & 4 != 0 })).map_err(|e| format!("{e:?}"))?;
        }
    }
    let n_edges = r.u()? as usize;
    let mut prev_a = 0i64;
    for _ in 0..n_edges {
        let id = net.alloc_edge();
        let a = prev_a + r.i()?;
        let b = a + r.i()?;
        prev_a = a;
        let tb = r.u()?;
        let (tracks, built) = ((tb & 3) as u8, tb & 4 != 0);
        let n = r.u()? as usize;
        let (mut px, mut py) = (net.node_x[a as usize], net.node_y[a as usize]);
        let mut pis = Vec::with_capacity(n);
        for _ in 0..n {
            let (x, y) = (quant(px + r.mm()?), quant(py + r.mm()?));
            (px, py) = (x, y);
            let radius = r.u()? as f64 * QUANT;
            let level = r.i()? as i8;
            pis.push(Pi { x, y, radius, level });
        }
        let mut thru = vec![];
        if tb & 8 != 0 {
            let (mut px, mut py) = (net.node_x[a as usize], net.node_y[a as usize]);
            for _ in 0..r.u()? {
                let (x, y) = (quant(px + r.mm()?), quant(py + r.mm()?));
                (px, py) = (x, y);
                thru.push(Pi { x, y, radius: 0.0, level: r.i()? as i8 });
            }
        }
        net.set_edge(id, Some(EdgeData { a: a as u32, b: b as u32, tracks, pis, built, thru })).map_err(|e| format!("{e:?}"))?;
    }
    let n_lines = r.u()? as usize;
    for _ in 0..n_lines {
        let id = net.alloc_line();
        let name = r.s()?;
        let colour = r.u()? as u32;
        let tph = [r.tenth()?, r.tenth()?, r.tenth()?];
        let (dwell_s, turnaround_s) = (r.tenth()?, r.tenth()?);
        let cars = r.u()? as u8;
        let mut stops = vec![];
        let mut prev = 0i64;
        for _ in 0..r.u()? {
            prev += r.i()?;
            stops.push(prev as u32);
        }
        let mut path = vec![];
        let mut prev = 0i64;
        for _ in 0..r.u()? {
            let v = r.i()?;
            prev += v >> 1;
            path.push((prev as u32) << 1 | (v & 1) as u32);
        }
        net.set_line(id, Some(LineData { name, colour, stops, path, tph, dwell_s, turnaround_s, cars })).map_err(|e| format!("{e:?}"))?;
    }
    net.take_touch();
    Ok(net)
}
