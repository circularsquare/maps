//! The network part of a save (SPEC 11, notes/T-008.md 4.6): inputs only, ids renumbered densely,
//! varints, positions as millimetre deltas. Compression (deflate, `CompressionStream` in the
//! browser) is the caller's. Derived geometry is rebuilt on load and is bit-identical, because
//! positions and radii are already on the 1 mm grid.

use super::frame::FrameKind;
use super::geom::Pi;
use super::net::{EdgeData, LineData, Network, NodeData, StationData};
use super::params::*;

/// TWT4 (T-104) is the oldest format read (TWT2 and TWT3 stopped loading, T-1146). A line carries
/// its whole-line train allocation; derived frequencies are not saved for allocated lines. A line
/// saved without one (a flag of 0, then its trains per hour) is given trains from them when the
/// network is derived (`TrackWorld::allocate_unallocated`). An edge may carry the points it was
/// drawn through (tracks | 8, then count and points like PIs, no radius; `EdgeData::thru`): kept
/// so such an edge reshapes from its clicks, and written back as long as it is unchanged.
/// Optional trailing blocks follow the lines: `LUI1` (line numbers and symbols) and, when the
/// player linked stations, `LNK1` (T-197: count, then node pairs in the dense ids). A save without
/// them loads with none. An edge's flags `| 32` mark a crossover's diagonal (T-338), `| 64`
/// high-speed track (T-817); older saves have none. A trailing `LHS1` block (T-817) lists the
/// lines running high-speed trains; without it all are ordinary.
const MAGIC: &[u8; 4] = b"TWT4";
/// TWT5 (T-125) is TWT4 byte for byte except that the metres are in the Lambert conformal world
/// frame (`frame.rs`). TWT4 and older are always the legacy equirectangular frame, and a network
/// in that frame still encodes as TWT4, identical to what earlier builds wrote.
const MAGIC_TWT5: &[u8; 4] = b"TWT5";
/// TWT6 (T-306) is TWT5 byte for byte except that the metres are in the Mercator frame of the whole
/// Earth (`frame.rs`); the origin fields keep the game's home origin, which the frame does not use.
/// Radii are ground metres, as the player set them.
const MAGIC_TWT6: &[u8; 4] = b"TWT6";

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
    let mut w = W(match net.frame_kind {
        FrameKind::Lcc => MAGIC_TWT5,
        FrameKind::Merc => MAGIC_TWT6,
        FrameKind::Equirect => MAGIC,
    }
    .to_vec());
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
        w.u(net.node_flying[i] as u64 | ((plat > 0) as u64) << 1 | ((plat > 0 && net.node_built[i]) as u64) << 2 | ((plat > 0 && net.node_off[i] && !net.node_built[i]) as u64) << 3);
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
        w.u(d.tracks as u64 | (d.built as u64) << 2 | (!d.thru.is_empty() as u64) << 3 | ((d.off && !d.built) as u64) << 4 | (d.crossover as u64) << 5 | (d.hsr as u64) << 6);
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
        for t in if d.trains.is_some() { [0.0; 3] } else { d.tph } {
            w.tenth(t);
        }
        w.u(d.trains.is_some() as u64);
        if let Some(trains) = d.trains { for count in trains { w.u(count as u64); } }
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
    // Optional trailing inputs: older TWT4 readers safely ignore this extension.
    w.0.extend_from_slice(b"LUI1");
    w.u(net.next_line_number as u64);
    w.u(lines.len() as u64);
    for &l in &lines { w.s(&net.line_symbol[l as usize]); }
    // T-197: the player's station links, after LUI1 and only when there are any, so a network
    // without links encodes exactly as before and older readers stop at LUI1.
    let links: Vec<[u32; 2]> = net.links_live().collect();
    if !links.is_empty() {
        w.0.extend_from_slice(b"LNK1");
        w.u(links.len() as u64);
        for [a, b] in links {
            w.u(node_id[a as usize] as u64);
            w.u(node_id[b as usize] as u64);
        }
    }
    // T-625: the lines whose train counts follow a headway, after LNK1 and only when there are any
    let autos: Vec<(usize, [u32; 3])> = lines.iter().enumerate().filter_map(|(i, &l)| net.line_auto[l as usize].map(|a| (i, a))).collect();
    if !autos.is_empty() {
        w.0.extend_from_slice(b"LAU1");
        w.u(autos.len() as u64);
        for (i, a) in autos {
            w.u(i as u64);
            for s in a { w.u(s as u64); }
        }
    }
    // T-817: the lines running high-speed trains, after LAU1 and only when there are any
    let fast: Vec<usize> = lines.iter().enumerate().filter(|(_, &l)| net.line_fast[l as usize]).map(|(i, _)| i).collect();
    if !fast.is_empty() {
        w.0.extend_from_slice(b"LHS1");
        w.u(fast.len() as u64);
        for i in fast { w.u(i as u64); }
    }
    w.0
}

/// Re-express a save's network in another frame (T-125). Every node, PI and drawn-through point
/// keeps its place on the ground (it goes through longitude and latitude and is put back on the
/// 1 mm grid); ids, levels, radii, stations, lines and flags are copied as they are. The geometry
/// is re-derived on load, so lengths and costs follow the new frame (a legacy network gets about
/// 1.2% longer around Philadelphia); money, schedules and fares are not touched. Not applied to
/// any save unasked: a legacy save stays in its frame unless a player converts it.
pub fn convert_frame(bytes: &[u8], to: FrameKind) -> Result<Vec<u8>, String> {
    let net = decode(bytes)?;
    if net.frame_kind == to {
        return Ok(encode(&net));
    }
    // The target frame at the save's home origin (a Mercator frame keeps it but does not use it).
    let (from, into) = (net.frame(), super::frame::Frame::new(to, net.origin_lon, net.origin_lat));
    let at = |x: f64, y: f64| {
        let (lon, lat) = from.to_lonlat(x, y);
        let (x, y) = into.from_lonlat(lon, lat);
        (quant(x), quant(y))
    };
    let mut out = Network::new_in(to, net.origin_lon, net.origin_lat, net.right_hand);
    for n in 0..net.node_count() as u32 {
        let id = out.alloc_node();
        if !net.node_ok(n) {
            continue;
        }
        let i = n as usize;
        let (x, y) = at(net.node_x[i], net.node_y[i]);
        out.set_node(id, Some(NodeData { x, y, level: net.node_level[i], flying: net.node_flying[i] })).map_err(|e| format!("{e:?}"))?;
        if net.node_platform[i] > 0 {
            out.set_station(id, Some(StationData { platform: net.node_platform[i], name: net.node_name[i].clone(), built: net.node_built[i], off: net.node_off[i] })).map_err(|e| format!("{e:?}"))?;
        }
    }
    let moved = |p: &Pi| {
        let (x, y) = at(p.x, p.y);
        Pi { x, y, ..*p }
    };
    for e in 0..net.edge_count() as u32 {
        let id = out.alloc_edge();
        if let Some(d) = net.edge_data(e) {
            let d = EdgeData { pis: d.pis.iter().map(&moved).collect(), thru: d.thru.iter().map(&moved).collect(), ..d };
            out.set_edge(id, Some(d)).map_err(|e| format!("{e:?}"))?;
        }
    }
    for l in 0..net.line_count() as u32 {
        let id = out.alloc_line();
        if let Some(d) = net.line_data(l) {
            out.set_line(id, Some(d)).map_err(|e| format!("{e:?}"))?;
            out.line_symbol[id as usize] = net.line_symbol[l as usize].clone();
        }
    }
    out.next_line_number = net.next_line_number;
    out.links = net.links.clone();
    Ok(encode(&out))
}

/// Decode into a fresh network (dense ids). Derive with `TrackWorld::new`.
pub fn decode(bytes: &[u8]) -> Result<Network, String> {
    if bytes.len() < 20 || (&bytes[..4] != MAGIC && &bytes[..4] != MAGIC_TWT5 && &bytes[..4] != MAGIC_TWT6) {
        return Err("not a trainworld track save".into());
    }
    let f = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let kind = match &bytes[..4] {
        b"TWT5" => FrameKind::Lcc,
        b"TWT6" => FrameKind::Merc,
        _ => FrameKind::Equirect,
    };
    let mut net = Network::new_in(kind, f(4), f(12), false);
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
            net.set_station(id, Some(StationData { platform, name, built: flags & 4 != 0, off: flags & 8 != 0 })).map_err(|e| format!("{e:?}"))?;
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
        let (tracks, built, off, crossover, hsr) = ((tb & 3) as u8, tb & 4 != 0, tb & 16 != 0, tb & 32 != 0, tb & 64 != 0);
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
        net.set_edge(id, Some(EdgeData { a: a as u32, b: b as u32, tracks, pis, built, off, thru, crossover, hsr })).map_err(|e| format!("{e:?}"))?;
    }
    let n_lines = r.u()? as usize;
    for _ in 0..n_lines {
        let id = net.alloc_line();
        let name = r.s()?;
        let colour = r.u()? as u32;
        let tph = [r.tenth()?, r.tenth()?, r.tenth()?];
        let trains = if r.u()? != 0 { Some([r.u()? as u32, r.u()? as u32, r.u()? as u32]) } else { None };
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
        net.set_line(id, Some(LineData { symbol: String::new(), trains, auto: None, name, colour, stops, path, tph, dwell_s, turnaround_s, cars, fast: false })).map_err(|e| format!("{e:?}"))?;
    }
    net.next_line_number = n_lines as u32 + 1;
    if r.0.get(r.1..r.1 + 4) == Some(b"LUI1") {
        r.1 += 4;
        net.next_line_number = (r.u()? as u32).max(1);
        if r.u()? as usize != n_lines { return Err("bad line UI extension".into()); }
        for i in 0..n_lines { net.line_symbol[i] = r.s()?; }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LNK1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let (a, b) = (r.u()? as u32, r.u()? as u32);
            // a pair naming no station is dropped rather than refusing the whole save
            net.link(a, b);
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LAU1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let auto = [r.u()? as u32, r.u()? as u32, r.u()? as u32];
            if i < n_lines && net.line_trains[i].is_some() { net.line_auto[i] = Some(auto); }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LHS1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            if i < n_lines { net.line_fast[i] = true; }
        }
    }
    net.take_touch();
    Ok(net)
}
