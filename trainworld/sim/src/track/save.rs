//! The network part of a save (SPEC 11, notes/T-008.md 4.6): inputs only, ids renumbered densely,
//! varints, positions as millimetre deltas. Compression (deflate, `CompressionStream` in the
//! browser) is the caller's. Derived geometry is rebuilt on load and is bit-identical, because
//! positions and radii are already on the 1 mm grid.

use super::frame::FrameKind;
use super::geom::Pi;
use super::net::{EdgeData, GroupLook, LineData, Network, NodeData, StationData};
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
/// lines running high-speed trains; without it all are ordinary. A trailing `LBS2` block (T-1465;
/// `LBS1`, T-1464, is the same without a kind per point) gives the bus lines their routes; without
/// it there are none. A trailing `LRT1` block (T-1787) carries the settled rates of the lines with
/// trains to the bit, and the lines the settle left circling; without it a load settles in full.
/// A trailing `LFR1` block (T-2193) makes stations ports (their berths) and gives the ferry lines
/// their routes; without it there are none.
const MAGIC: &[u8; 4] = b"TWT4";
/// TWT5 (T-125) is TWT4 byte for byte except that the metres are in the Lambert conformal world
/// frame (`frame.rs`). TWT4 and older are always the legacy equirectangular frame, and a network
/// in that frame still encodes as TWT4, identical to what earlier builds wrote.
const MAGIC_TWT5: &[u8; 4] = b"TWT5";
/// TWT6 (T-306) is TWT5 byte for byte except that the metres are in the Mercator frame of the whole
/// Earth (`frame.rs`); the origin fields keep the game's home origin, which the frame does not use.
/// Radii are ground metres, as the player set them.
const MAGIC_TWT6: &[u8; 4] = b"TWT6";

/// T-1826: the trailing blocks this build reads, in the order they are written. A block it does not
/// know stops the decoder, which then reports the bytes left over (`decode_read`).
pub const BLOCKS: [&[u8; 4]; 12] = [b"LUI1", b"LNK1", b"LAU1", b"LHS1", b"LBS1", b"LBS2", b"LRT1", b"LGR1", b"LGL1", b"LOR1", b"LPH1", b"LFR1"];
/// T-1826: the save level this build writes into the game header (`level`) and reads up to. Raised
/// with every new block, every new field inside a header field (a new top-level header field is seen
/// as unknown on its own) and every new meaning of existing bytes, so an older build knows a newer
/// save even when it has nothing left over. Saves from before it have none, read as 0.
pub const LEVEL: u32 = 3;

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
        // T-1281: a station's upgrades in bits 4 to 6, which older readers ignore
        let ups = if plat > 0 { net.node_upgrades[i] as u64 } else { 0 };
        w.u(net.node_flying[i] as u64 | ((plat > 0) as u64) << 1 | ((plat > 0 && net.node_built[i]) as u64) << 2 | ((plat > 0 && net.node_off[i] && !net.node_built[i]) as u64) << 3 | ups << 4);
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
        w.u(d.tracks as u64 | (d.built as u64) << 2 | (!d.thru.is_empty() as u64) << 3 | ((d.off && !d.built) as u64) << 4 | (d.crossover as u64) << 5 | (d.hsr as u64) << 6 | (d.sep as u64) << 7);
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
    // T-1464, T-1465: the bus lines' routes, after LHS1 and only when there are any: count, then per
    // line its index, constructed (1/0), point count and per point its mm deltas from the previous
    // and its `bus::K_*` bits
    let buses: Vec<(usize, &super::bus::BusRoute)> = lines.iter().enumerate().filter_map(|(i, &l)| net.line_bus[l as usize].as_ref().map(|r| (i, r))).collect();
    if !buses.is_empty() {
        w.0.extend_from_slice(b"LBS2");
        w.u(buses.len() as u64);
        for (i, r) in buses {
            w.u(i as u64);
            w.u(r.built as u64);
            w.u(r.points.len() as u64);
            let (mut px, mut py) = (0.0, 0.0);
            for (p, &k) in r.points.iter().zip(&r.kinds) {
                w.mm(p[0] - px);
                w.mm(p[1] - py);
                w.u(k as u64);
                (px, py) = (p[0], p[1]);
            }
        }
    }
    // T-1787: the settled rates of the lines with trains, to the bit (count, then index and three
    // f32 bits), then the circling lines the settle kept (count, then index, level, rate bits, round
    // trip bits, trains), after LBS2 and only once a settle has run
    let mut index = vec![u32::MAX; net.line_count()];
    for (i, &l) in lines.iter().enumerate() { index[l as usize] = i as u32; }
    let rated: Vec<usize> = lines.iter().enumerate().filter(|(_, &l)| net.line_trains[l as usize].is_some()).map(|(i, _)| i).collect();
    if net.rates_settled && !rated.is_empty() {
        w.0.extend_from_slice(b"LRT1");
        w.u(rated.len() as u64);
        for i in rated {
            w.u(i as u64);
            for r in net.line_tph[lines[i] as usize] { w.u(r.to_bits() as u64); }
        }
        let kept: Vec<_> = net.settle_kept.iter().filter(|k| index.get(k.0 as usize).is_some_and(|&i| i != u32::MAX)).collect();
        w.u(kept.len() as u64);
        for &(l, lev, rate, trip, trains) in kept {
            w.u(index[l as usize] as u64);
            w.u(lev as u64);
            w.u(rate as u64);
            w.u(trip);
            w.u(trains as u64);
        }
    }
    // T-1791: the line groups, after LRT1 and only when there are any: count, then per
    // line its index and group; then count, then per named group its number and name
    let grouped: Vec<(usize, u32)> = lines.iter().enumerate().filter(|(_, &l)| net.line_group[l as usize] != 0).map(|(i, &l)| (i, net.line_group[l as usize])).collect();
    if !grouped.is_empty() {
        w.0.extend_from_slice(b"LGR1");
        w.u(grouped.len() as u64);
        for &(i, g) in &grouped {
            w.u(i as u64);
            w.u(g as u64);
        }
        let mut names: Vec<(u32, &str)> = vec![];
        for &l in &lines {
            let (g, s) = (net.line_group[l as usize], net.line_group_name[l as usize].as_str());
            if g != 0 && !s.is_empty() && !names.iter().any(|x| x.0 == g) { names.push((g, s)); }
        }
        w.u(names.len() as u64);
        for (g, s) in names {
            w.u(g as u64);
            w.s(s);
        }
    }
    // T-1813: the groups' bullets set by hand, after LGR1: count, then per group its number, text
    // and colour (0 for none, else the colour + 1)
    let mut looks: Vec<(u32, &GroupLook)> = vec![];
    for &l in &lines {
        let (g, k) = (net.line_group[l as usize], &net.line_group_look[l as usize]);
        if g != 0 && *k != GroupLook::default() && !looks.iter().any(|x| x.0 == g) { looks.push((g, k)); }
    }
    if !looks.is_empty() {
        w.0.extend_from_slice(b"LGL1");
        w.u(looks.len() as u64);
        for (g, k) in looks {
            w.u(g as u64);
            w.s(&k.symbol);
            w.u(k.colour.map_or(0, |c| c as u64 + 1));
        }
    }
    // T-1813: the Lines and Stations tabs' hand order: count, then the lines' indices first
    // first; then the same for nodes
    let rank_of = |rank: &[u32], id: u32| rank.get(id as usize).copied().unwrap_or(0);
    let mut lo: Vec<(u32, usize)> = lines.iter().enumerate().filter_map(|(i, &l)| Some((rank_of(&net.line_rank, l), i)).filter(|x| x.0 > 0)).collect();
    let mut no: Vec<(u32, usize)> = nodes.iter().enumerate().filter_map(|(i, &k)| Some((rank_of(&net.node_rank, k), i)).filter(|x| x.0 > 0)).collect();
    if !lo.is_empty() || !no.is_empty() {
        lo.sort_unstable();
        no.sort_unstable();
        w.0.extend_from_slice(b"LOR1");
        for v in [lo, no] {
            w.u(v.len() as u64);
            for (_, i) in v {
                w.u(i as u64);
            }
        }
    }
    // T-1925: the lines' departure phases, after LOR1 and only when some line has one: count, then
    // per line its index and three f32 bits
    let phased: Vec<(usize, [f32; 3])> = lines.iter().enumerate().map(|(i, &l)| (i, net.line_phase_of(l))).filter(|x| x.1 != [0.0; 3]).collect();
    if !phased.is_empty() {
        w.0.extend_from_slice(b"LPH1");
        w.u(phased.len() as u64);
        for (i, p) in phased {
            w.u(i as u64);
            for x in p { w.u(x.to_bits() as u64); }
        }
    }
    // T-2193: the ports and the ferry lines, after LPH1 and only when there are any: count, then per
    // port its node index and its berth's mm deltas from the node; count, then per ferry line its
    // index, constructed (1/0), point count and per point its mm deltas from the previous, its
    // `ferry::K_*` kind and its speed in mm/s
    let ports: Vec<(usize, [f64; 2])> = nodes.iter().enumerate().filter_map(|(i, &n)| net.node_berth[n as usize].filter(|_| net.is_station(n)).map(|b| (i, b))).collect();
    let ferries: Vec<(usize, &super::ferry::FerryRoute)> = lines.iter().enumerate().filter_map(|(i, &l)| net.line_ferry[l as usize].as_ref().map(|r| (i, r))).collect();
    if !ports.is_empty() || !ferries.is_empty() {
        w.0.extend_from_slice(b"LFR1");
        w.u(ports.len() as u64);
        for (i, b) in ports {
            let n = nodes[i] as usize;
            w.u(i as u64);
            w.mm(b[0] - net.node_x[n]);
            w.mm(b[1] - net.node_y[n]);
        }
        w.u(ferries.len() as u64);
        for (i, r) in ferries {
            w.u(i as u64);
            w.u(r.built as u64);
            w.u(r.points.len() as u64);
            let (mut px, mut py) = (0.0, 0.0);
            for ((p, &k), &v) in r.points.iter().zip(&r.kinds).zip(&r.v) {
                w.mm(p[0] - px);
                w.mm(p[1] - py);
                w.u(k as u64);
                w.u(super::ferry::v_mm(v));
                (px, py) = (p[0], p[1]);
            }
        }
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
            let d = net.station_data(n).map(|d| StationData { berth: d.berth.map(|b| { let (x, y) = at(b[0], b[1]); [x, y] }), ..d });
            out.set_station(id, d).map_err(|e| format!("{e:?}"))?;
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
        if let Some(mut d) = net.line_data(l) {
            if let Some(r) = d.bus.as_mut() {
                r.points = r.points.iter().map(|p| { let (x, y) = at(p[0], p[1]); [x, y] }).collect();
            }
            if let Some(r) = d.ferry.as_mut() {
                r.points = r.points.iter().map(|p| { let (x, y) = at(p[0], p[1]); [x, y] }).collect();
            }
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
    decode_read(bytes).map(|(net, _)| net)
}

/// `decode`, and how much of the save this build did not read (T-1826): the bytes after the last
/// block it knows, plus one per node or edge with flag bits it does not know. Not 0 means a newer
/// build wrote it, and writing the network back would lose that part.
pub fn decode_read(bytes: &[u8]) -> Result<(Network, usize), String> {
    let mut unknown = 0usize;
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
        unknown += (flags >> 12 != 0) as usize;
        net.set_node(id, Some(NodeData { x, y, level, flying: flags & 1 == 1 })).map_err(|e| format!("{e:?}"))?;
        if flags & 2 != 0 {
            let platform = r.u()? as u16;
            let name = r.s()?;
            net.set_station(id, Some(StationData { platform, name, built: flags & 4 != 0, off: flags & 8 != 0, upgrades: (flags >> 4) as u8, berth: None })).map_err(|e| format!("{e:?}"))?;
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
        unknown += (tb >> 8 != 0) as usize;
        let (tracks, built, off, crossover, hsr, sep) = ((tb & 3) as u8, tb & 4 != 0, tb & 16 != 0, tb & 32 != 0, tb & 64 != 0, tb & 128 != 0);
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
        net.set_edge(id, Some(EdgeData { a: a as u32, b: b as u32, tracks, pis, built, off, thru, crossover, hsr, sep })).map_err(|e| format!("{e:?}"))?;
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
        net.set_line(id, Some(LineData { symbol: String::new(), trains, auto: None, name, colour, stops, path, tph, dwell_s, turnaround_s, cars, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default() })).map_err(|e| format!("{e:?}"))?;
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
    // LBS1 (T-1464) has no kinds: every point a plain corner, so every stretch straight
    let bus_block = match r.0.get(r.1..r.1 + 4) {
        Some(b"LBS1") => Some(false),
        Some(b"LBS2") => Some(true),
        _ => None,
    };
    if let Some(with_kinds) = bus_block {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let built = r.u()? != 0;
            let (mut px, mut py) = (0.0, 0.0);
            let (mut points, mut kinds) = (vec![], vec![]);
            for _ in 0..r.u()? {
                let (x, y) = (quant(px + r.mm()?), quant(py + r.mm()?));
                (px, py) = (x, y);
                points.push([x, y]);
                kinds.push(if with_kinds { r.u()? as u8 } else { super::bus::K_CORNER });
            }
            if i < n_lines { net.line_bus[i] = Some(super::bus::BusRoute::new(points, kinds, built)); }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LRT1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let rates = [r.u()? as u32, r.u()? as u32, r.u()? as u32].map(f32::from_bits);
            if i < n_lines && net.line_trains[i].is_some() { net.line_tph[i] = rates; }
        }
        for _ in 0..r.u()? {
            let (l, lev, rate, trip, trains) = (r.u()? as u32, r.u()? as usize, r.u()? as u32, r.u()?, r.u()? as u32);
            if (l as usize) < n_lines && lev < 3 { net.settle_kept.push((l, lev, rate, trip, trains)); }
        }
        net.rates_settled = true;
        net.rates_from_save = true;
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LGR1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let (i, g) = (r.u()? as usize, r.u()? as u32);
            if i < n_lines { net.line_group[i] = g; }
        }
        for _ in 0..r.u()? {
            let (g, s) = (r.u()? as u32, r.s()?);
            for i in 0..n_lines {
                if g != 0 && net.line_group[i] == g { net.line_group_name[i] = s.clone(); }
            }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LGL1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let (g, symbol, c) = (r.u()? as u32, r.s()?, r.u()?);
            let look = GroupLook { symbol, colour: c.checked_sub(1).map(|c| (c as u32) & 0xff_ffff) };
            for i in 0..n_lines {
                if g != 0 && net.line_group[i] == g { net.line_group_look[i] = look.clone(); }
            }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LOR1") {
        r.1 += 4;
        for kind in 0..2 {
            let len = if kind == 0 { n_lines } else { net.node_count() };
            let mut rank = vec![0u32; len];
            for k in 0..r.u()? {
                let i = r.u()? as usize;
                if i < len && rank[i] == 0 { rank[i] = k as u32 + 1; }
            }
            if kind == 0 { net.line_rank = rank } else { net.node_rank = rank }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LPH1") {
        r.1 += 4;
        let mut phase = vec![[0.0f32; 3]; n_lines];
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let p = [r.u()? as u32, r.u()? as u32, r.u()? as u32].map(f32::from_bits);
            if i < n_lines { phase[i] = p; }
        }
        net.line_phase = phase;
        net.phases_placed = true;
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LFR1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let (dx, dy) = (r.mm()?, r.mm()?);
            if i < net.node_count() && net.is_station(i as u32) {
                let b = [quant(net.node_x[i] + dx), quant(net.node_y[i] + dy)];
                let d = super::net::StationData { berth: Some(b), ..net.station_data(i as u32).unwrap() };
                net.set_station(i as u32, Some(d)).map_err(|e| format!("{e:?}"))?;
            }
        }
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let built = r.u()? != 0;
            let (mut px, mut py) = (0.0, 0.0);
            let (mut points, mut kinds, mut v) = (vec![], vec![], vec![]);
            for _ in 0..r.u()? {
                let (x, y) = (quant(px + r.mm()?), quant(py + r.mm()?));
                (px, py) = (x, y);
                points.push([x, y]);
                kinds.push(r.u()? as u8);
                v.push(r.u()? as f64 / 1000.0);
            }
            if i < n_lines { net.line_ferry[i] = Some(super::ferry::FerryRoute::new(points, kinds, v, built)); }
        }
    }
    net.take_touch();
    Ok((net, bytes.len() - r.1 + unknown))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small network with a station, an edge and a line, in the world frame.
    fn sample() -> Vec<u8> {
        let mut net = Network::new_in(FrameKind::Merc, -73.985, 40.758, true);
        for x in [0.0f64, 1000.0] {
            let n = net.alloc_node();
            net.set_node(n, Some(NodeData { x, y: 0.0, level: 0, flying: false })).unwrap();
            net.set_station(n, Some(StationData { platform: 2, name: format!("S{x}"), built: true, off: false, upgrades: 0, berth: None })).unwrap();
        }
        let e = net.alloc_edge();
        net.set_edge(e, Some(EdgeData { a: 0, b: 1, tracks: 2, pis: vec![], built: true, off: false, thru: vec![], crossover: false, hsr: false, sep: false })).unwrap();
        let l = net.alloc_line();
        net.set_line(l, Some(LineData { symbol: String::new(), trains: Some([2, 2, 1]), auto: None, name: "A".into(), colour: 0xff0000, stops: vec![0, 1], path: vec![0], tph: [0.0; 3], dwell_s: 30.0, turnaround_s: 120.0, cars: 4, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default() })).unwrap();
        encode(&net)
    }

    #[test]
    fn a_save_this_build_wrote_is_read_whole() {
        let bytes = sample();
        let (net, unread) = decode_read(&bytes).unwrap();
        assert_eq!(unread, 0);
        assert_eq!(encode(&net), bytes);
    }

    #[test]
    fn an_unknown_trailing_block_is_reported() {
        let mut bytes = sample();
        let known = decode(&bytes).unwrap();
        bytes.extend_from_slice(b"ZZZ1");
        bytes.extend_from_slice(&[3, 1, 2, 3]);
        let (net, unread) = decode_read(&bytes).unwrap();
        assert_eq!(unread, 8, "the unknown block's tag and body are left over");
        // what it does know loads as before
        assert_eq!(encode(&net), encode(&known));
    }

    #[test]
    fn an_unknown_block_before_known_ones_leaves_them_unread_too() {
        let bytes = sample();
        let at = bytes.windows(4).position(|w| w == b"LUI1").unwrap();
        let mut newer = bytes[..at].to_vec();
        newer.extend_from_slice(b"ZZZ1\x00");
        newer.extend_from_slice(&bytes[at..]);
        assert_eq!(decode_read(&newer).unwrap().1, bytes.len() - at + 5);
    }

    #[test]
    fn unknown_flag_bits_are_reported() {
        // two plain nodes and a single-track edge, written by hand with the flags given
        let raw = |node: u64, edge: u64| {
            let mut w = W(MAGIC_TWT6.to_vec());
            w.0.extend_from_slice(&0f64.to_le_bytes());
            w.0.extend_from_slice(&0f64.to_le_bytes());
            w.u(1);
            w.u(2);
            for dx in [0.0, 100.0] {
                w.mm(dx);
                w.mm(0.0);
                w.i(0);
                w.u(node);
            }
            w.u(1);
            w.i(0);
            w.i(1);
            w.u(edge);
            w.u(0);
            w.u(0);
            w.0
        };
        assert_eq!(decode_read(&raw(0, 1)).unwrap().1, 0);
        assert_eq!(decode_read(&raw(1 << 12, 1)).unwrap().1, 2, "both nodes");
        assert_eq!(decode_read(&raw(0, 1 | 1 << 8)).unwrap().1, 1);
    }

    /// Adding a block means listing it in BLOCKS and raising LEVEL; this pin fails until both are done.
    #[test]
    fn every_written_block_is_listed_and_counted_in_the_level() {
        let src = include_str!("save.rs");
        let mut written: Vec<&str> = vec![];
        for part in src.split("w.0.extend_from_slice(b\"").skip(1) {
            let tag = &part[..4];
            if !written.contains(&tag) { written.push(tag); }
        }
        for tag in &written {
            assert!(BLOCKS.iter().any(|b| &b[..] == tag.as_bytes()), "block {tag} is written but not in BLOCKS");
        }
        assert_eq!((LEVEL, BLOCKS.len()), (3, 12), "a new block raises LEVEL: raise it, then update this pin");
    }
}
