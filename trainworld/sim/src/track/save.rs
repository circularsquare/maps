//! The network part of a save (SPEC 11, notes/T-008.md 4.6): inputs only, ids renumbered densely,
//! varints, positions as millimetre deltas. Compression (deflate, `CompressionStream` in the
//! browser) is the caller's. Derived geometry is rebuilt on load and is bit-identical, because
//! positions and radii are already on the 1 mm grid.

use super::frame::FrameKind;
use super::geom::Pi;
use super::net::{EdgeData, GroupLook, LineData, Network, NodeData, StationData, DAY_UNKNOWN};
use super::params::*;
use super::twin::Twin;

/// TWT4 (T-104) is the oldest format read (TWT2 and TWT3 stopped loading, T-1146). A line carries
/// its whole-line train allocation; derived frequencies are not saved for allocated lines. A line
/// saved without one (a flag of 0, then its trains per hour) is given trains from them when the
/// network is derived (`TrackWorld::allocate_unallocated`). An edge may carry the points it was
/// drawn through (tracks | 8, then count and points like PIs, no radius; `EdgeData::thru`): kept
/// so such an edge reshapes from its clicks, and written back as long as it is unchanged.
/// Optional trailing blocks follow the lines: `LUI1` (line numbers and symbols) and, when the
/// player linked stations, `LNK1` (T-197: count, then node pairs in the dense ids). A save without
/// them loads with none. An edge's flags `| 32` mark a crossover's diagonal (T-338), `| 64`
/// high-speed track (T-817); older saves have none; bits 11-12 give one-way single track (T-3094: 0 not set, 1 a to b, 2 b to a, 3 both ways, T-3137), bits 13-14 and 15-16 the joins of its ends a and b (T-3137, `EdgeData::join`). A trailing `LHS1` block (T-817) lists the
/// lines running high-speed trains; without it all are ordinary. A trailing `LBS2` block (T-1465;
/// `LBS1`, T-1464, is the same without a kind per point) gives the bus lines their routes; without
/// it there are none. A trailing `LRT1` block (T-1787) carries the settled rates of the lines with
/// trains to the bit, and the lines the settle left circling; without it a load settles in full.
/// A trailing `LFR2` block (T-2193, T-2450) makes stations ports (their berths) and gives the ferry
/// lines their routes; without it there are none. `LFR1` is the same with its speeds searched at 25
/// km/h in open water, raised on load (`ferry::v_from_lfr1`).
/// A trailing `BLD1` block (T-2217) carries the station buildings; without it there are none.
/// A trailing `LBD1` block (T-2871) carries the day each constructed station, edge and line was
/// first built; without it, or where it is damaged, they read as built before recording began.
/// A trailing `LBP1` block (T-2885) gives the bus lines with a stop spacing of their own theirs (count,
/// then index and metres); without it every bus line's stops place themselves.
/// A trailing `LHO1` block (T-2895) lists the lines whose stop order the player set by hand (count,
/// then indices); without it none is.
/// A trailing `LMN1` block (T-3023, written until T-3109 dropped manual mode) lists lines that were in manual
/// mode (count, then indices); a plain out-and-back among them folds into its open line (`fold_manual`).
/// A trailing `LBH1` block (T-2891) gives the bus lines whose stops the player places those stops
/// (count, then per line its index, its stop count and the stops like PIs, no radius); without it
/// no line's are.
/// A trailing `LOW1` block (T-2981) lists the lines whose loop runs the other way from their mode's
/// default (count, then indices): a train line one way, a bus or ferry line both ways.
/// A trailing `LSL1` block (T-2987) gives the rail lines with a speed limit of their own (count, then
/// index and km/h); without it none has one.
/// A trailing `LPC1` block (T-3233), written with `LRT1` when any line is paced, gives the count
/// lines paced at a merge (count, then index and the three levels' headways as f32 bits); a load
/// keeps that pacing, and without the block it paces them again (T-2973).
/// A trailing `LTW1` block (T-3333) gives the lines run on both pairs of quad track and their hidden
/// twins (count, then index and kind: 1 a line on one pair for now, 2 on both, 3 a twin, then its
/// line's index); without it no line has one.
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
pub const BLOCKS: [&[u8; 4]; 24] = [b"LUI1", b"LNK1", b"LAU1", b"LHS1", b"LBS1", b"LBS2", b"LRT1", b"LGR1", b"LGL1", b"LOR1", b"LPH1", b"LSP1", b"LFR1", b"LFR2", b"BLD1", b"LBD1", b"LBP1", b"LHO1", b"LMN1", b"LBH1", b"LOW1", b"LSL1", b"LPC1", b"LTW1"];
/// T-1826: the save level this build writes into the game header (`level`) and reads up to. Raised
/// with every new block, every new field inside a header field (a new top-level header field is seen
/// as unknown on its own) and every new meaning of existing bytes, so an older build knows a newer
/// save even when it has nothing left over. Saves from before it have none, read as 0.
pub const LEVEL: u32 = 18;

/// An edge's kind in the flag bits 8-10. Regular and high speed write nothing here (high speed keeps
/// bit 6), so their saves are byte for byte what earlier builds wrote and earlier builds read them.
fn edge_kind_bits(kind: TrackKind) -> u64 {
    if (kind as u8) < 2 { 0 } else { (kind as u64) << 8 }
}

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
        // T-1281: a station's upgrades in bits 4 to 6, which older readers ignore; T-2683: bit 8 is
        // a name the player typed (masked off as an upgrade there); T-3332: bits 9 and 10 a one-track
        // station's way (`StationData::one`)
        let ups = if plat > 0 { net.node_upgrades[i] as u64 } else { 0 };
        let one = if plat > 0 { net.node_one[i] as u64 } else { 0 };
        w.u(net.node_flying[i] as u64 | ((plat > 0) as u64) << 1 | ((plat > 0 && net.node_built[i]) as u64) << 2 | ((plat > 0 && net.node_off[i] && !net.node_built[i]) as u64) << 3 | ups << 4 | ((plat > 0 && net.node_named[i]) as u64) << 8 | one << 9);
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
        w.u(d.tracks as u64 | (d.built as u64) << 2 | (!d.thru.is_empty() as u64) << 3 | ((d.off && !d.built) as u64) << 4 | (d.crossover as u64) << 5 | (d.kind.is_hsr() as u64) << 6 | (d.sep as u64) << 7 | edge_kind_bits(d.kind) | (if d.tracks == 1 { d.oneway as u64 | (d.join[0] as u64) << 2 | (d.join[1] as u64) << 4 } else { 0 }) << 11);
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
            let v = node_id.get(s as usize).copied().unwrap_or(u32::MAX) as i64;
            w.i(v - prev);
            prev = v;
        }
        w.u(d.path.len() as u64);
        let mut prev = 0i64;
        for &p in &d.path {
            // T-2820: a broken line's deleted edge is written as none (`edge_id`'s u32::MAX)
            let v = edge_id.get((p >> 1) as usize).copied().unwrap_or(u32::MAX) as i64;
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
    // T-2972: where the lines' spare time stands, after LPH1 and only when some line's is not half at
    // each end: count, then per line its index and three f32 bits
    let split: Vec<(usize, [f32; 3])> = lines.iter().enumerate().map(|(i, &l)| (i, net.line_split_of(l))).filter(|x| x.1 != [0.5; 3]).collect();
    if !split.is_empty() {
        w.0.extend_from_slice(b"LSP1");
        w.u(split.len() as u64);
        for (i, p) in split {
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
        w.0.extend_from_slice(b"LFR2");
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
    // T-2217: the station buildings, after LFR2 and only when there are any: count, then per building
    // its name, corner count and corners as mm deltas from the previous (the first from 0, 0)
    let halls: Vec<&super::building::Building> = net.buildings.iter().flatten().collect();
    if !halls.is_empty() {
        w.0.extend_from_slice(b"BLD1");
        w.u(halls.len() as u64);
        for b in halls {
            w.s(&b.name);
            w.u(b.corners.len() as u64);
            let (mut px, mut py) = (0.0, 0.0);
            for p in &b.corners {
                w.mm(p[0] - px);
                w.mm(p[1] - py);
                (px, py) = (p[0], p[1]);
            }
        }
    }
    // T-2871: the build days, last, in the dense order of the constructed stations, edges and lines
    // (each list its count, then its days as deltas from the previous). A network with no known day
    // (an older save played on with no edit) writes nothing, so it saves back to the same bytes
    let stations: Vec<u32> = nodes.iter().filter(|&&n| net.is_station(n) && net.node_built[n as usize]).map(|&n| net.node_day[n as usize]).collect();
    let built_edges: Vec<u32> = edges.iter().filter(|&&e| net.edge_built[e as usize]).map(|&e| net.edge_day[e as usize]).collect();
    let built_lines: Vec<u32> = lines.iter().filter(|&&l| net.route_built(l)).map(|&l| net.line_day[l as usize]).collect();
    if stations.iter().chain(&built_edges).chain(&built_lines).any(|&d| d > DAY_UNKNOWN) {
        w.0.extend_from_slice(b"LBD1");
        for list in [&stations, &built_edges, &built_lines] {
            w.u(list.len() as u64);
            let mut prev = 0i64;
            for &d in list {
                w.i(d as i64 - prev);
                prev = d as i64;
            }
        }
    }
    let spaced: Vec<(usize, u32)> = lines.iter().enumerate().filter_map(|(i, &l)| net.line_bus[l as usize].as_ref().filter(|r| r.spacing > 0).map(|r| (i, r.spacing))).collect();
    if !spaced.is_empty() {
        w.0.extend_from_slice(b"LBP1");
        w.u(spaced.len() as u64);
        for (i, m) in spaced {
            w.u(i as u64);
            w.u(m as u64);
        }
    }
    let hand: Vec<usize> = lines.iter().enumerate().filter(|(_, &l)| net.line_hand[l as usize]).map(|(i, _)| i).collect();
    if !hand.is_empty() {
        w.0.extend_from_slice(b"LHO1");
        w.u(hand.len() as u64);
        for i in hand { w.u(i as u64); }
    }
    let by_hand: Vec<(usize, &[[f64; 2]])> = lines.iter().enumerate().filter_map(|(i, &l)| net.line_bus[l as usize].as_ref().and_then(|r| r.hand.as_deref()).map(|h| (i, h))).collect();
    if !by_hand.is_empty() {
        w.0.extend_from_slice(b"LBH1");
        w.u(by_hand.len() as u64);
        for (i, stops) in by_hand {
            w.u(i as u64);
            w.u(stops.len() as u64);
            let (mut px, mut py) = (0.0, 0.0);
            for p in stops {
                w.mm(p[0] - px);
                w.mm(p[1] - py);
                (px, py) = (p[0], p[1]);
            }
        }
    }
    let flipped: Vec<usize> = lines.iter().enumerate().filter(|(_, &l)| net.line_one_way[l as usize] != one_way_default(net, l as usize)).map(|(i, _)| i).collect();
    if !flipped.is_empty() {
        w.0.extend_from_slice(b"LOW1");
        w.u(flipped.len() as u64);
        for i in flipped { w.u(i as u64); }
    }
    let limited: Vec<(usize, u16)> = lines.iter().enumerate().filter(|(_, &l)| net.line_limit[l as usize] > 0).map(|(i, &l)| (i, net.line_limit[l as usize])).collect();
    if !limited.is_empty() {
        w.0.extend_from_slice(b"LSL1");
        w.u(limited.len() as u64);
        for (i, v) in limited {
            w.u(i as u64);
            w.u(v as u64);
        }
    }
    // T-3233: the pacing the settled rates were settled with, so a load starts settled
    let paced: Vec<(usize, [f32; 3])> = lines.iter().enumerate().map(|(i, &l)| (i, net.line_pace_of(l))).filter(|(_, p)| p.iter().any(|&x| x != 0.0)).collect();
    if net.rates_settled && lines.iter().any(|&l| net.line_trains[l as usize].is_some()) && !paced.is_empty() {
        w.0.extend_from_slice(b"LPC1");
        w.u(paced.len() as u64);
        for (i, p) in paced {
            w.u(i as u64);
            for x in p { w.u(x.to_bits() as u64); }
        }
    }
    // T-3333: lines run on both pairs and their twins
    let index_of = |l: u32| lines.iter().position(|&x| x == l);
    let twins: Vec<(usize, u64, usize)> = lines.iter().enumerate().filter_map(|(i, &l)| match net.line_twin_of(l) {
        Twin::None => None,
        Twin::Parent { split } => Some((i, 1 + split as u64, 0)),
        Twin::Of(p) => index_of(p).map(|j| (i, 3, j)),
    }).collect();
    if !twins.is_empty() {
        w.0.extend_from_slice(b"LTW1");
        w.u(twins.len() as u64);
        for (i, kind, j) in twins {
            w.u(i as u64);
            w.u(kind);
            if kind == 3 { w.u(j as u64); }
        }
    }
    w.0
}

/// T-2981: the way a line's loop runs unless `LOW1` lists it: one way for a bus or ferry line.
fn one_way_default(net: &Network, l: usize) -> bool {
    net.line_bus[l].is_some() || net.line_ferry[l].is_some()
}

/// T-2871: the days of an `LBD1` block, one list per kind; None if a list is not the length this
/// network has, or a day is outside what a build could have written.
fn read_days(r: &mut R, n: [usize; 3]) -> Option<[Vec<u32>; 3]> {
    let mut out: [Vec<u32>; 3] = [vec![], vec![], vec![]];
    for (list, &n) in out.iter_mut().zip(&n) {
        if r.u().ok()? as usize != n {
            return None;
        }
        let mut prev = 0i64;
        for _ in 0..n {
            prev = prev.checked_add(r.i().ok()?)?;
            if !(DAY_UNKNOWN as i64..u32::MAX as i64).contains(&prev) {
                return None;
            }
            list.push(prev as u32);
        }
    }
    Some(out)
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
    out.buildings = net.buildings.iter().map(|b| b.as_ref().map(|b| super::building::Building { name: b.name.clone(), corners: b.corners.iter().map(|p| { let (x, y) = at(p[0], p[1]); [x, y] }).collect() })).collect();
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
            net.set_station(id, Some(StationData { platform, name, named: flags >> 8 & 1 != 0, built: flags & 4 != 0, off: flags & 8 != 0, upgrades: (flags >> 4) as u8, berth: None, day: 0, one: (flags >> 9 & 3) as u8 })).map_err(|e| format!("{e:?}"))?;
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
        unknown += (tb >> 17 != 0) as usize;
        let oneway = (tb >> 11 & 3) as u8;
        let (tracks, built, off, crossover, sep) = ((tb & 3) as u8, tb & 4 != 0, tb & 16 != 0, tb & 32 != 0, tb & 128 != 0);
        let kind = match (tb >> 8 & 7) as u8 {
            0 => TrackKind::from_hsr(tb & 64 != 0),
            k => TrackKind::from_u8(k).unwrap_or_else(|| {
                unknown += 1;
                TrackKind::Regular
            }),
        };
        let n = r.u()? as usize;
        // T-2820: ends and sizes that cannot be are a damaged save, not a panic
        if a < 0 || b < 0 || a as usize >= n_nodes || b as usize >= n_nodes {
            return Err("bad edge".into());
        }
        let (mut px, mut py) = (net.node_x[a as usize], net.node_y[a as usize]);
        let mut pis = Vec::with_capacity(n.min(bytes.len()));
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
        net.set_edge(id, Some(EdgeData { a: a as u32, b: b as u32, tracks, pis, built, off, thru, crossover, kind, sep, day: 0, oneway, join: [(tb >> 13 & 3) as u8, (tb >> 15 & 3) as u8] })).map_err(|e| format!("{e:?}"))?;
    }
    let n_lines = r.u()? as usize;
    let (mut dead_node, mut dead_edge) = (None, None);
    for _ in 0..n_lines {
        let id = net.alloc_line();
        let name = r.s()?;
        let colour = r.u()? as u32;
        let tph = [r.tenth()?, r.tenth()?, r.tenth()?];
        let trains = if r.u()? != 0 { Some([r.u()? as u32, r.u()? as u32, r.u()? as u32]) } else { None };
        let (dwell_s, turnaround_s) = (r.tenth()?, r.tenth()?);
        let cars = r.u()? as u8;
        // T-2820: a broken line keeps what was deleted under it, written as an id past the dense
        // ones. Those point at one dead slot, in range and never reused, as in the game that saved it
        let mut stops = vec![];
        let mut prev = 0i64;
        for _ in 0..r.u()? {
            prev += r.i()?;
            let s = prev as u32;
            stops.push(if (s as usize) < n_nodes { s } else { *dead_node.get_or_insert_with(|| net.alloc_node()) });
        }
        let mut path = vec![];
        let mut prev = 0i64;
        for _ in 0..r.u()? {
            let v = r.i()?;
            prev += v >> 1;
            let e = prev as u32;
            let e = if (e as usize) < n_edges { e } else { *dead_edge.get_or_insert_with(|| net.alloc_edge()) };
            path.push(e << 1 | (v & 1) as u32);
        }
        net.set_line(id, Some(LineData { symbol: String::new(), trains, auto: None, name, colour, stops, path, tph, dwell_s, turnaround_s, cars, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false, one_way: false, limit: 0, twin: Default::default() })).map_err(|e| format!("{e:?}"))?;
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
    if r.0.get(r.1..r.1 + 4) == Some(b"LSP1") {
        r.1 += 4;
        let mut split = vec![[0.5f32; 3]; n_lines];
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let p = [r.u()? as u32, r.u()? as u32, r.u()? as u32].map(f32::from_bits);
            if i < n_lines { split[i] = p; }
        }
        net.line_split = split;
    }
    let fr = r.0.get(r.1..r.1 + 4);
    if fr == Some(b"LFR1") || fr == Some(b"LFR2") {
        let old = fr == Some(b"LFR1");
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
                let mps = r.u()? as f64 / 1000.0;
                v.push(if old { super::ferry::v_from_lfr1(mps) } else { mps });
            }
            if i < n_lines { net.line_ferry[i] = Some(super::ferry::FerryRoute::new(points, kinds, v, built)); }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"BLD1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let name = r.s()?;
            let (mut px, mut py) = (0.0, 0.0);
            let mut corners = vec![];
            for _ in 0..r.u()? {
                let (x, y) = (quant(px + r.mm()?), quant(py + r.mm()?));
                (px, py) = (x, y);
                corners.push([x, y]);
            }
            net.buildings.push(Some(super::building::Building { name, corners }));
        }
    }
    // a line is dated when decoded, before the blocks that make it a bus or ferry line: a planned
    // route has no day yet
    for l in 0..n_lines as u32 {
        if !net.route_built(l) { net.line_day[l as usize] = 0; }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LBD1") {
        r.1 += 4;
        let stations: Vec<usize> = (0..n_nodes).filter(|&n| net.is_station(n as u32) && net.node_built[n]).collect();
        let built_edges: Vec<usize> = (0..n_edges).filter(|&e| net.edge_built[e]).collect();
        let built_lines: Vec<usize> = (0..n_lines).filter(|&l| net.route_built(l as u32)).collect();
        // a damaged block costs only the days, never the game: it is dropped with the bytes after it
        match read_days(&mut r, [stations.len(), built_edges.len(), built_lines.len()]) {
            Some([st, ed, li]) => {
                for (n, d) in stations.into_iter().zip(st) { net.node_day[n] = d; }
                for (e, d) in built_edges.into_iter().zip(ed) { net.edge_day[e] = d; }
                for (l, d) in built_lines.into_iter().zip(li) { net.line_day[l] = d; }
            }
            None => r.1 = bytes.len(),
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LBP1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let (i, m) = (r.u()? as usize, r.u()?);
            if let Some(Some(b)) = net.line_bus.get_mut(i) { b.spacing = super::bus::spacing_from(m as f64); }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LHO1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            if i < n_lines && net.line_bus[i].is_none() && net.line_ferry[i].is_none() { net.line_hand[i] = true; }
        }
    }
    let mut manual = vec![];
    if r.0.get(r.1..r.1 + 4) == Some(b"LMN1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            if i < n_lines && net.line_bus[i].is_none() && net.line_ferry[i].is_none() { manual.push(i as u32); }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LBH1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let mut stops = vec![];
            let (mut px, mut py) = (0.0, 0.0);
            for _ in 0..r.u()? {
                let (x, y) = (quant(px + r.mm()?), quant(py + r.mm()?));
                (px, py) = (x, y);
                stops.push([x, y]);
            }
            if let Some(Some(b)) = net.line_bus.get_mut(i) { b.hand = Some(stops); }
        }
    }
    // T-2981: each line its mode's way round, then the ones the block lists the other way
    for i in 0..n_lines {
        net.line_one_way[i] = one_way_default(&net, i);
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LOW1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            if i < n_lines { net.line_one_way[i] = !net.line_one_way[i]; }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LSL1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let v = r.u()?;
            if i < n_lines { net.line_limit[i] = v.min(u16::MAX as u64) as u16; }
        }
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LPC1") {
        r.1 += 4;
        let mut pace = vec![[0.0f32; 3]; n_lines];
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let p = [r.u()? as u32, r.u()? as u32, r.u()? as u32].map(f32::from_bits);
            if i < n_lines { pace[i] = p; }
        }
        net.line_pace = pace;
        net.pace_from_save = net.rates_from_save;
    }
    if r.0.get(r.1..r.1 + 4) == Some(b"LTW1") {
        r.1 += 4;
        for _ in 0..r.u()? {
            let i = r.u()? as usize;
            let kind = r.u()?;
            let t = match kind {
                1 | 2 => Twin::Parent { split: kind == 2 },
                3 => Twin::Of(r.u()? as u32),
                _ => return Err(format!("LTW1: kind {kind}")),
            };
            if i < n_lines { net.line_twin[i] = t; }
        }
    }
    for l in manual {
        fold_manual(&mut net, l);
    }
    net.take_touch();
    Ok((net, bytes.len() - r.1 + unknown))
}

/// A line saved in manual mode (T-3023) whose stops and path read the same both ways, `[A, B, C, B, A]`,
/// becomes the open line `[A, B, C]` run both ways; any other list stays as it is (T-3109).
fn fold_manual(net: &mut Network, l: u32) {
    let Some(mut d) = net.line_data(l) else { return };
    let (n, m) = (d.stops.len(), d.path.len());
    if n < 3 || n % 2 == 0 || (0..n).any(|i| d.stops[i] != d.stops[n - 1 - i]) || (0..m).any(|i| d.path[i] != d.path[m - 1 - i] ^ 1) {
        return;
    }
    d.stops.truncate(n / 2 + 1);
    d.path.truncate(m / 2);
    d.hand = false;
    d.one_way = false;
    let _ = net.set_line(l, Some(d));
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
            net.set_station(n, Some(StationData { platform: 2, name: format!("S{x}"), named: false, built: true, off: false, upgrades: 0, berth: None, day: 0, one: 0 })).unwrap();
        }
        let e = net.alloc_edge();
        net.set_edge(e, Some(EdgeData { a: 0, b: 1, tracks: 2, pis: vec![], built: true, off: false, thru: vec![], crossover: false, kind: TrackKind::Regular, sep: false, day: 0, oneway: 0, join: [0; 2] })).unwrap();
        let l = net.alloc_line();
        net.set_line(l, Some(LineData { symbol: String::new(), trains: Some([2, 2, 1]), auto: None, name: "A".into(), colour: 0xff0000, stops: vec![0, 1], path: vec![0], tph: [0.0; 3], dwell_s: 30.0, turnaround_s: 120.0, cars: 4, fast: false, bus: None, ferry: None, group: 0, group_name: String::new(), group_look: Default::default(), day: 0, hand: false, one_way: false, limit: 0, twin: Default::default() })).unwrap();
        encode(&net)
    }

    /// T-2820: a line whose track was deleted with no way round keeps its path on the dead edge (a
    /// broken line). The save writes that edge as "none", which reads back as an id past the edges.
    #[test]
    fn a_broken_line_saves_and_loads_again_and_again() {
        let mut net = decode(&sample()).unwrap();
        net.set_edge(0, None).unwrap();
        let first = encode(&net);
        let mut w = crate::track::world::TrackWorld::new(decode(&first).unwrap());
        assert!(w.net.line_ok(0), "the line is kept");
        assert!(w.net.check_line(0).is_err(), "and is broken");
        assert!(w.net.line_path(0).iter().all(|&p| ((p >> 1) as usize) < w.net.edge_count()), "its path stays in range");
        let second = encode(&w.net);
        let w2 = crate::track::world::TrackWorld::new(decode(&second).unwrap());
        assert!(w2.net.line_ok(0) && w2.net.check_line(0).is_err(), "still there and still broken on the next load");
        encode(&w2.net);
    }

    /// T-3109: a line saved in manual mode as a plain out-and-back loads as its open line.
    #[test]
    fn a_manual_out_and_back_folds_on_load() {
        let mut net = decode(&sample()).unwrap();
        let mut d = net.line_data(0).unwrap();
        d.stops = vec![0, 1, 0];
        d.path = vec![0, 1];
        d.hand = true;
        d.one_way = true;
        net.set_line(0, Some(d)).unwrap();
        fold_manual(&mut net, 0);
        let d = net.line_data(0).unwrap();
        assert_eq!((d.stops, d.path, d.hand, d.one_way), (vec![0, 1], vec![0], false, false));
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
    fn edge_kind_bits_round_trip_and_old_flags_still_read() {
        for k in TrackKind::ALL {
            let mut net = decode(&sample()).unwrap();
            let mut d = net.edge_data(0).unwrap();
            d.kind = k;
            d.tracks = 2;
            net.set_edge(0, Some(d)).unwrap();
            let (back, unread) = decode_read(&encode(&net)).unwrap();
            assert_eq!(unread, 0);
            assert_eq!(back.edge_kind[0], k);
        }
        assert_eq!(edge_kind_bits(TrackKind::Regular), 0);
        assert_eq!(edge_kind_bits(TrackKind::HighSpeed), 0, "high speed keeps bit 6, as old builds wrote it");
        assert_eq!(edge_kind_bits(TrackKind::StreetTram), 2 << 8);
    }

    /// An old save has high speed in bit 6 and nothing in bits 8-10.
    #[test]
    fn bit_six_alone_still_means_high_speed() {
        let mut net = decode(&sample()).unwrap();
        let mut d = net.edge_data(0).unwrap();
        d.kind = TrackKind::HighSpeed;
        net.set_edge(0, Some(d)).unwrap();
        let bytes = encode(&net);
        assert_eq!(decode(&bytes).unwrap().edge_kind[0], TrackKind::HighSpeed);
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
        assert_eq!(decode_read(&raw(0, 1 | 1 << 11)).unwrap().1, 0, "one way, a to b");
        assert_eq!(decode_read(&raw(0, 1 | 1 << 13 | 2 << 15)).unwrap().1, 0, "joins (T-3137)");
        assert_eq!(decode_read(&raw(0, 1 | 1 << 17)).unwrap().1, 1);
        assert_eq!(decode_read(&raw(0, 1 | 7 << 8)).unwrap().1, 1, "a kind this build lacks");
        assert_eq!(decode_read(&raw(0, 2 | 3 << 8)).unwrap().1, 0, "a known kind");
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
        assert_eq!((LEVEL, BLOCKS.len()), (18, 24), "a new block raises LEVEL: raise it, then update this pin");
    }

    /// T-2217: buildings come back as they were, in their order; a network without them encodes as
    /// it did before the block existed (it ends where `sample` always ended).
    #[test]
    fn buildings_round_trip_and_leave_a_network_without_them_as_it_was() {
        let bytes = sample();
        let mut net = decode(&bytes).unwrap();
        assert!(net.buildings.is_empty());
        assert_eq!(encode(&net), bytes, "no buildings: not a byte more");
        let hall = |name: &str, x: f64| super::super::building::Building { name: name.into(), corners: vec![[x, -40.0], [x + 300.0, -40.0], [x + 300.0, 40.5], [quant(x + 0.001), 40.0]] };
        net.buildings = vec![Some(hall("West", -150.0)), None, Some(hall("East ✓", 850.0))];
        let with = encode(&net);
        assert_eq!(&with[..bytes.len()], &bytes[..], "the block only follows what was there");
        assert_eq!(&with[bytes.len()..bytes.len() + 4], b"BLD1");
        let (back, unread) = decode_read(&with).unwrap();
        assert_eq!(unread, 0);
        assert_eq!(back.buildings, vec![Some(hall("West", -150.0)), Some(hall("East ✓", 850.0))], "renumbered densely");
        assert_eq!(encode(&back), with);
    }
}
