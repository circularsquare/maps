use super::*;

/// What a site table needs to be built into the market: one input per place.
pub(super) struct Built {
    pub(super) zones: Vec<Rc<ZoneIn>>,
    pub(super) rule: Rule,
    pub(super) out: Option<PackFrame>,
    pub(super) places: usize,
    pub(super) rows: Option<Vec<Row>>,
    pub(super) world: bool,
    pub(super) closed: bool,
    /// legacy: each pack site's place and index in its place's input (commutes name pack sites)
    pub(super) legacy_commutes: Vec<(u32, u32, f32)>,
    pub(super) drives: Rc<Drives>,
    pub(super) attractions: Vec<[f64; 3]>,
}
pub(super) fn single_pack(pack: LegacyPack) -> Result<Built, String> {
    let zm = pack.zone_m;
    let markets = pack.markets.len();
    if markets == 0 || !zm.is_finite() || zm <= 0. || !pack.trip_rate.is_finite() || pack.trip_rate < 0. {
        return Err("invalid market parameters".into());
    }
    if let Some(f) = &pack.frame {
        make_frame(f)?;
    }
    if !pack.site_park.is_empty() && (pack.site_park.len() != pack.sites.len() || pack.site_park.iter().any(|p| !p.is_finite() || *p < 0.)) {
        return Err("invalid site parking".into());
    }
    let mut zones: Vec<ZoneRaw> = (0..markets).map(|m| ZoneRaw { place: m, frame: pack.frame.clone(), zone_m: zm, inc: 1., ..Default::default() }).collect();
    let mut keys: Vec<HashMap<(i32, i32), usize>> = vec![HashMap::new(); markets];
    let mut at = Vec::with_capacity(pack.sites.len());
    for (si, s) in pack.sites.iter().enumerate() {
        if s.iter().any(|x| !x.is_finite()) || s[2] < 0. || s[3] < 0. || s[4] < 0. || s[4].fract() != 0. || s[4] >= markets as f64 {
            return Err("invalid intercity weights".into());
        }
        let m = s[4] as usize;
        let k = ((s[0] as f32 / zm as f32).floor() as i32, (s[1] as f32 / zm as f32).floor() as i32);
        let zi = &mut zones[m];
        let n = zi.zones.len();
        let z = *keys[m].entry(k).or_insert(n);
        if z == n {
            zi.zones.push([k.0, k.1]);
        }
        at.push((m as u32, zi.sites.len() as u32));
        zi.sites.push([s[0], s[1], s[2], s[3], z as f64]);
        if let Some(&p) = pack.site_park.get(si) {
            zi.site_park.push(p);
        }
    }
    let nc = pack.commute_trips.len();
    if pack.commute_home.len() != nc || pack.commute_work.len() != nc {
        return Err("commute arrays disagree".into());
    }
    let mut legacy_commutes = Vec::with_capacity(nc);
    for i in 0..nc {
        let (h, w, n) = (pack.commute_home[i] as usize, pack.commute_work[i] as usize, pack.commute_trips[i]);
        if h >= at.len() || w >= at.len() || !n.is_finite() || n < 0. {
            return Err("invalid commute between zones".into());
        }
        legacy_commutes.push((h as u32, w as u32, n));
    }
    // the commutes name pack sites: keep each site's place and place-local index for them
    let site_at = at;
    let zones = zones.into_iter().map(|z| Rc::new(ZoneIn::from(z))).collect();
    let mut built = Built { zones, rule: Rule::closed(pack.trip_rate), out: pack.frame.clone(), places: markets, rows: None, world: false, closed: true, legacy_commutes: vec![], drives: Rc::default(), attractions: vec![] };
    built.legacy_commutes = legacy_commutes
        .into_iter()
        .map(|(h, w, n)| {
            let (hp, hs) = site_at[h as usize];
            let (wp, ws) = site_at[w as usize];
            // packed: place in the top 8 bits is enough for the old packs (at most 255 places)
            ((hp << 24) | hs, (wp << 24) | ws, n)
        })
        .collect();
    Ok(built)
}
pub(super) fn compose(c: ComposeIn) -> Result<Built, String> {
    if !c.world.rule.ok() {
        return Err("invalid market rule".into());
    }
    make_frame(&c.world.frame)?;
    Ok(Built { zones: c.zones.into_iter().map(Rc::new).collect(), rule: c.world.rule, out: Some(c.world.frame), places: c.world.places, rows: c.world.rows, world: true, closed: false, legacy_commutes: vec![], drives: Rc::default(), attractions: c.world.attractions })
}

pub(super) fn build(b: Built) -> Result<Market, String> {
    build_for(b, None, None)
}
/// `build`, with the near totals added up only for pairs of places with an end in `near_for`
/// (indexed by place), and only those places' commutes (T-929: the rows of those places alone).
/// `reuse`: the corridors' figures of an earlier composition of the same world with rows.
pub(super) fn build_for(b: Built, near_for: Option<&[bool]>, reuse: Option<Rc<Corridors>>) -> Result<Market, String> {
    let Built { mut zones, rule, out, places: np, rows, world, closed, legacy_commutes, drives, attractions } = b;
    let out_frame = match &out {
        Some(f) => Some(make_frame(f)?),
        None => None,
    };
    let ground = out_frame.is_some();
    zones.sort_by_key(|z| z.place);
    if zones.windows(2).any(|w| w[0].place == w[1].place) || zones.iter().any(|z| z.place >= np) {
        return Err("market zones: a place twice or out of range".into());
    }
    if let Some(rows) = &rows {
        if rows.iter().any(|r| r.place >= np) {
            return Err("market row out of range".into());
        }
    }
    let mut countries: Vec<String> = vec![];
    let mut country_id = |s: &str| -> u32 {
        match countries.iter().position(|c| c == s) {
            Some(i) => i as u32,
            None => {
                countries.push(s.to_owned());
                countries.len() as u32 - 1
            }
        }
    };
    let mut places: Vec<Place> = vec![Place::default(); np];
    // global zone offsets: the rows' zone counts in place order, or (no rows) the packs'
    let counts: Vec<u32> = match &rows {
        Some(rows) => {
            let mut c = vec![0u32; np];
            for r in rows {
                c[r.place] = r.zones;
            }
            c
        }
        None => {
            let mut c = vec![0u32; np];
            for z in &zones {
                c[z.place] = z.zones.len() as u32;
            }
            c
        }
    };
    let mut off = 0u32;
    for (p, pl) in places.iter_mut().enumerate() {
        pl.off = off;
        pl.zones = counts[p];
        off += counts[p];
    }
    let n_gid = off as usize;
    let mut gid_place = vec![0u32; n_gid];
    for (p, pl) in places.iter().enumerate() {
        for g in pl.off..pl.off + pl.zones {
            gid_place[g as usize] = p as u32;
        }
    }
    // ---- sites and zones, grouped by frame
    let mut mz: Vec<Zone> = vec![];
    let mut cell_zone: Vec<usize> = vec![];
    let mut groups: Vec<Group> = vec![];
    let mut site_park: Vec<f32> = vec![];
    let mut site_u: Vec<f64> = vec![];
    let mut site_mult: Vec<(f64, f64)> = vec![];
    let mut pack_first_zone: Vec<usize> = vec![0; np];
    let mut pack_first_site: Vec<usize> = vec![0; np];
    // sites are laid out group by group, so one group's are contiguous
    let mut order: Vec<usize> = (0..zones.len()).collect();
    let frame_of = |z: &ZoneIn| -> Result<Option<Frame>, String> {
        match &z.frame {
            Some(f) => Ok(Some(make_frame(f)?)),
            None if !world => Ok(None),
            None => Err("a market zone without a frame".into()),
        }
    };
    let frames: Vec<Option<Frame>> = zones.iter().map(|z| frame_of(z)).collect::<Result<_, _>>()?;
    // one group per place: its access and its rides' 2 km zones depend on its own sites alone, so
    // a place's figures are the same whichever other packs are open (T-377)
    let gkeys: Vec<Option<Frame>> = frames.clone();
    let gidx: Vec<usize> = (0..zones.len()).collect();
    order.sort_by_key(|&i| zones[i].place);
    let mut group_cities: Vec<City> = gkeys.iter().map(|_| City { name: String::new(), h3: vec![], x: vec![], y: vec![], pop: vec![], jobs: vec![], region: vec![], gravity: None }).collect();
    let mut group_lo = vec![usize::MAX; gkeys.len()];
    for &zi_ in &order {
        let zin = &zones[zi_];
        let g = gidx[zi_];
        let gc = &mut group_cities[g];
        let place = zin.place;
        let nz = zin.zones.len();
        if !zin.zone_m.is_finite() || zin.zone_m <= 0. || !zin.inc.is_finite() || zin.inc <= 0. {
            return Err("invalid market zone parameters".into());
        }
        if world && (zin.zone_ng.len() != nz || zin.zone_nt.len() != nz || zin.zone_ng.iter().chain(&zin.zone_nt).any(|v| !v.is_finite() || *v < 0.)) {
            return Err(format!("place {place}: normalisers do not match its zones"));
        }
        if !zin.site_park.is_empty() && zin.site_park.len() != zin.sites.len() {
            return Err("invalid site parking".into());
        }
        if zin.cell_pop.len() != 0 && zin.cell_pop.len() != zin.sites.len() {
            return Err("invalid cell people".into());
        }
        if counts[place] as usize != nz {
            return Err(format!("place {place}: {nz} zones in its pack, {} in its row (stale registry row: run pipeline/market_zones.py)", counts[place]));
        }
        if group_lo[g] == usize::MAX {
            group_lo[g] = cell_zone.len();
        } else if group_lo[g] + gc.len() != cell_zone.len() {
            return Err("market groups out of order".into());
        }
        let first = mz.len();
        pack_first_zone[place] = first;
        pack_first_site[place] = cell_zone.len();
        let f = frames[zi_];
        for k in 0..nz {
            mz.push(Zone { x: 0., y: 0., u: [0.; 3], pop: 0., w_attr: 0., jobs: 0., g: 0., t: 0., a: 0., ta: 0., place, park: 0., gid: places[place].off + k as u32, dep: 0., tour: 0. });
        }
        let s_inc = zin.inc.min(1.).powf(rule.eps);
        let total_pop: f64 = zin.sites.pop.iter().sum();
        let total_attr: f64 = zin.sites.pop.iter().zip(zin.sites.jobs.iter()).map(|(p, j)| p + 2. * j).sum();
        // T-377: a pack's attraction rescaled so its total is its people (times income), as the
        // world field's cells count people
        let k_attr = if closed || total_attr <= 0. { 1. } else { total_pop / total_attr * s_inc };
        let mut park_w = vec![0f64; nz];
        let mut park = vec![0f64; nz];
        for si in 0..zin.sites.len() {
            let (x, y, pop, jobs, z) = zin.sites.at(si);
            if !x.is_finite() || !y.is_finite() || pop < 0. || jobs < 0. || z as usize >= nz {
                return Err("invalid intercity weights".into());
            }
            let z = z as usize;
            let attr = pop + 2. * jobs;
            let zz = &mut mz[first + z];
            let w = pop + attr;
            zz.x += x * w as f32;
            zz.y += y * w as f32;
            zz.pop += pop;
            zz.w_attr += attr;
            zz.jobs += jobs;
            zz.a += attr * k_attr;
            let u = (zin.cell_pop.at(si).unwrap_or(0.).max(1.) / rule.u_ref_people).max(1e-12);
            zz.ta += attr * k_attr * u.powf(rule.th);
            site_u.push(u);
            let (sg, st) = (s_inc, s_inc * u.powf(rule.tho) / rule.u_mean_us);
            site_mult.push((sg, st));
            zz.dep += pop * (rule.r * sg + rule.rt * st);
            if let Some(&p) = zin.site_park.get(si) {
                if !p.is_finite() || p < 0. {
                    return Err("invalid site parking".into());
                }
                let jw = jobs + 1e-9;
                park[z] += p as f64 * jw;
                park_w[z] += jw;
            }
            site_park.push(zin.site_park.get(si).copied().unwrap_or(0.));
            gc.x.push(x);
            gc.y.push(y);
            gc.pop.push(pop as f32);
            gc.jobs.push(attr as f32);
            cell_zone.push(first + z);
        }
        for k in 0..nz {
            let z = &mut mz[first + k];
            let w = (z.pop + z.w_attr).max(1e-12) as f32;
            z.x /= w;
            z.y /= w;
            if park_w[k] > 0. {
                z.park = (park[k] / park_w[k]) as f32;
            }
            // the centre on the ground (an equirectangular frame is affine in lon/lat, so the
            // plane's weighted centre is the sites' weighted lon/lat), and in the output frame
            if let Some(f) = &f {
                let (lon, lat) = f.to_lonlat(z.x as f64, z.y as f64);
                z.u = unit(lon, lat);
                if let Some(o) = &out_frame {
                    let (ox, oy) = o.from_lonlat(lon, lat);
                    z.x = ox as f32;
                    z.y = oy as f32;
                }
            }
        }
        // departures over the normalisers (a closed world's come after every zone is known)
        if !closed {
            let lo = pack_first_site[place];
            for si in 0..zin.sites.len() {
                let (_, _, pop, _, zk) = zin.sites.at(si);
                let z = first + zk as usize;
                let (sg, st) = site_mult[lo + si];
                let (ng, nt) = (zin.zone_ng[zk as usize], zin.zone_nt[zk as usize]);
                let zz = &mut mz[z];
                if ng > 0. {
                    zz.g += pop * rule.r * sg / ng;
                }
                if nt > 0. {
                    zz.t += pop * rule.rt * st / nt;
                }
            }
        }
        // T-378: the drive from each zone's centre to each airport, people-weighted
        if !zin.air_direct.is_empty() && zin.air_direct.len() != zin.airports.len() || zin.airports.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("place {place}: invalid airports"));
        }
        let zs = &mz[first..first + nz];
        let w: f64 = zs.iter().map(|z| z.pop).sum();
        let air: Vec<(f64, Vec<u32>)> = if !ground {
            vec![]
        } else {
            zin.airports
                .iter()
                .enumerate()
                .map(|(k, a)| {
                    let u = unit(a[0], a[1]);
                    let s: f64 = zs.iter().map(|z| z.pop * drive_min(chord_km(&z.u, &u) as f32) as f64).sum();
                    (if w > 0. { s / w } else { drive_min(0.) as f64 }, sorted(zin.air_direct.get(k).cloned().unwrap_or_default()))
                })
                .collect()
        };
        let aw: f64 = zs.iter().map(|z| z.w_attr).sum();
        let park = if aw > 0. { zs.iter().map(|z| z.park as f64 * z.w_attr).sum::<f64>() / aw } else { 0. };
        let pl = &mut places[place];
        pl.open = true;
        pl.country = country_id(&zin.country);
        pl.inc = zin.inc.min(1.);
        pl.road = zin.road;
        pl.zone = zin.zone.clone();
        pl.air = air;
        pl.park = park;
    }
    for (g, c) in group_cities.into_iter().enumerate() {
        let (kz, kplace) = ride_zones(&c);
        groups.push(Group { frame: gkeys[g], lo: group_lo[g], city: c, kz, kplace });
    }
    groups.sort_by_key(|g| g.lo);
    // ---- a closed world: each zone's normaliser over the others (the old pack's market)
    let ground_km = |a: &Zone, b: &Zone| if ground { chord_km(&a.u, &b.u) } else { ((a.x - b.x).hypot(a.y - b.y) / 1000.) as f64 };
    if closed {
        let mut n = vec![0f64; mz.len()];
        for a in 0..mz.len() {
            for b in a + 1..mz.len() {
                let f = rule.ground(ground_km(&mz[a], &mz[b]));
                if f > 0. {
                    n[a] += mz[b].a * f;
                    n[b] += mz[a].a * f;
                }
            }
        }
        for (z, zz) in mz.iter_mut().enumerate() {
            zz.g = if n[z] > 0. { zz.pop * rule.r / n[z] } else { 0. };
        }
    }
    // ---- the places' rows: the world's, or added up here
    let mut near: HashMap<(u32, u32), f64> = HashMap::new();
    let mut commute_rows: Vec<Vec<(u32, f64)>> = vec![vec![]; np];
    let from_rows = rows.is_some();
    match rows {
        Some(rows) => {
            for r in rows {
                if [r.lon, r.lat, r.inc, r.g, r.t, r.a, r.ta, r.pop, r.dep].iter().any(|v| !v.is_finite()) || r.inc <= 0. {
                    return Err("invalid market row".into());
                }
                let pl = &mut places[r.place];
                if pl.open && (pl.country != country_id(&r.country) || pl.road != r.road) {
                    return Err(format!("place {}: its pack and its row disagree (run pipeline/market_zones.py)", r.place));
                }
                pl.has = true;
                pl.lon = r.lon;
                pl.lat = r.lat;
                pl.u = unit(r.lon, r.lat);
                pl.country = country_id(&r.country);
                pl.inc = r.inc.min(1.);
                pl.road = r.road;
                pl.g = r.g;
                pl.t = r.t;
                pl.a = r.a;
                pl.ta = r.ta;
                pl.pop = r.pop;
                pl.dep = r.dep;
                pl.zone = r.zone.clone();
                pl.air = r.air.iter().filter(|a| a.0.is_finite() && a.0 >= 0.).map(|a| (a.0, sorted(a.1.clone()))).collect();
                pl.park = if r.park.is_finite() { r.park.max(0.) } else { 0. };
                pl.fw = if r.fw.is_finite() && r.fw > 0. { r.fw } else { 0. };
                pl.cu = if r.cu.is_finite() && r.cu > 0. { r.cu } else { 0. };
                (pl.gr, pl.ng) = income_groups(&r.fg)?;
                for &(j, t) in &r.near {
                    if (j as usize) < np && t.is_finite() && t >= 0. {
                        near.insert((r.place.min(j as usize) as u32, r.place.max(j as usize) as u32), t);
                    }
                }
                commute_rows[r.place] = r.commute.clone();
            }
        }
        None => {
            for z in &mz {
                let pl = &mut places[z.place];
                pl.has = true;
                pl.g += z.g;
                pl.t += z.t;
                pl.a += z.a;
                pl.ta += z.ta;
                pl.pop += z.pop;
            }
            for (p, pl) in places.iter_mut().enumerate() {
                if !pl.has {
                    continue;
                }
                let zs = &mz[if pl.zones > 0 { pack_first_zone[p] } else { 0 }..if pl.zones > 0 { pack_first_zone[p] + pl.zones as usize } else { 0 }];
                let w: f64 = zs.iter().map(|z| z.pop + z.w_attr).sum::<f64>().max(1e-12);
                if ground {
                    let mut u = [0f64; 3];
                    for z in zs {
                        for k in 0..3 {
                            u[k] += z.u[k] * (z.pop + z.w_attr) / w;
                        }
                    }
                    let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-12);
                    pl.u = [u[0] / n, u[1] / n, u[2] / n];
                    (pl.lon, pl.lat) = lonlat_of(&pl.u);
                } else {
                    pl.x = (zs.iter().map(|z| z.x as f64 * (z.pop + z.w_attr)).sum::<f64>() / w) as f32;
                    pl.y = (zs.iter().map(|z| z.y as f64 * (z.pop + z.w_attr)).sum::<f64>() / w) as f32;
                }
            }
            // departures: each site's ground and tail rate
            for (s, &z) in cell_zone.iter().enumerate() {
                let (sg, st) = site_mult[s];
                let gr = &groups[match groups.binary_search_by(|g| g.lo.cmp(&s)) {
                    Ok(i) => i,
                    Err(i) => i - 1,
                }];
                let pop = gr.city.pop[s - gr.lo] as f64;
                places[mz[z].place].dep += pop * (rule.r * sg + rule.rt * st);
            }
        }
    }
    for pl in places.iter_mut() {
        if pl.has {
            if let Some(o) = &out_frame {
                let (x, y) = o.from_lonlat(pl.lon, pl.lat);
                pl.x = x as f32;
                pl.y = y as f32;
            }
        }
    }
    // ---- per pair of places with an open one (T-1204: every pair was 300 MB a worker)
    let mp = market_params();
    let mut rules = Rules { slot: vec![u32::MAX; np], ..Rules::default() };
    for (p, pl) in places.iter().enumerate() {
        if pl.open {
            rules.slot[p] = rules.open.len() as u32;
            rules.open.push(p);
        }
    }
    rules.out.reserve_exact(rules.open.len() * rules.open.len());
    for &i in &rules.open {
        rules.out.extend(rules.open.iter().map(|&j| pair_rule(&places, &countries, &rule, ground, &mp, &drives, i, j)));
    }
    let grouped = places.iter().any(|p| p.groups().iter().any(|g| g.1 != 1.));
    let mut m = Market {
        grouped,
        rule,
        ground,
        world,
        from_rows,
        anchor: groups.first().and_then(|g| g.frame),
        places,
        countries,
        rules,
        mp,
        drives,
        near,
        zones: mz,
        n_gid,
        gid_place,
        comp_of_gid: vec![u32::MAX; n_gid],
        groups,
        cell_zone,
        site_key: SiteKeys::default(),
        commutes: vec![],
        c_home: (vec![], vec![]),
        c_work: (vec![], vec![]),
        everyday: vec![],
        e_home: (vec![], vec![]),
        e_act: (vec![], vec![]),
        e_base: vec![],
        cw: Rc::new(Corridors { id: 0, trips_at: vec![], trips: vec![], commute: vec![], commuters: 0., leaving_by: vec![], daily: 0., leaving: 0., place_pairs: 0, rows: Default::default() }),
    };
    for (c, z) in m.zones.iter().enumerate() {
        m.comp_of_gid[z.gid as usize] = c as u32;
    }
    // ---- near totals added up from the zones (no rows: the pipeline, a closed world)
    if !from_rows {
        // near places' zone pairs only (T-929): each pair of places sums its zone pairs (a, b > a)
        // in the order a scan of every zone pair would, so the totals are the same to the bit
        let spans = m.place_spans();
        for (k, &(p, lo, hi)) in spans.iter().enumerate() {
            for &(q, qlo, qhi) in &spans[k..] {
                if !m.pr(p, q).near || near_for.is_some_and(|f| !f[p] && !f[q]) {
                    continue;
                }
                let mut sum = None;
                for a in lo..hi {
                    for b in qlo.max(a + 1)..qhi {
                        let t = m.eachway(&m.zones[a], &m.zones[b]);
                        if t >= m.rule.min_trips {
                            *sum.get_or_insert(0f64) += t;
                        }
                    }
                }
                if let Some(t) = sum {
                    m.near.insert((p as u32, q as u32), t);
                }
            }
        }
    }
    // ---- commutes
    let mut by_pair: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    if closed {
        for &(h, w, n) in &legacy_commutes {
            let site = |v: u32| pack_first_site[(v >> 24) as usize] + (v & 0xff_ffff) as usize;
            let (zh, zw) = (m.cell_zone[site(h)], m.cell_zone[site(w)]);
            if zh != zw {
                *by_pair.entry((zh as u32, m.zones[zw].gid)).or_default() += n as f64;
            }
        }
    } else {
        for zin in zones.iter().filter(|z| near_for.is_none_or(|f| f[z.place])) {
            let nc = zin.commute_trips.len();
            if zin.commute_home.len() != nc || zin.commute_place.len() != nc || zin.commute_work.len() != nc {
                return Err("commute arrays disagree".into());
            }
            let first = pack_first_zone[zin.place];
            for i in 0..nc {
                let (h, wp, w, n) = (zin.commute_home.at(i) as usize, zin.commute_place.at(i) as usize, zin.commute_work.at(i), zin.commute_trips[i]);
                if h >= zin.zones.len() || wp >= np || w >= m.places[wp].zones || !n.is_finite() || n < 0. {
                    return Err("invalid commute between zones".into());
                }
                let wg = m.places[wp].off + w;
                if m.zones[first + h].gid != wg {
                    *by_pair.entry(((first + h) as u32, wg)).or_default() += n as f64;
                }
            }
        }
    }
    m.commutes = by_pair.into_iter().map(|((h, w), n)| (h, w, n as f32)).collect();
    // T-1175: everyday outings across a zone line, from the open places' own files
    let mut by_kind: BTreeMap<(u32, u32, u8), (f64, f64)> = BTreeMap::new();
    for zin in zones.iter().filter(|z| !closed && near_for.is_none_or(|f| f[z.place])) {
        let ne = zin.everyday_trips.len();
        if [zin.everyday_home.len(), zin.everyday_place.len(), zin.everyday_zone.len(), zin.everyday_kind.len(), zin.everyday_walk.len()].iter().any(|&l| l != ne) {
            return Err("everyday arrays disagree".into());
        }
        let first = pack_first_zone[zin.place];
        for i in 0..ne {
            let (h, ap, a, k, n, w) = (zin.everyday_home.at(i) as usize, zin.everyday_place.at(i) as usize, zin.everyday_zone.at(i), zin.everyday_kind[i], zin.everyday_trips[i], zin.everyday_walk[i]);
            if h >= zin.zones.len() || ap >= np || a >= m.places[ap].zones || k as usize >= super::super::everyday::KIND_COUNT || !n.is_finite() || n < 0. || !(0.0..=1.0).contains(&w) {
                return Err("invalid everyday pair between zones".into());
            }
            let ag = m.places[ap].off + a;
            if m.zones[first + h].gid != ag {
                let e = by_kind.entry(((first + h) as u32, ag, k)).or_default();
                e.0 += n as f64;
                e.1 += (n * w) as f64;
            }
        }
    }
    m.everyday = by_kind.into_iter().map(|((h, a, kind), (n, w))| EPair { h, a, kind, trips: n as f32, walk: (w / n.max(1e-12)) as f32 }).collect();
    let csr = |key: &dyn Fn(&(u32, u32, f32)) -> Option<u32>, n: usize, list: &[(u32, u32, f32)]| {
        let mut off = vec![0u32; n + 1];
        for c in list {
            if let Some(k) = key(c) {
                off[k as usize + 1] += 1;
            }
        }
        for z in 0..n {
            off[z + 1] += off[z];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; off[n] as usize];
        for (i, c) in list.iter().enumerate() {
            if let Some(k) = key(c) {
                items[fill[k as usize] as usize] = i as u32;
                fill[k as usize] += 1;
            }
        }
        (off, items)
    };
    let nz = m.zones.len();
    m.c_home = csr(&|c| Some(c.0), nz, &m.commutes);
    let cog = m.comp_of_gid.clone();
    m.c_work = csr(&|c| Some(cog[c.1 as usize]).filter(|&w| w != u32::MAX), nz, &m.commutes);
    // T-1229: the everyday pairs by home zone and by activity zone, and per home place and kind
    let csr_e = |key: &dyn Fn(&EPair) -> Option<u32>| {
        let mut off = vec![0u32; nz + 1];
        for e in &m.everyday {
            if let Some(k) = key(e) {
                off[k as usize + 1] += 1;
            }
        }
        for z in 0..nz {
            off[z + 1] += off[z];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; off[nz] as usize];
        for (i, e) in m.everyday.iter().enumerate() {
            if let Some(k) = key(e) {
                items[fill[k as usize] as usize] = i as u32;
                fill[k as usize] += 1;
            }
        }
        (off, items)
    };
    let (e_home, e_act) = (csr_e(&|e| Some(e.h)), csr_e(&|e| Some(cog[e.a as usize]).filter(|&w| w != u32::MAX)));
    m.e_home = e_home;
    m.e_act = e_act;
    m.e_base = vec![(0., 0.); np * KIND_COUNT];
    for e in &m.everyday {
        let b = &mut m.e_base[m.zones[e.h as usize].place * KIND_COUNT + e.kind as usize];
        b.0 += e.trips as f64;
        b.1 += (e.trips * e.walk) as f64;
    }
    // ---- the corridors' figures: with rows the same for every composition, so kept (T-1204)
    m.cw = match reuse {
        Some(c) if from_rows => c,
        _ => Rc::new(m.corridors_of(from_rows.then_some(&commute_rows[..]))),
    };
    // ---- each 600 m square's site
    let site_key = SiteKeys::of((0..m.n_sites()).map(|s| m.site_key_of(s)));
    m.site_key = site_key;
    m.set_tourism(&attractions);
    Ok(m)
}
/// T-377: the packs of the places a worker has open, parsed once, and the world they belong to;
/// `market` composes them into a market (the worker does so whenever a pack opens or closes).
#[wasm_bindgen]
pub struct MarketPacks {
    pub(super) world: WorldIn,
    pub(super) zones: BTreeMap<usize, Rc<ZoneIn>>,
    /// T-1204, T-1220: the corridors' figures, worked out at the first composition and kept for
    /// the others (with rows they depend on the places alone)
    pub(super) corridors: RefCell<Option<Rc<Corridors>>>,
    pub(super) drives: Rc<Drives>,
}
#[wasm_bindgen]
impl MarketPacks {
    /// `world`: the registry's market frame, rule, place count and rows.
    #[wasm_bindgen(constructor)]
    pub fn new(world: &str) -> Result<MarketPacks, String> {
        let world: WorldIn = serde_json::from_str(world).map_err(|e| e.to_string())?;
        if !world.rule.ok() {
            return Err("invalid market rule".into());
        }
        make_frame(&world.frame)?;
        Ok(MarketPacks { world, zones: BTreeMap::new(), corridors: RefCell::new(None), drives: Rc::default() })
    }
    /// T-279: the routed drives between places (`market-drives.bin`, pipeline/market_drives.py),
    /// for the compositions after; the pairs placed. Without them every drive is the rule's.
    pub fn set_drives(&mut self, bytes: &[u8]) -> Result<u32, String> {
        let rows = self.world.rows.as_deref().unwrap_or(&[]);
        let place_of: HashMap<&str, usize> = rows.iter().map(|r| (r.zone.as_str(), r.place)).collect();
        let d = Drives::decode(bytes, &place_of, self.world.places)?;
        let n = d.len() as u32;
        self.drives = Rc::new(d);
        Ok(n)
    }
    /// Add (or replace) a place's pack; its place.
    pub fn add(&mut self, pack: &str) -> Result<u32, String> {
        let z: ZoneIn = serde_json::from_str(pack).map_err(|e| e.to_string())?;
        if z.place >= self.world.places {
            return Err("market pack's place out of range".into());
        }
        let p = z.place;
        self.zones.insert(p, Rc::new(z));
        Ok(p as u32)
    }
    pub fn remove(&mut self, place: u32) -> bool {
        self.zones.remove(&(place as usize)).is_some()
    }
    pub fn places(&self) -> Vec<u32> {
        self.zones.keys().map(|&p| p as u32).collect()
    }
    /// T-1174: drop the corridors kept from the first composition, so the next composition works
    /// them out from the packs held now, as a worker that opened them all at once does (they differ
    /// with the packs open at the first composition: T-1354).
    pub fn forget_corridors(&self) {
        *self.corridors.borrow_mut() = None;
    }
    /// The market of every pack held, with no network yet.
    pub fn market(&self) -> Result<IntercityApi, String> {
        let w = self.world.clone();
        let rows = w.rows.is_some();
        let b = Built { zones: self.zones.values().cloned().collect(), rule: w.rule, out: Some(w.frame), places: w.places, rows: w.rows, world: true, closed: false, legacy_commutes: vec![], drives: self.drives.clone(), attractions: w.attractions };
        let m = build_for(b, None, self.corridors.borrow().clone())?;
        if rows {
            *self.corridors.borrow_mut() = Some(m.cw.clone());
        }
        Ok(IntercityApi::on(Rc::new(m)))
    }
}
pub(super) fn parse(header: &str) -> Result<Market, String> {
    build(parse_built(header)?)
}
pub(super) fn parse_built(header: &str) -> Result<Built, String> {
    // a composition as bin/market_pack.rs writes it, straight into its struct (no Value between)
    if header.starts_with("{\"world\":") {
        return compose(serde_json::from_str(header).map_err(|e| e.to_string())?);
    }
    let v: serde_json::Value = serde_json::from_str(header).map_err(|e| e.to_string())?;
    if v.get("sites").is_some() {
        single_pack(serde_json::from_value(v).map_err(|e| e.to_string())?)
    } else {
        compose(serde_json::from_value(v).map_err(|e| e.to_string())?)
    }
}
/// T-276: the market's zones as the game built them from an old single pack's sites, for the
/// joint commute gravity (`bin/joint_gravity.rs`): each site's zone, the first site of each zone,
/// and each site's 600 m square key (x, y / 600, floored; the market).
pub fn market_site_zones(header: &str) -> Result<(Vec<u32>, Vec<u32>, Vec<(i32, i32, usize)>), String> {
    let pack: LegacyPack = serde_json::from_str(header).map_err(|e| e.to_string())?;
    let zm = pack.zone_m as f32;
    let mut keys = BTreeMap::new();
    let (mut site_zone, mut first, mut sq) = (vec![], vec![], vec![]);
    for (s, nums) in pack.sites.iter().enumerate() {
        let (x, y, m) = (nums[0] as f32, nums[1] as f32, nums[4] as usize);
        let n = keys.len();
        let z = *keys.entry(((x / zm).floor() as i32, (y / zm).floor() as i32, m)).or_insert(n);
        if z == first.len() {
            first.push(s as u32);
        }
        site_zone.push(z as u32);
        sq.push(((x / SITE_M).floor() as i32, (y / SITE_M).floor() as i32, m));
    }
    Ok((site_zone, first, sq))
}
/// T-377: the registry rows of every place a composition holds (the world given without rows):
/// what `bin/market_pack.rs` writes into the registry.
pub fn world_rows(header: &str) -> Result<Vec<Row>, String> {
    world_rows_for(header, None)
}
/// T-929: `world_rows` of the places in `only` (by place) alone. Each is the same to the bit as in
/// a composition of every place, given the composition holds every place under `near_km` of it and
/// every place it commutes to (a missing commute place is an error; a missing near one is not).
pub fn world_rows_for(header: &str, only: Option<&[bool]>) -> Result<Vec<Row>, String> {
    let b = parse_built(header)?;
    if only.is_some_and(|f| f.len() != b.places) {
        return Err("rows wanted for a place list of another length".into());
    }
    let m = build_for(b, only, None)?;
    if m.from_rows {
        return Err("the composition carries rows already".into());
    }
    let np = m.places.len();
    let mut commutes: Vec<BTreeMap<u32, f64>> = vec![BTreeMap::new(); np];
    for &(h, w, n) in &m.commutes {
        *commutes[m.zones[h as usize].place].entry(m.gid_place[w as usize]).or_default() += n as f64;
    }
    let mut out = vec![];
    for (p, commute) in commutes.into_iter().enumerate() {
        let pl = &m.places[p];
        if !pl.open || only.is_some_and(|f| !f[p]) {
            continue;
        }
        let near: Vec<(u32, f64)> = (0..np).filter(|&j| m.places[j].has && m.pr(p, j).near).filter_map(|j| m.near.get(&(p.min(j) as u32, p.max(j) as u32)).map(|&t| (j as u32, t))).collect();
        out.push(Row {
            place: p,
            zone: pl.zone.clone(),
            lon: pl.lon,
            lat: pl.lat,
            country: m.countries[pl.country as usize].clone(),
            inc: pl.inc,
            road: pl.road,
            g: pl.g,
            t: pl.t,
            a: pl.a,
            ta: pl.ta,
            pop: pl.pop,
            dep: pl.dep,
            zones: pl.zones,
            near,
            commute: commute.into_iter().collect(),
            air: pl.air.clone(),
            park: pl.park,
            fw: 0.,
            cu: 0.,
            fg: vec![],
        });
    }
    Ok(out)
}
/// T-377: trips a day each way between two places of a composition (rows given or not), and
/// that pair's trips over `min_km` (zone centre to zone centre; T-299's comparison), both from
/// the zones when both places are open.
pub fn place_trips(header: &str, pairs: &[(usize, usize)], min_km: f64) -> Result<Vec<(f64, f64)>, String> {
    Ok(parse(header)?.place_trips(pairs, min_km))
}
impl Market {
    pub(super) fn place_trips(&self, pairs: &[(usize, usize)], min_km: f64) -> Vec<(f64, f64)> {
        let spans = self.place_spans();
        let span = |p: usize| spans.iter().find(|s| s.0 == p).map_or(0..0, |s| s.1..s.2);
        pairs
            .iter()
            .map(|&(i, j)| {
                let all = self.place_eachway(i, j);
                let mut over = 0.;
                if self.places[i].open && self.places[j].open {
                    for a in span(i) {
                        for b in span(j) {
                            let (za, zb) = (&self.zones[a], &self.zones[b]);
                            if (i == j && b <= a) || self.zone_km(za, zb) < min_km {
                                continue;
                            }
                            let t = self.eachway(za, zb);
                            if t >= self.rule.min_trips {
                                over += t;
                            }
                        }
                    }
                }
                (all, over)
            })
            .collect()
    }
}
