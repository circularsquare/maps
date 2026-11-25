//! T-4199: a market whose corridors and book come from the corridor file gives every corridor figure the
//! market working them out gives, to the bit; a file of another world, or an unreadable one, is not used.

use super::t2216::zone_island_packs;
use super::tests::{rows_json, world_packs, RULE};
use super::*;

fn world(rows: &str) -> String {
    format!(r#"{{"frame":{{"kind":"equirect","lon":0,"lat":50}},"rule":{RULE},"places":5,"rows":{rows},"attractions":[]}}"#)
}
fn packs_of(world: &str, file: Option<&[u8]>, held: &[String]) -> MarketPacks {
    let mut m = MarketPacks::new(world).unwrap();
    if let Some(f) = file {
        m.set_corridors(f);
    }
    for p in held {
        m.add(p).unwrap();
    }
    m
}
fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn corridors_from_the_file_are_those_worked_out() {
    // (place 0's second zone on an island of its own)
    let packs = zone_island_packs();
    let w = world(&rows_json(&packs));
    let file = packs_of(&w, None, &[]).corridor_file().unwrap();
    // every set of packs held, some with a zone on another land
    let mut crossed = 0;
    for held in [&packs[1..2], &packs[..1], &packs[..2], &packs[..]] {
        let (plain, read) = (packs_of(&w, None, held), packs_of(&w, Some(&file), held));
        let (a, b) = (plain.market().unwrap(), read.market().unwrap());
        assert_eq!(read.corridors_state(), FILE_USED);
        assert!(b.m.cw.table.get().is_none(), "no table made from the file");
        let np = a.m.places.len();
        let (ca, cb) = (&a.m.cw, &b.m.cw);
        assert_eq!((ca.daily.to_bits(), ca.leaving.to_bits(), ca.place_pairs, ca.commuters.to_bits()), (cb.daily.to_bits(), cb.leaving.to_bits(), cb.place_pairs, cb.commuters.to_bits()));
        assert_eq!(bits(&ca.leaving_by), bits(&cb.leaving_by));
        assert_eq!(ca.commute.iter().map(|e| (e.0, e.1.to_bits())).collect::<Vec<_>>(), cb.commute.iter().map(|e| (e.0, e.1.to_bits())).collect::<Vec<_>>());
        // (with zones on two lands, the file's book and the rows whose modes they change)
        crossed += b.m.apart().cross.len();
        assert_eq!(bits(&a.table_summary(np as u32)), bits(&b.table_summary(np as u32)), "the book, {} packs", held.len());
        assert!(b.m.cw.table.get().is_none(), "the book made without the table");
        let all: Vec<u32> = (0..np as u32).collect();
        assert_eq!(bits(&a.place_rows(&all)), bits(&b.place_rows(&all)));
        let cs: Vec<u32> = (0..a.m.n_corridors() as u32).collect();
        assert_eq!(bits(&a.rows_at(&cs)), bits(&b.rows_at(&cs)));
        assert!(b.m.cw.table.get().is_none(), "rows answered without the table");
        // asked for every corridor, the table is made, the same
        assert_eq!(bits(&a.corridor_trips()), bits(&b.corridor_trips()));
        assert_eq!(a.corridor_rows(), b.corridor_rows());
        assert_eq!(bits(&a.elsewhere(0)), bits(&b.elsewhere(0)));
    }
    assert!(crossed > 0, "no corridor between zones on two lands");
}

#[test]
fn a_file_of_another_world_or_unreadable_is_not_used() {
    let packs = world_packs();
    let w = world(&rows_json(&packs));
    let file = packs_of(&w, None, &[]).corridor_file().unwrap();
    // the attractions are not part of the key
    let other_attractions = w.replace(r#""attractions":[]"#, r#""attractions":[[0.1,50.0,1000000.0]]"#);
    let m = packs_of(&other_attractions, Some(&file), &packs[..1]);
    m.market().unwrap();
    assert_eq!(m.corridors_state(), FILE_USED);
    // another rule
    let other = w.replace(r#""min_trips":0.01"#, r#""min_trips":0.02"#);
    assert_ne!(other, w);
    let m = packs_of(&other, Some(&file), &packs[..1]);
    let a = m.market().unwrap();
    assert_eq!(m.corridors_state(), FILE_OTHER_WORLD);
    assert!(m.held.file.borrow().is_none(), "a refused file is not kept");
    assert!(a.m.cw.book.is_none() && a.m.cw.table.get().is_some());
    let mut bad = file.clone();
    bad.truncate(bad.len() - 1);
    let m = packs_of(&w, Some(&bad), &packs[..1]);
    assert_eq!(m.corridors_state(), FILE_UNREADABLE);
    assert!(m.market().unwrap().m.cw.book.is_none());
    // a flipped bit in the body, whatever it is in
    let mut flipped = file.clone();
    let k = flipped.len() - 16;
    flipped[k] ^= 1;
    let m = packs_of(&w, Some(&flipped), &packs[..1]);
    assert_eq!(m.corridors_state(), FILE_UNREADABLE);
    // counts that would wrap a 32-bit length: refused, never a panic
    for at in [32, 36] {
        let mut wrapped = file.clone();
        wrapped[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(CorridorFile::parse(&wrapped).is_err());
        wrapped[at..at + 4].copy_from_slice(&0x2aaa_aaabu32.to_le_bytes());
        assert!(CorridorFile::parse(&wrapped).is_err());
    }
    // a sample whose trips differ (the checksum made again, as a file made wrongly would have it)
    let mut off = file.clone();
    let k = off.len() - 16;
    off[k] ^= 1;
    let sum = body_sum(&off);
    off[68..72].copy_from_slice(&sum.to_le_bytes());
    let m = packs_of(&w, Some(&off), &packs[..1]);
    m.market().unwrap();
    assert_eq!(m.corridors_state(), FILE_DIFFERS);
    assert!(m.held.file.borrow().is_none());
}

#[test]
fn a_file_of_other_drives_or_another_build_is_not_used() {
    let packs = world_packs();
    let w = world(&rows_json(&packs));
    // made with no drives table
    let file = packs_of(&w, None, &[]).corridor_file().unwrap();
    // (a table with no pairs: "TWD1", no places, no ferries; the second has the groups' count too)
    let empty = |extra: bool| {
        let mut b = b"TWD1".to_vec();
        b.extend([0u8; 8]);
        if extra {
            b.extend([0u8; 4]);
        }
        b
    };
    let mut m = MarketPacks::new(&w).unwrap();
    m.set_drives(&empty(false)).unwrap();
    m.set_corridors(&file);
    m.add(&packs[0]).unwrap();
    m.market().unwrap();
    assert_eq!(m.corridors_state(), FILE_OTHER_WORLD, "a file made with no drives, read with some");
    assert!(m.held.file.borrow().is_none());
    // made with a drives table, read with another, and with none
    let mut maker = MarketPacks::new(&w).unwrap();
    maker.set_drives(&empty(false)).unwrap();
    let file = maker.corridor_file().unwrap();
    for read in [Some(empty(true)), None] {
        let mut m = MarketPacks::new(&w).unwrap();
        if let Some(d) = &read {
            m.set_drives(d).unwrap();
        }
        m.set_corridors(&file);
        m.add(&packs[0]).unwrap();
        m.market().unwrap();
        assert_eq!(m.corridors_state(), FILE_OTHER_WORLD);
    }
    let mut same = MarketPacks::new(&w).unwrap();
    same.set_drives(&empty(false)).unwrap();
    same.set_corridors(&file);
    same.add(&packs[0]).unwrap();
    same.market().unwrap();
    assert_eq!(same.corridors_state(), FILE_USED);
    // a file made by another build of the sim (its header's source key; the body is the same)
    let mut other = file.clone();
    other[24] ^= 1;
    let m = packs_of(&w, Some(&other), &packs[..1]);
    assert_eq!(m.corridors_state(), FILE_OTHER_BUILD);
    assert!(m.held.file.borrow().is_none());
    m.market().unwrap();
    assert_eq!(m.corridors_state(), FILE_OTHER_BUILD);
}
