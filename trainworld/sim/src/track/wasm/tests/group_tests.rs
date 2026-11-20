use super::*;

#[test]
fn line_groups_join_copy_leave_and_save() {
    let mut a = api();
    a.set_cash(1e6);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let l4 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l4, &[n0, n1], 4, 4, 2, 30.0, 180.0, 0, 0x00933c, "4".into()), 0);
    let l5 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l5, &[n0, n1], 4, 4, 2, 30.0, 180.0, 0, 0x00933c, "5".into()), 0);
    assert_eq!(a.construct_line(l4), 0, "{}", a.issues());
    let times = a.stop_times(l4, 0, 0);
    // a join: no service edit
    let g = a.new_line_group();
    assert_eq!(a.set_line_group(&[l4, l5], g), 0, "{}", a.issues());
    assert!(a.dirty_demand_lines().is_empty());
    assert_eq!((a.line_group(l4), a.line_group(l5)), (g, g));
    assert_eq!(a.stop_times(l4, 0, 0), times);
    assert_eq!(a.undo(), 0);
    assert_eq!((a.line_group(l4), a.line_group(l5)), (0, 0));
    assert_eq!(a.redo(), 0);
    assert_eq!(a.line_group(l5), g);
    // a name, kept by a line that joins
    assert_eq!(a.set_line_group_name(g, "Lex".into()), 0);
    assert_eq!(a.line_group_name(l5), "Lex");
    let c = a.new_line_id();
    assert_eq!(a.copy_line(l4, c, "4 copy".into()), 0, "{}", a.issues());
    assert_eq!((a.line_group(c), a.line_group_name(c)), (g, "Lex".to_string()));
    // leaving: the copy goes, then the 5 leaves the 4 alone, which stops being a group
    assert_eq!(a.set_line_group(&[c], 0), 0);
    assert_eq!((a.line_group(c), a.line_group_name(c), a.line_group(l4)), (0, String::new(), g));
    assert_eq!(a.set_line_group(&[l5], 0), 0);
    assert_eq!((a.line_group(l4), a.line_group(l5)), (0, 0));
    assert_eq!(a.undo(), 0, "one step");
    assert_eq!((a.line_group(l4), a.line_group(l5)), (g, g));
    // a copy of an ungrouped line makes a group of the two
    let c2 = a.new_line_id();
    assert_eq!(a.copy_line(c, c2, "4 copy 2".into()), 0, "{}", a.issues());
    let g2 = a.line_group(c);
    assert!(g2 != 0 && g2 != g && a.line_group(c2) == g2);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_group(c), 0, "the copy's grouping goes with it");
    assert_eq!(a.redo(), 0);
    // deleting one of two ends the group
    assert_eq!(a.remove_line(c2), 0);
    assert!(!a.w.net.line_ok(c2), "the deleted line is gone");
    assert_eq!(a.line_group(c), 0);
    // saves keep groups and names; a save without the block loads ungrouped
    let bytes = a.save();
    let at = bytes.windows(4).rposition(|w| w == b"LGR1").expect("LGR1 written");
    let mut b = api();
    assert!(b.load(&bytes));
    let (b4, b5) = (0u32, 1u32);
    assert_eq!((b.line_group(b4), b.line_group(b5), b.line_group_name(b5)), (g, g, "Lex".to_string()));
    assert!(save::decode(&bytes[..at]).map(|n| n.line_group.iter().all(|&g| g == 0)).unwrap());
    assert_eq!(a.set_line_group(&[l4], 0), 0);
    assert!(!a.save().windows(4).any(|w| w == b"LGR1"));
}

#[test]
fn group_looks_and_hand_order_save() {
    let mut a = api();
    a.set_cash(1e6);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let ls: Vec<u32> = (0..3).map(|k| {
        let l = a.new_line_id();
        assert_eq!(a.set_line_with_trains(l, &[n0, n1], 4, 4, 2, 30.0, 180.0, 0, 0x00933c, format!("{k}")), 0);
        l
    }).collect();
    let g = a.new_line_group();
    assert_eq!(a.set_line_group(&ls[..2], g), 0);
    // a look set by hand, undone in one step, taken by a line that joins, dropped by one leaving
    assert_eq!(a.set_line_group_look(g, "Lex".into(), 0xff0000), 0);
    assert_eq!((a.line_group_symbol(ls[1]), a.line_group_colour(ls[1])), ("Lex".to_string(), 0xff0000));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.line_group_colour(ls[0]), -1);
    assert_eq!(a.redo(), 0);
    assert_eq!(a.set_line_group(&[ls[2]], g), 0);
    assert_eq!(a.line_group_symbol(ls[2]), "Lex");
    assert_eq!(a.set_line_group(&[ls[2]], 0), 0);
    assert_eq!((a.line_group_symbol(ls[2]), a.line_group_colour(ls[2])), (String::new(), -1));
    // the hand order, outside the undo history
    a.set_list_order(0, &[ls[2], ls[0], ls[1]]);
    a.set_list_order(1, &[n1, n0]);
    assert_eq!(a.list_order(0), vec![ls[2], ls[0], ls[1]]);
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!((b.line_group_symbol(1), b.line_group_colour(0)), ("Lex".to_string(), 0xff0000));
    assert_eq!(b.list_order(0), vec![2, 0, 1]);
    let st = b.list_order(1);
    assert_eq!(st.len(), 2);
    assert!(b.node_name(st[0]) == "East" && b.node_name(st[1]) == "West", "{st:?}");
    a.set_list_order(0, &[]);
    a.set_list_order(1, &[]);
    assert!(!a.save().windows(4).any(|w| w == b"LOR1"));
    assert!(a.list_order(0).is_empty());
}

/// T-3423: groups inside groups, three deep: names and bullets on every line below, no group in
/// itself, a group left with one member giving it to the group around it, and saves.
#[test]
fn nested_line_groups() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let ls: Vec<u32> = (0..5).map(|k| {
        let l = a.new_line_id();
        assert_eq!(a.set_line_with_trains(l, &[n0, n1], 4, 4, 2, 30.0, 180.0, 0, 0x00933c, format!("{k}")), 0);
        l
    }).collect();
    let (la, lb, lc, ld, le) = (ls[0], ls[1], ls[2], ls[3], ls[4]);
    let chain = |a: &TrackApi, l: u32| { let mut c = vec![a.line_group(l)]; c.extend(a.line_group_outer(l)); c.retain(|&g| g != 0); c };
    let g1 = a.new_line_group();
    assert_eq!(a.set_line_group(&[la, lb], g1), 0);
    let g2 = a.new_line_group();
    assert_eq!(a.set_group_members(&[lc], &[g1], g2, 0), 0, "{}", a.issues());
    let g3 = a.new_line_group();
    assert!(g3 > g2 && g2 > g1);
    assert_eq!(a.set_group_members(&[ld], &[g2], g3, 0), 0, "{}", a.issues());
    assert_eq!((chain(&a, la), chain(&a, lc), chain(&a, ld), chain(&a, le)), (vec![g1, g2, g3], vec![g2, g3], vec![g3], vec![]));
    assert!(a.dirty_demand_lines().is_empty());
    // a new group inside another, from two of its three members; from both of two, the one around
    // it keeps one member and goes
    assert_eq!(a.set_line_group(&[le], g2), 0);
    let g4 = a.new_line_group();
    assert_eq!(a.set_group_members(&[lc], &[g1], g4, g2), 0, "{}", a.issues());
    assert_eq!((chain(&a, la), chain(&a, lc), chain(&a, le)), (vec![g1, g4, g2, g3], vec![g4, g2, g3], vec![g2, g3]));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!(a.set_group_members(&[lc], &[g1], g4, g2), 0, "{}", a.issues());
    assert_eq!((chain(&a, la), chain(&a, lc)), (vec![g1, g4, g3], vec![g4, g3]));
    assert_eq!(a.undo(), 0);
    assert_eq!((chain(&a, la), chain(&a, lc)), (vec![g1, g2, g3], vec![g2, g3]));
    // the outer group's name and bullet are on every line in it, at any depth
    assert_eq!(a.set_line_group_name(g3, "Top".into()), 0);
    assert_eq!(a.set_line_group_look(g2, "Mid".into(), 0xff0000), 0);
    assert_eq!((a.line_group_name(ld), a.line_group_outer_name(la, 1), a.line_group_name(lc)), ("Top".to_string(), "Top".to_string(), String::new()));
    assert_eq!((a.line_group_outer_symbol(la, 0), a.line_group_outer_colour(la, 0), a.line_group_symbol(lc)), ("Mid".to_string(), 0xff0000, "Mid".to_string()));
    // no group in itself or in one inside it
    assert_ne!(a.set_group_members(&[], &[g3], g1, 0), 0);
    assert_ne!(a.set_group_members(&[], &[g2], g2, 0), 0);
    assert_eq!(chain(&a, la), vec![g1, g2, g3]);
    // a line joining a nested group takes the whole chain
    assert_eq!(a.set_line_group(&[le], g1), 0);
    assert_eq!((chain(&a, le), a.line_group_outer_name(le, 1)), (vec![g1, g2, g3], "Top".to_string()));
    assert_eq!(a.undo(), 0);
    // saves keep the chains; a save cut before LGN1 keeps the groups the lines are straight in
    let bytes = a.save();
    let at = bytes.windows(4).rposition(|w| w == b"LGN1").expect("LGN1 written");
    let mut b = api();
    assert!(b.load(&bytes));
    let (ba, bc) = (0u32, 2u32);
    assert_eq!((chain(&b, ba), chain(&b, bc), b.line_group_outer_name(ba, 1), b.line_group_outer_symbol(ba, 0)), (vec![g1, g2, g3], vec![g2, g3], "Top".to_string(), "Mid".to_string()));
    assert_eq!(b.save(), bytes);
    let flat = save::decode(&bytes[..at]).unwrap();
    assert!(flat.line_group_outer.iter().all(|o| o.is_empty()) && flat.line_group[0] == g1);
    // the middle group left with one member stops being one, and that member moves up a level
    assert_eq!(a.set_line_group(&[lc], 0), 0);
    assert_eq!((chain(&a, la), chain(&a, lc), chain(&a, ld)), (vec![g1, g3], vec![], vec![g3]));
    assert_eq!(a.line_group_outer_name(la, 0), "Top");
    assert_eq!(a.undo(), 0, "one step");
    assert_eq!((chain(&a, la), chain(&a, lc)), (vec![g1, g2, g3], vec![g2, g3]));
    // so does deleting a line: the top group keeps only the middle one, so it goes
    assert_eq!(a.remove_line(ld), 0);
    assert!(!a.w.net.line_ok(ld), "the deleted line is gone");
    assert_eq!((chain(&a, la), chain(&a, lc)), (vec![g1, g2], vec![g2]));
    assert_eq!(a.undo(), 0);
    // a group out to the top: the one it leaves keeps one line and stops being a group
    assert_eq!(a.set_group_members(&[], &[g1], 0, 0), 0);
    assert_eq!((chain(&a, la), chain(&a, lc), chain(&a, ld)), (vec![g1], vec![g3], vec![g3]), "g2 kept only c, which goes up to g3");
    assert!(!a.save().windows(4).any(|w| w == b"LGN1"), "nothing nested: no block");
}
