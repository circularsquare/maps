//! T-1442: tidy names that came from a real network. Station names arrive in capitals with a
//! "STATION" suffix ("PORT IMPERIAL HBLR STATION") and lines as "<feed> <route>" ("subway 6X").
//! Used where a real network becomes a save (`bin/realsave.rs`) and on every load, for saves made
//! before; a save records no "imported" flag, so a name is tidied only when it has the untidy
//! shape exactly: a station name with a part in capitals (a word of five letters or more, or
//! two words), a line name that is a known feed's key, a space and the route, with no symbol.
//! Anything else, a player's own names included, is left alone, so tidying twice changes nothing.

use super::net::Network;

/// Capitals that stay capitals: initialisms the no-vowel rule would not keep.
const ACRONYMS: [&str; 24] = [
    "PATH", "NJT", "LIRR", "MNR", "HBLR", "JFK", "LGA", "EWR", "MSU", "NJPAC", "NYU", "NYC", "NJ", "NY", "DC", "US", "USA", "UN", "LIC", "SUNY", "CUNY", "MTA", "BWI", "SIR",
];
/// Abbreviations written as a capital and lower case: "ST" is "St", "JCT" "Jct".
const ABBREVIATIONS: [&str; 16] = ["ST", "MT", "FT", "DR", "RD", "JCT", "BLVD", "PKWY", "HTS", "SQ", "PK", "AVE", "AV", "PL", "HWY", "CTR"];
/// Small words, lower case away from the start.
const SMALL: [&str; 4] = ["OF", "THE", "AND", "AT"];
/// The `feed` keys of `pipeline/realnet.py`'s sources, the first word of an imported line's name.
const FEEDS: [&str; 15] = ["subway", "path", "lirr", "mnr", "njt", "tfl", "nr", "idfm", "osm", "seo", "ttc", "go", "up", "s12", "n02"];
/// Joins the names a complex's stations had.
const JOIN: &str = " / ";

/// A line's name without its feed ("subway 6X" is "6X"), or None where it is not one.
pub fn tidy_line(name: &str, symbol: &str) -> Option<String> {
    if !symbol.is_empty() {
        return None;
    }
    let (feed, rest) = name.split_once(' ')?;
    let rest = rest.trim();
    (FEEDS.contains(&feed) && !rest.is_empty()).then(|| rest.to_string())
}

/// A station name in title case without "STATION", or None where it is not in capitals.
/// "PORT IMPERIAL HBLR STATION" is "Port Imperial HBLR"; in a complex's joined name only the
/// parts in capitals change, and a part another one repeats ("Penn Station" beside "NEW YORK PENN
/// STATION") goes.
pub fn tidy_station(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split(JOIN).collect();
    if !parts.iter().any(|p| shouting(p)) {
        return None;
    }
    let shown: Vec<String> = parts.iter().map(|p| if shouting(p) { title_part(p) } else { p.trim().to_string() }).collect();
    let words: Vec<Vec<String>> = shown.iter().map(|s| key_words(s)).collect();
    let keep: Vec<&String> = shown
        .iter()
        .enumerate()
        .filter(|&(i, _)| {
            !(0..shown.len()).any(|j| {
                j != i && words[j].len() >= words[i].len() && contains(&words[j], &words[i]) && (words[j].len() > words[i].len() || j < i)
            })
        })
        .map(|(_, s)| s)
        .collect();
    let out = keep.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(JOIN);
    (out != name && !out.is_empty()).then_some(out)
}

/// Whether a part is in capitals and long enough to be an import's: no lower-case letter, and
/// two words or one of five letters. "MSU" and "UCLA" stay.
fn shouting(part: &str) -> bool {
    let mut upper = 0;
    for c in part.chars() {
        if c.is_lowercase() {
            return false;
        }
        if c.is_uppercase() {
            upper += 1;
        }
    }
    if upper == 0 {
        return false;
    }
    let words = part.split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty());
    let (mut n, mut longest) = (0, 0);
    for w in words {
        n += 1;
        longest = longest.max(w.chars().count());
    }
    n >= 2 || longest >= 5
}

fn key_words(s: &str) -> Vec<String> {
    let mut w: Vec<String> = s.to_lowercase().split_whitespace().map(|x| x.to_string()).collect();
    if w.len() > 1 && w.last().map(|l| l == "station") == Some(true) {
        w.pop();
    }
    w
}

fn contains(big: &[String], small: &[String]) -> bool {
    !small.is_empty() && big.windows(small.len()).any(|w| w == small)
}

fn title_part(part: &str) -> String {
    let mut p = part.trim();
    for tail in [" STATION", " STA"] {
        if let Some(rest) = p.strip_suffix(tail) {
            if !rest.trim().is_empty() {
                p = rest.trim_end();
                break;
            }
        }
    }
    let chars: Vec<char> = p.chars().collect();
    let (mut out, mut i, mut first) = (String::new(), 0, true);
    while i < chars.len() {
        if !chars[i].is_alphanumeric() {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && (chars[j].is_alphanumeric() || (matches!(chars[j], '\'' | '\u{2019}') && j + 1 < chars.len() && chars[j + 1].is_alphanumeric())) {
            j += 1;
        }
        let w: String = chars[i..j].iter().collect();
        out.push_str(&word(&w, first));
        if w.chars().any(|c| c.is_alphabetic()) {
            first = false;
        }
        i = j;
    }
    out
}

fn word(w: &str, first: bool) -> String {
    if w.chars().any(|c| c.is_ascii_digit()) {
        // 22ND is 22nd; a code with digits in it stays
        let tail = w.trim_start_matches(|c: char| c.is_ascii_digit());
        let n = w.len() - tail.len();
        if n > 0 && matches!(tail, "ST" | "ND" | "RD" | "TH") {
            return format!("{}{}", &w[..n], tail.to_lowercase());
        }
        return w.to_string();
    }
    if ABBREVIATIONS.contains(&w) {
        return title(w);
    }
    if ACRONYMS.contains(&w) || (w.chars().count() <= 4 && !w.chars().any(|c| "AEIOUY".contains(c))) {
        return w.to_string();
    }
    if !first && SMALL.contains(&w) {
        return w.to_lowercase();
    }
    if let Some(rest) = w.strip_prefix("MC") {
        if rest.chars().count() >= 3 {
            return format!("Mc{}", title(rest));
        }
    }
    // O'NEIL is O'Neil, KING'S is King's
    let mut out = String::new();
    for (k, seg) in w.split(|c| c == '\'' || c == '\u{2019}').enumerate() {
        if k > 0 {
            out.push(if w.contains('\u{2019}') { '\u{2019}' } else { '\'' });
        }
        if k > 0 && seg.chars().count() == 1 {
            out.push_str(&seg.to_lowercase());
        } else {
            out.push_str(&title(seg));
        }
    }
    out
}

fn title(w: &str) -> String {
    let mut c = w.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
        None => String::new(),
    }
}

/// Tidy every station and line name of a network in place; the counts changed.
pub fn tidy_network(net: &mut Network) -> (usize, usize) {
    let (mut stations, mut lines) = (0, 0);
    for i in 0..net.node_name.len() {
        if net.node_name[i].is_empty() {
            continue;
        }
        if let Some(n) = tidy_station(&net.node_name[i]) {
            net.node_name[i] = n;
            stations += 1;
        }
    }
    for l in 0..net.line_name.len() {
        if !net.line_alive.get(l).copied().unwrap_or(false) {
            continue;
        }
        let symbol = net.line_symbol.get(l).map_or("", |s| s.as_str());
        if let Some(n) = tidy_line(&net.line_name[l], symbol) {
            net.line_name[l] = n;
            lines += 1;
        }
    }
    (stations, lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(s: &str) -> String {
        tidy_station(s).unwrap_or_else(|| s.to_string())
    }

    #[test]
    fn stations_in_capitals() {
        assert_eq!(st("PORT IMPERIAL HBLR STATION"), "Port Imperial HBLR");
        assert_eq!(st("EDISON STATION"), "Edison");
        assert_eq!(st("NEWARK AIRPORT RAILROAD STATION"), "Newark Airport Railroad");
        assert_eq!(st("22ND STREET LIGHT RAIL STATION"), "22nd Street Light Rail");
        assert_eq!(st("NJPAC/CENTRE STREET LIGHT RAIL STATION"), "NJPAC/Centre Street Light Rail");
        assert_eq!(st("MLK DRIVE LIGHT RAIL STATION"), "MLK Drive Light Rail");
        assert_eq!(st("LIBERTY STATE PARK-RIDE LIGHT RAIL STA"), "Liberty State Park-Ride Light Rail");
        assert_eq!(st("WAYNE/ROUTE 23 TRANSIT CENTER [RR]"), "Wayne/Route 23 Transit Center [RR]");
        assert_eq!(st("JERSEY AVE."), "Jersey Ave.");
        assert_eq!(st("PRINCETON JCT."), "Princeton Jct.");
        assert_eq!(st("MIDDLETOWN NJ"), "Middletown NJ");
        assert_eq!(st("ABERDEEN-MATAWAN"), "Aberdeen-Matawan");
        assert_eq!(st("HOBOKEN"), "Hoboken");
        assert_eq!(st("MCCARTER HWY"), "McCarter Hwy");
        assert_eq!(st("KING'S POINT"), "King's Point");
        assert_eq!(st("PORT OF SPAIN"), "Port of Spain");
    }

    #[test]
    fn joined_names() {
        assert_eq!(st("Penn Station / NEW YORK PENN STATION"), "New York Penn");
        assert_eq!(st("Newark / NEWARK PENN STATION"), "Newark Penn");
        assert_eq!(
            st("Newark / NEWARK PENN STATION / PENN STATION LIGHT RAIL DEPARTURE"),
            "Newark Penn / Penn Station Light Rail Departure"
        );
    }

    #[test]
    fn names_left_alone() {
        for n in ["Battersea Power Station", "Seoul Station", "34 St-Penn Station", "Times Sq-42 St/42 St-Port Authority Bus Terminal", "Abbey Wood / Abbey Wood", "MSU", "UCLA", "Central", "仙台駅", "", "Port Imperial HBLR"] {
            assert_eq!(tidy_station(n), None, "{n}");
        }
    }

    #[test]
    fn station_tidy_is_idempotent() {
        for n in ["PORT IMPERIAL HBLR STATION", "Penn Station / NEW YORK PENN STATION", "WAYNE/ROUTE 23 TRANSIT CENTER [RR]", "22ND STREET LIGHT RAIL STATION"] {
            let once = st(n);
            assert_eq!(tidy_station(&once), None, "{n}");
        }
    }

    #[test]
    fn lines_lose_their_feed() {
        assert_eq!(tidy_line("subway 6X", "").as_deref(), Some("6X"));
        assert_eq!(tidy_line("path PATH", "").as_deref(), Some("PATH"));
        assert_eq!(tidy_line("lirr Babylon Branch", "").as_deref(), Some("Babylon Branch"));
        assert_eq!(tidy_line("s12 東海道本線", "").as_deref(), Some("東海道本線"));
        assert_eq!(tidy_line("subway 6X", "X"), None);
        assert_eq!(tidy_line("6X", ""), None);
        assert_eq!(tidy_line("Red line", ""), None);
        assert_eq!(tidy_line("subway", ""), None);
        assert_eq!(tidy_line("Line 7", ""), None);
    }
}
