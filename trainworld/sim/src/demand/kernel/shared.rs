//! T-3160: read-only arrays held once per thread for every holder with the same numbers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;
use std::thread::LocalKey;

/// An array that is never written in place, held once per thread for every holder whose numbers
/// have the same bits. Each population on a demand worker builds its own graph for each period, and
/// most of those arrays come out the same: a city's commutes and everyday layers have the same
/// inputs, and the stops, edges and fares agree across periods and cities. Made from a `Vec`
/// (`from`, `collect`), which finds an equal array already held; a change makes a new one (`edit`),
/// which finds its own equal, so populations crowded alike share again. Equal means every element's
/// bits, so a holder reads exactly the numbers it made.
pub struct Shared<T: Bits> {
    rc: Rc<[T]>,
    /// its bucket in the thread's registry; 0 when not held there (an empty array, or one made while
    /// the thread ends)
    key: u64,
}

/// The element types a `Shared` holds: their bits, and their thread's registry.
pub trait Bits: Copy + PartialEq + 'static {
    fn bits(self) -> u32;
    fn registry() -> &'static LocalKey<RefCell<Registry<Self>>>;
}

/// The arrays held on one thread, by the hash of their bits.
pub struct Registry<T>(HashMap<u64, Vec<Rc<[T]>>>);

thread_local! {
    static U32S: RefCell<Registry<u32>> = RefCell::new(Registry(HashMap::new()));
    static F32S: RefCell<Registry<f32>> = RefCell::new(Registry(HashMap::new()));
}

impl Bits for u32 {
    #[inline]
    fn bits(self) -> u32 {
        self
    }
    fn registry() -> &'static LocalKey<RefCell<Registry<u32>>> {
        &U32S
    }
}

impl Bits for f32 {
    #[inline]
    fn bits(self) -> u32 {
        self.to_bits()
    }
    fn registry() -> &'static LocalKey<RefCell<Registry<f32>>> {
        &F32S
    }
}

fn key_of<T: Bits>(v: &[T]) -> u64 {
    let mut h = v.len() as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for x in v {
        h = (h.rotate_left(5) ^ x.bits() as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    // (never 0, which marks an array not held)
    h | 1
}

fn same_bits<T: Bits>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.bits() == y.bits())
}

impl<T: Bits> From<Vec<T>> for Shared<T> {
    fn from(mut v: Vec<T>) -> Shared<T> {
        if v.is_empty() {
            return Shared { rc: Rc::from(v), key: 0 };
        }
        let key = key_of(&v);
        let held = T::registry().try_with(|cell| {
            let mut r = cell.borrow_mut();
            let bucket = r.0.entry(key).or_default();
            match bucket.iter().find(|x| same_bits(x, &v)) {
                Some(x) => x.clone(),
                None => {
                    let rc: Rc<[T]> = Rc::from(std::mem::take(&mut v));
                    bucket.push(rc.clone());
                    rc
                }
            }
        });
        match held {
            Ok(rc) => Shared { rc, key },
            Err(_) => Shared { rc: Rc::from(v), key: 0 },
        }
    }
}

impl<T: Bits> FromIterator<T> for Shared<T> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Shared<T> {
        it.into_iter().collect::<Vec<T>>().into()
    }
}

impl<T: Bits> Drop for Shared<T> {
    fn drop(&mut self) {
        // this holder and the registry: the last holder is going, and the registry's copy with it
        if self.key == 0 || Rc::strong_count(&self.rc) != 2 {
            return;
        }
        let _ = T::registry().try_with(|cell| {
            let Ok(mut r) = cell.try_borrow_mut() else { return };
            if let Some(bucket) = r.0.get_mut(&self.key) {
                bucket.retain(|x| !Rc::ptr_eq(x, &self.rc));
                if bucket.is_empty() {
                    r.0.remove(&self.key);
                }
            }
        });
    }
}

impl<T: Bits> Clone for Shared<T> {
    fn clone(&self) -> Shared<T> {
        Shared { rc: self.rc.clone(), key: self.key }
    }
}

impl<T: Bits> Default for Shared<T> {
    fn default() -> Shared<T> {
        Vec::new().into()
    }
}

impl<T: Bits> Deref for Shared<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        &self.rc
    }
}

/// As two `Vec`s compare (element by element, so 0.0 equals -0.0).
impl<T: Bits> PartialEq for Shared<T> {
    fn eq(&self, o: &Shared<T>) -> bool {
        Rc::ptr_eq(&self.rc, &o.rc) || self[..] == o[..]
    }
}

impl<T: Bits> PartialEq<Vec<T>> for Shared<T> {
    fn eq(&self, o: &Vec<T>) -> bool {
        self[..] == o[..]
    }
}

impl<T: Bits> PartialEq<Shared<T>> for Vec<T> {
    fn eq(&self, o: &Shared<T>) -> bool {
        self[..] == o[..]
    }
}

impl<'a, T: Bits> IntoIterator for &'a Shared<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

impl<T: Bits + std::fmt::Debug> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self[..].fmt(f)
    }
}

impl<T: Bits> Shared<T> {
    /// A copy changed by `f`, held as any other array made from a `Vec`.
    pub fn edit(&mut self, f: impl FnOnce(&mut Vec<T>)) {
        let mut v = self.to_vec();
        f(&mut v);
        *self = v.into();
    }
    /// This holder's share of the array's bytes, so that every holder's together count it once
    /// (`demand::mem`).
    pub fn heap_share(&self) -> usize {
        let holders = Rc::strong_count(&self.rc) - usize::from(self.key != 0);
        std::mem::size_of_val(&*self.rc) / holders.max(1)
    }
    /// Whether two holders hold one array.
    pub fn same_array(&self, o: &Shared<T>) -> bool {
        Rc::ptr_eq(&self.rc, &o.rc)
    }
}

/// The arrays of `T` this thread holds.
#[cfg(test)]
pub fn held<T: Bits>() -> usize {
    T::registry().with(|r| r.borrow().0.values().map(|b| b.len()).sum())
}

/// The bytes of the arrays this thread holds, once each, and as they would be with every holder its own copy
/// (`bin/demand_mem.rs`).
pub fn held_bytes() -> (usize, usize) {
    fn of<T: Bits>() -> (usize, usize) {
        T::registry().with(|r| {
            r.borrow().0.values().flatten().fold((0, 0), |(once, each), x| {
                let b = std::mem::size_of_val(&**x);
                (once + b, each + b * (Rc::strong_count(x) - 1))
            })
        })
    }
    let (u, f) = (of::<u32>(), of::<f32>());
    (u.0 + f.0, u.1 + f.1)
}
