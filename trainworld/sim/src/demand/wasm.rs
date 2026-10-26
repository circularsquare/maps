//! The wasm entry outside `DemandApi` (api.rs): the linear memory size and the heap in use
//! (T-1205), and what the last panic was (T-1174).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use wasm_bindgen::prelude::*;

/// Current size of the wasm linear memory (it never shrinks, so this is also its peak).
#[wasm_bindgen]
pub fn wasm_memory_bytes() -> f64 {
    (core::arch::wasm32::memory_size(0) * 65536) as f64
}

/// Bytes the allocator has handed out and not had back (T-1205): the page sets a worker's memory
/// against what its items and a solve need, to tell a high-water mark from a need.
#[wasm_bindgen]
pub fn wasm_heap_bytes() -> f64 {
    LIVE.load(Ordering::Relaxed) as f64
}
/// The most `wasm_heap_bytes` has been since the last call with `reset` (since the start before one).
#[wasm_bindgen]
pub fn wasm_heap_peak(reset: bool) -> f64 {
    let peak = PEAK.load(Ordering::Relaxed);
    if reset { PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed); }
    peak as f64
}

/// T-1174: an allocation the memory could not grow for aborts the instance as a bare `unreachable`
/// with no message (the alloc error hook prints nowhere on wasm32), so its size is noted here first.
struct NotingAlloc;
static FAILED_ALLOC: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(n: usize) {
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    if live > PEAK.load(Ordering::Relaxed) { PEAK.store(live, Ordering::Relaxed); }
}

unsafe impl GlobalAlloc for NotingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if p.is_null() { FAILED_ALLOC.store(l.size().max(1), Ordering::Relaxed); } else { grew(l.size()); }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if p.is_null() { FAILED_ALLOC.store(l.size().max(1), Ordering::Relaxed); } else { grew(l.size()); }
        p
    }
    unsafe fn realloc(&self, ptr: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, l, new_size);
        if p.is_null() {
            FAILED_ALLOC.store(new_size.max(1), Ordering::Relaxed);
        } else {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            grew(new_size);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(ptr, l)
    }
}

#[global_allocator]
static ALLOC: NotingAlloc = NotingAlloc;

thread_local! {
    static LAST_PANIC: RefCell<String> = RefCell::new(String::new());
}

#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| {
        let text = info.to_string();
        LAST_PANIC.with(|p| if let Ok(mut p) = p.try_borrow_mut() { *p = text; });
    }));
}

use super::api::DemandApi;

#[wasm_bindgen]
impl DemandApi {
    /// T-1174: panic inside a `&mut self` call, leaving this object's borrow held as a panic in a
    /// solve does (the dev server's `debugPanic`).
    pub fn debug_panic(&mut self) {
        panic!("a test panic (T-1174)");
    }
}

/// T-1174: the last panic's message, or the allocation that ran out of memory; empty if neither.
/// Safe to call on an instance a panic has left unusable: it borrows no exported object.
#[wasm_bindgen]
pub fn last_panic() -> String {
    let bytes = FAILED_ALLOC.load(Ordering::Relaxed);
    let panic = LAST_PANIC.with(|p| p.try_borrow().map(|p| p.clone()).unwrap_or_default());
    match (bytes, panic.is_empty()) {
        (0, _) => panic,
        (b, true) => format!("out of memory: {b} bytes could not be allocated"),
        (b, false) => format!("out of memory: {b} bytes could not be allocated; last panic: {panic}"),
    }
}
