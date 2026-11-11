//! The wasm modules' allocator (T-1594, T-2699, T-2716): dlmalloc, as std's wasm32 allocator is, over
//! a system side that grows the linear memory in big steps. dlmalloc on its own grows the memory by
//! what a request lacks (64 KB at a time for small ones), and each grow costs about as much as the
//! whole memory, not the bytes added: a pass that allocated 30 MB in 9 KB pieces took 30 s longer in
//! a demand worker of 1.3 GB, and the clock worker's boarding account took 100-300 ms a step on all of
//! Japan, nearly all of it in grows (7 ms natively).
//!
//! T-1594 and T-2699 got the same steps from std's allocator by allocating and freeing a big block
//! after any allocation that grew the memory. That made the allocator's alloc too big to inline, and
//! T-2716 found what that exposed (notes/T-2716.md): LLVM takes `__rust_alloc_zeroed` and
//! `__rust_alloc` for fresh objects (zeroed, or uninitialized), so where a function kept such a call
//! and had a free of the same block inlined, it folded dlmalloc's read of the block's header (just
//! before the block) to 0. In a `--release` build of the track module that made dlmalloc's size check
//! abort loading the Japan save. Hence the methods below are never inlined: no caller sees the
//! headers. Growing in the system side also does no allocation inside an allocation.

use core::arch::wasm32;
use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout};
use wasm_bindgen::prelude::*;

#[cfg(target_feature = "atomics")]
compile_error!("crate::heap takes no lock: the wasm modules are single threaded");

const PAGE: usize = 65536;

/// T-2802: the least grow in pages when set (0: each `Heap`'s own `min`). The module's one memory, so one
/// setting for its allocator.
static MIN_PAGES: AtomicUsize = AtomicUsize::new(0);

/// T-2802: grows of at least `bytes` from now on, in place of the module's 32 MB: a small device's page sets
/// 8 MB, so a worker's memory ends nearer its peak (a demand worker's 65 MB held 31-40).
#[wasm_bindgen]
pub fn set_heap_step(bytes: f64) {
    MIN_PAGES.store((bytes.max(0.0) as usize).div_ceil(PAGE), Ordering::Relaxed);
}

/// dlmalloc's system side: each grow is at least `min` pages, or `1/frac` of the memory once that
/// is more (`frac` 0: no share), so a growing heap grows the memory a few dozen times in all.
pub struct Grow {
    min: usize,
    frac: usize,
    /// whether the linker's spare heap (`__heap_base` to `__heap_end`) has been handed out
    spare: UnsafeCell<bool>,
}

extern "C" {
    static __heap_base: u8;
    static __heap_end: u8;
}

// The wasm modules are single threaded (no `atomics`, above).
unsafe impl Send for Grow {}

impl Grow {
    /// Grows the memory by `pages`, or by what `size` bytes need if that fails.
    fn grow(&self, size: usize, pages: usize) -> (*mut u8, usize, u32) {
        let need = size.div_ceil(PAGE);
        let mut got = pages.max(need);
        let mut prev = wasm32::memory_grow(0, got);
        if prev == usize::MAX && got > need {
            got = need;
            prev = wasm32::memory_grow(0, got);
        }
        if prev == usize::MAX {
            return (ptr::null_mut(), 0, 0);
        }
        let base = prev as u64 * PAGE as u64;
        let len = got as u64 * PAGE as u64;
        // The last bytes of the address space can't be handed out (a pointer one past the end would
        // wrap to 0), as in dlmalloc's own wasm side.
        let len = if base + len >= 1 << 32 { (1u64 << 32) - 16 - base } else { len };
        (base as usize as *mut u8, len as usize, 0)
    }
}

unsafe impl dlmalloc::Allocator for Grow {
    fn alloc(&self, size: usize) -> (*mut u8, usize, u32) {
        // dlmalloc's wasm side hands out the linker's spare heap first, once, if it is big enough.
        unsafe {
            if !*self.spare.get() {
                *self.spare.get() = true;
                let base = &__heap_base as *const u8 as usize;
                let end = &__heap_end as *const u8 as usize;
                if base != 0 && end > base && end - base >= size {
                    return (base as *mut u8, end - base, 0);
                }
            }
        }
        let now = wasm32::memory_size(0);
        let share = if self.frac == 0 { 0 } else { now / self.frac };
        let min = match MIN_PAGES.load(Ordering::Relaxed) { 0 => self.min, set => set };
        self.grow(size, min.max(share))
    }
    fn remap(&self, _ptr: *mut u8, _old: usize, _new: usize, _can_move: bool) -> *mut u8 {
        ptr::null_mut()
    }
    fn free_part(&self, _ptr: *mut u8, _old: usize, _new: usize) -> bool {
        false
    }
    fn free(&self, _ptr: *mut u8, _size: usize) -> bool {
        false
    }
    fn can_release_part(&self, _flags: u32) -> bool {
        false
    }
    fn allocates_zeros(&self) -> bool {
        true
    }
    fn page_size(&self) -> usize {
        PAGE
    }
}

/// A global allocator: dlmalloc over [`Grow`]. Its methods stay out of line (the module's comment).
pub struct Heap(UnsafeCell<dlmalloc::Dlmalloc<Grow>>);

// Single threaded, as above.
unsafe impl Sync for Heap {}

impl Heap {
    /// Grows of at least `min_bytes`, or `1/frac` of the memory once that is more (`frac` 0: none).
    pub const fn new(min_bytes: usize, frac: usize) -> Heap {
        let grow = Grow { min: min_bytes.div_ceil(PAGE), frac, spare: UnsafeCell::new(false) };
        Heap(UnsafeCell::new(dlmalloc::Dlmalloc::new_with_allocator(grow)))
    }
}

unsafe impl GlobalAlloc for Heap {
    #[inline(never)]
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        (*self.0.get()).malloc(l.size(), l.align())
    }
    #[inline(never)]
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        (*self.0.get()).calloc(l.size(), l.align())
    }
    #[inline(never)]
    unsafe fn dealloc(&self, ptr: *mut u8, l: Layout) {
        (*self.0.get()).free(ptr, l.size(), l.align())
    }
    #[inline(never)]
    unsafe fn realloc(&self, ptr: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        (*self.0.get()).realloc(ptr, l.size(), l.align(), new_size)
    }
}

/// The clock worker's module (the demand module's is `demand::wasm`'s, over its own `Heap`): grows of
/// 32 MB, or an eighth of the memory once that is more.
#[cfg(not(feature = "demand"))]
#[global_allocator]
static ALLOC: Heap = Heap::new(32 << 20, 8);
