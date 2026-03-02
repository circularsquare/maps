#!/usr/bin/env python3
"""
clean_line.py - Clean up pixel art line exports to 1px-wide paths.

Usage:
    python clean_line.py input.png output.png [--threshold 128] [--passes 3]

Steps:
    1. Binarize: dark pixels become foreground (1), light become background (0)
    2. Remove spurs: foreground pixels with only 1 neighbor (or chain of 2) that
       aren't needed for connectivity are deleted.
    3. Remove 2x2 blocks: if a 2x2 region is all foreground, remove the pixel
       whose removal keeps the line connected (Zhang-Suen style thinning).
    4. Full skeletonization via Zhang-Suen thinning to guarantee 1px width.
"""

import argparse
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import label


def binarize(img_array, threshold=128):
    """Convert to binary: dark pixels = 1 (foreground), light pixels = 0 (background)."""
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        alpha_mask = img_array[:, :, 3] > threshold
        gray = np.mean(img_array[:, :, :3], axis=2)
        return (alpha_mask & (gray < threshold)).astype(np.uint8)
    elif img_array.ndim == 3:
        gray = np.mean(img_array[:, :, :3], axis=2)
    else:
        gray = img_array.astype(float)
    return (gray < threshold).astype(np.uint8)


def count_neighbors(binary):
    """For each pixel, count how many of its 8 neighbors are foreground."""
    from scipy.ndimage import convolve
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    return convolve(binary.astype(np.uint8), kernel, mode='constant', cval=0)


def crossing_number(binary):
    """
    Compute crossing number (number of 0->1 transitions in 8-neighbors clockwise).
    Used for Zhang-Suen thinning.
    """
    p = {}
    p[2] = np.roll(np.roll(binary, -1, axis=0),  0, axis=1)   # N
    p[3] = np.roll(np.roll(binary, -1, axis=0), -1, axis=1)   # NE
    p[4] = np.roll(np.roll(binary,  0, axis=0), -1, axis=1)   # E
    p[5] = np.roll(np.roll(binary,  1, axis=0), -1, axis=1)   # SE
    p[6] = np.roll(np.roll(binary,  1, axis=0),  0, axis=1)   # S
    p[7] = np.roll(np.roll(binary,  1, axis=0),  1, axis=1)   # SW
    p[8] = np.roll(np.roll(binary,  0, axis=0),  1, axis=1)   # W
    p[9] = np.roll(np.roll(binary, -1, axis=0),  1, axis=1)   # NW

    order = [p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]]
    cn = np.zeros_like(binary, dtype=np.int32)
    for i in range(8):
        a = order[i].astype(np.int32)
        b = order[(i+1) % 8].astype(np.int32)
        cn += np.maximum(b - a, 0)
    return cn, p


def zhang_suen_thin(binary):
    """Standard Zhang-Suen thinning algorithm."""
    img = binary.copy().astype(np.uint8)
    
    while True:
        changed = False
        
        for step in [1, 2]:
            cn, p = crossing_number(img)
            N = count_neighbors(img)
            
            # Common conditions
            cond1 = (img == 1)
            cond2 = (N >= 2) & (N <= 6)
            cond3 = (cn == 1)
            
            if step == 1:
                cond4 = (p[2] * p[4] * p[6] == 0)
                cond5 = (p[4] * p[6] * p[8] == 0)
            else:
                cond4 = (p[2] * p[4] * p[8] == 0)
                cond5 = (p[2] * p[6] * p[8] == 0)
            
            remove = cond1 & cond2 & cond3 & cond4 & cond5
            
            if np.any(remove):
                changed = True
                img[remove] = 0
        
        if not changed:
            break
    
    return img


def fix_diagonal_connections(binary):
    """
    After thinning, Zhang-Suen can leave pixels connected only diagonally:
    
        . X        X .
        X .   or   . X
    
    For each such diagonal pair (A at (r,c) and B at (r+1,c+1), or A at (r,c+1)
    and B at (r+1,c)), if both shared orthogonal neighbors are background,
    add one bridging pixel to restore orthogonal connectivity.
    
    We pick which of the two bridge positions to fill by checking which one
    has more existing foreground neighbors (keeps the line looking natural).
    """
    img = binary.copy()
    rows, cols = img.shape

    changed = True
    while changed:
        changed = False

        # Pattern 1: diagonal \  (top-left & bottom-right set, top-right & bottom-left empty)
        #   A .
        #   . B
        A = img[:-1, :-1]
        B = img[1:,  1:]
        TR = img[:-1, 1:]   # top-right (shared neighbor)
        BL = img[1:,  :-1]  # bottom-left (shared neighbor)
        mask1 = (A == 1) & (B == 1) & (TR == 0) & (BL == 0)

        # Pattern 2: diagonal /  (top-right & bottom-left set, top-left & bottom-right empty)
        #   . A
        #   B .
        A2  = img[:-1, 1:]
        B2  = img[1:,  :-1]
        TL2 = img[:-1, :-1]
        BR2 = img[1:,  1:]
        mask2 = (A2 == 1) & (B2 == 1) & (TL2 == 0) & (BR2 == 0)

        locs1 = np.argwhere(mask1)
        locs2 = np.argwhere(mask2)

        if len(locs1) == 0 and len(locs2) == 0:
            break

        for (r, c) in locs1:
            # Bridge candidates: (r, c+1) [top-right] or (r+1, c) [bottom-left]
            tr_nb = count_neighbors_single(img, r,   c+1)
            bl_nb = count_neighbors_single(img, r+1, c)
            if tr_nb >= bl_nb:
                img[r,   c+1] = 1
            else:
                img[r+1, c  ] = 1
            changed = True

        for (r, c) in locs2:
            # Bridge candidates: (r, c) [top-left] or (r+1, c+1) [bottom-right]
            tl_nb = count_neighbors_single(img, r,   c)
            br_nb = count_neighbors_single(img, r+1, c+1)
            if tl_nb >= br_nb:
                img[r,   c  ] = 1
            else:
                img[r+1, c+1] = 1
            changed = True

    return img


def remove_spurs(binary, spur_length=2):
    """
    Remove spur pixels: foreground pixels that are endpoints (1 neighbor)
    in chains of length <= spur_length, unless they are isolated (would
    become disconnected).
    """
    img = binary.copy()
    
    for _ in range(spur_length):
        nb = count_neighbors(img)
        # Endpoints: foreground pixels with exactly 1 neighbor
        endpoints = (img == 1) & (nb == 1)
        img[endpoints] = 0
    
    return img


def remove_2x2_blocks(binary):
    """
    Wherever a 2x2 block of foreground pixels exists, remove the one pixel
    that is most 'redundant' (i.e., its removal keeps connectivity).
    We use a simple approach: remove pixel if it still has >= 2 neighbors after removal
    and connectivity number is 1.
    """
    img = binary.copy()
    rows, cols = img.shape
    
    changed = True
    while changed:
        changed = False
        # Find top-left corners of 2x2 blocks
        r, c = np.where(
            (img[:-1, :-1] == 1) & 
            (img[:-1, 1:] == 1) & 
            (img[1:, :-1] == 1) & 
            (img[1:, 1:] == 1)
        )
        
        if len(r) == 0:
            break
            
        # For each 2x2 block, try to remove each of the 4 pixels
        # Remove the one with highest neighbor count (most redundant)
        for i in range(len(r)):
            ri, ci = r[i], c[i]
            candidates = [(ri, ci), (ri, ci+1), (ri+1, ci), (ri+1, ci+1)]
            
            for (pr, pc) in candidates:
                if img[pr, pc] == 0:
                    continue
                # Check if still part of a 2x2 block
                # Temporarily remove and check connectivity
                img[pr, pc] = 0
                # Count neighbors of the removed pixel's neighbors to verify connectivity
                # Simple check: the 4 corners should all still be connected
                # Use crossing number: if it was 1 before, removal is safe
                neighborhood = img[max(0,pr-1):pr+2, max(0,pc-1):pc+2]
                # Just check we haven't isolated anything: each former neighbor
                # should still have >= 1 neighbor
                safe = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr2, nc2 = pr+dr, pc+dc
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and img[nr2, nc2] == 1:
                            nb_count = count_neighbors_single(img, nr2, nc2)
                            if nb_count == 0:
                                safe = False
                                break
                    if not safe:
                        break
                
                if safe:
                    changed = True
                    break
                else:
                    img[pr, pc] = 1  # restore
    
    return img


def count_neighbors_single(img, r, c):
    """Count 8-neighbors of a single pixel."""
    rows, cols = img.shape
    count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                count += img[nr, nc]
    return count


def main():
    parser = argparse.ArgumentParser(description='Clean pixel art lines to 1px width')
    parser.add_argument('input', help='Input PNG file')
    parser.add_argument('output', help='Output PNG file')
    parser.add_argument('--threshold', type=int, default=128,
                        help='Threshold for binarization (0-255, default 128). '
                             'Pixels darker than this become foreground.')
    parser.add_argument('--spur-length', type=int, default=2,
                        help='Max length of spurs to remove (default 2)')
    parser.add_argument('--no-thin', action='store_true',
                        help='Skip Zhang-Suen thinning (only do spur/2x2 removal)')
    parser.add_argument('--fg-color', type=str, default='000000',
                        help='Foreground color in hex (default: 000000 = black)')
    parser.add_argument('--bg-color', type=str, default='ffffff',
                        help='Background color in hex (default: ffffff = white)')
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    img = Image.open(args.input)
    
    # Keep RGBA so we can binarize on alpha; convert others to RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Step 1: Binarize
    print("Step 1: Binarizing...")
    binary = binarize(arr, threshold=args.threshold)
    fg_count = np.sum(binary)
    print(f"  Foreground pixels: {fg_count:,}")
    
    # Step 2: Remove spurs
    print(f"Step 2: Removing spurs (length <= {args.spur_length})...")
    binary = remove_spurs(binary, spur_length=args.spur_length)
    print(f"  Foreground pixels after spur removal: {np.sum(binary):,}")
    
    # Step 3: Zhang-Suen thinning (handles 2x2 blocks and wider lines)
    if not args.no_thin:
        print("Step 3: Thinning (Zhang-Suen)...")
        binary = zhang_suen_thin(binary)
        print(f"  Foreground pixels after thinning: {np.sum(binary):,}")
        
        # Step 4: Fix diagonal-only connections
        print("Step 4: Fixing diagonal-only connections...")
        binary = fix_diagonal_connections(binary)
        print(f"  Foreground pixels after diagonal fix: {np.sum(binary):,}")

        # Step 5: Final spur cleanup after thinning
        print("Step 5: Final spur cleanup...")
        binary = remove_spurs(binary, spur_length=args.spur_length)
        print(f"  Foreground pixels after final cleanup: {np.sum(binary):,}")
    
    # Convert back to image
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    fg_rgb = hex_to_rgb(args.fg_color)
    
    # RGBA output: foreground = fg_color fully opaque, background = transparent
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[binary == 1] = (*fg_rgb, 255)   # opaque foreground
    out[binary == 0] = (0, 0, 0, 0)     # fully transparent background
    
    print(f"Saving to {args.output}...")
    Image.fromarray(out, 'RGBA').save(args.output)
    print("Done!")


if __name__ == '__main__':
    main()