import argparse
import math
import os
import random
from typing import List, Tuple

from PIL import Image, ImageDraw

# =========================
# Constants Section
# =========================

# General card/image constants
CARD_IMAGE_COUNT = 8
DEFAULT_CANVAS_SIZE = 1200
DEFAULT_CARD_DIAM_FRAC = 0.92
DEFAULT_BG_COLOR = "#FFFFFF"

# Card base drawing
EDGE_STROKE_COLOR = (210, 210, 210, 255)
EDGE_STROKE_WIDTH = 4
EDGE_STROKE_INSET = 2

# Placement/packing constants
PLACEMENT_NUM_SECTORS = 12
PLACEMENT_SECTOR_OVERFULL_PROB = 0.6
PLACEMENT_ATTEMPTS_PER_SYMBOL = 2000
PLACEMENT_ATTEMPTS_PER_SYMBOL_MAX = 3000

# Allow icons to overlap a little bit (fraction of icon diameter)
ICON_OVERLAP_FRAC = 0.20  # Allow up to 25% overlap between icons

# When relaxing, allow a bit more overlap
PLACEMENT_RELAX_ATTEMPTS = (600, 1200, 1800)
PLACEMENT_RELAX_GAP_SHRINK = 0.9
PLACEMENT_RELAX_MIN_FRAC_SHRINK = 0.9
PLACEMENT_RELAX_MIN_FRAC_FLOOR = 0.12
PLACEMENT_RELAX_HI_FRAC_SHRINK = 0.95
PLACEMENT_RELAX_HI_FRAC_FLOOR = 0.35

# Banded greedy placement
BANDED_GREEDY_BAND_COUNTS = (2, 3, 3)
BANDED_GREEDY_MAX_RESTARTS = 5
BANDED_GREEDY_ATTEMPTS_PER_SYMBOL = 600
BANDED_GREEDY_BAND_INNER_MAX = 0.40
BANDED_GREEDY_BAND_MID_LO = 0.30
BANDED_GREEDY_BAND_MID_HI = 0.75
BANDED_GREEDY_BAND_MID_FALLBACK_LO = 0.2
BANDED_GREEDY_BAND_MID_FALLBACK_HI = 0.6
BANDED_GREEDY_BAND_OUTER_LO = 0.70
BANDED_GREEDY_BAND_OUTER_HI = 1.00
BANDED_GREEDY_GAP_SHRINK_PER_RESTART = 0.95
BANDED_GREEDY_SCALE_SHRINK_PER_RESTART = 0.96

# Force placement
FORCE_PLACE_MAX_OUTER_TRIES = 6
FORCE_PLACE_MAX_ITERS = 900
FORCE_PLACE_SHRINK_PER_FAIL = 0.93
FORCE_PLACE_BOUNDARY_PADDING = 1.0
FORCE_PLACE_OVERLAP_RESOLVE = 0.55
FORCE_PLACE_RADIAL_ADJ = 0.06

# Template placement
TEMPLATE_TRIES = 120
TEMPLATE_JITTER_TRIES_PER_TEMPLATE = 40
TEMPLATE_SHRINK_PER_FAIL = 0.93
TEMPLATE_ANGLE_JITTER_DEG = 12.0
TEMPLATE_SCALE_STEPS = 6
TEMPLATE_RING_RADIUS_JITTER = 0.06

TEMPLATES = [
    [(1, 0.0), (7, 0.60)],      # 1 center + 7 around
    [(3, 0.34), (5, 0.62)],     # 3 inner + 5 outer
    [(4, 0.38), (4, 0.64)],     # 4 inner + 4 outer
    [(2, 0.30), (6, 0.66)],     # 2 inner + 6 outer
]

# Fallback ring placement
FALLBACK_RING_BASE_FRAC = 0.23
FALLBACK_RING_SIZE_JITTER_FRAC = 0.12
FALLBACK_RING_PAD_FRAC = 0.000  # No padding to allow overlap
FALLBACK_RING_ANGLE_JITTER_DEG = 6.0
FALLBACK_RING_MAX_OUTER_LOOPS = 50
FALLBACK_RING_ANGLE_JITTER_RAD = math.radians(FALLBACK_RING_ANGLE_JITTER_DEG)
FALLBACK_RING_SHRINK_PER_FAIL = 0.93
FALLBACK_RING_LAST_RESORT_SCALE = 0.7
FALLBACK_RING_MIN_CENTER_R = 10.0

# Generate card base size fractions
BASE_MIN_FRAC = 0.25
BASE_MAX_FRAC = 0.40
BASE_PAD_FRAC = 0.00  # No padding between symbols to allow overlap

# Attempts config for generate_card
ATTEMPTS_CONFIG = [
    (BASE_MIN_FRAC, BASE_MAX_FRAC, BASE_PAD_FRAC),
    (BASE_MIN_FRAC * 0.95, BASE_MAX_FRAC * 0.95, BASE_PAD_FRAC),
    (BASE_MIN_FRAC * 0.9, BASE_MAX_FRAC * 0.9, BASE_PAD_FRAC),
    (max(0.14, BASE_MIN_FRAC * 0.85), max(BASE_MIN_FRAC * 0.85 + 0.05, BASE_MAX_FRAC * 0.85), BASE_PAD_FRAC),
    (max(0.13, BASE_MIN_FRAC * 0.8), max(BASE_MIN_FRAC * 0.8 + 0.05, BASE_MAX_FRAC * 0.8), BASE_PAD_FRAC),
]

# Relax steps for fallback
RELAX_STEPS = [
    (0.95, 0.95, 1.0),
    (0.92, 0.92, 1.0),
    (0.90, 0.90, 1.0),
]

# Size fractions for visible diversity
SIZE_FRACS_TEMPLATE = [
    BASE_MAX_FRAC * 1.18,  # very large
    BASE_MAX_FRAC * 1.05,  # large
    (BASE_MIN_FRAC + BASE_MAX_FRAC) * 0.60,  # medium-large
    (BASE_MIN_FRAC + BASE_MAX_FRAC) * 0.48,  # medium
    (BASE_MIN_FRAC + BASE_MAX_FRAC) * 0.38,  # medium-small
    BASE_MIN_FRAC * 1.10,  # small
    BASE_MIN_FRAC * 0.95,  # smaller
    BASE_MIN_FRAC * 0.80,  # smallest
]

BANDS_TEMPLATE = ["inner", "inner", "mid", "mid", "mid", "outer", "outer", "outer"]

# Debug overlay
DEBUG_OVERLAY_CIRCLE_OUTLINE = (0, 0, 0, 80)
DEBUG_OVERLAY_CIRCLE_WIDTH = 2
DEBUG_OVERLAY_PLACEMENT_OUTLINE = (255, 0, 0, 120)
DEBUG_OVERLAY_PLACEMENT_WIDTH = 2
DEBUG_OVERLAY_CROSSHAIR_COLOR = (255, 0, 0, 180)
DEBUG_OVERLAY_CROSSHAIR_WIDTH = 2
DEBUG_OVERLAY_CROSSHAIR_SIZE = 6

# =========================
# End of Constants Section
# =========================

def load_images(image_paths: List[str]) -> List[Image.Image]:
    loaded_images: List[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGBA")
        # Auto-crop transparent borders to improve packing
        alpha = img.split()[-1]
        bbox = alpha.getbbox()
        if bbox is not None:
            img = img.crop(bbox)
        loaded_images.append(img)
    return loaded_images


def draw_card_base(canvas_size: int, card_diameter: int, background_color: str = DEFAULT_BG_COLOR) -> Image.Image:
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    cx = cy = canvas_size // 2
    radius = card_diameter // 2

    # Card face
    draw = ImageDraw.Draw(canvas)
    bounds = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bounds, fill=background_color)

    # Subtle edge stroke
    draw.ellipse(
        [
            bounds[0] + EDGE_STROKE_INSET,
            bounds[1] + EDGE_STROKE_INSET,
            bounds[2] - EDGE_STROKE_INSET,
            bounds[3] - EDGE_STROKE_INSET,
        ],
        outline=EDGE_STROKE_COLOR,
        width=EDGE_STROKE_WIDTH,
    )

    return canvas


def uniform_point_in_circle(max_radius: float) -> Tuple[float, float]:
    theta = random.random() * 2.0 * math.pi
    r = math.sqrt(random.random()) * max_radius
    return r * math.cos(theta), r * math.sin(theta)


def uniform_point_in_annulus(min_radius: float, max_radius: float) -> Tuple[float, float]:
    if max_radius <= 0:
        return 0.0, 0.0
    if min_radius < 0:
        min_radius = 0.0
    if min_radius > max_radius:
        min_radius = max(0.0, max_radius * 0.5)
    theta = random.random() * 2.0 * math.pi
    # Uniform by area between r^2 bounds
    r2 = random.uniform(min_radius * min_radius, max_radius * max_radius)
    r = math.sqrt(r2)
    return r * math.cos(theta), r * math.sin(theta)


def try_place_symbols(
    images: List[Image.Image],
    card_diameter: int,
    canvas_size: int,
    min_frac: float,
    max_frac: float,
    pad_frac: float,
    max_attempts_per_symbol: int = PLACEMENT_ATTEMPTS_PER_SYMBOL,
    size_fracs: List[float] | None = None,
    inner_bias_count: int = 0,
    inner_bias_scale: float = 0.5,
    bands: List[str] | None = None,
) -> List[dict]:
    cx = cy = canvas_size / 2.0
    card_radius = card_diameter / 2.0

    placements: List[dict] = []

    # Track angular sector occupancy to reduce crowding on one side
    num_sectors = PLACEMENT_NUM_SECTORS
    sector_counts = [0 for _ in range(num_sectors)]

    # Determine placement order: place inner-band and larger icons earlier
    indices = list(range(len(images)))
    def band_priority(i: int) -> int:
        if bands is None:
            return 1
        b = bands[i]
        return 0 if b == "inner" else (1 if b == "mid" else 2)
    if size_fracs is not None:
        indices.sort(key=lambda i: (-band_priority(i), -size_fracs[i] if size_fracs is not None else 0.0, random.random()))
    else:
        indices.sort(key=lambda i: (-band_priority(i), random.random()))

    # Base gap between symbols and padding from edge
    base_gap_px = int(pad_frac * card_diameter)

    placed_so_far = 0
    for idx in indices:
        img = images[idx]
        img_w, img_h = img.size

        placed = False
        attempt = 0
        # Progressive relaxation if placement becomes too hard
        gap_px = base_gap_px
        frac_lo = min_frac
        frac_hi = max_frac

        while attempt < max_attempts_per_symbol and not placed:
            attempt += 1

            # Target size as a fraction of card diameter (based on max dimension)
            if size_fracs is not None:
                base_frac = max(frac_lo, min(frac_hi, size_fracs[idx]))
                # Make the jitter range wider for more size difference
                target_max_dim_frac = base_frac * random.uniform(0.65, 1.50)
                target_max_dim_frac = max(frac_lo, min(frac_hi, target_max_dim_frac))
            else:
                target_max_dim_frac = random.uniform(frac_lo, frac_hi)
            scale = (target_max_dim_frac * card_diameter) / max(img_w, img_h)
            scaled_w = img_w * scale
            scaled_h = img_h * scale

            # Rotation random 0..360
            rotation_deg = random.random() * 360.0

            # Bounding circle radius (rotation-independent) with overlap allowance
            bounding_radius = 0.5 * math.hypot(scaled_w, scaled_h) * (1.0 - ICON_OVERLAP_FRAC * 0.5)

            # Ensure it fits in the card
            if bounding_radius >= card_radius - 2:
                # too big, retry with smaller size range
                continue

            # Candidate center within card so the entire bounding circle fits
            max_center_radius = card_radius - bounding_radius
            # Band-driven radial sampling to avoid ring-like layouts
            if bands is not None:
                band = bands[idx]
                if band == "inner":
                    r_min = 0.0
                    r_max = max(0.0, inner_bias_scale * max_center_radius)
                elif band == "mid":
                    r_min = max(0.0, 0.35 * max_center_radius)
                    r_max = max(0.0, 0.70 * max_center_radius)
                    if r_min > r_max:
                        r_min, r_max = 0.2 * max_center_radius, 0.6 * max_center_radius
                else:  # outer
                    r_min = max(0.0, 0.72 * max_center_radius)
                    r_max = max(0.0, 1.00 * max_center_radius)
                dx, dy = uniform_point_in_annulus(r_min, r_max)
            else:
                # Optional early inner bias for first few placements
                if placed_so_far < inner_bias_count:
                    biased_radius = max(0.0, inner_bias_scale * max_center_radius)
                    dx, dy = uniform_point_in_circle(biased_radius)
                else:
                    dx, dy = uniform_point_in_circle(max_center_radius)
            center_x = cx + dx
            center_y = cy + dy

            # Sector crowding control
            theta = math.atan2(center_y - cy, center_x - cx)
            if theta < 0:
                theta += 2.0 * math.pi
            sector_index = int(theta / (2.0 * math.pi) * num_sectors)
            min_count = min(sector_counts)
            # Probabilistically discourage placing in overfull sectors
            if sector_counts[sector_index] > min_count + 1:
                if random.random() < PLACEMENT_SECTOR_OVERFULL_PROB:
                    continue

            # Overlap check against already placed symbols - allow some overlap
            ok = True
            for p in placements:
                dist = math.hypot(center_x - p["cx"], center_y - p["cy"])
                # Allow overlap up to ICON_OVERLAP_FRAC of the smaller radius
                min_radius = min(bounding_radius, p["radius"])
                overlap_threshold = min_radius * ICON_OVERLAP_FRAC
                if dist < (bounding_radius + p["radius"] - overlap_threshold):
                    ok = False
                    break

            if not ok:
                # After many failures, slightly shrink and relax constraints
                if attempt in PLACEMENT_RELAX_ATTEMPTS:
                    gap_px = max(2, int(gap_px * PLACEMENT_RELAX_GAP_SHRINK))
                    frac_lo = max(min_frac * PLACEMENT_RELAX_MIN_FRAC_SHRINK, PLACEMENT_RELAX_MIN_FRAC_FLOOR)
                    frac_hi = max(frac_lo + 0.05, min(frac_hi * PLACEMENT_RELAX_HI_FRAC_SHRINK, PLACEMENT_RELAX_HI_FRAC_FLOOR))
                continue

            placements.append(
                {
                    "index": idx,
                    "scale": scale,
                    "rotation": rotation_deg,
                    "cx": center_x,
                    "cy": center_y,
                    "radius": bounding_radius,
                    "scaled_w": int(round(scaled_w)),
                    "scaled_h": int(round(scaled_h)),
                }
            )
            sector_counts[sector_index] += 1
            placed = True
            placed_so_far += 1

        if not placed:
            # If placement failed, raise to trigger a retry with wider canvas or adjusted params
            raise RuntimeError("Failed to place all symbols without overlap. Try increasing canvas size or lowering max size fraction.")

    return placements


def render_card(
    images: List[Image.Image],
    placements: List[dict],
    canvas: Image.Image,
) -> Image.Image:
    result = canvas.copy()
    # Draw from back to front in random order (already shuffled by placements)
    for p in placements:
        img = images[p["index"]]
        # Scale
        scaled = img.resize((p["scaled_w"], p["scaled_h"]), resample=Image.BICUBIC)
        # Rotate around center
        rotated = scaled.rotate(p["rotation"], resample=Image.BICUBIC, expand=True)
        # Paste
        x = int(round(p["cx"] - rotated.width / 2.0))
        y = int(round(p["cy"] - rotated.height / 2.0))
        result.alpha_composite(rotated, dest=(x, y))
    return result


def render_debug_overlay(
    canvas_size: int,
    card_diameter: int,
    placements: List[dict],
) -> Image.Image:
    overlay = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx = cy = canvas_size // 2
    r = card_diameter // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=DEBUG_OVERLAY_CIRCLE_OUTLINE, width=DEBUG_OVERLAY_CIRCLE_WIDTH)
    for i, p in enumerate(placements):
        x = int(round(p["cx"]))
        y = int(round(p["cy"]))
        rad = int(round(p["radius"]))
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], outline=DEBUG_OVERLAY_PLACEMENT_OUTLINE, width=DEBUG_OVERLAY_PLACEMENT_WIDTH)
        # crosshair
        draw.line([x - DEBUG_OVERLAY_CROSSHAIR_SIZE, y, x + DEBUG_OVERLAY_CROSSHAIR_SIZE, y], fill=DEBUG_OVERLAY_CROSSHAIR_COLOR, width=DEBUG_OVERLAY_CROSSHAIR_WIDTH)
        draw.line([x, y - DEBUG_OVERLAY_CROSSHAIR_SIZE, x, y + DEBUG_OVERLAY_CROSSHAIR_SIZE], fill=DEBUG_OVERLAY_CROSSHAIR_COLOR, width=DEBUG_OVERLAY_CROSSHAIR_WIDTH)
    return overlay


def compute_ring_stats(
    placements: List[dict],
    canvas_size: int,
    card_diameter: int,
) -> tuple[float, float, float, float, int, int, int, float]:
    cx = canvas_size / 2.0
    cy = canvas_size / 2.0
    card_radius = card_diameter / 2.0
    radii = [math.hypot(p["cx"] - cx, p["cy"] - cy) / card_radius for p in placements]
    avg_r = sum(radii) / len(radii)
    min_r = min(radii)
    max_r = max(radii)
    std_r = (sum((x - avg_r) ** 2 for x in radii) / len(radii)) ** 0.5
    inner_cnt = sum(1 for r in radii if r < 0.33)
    mid_cnt = sum(1 for r in radii if 0.33 <= r < 0.66)
    outer_cnt = sum(1 for r in radii if r >= 0.66)
    outer_ratio = outer_cnt / len(radii)
    ring_likeness = 0.5 * max(0.0, (0.25 - std_r)) / 0.25 + 0.5 * max(0.0, outer_ratio - 0.5) / 0.5
    return min_r, avg_r, max_r, std_r, inner_cnt, mid_cnt, outer_cnt, ring_likeness


def banded_greedy_place(
    images: List[Image.Image],
    card_diameter: int,
    canvas_size: int,
    min_frac: float,
    max_frac: float,
    pad_frac: float,
    size_fracs: List[float],
    band_counts: tuple[int, int, int] = BANDED_GREEDY_BAND_COUNTS,
    max_restarts: int = BANDED_GREEDY_MAX_RESTARTS,
    attempts_per_symbol: int = BANDED_GREEDY_ATTEMPTS_PER_SYMBOL,
) -> List[dict]:
    cx = canvas_size / 2.0
    cy = canvas_size / 2.0
    card_radius = card_diameter / 2.0

    indices = list(range(len(images)))
    # Sort by size descending so large icons place first
    indices.sort(key=lambda i: size_fracs[i] if size_fracs is not None else 0.0, reverse=True)

    # Build band assignment list
    inner_n, mid_n, outer_n = band_counts
    bands: List[str] = ["inner"] * inner_n + ["mid"] * mid_n + ["outer"] * outer_n
    if len(bands) != len(images):
        raise ValueError("band_counts must sum to 8")
    random.shuffle(bands)

    def band_ranges(max_center_radius: float, band: str) -> tuple[float, float]:
        if band == "inner":
            return 0.0, max(0.0, BANDED_GREEDY_BAND_INNER_MAX * max_center_radius)
        if band == "mid":
            lo = max(0.0, BANDED_GREEDY_BAND_MID_LO * max_center_radius)
            hi = max(0.0, BANDED_GREEDY_BAND_MID_HI * max_center_radius)
            if lo > hi:
                lo, hi = BANDED_GREEDY_BAND_MID_FALLBACK_LO * max_center_radius, BANDED_GREEDY_BAND_MID_FALLBACK_HI * max_center_radius
            return lo, hi
        # outer
        return max(0.0, BANDED_GREEDY_BAND_OUTER_LO * max_center_radius), max(0.0, BANDED_GREEDY_BAND_OUTER_HI * max_center_radius)

    gap_px_base = pad_frac * card_diameter

    for restart in range(max_restarts):
        placements: List[dict] = []
        centers: List[tuple[float, float]] = []
        radii_list: List[float] = []
        gap_px = max(2.0, gap_px_base * (BANDED_GREEDY_GAP_SHRINK_PER_RESTART ** restart))
        scale_adjust = 1.0 * (BANDED_GREEDY_SCALE_SHRINK_PER_RESTART ** restart)

        for pos, idx in enumerate(indices):
            img = images[idx]
            w, h = img.size
            base_frac = max(min_frac, min(max_frac, size_fracs[idx]))
            # Make the jitter range wider for more size difference
            target_frac = base_frac * scale_adjust * random.uniform(0.65, 1.50)
            s = (target_frac * card_diameter) / max(w, h)
            sw = int(round(w * s))
            sh = int(round(h * s))
            r_sym = 0.5 * math.hypot(sw, sh) * (1.0 - ICON_OVERLAP_FRAC * 0.5)
            if r_sym >= card_radius - 2:
                # too big; reduce scale and continue
                shrink = (card_radius - 4) / (r_sym if r_sym > 0 else (card_radius - 4))
                sw = int(round(sw * shrink))
                sh = int(round(sh * shrink))
                r_sym = 0.5 * math.hypot(sw, sh) * (1.0 - ICON_OVERLAP_FRAC * 0.5)

            band = bands[pos]
            max_center_R = max(0.0, card_radius - r_sym)
            r_min, r_max = band_ranges(max_center_R, band)

            placed = False
            for _ in range(attempts_per_symbol):
                dx, dy = uniform_point_in_annulus(r_min, r_max)
                cx_i = cx + dx
                cy_i = cy + dy
                # overlap check - allow some overlap
                ok = True
                for (px, py), pr in zip(centers, radii_list):
                    dist = math.hypot(cx_i - px, cy_i - py)
                    # Allow overlap up to ICON_OVERLAP_FRAC of the smaller radius
                    min_radius = min(r_sym, pr)
                    overlap_threshold = min_radius * ICON_OVERLAP_FRAC
                    if dist < (r_sym + pr - overlap_threshold):
                        ok = False
                        break
                if ok:
                    centers.append((cx_i, cy_i))
                    radii_list.append(r_sym)
                    placements.append(
                        {
                            "index": idx,
                            "scale": 0.0,
                            "rotation": random.random() * 360.0,
                            "cx": cx_i,
                            "cy": cy_i,
                            "radius": r_sym,
                            "scaled_w": sw,
                            "scaled_h": sh,
                        }
                    )
                    placed = True
                    break

            if not placed:
                break  # restart

        if len(placements) == len(images):
            # Randomize draw order for aesthetics
            random.shuffle(placements)
            return placements

    raise RuntimeError("Banded greedy placement failed")


def force_place_symbols(
    images: List[Image.Image],
    card_diameter: int,
    canvas_size: int,
    min_frac: float,
    max_frac: float,
    pad_frac: float,
    max_outer_tries: int = FORCE_PLACE_MAX_OUTER_TRIES,
    max_iters: int = FORCE_PLACE_MAX_ITERS,
    size_fracs: List[float] | None = None,
) -> List[dict]:
    cx = canvas_size / 2.0
    cy = canvas_size / 2.0
    card_radius = card_diameter / 2.0

    img_sizes: List[Tuple[int, int]] = [img.size for img in images]

    # Per-image target fractions (use provided sizes if available)
    if size_fracs is not None:
        target_fracs = [max(min_frac, min(max_frac, f)) for f in size_fracs]
    else:
        target_fracs = [random.uniform(min_frac, max_frac) for _ in images]

    def build_scaled(scale_factor: float) -> Tuple[List[Tuple[int, int]], List[float]]:
        scaled: List[Tuple[int, int]] = []
        radii: List[float] = []
        gap_px = pad_frac * card_diameter
        for (w, h), frac in zip(img_sizes, target_fracs):
            # Make the jitter range wider for more size difference
            s = (frac * scale_factor * card_diameter) / max(w, h)
            sw = int(round(w * s))
            sh = int(round(h * s))
            scaled.append((sw, sh))
            radii.append(0.5 * math.hypot(sw, sh) * (1.0 - ICON_OVERLAP_FRAC * 0.5))
        return scaled, radii

    scale_factor = 1.0

    for _attempt in range(max_outer_tries):
        scaled_sizes, radii = build_scaled(scale_factor)

        # Initialize centers within allowable region for each circle
        centers: List[Tuple[float, float]] = []
        for r_i in radii:
            max_center_R = max(0.0, card_radius - r_i - FORCE_PLACE_BOUNDARY_PADDING)
            dx, dy = uniform_point_in_circle(max_center_R)
            centers.append((cx + dx, cy + dy))

        # Force-directed relaxation
        ok = False
        for _ in range(max_iters):
            moved = False
            # Pairwise repulsion to resolve overlaps
            for i in range(len(centers)):
                xi, yi = centers[i]
                for j in range(i + 1, len(centers)):
                    xj, yj = centers[j]
                    dx = xj - xi
                    dy = yj - yi
                    dist = math.hypot(dx, dy)
                    min_dist = radii[i] + radii[j]
                    if dist < 1e-6:
                        # Random tiny nudge to break symmetry
                        dx = random.uniform(-1.0, 1.0)
                        dy = random.uniform(-1.0, 1.0)
                        dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        # Overlap penetration resolution vector
                        overlap = (min_dist - dist)
                        nx = dx / dist
                        ny = dy / dist
                        # Distribute movement between both
                        move = FORCE_PLACE_OVERLAP_RESOLVE * overlap
                        xi -= nx * move * (radii[j] / (radii[i] + radii[j]))
                        yi -= ny * move * (radii[j] / (radii[i] + radii[j]))
                        xj += nx * move * (radii[i] / (radii[i] + radii[j]))
                        yj += ny * move * (radii[i] / (radii[i] + radii[j]))
                        centers[i] = (xi, yi)
                        centers[j] = (xj, yj)
                        moved = True

            # Keep within circle boundary
            for i in range(len(centers)):
                xi, yi = centers[i]
                dx = xi - cx
                dy = yi - cy
                dist_c = math.hypot(dx, dy)
                max_c = card_radius - radii[i] - FORCE_PLACE_BOUNDARY_PADDING
                if dist_c > max_c and dist_c > 0.0:
                    pull = (dist_c - max_c)
                    xi -= (dx / dist_c) * pull
                    yi -= (dy / dist_c) * pull
                    centers[i] = (xi, yi)
                    moved = True

            # Mild dispersion to avoid collapsing into a ring: push away from average radius
            avg_r = sum(math.hypot(x - cx, y - cy) for (x, y) in centers) / len(centers)
            for i in range(len(centers)):
                xi, yi = centers[i]
                ri = math.hypot(xi - cx, yi - cy)
                if ri > 0.0:
                    # Small radial adjustment toward balanced distribution
                    adj = FORCE_PLACE_RADIAL_ADJ * (avg_r - ri)
                    xi += (xi - cx) / ri * adj
                    yi += (yi - cy) / ri * adj
                    centers[i] = (xi, yi)

            # Check if overlaps resolved - allow some overlap
            any_overlap = False
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                    # Allow overlap up to ICON_OVERLAP_FRAC of the smaller radius
                    min_radius = min(radii[i], radii[j])
                    overlap_threshold = min_radius * ICON_OVERLAP_FRAC
                    if dist < (radii[i] + radii[j] - overlap_threshold):
                        any_overlap = True
                        break
                if any_overlap:
                    break
            if not any_overlap and not moved:
                ok = True
                break

        if ok:
            # Build placements with random rotations; draw order randomized
            order = list(range(len(images)))
            random.shuffle(order)
            placements: List[dict] = []
            for k in order:
                sw, sh = scaled_sizes[k]
                placements.append(
                    {
                        "index": k,
                        "scale": 0.0,
                        "rotation": random.random() * 360.0,
                        "cx": centers[k][0],
                        "cy": centers[k][1],
                        "radius": radii[k],
                        "scaled_w": sw,
                        "scaled_h": sh,
                    }
                )
            return placements

        # If not ok, shrink globally and retry
        scale_factor *= FORCE_PLACE_SHRINK_PER_FAIL

    raise RuntimeError("Force placement failed")


def try_place_templates(
    images: List[Image.Image],
    card_diameter: int,
    canvas_size: int,
    min_frac: float,
    max_frac: float,
    pad_frac: float,
    template_tries: int = TEMPLATE_TRIES,
    jitter_tries_per_template: int = TEMPLATE_JITTER_TRIES_PER_TEMPLATE,
    size_fracs: List[float] | None = None,
) -> List[dict]:
    # Template-based placement that yields non-ring, realistic layouts
    cx = cy = canvas_size / 2.0
    card_radius = card_diameter / 2.0

    templates: List[List[Tuple[int, float]]] = TEMPLATES

    indices = list(range(len(images)))

    def compute_scaled_sizes(scale_global: float) -> Tuple[List[Tuple[int, int]], List[float]]:
        scaled_sizes: List[Tuple[int, int]] = []
        radii: List[float] = []
        for i, img in enumerate(images):
            w, h = img.size
            # Per-image size: use provided size_fracs with small jitter or random
            if size_fracs is not None:
                base_frac = max(min_frac, min(max_frac, size_fracs[i]))
                # Make the jitter range wider for more size difference
                target_frac = base_frac * random.uniform(0.65, 1.50)
            else:
                target_frac = random.uniform(min_frac, max_frac)
            target_frac *= scale_global
            s = (target_frac * card_diameter) / max(w, h)
            sw = int(round(w * s))
            sh = int(round(h * s))
            scaled_sizes.append((sw, sh))
            r = 0.5 * math.hypot(sw, sh) * (1.0 - ICON_OVERLAP_FRAC * 0.5)
            radii.append(r)
        return scaled_sizes, radii

    def angles_for_count(n: int, base_offset: float) -> List[float]:
        # Evenly spaced, with random jitter per angle
        angles = []
        for i in range(n):
            a = base_offset + (2.0 * math.pi * i) / n
            # jitter up to +/- TEMPLATE_ANGLE_JITTER_DEG degrees
            a += math.radians(random.uniform(-TEMPLATE_ANGLE_JITTER_DEG, TEMPLATE_ANGLE_JITTER_DEG))
            angles.append(a)
        return angles

    for _ in range(template_tries):
        random.shuffle(indices)
        template = random.choice(templates)

        # Gradually shrink if it doesn't fit
        scale_global = 1.0
        for _scale_step in range(TEMPLATE_SCALE_STEPS):
            scaled_sizes, radii = compute_scaled_sizes(scale_global)

            # Pre-assign which image goes to which slot next
            assignment: List[int] = []
            for count, _rf in template:
                for _ in range(count):
                    if not indices:
                        indices = list(range(len(images)))
                        random.shuffle(indices)
                    assignment.append(indices.pop())

            # Try multiple jitter attempts of angles and radial tweaks
            for _j in range(jitter_tries_per_template):
                centers: List[Tuple[float, float]] = []
                k = 0
                ok = True
                for (count, rf) in template:
                    # Choose slight ring radius jitter per slot
                    base_R = rf * card_radius
                    base_offset = random.random() * 2.0 * math.pi
                    angles = angles_for_count(count, base_offset)
                    for i in range(count):
                        idx = assignment[k]
                        r_i = radii[idx]
                        # Ensure against edge: clamp center radius
                        max_center_R = max(0.0, card_radius - r_i - 2.0)
                        # ring radius with jitter
                        ring_R = min(max_center_R, base_R + random.uniform(-TEMPLATE_RING_RADIUS_JITTER, TEMPLATE_RING_RADIUS_JITTER) * card_radius)
                        # If base ring exceeds allowable, pull inward
                        ring_R = max(0.0, min(max_center_R, ring_R))
                        theta = angles[i]
                        cx_i = cx + ring_R * math.cos(theta)
                        cy_i = cy + ring_R * math.sin(theta)

                        # Check against already placed - allow some overlap
                        for j in range(len(centers)):
                            prev_idx = assignment[j]
                            dist = math.hypot(cx_i - centers[j][0], cy_i - centers[j][1])
                            # Allow overlap up to ICON_OVERLAP_FRAC of the smaller radius
                            min_radius = min(r_i, radii[prev_idx])
                            overlap_threshold = min_radius * ICON_OVERLAP_FRAC
                            if dist < (r_i + radii[prev_idx] - overlap_threshold):
                                ok = False
                                break
                        if not ok:
                            break
                        centers.append((cx_i, cy_i))
                        k += 1
                    if not ok:
                        break

                if ok and len(centers) == CARD_IMAGE_COUNT:
                    # Build placements with rotations
                    placements: List[dict] = []
                    draw_order = list(range(CARD_IMAGE_COUNT))
                    random.shuffle(draw_order)
                    for pos in draw_order:
                        idx = assignment[pos]
                        sw, sh = scaled_sizes[idx]
                        placements.append(
                            {
                                "index": idx,
                                "scale": 0.0,
                                "rotation": random.random() * 360.0,
                                "cx": centers[pos][0],
                                "cy": centers[pos][1],
                                "radius": radii[idx],
                                "scaled_w": sw,
                                "scaled_h": sh,
                            }
                        )
                    return placements

            # Shrink globally and try again
            scale_global *= TEMPLATE_SHRINK_PER_FAIL

    raise RuntimeError("Template placement failed")


def fallback_place_ring(
    images: List[Image.Image],
    card_diameter: int,
    canvas_size: int,
    base_frac: float = FALLBACK_RING_BASE_FRAC,
    size_jitter_frac: float = FALLBACK_RING_SIZE_JITTER_FRAC,
    pad_frac: float = FALLBACK_RING_PAD_FRAC,
    angle_jitter_deg: float = FALLBACK_RING_ANGLE_JITTER_DEG,
    max_outer_loops: int = FALLBACK_RING_MAX_OUTER_LOOPS,
    size_fracs: List[float] | None = None,
) -> List[dict]:
    cx = cy = canvas_size / 2.0
    card_radius = card_diameter / 2.0

    # Initial per-image target size with small randomness
    img_dims: List[Tuple[int, int]] = [img.size for img in images]
    per_img_frac: List[float] = []
    for i, _ in enumerate(images):
        if size_fracs is not None:
            base = size_fracs[i]
        else:
            base = base_frac
        # Make the jitter range wider for more size difference
        jitter = 1.0 + random.uniform(-0.22, 0.22)
        per_img_frac.append(base * jitter)

    # Compute scaled sizes and radii
    def compute_scaled_and_radii(scale_factor: float) -> Tuple[List[Tuple[int, int]], List[float]]:
        scaled_sizes: List[Tuple[int, int]] = []
        radii: List[float] = []
        gap_px = pad_frac * card_diameter
        for (w, h), frac in zip(img_dims, per_img_frac):
            target_frac = max(0.08, frac * scale_factor)
            s = (target_frac * card_diameter) / max(w, h)
            sw = int(round(w * s))
            sh = int(round(h * s))
            scaled_sizes.append((sw, sh))
            r = 0.5 * math.hypot(sw, sh) * (1.0 - ICON_OVERLAP_FRAC * 0.5)
            radii.append(r)
        return scaled_sizes, radii

    scale_factor = 1.0
    angles_base = [i * (2.0 * math.pi / CARD_IMAGE_COUNT) for i in range(CARD_IMAGE_COUNT)]

    for _ in range(max_outer_loops):
        scaled_sizes, radii = compute_scaled_and_radii(scale_factor)
        max_r = max(radii)
        center_R = max(FALLBACK_RING_MIN_CENTER_R, card_radius - max_r)
        chord = 2.0 * center_R * math.sin(math.pi / CARD_IMAGE_COUNT)
        required_adjacent = max(radii[i] + radii[(i + 1) % CARD_IMAGE_COUNT] for i in range(CARD_IMAGE_COUNT))

        if chord + 0.1 >= required_adjacent:
            # Try jittered angles to break symmetry and avoid rare overlaps with bigger neighbors
            for _try in range(60):
                theta0 = random.random() * 2.0 * math.pi
                angle_jitter_rad = math.radians(angle_jitter_deg)
                thetas = [
                    (theta0 + base + random.uniform(-angle_jitter_rad, angle_jitter_rad)) % (2.0 * math.pi)
                    for base in angles_base
                ]
                # Sort to maintain order but keep relative jitter
                thetas.sort()

                centers: List[Tuple[float, float]] = [
                    (cx + center_R * math.cos(t), cy + center_R * math.sin(t)) for t in thetas
                ]

                # Pairwise overlap check - allow some overlap
                ok = True
                for i in range(CARD_IMAGE_COUNT):
                    for j in range(i + 1, CARD_IMAGE_COUNT):
                        dist = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                        # Allow overlap up to ICON_OVERLAP_FRAC of the smaller radius
                        min_radius = min(radii[i], radii[j])
                        overlap_threshold = min_radius * ICON_OVERLAP_FRAC
                        if dist < (radii[i] + radii[j] - overlap_threshold):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    # Compose placements with random rotations
                    placements: List[dict] = []
                    order = list(range(CARD_IMAGE_COUNT))
                    random.shuffle(order)
                    for k in order:
                        sw, sh = scaled_sizes[k]
                        placements.append(
                            {
                                "index": k,
                                "scale": 0.0,  # not used downstream except sizes
                                "rotation": random.random() * 360.0,
                                "cx": centers[k][0],
                                "cy": centers[k][1],
                                "radius": radii[k],
                                "scaled_w": sw,
                                "scaled_h": sh,
                            }
                        )
                    return placements

        # Not enough room between neighbors: shrink globally and retry
        scale_factor *= FALLBACK_RING_SHRINK_PER_FAIL

    # As a last resort, place with even smaller base size
    scaled_sizes, radii = compute_scaled_and_radii(FALLBACK_RING_LAST_RESORT_SCALE)
    max_r = max(radii)
    center_R = max(FALLBACK_RING_MIN_CENTER_R, card_radius - max_r)
    theta0 = random.random() * 2.0 * math.pi
    thetas = [(theta0 + i * (2.0 * math.pi / CARD_IMAGE_COUNT)) % (2.0 * math.pi) for i in range(CARD_IMAGE_COUNT)]
    centers = [(cx + center_R * math.cos(t), cy + center_R * math.sin(t)) for t in thetas]
    placements: List[dict] = []
    order = list(range(CARD_IMAGE_COUNT))
    random.shuffle(order)
    for k in order:
        sw, sh = scaled_sizes[k]
        placements.append(
            {
                "index": k,
                "scale": 0.0,
                "rotation": random.random() * 360.0,
                "cx": centers[k][0],
                "cy": centers[k][1],
                "radius": radii[k],
                "scaled_w": sw,
                "scaled_h": sh,
            }
        )
    return placements


def generate_card(
    image_paths: List[str],
    output_path: str,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    card_diameter: int | None = None,
    background_color: str = DEFAULT_BG_COLOR,
    seed: int | None = None,
    verbose: bool = False,
    debug_overlay_path: str | None = None,
    log_file_path: str | None = None,
) -> str:
    if seed is not None:
        random.seed(seed)

    if card_diameter is None:
        card_diameter = int(canvas_size * DEFAULT_CARD_DIAM_FRAC)

    if len(image_paths) != CARD_IMAGE_COUNT:
        raise ValueError(f"Exactly {CARD_IMAGE_COUNT} image paths are required.")

    images = load_images(image_paths)

    base_min_frac = BASE_MIN_FRAC
    base_max_frac = BASE_MAX_FRAC
    base_pad_frac = BASE_PAD_FRAC

    canvas = draw_card_base(canvas_size, card_diameter, background_color)

    # Attempt multiple progressively relaxed layouts for robustness
    attempts_config = ATTEMPTS_CONFIG.copy()

    last_error: Exception | None = None
    placements: List[dict] | None = None
    # Create explicit per-image size fractions to achieve visible size diversity
    size_fracs: List[float] = SIZE_FRACS_TEMPLATE.copy()
    random.shuffle(size_fracs)
    # Assign radial bands to reduce ring formation: 2 inner, 3 mid, 3 outer
    bands: List[str] = BANDS_TEMPLATE.copy()
    random.shuffle(bands)

    placement_method = ""
    for (min_frac, max_frac, pad_frac) in attempts_config:
        try:
            # First try free-form stochastic placement
            placements = try_place_symbols(
                images=images,
                card_diameter=card_diameter,
                canvas_size=canvas_size,
                min_frac=min_frac,
                max_frac=max_frac,
                pad_frac=pad_frac,
                max_attempts_per_symbol=PLACEMENT_ATTEMPTS_PER_SYMBOL_MAX,
                size_fracs=size_fracs,
                inner_bias_count=0,
                inner_bias_scale=0.45,
                bands=bands,
            )
            placement_method = "stochastic"
            break
        except Exception as e:  # catch and relax
            last_error = e
            continue

    if placements is None:
        # Auto-relaxation loop: shrink sizes/padding a bit across steps to avoid failure,
        # trying non-ring strategies in order: templates -> banded -> force.
        relax_steps = RELAX_STEPS
        for s_min, s_max, s_pad in relax_steps:
            try:
                placements = try_place_templates(
                    images=images,
                    card_diameter=card_diameter,
                    canvas_size=canvas_size,
                    min_frac=base_min_frac * s_min,
                    max_frac=base_max_frac * s_max,
                    pad_frac=base_pad_frac * s_pad,
                    size_fracs=size_fracs,
                )
                placement_method = f"template(relax={s_min:.2f}/{s_max:.2f})"
                break
            except Exception:
                try:
                    placements = banded_greedy_place(
                        images=images,
                        card_diameter=card_diameter,
                        canvas_size=canvas_size,
                        min_frac=base_min_frac * s_min,
                        max_frac=base_max_frac * s_max,
                        pad_frac=base_pad_frac * s_pad,
                        size_fracs=size_fracs,
                        band_counts=(3, 3, 2),
                        max_restarts=12,
                        attempts_per_symbol=1200,
                    )
                    placement_method = f"banded(relax={s_min:.2f}/{s_max:.2f})"
                    break
                except Exception:
                    try:
                        placements = force_place_symbols(
                            images=images,
                            card_diameter=card_diameter,
                            canvas_size=canvas_size,
                            min_frac=base_min_frac * s_min,
                            max_frac=base_max_frac * s_max,
                            pad_frac=base_pad_frac * s_pad,
                            max_outer_tries=12,
                            max_iters=2200,
                            size_fracs=size_fracs,
                        )
                        placement_method = f"force(relax={s_min:.2f}/{s_max:.2f})"
                        break
                    except Exception:
                        continue
        if placements is None:
            # As a last resort, allow ring fallback to ensure output, but only if user hasn't disabled it
            placements = fallback_place_ring(
                images=images,
                card_diameter=card_diameter,
                canvas_size=canvas_size,
                size_fracs=size_fracs,
            )
            placement_method = "ring-fallback(last-resort)"

    composed = render_card(images, placements, canvas)

    # Optional debug overlay
    if debug_overlay_path is not None:
        overlay = render_debug_overlay(canvas_size, card_diameter, placements)
        # Save overlay on transparent background
        overlay.save(debug_overlay_path)

    if verbose:
        # Detailed logs of placements to analyze ring-like layouts
        cx = canvas_size / 2.0
        cy = canvas_size / 2.0
        card_radius = card_diameter / 2.0

        log_lines: List[str] = []
        log_lines.append(f"Placement method: {placement_method}")

        rows: list[tuple[int, float, float, float, float]] = []
        for i, p in enumerate(placements):
            dx = p["cx"] - cx
            dy = p["cy"] - cy
            r = math.hypot(dx, dy)
            r_norm = r / card_radius
            theta_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            size_frac = max(p["scaled_w"], p["scaled_h"]) / card_diameter
            rows.append((i, r_norm, theta_deg, size_frac, p["rotation"]))

        log_lines.append("Icon placements (index r_norm theta_deg size_frac rot_deg):")
        for (i, r_norm, theta_deg, size_frac, rot_deg) in rows:
            log_lines.append(f"  {i:02d}  r={r_norm:0.3f}  θ={theta_deg:06.2f}°  size={size_frac:0.3f}  rot={rot_deg:06.2f}°")

        radii = [r for (_i, r, _t, _s, _rot) in rows]
        avg_r = sum(radii) / len(radii)
        min_r = min(radii)
        max_r = max(radii)
        std_r = (sum((x - avg_r) ** 2 for x in radii) / len(radii)) ** 0.5
        inner_cnt = sum(1 for r in radii if r < 0.33)
        mid_cnt = sum(1 for r in radii if 0.33 <= r < 0.66)
        outer_cnt = sum(1 for r in radii if r >= 0.66)

        # Heuristic ring-likeness: many are in outer band and low radial std
        outer_ratio = outer_cnt / len(radii)
        ring_likeness = 0.5 * max(0.0, (0.25 - std_r)) / 0.25 + 0.5 * max(0.0, outer_ratio - 0.5) / 0.5

        log_lines.append(f"Radial stats: min={min_r:.3f}  avg={avg_r:.3f}  max={max_r:.3f}  std={std_r:.3f}")
        log_lines.append(f"Band counts: inner={inner_cnt}  mid={mid_cnt}  outer={outer_cnt}")
        log_lines.append(f"Ring-likeness score (0..1): {ring_likeness:.2f}")

        for line in log_lines:
            print(line)
        if log_file_path:
            out_dir = os.path.dirname(log_file_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    composed.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Spot It-style card from 8 images.")
    parser.add_argument(
        "images",
        nargs="*",
        help=f"{CARD_IMAGE_COUNT} image file paths. If omitted, will try to read exactly {CARD_IMAGE_COUNT} files from ./images",
    )
    parser.add_argument("--out", dest="out", default=os.path.join("output", "card.png"), help="Output image path (PNG recommended)")
    parser.add_argument("--size", dest="size", type=int, default=DEFAULT_CANVAS_SIZE, help="Canvas size in pixels (square)")
    parser.add_argument("--card", dest="card", type=int, default=None, help=f"Card diameter in pixels (defaults to {int(DEFAULT_CARD_DIAM_FRAC*100)}%% of canvas size)")
    parser.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed for reproducibility (omit for randomness)")
    parser.add_argument("--bg", dest="bg", default=DEFAULT_BG_COLOR, help="Card background color (hex)")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print placement method and diagnostics")
    parser.add_argument("--debug-overlay", dest="debug_overlay", default=None, help="Optional path to save a debug overlay PNG")
    parser.add_argument("--log-file", dest="log_file", default=None, help="Optional path to save verbose placement logs")
    return parser.parse_args()


def discover_images_if_needed(paths: List[str]) -> List[str]:
    if paths:
        return paths
    default_dir = os.path.join("images")
    if not os.path.isdir(default_dir):
        raise FileNotFoundError("No images provided and ./images directory not found.")
    # Pick exactly CARD_IMAGE_COUNT files in a deterministic order but placement remains random
    candidates: List[str] = []
    for name in os.listdir(default_dir):
        lower = name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            candidates.append(os.path.join(default_dir, name))
    candidates.sort()
    if len(candidates) != CARD_IMAGE_COUNT:
        raise ValueError(f"Expected exactly {CARD_IMAGE_COUNT} images in ./images, found {len(candidates)}")
    return candidates


def main() -> None:
    args = parse_args()
    image_paths = discover_images_if_needed(args.images)
    out_path = generate_card(
        image_paths=image_paths,
        output_path=args.out,
        canvas_size=args.size,
        card_diameter=args.card,
        background_color=args.bg,
        seed=args.seed,
        verbose=args.verbose,
        debug_overlay_path=args.debug_overlay,
        log_file_path=args.log_file,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
