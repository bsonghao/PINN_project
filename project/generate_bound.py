import numpy as np
from typing import List, Tuple, Dict
import os
import torch

def generate_rectangular_bounds(
    dimensions: int,
    num_subdomains: List[int],
    ranges: List[Tuple[float, float]],
    overlap: float = 0.1,
    overlap_mode: str = "absolute"   # "relative" | "absolute"
) -> List[Dict]:
    """
    Generates rectangular subdomains with overlap for arbitrary input dimensions.

    Parameters:
    ----------
    dimensions   : int
        Number of input dimensions (e.g. 2 → 2D, 3 → 3D).
    num_subdomains : int
        Number of subdomains along EACH dimension.
        Total subdomains = num_subdomains ** dimensions.
    ranges       : List[Tuple[float, float]]
        Domain bounds per dimension → [(x_min, x_max), (y_min, y_max), ...].
    overlap      : float
        Overlap amount. Interpreted based on `overlap_mode`.
    overlap_mode : str
        - "relative" → overlap is a fraction of the subdomain width (e.g. 0.1 = 10%).
        - "absolute" → overlap is an absolute value in the same units as `ranges`.

    Returns:
    -------
    List[Dict]: Each dict contains:
        - "core"    : List[Tuple[float,float]] → tight bounds without overlap.
        - "extended": List[Tuple[float,float]] → bounds with overlap added.
        - "center"  : List[float]              → center point of the core subdomain.
        - "width"   : List[float]              → width of the core subdomain per dim.
        - "overlap" : List[float]              → actual overlap amount per dimension.
        - "index"   : Tuple[int, ...]          → N-D grid index of this subdomain.
    """

    # ── Validation ──────────────────────────────────────────────────────────
    if len(ranges) != dimensions:
        raise ValueError(
            f"Expected {dimensions} ranges, but got {len(ranges)}."
        )
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap_mode not in ("relative", "absolute"):
        raise ValueError('overlap_mode must be "relative" or "absolute".')

    # ── Step 1: Compute per-dimension subdomain edges (core) ─────────────────
    # For each dimension, divide [min, max] into `num_subdomains` equal parts
    dim_edges = []
    for dim in range(dimensions):
        lo, hi = ranges[dim]
        edges = np.linspace(lo, hi, num_subdomains[dim] + 1)  # n+1 edges for n subdomains
        dim_edges.append(edges)

    # ── Step 2: Compute overlap per dimension ───────────────────────────────
    # Each subdomain has a core width; overlap extends beyond the core bounds
    dim_core_widths = []
    for dim in range(dimensions):
        lo, hi = ranges[dim]
        core_width = (hi - lo) / num_subdomains[dim]
        dim_core_widths.append(core_width)

    def compute_overlap(dim: int) -> float:
        if overlap_mode == "relative":
            return overlap * dim_core_widths[dim]
        else:
            return overlap

    # ── Step 3: Generate all N-D subdomain index combinations ───────────────
    # Use meshgrid-style iteration over (dim_0_idx, dim_1_idx, ..., dim_N_idx)
    import itertools
    index_ranges = [range(num_subdomains[dim]) for dim in range(dimensions)]
    all_indices   = list(itertools.product(*index_ranges))

    # ── Step 4: Build subdomain dicts ────────────────────────────────────────
    subdomains = []

    for idx_tuple in all_indices:
        core_min        = []
        core_max        = []
        extended_min    = []
        extended_max    = []
        centers         = []
        widths          = []
        overlaps        = []

        for dim, idx in enumerate(idx_tuple):
            lo_core = float(dim_edges[dim][idx])
            hi_core = float(dim_edges[dim][idx + 1])

            ov = compute_overlap(dim)

            # Extended bounds — clipped to the global domain boundary
            global_lo, global_hi = ranges[dim]
            lo_ext = max(lo_core - ov, global_lo)
            hi_ext = min(hi_core + ov, global_hi)

            core_min.append(lo_core)
            core_max.append(hi_core)
            extended_min.append(lo_ext)
            extended_max.append(hi_ext)

            # core_bounds.append((round(lo_core, 10), round(hi_core, 10)))
            # extended_bounds.append((round(lo_ext, 10), round(hi_ext, 10)))

            centers.append(round((lo_ext + hi_ext) / 2.0, 10))
            widths.append(round(hi_ext - lo_ext, 10))
            overlaps.append(round(ov, 10))

        subdomains.append({
            "index"    : idx_tuple,
            "core_max" : torch.tensor(core_max),
            "core_min" : torch.tensor(core_min),
            "extended_max" : torch.tensor(extended_max),
            "extended_min" : torch.tensor(extended_min),
            "center"   : torch.tensor(centers),
            "width"    : torch.tensor(widths),
            "overlap"  : torch.tensor(overlaps),
        })

    return subdomains


# ─────────────────────────────────────────────────────────────────────────────
# Pretty Printer
# ─────────────────────────────────────────────────────────────────────────────
def print_subdomains(subdomains: List[Dict], dimensions: int) -> None:
    total = len(subdomains)
    print(f"\n{'─'*60}")
    print(f"  Total Subdomains : {total}")
    print(f"  Dimensions       : {dimensions}")
    print(f"{'─'*60}\n")

    for sd in subdomains:
        print(f"  Subdomain Index : {sd['index']}")
        for dim in range(dimensions):
            lo_c, hi_c = sd['core_min'][dim].item(), sd["core_max"][dim].item()
            lo_e, hi_e = sd['extended_min'][dim].item(), sd["extended_max"][dim].item()
            ov          = sd['overlap'][dim].item()
            print(f"    Dim {dim}: "
                  f"core=[{lo_c:.4f}, {hi_c:.4f}]  "
                  f"extended=[{lo_e:.4f}, {hi_e:.4f}]  "
                  f"overlap=±{ov:.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Example 1: 1D with relative overlap ──────────────────────────────────
    # print("═" * 60)
    # print("  EXAMPLE 1 — 1D, 4 subdomains, 10% relative overlap")
    # print("═" * 60)
    # sds_1d = generate_rectangular_bounds(
        # dimensions     = 1,
        # num_subdomains = [4],
        # ranges         = [(0.0, 1.0)],
        # overlap        = 0.1,
        # overlap_mode   = "relative"
    # )
    # print_subdomains(sds_1d, dimensions=1)


    # ── Example 2: 2D with relative overlap ──────────────────────────────────
    print("═" * 60)
    print("  EXAMPLE 2 — 2D, 2×3 subdomains, 15% relative overlap")
    print("═" * 60)
    sds_2d = generate_rectangular_bounds(
        dimensions     = 2,
        num_subdomains = [3, 4],
        ranges         = [(0.0, 1.0), (0.0, 2.0)],
        overlap        = 0.15,
        overlap_mode   = "relative"
    )
    print_subdomains(sds_2d, dimensions=2)


    # ── Example 3: 3D with absolute overlap ──────────────────────────────────
    # print("═" * 60)
    # print("  EXAMPLE 3 — 3D, 2×2×4 subdomains, absolute overlap=0.05")
    # print("═" * 60)
    # sds_3d = generate_rectangular_bounds(
        # dimensions     = 3,
        # num_subdomains = [2, 3, 4],
        # ranges         = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        # overlap        = 0.05,
        # overlap_mode   = "absolute"
    # )
    # print_subdomains(sds_3d, dimensions=3)
