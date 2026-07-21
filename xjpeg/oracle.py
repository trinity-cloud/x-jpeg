"""Rate allocation over sampled per-image rate--distortion curves.

The fractional result is an upper bound for the convex relaxation of the
*sampled* points.  It is not an upper bound for JPEG in general, nor for table
scales that were not sampled.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Allocation:
    """A feasible discrete allocation and its sampled-curve relaxation."""

    selection: np.ndarray
    total_rate: float
    total_quality: float
    fractional_total_quality: float
    fractional_segment: tuple[int, int, int, float] | None


def concave_hull_indices(rate, quality) -> np.ndarray:
    """Return sampled points on the increasing upper concave envelope.

    Duplicate-rate and dominated points are discarded.  Returned indices
    refer to the original input arrays and are ordered by increasing rate.
    """

    rate = np.asarray(rate, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if rate.ndim != 1 or quality.ndim != 1 or rate.shape != quality.shape:
        raise ValueError("rate and quality must be equal-length 1D arrays")
    if not len(rate):
        raise ValueError("at least one RD point is required")
    if not np.all(np.isfinite(rate)) or not np.all(np.isfinite(quality)):
        raise ValueError("rate and quality must be finite")

    # For an identical rate, retain the highest-quality point (and use the
    # lowest original index as a deterministic tie break).
    ordered = sorted(range(len(rate)), key=lambda i: (rate[i], -quality[i], i))
    unique = []
    for index in ordered:
        if unique and np.isclose(rate[index], rate[unique[-1]], rtol=0.0, atol=1e-12):
            continue
        unique.append(index)

    # Any point no better than a lower-rate point can never be optimal.
    increasing = []
    best_quality = -np.inf
    for index in unique:
        if quality[index] > best_quality + 1e-15:
            increasing.append(index)
            best_quality = quality[index]

    # Upper concave envelope: marginal quality/rate slopes must decrease.
    hull = []
    for index in increasing:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            slope_ab = (quality[b] - quality[a]) / (rate[b] - rate[a])
            slope_bi = (quality[index] - quality[b]) / (rate[index] - rate[b])
            if slope_ab > slope_bi + 1e-15:
                break
            hull.pop()
        hull.append(index)
    return np.asarray(hull, dtype=np.int64)


def allocate_sampled_rd(rate, quality, target_mean_rate: float) -> Allocation:
    """Allocate a mean rate budget across independent sampled RD curves.

    ``rate`` and ``quality`` have shape ``(num_images, num_points)``.  The
    returned discrete selection is feasible and realizable using sampled
    points.  ``fractional_total_quality`` is the optimum of the upper-concave
    fractional relaxation, and is therefore an upper bound on any discrete
    selection from those sampled points.
    """

    rate = np.asarray(rate, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if rate.ndim != 2 or quality.shape != rate.shape:
        raise ValueError("rate and quality must have the same 2D shape")
    if rate.shape[0] == 0 or rate.shape[1] == 0:
        raise ValueError("at least one image and one RD point are required")
    if not np.isfinite(target_mean_rate) or target_mean_rate <= 0:
        raise ValueError("target_mean_rate must be finite and positive")

    hulls = [concave_hull_indices(r, q) for r, q in zip(rate, quality)]
    selection = np.asarray([hull[0] for hull in hulls], dtype=np.int64)
    positions = np.zeros(len(hulls), dtype=np.int64)
    base_rate = float(rate[np.arange(len(hulls)), selection].sum())
    base_quality = float(quality[np.arange(len(hulls)), selection].sum())
    budget = float(target_mean_rate * len(hulls))
    tolerance = max(1e-12, abs(budget) * 1e-12)
    if base_rate > budget + tolerance:
        raise ValueError(
            f"sampled curves cannot meet the rate budget: minimum mean rate "
            f"is {base_rate / len(hulls):.6f}, target is {target_mean_rate:.6f}"
        )

    # Max-heap of the next marginal segment from each image.  Concavity means
    # that accepting segments in descending slope solves the fractional
    # relaxation.
    heap = []

    def push_next(image: int, position: int, target_heap):
        hull = hulls[image]
        if position + 1 >= len(hull):
            return
        left, right = int(hull[position]), int(hull[position + 1])
        delta_rate = float(rate[image, right] - rate[image, left])
        delta_quality = float(quality[image, right] - quality[image, left])
        slope = delta_quality / delta_rate
        heapq.heappush(
            target_heap,
            (-slope, image, position, left, right, delta_rate, delta_quality),
        )

    for image in range(len(hulls)):
        push_next(image, 0, heap)

    # First compute the exact fractional-relaxation value.
    fractional_heap = list(heap)
    heapq.heapify(fractional_heap)
    fractional_rate = base_rate
    fractional_quality = base_quality
    fractional_positions = np.zeros(len(hulls), dtype=np.int64)
    fractional_segment = None
    while fractional_heap:
        item = heapq.heappop(fractional_heap)
        _, image, position, left, right, delta_rate, delta_quality = item
        if position != fractional_positions[image]:
            continue
        remaining = budget - fractional_rate
        if remaining <= tolerance:
            break
        if delta_rate <= remaining + tolerance:
            fractional_rate += delta_rate
            fractional_quality += delta_quality
            fractional_positions[image] += 1
            push_next(image, position + 1, fractional_heap)
        else:
            fraction = max(0.0, min(1.0, remaining / delta_rate))
            fractional_quality += fraction * delta_quality
            fractional_rate += fraction * delta_rate
            fractional_segment = (image, left, right, fraction)
            break

    # Produce a realizable discrete solution.  A segment too large for the
    # remaining budget is skipped, while smaller later segments may still fit.
    total_rate = base_rate
    total_quality = base_quality
    while heap:
        item = heapq.heappop(heap)
        _, image, position, left, right, delta_rate, delta_quality = item
        if position != positions[image]:
            continue
        if total_rate + delta_rate <= budget + tolerance:
            selection[image] = right
            positions[image] += 1
            total_rate += delta_rate
            total_quality += delta_quality
            push_next(image, position + 1, heap)

    return Allocation(
        selection=selection,
        total_rate=total_rate,
        total_quality=total_quality,
        fractional_total_quality=fractional_quality,
        fractional_segment=fractional_segment,
    )
