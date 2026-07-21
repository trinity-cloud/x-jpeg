import numpy as np
import pytest

from xjpeg.oracle import allocate_sampled_rd, concave_hull_indices


def test_concave_hull_discards_duplicates_dominated_and_convex_points():
    rate = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    quality = np.asarray([0.10, 0.20, 0.19, 0.19, 0.50, 0.55])
    # rate=2 duplicate index 2 and rate=3 dominated index 3 disappear.  The
    # rate=2 point is below the chord from rate=1 to rate=4 and also disappears.
    assert concave_hull_indices(rate, quality).tolist() == [0, 4, 5]


def test_allocation_prefers_highest_marginal_gain_and_respects_budget():
    rate = np.asarray([[0.5, 1.0], [0.5, 1.0]])
    quality = np.asarray([[0.5, 1.0], [0.5, 0.8]])
    result = allocate_sampled_rd(rate, quality, target_mean_rate=0.75)

    assert result.selection.tolist() == [1, 0]
    assert result.total_rate == pytest.approx(1.5)
    assert result.total_quality == pytest.approx(1.5)
    assert result.fractional_total_quality == pytest.approx(1.5)


def test_fractional_relaxation_is_tighter_than_unspendable_discrete_budget():
    rate = np.asarray([[0.5, 1.1], [0.5, 1.1]])
    quality = np.asarray([[0.5, 1.0], [0.5, 0.8]])
    result = allocate_sampled_rd(rate, quality, target_mean_rate=0.65)

    assert result.total_rate == pytest.approx(1.0)
    assert result.total_quality == pytest.approx(1.0)
    assert result.fractional_total_quality == pytest.approx(1.25)
    assert result.fractional_segment == (0, 0, 1, pytest.approx(0.5))


def test_allocation_rejects_unreachable_budget():
    with pytest.raises(ValueError, match="cannot meet the rate budget"):
        allocate_sampled_rd(
            np.asarray([[0.9, 1.0]]),
            np.asarray([[0.8, 0.9]]),
            target_mean_rate=0.8,
        )
