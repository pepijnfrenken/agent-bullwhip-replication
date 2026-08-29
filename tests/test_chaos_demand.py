"""Tests for the chaotic + wild demand patterns (the crossover experiment)."""
import random
from agent_bullwhip.engine import make_demand


def test_chaotic_pattern_reproducible_and_bounded():
    d1 = make_demand(36, "chaotic", seed=1000)
    d2 = make_demand(36, "chaotic", seed=1000)
    assert d1 == d2, "chaotic must be seed-reproducible"
    d3 = make_demand(36, "chaotic", seed=1001)
    assert d1 != d3, "different seed -> different path"
    assert all(1 <= x <= 40 for x in d1)
    assert len(d1) == 36


def test_wild_pattern_reproducible_and_bounded():
    d1 = make_demand(36, "wild", seed=1000)
    d2 = make_demand(36, "wild", seed=1000)
    assert d1 == d2, "wild must be seed-reproducible"
    d3 = make_demand(36, "wild", seed=1001)
    assert d1 != d3
    assert all(1 <= x <= 60 for x in d1)
    assert len(d1) == 36


def test_chaotic_and_wild_are_more_volatile_than_noisy():
    """Sanity: the chaos arms should have higher variance than the mild noisy arm."""
    import statistics
    def sd(pattern, seed):
        return statistics.pstdev(make_demand(72, pattern, seed=seed))
    noisy_sd = sd("noisy", 1000)
    chaotic_sd = sd("chaotic", 1000)
    wild_sd = sd("wild", 1000)
    assert chaotic_sd > noisy_sd
    assert wild_sd > chaotic_sd
