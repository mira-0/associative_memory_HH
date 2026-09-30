import numpy as np

def random_patterns(
        n_patterns: int, n_blocks: int, active_blocks: int, *, include_default: bool = True, rng: np.random.Generator | None = None
    ) -> np.ndarray:
    
    if n_patterns <= 0:
        raise ValueError("n_patterns must be positive.")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")
    if not 0 <= active_blocks <= n_blocks:
        raise ValueError("active_blocks must be between 0 and n_blocks.")
    if rng is None:
        rng = np.random.default_rng()
    
    patterns = np.zeros((n_patterns, n_blocks), dtype=np.int8,)
    start = 0
    
    if include_default:
        patterns[0, :active_blocks] = 1
        start = 1
    
    for pattern_index in range(start, n_patterns):
        active_indices = rng.choice(n_blocks, active_blocks, replace=False)
        patterns[pattern_index, active_indices] = 1
    
    return patterns


def flip_bits(pattern, n_flips: int, *, rng: np.random.Generator | None = None):
    
    pattern = np.asarray(pattern)
    
    if pattern.ndim != 1:
        raise ValueError("pattern must be one-dimensional.")
    if not np.all(np.isin(pattern, [0, 1])):
        raise ValueError("pattern must contain only 0 and 1.")
    if n_flips < 0:
        raise ValueError("n_flips cannot be negative.")

    one_indices = np.flatnonzero(pattern == 1)
    zero_indices = np.flatnonzero(pattern == 0)

    if n_flips > min(len(one_indices), len(zero_indices)):
        raise ValueError("Not enough active or inactive entries.")
    if rng is None:
        rng = np.random.default_rng()

    removed_indices = rng.choice(one_indices, n_flips, replace=False)
    added_indices = rng.choice(zero_indices, n_flips, replace=False)

    corrupted_pattern = pattern.copy()
    corrupted_pattern[removed_indices] = 0
    corrupted_pattern[added_indices] = 1

    return (corrupted_pattern, added_indices, removed_indices)


def remove_bits(pattern, n_removals: int, *, rng: np.random.Generator | None = None):
    pattern = np.asarray(pattern)

    if pattern.ndim != 1:
        raise ValueError("pattern must be one-dimensional.")
    if not np.all(np.isin(pattern, [0, 1])):
        raise ValueError("pattern must contain only 0 and 1.")

    active_indices = np.flatnonzero(pattern == 1)

    if not 0 <= n_removals <= len(active_indices):
        raise ValueError("n_removals exceeds the number of active entries.")

    if rng is None:
        rng = np.random.default_rng()

    removed_indices = rng.choice(
        active_indices,
        n_removals,
        replace=False,
    )

    corrupted_pattern = pattern.copy()
    corrupted_pattern[removed_indices] = 0

    return corrupted_pattern, removed_indices