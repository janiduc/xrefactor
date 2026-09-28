"""
Shared train/val/test split utility.

Reads split ratios from the same `data.*` config keys every training script
already expects to matter (config.yaml's `data:` block) but that, before this
module existed, no script actually consumed - each one invented its own
inline split logic instead. New training scripts should call
`split_indices` rather than re-implementing a split.
"""

from typing import Optional, Tuple

import torch

from src.utils.helpers import ConfigManager


def split_indices(
    num_examples: int,
    config: ConfigManager,
    seed_override: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split `num_examples` indices into (train, val, test) using config's data.* ratios.

    Reads `data.train_split`, `data.val_split`, `data.test_split`, `data.random_seed`
    from the given ConfigManager. Falls back to 0.7/0.15/0.15 and seed 42 if any
    of these are missing from config.

    Raises:
        ValueError: if the three ratios don't sum to ~1.0, or num_examples < 3
            (too few to make a non-empty split of each kind).
    """
    train_split = config.get("data.train_split", 0.7)
    val_split = config.get("data.val_split", 0.15)
    test_split = config.get("data.test_split", 0.15)
    seed = seed_override if seed_override is not None else config.get("data.random_seed", 42)

    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"data.train_split + data.val_split + data.test_split must sum to 1.0, got {total}"
        )
    if num_examples < 3:
        raise ValueError(f"Need at least 3 examples to split into train/val/test, got {num_examples}")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_examples, generator=generator)

    train_size = max(1, int(round(num_examples * train_split)))
    val_size = max(1, int(round(num_examples * val_split)))
    # Test gets whatever remains, guaranteeing every index is assigned exactly once.
    train_size = min(train_size, num_examples - 2)
    val_size = min(val_size, num_examples - train_size - 1)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    return train_idx, val_idx, test_idx
