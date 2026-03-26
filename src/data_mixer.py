"""Utilities for mixing tokenized domain data according to mixture weights."""
import numpy as np
from pathlib import Path


def load_domain_tokens(domain_dir: str, max_tokens: int | None = None) -> np.ndarray:
    """Load tokenized domain data from a binary file.

    The binary file stores token IDs. We try uint16 first (vocab < 65536),
    falling back to uint32 for larger vocabularies like Dolma 2 (100352).
    """
    bin_path = Path(domain_dir) / "train.npy"
    # Dolma 2 tokenizer has vocab_size=100352, which requires uint32
    tokens = np.load(str(bin_path))
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def create_mixed_batch(
    domain_tokens: dict[str, np.ndarray],
    weights: dict[str, float],
    batch_tokens: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a batch of tokens by sampling from domains according to weights.

    Args:
        domain_tokens: dict mapping domain name to token array
        weights: dict mapping domain name to sampling weight (must sum to ~1)
        batch_tokens: total number of tokens in the batch
        rng: random number generator

    Returns:
        1D array of token IDs
    """
    if rng is None:
        rng = np.random.default_rng()

    # Normalize weights
    total = sum(weights.values())
    norm_weights = {k: v / total for k, v in weights.items()}

    parts = []
    for domain, weight in norm_weights.items():
        if domain not in domain_tokens or weight < 1e-6:
            continue
        n_tokens = int(batch_tokens * weight)
        available = domain_tokens[domain]
        if len(available) < n_tokens:
            # Repeat if needed (with warning in caller)
            repeats = (n_tokens // len(available)) + 1
            available = np.tile(available, repeats)
        start = rng.integers(0, max(1, len(available) - n_tokens))
        parts.append(available[start : start + n_tokens])

    if not parts:
        return np.zeros(batch_tokens, dtype=np.uint32)

    result = np.concatenate(parts)
    # Pad or truncate to exact size
    if len(result) < batch_tokens:
        result = np.pad(result, (0, batch_tokens - len(result)))
    return result[:batch_tokens]
