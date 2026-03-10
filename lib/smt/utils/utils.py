"""Utility helpers for the SMT solver module."""

import math


def powers_of_2_range(lo: int, hi: int) -> list[int]:
    """Return all powers of 2 in the closed interval [lo, hi].

    Args:
        lo: Lower bound (must itself be a power of 2).
        hi: Upper bound (must itself be a power of 2).

    Returns:
        Sorted list of powers of 2 from lo to hi inclusive.

    Example:
        >>> powers_of_2_range(32, 256)
        [32, 64, 128, 256]
    """
    lo_exp = int(math.log2(lo))
    hi_exp = int(math.log2(hi))
    return [2**k for k in range(lo_exp, hi_exp + 1)]


def default_symbol_domains() -> dict[str, list[int]]:
    """Return the default allowed values for each block-size symbol.

    BM, BN: powers of 2 from 32 to 4096.
    BK:     powers of 2 from 32 to 512.
    """
    return {
        "BM": powers_of_2_range(32, 4096),
        "BN": powers_of_2_range(32, 4096),
        "BK": powers_of_2_range(32, 512),
    }
