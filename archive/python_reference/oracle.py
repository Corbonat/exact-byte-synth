"""Black-box target function for 8-bit synthesis."""

import math

WIDTH = 8
MASK = (1 << WIDTH) - 1
DOMAIN_SIZE = 1 << WIDTH


def oracle(x: int) -> int:
    """Return the target value for the given 8-bit input."""
    return (50 + (math.ceil(50*math.cos((x-128)/4000.7437)))) & MASK
