from __future__ import annotations

import math
import sys


def probability_from_log(log_probability: float) -> float:
    if math.isinf(log_probability) and log_probability < 0:
        return 0.0
    min_log = math.log(sys.float_info.min)
    if log_probability < min_log:
        return 0.0
    return math.exp(log_probability)


def scientific_notation_from_log(log_probability: float, precision: int = 6) -> str:
    if math.isinf(log_probability) and log_probability < 0:
        return "0"
    log10_probability = log_probability / math.log(10)
    exponent = math.floor(log10_probability)
    mantissa = 10 ** (log10_probability - exponent)
    return f"{mantissa:.{precision}f}e{exponent}"
