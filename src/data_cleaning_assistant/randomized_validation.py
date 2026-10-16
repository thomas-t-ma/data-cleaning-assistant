from __future__ import annotations

from .quantitative_validation import QuantitativeStressReport, run_quantitative_cases
from .randomized_adversarial import (
    DEFAULT_ADVERSARIAL_SEEDS,
    randomized_adversarial_cases,
)


def run_randomized_validation(
    *,
    seeds: tuple[int, ...] = DEFAULT_ADVERSARIAL_SEEDS,
    rows_per_case: int = 100,
) -> QuantitativeStressReport:
    """Run seeded randomized data-cleaning trials through the quantitative scorer."""
    cases = randomized_adversarial_cases(
        seeds=seeds,
        rows_per_case=rows_per_case,
    )
    return run_quantitative_cases(
        cases,
        temp_prefix="data-cleaner-randomized-",
    )
