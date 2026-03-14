from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from hw_1.src.common import probability_from_log, scientific_notation_from_log
from hw_1.styles import styles, styles_count


@dataclass(slots=True)
class CategorySample:
    category: str
    value: str
    count: int
    prior_probability: float
    mle_probability: float
    bayesian_probability: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "value": self.value,
            "count": self.count,
            "prior_probability": self.prior_probability,
            "mle_probability": self.mle_probability,
            "bayesian_probability": self.bayesian_probability,
        }


class StyleGenerator:
    def __init__(self, alpha: float = 1.0, seed: int | None = None) -> None:
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        for category, options in styles.items():
            counts = styles_count.get(category)
            if counts is None:
                raise ValueError(f"Missing counts for category: {category}")
            if len(options) != len(counts):
                raise ValueError(f"Option/count mismatch for category: {category}")

    def _category_probabilities(self, category: str) -> tuple[np.ndarray, np.ndarray]:
        counts = np.asarray(styles_count[category], dtype=np.float64)
        total = counts.sum()
        prior = np.full(len(counts), 1.0 / len(counts), dtype=np.float64)
        mle = counts / total
        bayesian = (counts + self.alpha * prior * len(counts)) / (total + self.alpha * len(counts))
        return mle, bayesian

    def sample(self) -> dict[str, Any]:
        sampled_style: dict[str, str] = {}
        sampled_categories: list[CategorySample] = []
        mle_log_probability = 0.0
        bayesian_log_probability = 0.0
        mle_is_zero = False

        for category, options in styles.items():
            counts = styles_count[category]
            mle_probabilities, bayesian_probabilities = self._category_probabilities(category)
            prior_probability = 1.0 / len(options)
            index = int(self.rng.choice(len(options), p=bayesian_probabilities))

            sampled_style[category] = options[index]
            sampled_categories.append(
                CategorySample(
                    category=category,
                    value=options[index],
                    count=counts[index],
                    prior_probability=prior_probability,
                    mle_probability=float(mle_probabilities[index]),
                    bayesian_probability=float(bayesian_probabilities[index]),
                )
            )

            if mle_probabilities[index] == 0:
                mle_is_zero = True
            else:
                mle_log_probability += math.log(float(mle_probabilities[index]))

            bayesian_log_probability += math.log(float(bayesian_probabilities[index]))

        if mle_is_zero:
            mle_log_probability = float("-inf")

        return {
            "sampled_style": sampled_style,
            "sampled_categories": [item.as_dict() for item in sampled_categories],
            "mle_probability": probability_from_log(mle_log_probability),
            "mle_probability_scientific": scientific_notation_from_log(mle_log_probability),
            "mle_log_probability": mle_log_probability,
            "bayesian_probability": probability_from_log(bayesian_log_probability),
            "bayesian_probability_scientific": scientific_notation_from_log(bayesian_log_probability),
            "bayesian_log_probability": bayesian_log_probability,
            "bayesian_model": "dirichlet_posterior_predictive",
            "alpha": self.alpha,
        }

    def sample_many(self, count: int) -> list[dict[str, Any]]:
        return [self.sample() for _ in range(count)]
