from typing import Union


class RewardAggregator:
    """Isku daraan dhammaan verifier signals (accuracy + format + length) sida
    architecture diagram-ku qeexay — meel dhexe halkii reward funcs-ku is qariyaan."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def aggregate(self, signals: dict[str, list[float]]) -> list[float]:
        n = len(next(iter(signals.values())))
        totals = [0.0] * n
        for key, values in signals.items():
            w = self.weights.get(key, 1.0)
            for i, v in enumerate(values):
                totals[i] += w * v
        return totals