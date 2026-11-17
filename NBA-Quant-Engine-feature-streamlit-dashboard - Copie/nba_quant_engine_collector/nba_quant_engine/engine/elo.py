"""Margin-of-victory Elo rating, updated strictly after each game."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EloRatings:
    """Elo with margin-of-victory multiplier and between-season regression.

    Defaults follow the widely used NBA formulation: K=20, home court ~100
    Elo points, MOV multiplier ((|mov|+3)^0.8) / (7.5 + 0.006 * winner_diff)
    and a 25 % regression toward the league mean at each new season.
    """

    k: float = 20.0
    home_advantage: float = 100.0
    initial: float = 1500.0
    season_carryover: float = 0.75
    use_mov: bool = True
    ratings: dict[str, float] = field(default_factory=dict)
    seasons: dict[str, int] = field(default_factory=dict)

    def rating(self, team: str, season: int | None = None) -> float:
        current = self.ratings.get(team, self.initial)
        if season is not None:
            last = self.seasons.get(team)
            if last is not None and season > last:
                current = self.initial + self.season_carryover * (current - self.initial)
                self.ratings[team] = current
            self.seasons[team] = season if last is None else max(last, season)
        return current

    def expected_home(self, home: str, away: str, season: int | None = None) -> float:
        diff = self.rating(home, season) + self.home_advantage - self.rating(away, season)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def elo_diff(self, home: str, away: str, season: int | None = None) -> float:
        return self.rating(home, season) + self.home_advantage - self.rating(away, season)

    def update(self, home: str, away: str, home_score: float, away_score: float, season: int | None = None) -> None:
        expected = self.expected_home(home, away, season)
        home_win = 1.0 if home_score > away_score else 0.0
        multiplier = 1.0
        if self.use_mov:
            margin = abs(float(home_score) - float(away_score))
            diff = self.elo_diff(home, away, season)
            winner_diff = diff if home_win else -diff
            multiplier = ((margin + 3.0) ** 0.8) / (7.5 + 0.006 * winner_diff)
            multiplier = max(0.1, multiplier)
        shift = self.k * multiplier * (home_win - expected)
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift


def elo_to_spread(elo_diff: float) -> float:
    """Approximate point spread implied by an Elo difference (~28 Elo per point)."""
    return elo_diff / 28.0


def logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
