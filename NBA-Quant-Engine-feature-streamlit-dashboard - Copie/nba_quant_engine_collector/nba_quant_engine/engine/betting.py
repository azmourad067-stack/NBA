"""Value detection and out-of-sample bankroll simulation on PIT moneylines."""
from __future__ import annotations

from typing import Any


def expected_value(probability: float, decimal_odds: float) -> float:
    """Expected profit per unit staked."""
    return probability * decimal_odds - 1.0


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * probability - (1.0 - probability)) / b)


def value_bets(
    prediction: dict[str, Any],
    min_edge: float = 0.03,
    kelly_multiplier: float = 0.25,
    max_stake: float = 0.02,
) -> list[dict[str, Any]]:
    """Return the moneyline side(s) whose model EV exceeds ``min_edge``."""
    picks = []
    for side, prob_key, odds_key in (
        ("home", "home_win_probability", "market_home_odds"),
        ("away", "away_win_probability", "market_away_odds"),
    ):
        odds = prediction.get(odds_key)
        prob = prediction.get(prob_key)
        if odds is None or prob is None or odds <= 1.0:
            continue
        ev = expected_value(prob, odds)
        if ev < min_edge:
            continue
        stake = min(max_stake, kelly_multiplier * kelly_fraction(prob, odds))
        if stake <= 0:
            continue
        picks.append({"side": side, "odds": float(odds), "probability": float(prob), "edge": float(ev), "stake_fraction": float(stake)})
    picks.sort(key=lambda p: p["edge"], reverse=True)
    return picks[:1]


def simulate_bankroll(
    predictions: list[dict[str, Any]],
    min_edge: float = 0.03,
    kelly_multiplier: float = 0.25,
    max_stake: float = 0.02,
    initial_bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Chronological fractional-Kelly simulation on out-of-sample predictions.

    Only rows carrying verified pre-tipoff market odds are eligible. Stakes are
    sized on the bankroll available at bet time; games sharing a timestamp are
    sized on the same bankroll (no peeking at simultaneous results).
    """
    bankroll = peak = initial_bankroll
    max_drawdown = 0.0
    bets: list[dict[str, Any]] = []
    staked = profit = 0.0
    eligible = 0
    by_time: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(predictions, key=lambda r: r["played_at"]):
        by_time.setdefault(p["played_at"], []).append(p)

    for played_at, group in by_time.items():
        base = bankroll
        delta = 0.0
        for pred in group:
            if pred.get("market_home_odds") is None or pred.get("actual_home_win") is None:
                continue
            eligible += 1
            for pick in value_bets(pred, min_edge, kelly_multiplier, max_stake):
                stake = base * pick["stake_fraction"]
                won = (pick["side"] == "home") == bool(pred["actual_home_win"])
                pnl = stake * (pick["odds"] - 1.0) if won else -stake
                delta += pnl
                staked += stake
                profit += pnl
                bets.append({**pick, "game_provider_id": pred["game_provider_id"], "played_at": played_at,
                             "stake": stake, "won": won, "pnl": pnl})
        bankroll += delta
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak else 0.0)

    wins = sum(b["won"] for b in bets)
    return {
        "eligible_games": eligible,
        "bets": len(bets),
        "hit_rate": wins / len(bets) if bets else None,
        "average_odds": sum(b["odds"] for b in bets) / len(bets) if bets else None,
        "average_edge": sum(b["edge"] for b in bets) / len(bets) if bets else None,
        "total_staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else None,
        "final_bankroll": bankroll,
        "bankroll_growth": bankroll / initial_bankroll - 1.0,
        "max_drawdown": max_drawdown,
        "settings": {"min_edge": min_edge, "kelly_multiplier": kelly_multiplier, "max_stake": max_stake},
        "history": bets,
    }
