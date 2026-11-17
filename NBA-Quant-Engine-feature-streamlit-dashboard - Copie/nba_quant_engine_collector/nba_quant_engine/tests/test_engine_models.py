import copy
import math

import pytest

from engine.betting import expected_value, kelly_fraction, simulate_bankroll, value_bets
from engine.dataset import aggregate_verified_market, build_feature_rows, four_factors, possessions
from engine.elo import EloRatings
from engine.evaluation import walk_forward
from engine.service import (
    build_report_from_rows, build_rows, load_engine, predict_upcoming_games, save_engine, train_engine,
)
from engine.sources import generate_synthetic_league


@pytest.fixture(scope="module")
def league():
    return generate_synthetic_league(seasons=2, teams=12, games_per_team=60, upcoming_games=6, seed=3)


@pytest.fixture(scope="module")
def rows(league):
    return build_rows(league)


def test_elo_update_is_zero_sum_and_regresses_between_seasons():
    elo = EloRatings()
    elo.expected_home("A", "B", 2025)
    elo.update("A", "B", 120, 100, 2025)
    assert elo.ratings["A"] > 1500 > elo.ratings["B"]
    assert math.isclose(elo.ratings["A"] + elo.ratings["B"], 3000.0)
    before = elo.ratings["A"]
    after = elo.rating("A", 2026)
    assert math.isclose(after - 1500, 0.75 * (before - 1500))


def test_box_score_metrics_are_plausible():
    own = {"field_goals_made": 40, "field_goals_attempted": 85, "three_pointers_made": 12,
           "three_pointers_attempted": 35, "free_throws_made": 18, "free_throws_attempted": 22,
           "offensive_rebounds": 10, "defensive_rebounds": 33, "turnovers": 13}
    opp = dict(own, offensive_rebounds=9, defensive_rebounds=35)
    ff = four_factors(own, opp)
    assert ff["efg"] == pytest.approx((40 + 6) / 85)
    assert ff["orb_pct"] == pytest.approx(10 / 45)
    assert 85 < possessions(own, opp) < 110


def test_changing_a_result_never_changes_earlier_or_same_game_features(league):
    games, stats, _ = league
    games = [g for g in games if g["game_status"] == "final"][:300]
    base = build_feature_rows(games, stats)
    k = 150
    target_id = base[k]["game_provider_id"]
    mutated = copy.deepcopy(games)
    for g in mutated:
        if g["game_provider_id"] == target_id:
            g["home_score"], g["away_score"] = g["away_score"] + 40, g["home_score"]
    altered = build_feature_rows(mutated, [s for s in stats if s["game_provider_id"] != target_id])
    feature_keys = [c for c in base[0] if not c.startswith("target_") and c not in ("home_score", "away_score")]
    for i in range(k + 1):
        assert {c: base[i][c] for c in feature_keys} == {c: altered[i][c] for c in feature_keys}
    assert any(base[j]["home_elo_prob"] != altered[j]["home_elo_prob"] for j in range(k + 1, len(base)))


def test_walk_forward_trains_only_on_earlier_days(rows):
    completed = [r for r in rows if r["target_home_win"] is not None]
    preds = walk_forward(completed, min_train_games=200, retrain_every=100)
    assert preds
    order = sorted(completed, key=lambda r: (r["played_at"], r["game_provider_id"]))
    for pred in preds:
        last_train_day = order[pred["train_games"] - 1]["played_at"][:10]
        assert pred["played_at"][:10] > last_train_day


def test_market_aggregation_uses_only_verified_pretip_quotes():
    tip = "2026-01-01T20:00:00Z"
    base = {"game_provider_id": "g", "source_captured_at": "2026-01-01T18:00:00Z",
            "snapshot_time_quality": "verified_point_in_time"}
    odds = [
        dict(base, sportsbook="a", market_type="moneyline", selection="home", decimal_odds=1.8),
        dict(base, sportsbook="a", market_type="moneyline", selection="away", decimal_odds=2.1),
        dict(base, sportsbook="a", market_type="spread", selection="home", line_value=-3.5, decimal_odds=1.91),
        dict(base, sportsbook="a", market_type="total", selection="over", line_value=221.5, decimal_odds=1.91),
        dict(base, sportsbook="late", market_type="moneyline", selection="home", decimal_odds=1.1,
             source_captured_at="2026-01-01T21:00:00Z"),
        dict(base, sportsbook="late", market_type="moneyline", selection="away", decimal_odds=8.0,
             source_captured_at="2026-01-01T21:00:00Z"),
    ]
    quote = aggregate_verified_market(odds, {"g": tip})["g"]
    assert quote["home_odds"] == 1.8 and quote["away_odds"] == 2.1
    assert quote["home_prob"] == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 2.1))
    assert quote["home_spread"] == -3.5 and quote["total"] == 221.5


def test_betting_math():
    assert expected_value(0.5, 2.2) == pytest.approx(0.1)
    assert kelly_fraction(0.5, 2.2) == pytest.approx((1.2 * 0.5 - 0.5) / 1.2)
    assert kelly_fraction(0.3, 2.0) == 0.0
    pred = {"home_win_probability": 0.6, "away_win_probability": 0.4, "market_home_odds": 2.0, "market_away_odds": 1.9}
    picks = value_bets(pred, min_edge=0.05)
    assert len(picks) == 1 and picks[0]["side"] == "home"
    sim = simulate_bankroll([dict(pred, game_provider_id="g", played_at="2026-01-01", actual_home_win=1)])
    assert sim["bets"] == 1 and sim["profit"] > 0


def test_end_to_end_report_beats_naive_baseline(rows):
    report = build_report_from_rows(rows, min_train_games=300, retrain_every=150)
    assert report["status"] == "evaluated"
    models = report["models"]
    assert models["ensemble"]["brier_score"] < models["always_home"]["brier_score"]
    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0
    assert report["upcoming"]["status"] == "predicted"
    pred = report["upcoming"]["predictions"][0]
    assert 0.0 < pred["home_win_probability"] < 1.0
    assert pred["home_win_probability"] + pred["away_win_probability"] == pytest.approx(1.0)
    assert "predicted_total" in pred and "value_bet" in pred


def test_engine_round_trip(tmp_path, rows):
    engine = train_engine(rows)
    path = save_engine(engine, tmp_path / "engine.joblib")
    loaded = load_engine(path)
    upcoming = predict_upcoming_games(rows, engine=loaded)
    assert upcoming["status"] == "predicted"
    original = predict_upcoming_games(rows, engine=engine)
    assert upcoming["predictions"][0]["home_win_probability"] == pytest.approx(
        original["predictions"][0]["home_win_probability"])


def test_insufficient_history_is_reported_not_faked():
    games, stats, odds = generate_synthetic_league(seasons=1, teams=6, games_per_team=10, upcoming_games=2)
    report = build_report_from_rows(build_rows((games, stats, odds)), min_train_games=100)
    assert report["status"] == "insufficient_history"
    assert report["upcoming"]["status"] == "insufficient_history"
