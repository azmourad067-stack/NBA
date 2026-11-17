"""Data sources for the engine: Supabase (read-only), CSV files, a public
historical results file, and a synthetic league for offline demos/tests.

Every loader returns ``(games, team_stats, odds)`` lists of dicts using the
column names of the ``nba`` Supabase schema.
"""
from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from psycopg import sql

GameData = tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]

TEAM_STAT_COLUMNS = [
    "game_provider_id", "team_provider_id", "points", "field_goals_made", "field_goals_attempted",
    "three_pointers_made", "three_pointers_attempted", "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "assists", "steals", "blocks", "turnovers", "personal_fouls",
]


def load_from_database(db: Any, season_types: tuple[str, ...] = ("regular", "postseason", "play_in")) -> GameData:
    """Read-only extraction from Supabase. Never writes, never calls DunkAPI."""
    schema = sql.Identifier(db.schema)
    games = db.conn.execute(sql.SQL("""
        SELECT game_provider_id, season_end_year, season_type, game_date, scheduled_tipoff_at,
               game_status, home_team_provider_id, away_team_provider_id, home_score, away_score
        FROM {}.nba_games
        WHERE game_status IN ('final', 'scheduled') AND season_type = ANY(%s)
        ORDER BY game_date, scheduled_tipoff_at
    """).format(schema), (list(season_types),)).fetchall()
    stats = db.conn.execute(sql.SQL("SELECT {} FROM {}.nba_team_game_stats").format(
        sql.SQL(", ").join(sql.Identifier(c) for c in TEAM_STAT_COLUMNS), schema)).fetchall()
    odds = db.conn.execute(sql.SQL("""
        SELECT o.game_provider_id, o.sportsbook, o.market_type, o.selection, o.line_value,
               o.decimal_odds, o.source_captured_at, o.snapshot_time_quality
        FROM {}.nba_odds_snapshots AS o
        JOIN {}.nba_games AS g ON g.game_provider_id = o.game_provider_id
        WHERE o.market_type IN ('moneyline', 'spread', 'total')
          AND o.snapshot_time_quality = 'verified_point_in_time'
          AND g.scheduled_tipoff_at IS NOT NULL
          AND o.source_captured_at < g.scheduled_tipoff_at
          AND o.decimal_odds > 1
    """).format(schema, schema)).fetchall()
    return [dict(r) for r in games], [dict(r) for r in stats], [dict(r) for r in odds]


def load_team_names(db: Any) -> dict[str, str]:
    rows = db.conn.execute(sql.SQL("SELECT team_provider_id, abbreviation, display_name FROM {}.nba_teams").format(
        sql.Identifier(db.schema))).fetchall()
    return {str(r["team_provider_id"]): str(r.get("abbreviation") or r.get("display_name")) for r in rows}


def _read_csv(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in csv.DictReader(handle)]


def load_from_csv(games_csv: str | Path, team_stats_csv: str | Path | None = None, odds_csv: str | Path | None = None) -> GameData:
    """CSV exports with the same column names as the Supabase tables."""
    games = _read_csv(games_csv)
    for g in games:
        for key in ("home_score", "away_score", "season_end_year"):
            if g.get(key) is not None:
                g[key] = int(float(g[key]))
    return games, _read_csv(team_stats_csv), _read_csv(odds_csv)


def load_fivethirtyeight(path: str | Path, first_season: int = 2005, last_season: int | None = None) -> GameData:
    """Convert the public FiveThirtyEight ``nbaallelo.csv`` results file.

    Only scores are available (no box scores, no odds): box-score features stay
    missing and are imputed. Useful to benchmark the engine on real NBA history.
    """
    games: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["_iscopy"] != "0" or row["lg_id"] != "NBA":
                continue
            season = int(row["year_id"])
            if season < first_season or (last_season and season > last_season):
                continue
            month, day, year = (int(x) for x in row["date_game"].split("/"))
            team_home = row["game_location"] in ("H", "N")
            home, away = (row["team_id"], row["opp_id"]) if team_home else (row["opp_id"], row["team_id"])
            hs, as_ = (int(row["pts"]), int(row["opp_pts"])) if team_home else (int(row["opp_pts"]), int(row["pts"]))
            games.append({
                "game_provider_id": row["game_id"], "season_end_year": season,
                "season_type": "postseason" if row["is_playoffs"] == "1" else "regular",
                "game_date": f"{year:04d}-{month:02d}-{day:02d}", "scheduled_tipoff_at": None,
                "game_status": "final", "home_team_provider_id": home, "away_team_provider_id": away,
                "home_score": hs, "away_score": as_,
            })
    return games, [], []


# ---------------------------------------------------------------------------
# Synthetic league (offline demo + tests)
# ---------------------------------------------------------------------------

def _book_prices(prob: float, vig: float) -> tuple[float, float]:
    ph, pa = prob * (1 + vig / 2), (1 - prob) * (1 + vig / 2)
    return 1.0 / ph, 1.0 / pa


def generate_synthetic_league(
    seasons: int = 3,
    teams: int = 30,
    games_per_team: int = 82,
    upcoming_games: int = 10,
    seed: int = 7,
    market_noise: float = 0.035,
    vig: float = 0.045,
    first_season: int = 2024,
) -> GameData:
    """Generate a realistic-looking league with latent team strength.

    Strength drifts within a season and partially regresses between seasons.
    Box scores are produced from team-level four-factor tendencies, so points
    are consistent with the box score. Moneyline, spread and total quotes are
    generated from the latent truth with noise and bookmaker margin, captured
    two hours before tip-off (verified point-in-time).
    """
    rng = random.Random(seed)
    team_ids = [f"T{i:02d}" for i in range(teams)]
    strength = {t: rng.gauss(0, 4.5) for t in team_ids}      # net points per 100 possessions
    off_bias = {t: rng.gauss(0, 2.0) for t in team_ids}
    pace = {t: rng.gauss(99.0, 2.5) for t in team_ids}
    games: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    odds: list[dict[str, Any]] = []
    games_per_season = teams * games_per_team // 2
    counter = 0

    def box(team: str, opp: str, poss: float, efg: float, home: bool) -> dict[str, Any]:
        tov = max(5, round(rng.gauss(0.13 * poss, 2.0)))
        fta = max(5, round(rng.gauss(0.27 * (poss * 0.86), 4.0)))
        fga = max(60, round(poss - tov - 0.44 * fta + rng.gauss(11, 2)))
        tpa = max(15, round(fga * rng.uniform(0.33, 0.43)))
        tpm = max(3, min(tpa, round(tpa * rng.gauss(0.36, 0.04))))
        fgm = max(tpm, round(efg * fga - 0.5 * tpm))
        ftm = min(fta, round(fta * rng.gauss(0.78, 0.05)))
        oreb = max(2, round(rng.gauss(10.5, 3)))
        dreb = max(20, round(rng.gauss(34, 4)))
        return {
            "team_provider_id": team, "points": 2 * fgm + tpm + ftm,
            "field_goals_made": fgm, "field_goals_attempted": fga,
            "three_pointers_made": tpm, "three_pointers_attempted": tpa,
            "free_throws_made": ftm, "free_throws_attempted": fta,
            "offensive_rebounds": oreb, "defensive_rebounds": dreb,
            "assists": max(10, round(fgm * rng.uniform(0.55, 0.68))),
            "steals": max(2, round(rng.gauss(7.5, 2))), "blocks": max(0, round(rng.gauss(5, 2))),
            "turnovers": tov, "personal_fouls": max(10, round(rng.gauss(20, 3))),
        }

    for s in range(seasons):
        season = first_season + s
        if s:
            for t in team_ids:
                strength[t] = 0.6 * strength[t] + rng.gauss(0, 2.8)
        start = datetime(season - 1, 10, 22, 0, 0, tzinfo=timezone.utc)
        total_games = games_per_season + (upcoming_games if s == seasons - 1 else 0)
        per_day = max(1, teams // 4)
        day = 0
        made = 0
        while made < total_games:
            todays = rng.sample(team_ids, min(len(team_ids), 2 * per_day))
            for i in range(0, len(todays) - 1, 2):
                if made >= total_games:
                    break
                home, away = todays[i], todays[i + 1]
                tip = start + timedelta(days=day, hours=rng.choice([0, 0, 1, 2, 3]))
                counter += 1
                gid = f"S{counter:06d}"
                is_upcoming = s == seasons - 1 and made >= games_per_season
                exp_poss = 0.5 * (pace[home] + pace[away])
                exp_margin_per100 = strength[home] - strength[away] + 2.8
                exp_margin = exp_margin_per100 * exp_poss / 100.0
                true_prob = 0.5 * (1 + math.erf(exp_margin / (12.0 * math.sqrt(2))))
                game = {
                    "game_provider_id": gid, "season_end_year": season, "season_type": "regular",
                    "game_date": tip.date().isoformat(), "scheduled_tipoff_at": tip.isoformat(),
                    "game_status": "scheduled" if is_upcoming else "final",
                    "home_team_provider_id": home, "away_team_provider_id": away,
                    "home_score": None, "away_score": None,
                }
                if not is_upcoming:
                    poss = rng.gauss(exp_poss, 3.0)
                    shock = rng.gauss(0, 11.0)
                    base_efg = 0.535
                    h_efg = base_efg + (0.5 * exp_margin_per100 + off_bias[home] + shock / 2) / 220.0
                    a_efg = base_efg + (-0.5 * exp_margin_per100 + off_bias[away] - shock / 2) / 220.0
                    hb, ab = box(home, away, poss, h_efg, True), box(away, home, poss, a_efg, False)
                    if hb["points"] == ab["points"]:
                        hb["free_throws_made"] += 1
                        hb["free_throws_attempted"] += 1
                        hb["points"] += 1
                    game["home_score"], game["away_score"] = hb["points"], ab["points"]
                    stats.extend([{"game_provider_id": gid, **hb}, {"game_provider_id": gid, **ab}])
                games.append(game)
                # Market: noisy view of the truth, captured 2h before tipoff.
                market_prob = min(0.97, max(0.03, true_prob + rng.gauss(0, market_noise)))
                captured = (tip - timedelta(hours=2)).isoformat()
                for book in ("bookA", "bookB"):
                    bp = min(0.97, max(0.03, market_prob + rng.gauss(0, 0.01)))
                    h_odds, a_odds = _book_prices(bp, vig)
                    spread_line = round(-exp_margin * 2 + rng.gauss(0, 1.5)) / 2
                    total_line = round((2 * exp_poss * 1.12 + rng.gauss(0, 3)) * 2) / 2
                    for market, selection, line, price in (
                        ("moneyline", "home", None, h_odds), ("moneyline", "away", None, a_odds),
                        ("spread", "home", spread_line, 1.91), ("spread", "away", -spread_line, 1.91),
                        ("total", "over", total_line, 1.91), ("total", "under", total_line, 1.91),
                    ):
                        odds.append({
                            "game_provider_id": gid, "sportsbook": book, "market_type": market,
                            "selection": selection, "line_value": line, "decimal_odds": round(price, 3),
                            "source_captured_at": captured, "snapshot_time_quality": "verified_point_in_time",
                        })
                made += 1
            for t in team_ids:
                strength[t] += rng.gauss(0, 0.12)
            day += 1
    return games, stats, odds
