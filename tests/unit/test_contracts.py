"""Ingestion contracts: every violation must fail the load loudly."""

import pandas as pd
import pytest

from wc26.data import contracts
from wc26.data.contracts import ContractViolation, validate_fixtures, validate_matches


def _matches(n: int = 4, year: int = 2019, mean_goals_each: int = 1) -> pd.DataFrame:
    """A minimal contract-clean matches table; n rows in one decade."""
    dates = pd.to_datetime([f"{year}-03-{(i % 28) + 1:02d}" for i in range(n)])
    return pd.DataFrame(
        {
            "match_id": [f"id{i}" for i in range(n)],
            "match_date": dates,
            "home_id": ["Brazil"] * n,
            "away_id": ["Argentina"] * n,
            "home_score": [mean_goals_each] * n,
            "away_score": [mean_goals_each] * n,
            "tournament": ["Friendly"] * n,
            "comp_tier": ["friendly"] * n,
            "neutral": [False] * n,
        }
    )


def test_valid_table_passes() -> None:
    validate_matches(_matches())


def test_wrong_columns_fail() -> None:
    bad = _matches().rename(columns={"home_id": "home_team"})
    with pytest.raises(ContractViolation, match="columns"):
        validate_matches(bad)


def test_empty_table_fails() -> None:
    with pytest.raises(ContractViolation, match="empty"):
        validate_matches(_matches().iloc[0:0])


def test_na_score_fails() -> None:
    bad = _matches()
    bad["home_score"] = [None] + [1] * (len(bad) - 1)
    with pytest.raises(ContractViolation, match="NA scores"):
        validate_matches(bad)


def test_score_out_of_range_fails() -> None:
    bad = _matches()
    bad.loc[0, "home_score"] = 99
    with pytest.raises(ContractViolation, match=r"\[0, 31\]"):
        validate_matches(bad)


def test_conflicting_duplicate_match_id_fails() -> None:
    bad = _matches()
    bad.loc[1, "match_id"] = bad.loc[0, "match_id"]
    bad.loc[1, "home_score"] = 5
    with pytest.raises(ContractViolation, match="conflicting scores"):
        validate_matches(bad)


def test_repeated_identical_result_passes() -> None:
    table = _matches()
    table.loc[1] = table.loc[0]
    validate_matches(table)


def test_insane_year_fails() -> None:
    bad = _matches(year=1850)
    with pytest.raises(ContractViolation, match="sane range"):
        validate_matches(bad)


def test_unparsed_date_fails() -> None:
    bad = _matches()
    bad["match_date"] = bad["match_date"].dt.strftime("%Y-%m-%d")  # raw strings
    with pytest.raises(ContractViolation, match="parsed date"):
        validate_matches(bad)


def test_unknown_tier_fails() -> None:
    bad = _matches()
    bad["comp_tier"] = ["testimonial"] * len(bad)
    with pytest.raises(ContractViolation, match="comp_tier"):
        validate_matches(bad)


def test_empty_team_id_fails() -> None:
    bad = _matches()
    bad.loc[0, "home_id"] = ""
    with pytest.raises(ContractViolation, match="empty team id"):
        validate_matches(bad)


def test_non_bool_neutral_fails() -> None:
    bad = _matches()
    bad["neutral"] = [0] * len(bad)
    with pytest.raises(ContractViolation, match="bool"):
        validate_matches(bad)


def test_modern_decade_goal_drought_fails() -> None:
    bad = _matches(n=120, year=1995, mean_goals_each=0)
    with pytest.raises(ContractViolation, match="modern decades"):
        validate_matches(bad)


def test_early_decade_high_scoring_passes() -> None:
    # 1930s internationals really did average >3.5 goals; the loose early bound
    # [1.0, 7.0] must accept them (DECISION in contracts.py).
    validate_matches(_matches(n=120, year=1935, mean_goals_each=2))


def test_early_decade_parsing_disaster_fails() -> None:
    bad = _matches(n=120, year=1935, mean_goals_each=4)  # mean 8 goals/match
    with pytest.raises(ContractViolation, match=r"\[1.0, 7.0\]"):
        validate_matches(bad)


def _fixtures(n: int = 3) -> pd.DataFrame:
    dates = pd.to_datetime([f"2026-06-{11 + i:02d}" for i in range(n)])
    return pd.DataFrame(
        {
            "match_id": [f"fid{i}" for i in range(n)],
            "match_date": dates,
            "home_id": ["Mexico"] * n,
            "away_id": ["South Africa"] * n,
            "tournament": ["FIFA World Cup"] * n,
            "neutral": [False] * n,
        }
    )


def test_valid_fixtures_pass() -> None:
    validate_fixtures(_fixtures())


def test_fixture_wrong_columns_fail() -> None:
    with pytest.raises(ContractViolation, match="columns"):
        validate_fixtures(_fixtures().drop(columns=["neutral"]))


def test_duplicate_fixture_id_fails() -> None:
    bad = _fixtures()
    bad.loc[1, "match_id"] = bad.loc[0, "match_id"]
    with pytest.raises(ContractViolation, match="duplicate match_id"):
        validate_fixtures(bad)


def test_column_constants_are_pinned() -> None:
    # Ingest slices frames by these constants; pin them so schema drift is loud.
    assert contracts.MATCH_COLUMNS == [
        "match_id",
        "match_date",
        "home_id",
        "away_id",
        "home_score",
        "away_score",
        "tournament",
        "comp_tier",
        "neutral",
    ]
    assert contracts.FIXTURE_COLUMNS == [
        "match_id",
        "match_date",
        "home_id",
        "away_id",
        "tournament",
        "neutral",
    ]
