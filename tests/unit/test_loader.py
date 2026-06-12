"""Loader: the only read path, with the leakage cutoff."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from wc26.data.loader import load_fixtures, load_matches

MATCHES_CSV = """\
match_id,match_date,home_id,away_id,home_score,away_score,tournament,comp_tier,neutral
a1,2014-06-12,Brazil,Croatia,3,1,FIFA World Cup,worldcup,False
a2,2018-06-14,Russia,Saudi Arabia,5,0,FIFA World Cup,worldcup,False
a3,2018-06-15,Egypt,Uruguay,0,1,FIFA World Cup,worldcup,True
"""

FIXTURES_CSV = """\
match_id,match_date,home_id,away_id,tournament,neutral
f1,2026-06-11,Mexico,South Africa,FIFA World Cup,False
"""


@pytest.fixture
def snapshots(tmp_path: Path) -> Path:
    (tmp_path / "matches.csv").write_text(MATCHES_CSV)
    (tmp_path / "fixtures.csv").write_text(FIXTURES_CSV)
    return tmp_path


def test_no_cutoff_returns_everything(snapshots: Path) -> None:
    assert len(load_matches(snapshots_dir=snapshots)) == 3


def test_cutoff_is_strict(snapshots: Path) -> None:
    # A model fitted "as of" 2018-06-14 must not see that day's results.
    matches = load_matches(before=date(2018, 6, 14), snapshots_dir=snapshots)
    assert list(matches["match_id"]) == ["a1"]


def test_cutoff_excludes_all_future(snapshots: Path) -> None:
    matches = load_matches(before=date(2014, 1, 1), snapshots_dir=snapshots)
    assert matches.empty


def test_dates_are_parsed(snapshots: Path) -> None:
    matches = load_matches(snapshots_dir=snapshots)
    assert pd.api.types.is_datetime64_any_dtype(matches["match_date"])


def test_load_fixtures(snapshots: Path) -> None:
    fixtures = load_fixtures(snapshots_dir=snapshots)
    assert list(fixtures["match_id"]) == ["f1"]
    assert pd.api.types.is_datetime64_any_dtype(fixtures["match_date"])
