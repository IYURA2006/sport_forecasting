"""The leakage firewall: the only read path for historical data.

Model fitting and backtest code must import :func:`load_matches` and never read
data files directly -- the ``before`` cutoff is what makes the walk-forward
backtest structurally unable to see the future. Enforced by the control tests
in ``tests/controls/``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

SNAPSHOTS_DIR = Path("data/snapshots")


def load_matches(
    before: date | None = None, snapshots_dir: Path = SNAPSHOTS_DIR
) -> pd.DataFrame:
    """Completed matches with match_date strictly before ``before`` (all if None).

    Strict inequality: a model fitted "as of" a tournament's first day must not
    see that day's results.
    """
    frame = pd.read_csv(snapshots_dir / "matches.csv", parse_dates=["match_date"])
    if before is not None:
        frame = frame.loc[frame["match_date"] < pd.Timestamp(before)].reset_index(drop=True)
    return frame


def load_fixtures(snapshots_dir: Path = SNAPSHOTS_DIR) -> pd.DataFrame:
    """Scheduled (unplayed) fixtures -- the 2026 group stage at ingest time."""
    return pd.read_csv(snapshots_dir / "fixtures.csv", parse_dates=["match_date"])
