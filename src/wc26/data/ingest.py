"""Raw results CSV -> typed Parquet snapshot.

Single raw input: ``data/raw/results.csv`` -- the martj42 "International football
results from 1872" table (pinned source below) with the 72 scheduled 2026 World Cup
group-stage fixtures appended as rows whose scores are NA. Completed rows become the
bitemporal ``matches`` table; NA-score rows become the ``fixtures`` table.

Snapshots are content-addressed: the snapshot id is the truncated SHA-256 of the raw
file bytes, so identical input always yields the identical snapshot id and re-running
ingest is a no-op rather than a fork.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc26.data import contracts
from wc26.data.entities import canonical_team_id, comp_tier

#: Pinned upstream source. The local raw file additionally carries the 2026
#: group-stage schedule as NA-score rows; the manifest fingerprints its exact
#: bytes, so provenance is checksummed even though ingest itself never downloads.
RAW_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

#: knowledge_time = event_time + 3h: a result is knowable roughly 3h after kickoff.
KNOWLEDGE_LAG = pd.Timedelta(hours=3)

# DECISION: the upstream table carries two rows for the 1974-02-17 Tahiti v
# New Caledonia friendly with contradictory scores (2-1 and 1-2). There is no way to
# tell which is right, so both rows are dropped. The exclusion is an explicit list so
# that any NEW conflict in a future data refresh still fails the ingestion contract
# instead of being silently absorbed.
KNOWN_CONFLICTING_ROWS: frozenset[tuple[str, str, str, str]] = frozenset(
    {("1974-02-17", "Tahiti", "New Caledonia", "Friendly")}
)


# DECISION: match_id hashes canonical ids rather than raw source names, so a future
# odds feed that says "USA" joins to the same match_id once canonicalised.
def match_id(date_iso: str, home_id: str, away_id: str, tournament: str) -> str:
    """SHA-1 of (date, home, away, tournament) -- no score fields, so a fixture's id
    is identical before and after the match is played."""
    return hashlib.sha1("|".join((date_iso, home_id, away_id, tournament)).encode()).hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV with scores as nullable ints (NA score = scheduled fixture)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download the results table from {RAW_RESULTS_URL} "
            "and append the 2026 group-stage schedule as NA-score rows."
        )
    return pd.read_csv(path, dtype={"home_score": "Int16", "away_score": "Int16", "date": str})


def build_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw rows into typed (completed matches, scheduled fixtures) tables."""
    frame = raw.copy()
    if frame["neutral"].isna().any():
        raise contracts.ContractViolation("neutral column has missing values")

    keys = zip(frame["date"], frame["home_team"], frame["away_team"], frame["tournament"])
    frame = frame.loc[[key not in KNOWN_CONFLICTING_ROWS for key in keys]]

    frame["home_id"] = frame["home_team"].map(canonical_team_id)
    frame["away_id"] = frame["away_team"].map(canonical_team_id)
    frame["comp_tier"] = frame["tournament"].map(comp_tier)
    frame["match_id"] = [
        match_id(date, home, away, tournament)
        for date, home, away, tournament in zip(
            frame["date"], frame["home_id"], frame["away_id"], frame["tournament"]
        )
    ]
    # Exact re-statements of one result are harmless duplicates; keep the first.
    # Scores stay in the dedup key deliberately: a conflicting pair must NOT
    # collapse here -- both rows survive so the conflict contract can fire.
    frame = frame.drop_duplicates(subset=["match_id", "home_score", "away_score"])

    frame["event_time"] = pd.to_datetime(frame["date"], utc=True)
    frame["knowledge_time"] = frame["event_time"] + KNOWLEDGE_LAG
    # DECISION: host_home is derived from the source's own neutral flag -- in a final
    # tournament the only non-neutral fixtures are the hosts' (verified on the 2026
    # rows: the 9 non-neutral fixtures are exactly the USA/Mexico/Canada group games).
    frame["host_home"] = ~frame["neutral"] & frame["comp_tier"].isin(["worldcup", "continental"])

    completed = frame["home_score"].notna() & frame["away_score"].notna()
    # Guard the narrowing cast below: int8 wraps silently (287 -> 31), so the
    # [0, 31] contract must see the raw parsed values.
    contracts.require_score_range(frame.loc[completed, ["home_score", "away_score"]])

    matches = (
        frame.loc[completed, contracts.MATCH_COLUMNS]
        .sort_values(["event_time", "match_id"], kind="mergesort")
        .reset_index(drop=True)
        .astype(
            {
                "home_score": "int8",
                "away_score": "int8",
                "tournament": "category",
                "comp_tier": "category",
            }
        )
    )
    fixtures = (
        frame.loc[~completed, contracts.FIXTURE_COLUMNS]
        .sort_values(["event_time", "match_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    # A schedule row goes stale once its played result lands in a data refresh;
    # the result wins and the fixture row is dropped.
    fixtures = fixtures.loc[~fixtures["match_id"].isin(set(matches["match_id"]))].reset_index(
        drop=True
    )
    return matches, fixtures


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    directory: Path
    n_matches: int
    n_fixtures: int


def build_snapshot(raw_path: Path, snapshots_dir: Path) -> Snapshot:
    """Validate the raw file, write Parquet under a content-addressed directory, and
    record provenance (source URL, raw checksum, row counts) in manifest.json."""
    matches, fixtures = build_tables(load_raw(raw_path))  # load_raw owns the missing-file error
    contracts.validate_matches(matches)
    contracts.validate_fixtures(fixtures)

    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    snapshot_id = raw_sha256[:12]

    directory = snapshots_dir / snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(directory / "matches.parquet", index=False)
    fixtures.to_parquet(directory / "fixtures.parquet", index=False)

    manifest_path = snapshots_dir / "manifest.json"
    manifest: dict[str, dict[str, object]] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    manifest[snapshot_id] = {
        "source_url": RAW_RESULTS_URL,
        "source_note": (
            "local file = upstream results plus the 72 appended 2026 group-stage "
            "schedule rows (NA scores); raw_sha256 fingerprints the local file, "
            "not the upstream download"
        ),
        "raw_file": str(raw_path),
        "raw_sha256": raw_sha256,
        "created": datetime.now(UTC).isoformat(timespec="microseconds"),
        "n_matches": int(len(matches)),
        "n_fixtures": int(len(fixtures)),
        "last_event_time": str(matches["event_time"].max().date()),
        "known_conflicts_excluded": sorted(map(list, KNOWN_CONFLICTING_ROWS)),
    }
    # Atomic replace so an interrupted run can't leave a half-written manifest.
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(manifest_path)
    return Snapshot(snapshot_id, directory, len(matches), len(fixtures))


def load_snapshot(snapshots_dir: Path, snapshot_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back a snapshot's (matches, fixtures) tables."""
    directory = snapshots_dir / snapshot_id
    return (
        pd.read_parquet(directory / "matches.parquet"),
        pd.read_parquet(directory / "fixtures.parquet"),
    )


def latest_snapshot_id(snapshots_dir: Path) -> str:
    """The most recently created snapshot id per manifest.json.

    Ties on `created` break deterministically by snapshot id.
    """
    manifest_path = snapshots_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found -- run `make data` first.")
    manifest: dict[str, dict[str, str]] = json.loads(manifest_path.read_text())
    return max(manifest, key=lambda sid: (manifest[sid]["created"], sid))


def main() -> None:
    snapshot = build_snapshot(Path("data/raw/results.csv"), Path("data/snapshots"))
    print(
        f"snapshot {snapshot.snapshot_id}: {snapshot.n_matches} matches, "
        f"{snapshot.n_fixtures} fixtures -> {snapshot.directory}"
    )


if __name__ == "__main__":
    main()
