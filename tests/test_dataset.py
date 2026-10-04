"""Tests for failfast.dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from failfast.dataset import (
    build_labeled_dataset,
    load_dataset,
    save_dataset,
    time_split,
)


def make_df(node_ids: list[str]) -> pd.DataFrame:
    """Return a minimal feature DataFrame indexed by node_id."""
    return pd.DataFrame(
        {
            "overlap_lines": [0] * len(node_ids),
            "failure_rate": [0.0] * len(node_ids),
        },
        index=pd.Index(node_ids, name="node_id"),
    )


class TestBuildLabeledDataset:
    def test_failed_outcome_gets_label_1(self) -> None:
        df = make_df(["t::a", "t::b"])
        outcome_map = {"t::a": "failed", "t::b": "passed"}
        labeled = build_labeled_dataset(df, outcome_map)
        assert labeled.loc["t::a", "label"] == 1
        assert labeled.loc["t::b", "label"] == 0

    def test_error_outcome_gets_label_1(self) -> None:
        df = make_df(["t::a"])
        labeled = build_labeled_dataset(df, {"t::a": "error"})
        assert labeled.loc["t::a", "label"] == 1

    def test_skipped_outcome_gets_label_0(self) -> None:
        df = make_df(["t::a"])
        labeled = build_labeled_dataset(df, {"t::a": "skipped"})
        assert labeled.loc["t::a", "label"] == 0

    def test_missing_outcome_gets_label_0(self) -> None:
        df = make_df(["t::a"])
        labeled = build_labeled_dataset(df, {})
        assert labeled.loc["t::a", "label"] == 0

    def test_is_synthetic_column_added_when_flagged(self) -> None:
        df = make_df(["t::a"])
        labeled = build_labeled_dataset(df, {}, is_synthetic=True)
        assert "is_synthetic" in labeled.columns
        assert labeled.loc["t::a", "is_synthetic"] == 1

    def test_is_synthetic_column_absent_by_default(self) -> None:
        df = make_df(["t::a"])
        labeled = build_labeled_dataset(df, {})
        assert "is_synthetic" not in labeled.columns

    def test_original_df_not_mutated(self) -> None:
        df = make_df(["t::a"])
        original_cols = list(df.columns)
        build_labeled_dataset(df, {"t::a": "failed"})
        assert list(df.columns) == original_cols


class TestTimeSplit:
    def _make_temporal_df(self, n: int) -> pd.DataFrame:
        base = 1_000_000.0
        return pd.DataFrame(
            {
                "commit_timestamp": [base + i for i in range(n)],
                "feature": list(range(n)),
            }
        )

    def test_basic_split_ratio(self) -> None:
        df = self._make_temporal_df(10)
        train, test = time_split(df, train_ratio=0.8)
        assert len(train) == 8
        assert len(test) == 2

    def test_train_is_earlier_than_test(self) -> None:
        df = self._make_temporal_df(20)
        train, test = time_split(df, train_ratio=0.7)
        assert train["commit_timestamp"].max() <= test["commit_timestamp"].min()

    def test_no_data_lost(self) -> None:
        df = self._make_temporal_df(15)
        train, test = time_split(df, train_ratio=0.6)
        assert len(train) + len(test) == 15

    def test_invalid_ratio_raises(self) -> None:
        df = self._make_temporal_df(10)
        with pytest.raises(ValueError, match="train_ratio"):
            time_split(df, train_ratio=0.0)
        with pytest.raises(ValueError, match="train_ratio"):
            time_split(df, train_ratio=1.0)

    def test_missing_timestamp_column_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(KeyError):
            time_split(df)

    def test_rows_are_sorted_by_timestamp(self) -> None:
        """Shuffled input must be sorted before splitting."""
        df = pd.DataFrame(
            {
                "commit_timestamp": [30.0, 10.0, 20.0],
                "v": ["c", "a", "b"],
            }
        )
        train, test = time_split(df, train_ratio=0.67)
        assert train["v"].tolist() == ["a", "b"]
        assert test["v"].tolist() == ["c"]


class TestSaveAndLoadDataset:
    def test_roundtrip_parquet(self, tmp_path: Path) -> None:
        df = make_df(["t::a", "t::b"])
        df["label"] = [0, 1]
        path = save_dataset(df, tmp_path, "test_ds")
        loaded = load_dataset(path)
        assert list(loaded.index) == list(df.index)
        assert "label" in loaded.columns

    def test_manifest_written(self, tmp_path: Path) -> None:
        df = make_df(["t::a"])
        df["label"] = [1]
        save_dataset(df, tmp_path, "my_ds")
        manifest_path = tmp_path / "my_ds_manifest.json"
        assert manifest_path.exists()

    def test_manifest_synthetic_flag(self, tmp_path: Path) -> None:
        import json

        df = make_df(["t::a"])
        df["label"] = [0]
        save_dataset(df, tmp_path, "synth_ds", is_synthetic=True)
        manifest = json.loads((tmp_path / "synth_ds_manifest.json").read_text())
        assert manifest["synthetic"] is True

    def test_output_dir_created(self, tmp_path: Path) -> None:
        df = make_df(["t::a"])
        df["label"] = [0]
        new_dir = tmp_path / "sub" / "dir"
        save_dataset(df, new_dir, "ds")
        assert new_dir.exists()

    def test_load_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "file.txt"
        bad.write_text("oops")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(bad)
