"""Tests for the validation module."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hf_bids_nifti.validation import (
    ValidationCheck,
    ValidationResult,
    count_files,
    count_subjects,
    spot_check_nifti_files,
    validate_bids_required_files,
    validate_count,
    validate_generic_bids,
)


@pytest.fixture
def minimal_bids_dir() -> Generator[Path, None, None]:
    """Create a minimal BIDS-like directory structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = Path(tmpdir)

        # Create required BIDS files
        (bids_root / "dataset_description.json").write_text('{"Name": "Test"}')
        (bids_root / "participants.tsv").write_text("participant_id\nsub-001\nsub-002\n")

        # Create subject directories with NIfTI files
        for i in range(3):
            subj_dir = bids_root / f"sub-{i:03d}" / "anat"
            subj_dir.mkdir(parents=True)

            # Create minimal NIfTI file
            data = np.ones((2, 2, 2), dtype=np.float32) * (i + 1)
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, subj_dir / f"sub-{i:03d}_T1w.nii.gz")

        yield bids_root


class TestValidationCheck:
    """Tests for ValidationCheck dataclass."""

    def test_passed_check_str(self) -> None:
        """Test string representation of passed check."""
        check = ValidationCheck(
            name="test_check",
            passed=True,
            expected="100",
            actual="100",
        )
        result = str(check)
        assert "[PASS]" in result
        assert "test_check" in result

    def test_failed_check_str(self) -> None:
        """Test string representation of failed check."""
        check = ValidationCheck(
            name="test_check",
            passed=False,
            expected="100",
            actual="50",
            details="Missing 50 items",
        )
        result = str(check)
        assert "[FAIL]" in result
        assert "test_check" in result
        assert "Missing 50 items" in result


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_all_passed_true(self, minimal_bids_dir: Path) -> None:
        """Test all_passed property when all checks pass."""
        result = ValidationResult(minimal_bids_dir)
        result.add_check(ValidationCheck("a", True, "1", "1"))
        result.add_check(ValidationCheck("b", True, "2", "2"))
        assert result.all_passed is True

    def test_all_passed_false(self, minimal_bids_dir: Path) -> None:
        """Test all_passed property when some checks fail."""
        result = ValidationResult(minimal_bids_dir)
        result.add_check(ValidationCheck("a", True, "1", "1"))
        result.add_check(ValidationCheck("b", False, "2", "0"))
        assert result.all_passed is False

    def test_failed_checks(self, minimal_bids_dir: Path) -> None:
        """Test failed_checks property."""
        result = ValidationResult(minimal_bids_dir)
        result.add_check(ValidationCheck("a", True, "1", "1"))
        result.add_check(ValidationCheck("b", False, "2", "0"))
        result.add_check(ValidationCheck("c", False, "3", "0"))

        failed = result.failed_checks
        assert len(failed) == 2
        assert all(not c.passed for c in failed)

    def test_summary_contains_all_checks(self, minimal_bids_dir: Path) -> None:
        """Test that summary includes all checks."""
        result = ValidationResult(minimal_bids_dir)
        result.add_check(ValidationCheck("check_one", True, "1", "1"))
        result.add_check(ValidationCheck("check_two", False, "2", "0"))

        summary = result.summary()
        assert "check_one" in summary
        assert "check_two" in summary
        assert "PASS" in summary
        assert "FAIL" in summary


class TestValidateBidsRequiredFiles:
    """Tests for validate_bids_required_files function."""

    def test_all_files_present(self, minimal_bids_dir: Path) -> None:
        """Test validation passes when all required files exist."""
        check = validate_bids_required_files(minimal_bids_dir)
        assert check.passed is True
        assert check.actual == "all present"

    def test_missing_files(self, minimal_bids_dir: Path) -> None:
        """Test validation fails when required files are missing."""
        # Remove a required file
        (minimal_bids_dir / "dataset_description.json").unlink()

        check = validate_bids_required_files(minimal_bids_dir)
        assert check.passed is False
        assert "missing" in check.actual.lower()
        assert "dataset_description.json" in check.actual

    def test_custom_required_files(self, minimal_bids_dir: Path) -> None:
        """Test validation with custom required files list."""
        check = validate_bids_required_files(
            minimal_bids_dir,
            required_files=["dataset_description.json", "nonexistent.json"],
        )
        assert check.passed is False
        assert "nonexistent.json" in check.actual


class TestSpotCheckNiftiFiles:
    """Tests for spot_check_nifti_files function."""

    def test_valid_nifti_files(self, minimal_bids_dir: Path) -> None:
        """Test spot check passes for valid NIfTI files."""
        check = spot_check_nifti_files(minimal_bids_dir, pattern="**/*_T1w.nii.gz")
        assert check.passed is True
        assert "passed" in check.actual

    def test_no_files_found(self, minimal_bids_dir: Path) -> None:
        """Test spot check fails when no files match pattern."""
        check = spot_check_nifti_files(minimal_bids_dir, pattern="**/*_nonexistent.nii.gz")
        assert check.passed is False
        assert "no files found" in check.actual.lower()

    def test_sample_size_respected(self, minimal_bids_dir: Path) -> None:
        """Test that sample size limits how many files are checked."""
        check = spot_check_nifti_files(
            minimal_bids_dir,
            pattern="**/*_T1w.nii.gz",
            sample_size=2,
        )
        assert check.passed is True
        # Should check 2 of 3 files
        assert "2/2" in check.actual


class TestCountFunctions:
    """Tests for count_subjects and count_files functions."""

    def test_count_subjects(self, minimal_bids_dir: Path) -> None:
        """Test subject counting."""
        count = count_subjects(minimal_bids_dir)
        assert count == 3

    def test_count_files(self, minimal_bids_dir: Path) -> None:
        """Test file counting with glob pattern."""
        count = count_files(minimal_bids_dir, "*_T1w.nii.gz")
        assert count == 3


class TestValidateCount:
    """Tests for validate_count function."""

    def test_count_within_range(self) -> None:
        """Test validation passes when count is within range."""
        check = validate_count("test", actual=50, expected_min=40, expected_max=60)
        assert check.passed is True

    def test_count_below_min(self) -> None:
        """Test validation fails when count is below minimum."""
        check = validate_count("test", actual=30, expected_min=40, expected_max=60)
        assert check.passed is False

    def test_count_above_max(self) -> None:
        """Test validation fails when count is above maximum."""
        check = validate_count("test", actual=70, expected_min=40, expected_max=60)
        assert check.passed is False

    def test_count_min_only(self) -> None:
        """Test validation with only minimum specified."""
        check = validate_count("test", actual=100, expected_min=50)
        assert check.passed is True
        assert ">=" in check.expected


class TestValidateGenericBids:
    """Tests for validate_generic_bids function."""

    def test_valid_bids_dataset(self, minimal_bids_dir: Path) -> None:
        """Test validation passes for valid BIDS dataset."""
        result = validate_generic_bids(minimal_bids_dir)
        assert result.all_passed is True
        assert len(result.checks) >= 2  # At least required files and NIfTI check

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        """Test validation fails for nonexistent path."""
        result = validate_generic_bids(tmp_path / "nonexistent")
        assert result.all_passed is False
        assert "bids_root_exists" in result.checks[0].name

    def test_missing_required_files(self, minimal_bids_dir: Path) -> None:
        """Test validation fails when required files are missing."""
        (minimal_bids_dir / "dataset_description.json").unlink()
        result = validate_generic_bids(minimal_bids_dir)
        assert result.all_passed is False
