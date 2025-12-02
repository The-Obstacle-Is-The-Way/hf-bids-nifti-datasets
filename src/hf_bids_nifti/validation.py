"""
Generic BIDS dataset validation framework.

This module provides reusable validation utilities for BIDS datasets:
- ValidationCheck and ValidationResult dataclasses for structured results
- Generic BIDS validation (required files exist)
- NIfTI integrity spot-checking via nibabel

Why validation matters:
OpenNeuro datasets are DataLad/git-annex repos with internal checksums,
but raw AWS S3 downloads (the most common method) bypass this integrity layer.
This module helps verify that downloaded data matches expectations before
pushing to HuggingFace Hub.

Usage pattern for dataset-specific validation:
    1. Copy this module's patterns to your dataset module
    2. Add dataset-specific checks (expected counts, series validation)
    3. Call generic checks + your specific checks in a validate function

Example:
    ```python
    from hf_bids_nifti.validation import (
        ValidationResult,
        validate_bids_required_files,
        spot_check_nifti_files,
    )

    def validate_my_dataset(bids_root: Path) -> ValidationResult:
        result = ValidationResult(bids_root)

        # Generic BIDS checks
        result.add_check(validate_bids_required_files(bids_root))
        result.add_check(spot_check_nifti_files(bids_root, sample_size=10))

        # Dataset-specific checks
        result.add_check(check_subject_count(bids_root, expected=100))

        return result
    ```
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    expected: str
    actual: str
    details: str | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = f"[{status}] {self.name}: expected={self.expected}, actual={self.actual}"
        if self.details:
            msg += f" ({self.details})"
        return msg


@dataclass
class ValidationResult:
    """Aggregated result of multiple validation checks."""

    bids_root: Path
    checks: list[ValidationCheck] = field(default_factory=list)

    def add_check(self, check: ValidationCheck) -> None:
        """Add a validation check to the results."""
        self.checks.append(check)

    @property
    def all_passed(self) -> bool:
        """Return True if all checks passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list[ValidationCheck]:
        """Return list of failed checks."""
        return [check for check in self.checks if not check.passed]

    def summary(self) -> str:
        """Generate human-readable summary of validation results."""
        lines = [
            f"Validation Results for: {self.bids_root}",
            "=" * 60,
        ]

        for check in self.checks:
            status = "\u2705 PASS" if check.passed else "\u274c FAIL"
            lines.append(f"{status} {check.name}")
            lines.append(f"       Expected: {check.expected}")
            lines.append(f"       Actual:   {check.actual}")
            if check.details:
                lines.append(f"       Details:  {check.details}")

        lines.append("=" * 60)
        if self.all_passed:
            lines.append("\u2705 All validations passed! Data is ready for HF push.")
        else:
            failed_names = [c.name for c in self.failed_checks]
            lines.append(f"\u274c {len(failed_names)} check(s) failed: {failed_names}")
            lines.append("   Check download completion or data integrity.")

        return "\n".join(lines)


def validate_bids_required_files(
    bids_root: Path,
    required_files: list[str] | None = None,
) -> ValidationCheck:
    """
    Validate that required BIDS files exist.

    Args:
        bids_root: Path to BIDS dataset root.
        required_files: List of required files. Defaults to standard BIDS files.

    Returns:
        ValidationCheck with pass/fail result.
    """
    if required_files is None:
        required_files = [
            "dataset_description.json",
            "participants.tsv",
        ]

    missing = [f for f in required_files if not (bids_root / f).exists()]

    if missing:
        return ValidationCheck(
            name="bids_required_files",
            passed=False,
            expected="all present",
            actual=f"missing: {', '.join(missing)}",
            details=f"Required: {required_files}",
        )
    else:
        return ValidationCheck(
            name="bids_required_files",
            passed=True,
            expected="all present",
            actual="all present",
        )


def spot_check_nifti_files(
    bids_root: Path,
    pattern: str = "**/*_T1w.nii.gz",
    sample_size: int = 10,
) -> ValidationCheck:
    """
    Spot-check that a sample of NIfTI files are loadable.

    This catches corrupted downloads without having to load all files.
    Uses nibabel to verify file headers are readable.

    Args:
        bids_root: Path to BIDS dataset root.
        pattern: Glob pattern to find NIfTI files. Default: T1w images.
        sample_size: Number of files to check. Default: 10.

    Returns:
        ValidationCheck with pass/fail result.
    """
    nifti_files = list(bids_root.glob(pattern))

    if not nifti_files:
        return ValidationCheck(
            name="nifti_integrity",
            passed=False,
            expected="loadable NIfTI files",
            actual="no files found",
            details=f"Pattern: {pattern}",
        )

    # Sample randomly for large datasets
    sample = random.sample(nifti_files, min(sample_size, len(nifti_files)))
    actual_sample_size = len(sample)

    try:
        import nibabel as nib

        for f in sample:
            # Load header only (fast, catches most corruption)
            _ = nib.load(f).header
    except Exception as e:  # noqa: BLE001 - intentionally broad to catch any IO/nibabel error
        return ValidationCheck(
            name="nifti_integrity",
            passed=False,
            expected=f"{actual_sample_size}/{actual_sample_size} loadable",
            actual=f"ERROR: {e}",
            details=f"Pattern: {pattern}",
        )

    return ValidationCheck(
        name="nifti_integrity",
        passed=True,
        expected=f"{actual_sample_size}/{actual_sample_size} loadable",
        actual=f"{actual_sample_size}/{actual_sample_size} passed",
        details=f"Checked {actual_sample_size} of {len(nifti_files)} files",
    )


def count_subjects(bids_root: Path) -> int:
    """Count subject directories in a BIDS dataset."""
    return sum(1 for p in bids_root.glob("sub-*") if p.is_dir())


def count_files(bids_root: Path, pattern: str) -> int:
    """Count files matching a glob pattern."""
    return len(list(bids_root.rglob(pattern)))


def validate_count(
    name: str,
    actual: int,
    expected_min: int,
    expected_max: int | None = None,
) -> ValidationCheck:
    """
    Create a validation check for a count value.

    Args:
        name: Name of the check (e.g., "subjects", "t1w_files").
        actual: Actual count.
        expected_min: Minimum expected count.
        expected_max: Maximum expected count (optional).

    Returns:
        ValidationCheck with pass/fail result.
    """
    if expected_max is None:
        expected_str = f">= {expected_min}"
        passed = actual >= expected_min
    else:
        expected_str = f"{expected_min}-{expected_max}"
        passed = expected_min <= actual <= expected_max

    return ValidationCheck(
        name=name,
        passed=passed,
        expected=expected_str,
        actual=str(actual),
        details=f"Expected {expected_str}, got {actual}" if not passed else None,
    )


def validate_generic_bids(
    bids_root: Path,
    nifti_pattern: str = "**/*_T1w.nii.gz",
    nifti_sample_size: int = 10,
) -> ValidationResult:
    """
    Run generic BIDS validation checks.

    This provides baseline validation applicable to any BIDS dataset:
    - Required BIDS files exist
    - Sample NIfTI files are loadable

    For dataset-specific validation (subject counts, series counts),
    implement a custom validation function in your dataset module.

    Args:
        bids_root: Path to BIDS dataset root.
        nifti_pattern: Glob pattern for NIfTI files to spot-check.
        nifti_sample_size: Number of NIfTI files to sample.

    Returns:
        ValidationResult with generic checks.

    Example:
        ```python
        result = validate_generic_bids(Path("/data/ds000001"))
        if result.all_passed:
            print("Basic BIDS structure is valid")
        else:
            print(result.summary())
        ```
    """
    if not bids_root.exists():
        result = ValidationResult(bids_root)
        result.add_check(
            ValidationCheck(
                name="bids_root_exists",
                passed=False,
                expected="directory exists",
                actual="not found",
                details=str(bids_root),
            )
        )
        return result

    result = ValidationResult(bids_root)

    # 1. Required BIDS files
    result.add_check(validate_bids_required_files(bids_root))

    # 2. NIfTI integrity spot-check
    result.add_check(
        spot_check_nifti_files(bids_root, pattern=nifti_pattern, sample_size=nifti_sample_size)
    )

    return result
