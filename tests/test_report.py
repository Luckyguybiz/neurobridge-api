"""Tests for full report generation."""

from analysis.report import generate_full_report


def test_full_report(synthetic_data):
    result = generate_full_report(synthetic_data)
    assert isinstance(result, dict)
    # Should contain multiple analysis sections
    assert len(result) >= 5, f"Full report should have 5+ sections, got {len(result)}"


def test_full_report_small(small_data):
    """Test full report with minimal data."""
    result = generate_full_report(small_data)
    assert isinstance(result, dict)
    assert len(result) >= 3
