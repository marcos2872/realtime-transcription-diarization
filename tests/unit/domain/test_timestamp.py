"""Tests for the Timestamp value object."""

import pytest

from transcript.domain.value_objects import Timestamp


def test_valid_timestamp_keeps_bounds():
    ts = Timestamp(start=1.5, end=4.0)

    assert ts.start == 1.5
    assert ts.end == 4.0
    assert ts.duration == 2.5


def test_zero_length_timestamp_is_allowed():
    ts = Timestamp(start=2.0, end=2.0)

    assert ts.duration == 0.0


def test_negative_start_is_rejected():
    with pytest.raises(ValueError, match="start"):
        Timestamp(start=-0.1, end=1.0)


def test_end_before_start_is_rejected():
    with pytest.raises(ValueError, match="end"):
        Timestamp(start=5.0, end=4.0)


def test_overlap_seconds_of_interleaved_intervals():
    a = Timestamp(start=0.0, end=10.0)
    b = Timestamp(start=4.0, end=12.0)

    assert a.overlap_seconds(b) == pytest.approx(6.0)
    assert b.overlap_seconds(a) == pytest.approx(6.0)


def test_disjoint_intervals_do_not_overlap():
    a = Timestamp(start=0.0, end=1.0)
    b = Timestamp(start=1.0, end=2.0)  # touching is not overlapping

    assert a.overlap_seconds(b) == 0.0
    assert not a.overlaps(b)
