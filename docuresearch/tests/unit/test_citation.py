"""Tests for citation location formatting (Epic 6)."""

from __future__ import annotations

import pytest

from app.citation import format_location


@pytest.mark.parametrize(
    ("location", "label"),
    [
        ("page:2:chunk:1", "Page 2"),
        ("page:10:chunk:14", "Page 10"),
        ("paragraph:5:chunk:4", "Paragraph 5"),
        ("Sample Technical Document / Security Requirements:chunk:1",
         "Section: Sample Technical Document › Security Requirements"),
        ("Introduction:chunk:0", "Section: Introduction"),
        ("¶chunk:3", "Passage 4"),
        ("page:1", "Page 1"),  # no chunk suffix
        ("unknown-format", "unknown-format"),
    ],
)
def test_format_location(location, label):
    assert format_location(location) == label
