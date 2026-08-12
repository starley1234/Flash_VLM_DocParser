from __future__ import annotations

import pytest

from flash_vlm.utils import parse_pages


def test_parse_pages_all():
    assert parse_pages(None, 10) == list(range(1, 11))
    assert parse_pages("", 10) == list(range(1, 11))


def test_parse_pages_spec():
    assert parse_pages("1-3,5", 10) == [1, 2, 3, 5]
    assert parse_pages(" 2 , 4 - 6 ", 10) == [2, 4, 5, 6]


def test_parse_pages_list_and_int():
    assert parse_pages([3, 1, 2], 5) == [3, 1, 2]
    assert parse_pages(4, 5) == [4]


def test_parse_pages_dedup():
    assert parse_pages("1,1,2-3,3", 5) == [1, 2, 3]


def test_parse_pages_invalid():
    with pytest.raises(ValueError):
        parse_pages("0", 5)
    with pytest.raises(ValueError):
        parse_pages("99", 5)
    with pytest.raises(ValueError):
        parse_pages("3-1", 5)  # некорректный диапазон? (start > end отдаст пустой range)
