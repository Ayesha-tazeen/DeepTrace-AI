"""Tests for filename / path safety helpers."""

import pytest

from deeptrace.ingestion.path_safety import (
    check_filename,
    check_path_string,
    sanitize_filename,
)


@pytest.mark.parametrize("name", ["..", "a..b.jpg", "CON.png", "bad|name.png", "x.png.", " x.png", "a\x00b.png", ""])
def test_unsafe_filenames_rejected(name):
    assert check_filename(name)


def test_safe_filename_ok():
    assert check_filename("holiday photo (1).jpg") == []


def test_path_traversal_rejected():
    assert check_path_string("../../etc/passwd")
    assert check_path_string("data/../secret.png")


def test_unc_path_rejected():
    assert check_path_string("\\\\server\\share\\x.png")


def test_sanitize_filename():
    assert sanitize_filename("my file (1)!.png") == "my_file_1_.png"
    assert sanitize_filename("../../evil") == "evil"
    assert sanitize_filename("CON") == "file"
