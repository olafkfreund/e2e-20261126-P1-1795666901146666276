"""Unit and CLI tests for the wordfreq package."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wordfreq.cli import count_words, format_table, main

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_sample(tmp_path: Path, text: str, name: str = "sample.txt") -> Path:
    sample = tmp_path / name
    sample.write_text(text, encoding="utf-8")
    return sample


def _run_module(*args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wordfreq", *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


# --- count_words -----------------------------------------------------------


def test_count_words_basic_ranking() -> None:
    text = "Apple banana apple, cherry! Banana apple."
    assert count_words(text) == [("apple", 3), ("banana", 2), ("cherry", 1)]


def test_count_words_higher_count_beats_alphabetical_order() -> None:
    assert count_words("zebra zebra apple") == [("zebra", 2), ("apple", 1)]


@pytest.mark.parametrize(
    "text",
    ["zeta alpha mid", "mid zeta alpha", "alpha mid zeta"],
)
def test_count_words_ties_are_alphabetical_regardless_of_input_order(
    text: str,
) -> None:
    assert count_words(text) == [("alpha", 1), ("mid", 1), ("zeta", 1)]


def test_count_words_tie_on_count_breaks_alphabetically() -> None:
    assert count_words("pear apple fig fig pear kiwi") == [
        ("fig", 2),
        ("pear", 2),
        ("apple", 1),
        ("kiwi", 1),
    ]


def test_count_words_lowercases_by_default() -> None:
    assert count_words("Hello HELLO hello") == [("hello", 3)]


def test_count_words_preserves_case_when_ignore_case_is_false() -> None:
    assert count_words("Hello HELLO hello", ignore_case=False) == [
        ("HELLO", 1),
        ("Hello", 1),
        ("hello", 1),
    ]


def test_count_words_splits_on_non_alphanumeric_characters() -> None:
    assert count_words("foo_bar don't a-b c1 c1") == [
        ("c1", 2),
        ("a", 1),
        ("b", 1),
        ("bar", 1),
        ("don", 1),
        ("foo", 1),
        ("t", 1),
    ]


def test_count_words_only_matches_ascii_letters_and_digits() -> None:
    assert count_words("café") == [("caf", 1)]


def test_count_words_min_len_filters_short_words() -> None:
    assert count_words("a an the and to be", min_len=3) == [("and", 1), ("the", 1)]


def test_count_words_top_limits_results() -> None:
    assert count_words("a a b b c", top=2) == [("a", 2), ("b", 2)]


def test_count_words_top_zero_returns_empty_list() -> None:
    assert count_words("a b", top=0) == []


def test_count_words_empty_text_returns_empty_list() -> None:
    assert count_words("") == []


@pytest.mark.parametrize("min_len", [0, -1])
def test_count_words_rejects_min_len_below_one(min_len: int) -> None:
    with pytest.raises(ValueError, match=r"min_len must be >= 1"):
        count_words("a", min_len=min_len)


def test_count_words_rejects_negative_top() -> None:
    with pytest.raises(ValueError, match=r"top must be >= 0"):
        count_words("a", top=-1)


# --- format_table ----------------------------------------------------------


def test_format_table_tsv() -> None:
    assert format_table([("apple", 3), ("banana", 2)]) == "apple\t3\nbanana\t2"


def test_format_table_empty_returns_empty_string() -> None:
    assert format_table([]) == ""


def test_format_table_single_row_has_no_trailing_newline() -> None:
    assert format_table([("a", 1)]) == "a\t1"


# --- main: input and output ------------------------------------------------


def test_cli_top_and_min_len(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("to be or not to be that is the question", encoding="utf-8")
    rc = main([str(sample), "--min-len", "3", "--top", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines() == ["not\t1", "question\t1"]


def test_main_tie_breaking_through_top(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("c b a"))
    assert main(["--top", "2"]) == 0
    assert capsys.readouterr().out == "a\t1\nb\t1\n"


def test_main_reads_stdin_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("b a b\n"))
    assert main([]) == 0
    assert capsys.readouterr().out == "b\t2\na\t1\n"


def test_main_dash_path_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("hi hi"))
    assert main(["-"]) == 0
    assert capsys.readouterr().out == "hi\t2\n"


def test_main_tsv_output_ends_with_single_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_sample(tmp_path, "b a b")
    assert main([str(sample)]) == 0
    out = capsys.readouterr().out
    assert out == "b\t2\na\t1\n"
    assert out.endswith("\n")
    assert not out.endswith("\n\n")


def test_main_empty_tsv_output_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_sample(tmp_path, "  !!  ")
    assert main([str(sample)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_top_zero_tsv_output_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_sample(tmp_path, "a b c")
    assert main([str(sample), "--top", "0"]) == 0
    assert capsys.readouterr().out == ""


def test_main_json_output_is_array_with_trailing_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_sample(tmp_path, "b a b")
    assert main([str(sample), "--json"]) == 0
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert not out.endswith("\n\n")
    assert json.loads(out) == [
        {"word": "b", "count": 2},
        {"word": "a", "count": 1},
    ]


def test_main_json_empty_input_prints_empty_array(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["--json"]) == 0
    assert capsys.readouterr().out == "[]\n"


def test_main_missing_file_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.txt"
    assert main([str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.txt" in captured.err


# --- main: --case-sensitive ------------------------------------------------


def test_main_default_folds_case(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Apple apple APPLE apple"))
    assert main([]) == 0
    assert capsys.readouterr().out == "apple\t4\n"


def test_main_case_sensitive_counts_spellings_separately(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Apple apple APPLE apple"))
    assert main(["--case-sensitive"]) == 0
    assert capsys.readouterr().out == "apple\t2\nAPPLE\t1\nApple\t1\n"


# --- main: argument validation ---------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--top", "-1"],
        ["--min-len", "0"],
        ["--top", "abc"],
        ["--min-len", "x"],
        ["--bogus"],
    ],
)
def test_main_invalid_arguments_return_2_with_stderr(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("a b"))
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


# --- python -m wordfreq ----------------------------------------------------


def test_python_m_wordfreq_reads_stdin_and_prints_tsv() -> None:
    result = _run_module("--top", "2", stdin="the cat the dog cat the\n")
    assert result.returncode == 0
    assert result.stdout == "the\t3\ncat\t2\n"
    assert result.stderr == ""


def test_python_m_wordfreq_dash_path_reads_stdin() -> None:
    result = _run_module("-", stdin="x x y\n")
    assert result.returncode == 0
    assert result.stdout == "x\t2\ny\t1\n"


def test_python_m_wordfreq_empty_tsv_output_writes_nothing() -> None:
    result = _run_module(stdin="   \n")
    assert result.returncode == 0
    assert result.stdout == ""


def test_python_m_wordfreq_json_output_has_trailing_newline(tmp_path: Path) -> None:
    sample = _write_sample(tmp_path, "x y x")
    result = _run_module(str(sample), "--json")
    assert result.returncode == 0
    assert result.stdout.endswith("\n")
    assert not result.stdout.endswith("\n\n")
    assert json.loads(result.stdout) == [
        {"word": "x", "count": 2},
        {"word": "y", "count": 1},
    ]


def test_python_m_wordfreq_case_sensitive_flag() -> None:
    result = _run_module("--case-sensitive", stdin="Ab ab\n")
    assert result.returncode == 0
    assert result.stdout == "Ab\t1\nab\t1\n"


def test_python_m_wordfreq_missing_file_exits_2(tmp_path: Path) -> None:
    result = _run_module(str(tmp_path / "nope.txt"))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr != ""


def test_python_m_wordfreq_invalid_argument_exits_2() -> None:
    result = _run_module("--min-len", "0", stdin="a\n")
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr != ""
