"""Answers cite numbered sources; citations are parsed and checked deterministically."""

from rag_lab.domain.answers import ABSTENTION, check_citations, extract_citations, is_abstention


def test_extracts_bracketed_source_numbers_in_order_of_first_use() -> None:
    text = "Rates rose [2]. Savings too [1, 3][2]. See also [ 4 ]."

    assert extract_citations(text) == (2, 1, 3, 4)


def test_ignores_brackets_that_are_not_citations() -> None:
    assert extract_citations("Use list[0] or [a] or [] or [1.5] or [-2].") == ()


def test_citations_are_valid_only_when_present_and_within_the_sources() -> None:
    assert check_citations("Yes [1] and [2].", source_count=2).valid
    missing = check_citations("Yes, rates rose.", source_count=2)
    unknown = check_citations("Yes [1] and [7].", source_count=2)

    assert not missing.valid
    assert missing.cited == ()
    assert not unknown.valid
    assert unknown.invalid == (7,)
    assert check_citations("Zero [0].", source_count=2).invalid == (0,)


def test_abstention_is_the_sentinel_at_the_start_of_the_answer() -> None:
    assert is_abstention(f"{ABSTENTION}")
    assert is_abstention(f"  {ABSTENTION.lower()}: the sources do not say.")
    assert is_abstention(f"**{ABSTENTION}**: nothing about dates.")
    assert is_abstention(f"> {ABSTENTION}")
    assert is_abstention(f"\r\n\u00a0**{ABSTENTION}**")
    assert not is_abstention(f"The answer is 42 [1]. Not {ABSTENTION}.")
