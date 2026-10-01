import sys
sys.path.insert(0, "src")

from pipeline import contains_unsafe_promise, strip_urls


def test_catches_refund_promise():
    found, phrase = contains_unsafe_promise(
        "Sure, we can process a refund for you today."
    )
    assert found is True


def test_catches_merge_promise():
    found, phrase = contains_unsafe_promise(
        "We can merge your two accounts right away."
    )
    assert found is True


def test_does_not_flag_honest_hedging():
    # Critical negative test: honest hedging should not be flagged.
    found, phrase = contains_unsafe_promise(
        "I can't guarantee a timeline, but I'll check on it."
    )
    assert found is False


def test_does_not_flag_normal_reply():
    found, phrase = contains_unsafe_promise(
        "Could you DM us your account email so we can take a look?"
    )
    assert found is False


def test_strip_urls_removes_link():
    result = strip_urls(
        "Check this out [https://t.co/abc123](https://t.co/abc123) thanks"
    )
    assert "http" not in result


def test_strip_urls_leaves_normal_text_alone():
    result = strip_urls("No links here at all")
    assert result == "No links here at all"