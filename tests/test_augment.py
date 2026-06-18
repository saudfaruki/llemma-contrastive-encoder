import random

from src.augment import augment, normalize_whitespace, strip_comments


# ---------------- normalize_whitespace ----------------

def test_normalize_preserves_nonwhitespace_content():
    # Collapsing/expanding whitespace must not change the token sequence.
    random.seed(0)
    s = "theorem  foo\t: 1   = 1 :=\n  by\trfl  \n"
    out = normalize_whitespace(s)
    assert "".join(out.split()) == "".join(s.split())


def test_normalize_preserves_newline_count():
    random.seed(1)
    s = "line one\n  line two\n\nline four\n"
    out = normalize_whitespace(s)
    assert out.count("\n") == s.count("\n")


def test_normalize_removes_tabs_and_bounds_space_runs():
    random.seed(2)
    s = "a\t\tb     c\td"
    for _ in range(20):
        out = normalize_whitespace(s)
        assert "\t" not in out
        assert "    " not in out  # no run of 4+ spaces (runs collapse to 1-3)


def test_normalize_is_a_variant_not_just_canonical():
    # Over many randomized calls the surface form should actually vary.
    random.seed(3)
    s = "a   b   c   d   e"
    seen = {normalize_whitespace(s) for _ in range(50)}
    assert len(seen) > 1


# ---------------- strip_comments ----------------

def test_strip_line_comment_keeps_code_and_newline():
    s = "x := 1 -- set x to one\ny := 2"
    out = strip_comments(s)
    assert "set x to one" not in out
    assert "y := 2" in out
    assert out.count("\n") == 1


def test_strip_simple_block_comment():
    s = "a /- comment -/ b"
    out = strip_comments(s)
    assert "comment" not in out
    assert out == "a  b"


def test_strip_nested_block_comment():
    # The decisive case: a regex stops at the first '-/' and leaks 'still outer'.
    s = "a /- outer /- inner -/ still outer -/ b"
    out = strip_comments(s)
    assert "outer" not in out
    assert "inner" not in out
    assert "still" not in out
    assert out == "a  b"


def test_strip_doc_comment_treated_like_block():
    s = "/-- This is a doc comment -/\ndef foo := 1"
    out = strip_comments(s)
    assert "doc comment" not in out
    assert "def foo := 1" in out


def test_string_literal_protects_comment_markers():
    s = 'let s := "/- not a comment -/"  -- real comment'
    out = strip_comments(s)
    assert "/- not a comment -/" in out      # markers inside the string survive
    assert "real comment" not in out          # the real trailing comment is gone


def test_escaped_quote_inside_string():
    s = r'x := "a\"b" -- c'
    out = strip_comments(s)
    assert r'a\"b' in out      # escaped quote does not prematurely close the string
    assert "c" not in out      # trailing comment removed


def test_strip_comments_idempotent_on_clean_code():
    s = "theorem foo : 1 = 1 := rfl"
    assert strip_comments(s) == s


# ---------------- augment ----------------

def test_augment_returns_str_and_preserves_code_tokens():
    random.seed(0)
    s = "theorem foo : 1 = 1 := rfl -- trivial"
    code_tokens = "theorem foo : 1 = 1 := rfl".split()
    for _ in range(30):
        out = augment(s)
        assert isinstance(out, str)
        assert set(code_tokens).issubset(set(out.split()))


def test_augment_strips_comments_sometimes_and_keeps_them_sometimes():
    # comment-strip fires with prob ~0.5; over 80 trials we should see both outcomes.
    random.seed(7)
    s = "def foo := 1 -- UNIQUEMARKERXYZ"
    stripped = kept = 0
    for _ in range(80):
        out = augment(s)
        if "UNIQUEMARKERXYZ" in out:
            kept += 1
        else:
            stripped += 1
    assert stripped > 0
    assert kept > 0
