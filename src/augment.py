"""Tier-1 data augmentations for Lean proofs.

Only two augmentations are enabled for v1 (alpha-renaming is deliberately dropped):
  - whitespace perturbation: produce a surface *variant* of the spacing
  - comment stripping: remove Lean comments (line, block, doc), nesting-aware

Augmentations are applied fresh at data-loading time, never pre-baked.
"""

import random
import re

_WS_RUN = re.compile(r"[ \t]+")


def normalize_whitespace(s: str) -> str:
    """Return a surface variant: each run of spaces/tabs becomes 1-3 spaces,
    trailing whitespace is randomly stripped, newline structure is preserved.

    Content-preserving: the non-whitespace token sequence is unchanged.
    """
    out_lines = []
    for line in s.split("\n"):
        line = _WS_RUN.sub(lambda _m: " " * random.randint(1, 3), line)
        if random.random() < 0.5:
            line = line.rstrip()
        out_lines.append(line)
    return "\n".join(out_lines)


def strip_comments(s: str) -> str:
    """Remove Lean comments with a stateful scanner (regex cannot handle nesting).

    Handles:
      - ``--`` line comments (to end of line; newline kept)
      - ``/- ... -/`` block comments, which NEST
      - ``/-- ... -/`` doc comments (treated like block comments for stripping)
      - string literals ``"..."`` are protected (markers inside are not comments),
        including escaped quotes ``\\"``
    """
    out = []
    i = 0
    n = len(s)
    depth = 0           # block-comment nesting depth
    in_string = False
    escaped = False

    while i < n:
        c = s[i]

        if depth > 0:
            # Inside a block comment: only nesting open/close matter.
            if c == "/" and i + 1 < n and s[i + 1] == "-":
                depth += 1
                i += 2
                continue
            if c == "-" and i + 1 < n and s[i + 1] == "/":
                depth -= 1
                i += 2
                continue
            i += 1
            continue

        if in_string:
            out.append(c)
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            i += 1
            continue

        # Normal code.
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "-":
            # Opens both /- and /-- (doc); matching /- first handles both.
            depth += 1
            i += 2
            continue
        if c == "-" and i + 1 < n and s[i + 1] == "-":
            # Line comment: skip to (but not including) the next newline.
            j = i + 2
            while j < n and s[j] != "\n":
                j += 1
            i = j
            continue

        out.append(c)
        i += 1

    return "".join(out)


def augment(s: str, comment_prob: float = 0.5) -> str:
    """Apply whitespace perturbation always, comment stripping with prob ~0.5.

    Returns a fresh variant on each call (relies on the global RNG state).
    """
    if random.random() < comment_prob:
        s = strip_comments(s)
    return normalize_whitespace(s)
