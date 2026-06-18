"""Demo: show augmentations on 5 sampled real proofs + confirm the tricky
comment cases (nested block comment, doc comment). Run:

    .venv/bin/python scripts/augment_demo.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.augment import normalize_whitespace, strip_comments, augment  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "elementary_nt_proofs_v3.jsonl")


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    random.seed(0)
    records = load(DATA)
    # Prefer proofs that actually contain comments so stripping is visible.
    with_comments = [r for r in records
                     if "--" in r["full_source"] or "/-" in r["full_source"]]
    pool = with_comments if len(with_comments) >= 5 else records
    sample = random.sample(pool, 5)

    banner("5 SAMPLED PROOFS: before / after")
    for k, rec in enumerate(sample, 1):
        src = rec["full_source"]
        print(f"\n--- [{k}] {rec['full_name']} "
              f"({'has-comments' if ('--' in src or '/-' in src) else 'no-comments'}) ---")
        print("ORIGINAL:\n" + src)
        print("\nnormalize_whitespace ->\n" + normalize_whitespace(src))
        print("\nstrip_comments ->\n" + strip_comments(src))
        print("\naugment (fresh variant) ->\n" + augment(src))

    banner("NESTED BLOCK COMMENT (regex would leak the tail)")
    nested = "lemma L : True := by\n  /- outer /- inner -/ still outer -/ trivial"
    out = strip_comments(nested)
    print("ORIGINAL:\n" + nested)
    print("\nstrip_comments ->\n" + out)
    assert "outer" not in out and "inner" not in out and "still" not in out, "nested FAILED"
    assert "trivial" in out, "nested dropped code"
    print("\n[OK] nested block comment fully removed, code preserved.")

    banner("DOC COMMENT (/-- ... -/) treated like a block comment")
    doc = "/-- The fundamental theorem. -/\ntheorem ft : 1 = 1 := rfl"
    out = strip_comments(doc)
    print("ORIGINAL:\n" + doc)
    print("\nstrip_comments ->\n" + out)
    assert "fundamental theorem" not in out, "doc FAILED"
    assert "theorem ft : 1 = 1 := rfl" in out, "doc dropped code"
    print("\n[OK] doc comment removed, theorem preserved.")

    print("\nALL DEMO CHECKS PASSED")


if __name__ == "__main__":
    main()
