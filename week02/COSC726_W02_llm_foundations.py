#!/usr/bin/env python3
"""
COSC726 - Agentic Artificial Intelligence
Week 2 / Lab 1 - llm_foundations.py  (COMPLETED SOLUTION)

Treat a model interface as an object of measurement. This file uses ONLY the
Python standard library - no API key, no network, no third-party packages.

You will implement THREE functions, each marked with `TODO`:
    1. count_tokens(text, tokenizer)   - token counting for a teaching tokenizer
    2. prepare_context(...)            - explicit context budgeting
    3. sample_next(distribution, ...)  - a transparent sampler

Run the self-test when you are done:
    python COSC726_W02_llm_foundations.py --self-test

Everything is deterministic. The running example is the customer-support agent
helping Layla with order #A1032, carried over from Week 1.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Two deliberately DIFFERENT teaching tokenizers. Neither is authoritative; the
# point is that the same text costs a different number of tokens under each.
# ─────────────────────────────────────────────────────────────────────────────
TOKENIZER_A = {
    "vocab": ["order", "agent", "the", "credit", "policy", "late", "1043",
              "ic", "ing", "ed", " ", "-", "A", "#"],
    "name": "teach-A",
}
TOKENIZER_B = {
    "vocab": ["order", "ag", "ent", "the", "cred", "it", "pol", "icy", "late",
              "10", "43", "ic", "ing", "ed", " ", "-", "A", "#"],
    "name": "teach-B",
}


def _greedy_split(text: str, vocab: list[str]) -> list[str]:
    """Greedy longest-match tokenisation against a vocab; unknown chars stand alone."""
    vocab = sorted(vocab, key=len, reverse=True)
    text, out, i = text.lower(), [], 0
    while i < len(text):
        for v in vocab:
            if v and text.startswith(v.lower(), i):
                out.append(v)
                i += len(v)
                break
        else:
            out.append(text[i])
            i += 1
    return out


def count_tokens(text: str, tokenizer: dict) -> int:
    """
    TODO 1
    Return the NUMBER OF TOKENS `text` produces under `tokenizer`.

    Use the provided helper `_greedy_split(text, tokenizer["vocab"])` to get the
    list of token strings, then return how many there are.
    """
    tokens = _greedy_split(text, tokenizer["vocab"])
    return len(tokens)


@dataclass
class ContextPlan:
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    rejected: bool = False
    reason: str = ""


def prepare_context(messages: list[str], context_limit: int, reserved_output: int,
                    tokenizer: dict, strategy: str = "drop_oldest") -> ContextPlan:
    """
    TODO 2
    Fit `messages` into an explicit token budget and RECORD what happened.
    """
    budget = context_limit - reserved_output
    total_tokens = lambda msgs: sum(count_tokens(m, tokenizer) for m in msgs)

    if strategy == "reject":
        current_total = total_tokens(messages)
        if current_total > budget:
            return ContextPlan(
                kept=[],
                dropped=[],
                rejected=True,
                reason=f"input {current_total} > budget {budget}"
            )
        return ContextPlan(
            kept=list(messages),
            dropped=[],
            rejected=False,
            reason=""
        )

    elif strategy == "drop_oldest":
        kept = list(messages)
        dropped = []

        while total_tokens(kept) > budget and len(kept) > 1:
            dropped.append(kept.pop(1))  # Keep system message at index 0

        is_rejected = total_tokens(kept) > budget
        reason = f"input {total_tokens(kept)} > budget {budget}" if is_rejected else ""

        return ContextPlan(
            kept=kept,
            dropped=dropped,
            rejected=is_rejected,
            reason=reason
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def sample_next(distribution: dict[str, float], temperature: float,
                rng: random.Random) -> str:
    """
    TODO 3
    Return ONE token sampled from `distribution` (token -> probability).
    """
    if temperature <= 0:
        max_val = max(distribution.values())
        # Break ties alphabetically
        best_tokens = sorted([k for k, v in distribution.items() if v == max_val])
        return best_tokens[0]

    # Temperature scaling & softmax normalization
    scaled = {k: math.exp(math.log(v) / temperature) for k, v in distribution.items() if v > 0}
    total_weight = sum(scaled.values())

    # Cumulative random walk using rng
    r = rng.random()
    acc = 0.0
    for k in sorted(scaled.keys()):
        acc += scaled[k] / total_weight
        if r <= acc:
            return k

    return sorted(scaled.keys())[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test harness (provided - do not edit). Run with --self-test.
# ─────────────────────────────────────────────────────────────────────────────
def _run_self_test() -> int:
    failures = []

    # 1 - token counting differs across tokenizers
    try:
        a = count_tokens("order A-1043", TOKENIZER_A)
        b = count_tokens("order A-1043", TOKENIZER_B)
        assert isinstance(a, int) and isinstance(b, int), "counts must be ints"
        assert a > 0 and b > 0, "counts must be positive"
    except Exception as e:  # noqa: BLE001
        failures.append(f"count_tokens: {e}")

    # 2 - context budgeting: reject vs drop_oldest
    try:
        msgs = ["system rules", "turn one", "turn two", "turn three about the credit"]
        rej = prepare_context(msgs, context_limit=8, reserved_output=4,
                              tokenizer=TOKENIZER_A, strategy="reject")
        assert rej.rejected is True and rej.reason, "reject must set rejected + reason"
        drop = prepare_context(msgs, context_limit=40, reserved_output=4,
                               tokenizer=TOKENIZER_A, strategy="drop_oldest")
        assert drop.kept and drop.kept[0] == "system rules", "system message must be kept"
    except Exception as e:  # noqa: BLE001
        failures.append(f"prepare_context: {e}")

    # 3 - sampler: greedy is deterministic; argmax picks the mode
    try:
        dist = {"Paris": 0.82, "London": 0.11, "Lyon": 0.05, "Rome": 0.02}
        picks = {sample_next(dist, 0.0, random.Random(s)) for s in range(5)}
        assert picks == {"Paris"}, "temperature 0 must always return the mode"
        hot = sample_next(dist, 1.0, random.Random(1))
        assert hot in dist, "temperature>0 must return a token from the distribution"
    except Exception as e:  # noqa: BLE001
        failures.append(f"sample_next: {e}")

    if failures:
        print("SELF-TEST FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL SELF-TESTS PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="COSC726 Week 2 lab - LLM foundations")
    parser.add_argument("--self-test", action="store_true", help="run the self-test suite")
    args = parser.parse_args(argv)
    if args.self_test:
        return _run_self_test()
    print("Nothing to run. Implement the three TODOs, then: --self-test")
    print("You have 3 TODO(s): count_tokens, prepare_context, sample_next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())