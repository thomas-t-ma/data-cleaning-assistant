"""Run the fresh single-edit scientific-neighbor holdout with local Ollama.

This is intentionally separate from the original forty-case semantic stress corpus.
Use it after prompt development on that original corpus to estimate whether typo
recall gains generalize without merging valid one-edit scientific terms.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data_cleaning_assistant.ai_planner import resolve_model
from data_cleaning_assistant.ai_provider import OllamaProvider
from data_cleaning_assistant.semantic_holdout import semantic_neighbor_holdout_cases

from semantic_stress_test import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Exact installed Ollama model; omit if only one")
    parser.add_argument("--url", help="Ollama URL; defaults to local")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--think", action="store_true", help="Optional thinking-mode experiment")
    parser.add_argument(
        "--prompt-variant",
        choices=("baseline", "evidence_calibrated"),
        default="baseline",
        help="Compare the frozen production prompt with the benchmark-only calibrated addendum",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Check detector/candidate coverage without calling Ollama",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    provider = None if args.audit_only else OllamaProvider(args.url, think=args.think)
    model = None if args.audit_only else resolve_model(provider, args.model)
    run_suite(
        provider,
        model,
        runs=args.runs,
        output=args.output,
        audit_only=args.audit_only,
        cases=semantic_neighbor_holdout_cases(),
        prompt_variant=args.prompt_variant,
    )


if __name__ == "__main__":
    main()
