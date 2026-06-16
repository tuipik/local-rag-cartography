"""Small reporting helpers shared by command-line scripts."""

from __future__ import annotations

from collections import Counter


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"\n{title}:")
    if not counter:
        print("  none")
        return
    for value, count in sorted(counter.items()):
        print(f"  {value}: {count}")
