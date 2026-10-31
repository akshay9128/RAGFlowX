#!/usr/bin/env python3
"""
RAGFlowX Cross-Platform Test Runner Script (Integration Testing)
Runs all test suites across tiers (unit, e2e, quality) and reports summary results.
"""

import sys
import subprocess
from pathlib import Path


def main():
    print("=" * 60)
    print("RAGFlowX Test Suite Runner — End-to-End Integration")
    print("=" * 60)

    root_dir = Path(__file__).resolve().parent.parent

    suites = [
        ("Unit & General Test Suite", [sys.executable, "-m", "pytest", "-m", "not e2e and not quality and not live", "-v"]),
        ("E2E Integration Tier", [sys.executable, "-m", "pytest", "-m", "e2e", "-v"]),
        ("Quality Tier (Embeddings & Reranker)", [sys.executable, "-m", "pytest", "-m", "quality", "-v"]),
    ]

    failed_suites = []

    for name, cmd in suites:
        print(f"\n[-] Running: {name}")
        print(f"    Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(root_dir))
        if result.returncode != 0:
            failed_suites.append(name)
            print(f"[X] FAILED: {name}")
        else:
            print(f"[✔] PASSED: {name}")

    print("\n" + "=" * 60)
    if failed_suites:
        print(f"[X] Test run completed with failures in: {', '.join(failed_suites)}")
        sys.exit(1)
    else:
        print("[✔] All test suites passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
