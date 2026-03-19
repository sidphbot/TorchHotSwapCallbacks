#!/usr/bin/env python3
"""
Build test: run all policy pack scenarios headless and report results.

Not a pytest test — meant for pre-commit or CI verification.
Exits with code 0 if all pass, 1 otherwise.

Usage:
    python tests/build_test_scenarios.py
    python tests/build_test_scenarios.py --pack stability_basics
"""
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Run policy pack scenario tests")
    parser.add_argument("--pack", default=None, help="Only run scenarios for this pack")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from hotcb.scenarios.runner import run_all

    print("=" * 60)
    print("  hotcb scenario build test")
    print("=" * 60)
    print()

    start = time.time()
    results = run_all(pack=args.pack, step_delay=0, verbose=True)
    elapsed = time.time() - start

    print()
    print("-" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"  {passed}/{total} scenarios passed in {elapsed:.1f}s")

    if passed < total:
        print()
        for r in results:
            if not r.passed:
                print(f"  FAIL: {r.name}")
                print(f"    Expected: {r.expected_rules}")
                print(f"    Fired:    {r.rules_fired}")
                print(f"    Missing:  {r.missing_rules}")
                if r.error:
                    print(f"    Error:    {r.error}")
        print()
        print("BUILD FAILED")
        sys.exit(1)
    else:
        print()
        print("BUILD PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
