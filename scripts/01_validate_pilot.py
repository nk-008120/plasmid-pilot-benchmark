#!/usr/bin/env python3
"""Thin wrapper so the step is runnable as `python scripts/01_validate_pilot.py...`.

All logic lives in src/pilot/validate.py (install the package first: `pip install -e.`).
"""
from pilot.validate import main

if __name__ == "__main__":
    raise SystemExit(main())
