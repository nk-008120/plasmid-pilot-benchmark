#!/usr/bin/env python3
"""Pre-fill the manual audit worksheet: `python scripts/02_make_audit_sheet.py --results-dir results/baseline`.

All logic lives in src/pilot/audit.py (install the package first: `pip install -e.`).
"""
from pilot.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
