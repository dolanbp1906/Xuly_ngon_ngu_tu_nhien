"""Chạy pipeline hồ sơ nhân khẩu Việt Nam.

Cách dùng, tại thư mục do_an_mon_du_lieu_lon:

    python run.py
    python run.py --limit 20000
    python run.py --skip-extract
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analyze import analyze
from extract import extract
from plots import plot_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = đủ 1.000.000 dòng")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    if not args.skip_extract:
        extract(limit=args.limit)
    analyze()
    plot_all()
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
