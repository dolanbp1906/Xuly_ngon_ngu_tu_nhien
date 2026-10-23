"""Rút cột có cấu trúc từ nvidia/Nemotron-Personas-Vietnam ra Parquet."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data" / "hf_home" / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DATASET = "nvidia/Nemotron-Personas-Vietnam"
LICENSE = "CC-BY-4.0"
COLUMNS = [
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "zone",
    "region",
    "skills_and_expertise_list",
]
SCHEMA = pa.schema(
    [
        ("sex", pa.string()),
        ("age", pa.int64()),
        ("marital_status", pa.string()),
        ("education_level", pa.string()),
        ("occupation", pa.string()),
        ("zone", pa.string()),
        ("region", pa.string()),
        ("skills_and_expertise_list", pa.string()),
    ]
)
PROCESSED = ROOT / "data" / "processed"
BATCH = 20_000


def log(message: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {message}"
    print(line, flush=True)
    log_path = PROCESSED / "extract.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def extract(limit: int = 0) -> Path:
    """Tải Parquet gốc và chỉ giữ các cột thống kê.

    limit=0 nghĩa là lấy đủ 1.000.000 dòng.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    target = PROCESSED / "personas_struct.parquet"
    temporary = PROCESSED / "personas_struct.parquet.tmp"
    if temporary.exists():
        temporary.unlink()

    log(f"Đang tải {DATASET}. Sau đó chỉ giữ {len(COLUMNS)} cột có cấu trúc.")
    local = snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
    )
    source = ds.dataset(Path(local) / "data", format="parquet")
    log("Đã có file nguồn. Đang rút cột.")

    writer = None
    sample: list[dict] = []
    seen = 0
    scanner = source.scanner(columns=COLUMNS, batch_size=BATCH)
    for batch in scanner.to_batches():
        if limit and seen >= limit:
            break
        if limit and seen + batch.num_rows > limit:
            batch = batch.slice(0, limit - seen)
        table = pa.Table.from_batches([batch]).cast(SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
        writer.write_table(table)
        if len(sample) < 3:
            sample.extend(table.slice(0, 3 - len(sample)).to_pylist())
        seen += table.num_rows
        log(f"đã ghi {seen:,} dòng")

    if writer is None:
        raise RuntimeError("Không đọc được dòng nào từ bộ dữ liệu.")
    writer.close()
    temporary.replace(target)

    meta = {
        "dataset": DATASET,
        "url": "https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam",
        "license": LICENSE,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "rows": seen,
        "columns": COLUMNS,
        "note": "Hồ sơ nhân khẩu tổng hợp, phân bố neo theo Niên giám thống kê Việt Nam 2024.",
    }
    (PROCESSED / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (PROCESSED / "sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Xong {seen:,} dòng -> {target}")
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    extract(limit=args.limit)
