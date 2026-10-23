"""Thống kê hồ sơ nhân khẩu và so sánh Pandas với DuckDB."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pandas as pd
import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "processed" / "personas_struct.parquet"
TABLES = ROOT / "output" / "tables"

AGE_SQL = """
CASE
    WHEN age < 25 THEN '18-24'
    WHEN age < 35 THEN '25-34'
    WHEN age < 45 THEN '35-44'
    WHEN age < 55 THEN '45-54'
    WHEN age < 65 THEN '55-64'
    ELSE '65+'
END
"""


def _write(frame: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _distribution(con: duckdb.DuckDBPyConnection, dimension: str, value_sql: str) -> pd.DataFrame:
    frame = con.execute(
        f"""
        SELECT
            '{dimension}' AS dimension,
            CAST({value_sql} AS VARCHAR) AS value,
            count(*) AS n
        FROM personas
        GROUP BY 2
        ORDER BY n DESC
        """
    ).fetchdf()
    total = int(frame["n"].sum())
    frame["pct"] = (frame["n"] / total * 100).round(2)
    return frame


def _skills(parquet: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pq.read_table(parquet, columns=["occupation", "skills_and_expertise_list"])
    occupations = table.column("occupation").to_pylist()
    raw_lists = table.column("skills_and_expertise_list").to_pylist()
    by_job: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for occupation, raw in zip(occupations, raw_lists):
        try:
            skills = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        except json.JSONDecodeError:
            skills = []
        for skill in skills:
            text = str(skill).strip()
            if not text:
                continue
            by_job[str(occupation)][text] += 1
            overall[text] += 1

    top_rows = [
        {"skill": skill, "n": count}
        for skill, count in overall.most_common(20)
    ]
    per_job = []
    for occupation, counts in by_job.items():
        for rank, (skill, count) in enumerate(counts.most_common(5), start=1):
            per_job.append(
                {"occupation": occupation, "rank": rank, "skill": skill, "n": count}
            )
    per_job.sort(key=lambda item: (item["occupation"], item["rank"]))
    return pd.DataFrame(top_rows), pd.DataFrame(per_job)


def _benchmark(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Cùng một phép nhóm, cả hai bên đều tính cả thời gian đọc Parquet."""
    process = psutil.Process()
    query = """
        SELECT region, occupation, count(*) AS n
        FROM read_parquet(?)
        GROUP BY region, occupation
    """

    started = time.perf_counter()
    duck = con.execute(query, [PARQUET.as_posix()]).fetchdf()
    duck_seconds = time.perf_counter() - started
    duck_rss = process.memory_info().rss / (1024 * 1024)

    started = time.perf_counter()
    frame = pd.read_parquet(PARQUET, columns=["region", "occupation"])
    pandas_counts = frame.groupby(["region", "occupation"], observed=True).size()
    pandas_seconds = time.perf_counter() - started
    pandas_rss = process.memory_info().rss / (1024 * 1024)

    if len(duck) != len(pandas_counts):
        raise RuntimeError(
            f"Số nhóm lệch: DuckDB {len(duck)} nhóm, Pandas {len(pandas_counts)} nhóm."
        )

    return pd.DataFrame(
        [
            {
                "engine": "duckdb",
                "task": "doc_va_nhom_vung_nghe",
                "seconds": round(duck_seconds, 3),
                "groups": len(duck),
                "rss_mb": round(duck_rss, 1),
            },
            {
                "engine": "pandas",
                "task": "doc_va_nhom_vung_nghe",
                "seconds": round(pandas_seconds, 3),
                "groups": int(pandas_counts.shape[0]),
                "rss_mb": round(pandas_rss, 1),
            },
        ]
    )


def analyze() -> None:
    if not PARQUET.exists():
        raise FileNotFoundError(f"Chưa có file dữ liệu: {PARQUET}")

    con = duckdb.connect()
    parquet_path = PARQUET.as_posix().replace("'", "''")
    con.execute(
        f"CREATE VIEW personas AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    total = con.execute("SELECT count(*) FROM personas").fetchone()[0]
    print(f"Đọc {total:,} dòng từ {PARQUET.name}", flush=True)

    parts = [
        _distribution(con, "sex", "sex"),
        _distribution(con, "zone", "zone"),
        _distribution(con, "region", "region"),
        _distribution(con, "age_group", AGE_SQL),
    ]
    distribution = pd.concat(parts, ignore_index=True)
    _write(distribution, "phan_bo.csv")

    top_jobs = con.execute(
        """
        WITH counts AS (
            SELECT region, occupation, count(*) AS n
            FROM personas
            GROUP BY region, occupation
        ),
        ranked AS (
            SELECT
                region,
                occupation,
                n,
                row_number() OVER (PARTITION BY region ORDER BY n DESC) AS rank
            FROM counts
        )
        SELECT region, rank, occupation, n
        FROM ranked
        WHERE rank <= 15
        ORDER BY region, rank
        """
    ).fetchdf()
    _write(top_jobs, "top15_nghe_theo_vung.csv")

    education = con.execute(
        f"""
        WITH grouped AS (
            SELECT {AGE_SQL} AS age_group, education_level, count(*) AS n
            FROM personas
            GROUP BY 1, 2
        )
        SELECT
            age_group,
            education_level,
            n,
            round(100.0 * n / sum(n) OVER (PARTITION BY age_group), 2) AS pct_trong_nhom
        FROM grouped
        ORDER BY age_group, n DESC
        """
    ).fetchdf()
    _write(education, "hoc_van_theo_nhom_tuoi.csv")

    marital = con.execute(
        f"""
        WITH grouped AS (
            SELECT sex, {AGE_SQL} AS age_group, marital_status, count(*) AS n
            FROM personas
            GROUP BY 1, 2, 3
        )
        SELECT
            sex,
            age_group,
            marital_status,
            n,
            round(100.0 * n / sum(n) OVER (PARTITION BY sex, age_group), 2) AS pct
        FROM grouped
        ORDER BY sex, age_group, n DESC
        """
    ).fetchdf()
    _write(marital, "hon_nhan_theo_tuoi_gioi.csv")

    print("Đang tách danh sách kỹ năng...", flush=True)
    top_skills, skills_by_job = _skills(PARQUET)
    _write(top_skills, "top_ky_nang.csv")
    _write(skills_by_job, "ky_nang_theo_nghe.csv")

    print("Đang so sánh Pandas và DuckDB...", flush=True)
    benchmark = _benchmark(con)
    _write(benchmark, "benchmark.csv")

    sex = distribution[distribution["dimension"] == "sex"]
    lines = [
        f"Số hồ sơ: {total:,}",
        "Giới tính:",
    ]
    for row in sex.itertuples(index=False):
        lines.append(f"  - {row.value}: {row.n:,} ({row.pct}%)")
    lines.append("So sánh nhóm theo vùng và nghề:")
    for row in benchmark.itertuples(index=False):
        lines.append(
            f"  - {row.engine}: {row.seconds} giây, {row.groups:,} nhóm, RSS {row.rss_mb} MB"
        )
    summary = "\n".join(lines) + "\n"
    (TABLES / "tom_tat.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    analyze()
