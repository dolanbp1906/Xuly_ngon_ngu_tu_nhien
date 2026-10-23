"""Vẽ biểu đồ từ các bảng thống kê."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
REGION_ORDER = [
    "Trung du và miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ và Duyên hải miền Trung",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]
EDU_ORDER = [
    "Không có học vấn",
    "Tiểu học",
    "Trung học cơ sở",
    "Trung học phổ thông",
    "Trung cấp / Cao đẳng",
    "Đại học",
    "Sau đại học",
]
MARITAL_ORDER = ["Độc thân", "Đã kết hôn", "Ly hôn", "Ly thân", "Góa", "Goá"]


def _font() -> None:
    names = {font.name for font in font_manager.fontManager.ttflist}
    if "Segoe UI" in names:
        plt.rcParams["font.family"] = "Segoe UI"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def _ordered(values: list[str], preferred: list[str]) -> list[str]:
    head = [item for item in preferred if item in values]
    tail = sorted(item for item in values if item not in head)
    return head + tail


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, encoding="utf-8-sig")


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=140)
    plt.close(fig)


def plot_distribution(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    specs = [
        ("sex", "Giới tính"),
        ("zone", "Khu vực"),
        ("age_group", "Nhóm tuổi"),
    ]
    for axis, (dimension, title) in zip(axes, specs):
        part = frame[frame["dimension"] == dimension].copy()
        if dimension == "age_group":
            order = _ordered(part["value"].tolist(), AGE_ORDER)
            part = part.set_index("value").loc[order].reset_index()
        axis.barh(part["value"].astype(str)[::-1], part["n"][::-1], color="#1f6f8b")
        axis.set_title(title)
        axis.set_xlabel("Số hồ sơ")
    fig.suptitle("Phân bố nhân khẩu trên hồ sơ tổng hợp")
    _save(fig, "01_phan_bo.png")

    regions = frame[frame["dimension"] == "region"].copy()
    order = _ordered(regions["value"].tolist(), REGION_ORDER)
    regions = regions.set_index("value").loc[order].reset_index()
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.barh(regions["value"].astype(str)[::-1], regions["n"][::-1], color="#c44900")
    axis.set_xlabel("Số hồ sơ")
    axis.set_title("Số hồ sơ theo vùng")
    _save(fig, "02_vung.png")


def plot_jobs(frame: pd.DataFrame) -> None:
    regions = _ordered(frame["region"].dropna().unique().tolist(), REGION_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for axis, region in zip(axes.ravel(), regions):
        part = frame[frame["region"] == region].nsmallest(8, "rank")
        axis.barh(part["occupation"][::-1], part["n"][::-1], color="#1f6f8b")
        axis.set_title(region, fontsize=10)
        axis.tick_params(axis="y", labelsize=8)
    for axis in axes.ravel()[len(regions) :]:
        axis.axis("off")
    fig.suptitle("Tám nghề phổ biến nhất trong mỗi vùng")
    _save(fig, "03_nghe_theo_vung.png")


def plot_education(frame: pd.DataFrame) -> None:
    ages = _ordered(frame["age_group"].unique().tolist(), AGE_ORDER)
    levels = _ordered(frame["education_level"].unique().tolist(), EDU_ORDER)
    pivot = (
        frame.pivot(index="education_level", columns="age_group", values="pct_trong_nhom")
        .reindex(index=levels, columns=ages)
        .fillna(0)
    )
    fig, axis = plt.subplots(figsize=(9, 5.2))
    image = axis.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for i, level in enumerate(pivot.index):
        for j, age in enumerate(pivot.columns):
            axis.text(j, i, f"{pivot.loc[level, age]:.0f}", ha="center", va="center", fontsize=8)
    axis.set_title("Tỷ lệ trình độ học vấn trong từng nhóm tuổi (%)")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save(fig, "04_hoc_van.png")


def plot_marital(frame: pd.DataFrame) -> None:
    sexes = frame["sex"].dropna().unique().tolist()
    ages = _ordered(frame["age_group"].unique().tolist(), AGE_ORDER)
    statuses = _ordered(frame["marital_status"].unique().tolist(), MARITAL_ORDER)
    colors = ["#1f6f8b", "#c44900", "#5c6b73", "#9b5de5", "#2a9d8f", "#e9c46a"]
    fig, axes = plt.subplots(1, len(sexes), figsize=(6 * max(len(sexes), 1), 4.8), sharey=True)
    if len(sexes) == 1:
        axes = [axes]
    for axis, sex in zip(axes, sexes):
        part = frame[frame["sex"] == sex]
        bottom = [0.0] * len(ages)
        for status, color in zip(statuses, colors):
            values = []
            for age in ages:
                match = part[(part["age_group"] == age) & (part["marital_status"] == status)]
                values.append(float(match["pct"].iloc[0]) if len(match) else 0.0)
            axis.bar(ages, values, bottom=bottom, label=status, color=color)
            bottom = [base + extra for base, extra in zip(bottom, values)]
        axis.set_title(str(sex))
        axis.set_ylabel("Tỷ lệ (%)")
        axis.tick_params(axis="x", rotation=30)
    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Tình trạng hôn nhân theo nhóm tuổi và giới tính")
    _save(fig, "05_hon_nhan.png")


def plot_skills(frame: pd.DataFrame) -> None:
    part = frame.head(15).iloc[::-1]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(part["skill"], part["n"], color="#1f6f8b")
    axis.set_xlabel("Số lần xuất hiện")
    axis.set_title("Mười lăm kỹ năng được nhắc nhiều nhất")
    _save(fig, "06_ky_nang.png")


def plot_all() -> None:
    _font()
    plot_distribution(_read("phan_bo.csv"))
    plot_jobs(_read("top15_nghe_theo_vung.csv"))
    plot_education(_read("hoc_van_theo_nhom_tuoi.csv"))
    plot_marital(_read("hon_nhan_theo_tuoi_gioi.csv"))
    plot_skills(_read("top_ky_nang.csv"))
    print(f"Đã lưu biểu đồ trong {FIGURES}", flush=True)


if __name__ == "__main__":
    plot_all()
