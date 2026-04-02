#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from spmo_config import CORE_PARAMS, MATCH_VIS_EXCLUDE_COLS, _format_scene_value

def configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_matched_samples_visualization(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if df.empty or "wet_time" not in df.columns:
        return None

    configure_plot_style()

    plot_df = df.sort_values("wet_time").copy()
    preferred_cols = [
        "wet_weight",
        "印刷压力",
        "印刷高度偏移",
        "刮刀高度偏移",
        "刮刀高度_上",
        "刮刀高度_下",
        "墨刀高度_上",
        "墨刀高度_下",
        "印刷速度",
        "网版寿命",
        "time_diff_min",
    ]
    plot_cols: List[str] = []

    for col in preferred_cols:
        if col in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[col]):
            plot_cols.append(col)

    for col in plot_df.columns:
        if col in MATCH_VIS_EXCLUDE_COLS or col in plot_cols:
            continue
        if pd.api.types.is_numeric_dtype(plot_df[col]):
            plot_cols.append(col)

    if not plot_cols:
        return None

    n_cols = 2
    n_rows = int(np.ceil(len(plot_cols) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, max(4 * n_rows, 6)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)

    x = plot_df["wet_time"]
    for ax, col in zip(axes, plot_cols):
        ax.plot(x, plot_df[col], color="#1f77b4", linewidth=1.0, alpha=0.9)
        ax.scatter(x, plot_df[col], s=8, color="#d62728", alpha=0.45)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    for ax in axes[len(plot_cols):]:
        ax.axis("off")

    for ax in axes[: len(plot_cols)]:
        ax.set_xlabel("wet_time", fontsize=9)

    fig.suptitle("样本匹配结果可视化：wet_time 时间序列子图", fontsize=16)
    fig.autofmt_xdate(rotation=25)

    output_path = output_dir / "matched_samples_timeseries.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_unused_adjacent_sample_visualization(matched_df: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if matched_df is None or matched_df.empty or "wet_time" not in matched_df.columns or "wet_weight" not in matched_df.columns:
        return None

    configure_plot_style()
    plot_df = matched_df.copy()
    plot_df["wet_time"] = pd.to_datetime(plot_df["wet_time"], errors="coerce")
    plot_df["wet_weight"] = pd.to_numeric(plot_df["wet_weight"], errors="coerce")
    plot_df = plot_df.dropna(subset=["wet_time", "wet_weight"]).sort_values("wet_time").reset_index(drop=True)
    if plot_df.empty:
        return None

    used_cur = set()
    if deltas is not None and (not deltas.empty) and ("t_cur" in deltas.columns):
        used_cur = set(pd.to_datetime(deltas["t_cur"], errors="coerce").dropna())
    plot_df["is_unused_as_cur"] = ~plot_df["wet_time"].isin(used_cur)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(plot_df["wet_time"], plot_df["wet_weight"], color="#4e79a7", linewidth=1.2, alpha=0.9, label="全部 wet_weight")

    unused_df = plot_df[plot_df["is_unused_as_cur"]].copy()
    if not unused_df.empty:
        ax.scatter(
            unused_df["wet_time"],
            unused_df["wet_weight"],
            s=28,
            color="#d62728",
            alpha=0.9,
            label=f"未进入相邻差分的样本点 ({len(unused_df)})",
            zorder=3,
        )

    ax.set_title(f"wet_weight 折线图及未进入相邻差分的样本点（{len(unused_df)} 个）")
    ax.set_xlabel("wet_time")
    ax.set_ylabel("wet_weight")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate(rotation=25)

    output_path = output_dir / "wet_weight_unused_adjacent_points.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _truncate_label(text: str, limit: int = 42) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def save_hierarchical_scene_visualizations(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    if df.empty:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}

    coarse_counts = (
        df.groupby("scene_coarse_key", dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
    )
    coarse_counts["scene_coarse_key_short"] = coarse_counts["scene_coarse_key"].map(_truncate_label)

    fig, ax = plt.subplots(figsize=(14, max(6, 0.6 * len(coarse_counts) + 2)))
    ax.barh(coarse_counts["scene_coarse_key_short"], coarse_counts["sample_count"], color="#4e79a7")
    ax.invert_yaxis()
    ax.set_title("粗场景样本数排行")
    ax.set_xlabel("样本数")
    ax.set_ylabel("粗场景")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    coarse_bar_path = output_dir / "hierarchical_coarse_scene_ranking.png"
    fig.savefig(coarse_bar_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output_paths["coarse_scene_ranking"] = str(coarse_bar_path)

    boxplot_stats = coarse_counts.copy()
    top_scenes = boxplot_stats["scene_coarse_key"].head(min(12, len(boxplot_stats))).tolist()
    box_df = df[df["scene_coarse_key"].isin(top_scenes)].copy()
    if not box_df.empty:
        grouped = [box_df.loc[box_df["scene_coarse_key"] == scene, "wet_weight"].dropna().values for scene in top_scenes]
        labels = [_truncate_label(scene, 28) for scene in top_scenes]

        fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(labels) + 4), 6.5))
        bp = ax.boxplot(grouped, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#76b7b2", alpha=0.75)
        ax.set_title("不同场景湿重分布箱线图")
        ax.set_xlabel("粗场景")
        ax.set_ylabel("湿重")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        boxplot_path = output_dir / "hierarchical_wet_weight_boxplot.png"
        fig.savefig(boxplot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["wet_weight_boxplot"] = str(boxplot_path)

    scene_tree_df = (
        df.groupby(["scene_coarse_key", "life_segment", "scene_fine_key"], dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values(["sample_count", "scene_coarse_key", "life_segment"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    scene_tree_csv_path = output_dir / "hierarchical_scene_structure_tree.csv"
    scene_tree_df.to_csv(scene_tree_csv_path, index=False, encoding="utf-8-sig")
    output_paths["scene_structure_tree_csv"] = str(scene_tree_csv_path)

    coarse_feature_cols = [c for c in ["刮刀高度_上", "刮刀高度_下", "墨刀高度_上", "墨刀高度_下"] if c in scene_tree_df.columns]
    if coarse_feature_cols:
        table_show = scene_tree_df.copy()
        for col in coarse_feature_cols:
            vals = pd.to_numeric(table_show[col], errors="coerce")
            table_show[col] = vals.map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
        table_show["life_segment"] = table_show["life_segment"].astype(str)
        table_show["sample_count"] = table_show["sample_count"].astype(int).astype(str)

        top_header = ["粗场景"] * len(coarse_feature_cols) + ["细场景", "细场景"]
        second_header = coarse_feature_cols + ["网版寿命", "样本数"]
        body_cols = coarse_feature_cols + ["life_segment", "sample_count"]
        table_rows = [top_header, second_header] + table_show[body_cols].values.tolist()

        fig_h = max(8, 0.62 * len(table_rows) + 1.8)
        fig_w = max(18, 2.8 * len(body_cols) + 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        table = ax.table(
            cellText=table_rows,
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=[0.13] * len(coarse_feature_cols) + [0.22, 0.10],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.8)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#d9eaf7")
                cell.set_text_props(weight="bold", ha="center")
            elif row == 1:
                cell.set_facecolor("#edf4fb")
                cell.set_text_props(weight="bold", ha="center")
            else:
                cell.set_text_props(ha="center", va="center")

        ax.set_title(f"场景结构树状表（全量 {len(scene_tree_df)} 行）", pad=16)
        table_png_path = output_dir / "hierarchical_scene_structure_tree.png"
        fig.savefig(table_png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["scene_structure_tree_png"] = str(table_png_path)
        return output_paths

    table_show = scene_tree_df.copy()

    def _format_scene_cell(text: str) -> str:
        return str(text).replace(" | ", "\n")

    table_show["scene_coarse_key"] = table_show["scene_coarse_key"].map(_format_scene_cell)
    table_show["scene_fine_key"] = table_show["scene_fine_key"].map(_format_scene_cell)
    table_show["life_segment"] = table_show["life_segment"].astype(str)
    table_show["sample_count"] = table_show["sample_count"].astype(int).astype(str)

    table_rows = [
        ["粗场景", "粗场景", "细场景", "细场景"],
        ["粗场景键", "寿命分段", "细场景键", "样本数"],
    ] + table_show[["scene_coarse_key", "life_segment", "scene_fine_key", "sample_count"]].values.tolist()

    fig_h = max(8, 0.78 * len(table_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.33, 0.12, 0.43, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold", ha="center")
        elif row == 1:
            cell.set_facecolor("#edf4fb")
            cell.set_text_props(weight="bold", ha="center")
        else:
            cell.set_text_props(ha="left", va="center")

    ax.set_title(f"场景结构树状表（全量 {len(scene_tree_df)} 行）", pad=16)
    table_png_path = output_dir / "hierarchical_scene_structure_tree.png"
    fig.savefig(table_png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output_paths["scene_structure_tree_png"] = str(table_png_path)

    return output_paths


def save_coarse_scene_param_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    max_scenes: int = 12,
) -> Dict[str, str]:
    if df is None or df.empty or "scene_coarse_key" not in df.columns:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}
    numeric_candidates = [
        "wet_weight",
        "印刷压力",
        "印刷高度偏移",
        "刮刀高度偏移",
        "印刷速度",
        "网版寿命",
        "time_diff_min",
    ]
    plot_cols = [c for c in numeric_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not plot_cols:
        return {}

    scene_counts = df["scene_coarse_key"].value_counts().head(max_scenes)
    summary_rows: List[Dict] = []

    for idx, scene_key in enumerate(scene_counts.index, start=1):
        scene_df = df[df["scene_coarse_key"] == scene_key].copy()
        if scene_df.empty:
            continue

        n_cols = 2
        n_rows = int(np.ceil(len(plot_cols) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(15, max(4.6 * n_rows, 6)),
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, plot_cols):
            values = pd.to_numeric(scene_df[col], errors="coerce").dropna()
            if len(values) == 0:
                ax.axis("off")
                continue
            bins = min(18, max(6, int(np.sqrt(len(values)) + 2)))
            ax.hist(values, bins=bins, color="#4e79a7", alpha=0.82, edgecolor="white")
            mean_v = float(values.mean())
            med_v = float(values.median())
            ax.axvline(mean_v, color="#d62728", linestyle="--", linewidth=1.0, label=f"均值={mean_v:.4f}")
            ax.axvline(med_v, color="#59a14f", linestyle=":", linewidth=1.0, label=f"中位数={med_v:.4f}")
            ax.set_title(f"{col} 分布")
            ax.set_xlabel(col)
            ax.set_ylabel("频数")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.legend(fontsize=8)
            summary_rows.append(
                {
                    "scene_coarse_key": scene_key,
                    "param": col,
                    "samples": int(len(values)),
                    "mean": mean_v,
                    "median": med_v,
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

        for ax in axes[len(plot_cols):]:
            ax.axis("off")

        fig.suptitle(
            f"粗场景 {idx} 参数分布图 | 样本数={len(scene_df)} | {_truncate_label(scene_key, 90)}",
            fontsize=14,
        )
        out_path = output_dir / f"coarse_scene_{idx:02d}_param_distribution.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths[f"scene_{idx:02d}"] = str(out_path)

    if summary_rows:
        summary_csv = output_dir / "coarse_scene_param_distribution_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        output_paths["summary_csv"] = str(summary_csv)

    return output_paths


def run_scene_decision_tree(
    df: pd.DataFrame,
    output_dir: Path,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    if df is None or df.empty or "wet_weight" not in df.columns:
        return {"enabled": False, "message": "缺少湿重数据，无法训练决策树"}

    configure_plot_style()

    feature_candidates = feature_cols if feature_cols is not None else ["刮刀高度_上", "刮刀高度_下", "墨刀高度_上", "墨刀高度_下"]
    feature_cols = [c for c in feature_candidates if c in df.columns]
    if not feature_cols:
        return {"enabled": False, "message": "缺少场景特征列，无法训练决策树"}

    tree_df = df[feature_cols + ["wet_weight"]].copy()
    for col in feature_cols + ["wet_weight"]:
        tree_df[col] = pd.to_numeric(tree_df[col], errors="coerce")
    tree_df = tree_df.dropna(subset=["wet_weight"] + feature_cols).reset_index(drop=True)
    if len(tree_df) < max(20, min_samples_leaf * 2):
        return {"enabled": False, "message": "可用于决策树训练的样本不足"}

    X = tree_df[feature_cols]
    y = tree_df["wet_weight"]

    model = DecisionTreeRegressor(
        random_state=42,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X, y)

    tree_df["tree_leaf_id"] = model.apply(X)
    leaf_summary = (
        tree_df.groupby("tree_leaf_id")
        .agg(
            sample_count=("wet_weight", "size"),
            wet_weight_mean=("wet_weight", "mean"),
            wet_weight_std=("wet_weight", "std"),
            wet_weight_min=("wet_weight", "min"),
            wet_weight_max=("wet_weight", "max"),
        )
        .reset_index()
        .sort_values("wet_weight_mean")
        .reset_index(drop=True)
    )
    leaf_summary_csv = output_dir / "decision_tree_scene_leaf_summary.csv"
    leaf_summary.to_csv(leaf_summary_csv, index=False, encoding="utf-8-sig")

    assignments = df.copy()
    assign_df = assignments[feature_cols].copy()
    for col in feature_cols:
        assign_df[col] = pd.to_numeric(assign_df[col], errors="coerce")
    valid_mask = assign_df.notna().all(axis=1)
    assignments["tree_leaf_id"] = np.nan
    assignments.loc[valid_mask, "tree_leaf_id"] = model.apply(assign_df.loc[valid_mask, feature_cols])
    assign_csv = output_dir / "decision_tree_scene_assignments.csv"
    assignments.to_csv(assign_csv, index=False, encoding="utf-8-sig")

    rules_text = export_text(model, feature_names=feature_cols, decimals=3)
    rules_txt = output_dir / "decision_tree_scene_rules.txt"
    with open(rules_txt, "w", encoding="utf-8") as f:
        f.write(rules_text)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("以湿重为目标的场景最优分割决策树")
    tree_png = output_dir / "decision_tree_scene_split.png"
    fig.savefig(tree_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    leaf_box_df = tree_df.merge(leaf_summary[["tree_leaf_id"]], on="tree_leaf_id", how="left")
    ordered_leaves = leaf_summary["tree_leaf_id"].tolist()
    grouped = [leaf_box_df.loc[leaf_box_df["tree_leaf_id"] == leaf_id, "wet_weight"].dropna().values for leaf_id in ordered_leaves]
    labels = [f"Leaf {int(leaf_id)}" for leaf_id in ordered_leaves]
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(labels) + 3), 6))
    bp = ax.boxplot(grouped, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set(facecolor="#76b7b2", alpha=0.75)
    ax.set_title("决策树叶子场景湿重分布箱线图")
    ax.set_xlabel("叶子场景")
    ax.set_ylabel("湿重")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    leaf_boxplot_png = output_dir / "decision_tree_leaf_wet_weight_boxplot.png"
    fig.savefig(leaf_boxplot_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "enabled": True,
        "feature_cols": feature_cols,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "leaf_count": int(leaf_summary["tree_leaf_id"].nunique()),
        "leaf_summary_csv": str(leaf_summary_csv),
        "assignment_csv": str(assign_csv),
        "rules_txt": str(rules_txt),
        "tree_png": str(tree_png),
        "leaf_wet_weight_boxplot": str(leaf_boxplot_png),
    }


def save_coarse_scene_param_boxplots(
    df: pd.DataFrame,
    output_dir: Path,
    max_scenes: Optional[int] = None,
) -> Dict[str, str]:
    if df is None or df.empty or "scene_coarse_key" not in df.columns:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}
    plot_cols = [
        c
        for c in ["wet_weight", "印刷压力", "印刷高度偏移", "刮刀高度偏移"]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(plot_cols) == 0:
        return {}

    if max_scenes is None:
        max_scenes = int(df["scene_coarse_key"].nunique())
    scene_counts = df["scene_coarse_key"].value_counts().head(max_scenes)
    ordered_scenes = scene_counts.index.tolist()
    scene_labels = [f"S{i}" for i in range(1, len(ordered_scenes) + 1)]
    summary_rows: List[Dict] = []

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, plot_cols):
        grouped = []
        for idx, scene_key in enumerate(ordered_scenes):
            values = pd.to_numeric(df.loc[df["scene_coarse_key"] == scene_key, col], errors="coerce").dropna()
            grouped.append(values.values)
            if len(values):
                summary_rows.append(
                    {
                        "scene_coarse_key": scene_key,
                        "scene_label": scene_labels[idx],
                        "param": col,
                        "samples": int(len(values)),
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "std": float(values.std(ddof=0)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )

        bp = ax.boxplot(grouped, tick_labels=scene_labels, patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#76b7b2", alpha=0.75)
        ax.set_title(f"{col} 箱线图")
        ax.set_xlabel("粗场景")
        ax.set_ylabel(col)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    for ax in axes[len(plot_cols):]:
        ax.axis("off")

    scene_note = " | ".join([f"{scene_labels[i]}={_truncate_label(ordered_scenes[i], 24)}" for i in range(len(ordered_scenes))])
    fig.suptitle(f"{len(ordered_scenes)}种粗场景变量箱线图\n{scene_note}", fontsize=15)
    out_path = output_dir / "coarse_scene_param_boxplots.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output_paths["boxplot_figure"] = str(out_path)

    if summary_rows:
        summary_csv = output_dir / "coarse_scene_param_distribution_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        output_paths["summary_csv"] = str(summary_csv)

    return output_paths


def save_fixed_coarse_scene_structure_table(
    df: pd.DataFrame,
    output_dir: Path,
    max_scenes: Optional[int] = None,
) -> Dict[str, str]:
    if df is None or df.empty or "scene_coarse_key" not in df.columns:
        return {}

    configure_plot_style()
    if max_scenes is None:
        max_scenes = int(df["scene_coarse_key"].nunique())
    scene_counts = (
        df.groupby("scene_coarse_key", dropna=False)
        .agg(
            sample_count=("wet_weight", "size"),
            wet_weight_mean=("wet_weight", "mean"),
            wet_weight_std=("wet_weight", "std"),
        )
        .reset_index()
        .sort_values("sample_count", ascending=False)
        .head(max_scenes)
        .reset_index(drop=True)
    )
    scene_counts["scene_label"] = [f"S{i}" for i in range(1, len(scene_counts) + 1)]
    scene_counts["scene_desc"] = scene_counts["scene_coarse_key"].map(lambda x: _truncate_label(x, 48))

    csv_path = output_dir / "fixed_coarse_scene_structure_table.csv"
    scene_counts.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig_h = max(4.8, 0.6 * len(scene_counts) + 2.3)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=scene_counts[["scene_label", "scene_desc", "sample_count", "wet_weight_mean", "wet_weight_std"]].round(
            {"wet_weight_mean": 6, "wet_weight_std": 6}
        ).values,
        colLabels=["场景编号", "粗场景描述", "样本数", "湿重均值", "湿重标准差"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    ax.set_title(f"固定规则 {len(scene_counts)} 种粗场景结构树状表", pad=16)
    png_path = output_dir / "fixed_coarse_scene_structure_table.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {"csv": str(csv_path), "png": str(png_path)}


def _extract_tree_split_thresholds(model: DecisionTreeRegressor, feature_cols: List[str]) -> Dict[str, List[float]]:
    thresholds: Dict[str, List[float]] = {c: [] for c in feature_cols}
    tree = model.tree_
    for feature_idx, threshold in zip(tree.feature, tree.threshold):
        if feature_idx < 0:
            continue
        thresholds[feature_cols[int(feature_idx)]].append(float(threshold))
    return {k: sorted(set(v)) for k, v in thresholds.items()}


def _extract_fixed_split_thresholds(series: pd.Series, step: float) -> List[float]:
    if step <= 0:
        return []
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return []
    centers = np.sort(np.unique(np.round(values / step) * step))
    if len(centers) <= 1:
        return []
    return [float((centers[i] + centers[i + 1]) / 2.0) for i in range(len(centers) - 1)]


def save_coarse_scene_split_points_visualization(
    df: pd.DataFrame,
    output_dir: Path,
    scene_config: Dict,
) -> Dict[str, str]:
    if df is None or df.empty or "wet_weight" not in df.columns:
        return {}

    coarse_cfg = scene_config.get("coarse_scene", {})
    feature_cols = [c for c in coarse_cfg.get("features", []) if c in df.columns]
    if not feature_cols:
        return {}

    configure_plot_style()
    method = str(coarse_cfg.get("method", "fixed")).lower()
    split_map: Dict[str, List[float]] = {c: [] for c in feature_cols}

    if method == "tree":
        train_df = df[feature_cols + ["wet_weight"]].copy()
        for col in feature_cols + ["wet_weight"]:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        train_df = train_df.dropna(subset=feature_cols + ["wet_weight"]).reset_index(drop=True)
        if not train_df.empty:
            model = DecisionTreeRegressor(
                random_state=42,
                max_leaf_nodes=int(coarse_cfg.get("tree", {}).get("max_leaf_nodes", 6)),
                min_samples_leaf=int(coarse_cfg.get("tree", {}).get("min_samples_leaf", 20)),
            )
            model.fit(train_df[feature_cols], train_df["wet_weight"])
            split_map = _extract_tree_split_thresholds(model, feature_cols)
    else:
        step = float(coarse_cfg.get("fixed_round_step", 0.5))
        for col in feature_cols:
            split_map[col] = _extract_fixed_split_thresholds(df[col], step)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    summary_rows: List[Dict] = []

    for ax, col in zip(axes, feature_cols[:4]):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            ax.axis("off")
            continue
        ax.hist(values, bins=min(30, max(10, int(np.sqrt(len(values))))), color="#9ecae1", edgecolor="white", alpha=0.85)
        for split in split_map.get(col, []):
            ax.axvline(split, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.9)
            summary_rows.append({"param": col, "split_point": float(split), "method": method})
        ax.set_title(f"{col} 原始分布与分割点")
        ax.set_xlabel(col)
        ax.set_ylabel("样本数")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    for ax in axes[len(feature_cols[:4]):]:
        ax.axis("off")

    fig.suptitle(f"粗场景参数分割点可视化（{method}）", fontsize=15)
    png_path = output_dir / "coarse_scene_split_points.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    output_paths = {"png": str(png_path)}
    if summary_rows:
        csv_path = output_dir / "coarse_scene_split_points.csv"
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        output_paths["csv"] = str(csv_path)
    return output_paths


def save_life_segment_wet_weight_distribution(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    if df.empty or "life_segment" not in df.columns or "wet_weight" not in df.columns:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}
    plot_df = df.copy()
    plot_df["wet_weight"] = pd.to_numeric(plot_df["wet_weight"], errors="coerce")
    plot_df = plot_df[plot_df["wet_weight"].notna()].copy()
    if plot_df.empty:
        return {}

    segment_stats = (
        plot_df.groupby("life_segment", dropna=False)
        .agg(
            sample_count=("wet_weight", "size"),
            wet_mean=("wet_weight", "mean"),
            wet_std=("wet_weight", "std"),
            wet_min=("wet_weight", "min"),
            wet_q25=("wet_weight", lambda s: s.quantile(0.25)),
            wet_median=("wet_weight", "median"),
            wet_q75=("wet_weight", lambda s: s.quantile(0.75)),
            wet_max=("wet_weight", "max"),
        )
        .reset_index()
    )
    segment_stats["life_segment"] = segment_stats["life_segment"].astype(str)
    segment_stats = segment_stats.sort_values("life_segment").reset_index(drop=True)

    grouped = [
        plot_df.loc[plot_df["life_segment"].astype(str) == seg, "wet_weight"].values
        for seg in segment_stats["life_segment"].tolist()
    ]
    labels = [_truncate_label(seg, 24) for seg in segment_stats["life_segment"].tolist()]

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(labels) + 4), 6.5))
    bp = ax.boxplot(grouped, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set(facecolor="#edc948", alpha=0.75)
    ax.set_title("不同网版寿命区间湿重分布箱线图")
    ax.set_xlabel("网版寿命区间")
    ax.set_ylabel("湿重")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    boxplot_path = output_dir / "life_segment_wet_weight_boxplot.png"
    fig.savefig(boxplot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output_paths["life_segment_wet_weight_boxplot"] = str(boxplot_path)

    summary_csv = output_dir / "life_segment_wet_weight_summary.csv"
    segment_stats.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    output_paths["life_segment_wet_weight_summary"] = str(summary_csv)
    return output_paths


def save_decision_tree_scene_boxplots_and_table(
    df: pd.DataFrame,
    output_dir: Path,
    max_scenes: int = 6,
) -> Dict[str, str]:
    if df is None or df.empty or "tree_leaf_id" not in df.columns:
        return {}

    configure_plot_style()
    work = df.copy()
    work["tree_leaf_id"] = pd.to_numeric(work["tree_leaf_id"], errors="coerce")
    work = work.dropna(subset=["tree_leaf_id"]).copy()
    if work.empty:
        return {}
    work["tree_leaf_id"] = work["tree_leaf_id"].astype(int)

    leaf_counts = (
        work.groupby("tree_leaf_id", dropna=False)
        .agg(
            sample_count=("wet_weight", "size"),
            wet_weight_mean=("wet_weight", "mean"),
            wet_weight_std=("wet_weight", "std"),
        )
        .reset_index()
        .sort_values("wet_weight_mean")
        .head(max_scenes)
        .reset_index(drop=True)
    )
    leaf_counts["scene_label"] = [f"T{i}" for i in range(1, len(leaf_counts) + 1)]
    selected_leaf_ids = leaf_counts["tree_leaf_id"].tolist()

    plot_cols = [
        c
        for c in ["wet_weight", "印刷压力", "印刷高度偏移", "刮刀高度偏移"]
        if c in work.columns and pd.api.types.is_numeric_dtype(work[c])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, plot_cols):
        grouped = []
        for leaf_id in selected_leaf_ids:
            values = pd.to_numeric(work.loc[work["tree_leaf_id"] == leaf_id, col], errors="coerce").dropna()
            grouped.append(values.values)
        bp = ax.boxplot(grouped, tick_labels=leaf_counts["scene_label"].tolist(), patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#f28e2b", alpha=0.75)
        ax.set_title(f"{col} 箱线图")
        ax.set_xlabel("决策树场景")
        ax.set_ylabel(col)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    for ax in axes[len(plot_cols):]:
        ax.axis("off")

    scene_note = " | ".join([f"T{i+1}=Leaf {selected_leaf_ids[i]}" for i in range(len(selected_leaf_ids))])
    fig.suptitle(f"决策树 6 种场景变量箱线图\n{scene_note}", fontsize=15)
    boxplot_path = output_dir / "decision_tree_scene_param_boxplots.png"
    fig.savefig(boxplot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    table_df = leaf_counts.copy()
    feature_cols = [c for c in ["刮刀高度_上", "刮刀高度_下", "墨刀高度_上", "墨刀高度_下", "网版寿命"] if c in work.columns]
    if feature_cols:
        feature_ranges = (
            work.groupby("tree_leaf_id")[feature_cols]
            .agg(["min", "max"])
            .reset_index()
        )
        feature_ranges.columns = ["tree_leaf_id"] + [f"{a}_{b}" for a, b in feature_ranges.columns.tolist()[1:]]
        table_df = table_df.merge(feature_ranges, on="tree_leaf_id", how="left")

        def _format_scene_desc(row: pd.Series) -> str:
            parts: List[str] = []
            for col in feature_cols:
                vmin = row.get(f"{col}_min")
                vmax = row.get(f"{col}_max")
                if pd.isna(vmin) or pd.isna(vmax):
                    continue
                if abs(float(vmax) - float(vmin)) < 1e-9:
                    parts.append(f"{col}={float(vmin):.3f}")
                else:
                    parts.append(f"{col}={float(vmin):.3f}-{float(vmax):.3f}")
            return " | ".join(parts)

        table_df["scene_desc"] = table_df.apply(_format_scene_desc, axis=1)
    else:
        table_df["scene_desc"] = table_df["tree_leaf_id"].map(lambda x: f"Leaf {int(x)}")

    csv_path = output_dir / "decision_tree_scene_structure_table.csv"
    export_cols = ["scene_label", "scene_desc", "tree_leaf_id", "sample_count", "wet_weight_mean", "wet_weight_std"]
    other_cols = [c for c in table_df.columns if c not in export_cols]
    table_df[export_cols + other_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    show_df = table_df[["scene_label", "scene_desc", "tree_leaf_id", "sample_count", "wet_weight_mean", "wet_weight_std"]].copy()
    show_df["scene_desc"] = show_df["scene_desc"].map(lambda x: _truncate_label(x, 54))
    show_df.columns = ["场景编号", "场景描述", "叶子ID", "样本数", "湿重均值", "湿重标准差"]
    fig_h = max(4.8, 0.6 * len(show_df) + 2.3)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=show_df.round({"湿重均值": 6, "湿重标准差": 6}).values,
        colLabels=list(show_df.columns),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    ax.set_title("决策树 6 种场景结构树状表", pad=16)
    png_path = output_dir / "decision_tree_scene_structure_table.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {"boxplot_png": str(boxplot_path), "csv": str(csv_path), "png": str(png_path)}


def save_adjacent_delta_visualizations(deltas: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    if deltas.empty:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}

    if "changed_core_count" in deltas.columns:
        changed_counts = (
            deltas["changed_core_count"]
            .value_counts(dropna=False)
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(changed_counts.index.astype(str), changed_counts.values, color="#59a14f")
        ax.set_title("changed_core_count 分布条形图")
        ax.set_xlabel("changed_core_count")
        ax.set_ylabel("样本对数量")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        path = output_dir / "adjacent_changed_core_count_distribution.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["changed_core_count_distribution"] = str(path)

    if "context_change_count" in deltas.columns:
        context_counts = (
            deltas["context_change_count"]
            .value_counts(dropna=False)
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(context_counts.index.astype(str), context_counts.values, color="#f28e2b")
        ax.set_title("context_change_count 分布图")
        ax.set_xlabel("context_change_count")
        ax.set_ylabel("样本对数量")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        path = output_dir / "adjacent_context_change_count_distribution.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["context_change_count_distribution"] = str(path)

    scatter_params = [p for p in CORE_PARAMS if f"delta_{p}" in deltas.columns]
    if scatter_params and "delta_wet" in deltas.columns:
        n_cols = min(2, len(scatter_params))
        n_rows = int(np.ceil(len(scatter_params) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(8 * n_cols, max(5, 4.8 * n_rows)),
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(-1)

        for ax, param in zip(axes, scatter_params):
            x = pd.to_numeric(deltas[f"delta_{param}"], errors="coerce")
            y = pd.to_numeric(deltas["delta_wet"], errors="coerce")
            valid = x.notna() & y.notna()
            ax.scatter(x[valid], y[valid], s=15, alpha=0.55, color="#4e79a7", edgecolors="none")
            ax.axhline(0, color="#d62728", linewidth=0.9, alpha=0.8)
            ax.axvline(0, color="#555555", linewidth=0.9, alpha=0.6)
            ax.set_title(f"{param} 差分散点图")
            ax.set_xlabel(f"delta_{param}")
            ax.set_ylabel("delta_wet")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

        for ax in axes[len(scatter_params):]:
            ax.axis("off")

        path = output_dir / "adjacent_param_delta_scatter.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["param_delta_scatter"] = str(path)

    return output_paths


def save_single_param_visualizations(
    effect_df: pd.DataFrame,
    merged: pd.DataFrame,
    plan: Dict,
    output_dir: Path,
) -> Dict[str, str]:
    if effect_df is None or effect_df.empty:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}

    direction_counts = (
        effect_df.groupby(["参数", "方向结论"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if not direction_counts.empty:
        pivot = (
            direction_counts.pivot(index="参数", columns="方向结论", values="count")
            .fillna(0)
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bottom = np.zeros(len(pivot))
        color_pool = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
        for idx, col in enumerate(pivot.columns):
            vals = pivot[col].to_numpy(dtype=float)
            ax.bar(pivot.index, vals, bottom=bottom, label=str(col), color=color_pool[idx % len(color_pool)])
            bottom += vals
        ax.set_title("单参数影响方向条形图")
        ax.set_xlabel("参数")
        ax.set_ylabel("场景结论数量")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        path = output_dir / "single_param_direction_bar.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["direction_bar"] = str(path)

    return output_paths


def save_joint_param_visualizations(
    joint_effect_df: pd.DataFrame,
    merged: pd.DataFrame,
    plan: Dict,
    output_dir: Path,
) -> Dict[str, str]:
    if joint_effect_df is None or joint_effect_df.empty:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}

    r2_series = pd.to_numeric(joint_effect_df["model_r2"], errors="coerce").dropna()
    if not r2_series.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(r2_series, bins=min(10, max(5, len(r2_series))), color="#4e79a7", alpha=0.88, edgecolor="white")
        ax.axvline(r2_series.mean(), color="#d62728", linewidth=1.1, linestyle="--", label=f"均值={r2_series.mean():.3f}")
        ax.set_title("模型拟合优度分布图")
        ax.set_xlabel("model_r2")
        ax.set_ylabel("场景数量")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend()
        path = output_dir / "joint_model_r2_distribution.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["model_r2_distribution"] = str(path)

    main_cols = [f"coef_{p}" for p in CORE_PARAMS if f"coef_{p}" in joint_effect_df.columns]
    if main_cols:
        main_heatmap = (
            joint_effect_df.set_index("scene_key")[main_cols]
            .sort_index()
            .copy()
        )
        row_labels = [_truncate_label(x, 38) for x in main_heatmap.index]
        col_labels = [c.replace("coef_", "") for c in main_heatmap.columns]
        vals = main_heatmap.to_numpy(dtype=float)
        vmax = np.max(np.abs(vals)) if vals.size else 1.0
        vmax = vmax if vmax > 0 else 1.0

        fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(col_labels) + 2), max(5, 0.72 * len(row_labels) + 2)))
        im = ax.imshow(vals, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title("主效应系数热力图")
        ax.set_xlabel("参数")
        ax.set_ylabel("场景")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.2e}", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, label="主效应系数")
        path = output_dir / "joint_main_effect_heatmap.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["main_effect_heatmap"] = str(path)

    interaction_cols = [
        "coef_印刷压力x印刷高度偏移",
        "coef_印刷压力x刮刀高度偏移",
        "coef_印刷高度偏移x刮刀高度偏移",
        "coef_三参数交互",
    ]
    interaction_cols = [c for c in interaction_cols if c in joint_effect_df.columns]
    if interaction_cols:
        inter_heatmap = (
            joint_effect_df.set_index("scene_key")[interaction_cols]
            .sort_index()
            .copy()
        )
        row_labels = [_truncate_label(x, 38) for x in inter_heatmap.index]
        col_labels = [c.replace("coef_", "") for c in inter_heatmap.columns]
        vals = inter_heatmap.to_numpy(dtype=float)
        vmax = np.max(np.abs(vals)) if vals.size else 1.0
        vmax = vmax if vmax > 0 else 1.0

        fig, ax = plt.subplots(figsize=(max(10, 2.4 * len(col_labels) + 2), max(5, 0.72 * len(row_labels) + 2)))
        im = ax.imshow(vals, cmap="PuOr", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title("交互项系数热力图")
        ax.set_xlabel("交互项")
        ax.set_ylabel("场景")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.2e}", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, label="交互项系数")
        path = output_dir / "joint_interaction_effect_heatmap.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["interaction_effect_heatmap"] = str(path)

    latest_row = merged.loc[merged["wet_time"].idxmax()] if (not merged.empty and "wet_time" in merged.columns) else None
    src_scene = None
    joint_rec = plan.get("joint_recommendation") if isinstance(plan, dict) else None
    if isinstance(joint_rec, dict):
        src_scene = joint_rec.get("结论来源场景") or joint_rec.get("缁撹鏉ユ簮鍦烘櫙")
    if src_scene is None:
        src_scene = plan.get("latest_scene") if isinstance(plan, dict) else None

    local_row = pd.DataFrame()
    if src_scene is not None and "scene_key" in joint_effect_df.columns:
        local_row = joint_effect_df[joint_effect_df["scene_key"] == src_scene].head(1).copy()
    if local_row.empty and isinstance(plan, dict) and plan.get("latest_coarse_scene") is not None:
        local_row = (
            joint_effect_df[joint_effect_df["scene_coarse_key"] == plan.get("latest_coarse_scene")]
            .sort_values(["joint_samples", "model_r2"], ascending=False)
            .head(1)
            .copy()
        )
    if local_row.empty:
        local_row = joint_effect_df.sort_values(["joint_samples", "model_r2"], ascending=False).head(1).copy()

    if (not local_row.empty) and latest_row is not None:
        row = local_row.iloc[0]
        p1, p2, p3 = CORE_PARAMS
        cur = {}
        for p in CORE_PARAMS:
            if p in latest_row.index and pd.notna(latest_row[p]):
                cur[p] = float(latest_row[p])
            else:
                cur[p] = 0.0

        def predict_delta(x1: float, x2: float, x3: float) -> float:
            return float(
                row.get("coef_intercept", 0.0)
                + row.get(f"coef_{p1}", 0.0) * x1
                + row.get(f"coef_{p2}", 0.0) * x2
                + row.get(f"coef_{p3}", 0.0) * x3
                + row.get("coef_印刷压力x印刷高度偏移", 0.0) * x1 * x2
                + row.get("coef_印刷压力x刮刀高度偏移", 0.0) * x1 * x3
                + row.get("coef_印刷高度偏移x刮刀高度偏移", 0.0) * x2 * x3
                + row.get("coef_三参数交互", 0.0) * x1 * x2 * x3
            )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
        for ax, p in zip(axes, CORE_PARAMS):
            base = cur[p]
            span = max(abs(base) * 0.12, 0.3 if "偏移" in p else 5.0)
            xs = np.linspace(base - span, base + span, 60)

            if p == p1:
                ys = [predict_delta(x, cur[p2], cur[p3]) for x in xs]
            elif p == p2:
                ys = [predict_delta(cur[p1], x, cur[p3]) for x in xs]
            else:
                ys = [predict_delta(cur[p1], cur[p2], x) for x in xs]

            ax.plot(xs, ys, color="#4e79a7", linewidth=2.0)
            ax.axvline(base, color="#d62728", linestyle="--", linewidth=1.0, label="当前值")
            ax.axhline(0, color="#666666", linestyle=":", linewidth=0.9)
            ax.set_title(f"{p} 单场景局部响应")
            ax.set_xlabel(p)
            ax.set_ylabel("模型预测 delta_wet")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
            ax.legend()

        fig.suptitle(
            f"单场景局部响应图 | 场景: {_truncate_label(str(row.get('scene_key', 'NA')), 68)} | R2={float(row.get('model_r2', np.nan)):.3f}",
            fontsize=13,
        )
        path = output_dir / "joint_local_response.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["local_response"] = str(path)

    return output_paths


def save_joint_param_3d_surfaces(
    joint_effect_df: pd.DataFrame,
    deltas: pd.DataFrame,
    output_dir: Path,
    max_scenes: int = 6,
) -> Dict[str, str]:
    if joint_effect_df is None or joint_effect_df.empty:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}
    p1, p2, p3 = CORE_PARAMS

    scene_order = (
        joint_effect_df.sort_values(["joint_samples", "model_r2"], ascending=False)
        .head(max_scenes)
        .reset_index(drop=True)
    )

    summary_rows: List[Dict] = []

    for idx, (_, row) in enumerate(scene_order.iterrows(), start=1):
        scene_key = row.get("scene_key", f"scene_{idx}")
        dsrc = deltas[deltas["scene_key"] == scene_key].copy() if "scene_key" in deltas.columns else pd.DataFrame()
        if dsrc.empty:
            dsrc = deltas[deltas["scene_coarse_key"] == row.get("scene_coarse_key")].copy() if "scene_coarse_key" in deltas.columns else pd.DataFrame()

        def _axis_range(series: pd.Series, default_span: float) -> tuple[float, float]:
            vals = pd.to_numeric(series, errors="coerce").dropna()
            if len(vals):
                lo, hi = float(vals.min()), float(vals.max())
                if abs(hi - lo) < 1e-12:
                    lo -= default_span / 2
                    hi += default_span / 2
                pad = (hi - lo) * 0.12
                return lo - pad, hi + pad
            return -default_span, default_span

        r1 = _axis_range(dsrc.get(f"delta_{p1}", pd.Series(dtype=float)), 10.0)
        r2 = _axis_range(dsrc.get(f"delta_{p2}", pd.Series(dtype=float)), 0.5)
        r3 = _axis_range(dsrc.get(f"delta_{p3}", pd.Series(dtype=float)), 0.5)

        fix1 = float(pd.to_numeric(dsrc.get(f"delta_{p1}", pd.Series(dtype=float)), errors="coerce").median()) if not dsrc.empty else 0.0
        fix2 = float(pd.to_numeric(dsrc.get(f"delta_{p2}", pd.Series(dtype=float)), errors="coerce").median()) if not dsrc.empty else 0.0
        fix3 = float(pd.to_numeric(dsrc.get(f"delta_{p3}", pd.Series(dtype=float)), errors="coerce").median()) if not dsrc.empty else 0.0
        for vname, v in [("fix1", fix1), ("fix2", fix2), ("fix3", fix3)]:
            if not np.isfinite(v):
                if vname == "fix1":
                    fix1 = 0.0
                elif vname == "fix2":
                    fix2 = 0.0
                else:
                    fix3 = 0.0

        b0 = float(row.get("coef_intercept", 0.0))
        b1 = float(row.get(f"coef_{p1}", 0.0))
        b2 = float(row.get(f"coef_{p2}", 0.0))
        b3 = float(row.get(f"coef_{p3}", 0.0))
        b12 = float(row.get("coef_印刷压力x印刷高度偏移", 0.0))
        b13 = float(row.get("coef_印刷压力x刮刀高度偏移", 0.0))
        b23 = float(row.get("coef_印刷高度偏移x刮刀高度偏移", 0.0))
        b123 = float(row.get("coef_三参数交互", 0.0))

        def predict(x1, x2, x3):
            return (
                b0
                + b1 * x1
                + b2 * x2
                + b3 * x3
                + b12 * x1 * x2
                + b13 * x1 * x3
                + b23 * x2 * x3
                + b123 * x1 * x2 * x3
            )

        fig = plt.figure(figsize=(18, 5.8), constrained_layout=True)
        combos = [
            (p1, p2, fix3, r1, r2, lambda a, b: predict(a, b, fix3), f"{p3}固定={fix3:.3f}"),
            (p1, p3, fix2, r1, r3, lambda a, b: predict(a, fix2, b), f"{p2}固定={fix2:.3f}"),
            (p2, p3, fix1, r2, r3, lambda a, b: predict(fix1, a, b), f"{p1}固定={fix1:.3f}"),
        ]

        for j, (xlab, ylab, _, xr, yr, fn, fixed_note) in enumerate(combos, start=1):
            ax = fig.add_subplot(1, 3, j, projection="3d")
            xs = np.linspace(xr[0], xr[1], 32)
            ys = np.linspace(yr[0], yr[1], 32)
            X, Y = np.meshgrid(xs, ys)
            Z = fn(X, Y)
            ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.88)
            ax.set_title(f"{xlab} x {ylab}\n{fixed_note}", fontsize=10)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_zlabel("预测 delta_wet")

        fig.suptitle(
            f"场景{idx} 联动多项式模型 3D响应面 | samples={int(row.get('joint_samples', 0))} | R2={float(row.get('model_r2', np.nan)):.3f}\n{_truncate_label(scene_key, 88)}",
            fontsize=13,
        )
        out_path = output_dir / f"joint_scene_{idx:02d}_3d_surfaces.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths[f"scene_{idx:02d}"] = str(out_path)

        summary_rows.append(
            {
                "scene_index": idx,
                "scene_key": scene_key,
                "joint_samples": int(row.get("joint_samples", 0)),
                "model_r2": float(row.get("model_r2", np.nan)),
                "fixed_delta_印刷压力": fix1,
                "fixed_delta_印刷高度偏移": fix2,
                "fixed_delta_刮刀高度偏移": fix3,
                "figure_path": str(out_path),
            }
        )

    if summary_rows:
        summary_csv = output_dir / "joint_scene_3d_surfaces_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        output_paths["summary_csv"] = str(summary_csv)

    return output_paths


def _first_present(d: Dict, keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def save_latest_adjustment_visualizations(plan: Dict, output_dir: Path) -> Dict[str, str]:
    if not isinstance(plan, dict) or not plan:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}

    current_wet = float(plan.get("current_wet_weight", np.nan))
    target_wet = float(plan.get("target_wet_weight", np.nan))
    required_reduction = float(plan.get("required_reduction", np.nan))

    joint_rec = plan.get("joint_recommendation")
    if isinstance(joint_rec, dict):
        items = _first_present(joint_rec, ["组合调参", "缁勫悎璋冨弬"], default=[])
        if items:
            labels = []
            contributions = []
            for item in items:
                labels.append(str(_first_present(item, ["参数", "鍙傛暟"], default="NA")))
                contributions.append(
                    float(
                        _first_present(
                            item,
                            ["预计降幅贡献(线性近似)", "棰勮闄嶅箙璐＄尞(绾挎€ц繎浼?"],
                            default=0.0,
                        )
                    )
                )

            fig, ax = plt.subplots(figsize=(10, 5.2))
            cumulative = 0.0
            starts = []
            for c in contributions:
                starts.append(cumulative)
                cumulative += c

            bar_colors = ["#59a14f" if c >= 0 else "#e15759" for c in contributions]
            ax.bar(labels, contributions, bottom=starts, color=bar_colors, width=0.55)
            for i, c in enumerate(contributions):
                ax.text(i, starts[i] + c, f"{c:.6f}", ha="center", va="bottom", fontsize=9)
            ax.axhline(0, color="#666666", linewidth=0.9)
            ax.set_title("联动调参预计降幅贡献图")
            ax.set_ylabel("预计降幅贡献")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
            total_pred = float(
                _first_present(
                    joint_rec,
                    ["预计总降幅(线性近似)", "棰勮鎬婚檷骞?绾挎€ц繎浼?"],
                    default=np.nan,
                )
            )
            if np.isfinite(total_pred):
                ax.text(
                    0.98,
                    0.95,
                    f"合计预计降幅: {total_pred:.6f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox={"facecolor": "#f7f7f7", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
                )
            path = output_dir / "latest_adjustment_contribution_waterfall.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            output_paths["contribution_waterfall"] = str(path)

            table_rows = []
            for item in items:
                param = str(_first_present(item, ["参数", "鍙傛暟"], default="NA"))
                current_v = _first_present(item, ["当前值", "褰撳墠鍊?"], default=np.nan)
                action = str(_first_present(item, ["建议方向", "寤鸿鏂瑰悜"], default="NA"))
                delta_v = _first_present(item, ["建议变化量", "寤鸿鍙樺寲閲?"], default=np.nan)
                new_v = _first_present(item, ["建议新值", "寤鸿鏂板€?"], default=np.nan)
                contrib = _first_present(
                    item,
                    ["预计降幅贡献(线性近似)", "棰勮闄嶅箙璐＄尞(绾挎€ц繎浼?"],
                    default=np.nan,
                )
                table_rows.append(
                    [
                        param,
                        f"{float(current_v):.4f}" if pd.notna(current_v) else "",
                        action,
                        f"{abs(float(delta_v)):.4f}" if pd.notna(delta_v) else "",
                        f"{float(new_v):.4f}" if pd.notna(new_v) else "",
                        f"{float(contrib):.6f}" if pd.notna(contrib) else "",
                    ]
                )

            fig_h = max(4.5, 0.55 * len(table_rows) + 2.8)
            fig, ax = plt.subplots(figsize=(14, fig_h))
            ax.axis("off")
            title_lines = [
                "最新样本联动调参建议看板",
                f"时间: {plan.get('latest_time', 'NA')}",
                f"场景: {_truncate_label(plan.get('latest_scene', 'NA'), 72)}",
                f"当前湿重: {current_wet:.6f}  目标湿重: {target_wet:.6f}",
            ]
            table = ax.table(
                cellText=table_rows,
                colLabels=["参数", "当前值", "建议方向", "建议变化量", "建议新值", "预计降幅贡献"],
                loc="center",
                cellLoc="left",
                colLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.45)
            ax.set_title("\n".join(title_lines), pad=16)
            path = output_dir / "latest_adjustment_dashboard.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            output_paths["adjustment_dashboard"] = str(path)
    elif plan.get("recommendations"):
        recs = plan.get("recommendations", [])
        fig_h = max(4.5, 0.55 * len(recs) + 2.8)
        table_rows = []
        for item in recs:
            table_rows.append(
                [
                    str(_first_present(item, ["参数", "鍙傛暟"], default="NA")),
                    f"{float(_first_present(item, ['当前值', '褰撳墠鍊?'], np.nan)):.4f}",
                    str(_first_present(item, ["建议方向", "寤鸿鏂瑰悜"], default="NA")),
                    f"{abs(float(_first_present(item, ['建议变化量', '寤鸿鍙樺寲閲?'], np.nan))):.4f}",
                    f"{float(_first_present(item, ['建议新值', '寤鸿鏂板€?'], np.nan)):.4f}",
                    f"{float(_first_present(item, ['基于历史斜率预计降幅', '鍩轰簬鍘嗗彶鏂滅巼棰勮闄嶅箙'], np.nan)):.6f}",
                ]
            )
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.axis("off")
        table = ax.table(
            cellText=table_rows,
            colLabels=["参数", "当前值", "建议方向", "建议变化量", "建议新值", "预计降幅"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)
        ax.set_title("最新样本单参数建议看板", pad=16)
        path = output_dir / "latest_adjustment_dashboard.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["adjustment_dashboard"] = str(path)

    return output_paths


def save_backtest_visualizations(backtest_df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    if backtest_df is None or backtest_df.empty:
        return {}

    configure_plot_style()
    output_paths: Dict[str, str] = {}

    plot_df = backtest_df.copy()
    x = pd.to_datetime(plot_df["current_time"], errors="coerce")
    current_wet = pd.to_numeric(plot_df["current_wet_weight"], errors="coerce")
    target_wet = pd.to_numeric(plot_df["target_wet_weight"], errors="coerce")
    actual_next_wet = pd.to_numeric(plot_df["next_wet_weight"], errors="coerce")
    predicted_next_wet = pd.to_numeric(plot_df.get("predicted_next_wet_weight"), errors="coerce")
    noisy_predicted_next_wet = pd.to_numeric(plot_df.get("noisy_predicted_next_wet_weight"), errors="coerce")

    valid = x.notna()
    if valid.any():
        fig, ax = plt.subplots(figsize=(12, 5.2))
        if current_wet.notna().any():
            valid_current = x.notna() & current_wet.notna()
            ax.plot(x[valid_current], current_wet[valid_current], label="当前湿重", color="#bab0ab", linewidth=1.2, alpha=0.9)
        if target_wet.notna().any():
            valid_target = x.notna() & target_wet.notna()
            ax.plot(x[valid_target], target_wet[valid_target], label="目标湿重", color="#f28e2b", linewidth=1.4, linestyle="-.")
        if predicted_next_wet.notna().any():
            valid_pred = x.notna() & predicted_next_wet.notna()
            ax.plot(x[valid_pred], predicted_next_wet[valid_pred], label="预计下一步湿重", color="#e15759", linewidth=1.5, linestyle="--")
        if noisy_predicted_next_wet.notna().any():
            valid_noisy = x.notna() & noisy_predicted_next_wet.notna()
            ax.plot(
                x[valid_noisy],
                noisy_predicted_next_wet[valid_noisy],
                label="带扰动预计下一步湿重",
                color="#59a14f",
                linewidth=1.3,
                linestyle=":",
            )
        ax.set_title("最后10%样本回测时间序列（湿重值）")
        ax.set_xlabel("时间")
        ax.set_ylabel("湿重值")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate(rotation=25)
        path = output_dir / "backtest_timeline.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["timeline"] = str(path)

    valid = predicted_next_wet.notna() & actual_next_wet.notna()
    if valid.any():
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        ax.scatter(
            predicted_next_wet[valid],
            actual_next_wet[valid],
            s=22,
            alpha=0.65,
            color="#59a14f",
            edgecolors="none",
            label="原始预计",
        )
        if noisy_predicted_next_wet.notna().any():
            valid_noisy = noisy_predicted_next_wet.notna() & actual_next_wet.notna()
            ax.scatter(
                noisy_predicted_next_wet[valid_noisy],
                actual_next_wet[valid_noisy],
                s=18,
                alpha=0.45,
                color="#f28e2b",
                edgecolors="none",
                label="带扰动预计",
            )
        lim_low = min(float(predicted_next_wet[valid].min()), float(actual_next_wet[valid].min()))
        lim_high = max(float(predicted_next_wet[valid].max()), float(actual_next_wet[valid].max()))
        if noisy_predicted_next_wet.notna().any():
            lim_low = min(lim_low, float(noisy_predicted_next_wet[valid_noisy].min()))
            lim_high = max(lim_high, float(noisy_predicted_next_wet[valid_noisy].max()))
        if lim_high - lim_low < 1e-12:
            lim_high = lim_low + 1e-6
        ax.plot([lim_low, lim_high], [lim_low, lim_high], color="#d62728", linestyle="--", linewidth=1.0, label="理想对角线")
        ax.set_title("回测：预计下一步湿重 vs 实际下一步湿重")
        ax.set_xlabel("预计下一步湿重")
        ax.set_ylabel("实际下一步湿重")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend(loc="best")
        path = output_dir / "backtest_expected_vs_actual.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths["expected_vs_actual"] = str(path)

    return output_paths


