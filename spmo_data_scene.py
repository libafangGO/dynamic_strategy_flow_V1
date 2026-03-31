#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from spmo_config import CANDIDATES, _format_scene_value, find_col, parse_datetime_safe, uniquify_columns

def load_and_match(data_dir: Path, max_match_minutes: int) -> pd.DataFrame:
    process_path = data_dir / "丝网印刷参数数据-预处理后.csv"
    wet_path = data_dir / "丝网印刷近三个月湿重1-2(1).xls"

    if not process_path.exists():
        raise FileNotFoundError(f"未找到文件: {process_path}")
    if not wet_path.exists():
        raise FileNotFoundError(f"未找到文件: {wet_path}")

    process_df = uniquify_columns(pd.read_csv(process_path))
    wet_df = uniquify_columns(pd.read_excel(wet_path, sheet_name=0))

    process_time_col = find_col(process_df, CANDIDATES["process_time"])
    wet_time_col = find_col(wet_df, CANDIDATES["wet_time"])
    wet_value_col = find_col(wet_df, CANDIDATES["wet_value"])

    if process_time_col is None:
        raise ValueError("工艺参数文件缺少 INSERTTIME")
    if wet_time_col is None or wet_value_col is None:
        raise ValueError("湿重文件缺少采集时间或湿重列")

    process_df[process_time_col] = parse_datetime_safe(process_df[process_time_col])
    wet_df[wet_time_col] = parse_datetime_safe(wet_df[wet_time_col])

    process_df = process_df.dropna(subset=[process_time_col]).sort_values(process_time_col)
    wet_df = wet_df.dropna(subset=[wet_time_col, wet_value_col]).sort_values(wet_time_col)

    process_name_col = find_col(wet_df, CANDIDATES["process_name"])
    if process_name_col is not None:
        mask = wet_df[process_name_col].astype(str).str.contains("丝网|˿��", na=False)
        if mask.any():
            wet_df = wet_df[mask].copy()

    pressure_col = find_col(process_df, CANDIDATES["pressure"])
    print_offset_col = find_col(process_df, CANDIDATES["print_offset"])
    scraper_offset_col = find_col(process_df, CANDIDATES["scraper_offset"])

    if pressure_col is None or print_offset_col is None or scraper_offset_col is None:
        raise ValueError("无法识别三个核心参数列（印刷压力/印刷高度偏移/刮刀高度偏移）")

    scene_candidates = [
        ("刮刀高度_上", find_col(process_df, CANDIDATES["scraper_height_up"])),
        ("刮刀高度_下", find_col(process_df, CANDIDATES["scraper_height_down"])),
        ("墨刀高度_上", find_col(process_df, CANDIDATES["ink_height_up"])),
        ("墨刀高度_下", find_col(process_df, CANDIDATES["ink_height_down"])),
    ]

    cols = list(process_df.columns)
    fallback_idx_map = {
        "刮刀高度_上": 12,
        "刮刀高度_下": 13,
        "墨刀高度_上": 24,
        "墨刀高度_下": 25,
    }
    fixed_scene = {}
    for name, col in scene_candidates:
        if col is not None and col in process_df.columns:
            fixed_scene[name] = col
        else:
            idx = fallback_idx_map[name]
            if idx < len(cols):
                fixed_scene[name] = cols[idx]

    optional_cols = {
        "印刷速度": find_col(process_df, CANDIDATES["speed"]),
        "网版寿命": find_col(process_df, CANDIDATES["screen_life"]),
        "设备": find_col(process_df, CANDIDATES["equipment"]) or find_col(wet_df, CANDIDATES["equipment"]),
    }

    process_keep = [process_time_col, pressure_col, print_offset_col, scraper_offset_col] + list(fixed_scene.values())
    process_keep = [c for c in process_keep if c in process_df.columns]
    for c in optional_cols.values():
        if c is not None and c in process_df.columns and c not in process_keep:
            process_keep.append(c)

    process_small = process_df[process_keep].copy().rename(columns={process_time_col: "process_time"})
    wet_small = wet_df[[wet_time_col, wet_value_col]].copy().rename(columns={wet_time_col: "wet_time", wet_value_col: "wet_weight"})
    # Keep one conversion point so all downstream analyses/exports use the same wet-weight unit.
    wet_small["wet_weight"] = pd.to_numeric(wet_small["wet_weight"], errors="coerce") * 1000.0

    merged = pd.merge_asof(
        wet_small.sort_values("wet_time"),
        process_small.sort_values("process_time"),
        left_on="wet_time",
        right_on="process_time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=max_match_minutes),
    )

    merged["time_diff_min"] = (merged["wet_time"] - merged["process_time"]).dt.total_seconds() / 60.0
    merged = merged.dropna(subset=["process_time", "wet_weight"]).reset_index(drop=True)

    rename_map = {
        pressure_col: "印刷压力",
        print_offset_col: "印刷高度偏移",
        scraper_offset_col: "刮刀高度偏移",
    }
    for logical_name, col in fixed_scene.items():
        rename_map[col] = logical_name
    for logical_name, col in optional_cols.items():
        if col is not None:
            rename_map[col] = logical_name

    merged = merged.rename(columns=rename_map)

    numeric_cols = [c for c in merged.columns if c not in ["wet_time", "process_time", "设备"]]
    for c in numeric_cols:
        if merged[c].dtype == object:
            converted = pd.to_numeric(merged[c], errors="coerce")
            if converted.notna().mean() > 0.95:
                merged[c] = converted

    return merged


def _format_life_range_label(low: float, high: float, bin_size: float) -> str:
    if bin_size >= 10000:
        low_w = low / 10000.0
        high_w = high / 10000.0
        if abs(low_w - round(low_w)) < 1e-9 and abs(high_w - round(high_w)) < 1e-9:
            return f"{int(round(low_w))}-{int(round(high_w))}万"
        return f"{low_w:.1f}-{high_w:.1f}万"
    if abs(low - round(low)) < 1e-9 and abs(high - round(high)) < 1e-9:
        return f"{int(round(low))}-{int(round(high))}"
    return f"{low:.1f}-{high:.1f}"


def build_hierarchical_scene_keys(
    df: pd.DataFrame,
    coarse_scene_cols: List[str],
    screen_life_col: Optional[str],
    coarse_round_step: float = 0.5,
    life_bin_size: float = 200000,
) -> pd.DataFrame:
    out = df.copy()

    grouped_scene_cols: List[str] = []
    for c in coarse_scene_cols:
        gcol = f"{c}_组"
        grouped_scene_cols.append(gcol)
        num = pd.to_numeric(out[c], errors="coerce")
        if coarse_round_step > 0:
            out[gcol] = np.round(num / coarse_round_step) * coarse_round_step
        else:
            out[gcol] = num

    out["scene_coarse_key"] = out[grouped_scene_cols].apply(
        lambda row: " | ".join([f"{coarse_scene_cols[i]}={_format_scene_value(row[grouped_scene_cols[i]])}" for i in range(len(coarse_scene_cols))]),
        axis=1,
    )

    out["life_segment"] = "L1"
    if screen_life_col is not None and screen_life_col in out.columns:
        life = pd.to_numeric(out[screen_life_col], errors="coerce")
        if life_bin_size > 0:
            seg_idx = np.floor(life / life_bin_size)
            seg_idx = seg_idx.where(seg_idx >= 0)
            seg_idx = seg_idx.fillna(0).astype(int)

            lows = seg_idx * int(life_bin_size)
            highs = (seg_idx + 1) * int(life_bin_size)
            out["life_segment"] = [
                _format_life_range_label(float(lo), float(hi), float(life_bin_size))
                for lo, hi in zip(lows, highs)
            ]

    out["scene_fine_key"] = out["scene_coarse_key"] + " | life=" + out["life_segment"].astype(str)
    return out


def _build_tree_labels_from_features(df: pd.DataFrame, leaf_col: str, feature_cols: List[str], prefix: str) -> pd.Series:
    labels: Dict[int, str] = {}
    for leaf_id, g in df.groupby(leaf_col, dropna=False):
        parts: List[str] = []
        for col in feature_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            vmin = float(vals.min())
            vmax = float(vals.max())
            if abs(vmax - vmin) < 1e-9:
                parts.append(f"{col}={vmin:.3f}")
            else:
                parts.append(f"{col}={vmin:.3f}-{vmax:.3f}")
        if not parts:
            parts.append(f"{prefix}{int(leaf_id)}")
        labels[int(leaf_id)] = " | ".join(parts)
    return df[leaf_col].astype(int).map(labels)


def build_scene_keys_from_config(df: pd.DataFrame, scene_config: Dict) -> pd.DataFrame:
    out = df.copy()

    coarse_cfg = scene_config.get("coarse_scene", {})
    coarse_features = [c for c in coarse_cfg.get("features", []) if c in out.columns]
    if len(coarse_features) == 0:
        raise ValueError("配置中的 coarse_scene.features 未在数据中识别到")

    coarse_method = str(coarse_cfg.get("method", "fixed")).lower()
    if coarse_method == "tree":
        train_df = out[coarse_features + ["wet_weight"]].copy()
        for col in coarse_features + ["wet_weight"]:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        valid_mask = train_df.notna().all(axis=1)
        if not valid_mask.any():
            raise ValueError("粗场景树模型缺少可用训练样本")
        model = DecisionTreeRegressor(
            random_state=42,
            max_leaf_nodes=int(coarse_cfg.get("tree", {}).get("max_leaf_nodes", 6)),
            min_samples_leaf=int(coarse_cfg.get("tree", {}).get("min_samples_leaf", 20)),
        )
        model.fit(train_df.loc[valid_mask, coarse_features], train_df.loc[valid_mask, "wet_weight"])
        out["scene_coarse_leaf_id"] = -1
        out.loc[valid_mask, "scene_coarse_leaf_id"] = model.apply(train_df.loc[valid_mask, coarse_features])
        out["scene_coarse_leaf_id"] = out["scene_coarse_leaf_id"].astype(int)
        out["scene_coarse_key"] = _build_tree_labels_from_features(out, "scene_coarse_leaf_id", coarse_features, "coarse_leaf_")
    else:
        coarse_round_step = float(coarse_cfg.get("fixed_round_step", 0.5))
        grouped_scene_cols: List[str] = []
        for c in coarse_features:
            gcol = f"{c}_cfg_group"
            grouped_scene_cols.append(gcol)
            num = pd.to_numeric(out[c], errors="coerce")
            out[gcol] = np.round(num / coarse_round_step) * coarse_round_step if coarse_round_step > 0 else num
        out["scene_coarse_key"] = out[grouped_scene_cols].apply(
            lambda row: " | ".join([f"{coarse_features[i]}={_format_scene_value(row[grouped_scene_cols[i]])}" for i in range(len(coarse_features))]),
            axis=1,
        )

    life_cfg = scene_config.get("life_segment", {})
    out["life_segment"] = "L1"
    if bool(life_cfg.get("enabled", True)):
        life_feature = str(life_cfg.get("feature", "网版寿命"))
        if life_feature in out.columns:
            life = pd.to_numeric(out[life_feature], errors="coerce")
            life_method = str(life_cfg.get("method", "fixed")).lower()
            if life_method == "tree":
                valid_mask = life.notna() & pd.to_numeric(out["wet_weight"], errors="coerce").notna()
                if valid_mask.any():
                    model = DecisionTreeRegressor(
                        random_state=42,
                        max_leaf_nodes=int(life_cfg.get("tree", {}).get("max_leaf_nodes", 4)),
                        min_samples_leaf=int(life_cfg.get("tree", {}).get("min_samples_leaf", 20)),
                    )
                    model.fit(life.loc[valid_mask].to_frame(name=life_feature), pd.to_numeric(out.loc[valid_mask, "wet_weight"], errors="coerce"))
                    out["life_leaf_id"] = -1
                    out.loc[valid_mask, "life_leaf_id"] = model.apply(life.loc[valid_mask].to_frame(name=life_feature))
                    out["life_leaf_id"] = out["life_leaf_id"].astype(int)
                    labels: Dict[int, str] = {}
                    for leaf_id, g in out.loc[valid_mask].groupby("life_leaf_id"):
                        vals = pd.to_numeric(g[life_feature], errors="coerce").dropna()
                        if len(vals) == 0:
                            labels[int(leaf_id)] = "L1"
                        else:
                            vmin = float(vals.min())
                            vmax = float(vals.max())
                            labels[int(leaf_id)] = f"{life_feature}={vmin:.0f}" if abs(vmax - vmin) < 1e-9 else f"{life_feature}={vmin:.0f}-{vmax:.0f}"
                    out["life_segment"] = out["life_leaf_id"].map(labels).fillna("L1")
            else:
                life_bin_size = float(life_cfg.get("fixed_bin_size", 200000))
                if life_bin_size > 0:
                    seg_idx = np.floor(life / life_bin_size)
                    seg_idx = seg_idx.where(seg_idx >= 0)
                    seg_idx = seg_idx.fillna(0).astype(int)
                    lows = seg_idx * int(life_bin_size)
                    highs = (seg_idx + 1) * int(life_bin_size)
                    out["life_segment"] = [
                        _format_life_range_label(float(lo), float(hi), float(life_bin_size))
                        for lo, hi in zip(lows, highs)
                    ]

    out["scene_fine_key"] = out["scene_coarse_key"] if not bool(life_cfg.get("enabled", True)) else out["scene_coarse_key"] + " | life=" + out["life_segment"].astype(str)
    return out




