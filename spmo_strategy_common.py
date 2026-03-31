#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from spmo_analysis import analyze_joint_param_effects, analyze_single_param_effects, build_adjacent_deltas
from spmo_data_scene import build_hierarchical_scene_keys, build_scene_keys_from_config, load_and_match


def build_strategy_context(
    data_dir: Path,
    max_match_minutes: int,
    max_pair_gap_minutes: int,
    max_context_changes: int,
    min_samples: int,
    min_samples_joint: int,
    context_tol: float,
    coarse_round_step: float,
    life_bin_size: float,
    core_stable_tol: float,
    scene_config: Dict,
) -> Dict:
    merged = load_and_match(data_dir, max_match_minutes=max_match_minutes)

    default_scene_cols = [c for c in ["刮刀高度_上", "刮刀高度_下", "墨刀高度_上", "墨刀高度_下"] if c in merged.columns]
    if not default_scene_cols:
        raise ValueError("未识别到场景拆分列（刮刀/墨刀高度）")

    merged = build_hierarchical_scene_keys(
        merged,
        coarse_scene_cols=default_scene_cols,
        screen_life_col="网版寿命" if "网版寿命" in merged.columns else None,
        coarse_round_step=coarse_round_step,
        life_bin_size=life_bin_size,
    )
    merged = build_scene_keys_from_config(merged, scene_config=scene_config)

    scene_cols = [c for c in scene_config.get("coarse_scene", {}).get("features", []) if c in merged.columns]
    deltas = build_adjacent_deltas(
        merged,
        max_gap_minutes=max_pair_gap_minutes,
        context_tol=context_tol,
    )
    effects = analyze_single_param_effects(
        deltas,
        max_context_changes=max_context_changes,
        min_samples=min_samples,
        core_stable_tol=core_stable_tol,
    )
    if effects.empty:
        effects = analyze_single_param_effects(
            deltas,
            max_context_changes=max_context_changes + 4,
            min_samples=max(3, min_samples // 2),
            core_stable_tol=max(core_stable_tol, 0.1),
        )

    joint_effects = analyze_joint_param_effects(
        deltas,
        max_context_changes=max_context_changes,
        min_samples_joint=min_samples_joint,
        core_stable_tol=core_stable_tol,
    )

    scene_diag = (
        merged.groupby(["scene_fine_key", "scene_coarse_key", "life_segment"], dropna=False)
        .agg(scene_sample_count=("wet_weight", "size"))
        .reset_index()
        .sort_values(["scene_sample_count", "scene_fine_key"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if effects is not None and (not effects.empty):
        single_scene_counts = (
            effects.groupby("scene_key", dropna=False)
            .agg(
                single_param_result_rows=("scene_key", "size"),
                single_param_count=("参数", "nunique"),
                single_param_list=("参数", lambda s: " | ".join(sorted(map(str, set(s))))),
            )
            .reset_index()
            .rename(columns={"scene_key": "scene_fine_key"})
        )
        scene_diag = scene_diag.merge(single_scene_counts, on="scene_fine_key", how="left")
    else:
        scene_diag["single_param_result_rows"] = 0
        scene_diag["single_param_count"] = 0
        scene_diag["single_param_list"] = ""

    if joint_effects is not None and (not joint_effects.empty):
        joint_scene_counts = (
            joint_effects[["scene_key", "joint_samples", "model_r2"]]
            .drop_duplicates(subset=["scene_key"])
            .rename(columns={"scene_key": "scene_fine_key"})
        )
        joint_scene_counts["has_joint_model"] = 1
        scene_diag = scene_diag.merge(joint_scene_counts, on="scene_fine_key", how="left")
    else:
        scene_diag["joint_samples"] = np.nan
        scene_diag["model_r2"] = np.nan
        scene_diag["has_joint_model"] = 0

    scene_diag["single_param_result_rows"] = pd.to_numeric(scene_diag.get("single_param_result_rows"), errors="coerce").fillna(0).astype(int)
    scene_diag["single_param_count"] = pd.to_numeric(scene_diag.get("single_param_count"), errors="coerce").fillna(0).astype(int)
    scene_diag["single_param_list"] = scene_diag.get("single_param_list", "").fillna("")
    scene_diag["has_single_param"] = (scene_diag["single_param_result_rows"] > 0).astype(int)
    scene_diag["has_joint_model"] = pd.to_numeric(scene_diag.get("has_joint_model"), errors="coerce").fillna(0).astype(int)
    scene_diag["has_any_effect"] = ((scene_diag["has_single_param"] == 1) | (scene_diag["has_joint_model"] == 1)).astype(int)

    return {
        "merged": merged,
        "scene_cols": scene_cols,
        "deltas": deltas,
        "effects": effects,
        "joint_effects": joint_effects,
        "scene_diag": scene_diag,
    }
