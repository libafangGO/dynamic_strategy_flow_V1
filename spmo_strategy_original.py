#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from spmo_analysis import latest_scene_adjustment
from spmo_backtest import run_backtest
from spmo_config import CORE_PARAMS
from spmo_visualizations import (
    run_scene_decision_tree,
    save_coarse_scene_param_boxplots,
    save_coarse_scene_split_points_visualization,
    save_decision_tree_scene_boxplots_and_table,
    save_fixed_coarse_scene_structure_table,
    save_hierarchical_scene_visualizations,
    save_latest_adjustment_visualizations,
    save_matched_samples_visualization,
    save_single_param_visualizations,
    save_unused_adjacent_sample_visualization,
)


def run_original_strategy(
    context: Dict,
    output_dir: Path,
    max_pair_gap_minutes: int,
    backtest_ratio: float,
    backtest_noise_scale: float,
    backtest_noise_seed: int,
    max_context_changes: int,
    min_samples: int,
    min_samples_joint: int,
    context_tol: float,
    core_stable_tol: float,
    scene_config: Dict,
    joint_weight_config: Dict,
) -> Tuple[Dict, Dict]:
    merged = context["merged"]
    scene_cols = context["scene_cols"]
    deltas = context["deltas"]
    effects = context["effects"]
    joint_effects = context["joint_effects"]
    scene_diag = context["scene_diag"]

    output_dir.mkdir(parents=True, exist_ok=True)
    match_vis_path = save_matched_samples_visualization(merged, output_dir)
    hierarchical_vis_paths = save_hierarchical_scene_visualizations(merged, output_dir)
    coarse_scene_distribution_paths = save_coarse_scene_param_boxplots(
        merged,
        output_dir,
        max_scenes=int(merged["scene_coarse_key"].nunique()),
    )
    coarse_scene_split_point_paths = save_coarse_scene_split_points_visualization(
        merged,
        output_dir,
        scene_config=scene_config,
    )
    fixed_coarse_scene_table_paths = save_fixed_coarse_scene_structure_table(
        merged,
        output_dir,
        max_scenes=int(merged["scene_coarse_key"].nunique()),
    )
    decision_tree_scene_summary = run_scene_decision_tree(
        merged,
        output_dir,
        max_leaf_nodes=max(2, int(merged["scene_coarse_key"].nunique())),
        min_samples_leaf=max(20, int(len(merged) * 0.02)),
        feature_cols=scene_cols,
    )
    decision_tree_assignment_df = pd.read_csv(output_dir / "decision_tree_scene_assignments.csv")
    decision_tree_scene_display_paths = save_decision_tree_scene_boxplots_and_table(
        decision_tree_assignment_df,
        output_dir,
        max_scenes=6,
    )
    unused_adjacent_sample_vis = save_unused_adjacent_sample_visualization(merged, deltas, output_dir)

    plan = latest_scene_adjustment(
        merged,
        effects,
        joint_effects,
        deltas,
        core_stable_tol=core_stable_tol,
        joint_weight_config=joint_weight_config,
        target_wet_config=scene_config.get("target_wet_config"),
    )
    single_param_vis_paths = save_single_param_visualizations(effects, merged, plan, output_dir)
    latest_adjustment_vis_paths = save_latest_adjustment_visualizations(plan, output_dir)
    backtest_summary = run_backtest(
        merged=merged,
        output_dir=output_dir,
        test_ratio=backtest_ratio,
        max_pair_gap_minutes=max_pair_gap_minutes,
        max_context_changes=max_context_changes,
        min_samples=min_samples,
        min_samples_joint=min_samples_joint,
        context_tol=context_tol,
        core_stable_tol=core_stable_tol,
        noise_scale=backtest_noise_scale,
        noise_seed=backtest_noise_seed,
        joint_weight_config=joint_weight_config,
        target_wet_config=scene_config.get("target_wet_config"),
    )

    scene_diag_path = output_dir / "scene_model_coverage_diagnosis.csv"
    scene_diag.to_csv(scene_diag_path, index=False, encoding="utf-8-sig")
    merged.to_csv(output_dir / "matched_samples.csv", index=False, encoding="utf-8-sig")
    merged[
        ["wet_time", "wet_weight", "scene_coarse_key", "life_segment", "scene_fine_key"]
        + [c for c in CORE_PARAMS if c in merged.columns]
    ].to_csv(output_dir / "hierarchical_scene_samples.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(output_dir / "adjacent_deltas.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(output_dir / "scene_single_param_effects.csv", index=False, encoding="utf-8-sig")
    joint_effects.to_csv(output_dir / "scene_joint_param_effects.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "latest_scene_adjustment_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples": int(len(merged)),
        "coarse_scenes": int(merged["scene_coarse_key"].nunique()),
        "fine_scenes": int(merged["scene_fine_key"].nunique()),
        "adjacent_pairs": int(len(deltas)),
        "effect_rows": int(len(effects)),
        "joint_effect_rows": int(len(joint_effects)),
        "coarse_scene_cols": scene_cols,
        "core_params": CORE_PARAMS,
        "coarse_round_step": scene_config.get("coarse_scene", {}).get("fixed_round_step"),
        "life_bin_size": scene_config.get("life_segment", {}).get("fixed_bin_size"),
        "core_stable_tol": core_stable_tol,
        "scene_config": scene_config,
        "joint_weight_config": joint_weight_config,
        "matched_samples_visualization": str(match_vis_path) if match_vis_path is not None else None,
        "unused_adjacent_sample_visualization": str(unused_adjacent_sample_vis) if unused_adjacent_sample_vis is not None else None,
        "hierarchical_scene_visualizations": hierarchical_vis_paths,
        "coarse_scene_param_distributions": coarse_scene_distribution_paths,
        "coarse_scene_split_points": coarse_scene_split_point_paths,
        "scene_model_coverage_diagnosis": str(scene_diag_path),
        "fixed_coarse_scene_structure_table": fixed_coarse_scene_table_paths,
        "decision_tree_scene_split": decision_tree_scene_summary,
        "decision_tree_scene_display": decision_tree_scene_display_paths,
        "single_param_visualizations": single_param_vis_paths,
        "latest_adjustment_visualizations": latest_adjustment_vis_paths,
        "backtest_last_10pct": backtest_summary,
        "latest_scene": plan.get("latest_scene"),
        "latest_coarse_scene": plan.get("latest_coarse_scene"),
        "strategy_mode": "original",
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, plan
