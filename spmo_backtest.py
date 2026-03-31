#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spmo_analysis import (
    analyze_joint_param_effects,
    analyze_single_param_effects,
    build_adjacent_deltas,
    latest_scene_adjustment,
)
from spmo_visualizations import _first_present, save_backtest_visualizations


def run_backtest(
    merged: pd.DataFrame,
    output_dir: Path,
    test_ratio: float,
    max_pair_gap_minutes: int,
    max_context_changes: int,
    min_samples: int,
    min_samples_joint: int,
    context_tol: float,
    core_stable_tol: float,
    noise_scale: float,
    noise_seed: int,
    joint_weight_config: Optional[Dict] = None,
) -> Dict:
    work = merged.sort_values("wet_time").reset_index(drop=True).copy()
    n_total = len(work)
    if n_total < 30:
        return {
            "enabled": True,
            "message": "样本量不足，未执行回测",
            "test_ratio": test_ratio,
            "total_samples": n_total,
        }

    test_size = max(1, int(np.ceil(n_total * test_ratio)))
    test_start = max(1, n_total - test_size)
    records: List[Dict] = []
    rng = np.random.default_rng(noise_seed)

    for i in range(test_start, n_total - 1):
        hist = work.iloc[: i + 1].copy()
        future = work.iloc[i + 1]

        deltas = build_adjacent_deltas(
            hist,
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
        plan = latest_scene_adjustment(
            hist,
            effects,
            joint_effects,
            deltas,
            core_stable_tol=core_stable_tol,
            joint_weight_config=joint_weight_config,
        )

        current = hist.iloc[-1]
        current_wet = float(current["wet_weight"])
        next_wet = float(future["wet_weight"])
        actual_next_reduction = current_wet - next_wet

        scene_hist = (
            hist[hist.get("scene_fine_key") == current.get("scene_fine_key")]
            if "scene_fine_key" in hist.columns
            else pd.DataFrame()
        )
        if len(scene_hist) >= 8:
            noise_source = pd.to_numeric(scene_hist["wet_weight"].diff(), errors="coerce").dropna()
            noise_scope = "scene"
        else:
            noise_source = pd.to_numeric(hist["wet_weight"].diff(), errors="coerce").dropna()
            noise_scope = "global"
        noise_mean = float(noise_source.mean()) if len(noise_source) else 0.0
        noise_std = float(noise_source.std(ddof=0)) if len(noise_source) else 0.0
        sampled_noise = float(rng.normal(noise_mean, noise_std * max(noise_scale, 0.0))) if noise_std > 0 else noise_mean

        joint_rec = plan.get("joint_recommendation")
        if isinstance(joint_rec, dict):
            expected_reduction = float(
                _first_present(
                    joint_rec,
                    ["预计总降幅(线性近似)", "棰勮鎬婚檷骞?绾挎€ц繎浼?"],
                    default=0.0,
                )
            )
            strategy_type = "joint"
        else:
            recs = plan.get("recommendations", [])
            expected_reduction = 0.0
            for item in recs:
                expected_reduction += float(
                    _first_present(
                        item,
                        ["基于历史斜率预计降幅", "鍩轰簬鍘嗗彶鏂滅巼棰勮闄嶅箙"],
                        default=0.0,
                    )
                )
            strategy_type = "single" if recs else "none"

        noisy_expected_reduction = expected_reduction + sampled_noise
        predicted_next_wet = current_wet - expected_reduction
        noisy_predicted_next_wet = current_wet - noisy_expected_reduction

        records.append(
            {
                "current_time": str(current["wet_time"]),
                "next_time": str(future["wet_time"]),
                "current_scene": current.get("scene_fine_key", ""),
                "current_wet_weight": current_wet,
                "next_wet_weight": next_wet,
                "target_wet_weight": float(plan.get("target_wet_weight", np.nan)),
                "required_reduction": float(plan.get("required_reduction", 0.0)),
                "expected_reduction": expected_reduction,
                "noise_scope": noise_scope,
                "noise_mean": noise_mean,
                "noise_std": noise_std,
                "sampled_noise": sampled_noise,
                "noisy_expected_reduction": noisy_expected_reduction,
                "predicted_next_wet_weight": predicted_next_wet,
                "noisy_predicted_next_wet_weight": noisy_predicted_next_wet,
                "actual_next_reduction": actual_next_reduction,
                "strategy_type": strategy_type,
                "has_recommendation": int(strategy_type != "none"),
                "hit_reduce_direction": int(actual_next_reduction > 0),
                "hit_reduce_direction_noisy": int(noisy_expected_reduction > 0 and actual_next_reduction > 0),
                "signed_error": actual_next_reduction - expected_reduction,
                "abs_error": abs(actual_next_reduction - expected_reduction),
                "noisy_signed_error": actual_next_reduction - noisy_expected_reduction,
                "noisy_abs_error": abs(actual_next_reduction - noisy_expected_reduction),
            }
        )

    backtest_df = pd.DataFrame(records)
    backtest_csv = output_dir / "backtest_last_10pct_details.csv"
    backtest_df.to_csv(backtest_csv, index=False, encoding="utf-8-sig")
    backtest_vis_paths = save_backtest_visualizations(backtest_df, output_dir)

    actionable = backtest_df[backtest_df["has_recommendation"] == 1].copy() if not backtest_df.empty else pd.DataFrame()
    wet_metrics = {
        "当前湿重": "current_wet_weight",
        "带扰动预计下一步湿重": "noisy_predicted_next_wet_weight",
        "预计下一步湿重": "predicted_next_wet_weight",
        "目标湿重": "target_wet_weight",
    }
    wet_value_summary = {}
    for label, col in wet_metrics.items():
        if len(actionable):
            series = pd.to_numeric(actionable[col], errors="coerce")
            wet_value_summary[label] = {
                "mean": float(series.mean()),
                "var": float(series.var(ddof=0)),
                "sum": float(series.sum()),
            }
        else:
            wet_value_summary[label] = {
                "mean": np.nan,
                "var": np.nan,
                "sum": np.nan,
            }

    summary = {
        "enabled": True,
        "test_ratio": test_ratio,
        "total_samples": int(n_total),
        "test_samples": int(test_size),
        "evaluated_steps": int(len(backtest_df)),
        "actionable_steps": int(len(actionable)),
        "recommendation_coverage": float(len(actionable) / len(backtest_df)) if len(backtest_df) else 0.0,
        "direction_hit_rate": float(actionable["hit_reduce_direction"].mean()) if len(actionable) else np.nan,
        "direction_hit_rate_noisy": float(actionable["hit_reduce_direction_noisy"].mean()) if len(actionable) else np.nan,
        "avg_required_reduction": float(actionable["required_reduction"].mean()) if len(actionable) else np.nan,
        "avg_expected_reduction": float(actionable["expected_reduction"].mean()) if len(actionable) else np.nan,
        "avg_noisy_expected_reduction": float(actionable["noisy_expected_reduction"].mean()) if len(actionable) else np.nan,
        "avg_actual_next_reduction": float(actionable["actual_next_reduction"].mean()) if len(actionable) else np.nan,
        "avg_predicted_next_wet_weight": float(actionable["predicted_next_wet_weight"].mean()) if len(actionable) else np.nan,
        "avg_noisy_predicted_next_wet_weight": float(actionable["noisy_predicted_next_wet_weight"].mean()) if len(actionable) else np.nan,
        "avg_actual_next_wet_weight": float(actionable["next_wet_weight"].mean()) if len(actionable) else np.nan,
        "mean_abs_error": float(actionable["abs_error"].mean()) if len(actionable) else np.nan,
        "mean_noisy_abs_error": float(actionable["noisy_abs_error"].mean()) if len(actionable) else np.nan,
        "wet_value_summary": wet_value_summary,
        "noise_scale": noise_scale,
        "noise_seed": noise_seed,
        "details_csv": str(backtest_csv),
        "visualizations": backtest_vis_paths,
    }

    with open(output_dir / "backtest_last_10pct_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
