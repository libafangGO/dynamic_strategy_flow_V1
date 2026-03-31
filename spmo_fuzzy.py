#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spmo_analysis import latest_scene_adjustment
from spmo_backtest import run_backtest
from spmo_config import CORE_PARAMS
from spmo_visualizations import save_latest_adjustment_visualizations


KEY_SOURCE_SCENE = "结论来源场景"
KEY_PLAN_ITEMS = "组合调参"
KEY_PARAM = "参数"
KEY_CURRENT = "当前值"
KEY_ACTION = "建议方向"
KEY_DELTA = "建议变化量"
KEY_NEW_VALUE = "建议新值"
KEY_EXPECTED_TOTAL = "预计总降幅(线性近似)"
KEY_EXPECTED_CONTRIB = "预计降幅贡献(线性近似)"
KEY_STABILITY_SCORE = "stability_score"

DEFAULT_FUZZY_CONFIG = {
    "recent_window": 12,
    "scene_recent_window": 12,
    "min_scene_samples": 6,
    "min_scale": 0.35,
    "max_scale": 1.45,
    "param_gain_min": 0.60,
    "param_gain_max": 1.35,
}


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(out):
        return float(default)
    return out


def _compute_fuzzy_membership(
    merged: pd.DataFrame,
    plan: Dict,
    fuzzy_config: Optional[Dict] = None,
) -> Dict:
    cfg = dict(DEFAULT_FUZZY_CONFIG)
    if isinstance(fuzzy_config, dict):
        cfg.update(fuzzy_config)

    work = merged.sort_values("wet_time").reset_index(drop=True).copy()
    current_wet = _safe_float(plan.get("current_wet_weight"))
    target_wet = _safe_float(plan.get("target_wet_weight"), current_wet)
    error = current_wet - target_wet

    if len(work) >= 2:
        error_rate = current_wet - _safe_float(work.iloc[-2]["wet_weight"], current_wet)
    else:
        error_rate = 0.0

    latest_scene = plan.get("latest_scene")
    scene_work = work[work.get("scene_fine_key") == latest_scene].copy() if "scene_fine_key" in work.columns else pd.DataFrame()
    if len(scene_work) < int(cfg["min_scene_samples"]):
        scene_work = work

    recent_window = int(cfg["recent_window"])
    recent_weights = pd.to_numeric(work["wet_weight"], errors="coerce").tail(recent_window).dropna()
    scene_recent_window = int(cfg["scene_recent_window"])
    scene_recent_weights = pd.to_numeric(scene_work["wet_weight"], errors="coerce").tail(scene_recent_window).dropna()
    recent_sigma = float(recent_weights.std(ddof=0)) if len(recent_weights) >= 2 else 0.0
    scene_sigma = float(scene_recent_weights.std(ddof=0)) if len(scene_recent_weights) >= 2 else recent_sigma
    sigma = scene_sigma if scene_sigma > 0 else recent_sigma

    joint_rec = plan.get("joint_recommendation") if isinstance(plan.get("joint_recommendation"), dict) else {}
    joint_r2 = _safe_float(joint_rec.get("joint_model_r2"), 0.0)
    joint_samples = _safe_float(joint_rec.get("joint_samples"), 0.0)
    base_intensity = _safe_float(joint_rec.get(KEY_EXPECTED_TOTAL), 0.0)
    if base_intensity <= 0.0:
        for item in plan.get("recommendations", []) or []:
            base_intensity += abs(_safe_float(item.get(KEY_DELTA), 0.0))

    sigma_ref = max(abs(target_wet) * 0.005, recent_sigma, 0.08)
    error_ref = max(abs(target_wet) * 0.01, sigma_ref, 0.10)
    intensity_ref = max(error_ref, 0.15)

    large_error = _clip(abs(error) / (2.0 * error_ref), 0.0, 1.0)
    recovering = _clip(max(0.0, -error_rate) / error_ref, 0.0, 1.0)
    upward_trend = _clip(max(0.0, error_rate) / error_ref, 0.0, 1.0)
    high_vol = _clip(sigma / max(sigma_ref, 1e-6), 0.0, 1.0)
    scene_risk = _clip((1.0 - _clip(joint_r2, 0.0, 1.0)) * 0.55 + max(0.0, 20.0 - joint_samples) / 20.0 * 0.45, 0.0, 1.0)
    high_intensity = _clip(base_intensity / intensity_ref, 0.0, 1.0)

    return {
        "current_wet_weight": current_wet,
        "target_wet_weight": target_wet,
        "error": float(error),
        "error_rate": float(error_rate),
        "recent_sigma": float(sigma),
        "scene_risk": float(scene_risk),
        "base_intensity": float(base_intensity),
        "joint_model_r2": float(joint_r2),
        "joint_samples": int(joint_samples),
        "error_ref": float(error_ref),
        "sigma_ref": float(sigma_ref),
        "large_error": float(large_error),
        "recovering": float(recovering),
        "upward_trend": float(upward_trend),
        "high_vol": float(high_vol),
        "high_intensity": float(high_intensity),
    }


def _build_fuzzy_outputs(plan: Dict, membership: Dict, fuzzy_config: Optional[Dict] = None) -> Dict:
    cfg = dict(DEFAULT_FUZZY_CONFIG)
    if isinstance(fuzzy_config, dict):
        cfg.update(fuzzy_config)

    large_error = membership["large_error"]
    upward_trend = membership["upward_trend"]
    high_vol = membership["high_vol"]
    recovering = membership["recovering"]
    scene_risk = membership["scene_risk"]
    high_intensity = membership["high_intensity"]

    overall_scale = (
        1.0
        + 0.30 * large_error * upward_trend
        + 0.12 * large_error * (1.0 - high_vol)
        - 0.26 * high_vol
        - 0.18 * recovering
        - 0.12 * scene_risk
        - 0.08 * high_intensity * high_vol
    )
    overall_scale = _clip(overall_scale, cfg["min_scale"], cfg["max_scale"])
    stability_bias = _clip(0.55 * high_vol + 0.25 * recovering + 0.20 * scene_risk, 0.0, 1.0)

    per_param_gain: Dict[str, float] = {param: 1.0 for param in CORE_PARAMS}
    rule_rows: List[Dict] = []

    joint_rec = plan.get("joint_recommendation") if isinstance(plan.get("joint_recommendation"), dict) else {}
    plan_items = joint_rec.get(KEY_PLAN_ITEMS, []) if isinstance(joint_rec, dict) else []
    total_contrib = sum(max(0.0, _safe_float(item.get(KEY_EXPECTED_CONTRIB), 0.0)) for item in plan_items)

    for item in plan_items:
        param = str(item.get(KEY_PARAM, ""))
        base_contrib = max(0.0, _safe_float(item.get(KEY_EXPECTED_CONTRIB), 0.0))
        dominance = base_contrib / total_contrib if total_contrib > 1e-12 else 0.0
        stability_score = _clip(_safe_float(item.get(KEY_STABILITY_SCORE), 1.0), 0.0, 1.0)
        gain = 1.0
        gain += 0.22 * large_error * upward_trend * dominance
        gain += 0.08 * large_error * (1.0 - stability_bias)
        gain -= 0.18 * high_vol * (1.0 - stability_score)
        gain -= 0.10 * scene_risk * (1.0 - dominance)
        gain = _clip(gain, cfg["param_gain_min"], cfg["param_gain_max"])
        per_param_gain[param] = float(gain)
        rule_rows.append(
            {
                "param": param,
                "dominance": float(dominance),
                "stability_score": float(stability_score),
                "per_param_gain": float(gain),
                "overall_adjust_scale": float(overall_scale),
                "stability_bias": float(stability_bias),
                "large_error": float(large_error),
                "upward_trend": float(upward_trend),
                "recovering": float(recovering),
                "high_vol": float(high_vol),
                "scene_risk": float(scene_risk),
            }
        )

    rules = []
    if large_error > 0.55 and upward_trend > 0.25:
        rules.append("误差偏大且湿重仍在上升，适度放大主导参数动作。")
    if high_vol > 0.45:
        rules.append("近期波动偏高，整体降权并增加稳态偏置。")
    if recovering > 0.30 and high_vol > 0.35:
        rules.append("误差正在收敛但波动仍高，优先稳住，减小步长。")
    if scene_risk > 0.45:
        rules.append("场景模型风险较高，进一步压缩总调节强度。")
    if not rules:
        rules.append("当前场景相对平稳，按原始建议进行轻度二次修正。")

    return {
        "overall_adjust_scale": float(overall_scale),
        "stability_bias": float(stability_bias),
        "per_param_gain": per_param_gain,
        "rules": rules,
        "rule_rows": rule_rows,
    }


def apply_fuzzy_control_to_plan(
    plan: Dict,
    merged: pd.DataFrame,
    fuzzy_config: Optional[Dict] = None,
) -> Dict:
    fuzzy_plan = copy.deepcopy(plan)
    membership = _compute_fuzzy_membership(merged=merged, plan=plan, fuzzy_config=fuzzy_config)
    fuzzy_output = _build_fuzzy_outputs(plan=plan, membership=membership, fuzzy_config=fuzzy_config)

    joint_rec = fuzzy_plan.get("joint_recommendation")
    if isinstance(joint_rec, dict):
        expected_total = 0.0
        for item in joint_rec.get(KEY_PLAN_ITEMS, []) or []:
            param = str(item.get(KEY_PARAM, ""))
            action = str(item.get(KEY_ACTION, "保持"))
            original_delta = _safe_float(item.get(KEY_DELTA), 0.0)
            original_contrib = _safe_float(item.get(KEY_EXPECTED_CONTRIB), 0.0)
            param_gain = fuzzy_output["per_param_gain"].get(param, 1.0)
            ratio = fuzzy_output["overall_adjust_scale"] * param_gain if action in ("增大", "减小") else 0.0
            fuzzy_delta = original_delta * ratio
            current_value = _safe_float(item.get(KEY_CURRENT), 0.0)
            if action == "增大":
                fuzzy_new_value = current_value + fuzzy_delta
            elif action == "减小":
                fuzzy_new_value = current_value - fuzzy_delta
            else:
                fuzzy_new_value = current_value
                fuzzy_delta = 0.0
                ratio = 0.0
            fuzzy_contrib = original_contrib * ratio
            item["原始建议变化量"] = float(original_delta)
            item["原始预计降幅贡献(线性近似)"] = float(original_contrib)
            item["fuzzy_scale_factor"] = float(fuzzy_output["overall_adjust_scale"])
            item["fuzzy_param_gain"] = float(param_gain)
            item[KEY_DELTA] = float(fuzzy_delta)
            item[KEY_NEW_VALUE] = float(fuzzy_new_value)
            item[KEY_EXPECTED_CONTRIB] = float(fuzzy_contrib)
            expected_total += fuzzy_contrib

        joint_rec["original_expected_total_reduction"] = _safe_float(plan.get("joint_recommendation", {}).get(KEY_EXPECTED_TOTAL), 0.0)
        joint_rec[KEY_EXPECTED_TOTAL] = float(expected_total)
        joint_rec["fuzzy_control_enabled"] = True
        joint_rec["fuzzy_overall_adjust_scale"] = float(fuzzy_output["overall_adjust_scale"])
        joint_rec["fuzzy_stability_bias"] = float(fuzzy_output["stability_bias"])

    if fuzzy_plan.get("recommendations"):
        for item in fuzzy_plan.get("recommendations", []):
            param = str(item.get(KEY_PARAM, ""))
            action = str(item.get(KEY_ACTION, "保持"))
            original_delta = _safe_float(item.get(KEY_DELTA), 0.0)
            param_gain = fuzzy_output["per_param_gain"].get(param, 1.0)
            ratio = fuzzy_output["overall_adjust_scale"] * param_gain if action in ("增大", "减小") else 0.0
            fuzzy_delta = original_delta * ratio
            current_value = _safe_float(item.get(KEY_CURRENT), 0.0)
            expected = _safe_float(item.get("基于历史斜率预计降幅"), 0.0) * ratio
            if action == "增大":
                fuzzy_new_value = current_value + fuzzy_delta
            elif action == "减小":
                fuzzy_new_value = current_value - fuzzy_delta
            else:
                fuzzy_new_value = current_value
                fuzzy_delta = 0.0
                expected = 0.0
            item["原始建议变化量"] = float(original_delta)
            item["fuzzy_scale_factor"] = float(fuzzy_output["overall_adjust_scale"])
            item["fuzzy_param_gain"] = float(param_gain)
            item[KEY_DELTA] = float(fuzzy_delta)
            item[KEY_NEW_VALUE] = float(fuzzy_new_value)
            item["基于历史斜率预计降幅"] = float(expected)

    fuzzy_plan["fuzzy_control"] = {
        "inputs": membership,
        "outputs": {
            "overall_adjust_scale": float(fuzzy_output["overall_adjust_scale"]),
            "stability_bias": float(fuzzy_output["stability_bias"]),
            "per_param_gain": fuzzy_output["per_param_gain"],
        },
        "rules": fuzzy_output["rules"],
    }
    fuzzy_plan["fuzzy_rule_trace_rows"] = fuzzy_output["rule_rows"]
    return fuzzy_plan


def _fuzzy_planner(
    merged: pd.DataFrame,
    effect_df: pd.DataFrame,
    joint_effect_df: pd.DataFrame,
    deltas: pd.DataFrame,
    core_stable_tol: float,
    joint_weight_config: Optional[Dict] = None,
    target_wet_config: Optional[Dict] = None,
    fuzzy_config: Optional[Dict] = None,
) -> Dict:
    base_plan = latest_scene_adjustment(
        merged,
        effect_df,
        joint_effect_df,
        deltas,
        core_stable_tol=core_stable_tol,
        joint_weight_config=joint_weight_config,
        target_wet_config=target_wet_config,
    )
    return apply_fuzzy_control_to_plan(base_plan, merged=merged, fuzzy_config=fuzzy_config)


def run_fuzzy_strategy(
    context: Dict,
    output_dir: Path,
    core_stable_tol: float,
    backtest_ratio: float,
    backtest_noise_scale: float,
    backtest_noise_seed: int,
    max_pair_gap_minutes: int,
    max_context_changes: int,
    min_samples: int,
    min_samples_joint: int,
    context_tol: float,
    joint_weight_config: Optional[Dict] = None,
    target_wet_config: Optional[Dict] = None,
    fuzzy_config: Optional[Dict] = None,
) -> Dict:
    merged = context["merged"]
    effects = context["effects"]
    joint_effects = context["joint_effects"]
    deltas = context["deltas"]
    output_dir.mkdir(parents=True, exist_ok=True)
    base_plan = latest_scene_adjustment(
        merged,
        effects,
        joint_effects,
        deltas,
        core_stable_tol=core_stable_tol,
        joint_weight_config=joint_weight_config,
        target_wet_config=target_wet_config,
    )
    fuzzy_plan = apply_fuzzy_control_to_plan(base_plan, merged=merged, fuzzy_config=fuzzy_config)

    rule_trace_df = pd.DataFrame(fuzzy_plan.get("fuzzy_rule_trace_rows", []))
    if not rule_trace_df.empty:
        rule_trace_df.to_csv(output_dir / "fuzzy_rule_trace.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["param", "per_param_gain"]).to_csv(output_dir / "fuzzy_rule_trace.csv", index=False, encoding="utf-8-sig")

    vis_paths = save_latest_adjustment_visualizations(fuzzy_plan, output_dir)
    renamed_vis_paths: Dict[str, str] = {}
    for _, raw_path in vis_paths.items():
        src = Path(raw_path)
        if not src.exists():
            continue
        if src.name == "latest_adjustment_dashboard.png":
            dst = src.with_name("fuzzy_dashboard.png")
        elif src.name == "latest_adjustment_contribution_waterfall.png":
            dst = src.with_name("fuzzy_contribution_waterfall.png")
        else:
            dst = src.with_name(f"fuzzy_{src.name}")
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        renamed_vis_paths[dst.stem] = str(dst)

    with open(output_dir / "fuzzy_latest_adjustment_plan.json", "w", encoding="utf-8") as f:
        json.dump(fuzzy_plan, f, ensure_ascii=False, indent=2)

    fuzzy_backtest_summary = run_backtest(
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
        target_wet_config=target_wet_config,
        planner_fn=lambda hist, effect_df, joint_df, hist_deltas, **kwargs: _fuzzy_planner(
            hist,
            effect_df,
            joint_df,
            hist_deltas,
            fuzzy_config=fuzzy_config,
            **kwargs,
        ),
        output_prefix="fuzzy_",
        details_filename="fuzzy_backtest_details.csv",
        summary_filename="fuzzy_backtest_summary.json",
    )

    summary = {
        "enabled": True,
        "fuzzy_output_dir": str(output_dir),
        "latest_time": fuzzy_plan.get("latest_time"),
        "latest_scene": fuzzy_plan.get("latest_scene"),
        "current_wet_weight": fuzzy_plan.get("current_wet_weight"),
        "target_wet_weight": fuzzy_plan.get("target_wet_weight"),
        "required_reduction": fuzzy_plan.get("required_reduction"),
        "fuzzy_control": fuzzy_plan.get("fuzzy_control", {}),
        "rule_trace_csv": str(output_dir / "fuzzy_rule_trace.csv"),
        "latest_adjustment_plan_json": str(output_dir / "fuzzy_latest_adjustment_plan.json"),
        "visualizations": renamed_vis_paths,
        "backtest": fuzzy_backtest_summary,
    }

    with open(output_dir / "fuzzy_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
