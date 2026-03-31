#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spmo_config import CORE_PARAMS, DEFAULT_SCENE_CONFIG

def build_adjacent_deltas(
    df: pd.DataFrame,
    max_gap_minutes: int,
    context_tol: float,
) -> pd.DataFrame:
    work = df.sort_values("wet_time").reset_index(drop=True).copy()
    if "scene_fine_key" not in work.columns:
        raise ValueError("缺少 scene_fine_key，请先构建层级场景")

    scene_cols = [c for c in ["刮刀高度_上", "刮刀高度_下", "墨刀高度_上", "墨刀高度_下"] if c in work.columns]
    exclude = set(
        [
            "wet_time",
            "process_time",
            "wet_weight",
            "time_diff_min",
            "scene_fine_key",
            "scene_coarse_key",
            "life_segment",
        ]
        + CORE_PARAMS
        + scene_cols
    )
    context_cols = [c for c in work.columns if c not in exclude and pd.api.types.is_numeric_dtype(work[c])]

    records: List[Dict] = []

    for scene_key, g in work.groupby("scene_fine_key", sort=False):
        g = g.sort_values("wet_time").reset_index(drop=True)
        if len(g) < 2:
            continue

        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            cur = g.iloc[i]

            gap = (cur["wet_time"] - prev["wet_time"]).total_seconds() / 60.0
            if gap > max_gap_minutes:
                continue

            d = {
                "scene_key": scene_key,
                "scene_coarse_key": g.iloc[0].get("scene_coarse_key", "NA"),
                "life_segment": g.iloc[0].get("life_segment", "L1"),
                "t_prev": str(prev["wet_time"]),
                "t_cur": str(cur["wet_time"]),
                "gap_min": float(gap),
                "delta_wet": float(cur["wet_weight"] - prev["wet_weight"]),
            }

            changed_core = 0
            for p in CORE_PARAMS:
                dv = float(cur[p] - prev[p]) if pd.notna(cur[p]) and pd.notna(prev[p]) else 0.0
                d[f"delta_{p}"] = dv
                if abs(dv) > 1e-12:
                    changed_core += 1

            d["changed_core_count"] = changed_core

            ctx_changes = 0
            for c in context_cols:
                if pd.isna(cur[c]) or pd.isna(prev[c]):
                    continue
                if abs(float(cur[c] - prev[c])) > context_tol:
                    ctx_changes += 1
            d["context_change_count"] = ctx_changes

            records.append(d)

    return pd.DataFrame(records)


def analyze_single_param_effects(
    deltas: pd.DataFrame,
    max_context_changes: int,
    min_samples: int,
    core_stable_tol: float,
) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()

    clean = deltas[(deltas["context_change_count"] <= max_context_changes)].copy()
    rows: List[Dict] = []

    for scene_key, g in clean.groupby("scene_key", sort=False):
        scene_coarse_key = g["scene_coarse_key"].iloc[0] if "scene_coarse_key" in g.columns else "NA"
        life_segment = g["life_segment"].iloc[0] if "life_segment" in g.columns else "L1"

        for param in CORE_PARAMS:
            others = [p for p in CORE_PARAMS if p != param]
            cond = (
                (g[f"delta_{param}"].abs() > core_stable_tol)
                & (g[f"delta_{others[0]}"].abs() <= core_stable_tol)
                & (g[f"delta_{others[1]}"].abs() <= core_stable_tol)
            )
            s = g[cond].copy()
            n = len(s)
            if n < min_samples:
                continue

            ratio = s["delta_wet"] / s[f"delta_{param}"]
            mean_slope = float(ratio.mean())
            median_slope = float(ratio.median())

            inc = s[s[f"delta_{param}"] > 0]["delta_wet"]
            dec = s[s[f"delta_{param}"] < 0]["delta_wet"]

            if mean_slope > 0:
                direction = "参数增大 -> 湿重增大"
                reduce_advice = "降湿重建议：减小该参数"
            elif mean_slope < 0:
                direction = "参数增大 -> 湿重减小"
                reduce_advice = "降湿重建议：增大该参数"
            else:
                direction = "影响弱"
                reduce_advice = "优先调整其他参数"

            rows.append(
                {
                    "scene_key": scene_key,
                    "scene_coarse_key": scene_coarse_key,
                    "life_segment": life_segment,
                    "参数": param,
                    "single_change_samples": int(n),
                    "mean_delta_wet_per_unit": mean_slope,
                    "median_delta_wet_per_unit": median_slope,
                    "mean_delta_wet_when_param_increase": float(inc.mean()) if len(inc) else np.nan,
                    "mean_delta_wet_when_param_decrease": float(dec.mean()) if len(dec) else np.nan,
                    "typical_step_abs": float(s[f"delta_{param}"].abs().median()),
                    "方向结论": direction,
                    "降湿重建议": reduce_advice,
                }
            )

    return pd.DataFrame(rows)


def analyze_joint_param_effects(
    deltas: pd.DataFrame,
    max_context_changes: int,
    min_samples_joint: int,
    core_stable_tol: float,
) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()

    clean = deltas[(deltas["context_change_count"] <= max_context_changes)].copy()
    rows: List[Dict] = []

    p1, p2, p3 = CORE_PARAMS
    d1, d2, d3 = f"delta_{p1}", f"delta_{p2}", f"delta_{p3}"

    for scene_key, g in clean.groupby("scene_key", sort=False):
        scene_coarse_key = g["scene_coarse_key"].iloc[0] if "scene_coarse_key" in g.columns else "NA"
        life_segment = g["life_segment"].iloc[0] if "life_segment" in g.columns else "L1"

        gm = g[
            (g[d1].abs() > core_stable_tol)
            | (g[d2].abs() > core_stable_tol)
            | (g[d3].abs() > core_stable_tol)
        ].copy()

        n = len(gm)
        if n < min_samples_joint:
            continue

        x1 = gm[d1].astype(float).values
        x2 = gm[d2].astype(float).values
        x3 = gm[d3].astype(float).values
        y = gm["delta_wet"].astype(float).values

        X = np.column_stack(
            [
                np.ones(n),
                x1,
                x2,
                x3,
                x1 * x2,
                x1 * x3,
                x2 * x3,
                x1 * x2 * x3,
            ]
        )

        try:
            beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue

        if rank < 4:
            continue

        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

        rows.append(
            {
                "scene_key": scene_key,
                "scene_coarse_key": scene_coarse_key,
                "life_segment": life_segment,
                "joint_samples": int(n),
                "coef_intercept": float(beta[0]),
                "coef_印刷压力": float(beta[1]),
                "coef_印刷高度偏移": float(beta[2]),
                "coef_刮刀高度偏移": float(beta[3]),
                "coef_印刷压力x印刷高度偏移": float(beta[4]),
                "coef_印刷压力x刮刀高度偏移": float(beta[5]),
                "coef_印刷高度偏移x刮刀高度偏移": float(beta[6]),
                "coef_三参数交互": float(beta[7]),
                "model_r2": r2,
                "changed_core_eq1": int((gm["changed_core_count"] == 1).sum()),
                "changed_core_eq2": int((gm["changed_core_count"] == 2).sum()),
                "changed_core_eq3": int((gm["changed_core_count"] == 3).sum()),
            }
        )

    return pd.DataFrame(rows)


def latest_scene_adjustment(
    merged: pd.DataFrame,
    effect_df: pd.DataFrame,
    joint_effect_df: pd.DataFrame,
    deltas: pd.DataFrame,
    core_stable_tol: float,
    joint_weight_config: Optional[Dict] = None,
) -> Dict:
    latest_idx = merged["wet_time"].idxmax()
    latest = merged.loc[latest_idx]

    latest_scene = latest.get("scene_fine_key")
    latest_coarse_scene = latest.get("scene_coarse_key")

    scene_data = merged[merged.get("scene_fine_key") == latest_scene].sort_values("wet_time")
    if len(scene_data) == 0:
        scene_data = merged[merged.get("scene_coarse_key") == latest_coarse_scene].sort_values("wet_time")
    if len(scene_data) == 0:
        scene_data = merged.sort_values("wet_time")

    recent = scene_data.tail(min(80, len(scene_data)))
    current_wet = float(latest["wet_weight"])
    target_wet = float(recent["wet_weight"].quantile(0.35))
    need_reduce = max(0.0, current_wet - target_wet)
    joint_weight_config = joint_weight_config or DEFAULT_SCENE_CONFIG["joint_recommendation"]
    use_weighted_scale = bool(joint_weight_config.get("use_weighted_scale", False))
    joint_weights = joint_weight_config.get("weights", {})

    # ------- single-parameter recommendations (reference) -------
    recs = []
    if effect_df is not None and (not effect_df.empty) and ("scene_key" in effect_df.columns):
        e = effect_df[effect_df["scene_key"] == latest_scene].copy()
        if e.empty and latest_coarse_scene is not None:
            e = (
                effect_df[effect_df["scene_coarse_key"] == latest_coarse_scene]
                .sort_values("single_change_samples", ascending=False)
                .drop_duplicates(subset=["参数"], keep="first")
                .copy()
            )

        if e.empty:
            e = (
                effect_df.sort_values("single_change_samples", ascending=False)
                .drop_duplicates(subset=["参数"], keep="first")
                .copy()
            )

        for _, row in e.iterrows():
            p = row["参数"]
            slope = float(row["mean_delta_wet_per_unit"])
            step = float(row["typical_step_abs"])
            cur_v = float(latest[p])

            if abs(slope) < 1e-12 or step <= 0:
                action = "保持"
                delta = 0.0
                expected = 0.0
            else:
                action = "减小" if slope > 0 else "增大"
                one_step_reduce = abs(slope) * step
                scale = 1.0
                if one_step_reduce > 1e-12 and need_reduce > 0:
                    scale = min(2.0, max(1.0, need_reduce / one_step_reduce))
                delta = step * scale
                expected = one_step_reduce * scale

            new_v = cur_v + delta if action == "增大" else cur_v - delta if action == "减小" else cur_v

            recs.append(
                {
                    "参数": p,
                    "当前值": cur_v,
                    "建议方向": action,
                    "建议变化量": float(delta),
                    "建议新值": float(new_v),
                    "基于历史斜率预计降幅": float(expected),
                    "历史单位斜率": slope,
                    "样本数": int(row["single_change_samples"]),
                    "结论来源场景": row.get("scene_key", "fallback"),
                }
            )

    # ------- joint recommendation (primary) -------
    joint_recommendation = None
    if joint_effect_df is not None and (not joint_effect_df.empty) and ("scene_key" in joint_effect_df.columns):
        je = joint_effect_df[joint_effect_df["scene_key"] == latest_scene].copy()
        if je.empty and latest_coarse_scene is not None:
            je = (
                joint_effect_df[joint_effect_df["scene_coarse_key"] == latest_coarse_scene]
                .sort_values(["joint_samples", "model_r2"], ascending=False)
                .head(1)
                .copy()
            )
        if je.empty:
            je = (
                joint_effect_df.sort_values(["joint_samples", "model_r2"], ascending=False)
                .head(1)
                .copy()
            )

        if not je.empty:
            row = je.iloc[0]
            src_scene = row.get("scene_key", "fallback")

            dsrc = deltas[deltas.get("scene_key") == src_scene].copy() if "scene_key" in deltas.columns else pd.DataFrame()
            if dsrc.empty and latest_coarse_scene is not None and "scene_coarse_key" in deltas.columns:
                dsrc = deltas[deltas["scene_coarse_key"] == latest_coarse_scene].copy()
            if dsrc.empty:
                dsrc = deltas.copy()

            plan_items = []
            total_one_round_reduce = 0.0

            for p in CORE_PARAMS:
                coef = float(row.get(f"coef_{p}", 0.0))
                cur_v = float(latest[p])

                step_series = pd.to_numeric(dsrc.get(f"delta_{p}"), errors="coerce") if f"delta_{p}" in dsrc.columns else pd.Series(dtype=float)
                step = float(step_series.abs()[step_series.abs() > core_stable_tol].median()) if len(step_series) else np.nan

                if (not np.isfinite(step)) or step <= 0:
                    if effect_df is not None and (not effect_df.empty):
                        ref = effect_df[effect_df["参数"] == p].sort_values("single_change_samples", ascending=False)
                        if not ref.empty:
                            step = float(ref.iloc[0].get("typical_step_abs", 0.0))

                if (not np.isfinite(step)) or step <= 0:
                    step = 0.0

                if abs(coef) < 1e-12 or step <= 0:
                    action = "保持"
                    base_reduce = 0.0
                else:
                    action = "减小" if coef > 0 else "增大"
                    base_reduce = abs(coef) * step

                total_one_round_reduce += base_reduce
                plan_items.append(
                    {
                        "参数": p,
                        "当前值": cur_v,
                        "建议方向": action,
                        "基础步长": float(step),
                        "线性系数": coef,
                        "单步预计降幅(线性近似)": float(base_reduce),
                    }
                )

            scale = 1.0
            if total_one_round_reduce > 1e-12 and need_reduce > 0:
                scale = min(2.0, max(0.5, need_reduce / total_one_round_reduce))

            expected_total = 0.0
            for item in plan_items:
                weight = float(joint_weights.get(item["参数"], 1.0)) if use_weighted_scale else 1.0
                # Keep the existing scale_factor logic unchanged; only add a per-parameter weight on top.
                weighted_delta = item["基础步长"] * scale * weight if item["建议方向"] in ["增大", "减小"] else 0.0
                item["参数权重"] = float(weight)
                item["加权后建议变化量"] = float(weighted_delta)
                item["加权后建议新值"] = (
                    float(item["当前值"] + weighted_delta)
                    if item["建议方向"] == "增大"
                    else float(item["当前值"] - weighted_delta)
                    if item["建议方向"] == "减小"
                    else float(item["当前值"])
                )
                item["建议变化量"] = float(weighted_delta)
                item["建议新值"] = float(item["加权后建议新值"])
                item["预计降幅贡献(线性近似)"] = float(item["单步预计降幅(线性近似)"] * scale * weight)
                expected_total += item["预计降幅贡献(线性近似)"]

            joint_recommendation = {
                "结论来源场景": src_scene,
                "joint_samples": int(row.get("joint_samples", 0)),
                "joint_model_r2": float(row.get("model_r2", np.nan)),
                "required_reduction": need_reduce,
                "scale_factor": float(scale),
                "use_weighted_scale": use_weighted_scale,
                "joint_weights": {p: float(joint_weights.get(p, 1.0)) for p in CORE_PARAMS},
                "预计总降幅(线性近似)": float(expected_total),
                "组合调参": plan_items,
            }

    result = {
        "latest_time": str(latest["wet_time"]),
        "latest_scene": latest_scene,
        "latest_coarse_scene": latest_coarse_scene,
        "current_wet_weight": current_wet,
        "target_wet_weight": target_wet,
        "required_reduction": need_reduce,
        "joint_recommendation": joint_recommendation,
        "recommendations": recs,
    }

    if joint_recommendation is None and not recs:
        result["message"] = "差分样本总体不足，无法输出可靠调参建议。"

    return result


