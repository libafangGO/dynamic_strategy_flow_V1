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


def _clip_float(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _default_weight_detail(weight_source: str) -> Dict:
    return {
        "参数权重": 1.0,
        "权重来源": weight_source,
        "effect_score": 0.0,
        "confidence_score": 1.0 if weight_source == "disabled" else 0.0,
        "hit_score": 1.0 if weight_source == "disabled" else 0.75,
        "room_score": 1.0,
        "stability_score": 1.0,
        "sample_score": 0.0,
        "r2_score": 0.0,
        "direction_consistency": 0.5,
        "recent_hit_rate": 0.5,
        "available_room": 0.0,
        "vol_risk": 0.0,
        "effective_sample_count": 0,
        "raw_weight": 0.0,
        "norm_weight": 1.0,
    }


def _build_weight_scope(
    latest: pd.Series,
    merged: pd.DataFrame,
    deltas: pd.DataFrame,
    min_scene_samples: int,
) -> Dict:
    latest_scene = latest.get("scene_fine_key")
    latest_coarse_scene = latest.get("scene_coarse_key")

    fine_scene_data = merged[merged.get("scene_fine_key") == latest_scene].copy() if "scene_fine_key" in merged.columns else pd.DataFrame()
    fine_deltas = deltas[deltas.get("scene_key") == latest_scene].copy() if "scene_key" in deltas.columns else pd.DataFrame()
    if len(fine_scene_data) >= max(2, min_scene_samples) and len(fine_deltas) >= min_scene_samples:
        return {
            "weight_source": "scene_fine_key",
            "scene_data": fine_scene_data.sort_values("wet_time"),
            "deltas": fine_deltas,
        }

    coarse_scene_data = merged[merged.get("scene_coarse_key") == latest_coarse_scene].copy() if "scene_coarse_key" in merged.columns else pd.DataFrame()
    coarse_deltas = deltas[deltas.get("scene_coarse_key") == latest_coarse_scene].copy() if "scene_coarse_key" in deltas.columns else pd.DataFrame()
    if len(coarse_scene_data) >= max(2, min_scene_samples) and len(coarse_deltas) >= min_scene_samples:
        return {
            "weight_source": "scene_coarse_key",
            "scene_data": coarse_scene_data.sort_values("wet_time"),
            "deltas": coarse_deltas,
        }

    return {
        "weight_source": "global_default",
        "scene_data": pd.DataFrame(),
        "deltas": pd.DataFrame(),
    }


def build_dynamic_joint_weights(
    latest: pd.Series,
    merged: pd.DataFrame,
    deltas: pd.DataFrame,
    joint_row: pd.Series,
    plan_items: List[Dict],
    core_stable_tol: float,
    joint_weight_config: Optional[Dict] = None,
) -> Dict:
    joint_cfg = joint_weight_config or DEFAULT_SCENE_CONFIG["joint_recommendation"]
    dynamic_cfg = joint_cfg.get("dynamic_weight", {})
    k = max(float(dynamic_cfg.get("k", 8.0)), 1e-6)
    risk_lambda = max(float(dynamic_cfg.get("lambda", 0.8)), 0.0)
    recent_window = max(int(dynamic_cfg.get("recent_window", 20)), 3)
    min_scene_samples = max(int(dynamic_cfg.get("min_scene_samples", 8)), 3)
    eps = max(float(dynamic_cfg.get("eps", 1e-6)), 1e-12)

    scope_info = _build_weight_scope(
        latest=latest,
        merged=merged,
        deltas=deltas,
        min_scene_samples=min_scene_samples,
    )
    weight_source = scope_info["weight_source"]
    scope_scene = scope_info["scene_data"]
    scope_deltas = scope_info["deltas"]

    if weight_source == "global_default":
        return {
            "weight_source": "global_default",
            "weights": {param: _default_weight_detail("global_default") for param in CORE_PARAMS},
        }

    scope_deltas = scope_deltas.copy()
    if "t_cur" in scope_deltas.columns:
        scope_deltas["_t_cur_order"] = pd.to_datetime(scope_deltas["t_cur"], errors="coerce")
        scope_deltas = scope_deltas.sort_values("_t_cur_order")

    r2_score = float(joint_row.get("model_r2", np.nan))
    r2_score = _clip_float(r2_score, 0.0, 1.0) if np.isfinite(r2_score) else 0.0

    raw_weight_map: Dict[str, float] = {}
    detail_map: Dict[str, Dict] = {}

    for item in plan_items:
        param = item["参数"]
        coef = float(joint_row.get(f"coef_{param}", item.get("线性系数", 0.0)))
        step = float(item.get("基础步长", 0.0))
        action = item.get("建议方向", "保持")
        cur_value = float(item.get("当前值", latest.get(param, np.nan)))
        detail = _default_weight_detail(weight_source)

        effect_score = abs(coef) * step if np.isfinite(coef) and np.isfinite(step) else 0.0
        detail["effect_score"] = float(effect_score)

        if action not in ["增大", "减小"] or step <= 0 or abs(coef) <= eps or f"delta_{param}" not in scope_deltas.columns:
            raw_weight_map[param] = 0.0
            detail["权重来源"] = weight_source
            detail_map[param] = detail
            continue

        relevant = scope_deltas[pd.to_numeric(scope_deltas[f"delta_{param}"], errors="coerce").abs() > core_stable_tol].copy()
        n_p = int(len(relevant))
        sample_score = float(np.sqrt(n_p / (n_p + k))) if n_p > 0 else 0.0

        if n_p > 0:
            relevant["local_slope"] = pd.to_numeric(relevant["delta_wet"], errors="coerce") / (
                pd.to_numeric(relevant[f"delta_{param}"], errors="coerce") + eps
            )
            local_slopes = pd.to_numeric(relevant["local_slope"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        else:
            local_slopes = pd.Series(dtype=float)

        if len(local_slopes):
            slope_signs = np.sign(local_slopes.astype(float).values)
            direction_consistency = float(np.mean(slope_signs == np.sign(coef)))
        else:
            direction_consistency = 0.5

        confidence_score = float(sample_score * r2_score * direction_consistency)

        recent = relevant.tail(recent_window).copy()
        if len(recent):
            predicted_sign = np.sign(coef * pd.to_numeric(recent[f"delta_{param}"], errors="coerce").astype(float).values)
            actual_sign = np.sign(pd.to_numeric(recent["delta_wet"], errors="coerce").astype(float).values)
            valid = predicted_sign != 0
            recent_hit_rate = float(np.mean(predicted_sign[valid] == actual_sign[valid])) if np.any(valid) else 0.5
        else:
            recent_hit_rate = 0.5
        hit_score = float(0.5 + 0.5 * recent_hit_rate)

        hist_series = pd.to_numeric(scope_scene.get(param), errors="coerce").dropna() if param in scope_scene.columns else pd.Series(dtype=float)
        if len(hist_series):
            hist_min = float(hist_series.min())
            hist_max = float(hist_series.max())
            if action == "增大":
                available_room = max(0.0, hist_max - cur_value)
            else:
                available_room = max(0.0, cur_value - hist_min)
        else:
            available_room = 0.0
        room_score = _clip_float(available_room / (step + eps), 0.3, 1.2)

        recent_slopes = local_slopes.tail(recent_window)
        if len(recent_slopes) >= 2:
            slope_mean = abs(float(recent_slopes.mean()))
            vol_risk = float(recent_slopes.std(ddof=0) / (slope_mean + eps))
        else:
            vol_risk = 0.0
        stability_score = float(1.0 / (1.0 + risk_lambda * vol_risk))

        raw_weight = float(effect_score * confidence_score * hit_score * room_score * stability_score)

        detail.update(
            {
                "参数权重": 1.0,
                "权重来源": weight_source,
                "confidence_score": confidence_score,
                "hit_score": hit_score,
                "room_score": room_score,
                "stability_score": stability_score,
                "sample_score": sample_score,
                "r2_score": r2_score,
                "direction_consistency": direction_consistency,
                "recent_hit_rate": recent_hit_rate,
                "available_room": float(available_room),
                "vol_risk": vol_risk,
                "effective_sample_count": n_p,
                "raw_weight": raw_weight,
            }
        )
        raw_weight_map[param] = raw_weight
        detail_map[param] = detail

    mean_raw = float(np.mean([raw_weight_map.get(param, 0.0) for param in CORE_PARAMS])) if CORE_PARAMS else 0.0
    if mean_raw <= eps:
        return {
            "weight_source": "global_default",
            "weights": {param: _default_weight_detail("global_default") for param in CORE_PARAMS},
        }

    for param in CORE_PARAMS:
        raw_weight = float(raw_weight_map.get(param, 0.0))
        norm_weight = raw_weight / mean_raw
        final_weight = _clip_float(norm_weight, 0.6, 1.4)
        detail_map.setdefault(param, _default_weight_detail(weight_source))
        detail_map[param]["norm_weight"] = float(norm_weight)
        detail_map[param]["参数权重"] = float(final_weight)

    return {
        "weight_source": weight_source,
        "weights": detail_map,
    }


def compute_dual_target_wet(
    scene_data: pd.DataFrame,
    current_wet: float,
    target_wet_config: Optional[Dict] = None,
) -> Dict:
    cfg = target_wet_config or DEFAULT_SCENE_CONFIG["target_wet_config"]
    recent_window = max(int(cfg.get("recent_window", 80)), 1)
    stable_quantile = _clip_float(float(cfg.get("stable_quantile", 0.35)), 0.0, 1.0)
    trend_bottom_ratio = _clip_float(float(cfg.get("trend_bottom_ratio", 0.20)), 0.01, 0.50)
    trend_push_down = max(float(cfg.get("trend_push_down", 0.02)), 0.0)
    vol_low = max(float(cfg.get("vol_low", 0.15)), 0.0)
    vol_high = max(float(cfg.get("vol_high", 0.50)), vol_low + 1e-9)
    alpha_low = _clip_float(float(cfg.get("alpha_low", 0.30)), 0.0, 1.0)
    alpha_high = _clip_float(float(cfg.get("alpha_high", 0.70)), alpha_low, 1.0)
    min_recent_samples = max(int(cfg.get("min_recent_samples", 12)), 1)
    floor_q = _clip_float(float(cfg.get("target_floor_quantile", 0.05)), 0.0, 1.0)
    ceil_q = _clip_float(float(cfg.get("target_ceiling_quantile", 0.50)), floor_q, 1.0)
    enabled_dual_target = bool(cfg.get("enabled_dual_target", True))

    wet_series = pd.to_numeric(scene_data.get("wet_weight"), errors="coerce").dropna()
    recent_wet = wet_series.tail(min(recent_window, len(wet_series))) if len(wet_series) else pd.Series(dtype=float)
    recent_count = int(len(recent_wet))

    if recent_count == 0:
        return {
            "stable_target": float(current_wet),
            "trend_target": float(current_wet),
            "final_target": float(current_wet),
            "recent_vol": 0.0,
            "alpha": 1.0,
            "recent_count": 0,
            "target_method": "fallback_current",
        }

    stable_target = float(recent_wet.quantile(stable_quantile))
    lower_bound = float(recent_wet.quantile(floor_q))
    upper_bound = float(recent_wet.quantile(ceil_q))

    if (not enabled_dual_target) or recent_count < min_recent_samples:
        final_target = _clip_float(stable_target, lower_bound, upper_bound)
        return {
            "stable_target": stable_target,
            "trend_target": stable_target,
            "final_target": float(final_target),
            "recent_vol": float(recent_wet.std(ddof=0)) if recent_count >= 2 else 0.0,
            "alpha": 1.0,
            "recent_count": recent_count,
            "target_method": "fallback_q35",
        }

    bottom_n = max(1, int(np.ceil(recent_count * trend_bottom_ratio)))
    bottom_wet = recent_wet.nsmallest(bottom_n)
    trend_target = float(bottom_wet.mean() - trend_push_down) if len(bottom_wet) else stable_target
    recent_vol = float(recent_wet.std(ddof=0)) if recent_count >= 2 else 0.0

    if recent_vol >= vol_high:
        alpha = alpha_high
    elif recent_vol <= vol_low:
        alpha = alpha_low
    else:
        interp = (recent_vol - vol_low) / (vol_high - vol_low)
        alpha = float(alpha_low + interp * (alpha_high - alpha_low))

    final_target = float(alpha * stable_target + (1.0 - alpha) * trend_target)
    final_target = _clip_float(final_target, lower_bound, upper_bound)

    return {
        "stable_target": stable_target,
        "trend_target": trend_target,
        "final_target": float(final_target),
        "recent_vol": recent_vol,
        "alpha": float(alpha),
        "recent_count": recent_count,
        "target_method": "dual_target",
    }


def latest_scene_adjustment(
    merged: pd.DataFrame,
    effect_df: pd.DataFrame,
    joint_effect_df: pd.DataFrame,
    deltas: pd.DataFrame,
    core_stable_tol: float,
    joint_weight_config: Optional[Dict] = None,
    target_wet_config: Optional[Dict] = None,
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

    current_wet = float(latest["wet_weight"])
    target_info = compute_dual_target_wet(
        scene_data=scene_data,
        current_wet=current_wet,
        target_wet_config=target_wet_config,
    )
    target_wet = float(target_info["final_target"])
    need_reduce = max(0.0, current_wet - target_wet)
    joint_weight_config = joint_weight_config or DEFAULT_SCENE_CONFIG["joint_recommendation"]
    use_weighted_scale = bool(joint_weight_config.get("use_weighted_scale", False))
    configured_joint_weights = joint_weight_config.get("weights", {})

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

            if use_weighted_scale:
                dynamic_weight_result = build_dynamic_joint_weights(
                    latest=latest,
                    merged=merged,
                    deltas=deltas,
                    joint_row=row,
                    plan_items=plan_items,
                    core_stable_tol=core_stable_tol,
                    joint_weight_config=joint_weight_config,
                )
            else:
                dynamic_weight_result = {
                    "weight_source": "disabled",
                    "weights": {param: _default_weight_detail("disabled") for param in CORE_PARAMS},
                }

            expected_total = 0.0
            for item in plan_items:
                weight_detail = dynamic_weight_result["weights"].get(item["参数"], _default_weight_detail("disabled" if not use_weighted_scale else "global_default"))
                weight = float(weight_detail.get("参数权重", 1.0)) if use_weighted_scale else 1.0
                # Keep the existing scale_factor logic unchanged; only add a per-parameter weight on top.
                weighted_delta = item["基础步长"] * scale * weight if item["建议方向"] in ["增大", "减小"] else 0.0
                item["参数权重"] = float(weight)
                item["权重来源"] = weight_detail.get("权重来源", "disabled" if not use_weighted_scale else "global_default")
                item["effect_score"] = float(weight_detail.get("effect_score", 0.0))
                item["confidence_score"] = float(weight_detail.get("confidence_score", 1.0 if not use_weighted_scale else 0.0))
                item["hit_score"] = float(weight_detail.get("hit_score", 1.0 if not use_weighted_scale else 0.75))
                item["room_score"] = float(weight_detail.get("room_score", 1.0))
                item["stability_score"] = float(weight_detail.get("stability_score", 1.0))
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
                "weight_mode": "dynamic" if use_weighted_scale else "uniform",
                "weight_source": dynamic_weight_result.get("weight_source", "disabled" if not use_weighted_scale else "global_default"),
                "configured_joint_weights": {p: float(configured_joint_weights.get(p, 1.0)) for p in CORE_PARAMS},
                "joint_weights": {
                    p: float(dynamic_weight_result["weights"].get(p, {}).get("参数权重", 1.0)) if use_weighted_scale else 1.0
                    for p in CORE_PARAMS
                },
                "预计总降幅(线性近似)": float(expected_total),
                "组合调参": plan_items,
            }

    result = {
        "latest_time": str(latest["wet_time"]),
        "latest_scene": latest_scene,
        "latest_coarse_scene": latest_coarse_scene,
        "current_wet_weight": current_wet,
        "target_wet_weight": target_wet,
        "target_wet_stable": float(target_info["stable_target"]),
        "target_wet_trend": float(target_info["trend_target"]),
        "target_wet_final": float(target_info["final_target"]),
        "target_alpha": float(target_info["alpha"]),
        "target_recent_vol": float(target_info["recent_vol"]),
        "target_recent_count": int(target_info["recent_count"]),
        "target_method": target_info["target_method"],
        "required_reduction": need_reduce,
        "joint_recommendation": joint_recommendation,
        "recommendations": recs,
    }

    if joint_recommendation is None and not recs:
        result["message"] = "差分样本总体不足，无法输出可靠调参建议。"

    return result


