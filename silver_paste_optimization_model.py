#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
????????????????????+ ????????- ???????????- ??????????????????????????????????????
- ??????/?????????????????????????????????
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from spmo_analysis import analyze_joint_param_effects, analyze_single_param_effects, build_adjacent_deltas, latest_scene_adjustment
from spmo_backtest import run_backtest
from spmo_config import CORE_PARAMS, load_scene_config, resolve_joint_weight_config
from spmo_data_scene import build_hierarchical_scene_keys, build_scene_keys_from_config, load_and_match
from spmo_visualizations import (
    run_scene_decision_tree,
    save_coarse_scene_param_boxplots,
    save_decision_tree_scene_boxplots_and_table,
    save_fixed_coarse_scene_structure_table,
    save_hierarchical_scene_visualizations,
    save_latest_adjustment_visualizations,
    save_matched_samples_visualization,
    save_single_param_visualizations,
)

def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    max_match_minutes: int,
    max_pair_gap_minutes: int,
    max_context_changes: int,
    min_samples: int,
    min_samples_joint: int,
    context_tol: float,
    coarse_round_step: float,
    life_bin_size: float,
    core_stable_tol: float,
    backtest_ratio: float,
    backtest_noise_scale: float,
    backtest_noise_seed: int,
    scene_config: Dict,
    joint_weight_config: Dict,
) -> Dict:
    merged = load_and_match(data_dir, max_match_minutes=max_match_minutes)
    output_dir.mkdir(parents=True, exist_ok=True)
    match_vis_path = save_matched_samples_visualization(merged, output_dir)

    scene_cols = [c for c in ["刮刀高度_上", "刮刀高度_下", "墨刀高度_上", "墨刀高度_下"] if c in merged.columns]
    if len(scene_cols) == 0:
        raise ValueError("未识别到场景拆分列（刮刀/墨刀高度）")

    merged = build_hierarchical_scene_keys(
        merged,
        coarse_scene_cols=scene_cols,
        screen_life_col="网版寿命" if "网版寿命" in merged.columns else None,
        coarse_round_step=coarse_round_step,
        life_bin_size=life_bin_size,
    )
    merged = build_scene_keys_from_config(merged, scene_config=scene_config)
    scene_cols = [c for c in scene_config.get("coarse_scene", {}).get("features", []) if c in merged.columns]
    hierarchical_vis_paths = save_hierarchical_scene_visualizations(merged, output_dir)
    coarse_scene_distribution_paths = save_coarse_scene_param_boxplots(
        merged,
        output_dir,
        max_scenes=min(6, int(merged["scene_coarse_key"].nunique())),
    )
    fixed_coarse_scene_table_paths = save_fixed_coarse_scene_structure_table(
        merged,
        output_dir,
        max_scenes=min(6, int(merged["scene_coarse_key"].nunique())),
    )
    decision_tree_scene_summary = run_scene_decision_tree(
        merged,
        output_dir,
        max_leaf_nodes=max(2, int(merged["scene_coarse_key"].nunique())),
        min_samples_leaf=max(20, int(len(merged) * 0.02)),
    )
    decision_tree_assignment_df = pd.read_csv(output_dir / "decision_tree_scene_assignments.csv")
    decision_tree_scene_display_paths = save_decision_tree_scene_boxplots_and_table(
        decision_tree_assignment_df,
        output_dir,
        max_scenes=6,
    )

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

    plan = latest_scene_adjustment(
        merged,
        effects,
        joint_effects,
        deltas,
        core_stable_tol=core_stable_tol,
        joint_weight_config=joint_weight_config,
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
    )

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
        "coarse_round_step": coarse_round_step,
        "life_bin_size": life_bin_size,
        "core_stable_tol": core_stable_tol,
        "scene_config": scene_config,
        "joint_weight_config": joint_weight_config,
        "matched_samples_visualization": str(match_vis_path) if match_vis_path is not None else None,
        "hierarchical_scene_visualizations": hierarchical_vis_paths,
        "coarse_scene_param_distributions": coarse_scene_distribution_paths,
        "fixed_coarse_scene_structure_table": fixed_coarse_scene_table_paths,
        "decision_tree_scene_split": decision_tree_scene_summary,
        "decision_tree_scene_display": decision_tree_scene_display_paths,
        "single_param_visualizations": single_param_vis_paths,
        "latest_adjustment_visualizations": latest_adjustment_vis_paths,
        "backtest_last_10pct": backtest_summary,
        "latest_scene": plan.get("latest_scene"),
        "latest_coarse_scene": plan.get("latest_coarse_scene"),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, plan


def main() -> None:
    parser = argparse.ArgumentParser(description="三参数纯样本差分分析（层级场景）")
    parser.add_argument("--data-dir", default="D:/4.正泰新能-银浆单耗", help="数据目录")
    parser.add_argument("--output-dir", default="D:/4.正泰新能-银浆单耗/dynamic_strategy_flow_V1", help="输出根目录")
    parser.add_argument("--max-match-minutes", type=int, default=30, help="湿重与参数匹配最大时间差")
    parser.add_argument("--max-pair-gap-minutes", type=int, default=180, help="临近样本最大间隔(分钟)")
    parser.add_argument("--max-context-changes", type=int, default=3, help="允许的其他参数变化个数上限")
    parser.add_argument("--min-samples", type=int, default=4, help="每细场景-参数最少单参数样本数")
    parser.add_argument("--min-samples-joint", type=int, default=12, help="每细场景最少联动样本数")
    parser.add_argument("--context-tol", type=float, default=1e-6, help="其他参数是否变化阈值")
    parser.add_argument("--coarse-round-step", type=float, default=0.5, help="刮刀/墨刀高度粗分步长")
    parser.add_argument("--life-bin-size", type=float, default=200000, help="网版寿命固定分箱宽度")
    parser.add_argument("--core-stable-tol", type=float, default=0.1, help="核心参数近似不变阈值")
    parser.add_argument("--backtest-ratio", type=float, default=0.1, help="按时间顺序用于回测的最后样本占比")
    parser.add_argument("--backtest-noise-scale", type=float, default=1.0, help="回测扰动强度倍数，基于历史波动标准差")
    parser.add_argument("--backtest-noise-seed", type=int, default=42, help="回测扰动随机种子")
    parser.add_argument("--config", default=str(Path(__file__).with_name("scene_config.json")), help="场景配置文件(JSON)")
    parser.add_argument("--use-weighted-joint-scale", action="store_true", help="启用联动组合调参的参数加权缩放")
    parser.add_argument("--weight-pressure", type=float, default=None, help="印刷压力权重")
    parser.add_argument("--weight-print-offset", type=float, default=None, help="印刷高度偏移权重")
    parser.add_argument("--weight-scraper-offset", type=float, default=None, help="刮刀高度偏移权重")
    args = parser.parse_args()
    scene_config = load_scene_config(Path(args.config) if args.config else None)
    joint_weight_config = resolve_joint_weight_config(
        scene_config,
        use_weighted_joint_scale=args.use_weighted_joint_scale,
        weight_pressure=args.weight_pressure,
        weight_print_offset=args.weight_print_offset,
        weight_scraper_offset=args.weight_scraper_offset,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"sample_delta_outputs_{ts}"

    summary, plan = run_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=out_dir,
        max_match_minutes=args.max_match_minutes,
        max_pair_gap_minutes=args.max_pair_gap_minutes,
        max_context_changes=args.max_context_changes,
        min_samples=args.min_samples,
        min_samples_joint=args.min_samples_joint,
        context_tol=args.context_tol,
        coarse_round_step=args.coarse_round_step,
        life_bin_size=args.life_bin_size,
        core_stable_tol=args.core_stable_tol,
        backtest_ratio=args.backtest_ratio,
        backtest_noise_scale=args.backtest_noise_scale,
        backtest_noise_seed=args.backtest_noise_seed,
        scene_config=scene_config,
        joint_weight_config=joint_weight_config,
    )

    print("=" * 64)
    print("层级场景纯样本差分分析完成")
    print("=" * 64)
    print(f"输出目录: {out_dir}")
    print(f"样本数: {summary['samples']}")
    print(f"粗场景数: {summary['coarse_scenes']}")
    print(f"细场景数(含寿命段): {summary['fine_scenes']}")
    print(f"临近样本对: {summary['adjacent_pairs']}")
    print(f"单参数有效结论行数: {summary['effect_rows']}")
    print(f"联动有效结论行数: {summary['joint_effect_rows']}")
    if isinstance(summary.get("backtest_last_10pct"), dict):
        backtest_core = summary["backtest_last_10pct"].get("wet_value_summary", {})
        if backtest_core:
            print("最后10%回测湿重统计:")
            for metric, stats in backtest_core.items():
                print(
                    f"- {metric}: 均值={stats.get('mean', np.nan):.6f}, "
                    f"方差={stats.get('var', np.nan):.6f}, 累计和={stats.get('sum', np.nan):.6f}"
                )

    print(f"最新场景调参建议:")
    print(f"时间: {plan.get('latest_time')}")
    print(f"细场景: {plan.get('latest_scene')}")
    print(f"粗场景: {plan.get('latest_coarse_scene')}")

    if plan.get("joint_recommendation"):
        jr = plan["joint_recommendation"]
        print(
            f"当前湿重: {plan['current_wet_weight']:.6f}, 目标湿重: {plan['target_wet_weight']:.6f}, "
            f"联动预计降幅: {jr.get('预计总降幅(线性近似)', 0.0):.6f}"
        )
        print("联动组合建议:")
        for item in jr.get("组合调参", []):
            print(
                f"- {item['参数']}: {item['建议方向']} {abs(item['建议变化量']):.6f}, "
                f"新值={item['建议新值']:.6f}, 预计贡献={item['预计降幅贡献(线性近似)']:.6f}"
            )
    elif plan.get("recommendations"):
        print(f"当前湿重: {plan['current_wet_weight']:.6f}, 目标湿重: {plan['target_wet_weight']:.6f}")
        for r in plan["recommendations"]:
            print(
                f"- {r['参数']}: {r['建议方向']} {abs(r['建议变化量']):.6f}, "
                f"新值={r['建议新值']:.6f}, 预计降幅={r['基于历史斜率预计降幅']:.6f}"
            )
    else:
        print(plan.get("message", "暂无建议"))


if __name__ == "__main__":
    main()
