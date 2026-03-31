#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from spmo_config import load_scene_config, resolve_joint_weight_config
from spmo_fuzzy import run_fuzzy_strategy
from spmo_strategy_common import build_strategy_context
from spmo_strategy_original import run_original_strategy


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--strategy-mode", choices=["original", "fuzzy", "both"], default="both", help="执行原始策略、模糊策略或两者都执行")
    parser.add_argument("--enable-fuzzy-control", action="store_true", help="兼容旧调用方式，等价于 --strategy-mode both")
    parser.add_argument("--fuzzy-output-subdir", default="fuzzy_control_outputs", help="模糊控制输出子目录名")
    return parser


def _print_original_result(output_dir: Path, summary: Dict, plan: Dict) -> None:
    print("=" * 64)
    print("层级场景原始策略分析完成")
    print("=" * 64)
    print(f"输出目录: {output_dir}")
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
    print("最新场景调参建议:")
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
                f"新值={item['建议新值']:.6f}, 预计贡献={item.get('预计降幅贡献(线性近似)', 0.0):.6f}"
            )


def _print_fuzzy_result(output_dir: Path, summary: Dict) -> None:
    print("=" * 64)
    print("模糊控制策略分析完成")
    print("=" * 64)
    print(f"输出目录: {output_dir}")
    fuzzy_control = summary.get("fuzzy_control", {})
    outputs = fuzzy_control.get("outputs", {})
    print(f"最新场景: {summary.get('latest_scene')}")
    print(f"当前湿重: {summary.get('current_wet_weight'):.6f}")
    print(f"目标湿重: {summary.get('target_wet_weight'):.6f}")
    print(f"所需降幅: {summary.get('required_reduction'):.6f}")
    print(f"总调节系数: {outputs.get('overall_adjust_scale', np.nan):.6f}")
    print(f"稳态偏置: {outputs.get('stability_bias', np.nan):.6f}")
    for rule in fuzzy_control.get("rules", []):
        print(f"- {rule}")
    backtest = summary.get("backtest", {})
    if isinstance(backtest, dict):
        print(
            f"模糊回测 mean_abs_error={backtest.get('mean_abs_error', np.nan):.6f}, "
            f"avg_expected_reduction={backtest.get('avg_expected_reduction', np.nan):.6f}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    strategy_mode = "both" if args.enable_fuzzy_control and args.strategy_mode == "original" else args.strategy_mode
    scene_config = load_scene_config(Path(args.config) if args.config else None)
    joint_weight_config = resolve_joint_weight_config(
        scene_config,
        use_weighted_joint_scale=args.use_weighted_joint_scale,
        weight_pressure=args.weight_pressure,
        weight_print_offset=args.weight_print_offset,
        weight_scraper_offset=args.weight_scraper_offset,
    )

    context = build_strategy_context(
        data_dir=Path(args.data_dir),
        max_match_minutes=args.max_match_minutes,
        max_pair_gap_minutes=args.max_pair_gap_minutes,
        max_context_changes=args.max_context_changes,
        min_samples=args.min_samples,
        min_samples_joint=args.min_samples_joint,
        context_tol=args.context_tol,
        coarse_round_step=args.coarse_round_step,
        life_bin_size=args.life_bin_size,
        core_stable_tol=args.core_stable_tol,
        scene_config=scene_config,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"sample_delta_outputs_{ts}"

    if strategy_mode in ("original", "both"):
        original_summary, original_plan = run_original_strategy(
            context=context,
            output_dir=out_dir,
            max_pair_gap_minutes=args.max_pair_gap_minutes,
            backtest_ratio=args.backtest_ratio,
            backtest_noise_scale=args.backtest_noise_scale,
            backtest_noise_seed=args.backtest_noise_seed,
            max_context_changes=args.max_context_changes,
            min_samples=args.min_samples,
            min_samples_joint=args.min_samples_joint,
            context_tol=args.context_tol,
            core_stable_tol=args.core_stable_tol,
            scene_config=scene_config,
            joint_weight_config=joint_weight_config,
        )
        _print_original_result(out_dir, original_summary, original_plan)

    if strategy_mode in ("fuzzy", "both"):
        fuzzy_output_dir = out_dir / args.fuzzy_output_subdir
        fuzzy_summary = run_fuzzy_strategy(
            context=context,
            output_dir=fuzzy_output_dir,
            core_stable_tol=args.core_stable_tol,
            backtest_ratio=args.backtest_ratio,
            backtest_noise_scale=args.backtest_noise_scale,
            backtest_noise_seed=args.backtest_noise_seed,
            max_pair_gap_minutes=args.max_pair_gap_minutes,
            max_context_changes=args.max_context_changes,
            min_samples=args.min_samples,
            min_samples_joint=args.min_samples_joint,
            context_tol=args.context_tol,
            joint_weight_config=joint_weight_config,
            target_wet_config=scene_config.get("target_wet_config"),
        )
        _print_fuzzy_result(fuzzy_output_dir, fuzzy_summary)


if __name__ == "__main__":
    main()
