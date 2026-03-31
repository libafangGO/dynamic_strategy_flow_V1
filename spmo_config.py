#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


CANDIDATES = {
    "process_time": ["INSERTTIME"],
    "wet_time": ["采集时间", "閲囬泦鏃堕棿", "锟缴硷拷时锟斤拷", "閿熺即纭锋嫹鏃堕敓鏂ゆ嫹"],
    "wet_value": ["点数据", "湿重", "婀块噸", "鐐规暟鎹?", "閾佺偣鏁版嵁", "锟斤拷锟斤拷锟斤拷", "閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷?"],
    "process_name": ["工序", "宸ュ簭", "锟斤拷锟斤拷", "閿熸枻鎷烽敓鏂ゆ嫹"],
    "equipment": ["EQUIPMENT", "设备机台", "设备", "璁惧鏈哄彴", "璁惧", "锟借备锟斤拷台", "锟斤拷锟借备", "閿熷€熷閿熸枻鎷峰彴", "閿熸枻鎷烽敓鍊熷"],
    "pressure": ["印刷压力", "鍗板埛鍘嬪姏", "印刷压锟斤拷", "鍗板埛鍘嬮敓鏂ゆ嫹"],
    "print_offset": ["印刷高度偏移", "鍗板埛楂樺害鍋忕Щ", "印刷锟竭讹拷偏锟斤拷", "鍗板埛閿熺璁规嫹鍋忛敓鏂ゆ嫹"],
    "scraper_offset": ["刮刀高度偏移", "鍒垁楂樺害鍋忕Щ", "锟轿碉拷锟竭讹拷偏锟斤拷", "閿熻娇纰夋嫹閿熺璁规嫹鍋忛敓鏂ゆ嫹"],
    "scraper_height_up": ["刮刀高度_上", "鍒垁楂樺害_涓?", "锟轿碉拷锟竭讹拷_锟斤拷", "閿熻娇纰夋嫹閿熺璁规嫹_閿熸枻鎷?"],
    "scraper_height_down": ["刮刀高度_下", "鍒垁楂樺害_涓?", "锟轿碉拷锟竭讹拷_锟斤拷", "閿熻娇纰夋嫹閿熺璁规嫹_閿熸枻鎷?"],
    "ink_height_up": ["墨刀高度_上", "澧ㄥ垁楂樺害_涓?", "墨锟斤拷锟竭讹拷_锟斤拷", "澧ㄩ敓鏂ゆ嫹閿熺璁规嫹_閿熸枻鎷?"],
    "ink_height_down": ["墨刀高度_下", "澧ㄥ垁楂樺害_涓?", "墨锟斤拷锟竭讹拷_锟斤拷", "澧ㄩ敓鏂ゆ嫹閿熺璁规嫹_閿熸枻鎷?"],
    "screen_life": ["网版使用次数", "网板使用次数", "缃戠増浣跨敤娆℃暟", "缃戞澘浣跨敤娆℃暟", "锟斤拷锟斤拷使锟矫达拷锟斤拷", "閿熸枻鎷烽敓鏂ゆ嫹浣块敓鐭揪鎷烽敓鏂ゆ嫹"],
    "speed": ["印刷速度", "鍗板埛閫熷害", "印刷锟劫讹拷", "鍗板埛閿熷姭璁规嫹"],
}

CORE_PARAMS = ["印刷压力", "印刷高度偏移", "刮刀高度偏移"]
MATCH_VIS_EXCLUDE_COLS = {"wet_time", "process_time"}
DEFAULT_SCENE_CONFIG = {
    "coarse_scene": {
        "features": ["刮刀高度_上", "刮刀高度_下", "墨刀高度_上", "墨刀高度_下"],
        "method": "fixed",
        "fixed_round_step": 0.5,
        "tree": {
            "max_leaf_nodes": 6,
            "min_samples_leaf": 20,
        },
    },
    "life_segment": {
        "enabled": True,
        "feature": "网版寿命",
        "method": "fixed",
        "fixed_bin_size": 200000,
        "tree": {
            "max_leaf_nodes": 4,
            "min_samples_leaf": 20,
        },
    },
    "joint_recommendation": {
        "use_weighted_scale": False,
        "weights": {
            "印刷压力": 1.0,
            "印刷高度偏移": 1.0,
            "刮刀高度偏移": 1.0,
        },
        "dynamic_weight": {
            "k": 8.0,
            "lambda": 0.8,
            "recent_window": 20,
            "min_scene_samples": 8,
            "eps": 1e-6,
        },
    },
    "target_wet_config": {
        "enabled_dual_target": True,
        "recent_window": 80,
        "stable_quantile": 0.35,
        "trend_bottom_ratio": 0.20,
        "trend_push_down": 0.02,
        "vol_low": 0.15,
        "vol_high": 0.50,
        "alpha_low": 0.30,
        "alpha_high": 0.70,
        "min_recent_samples": 12,
        "target_floor_quantile": 0.05,
        "target_ceiling_quantile": 0.50,
    },
}


def _format_scene_value(v) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.3f}"
    return str(v)


def uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    cols: List[str] = []
    for c in df.columns:
        c = str(c)
        n = counts.get(c, 0)
        counts[c] = n + 1
        cols.append(c if n == 0 else f"{c}__dup{n}")
    out = df.copy()
    out.columns = cols
    return out


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    for name in candidates:
        if name in cols:
            return name
        if str(name).lower() in lower_map:
            return lower_map[str(name).lower()]

    for c in cols:
        c_low = str(c).lower()
        for name in candidates:
            n_low = str(name).lower()
            if c_low == n_low or c_low.startswith(n_low):
                return c

    for c in cols:
        c_low = str(c).lower()
        for name in candidates:
            if str(name).lower() in c_low:
                return c
    return None


def parse_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def deep_merge_dict(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_scene_config(config_path: Optional[Path]) -> Dict:
    config = json.loads(json.dumps(DEFAULT_SCENE_CONFIG, ensure_ascii=False))
    if config_path is None or (not config_path.exists()):
        return config
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
    return deep_merge_dict(config, user_config)


def resolve_joint_weight_config(
    scene_config: Dict,
    use_weighted_joint_scale: bool = False,
    weight_pressure: Optional[float] = None,
    weight_print_offset: Optional[float] = None,
    weight_scraper_offset: Optional[float] = None,
) -> Dict:
    joint_cfg = json.loads(
        json.dumps(scene_config.get("joint_recommendation", DEFAULT_SCENE_CONFIG["joint_recommendation"]), ensure_ascii=False)
    )
    weights = joint_cfg.setdefault("weights", {})

    overrides = {
        "印刷压力": weight_pressure,
        "印刷高度偏移": weight_print_offset,
        "刮刀高度偏移": weight_scraper_offset,
    }
    has_cli_weight = False
    for param, value in overrides.items():
        if value is not None:
            weights[param] = float(value)
            has_cli_weight = True

    if use_weighted_joint_scale or has_cli_weight:
        joint_cfg["use_weighted_scale"] = True

    for param in CORE_PARAMS:
        weights[param] = float(weights.get(param, 1.0))
    return joint_cfg
