# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 数据标准化处理模块（AGI L3跨域适配数据层核心）
核心逻辑：统一处理多领域原始数据（校验、缺失值填充、归一化），输出标准化数据，支撑跨域适配/代谢循环
设计原则：处理规则可配置、跨领域通用、与核心模块无缝衔接，无冗余数据处理逻辑
"""
import configparser
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")  # 屏蔽数据处理无关警告

class DataProcessor:
    """数据标准化处理核心类（多领域通用）"""
    def __init__(self):
        # 1. 加载配置（参数+路径，复用统一配置文件）
        self.param_cfg = self._load_config("parameters.ini")
        self.path_cfg = self._load_config("paths.ini")
        # 2. 定义数据处理规则（可配置，适配不同领域）
        self.process_rules = {
            "missing_value_strategy": "mean",  # 缺失值填充策略：mean/median/mode
            "normalization_range": (0, 1),     # 归一化范围（固定0~1，适配代谢因子）
            "data_type_check": ["float64", "int64"],  # 允许的数据类型
            "cache_size": int(self.param_cfg["BASIC"]["data_cache_size"])  # 数据缓存量
        }
        # 3. 初始化缓存（减少重复处理，提升效率）
        self.data_cache = {}

    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """统一加载参数/路径配置"""
        cfg = configparser.ConfigParser()
        cfg.read(config_file, encoding="utf-8")
        return cfg

    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        数据校验（AGI L3跨域适配的前置步骤）：检查空数据、格式错误、数据类型
        :param data: 原始输入数据
        :return: 校验结果（是否通过+错误信息）
        """
        validate_result = {
            "is_valid": True,
            "error_msg": "",
            "data_shape": data.shape
        }

        # 1. 检查是否为空数据
        if data.empty:
            validate_result["is_valid"] = False
            validate_result["error_msg"] = "输入数据为空"
            return validate_result

        # 2. 检查数据类型（仅允许数值型）
        non_numeric_cols = data.select_dtypes(exclude=self.process_rules["data_type_check"]).columns
        if len(non_numeric_cols) > 0:
            # 尝试转换非数值列（文本转数值失败则报错）
            for col in non_numeric_cols:
                try:
                    data[col] = pd.to_numeric(data[col], errors="raise")
                except:
                    validate_result["is_valid"] = False
                    validate_result["error_msg"] = f"非数值列转换失败：{non_numeric_cols.tolist()}"
                    return validate_result

        # 3. 检查缺失值比例（超过80%则报错）
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.8:
            validate_result["is_valid"] = False
            validate_result["error_msg"] = f"缺失值比例过高（{missing_ratio:.2f} > 0.8）"
            return validate_result

        return validate_result

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """缺失值处理（基于配置的策略，跨领域通用）"""
        strategy = self.process_rules["missing_value_strategy"]
        numeric_cols = data.select_dtypes(include=self.process_rules["data_type_check"]).columns

        for col in numeric_cols:
            if data[col].isnull().sum() == 0:
                continue
            # 按策略填充缺失值
            if strategy == "mean":
                fill_val = data[col].mean()
            elif strategy == "median":
                fill_val = data[col].median()
            elif strategy == "mode":
                fill_val = data[col].mode()[0]
            else:
                fill_val = 0.0  # 默认填充0
            data[col] = data[col].fillna(fill_val)

        return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据归一化（统一缩放到0~1，适配代谢因子取值范围）"""
        numeric_cols = data.select_dtypes(include=self.process_rules["data_type_check"]).columns
        min_val = self.process_rules["normalization_range"][0]
        max_val = self.process_rules["normalization_range"][1]

        for col in numeric_cols:
            col_min = data[col].min()
            col_max = data[col].max()
            # 避免除零（列值全相同则设为0.5）
            if col_max - col_min == 0:
                data[col] = 0.5
            else:
                data[col] = (data[col] - col_min) / (col_max - col_min) * (max_val - min_val) + min_val

        return data

    def standardize_data(self, data: pd.DataFrame, domain_name: str = "unknown", use_cache: bool = True) -> pd.DataFrame:
        """
        AGI L3核心：多领域数据统一标准化入口（被跨域适配/核心智能体调用）
        :param data: 原始输入数据（任意领域）
        :param domain_name: 领域名称（用于缓存标识）
        :param use_cache: 是否使用缓存（提升重复处理效率）
        :return: 标准化后的数据（可直接输入代谢循环/无监督适配）
        """
        # 1. 缓存检查（避免重复处理）
        cache_key = f"{domain_name}_{data.shape}"
        if use_cache and cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # 2. 数据校验（失败则抛出异常，由auto_recovery处理）
        validate_result = self._validate_data(data)
        if not validate_result["is_valid"]:
            raise ValueError(f"数据校验失败：{validate_result['error_msg']}")

        # 3. 缺失值处理
        data = self._handle_missing_values(data.copy())

        # 4. 数据归一化
        data = self._normalize_data(data)

        # 5. 写入缓存（控制缓存大小，避免内存溢出）
        if use_cache:
            if len(self.data_cache) >= self.process_rules["cache_size"]:
                # 移除最早的缓存项
                oldest_key = list(self.data_cache.keys())[0]
                del self.data_cache[oldest_key]
            self.data_cache[cache_key] = data

        # 6. 保存标准化后的数据（可选，支撑可解释性）
        self._save_processed_data(data, domain_name)

        return data

    def _save_processed_data(self, data: pd.DataFrame, domain_name: str) -> None:
        """保存标准化后的数据到paths.ini指定目录（支撑回溯/可解释性）"""
        processed_dir = self.path_cfg["PATH"]["processed_data_dir"]
        os.makedirs(processed_dir, exist_ok=True)
        save_path = os.path.join(processed_dir, f"{domain_name}_processed_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv")
        data.to_csv(save_path, index=False, encoding="utf-8")

    def clear_cache(self) -> None:
        """清空数据缓存（支撑内存管理）"""
        self.data_cache.clear()

# 数据处理核心验证入口（无冗余，快速测试）
if __name__ == "__main__":
    # 1. 初始化数据处理模块
    data_processor = DataProcessor()
    print("UMC数据标准化处理模块初始化完成")

    # 2. 模拟原始数据（量子领域，含缺失值、非数值列）
    raw_data = pd.DataFrame({
        "qubit_stability": [0.8, 0.7, None, 0.9, "0.85"],  # 含缺失值+文本数值
        "energy_consumption": [120, 110, 130, None, 115],  # 含缺失值
        "matter_output": [50, 45, 55, 48, 52]
    })
    print(f"\n=== 原始数据 ===")
    print(raw_data)
    print(f"原始数据类型：\n{raw_data.dtypes}")

    # 3. 执行数据标准化
    try:
        standardized_data = data_processor.standardize_data(raw_data, domain_name="quantum")
        print(f"\n=== 标准化后数据 ===")
        print(standardized_data)
        print(f"标准化数据类型：\n{standardized_data.dtypes}")
        print(f"缺失值检查：{standardized_data.isnull().sum().sum()}")
        print(f"数值范围：min={standardized_data.min().min():.2f}, max={standardized_data.max().max():.2f}")
    except ValueError as e:
        print(f"数据处理失败：{e}")

    # 4. 清空缓存
    data_processor.clear_cache()
    print(f"\n数据缓存已清空，当前缓存量：{len(data_processor.data_cache)}")