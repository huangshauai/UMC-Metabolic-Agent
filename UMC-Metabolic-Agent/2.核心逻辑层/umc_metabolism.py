# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 代谢循环核心（AGI L3底层动力）
核心逻辑：实现“信息-能量-物质”三位一体的代谢因子循环，支撑所有上层AGI L3能力
设计原则：仅聚焦代谢循环核心，不涉及上层逻辑；与参数配置联动，支持动态调参
"""
import numpy as np
import pandas as pd
import configparser
import time
from typing import Dict, Any, Optional

class MetabolicCycle:
    """代谢循环核心类（信息-能量-物质三位一体）"""
    def __init__(self):
        # 1. 加载核心参数（从统一配置文件，无硬编码）
        self.param_cfg = self._load_parameters()
        # 2. 初始化代谢核心因子（三位一体）
        self.core_factors = {
            "information": 0.0,  # 信息因子（数据特征/目标优先级）
            "energy": 0.0,       # 能量因子（计算资源/运行效率）
            "matter": 0.0        # 物质因子（数据量/结果产出）
        }
        # 3. 初始化运行状态（支撑闭环验证/可解释性）
        self.is_running = False
        self.cycle_count = 0
        self.stability_score = 0.0  # 代谢稳定性（对应parameters.ini的阈值）

    def _load_parameters(self) -> configparser.ConfigParser:
        """加载代谢循环核心参数（复用parameters.ini）"""
        cfg = configparser.ConfigParser()
        cfg.read("parameters.ini", encoding="utf-8")
        return cfg

    def _calculate_core_factors(self, data: pd.DataFrame, goal: str) -> Dict[str, float]:
        """
        核心：计算信息-能量-物质三位一体因子（AGI L3所有决策的底层依据）
        :param data: 标准化后的输入数据
        :param goal: 自主发现的优化目标
        :return: 归一化后的核心因子值（0~1）
        """
        # 1. 信息因子：基于目标相关特征的重要性（与自主目标发现联动）
        goal_feature = goal.split("：")[-1].split("（")[0] if "：" in goal else "metabolism_core"
        info_factor = self._calculate_info_factor(data, goal_feature)
        
        # 2. 能量因子：基于数据量+计算效率（与性能监控联动）
        energy_factor = self._calculate_energy_factor(data)
        
        # 3. 物质因子：基于结果产出潜力（与代谢循环速度联动）
        matter_factor = self._calculate_matter_factor(data)
        
        # 4. 归一化因子（0~1），避免因子值波动过大
        total = info_factor + energy_factor + matter_factor
        if total == 0:
            total = 1.0  # 避免除零
        
        return {
            "information": info_factor / total,
            "energy": energy_factor / total,
            "matter": matter_factor / total
        }

    def _calculate_info_factor(self, data: pd.DataFrame, goal_feature: str) -> float:
        """计算信息因子：目标相关特征的方差（方差越大，信息越有优化价值）"""
        if goal_feature in data.columns:
            return np.var(data[goal_feature]) * float(self.param_cfg["METABOLISM"]["core_factor_weight"])
        # 无目标特征时，取所有特征的平均方差
        return np.mean([np.var(data[col]) for col in data.columns]) * 0.8

    def _calculate_energy_factor(self, data: pd.DataFrame) -> float:
        """计算能量因子：数据量 × (1/循环速度)（循环越快，能量消耗越高）"""
        data_size = len(data) / 1000  # 归一化数据量（以千条为单位）
        cycle_speed = float(self.param_cfg["BASIC"]["cycle_speed"])
        energy_consumption_limit = float(self.param_cfg["METABOLISM"]["energy_consumption_limit"])
        energy_factor = data_size * (1 / cycle_speed)
        # 限制能量因子不超过上限（避免资源耗尽）
        return min(energy_factor, energy_consumption_limit)

    def _calculate_matter_factor(self, data: pd.DataFrame) -> float:
        """计算物质因子：非空值比例（非空值越多，结果产出潜力越大）"""
        non_null_ratio = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        return non_null_ratio * float(self.param_cfg["METABOLISM"]["core_factor_weight"])

    def _calculate_stability(self, factors: Dict[str, float]) -> float:
        """计算代谢稳定性（与parameters.ini的稳定性阈值联动，低于阈值触发优化）"""
        # 稳定性 = 因子的标准差倒数（因子越均衡，稳定性越高）
        factor_vals = list(factors.values())
        std = np.std(factor_vals)
        stability = 1 / (std + 0.1)  # 避免除零
        # 归一化到0~1
        return min(stability / 10, 1.0)  # 缩放系数保证取值范围

    def run(self, data: pd.DataFrame, goal: str, adapt_rules: Dict[str, Any], cycle_speed: Optional[float] = None) -> Dict[str, Any]:
        """
        代谢循环运行接口（被umc_v20_core.py调用）
        :param data: 标准化后的输入数据
        :param goal: 自主发现的优化目标
        :param adapt_rules: 跨域适配规则
        :param cycle_speed: 循环速度（优先使用参数配置，支持动态覆盖）
        :return: 代谢循环结果（核心因子+稳定性+运行日志）
        """
        self.is_running = True
        cycle_speed = cycle_speed or float(self.param_cfg["BASIC"]["cycle_speed"])
        stability_threshold = float(self.param_cfg["METABOLISM"]["stability_threshold"])
        
        # 初始化结果
        metabolic_result = {
            "core_factors": {},
            "stability_score": 0.0,
            "cycle_count": 0,
            "is_stable": False,
            "adapt_rules_used": adapt_rules["factor_mapping"]
        }

        # 运行代谢循环，直到稳定或达到最大循环次数
        max_cycles = 100  # 避免无限循环
        while self.is_running and self.cycle_count < max_cycles:
            # 1. 计算核心因子
            core_factors = self._calculate_core_factors(data, goal)
            self.core_factors = core_factors
            
            # 2. 计算代谢稳定性
            stability = self._calculate_stability(core_factors)
            self.stability_score = stability
            
            # 3. 更新循环状态
            self.cycle_count += 1
            metabolic_result["core_factors"] = core_factors
            metabolic_result["stability_score"] = stability
            metabolic_result["cycle_count"] = self.cycle_count
            metabolic_result["is_stable"] = stability >= stability_threshold
            
            # 4. 达到稳定阈值则停止循环
            if metabolic_result["is_stable"]:
                break
            
            # 5. 按循环速度休眠（模拟代谢节奏）
            time.sleep(cycle_speed)

        # 记录最终状态
        metabolic_result["final_goal_achieved"] = metabolic_result["is_stable"]
        return metabolic_result

    def stop(self) -> None:
        """停止代谢循环（被umc_v20_core.py调用）"""
        self.is_running = False
        # 重置状态（支撑下次运行）
        self.cycle_count = 0
        self.stability_score = 0.0
        self.core_factors = {"information": 0.0, "energy": 0.0, "matter": 0.0}

# 代谢循环核心验证入口（无冗余，快速测试）
if __name__ == "__main__":
    # 1. 初始化代谢循环
    metabolic_cycle = MetabolicCycle()
    print("代谢循环核心初始化完成")
    
    # 2. 模拟标准化数据（量子领域）
    test_data = pd.DataFrame({
        "qubit_stability": np.random.rand(100) * 0.9,
        "metabolism_core": np.random.rand(100) * 0.8
    })
    
    # 3. 运行代谢循环
    print("开始运行代谢循环...")
    result = metabolic_cycle.run(
        data=test_data,
        goal="提升量子比特代谢稳定性（能量维度）",
        adapt_rules={"factor_mapping": {"qubit_stability": "energy"}}
    )
    
    # 4. 输出核心结果（AGI L3底层依据）
    print("\n=== 代谢循环核心结果 ===")
    print(f"核心因子（信息-能量-物质）：{result['core_factors']}")
    print(f"代谢稳定性得分：{result['stability_score']:.2f}")
    print(f"是否稳定：{result['is_stable']}")
    print(f"循环次数：{result['cycle_count']}")
    
    # 5. 停止循环
    metabolic_cycle.stop()
    print("\n代谢循环已停止")