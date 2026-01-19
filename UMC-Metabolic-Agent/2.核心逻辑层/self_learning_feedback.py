# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 自主学习反馈模块（AGI L3核心）
核心逻辑：基于性能监控的得分结果，自动优化策略权重/代谢因子权重，反馈到配置文件
设计原则：优化幅度可配置、调整有边界、与核心模块深度联动，无冗余反馈逻辑
"""
import configparser
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# 联动核心模块（策略调整+性能监控）
from umc_strategy import UMCStrategy
from umc_performance import PerformanceMonitor

class SelfLearningFeedback:
    """自主学习反馈核心类（AGI L3闭环优化的关键）"""
    def __init__(self):
        # 1. 加载核心配置（复用parameters.ini的AGI_L3段）
        self.param_cfg = self._load_config("parameters.ini")
        # 2. 初始化联动模块（策略调整+性能监控）
        self.strategy_module = UMCStrategy()
        self.perf_monitor = PerformanceMonitor()
        # 3. 定义优化规则（可配置，避免权重波动过大）
        self.optimize_rules = {
            "feedback_rate": float(self.param_cfg["AGI_L3"]["self_learning_feedback_rate"]),  # 优化幅度
            "score_high_threshold": 0.9,  # 高得分：不优化
            "score_mid_threshold": 0.7,  # 中得分：小幅优化
            "score_low_threshold": 0.5,  # 低得分：大幅优化
            "weight_adjust_max": 0.2,    # 单次调整最大幅度
            "weight_adjust_min": 0.05    # 单次调整最小幅度
        }
        # 4. 初始化反馈状态（支撑可解释性）
        self.feedback_history = []
        self.current_optimize_result = None
        self.optimize_count = 0

    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """统一加载参数配置（读取反馈率/阈值）"""
        cfg = configparser.ConfigParser()
        cfg.read(config_file, encoding="utf-8")
        return cfg

    def _calculate_adjust_amount(self, perf_score: float) -> float:
        """
        核心：基于性能得分计算权重调整幅度（有边界，避免策略失控）
        :param perf_score: 性能监控模块输出的得分（0~1）
        :return: 权重调整幅度（正数=增加，负数=降低）
        """
        # 1. 按得分区间确定调整方向和基础幅度
        if perf_score >= self.optimize_rules["score_high_threshold"]:
            # 高得分：无需调整
            return 0.0
        elif perf_score >= self.optimize_rules["score_mid_threshold"]:
            # 中得分：小幅优化（基础幅度=反馈率×最小调整值）
            adjust_base = self.optimize_rules["feedback_rate"] * self.optimize_rules["weight_adjust_min"]
        elif perf_score >= self.optimize_rules["score_low_threshold"]:
            # 低得分：中幅优化
            adjust_base = self.optimize_rules["feedback_rate"] * (self.optimize_rules["weight_adjust_min"] + self.optimize_rules["weight_adjust_max"]) / 2
        else:
            # 极低得分：大幅优化
            adjust_base = self.optimize_rules["feedback_rate"] * self.optimize_rules["weight_adjust_max"]

        # 2. 随机扰动（模拟自主学习的探索性，幅度±10%）
        adjust_amount = adjust_base * (1 + np.random.uniform(-0.1, 0.1))
        # 3. 限制调整幅度（不超过最大/最小边界）
        adjust_amount = np.clip(adjust_amount, -self.optimize_rules["weight_adjust_max"], self.optimize_rules["weight_adjust_max"])
        
        return adjust_amount

    def _determine_adjust_target(self, metabolic_result: Dict[str, Any], goal_result: Dict[str, Any]) -> str:
        """确定优化目标（策略权重）：基于代谢因子+目标维度"""
        # 1. 提取代谢因子中得分最低的维度
        core_factors = metabolic_result["core_factors"]
        low_factor = min(core_factors.items(), key=lambda x: x[1])[0]
        # 2. 映射到策略优化目标
        factor_to_strategy = {
            "information": "unknown_domain" if goal_result["factor_dimension"] == "information" else goal_result["optimize_target"],
            "energy": "qubit_stability" if "energy" in goal_result["goal"] else "atomic_frequency",
            "matter": "logistics_efficiency" if "matter" in goal_result["goal"] else "unknown_domain",
            "stability": "unknown_domain"
        }
        # 3. 优先调整目标维度对应的策略
        adjust_target = goal_result.get("optimize_target", factor_to_strategy[low_factor])
        # 4. 兜底：若目标不存在，用陌生领域默认
        if adjust_target not in self.param_cfg["STRATEGY"]:
            adjust_target = "unknown_domain"
        
        return adjust_target

    def feedback_optimize(self, input_data: pd.DataFrame, metabolic_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        AGI L3核心：自主学习反馈优化（无人工干预，自动调整策略权重）
        :param input_data: 标准化后的数据（用于验证优化效果）
        :param metabolic_result: 代谢循环执行结果（含核心因子/稳定性）
        :return: 优化结果（调整目标/幅度/新旧权重/效果）
        """
        # 1. 获取性能得分（来自umc_performance.py）
        perf_score = self.perf_monitor.current_performance_score
        if perf_score == 0.0:
            # 无性能数据时，基于稳定性计算得分
            perf_score = metabolic_result["stability_score"]

        # 2. 获取自主发现的目标结果（来自goal_discovery.py）
        goal_result = self.perf_monitor.performance_history[-1] if self.perf_monitor.performance_history else {"factor_dimension": "stability", "optimize_target": "unknown_domain"}

        # 3. 计算权重调整幅度
        adjust_amount = self._calculate_adjust_amount(perf_score)
        if adjust_amount == 0.0:
            # 无需优化，直接返回
            optimize_result = {
                "adjust_target": "none",
                "adjust_amount": 0.0,
                "old_weight": 0.0,
                "new_weight": 0.0,
                "reason": f"性能得分{perf_score:.2f}≥高阈值{self.optimize_rules['score_high_threshold']}，无需优化",
                "optimize_status": "no_optimize"
            }
            self.current_optimize_result = optimize_result
            self.feedback_history.append(optimize_result)
            return optimize_result

        # 4. 确定优化目标（策略权重）
        adjust_target = self._determine_adjust_target(metabolic_result, goal_result)

        # 5. 执行策略权重调整（调用umc_strategy.py）
        adjust_result = self.strategy_module.adjust_strategy_weight(adjust_target, adjust_amount)

        # 6. 验证优化效果（简化版：重新计算特征重要性）
        from goal_discovery import AutonomousGoalDiscovery
        goal_discoverer = AutonomousGoalDiscovery()
        feature_importance = goal_discoverer._calculate_feature_importance(input_data)
        optimize_effect = "提升" if max(feature_importance.values()) > perf_score else "无提升"

        # 7. 构造优化结果（可解释性）
        optimize_result = {
            "adjust_target": adjust_target,
            "adjust_amount": adjust_amount,
            "old_weight": adjust_result["old_weight"],
            "new_weight": adjust_result["new_weight"],
            "perf_score_before": perf_score,
            "optimize_effect": optimize_effect,
            "reason": f"性能得分{perf_score:.2f}，调整{adjust_target}权重{adjust_amount:.3f}",
            "optimize_status": "success"
        }

        # 8. 更新状态，记录历史
        self.current_optimize_result = optimize_result
        self.optimize_count += 1
        self.feedback_history.append(optimize_result)

        return optimize_result

    def get_feedback_history(self, last_n: int = 5) -> list[Dict[str, Any]]:
        """获取最近n条反馈优化历史（支撑可视化/可解释性）"""
        return self.feedback_history[-last_n:] if self.feedback_history else []

# 自主学习反馈验证入口（无冗余，快速测试）
if __name__ == "__main__":
    # 1. 初始化自主学习反馈模块
    feedback_optimizer = SelfLearningFeedback()
    print("UMC自主学习反馈模块初始化完成")

    # 2. 模拟输入数据（标准化后）
    test_data = pd.DataFrame({
        "qubit_stability": np.random.rand(100) * 0.9,
        "energy_consumption": np.random.rand(100) * 0.8,
        "matter_output": np.random.rand(100) * 0.7
    })

    # 3. 模拟代谢循环结果（来自umc_metabolism.py）
    mock_metabolic_result = {
        "core_factors": {"information": 0.5, "energy": 0.4, "matter": 0.6},
        "stability_score": 0.65,
        "is_stable": False,
        "cycle_count": 15
    }

    # 4. 模拟性能得分（低得分，触发优化）
    feedback_optimizer.perf_monitor.current_performance_score = 0.6

    # 5. 执行自主学习反馈优化
    optimize_result = feedback_optimizer.feedback_optimize(test_data, mock_metabolic_result)
    print(f"\n=== 自主学习反馈优化结果 ===")
    print(f"优化目标：{optimize_result['adjust_target']}")
    print(f"调整幅度：{optimize_result['adjust_amount']:.3f}")
    print(f"权重变化：{optimize_result['old_weight']:.2f} → {optimize_result['new_weight']:.2f}")
    print(f"优化原因：{optimize_result['reason']}")
    print(f"优化效果：{optimize_result['optimize_effect']}")

    # 6. 查看反馈历史
    print(f"\n=== 反馈优化历史 ===")
    print(feedback_optimizer.get_feedback_history())