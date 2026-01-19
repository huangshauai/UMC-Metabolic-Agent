# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 自主决策策略模块（AGI L3核心）
核心逻辑：基于代谢核心因子（信息-能量-物质）动态选择最优策略，支持权重自主优化
设计原则：策略与因子联动、权重可配置、适配多领域，无硬编码策略
"""
import configparser
import numpy as np
from typing import Dict, Any, Optional

class UMCStrategy:
    """自主决策策略核心类（多领域动态适配）"""
    def __init__(self):
        # 1. 加载策略配置（复用parameters.ini的STRATEGY段）
        self.param_cfg = self._load_strategy_config()
        # 2. 定义领域-策略映射（AGI L3跨域适配的核心策略库）
        self.domain_strategies = {
            "quantum": {
                "name": "量子领域代谢优化策略",
                "factor_weight": {"information": 0.3, "energy": 0.5, "matter": 0.2},
                "optimize_target": "qubit_stability"
            },
            "atomic": {
                "name": "原子领域代谢优化策略",
                "factor_weight": {"information": 0.4, "energy": 0.3, "matter": 0.3},
                "optimize_target": "atomic_frequency"
            },
            "logistics": {
                "name": "物流领域代谢优化策略",
                "factor_weight": {"information": 0.2, "energy": 0.2, "matter": 0.6},
                "optimize_target": "logistics_efficiency"
            },
            "unknown": {
                "name": "陌生领域默认适配策略",
                "factor_weight": {"information": 0.33, "energy": 0.33, "matter": 0.34},
                "optimize_target": "unknown_domain"
            }
        }
        # 3. 初始化策略状态（支撑可解释性/自主优化）
        self.current_strategy = None
        self.current_strategy_weight = 0.0
        self.strategy_adjust_count = 0

    def _load_strategy_config(self) -> configparser.ConfigParser:
        """加载策略权重配置（复用parameters.ini，支持动态调整）"""
        cfg = configparser.ConfigParser()
        cfg.read("parameters.ini", encoding="utf-8")
        return cfg

    def select_optimal_strategy(self, domain_name: str, metabolic_factors: Dict[str, float]) -> Dict[str, Any]:
        """
        AGI L3核心：基于代谢因子动态选择最优策略
        :param domain_name: 领域名称（quantum/atomic/logistics/unknown）
        :param metabolic_factors: 代谢循环输出的核心因子（信息-能量-物质）
        :return: 最优策略+权重+适配建议
        """
        # 1. 确定基础策略（跨域适配）
        base_strategy = self.domain_strategies.get(domain_name, self.domain_strategies["unknown"])
        
        # 2. 加载该策略的配置权重（从parameters.ini，支持人工/自主调整）
        config_weight = float(self.param_cfg["STRATEGY"][base_strategy["optimize_target"]])
        
        # 3. 计算策略适配得分（代谢因子 × 策略因子权重）
        strategy_score = self._calculate_strategy_score(base_strategy, metabolic_factors)
        
        # 4. 融合配置权重，得到最终策略得分
        final_score = strategy_score * config_weight
        
        # 5. 更新策略状态（支撑可解释性/自主反馈）
        self.current_strategy = base_strategy["name"]
        self.current_strategy_weight = final_score
        self.strategy_adjust_count += 1
        
        # 6. 返回最终策略（含适配建议）
        return {
            "strategy_name": base_strategy["name"],
            "domain": domain_name,
            "factor_weight": base_strategy["factor_weight"],
            "optimize_target": base_strategy["optimize_target"],
            "strategy_score": final_score,
            "config_weight": config_weight,
            "adjust_count": self.strategy_adjust_count
        }

    def _calculate_strategy_score(self, strategy: Dict[str, Any], factors: Dict[str, float]) -> float:
        """计算策略适配得分（核心算法：因子加权求和）"""
        score = 0.0
        for factor_name, factor_val in factors.items():
            score += factor_val * strategy["factor_weight"][factor_name]
        # 归一化得分（0~1）
        return min(score, 1.0)

    def adjust_strategy_weight(self, target: str, weight_increase: float) -> Dict[str, float]:
        """
        支持自主学习反馈的策略权重调整（被self_learning_feedback.py调用）
        :param target: 调整目标（如qubit_stability）
        :param weight_increase: 权重增加量（0~0.5，由反馈模块决定）
        :return: 调整后的权重
        """
        if target not in self.param_cfg["STRATEGY"]:
            target = "unknown_domain"  # 陌生领域默认调整
        
        # 1. 获取当前权重
        current_weight = float(self.param_cfg["STRATEGY"][target])
        # 2. 调整权重（不超过1.0，不低于0.1）
        new_weight = np.clip(current_weight + weight_increase, 0.1, 1.0)
        # 3. 更新配置文件（实时生效，支撑AGI L3自主优化）
        self.param_cfg["STRATEGY"][target] = str(new_weight)
        with open("parameters.ini", "w", encoding="utf-8") as f:
            self.param_cfg.write(f)
        
        # 4. 返回调整结果
        return {
            "target": target,
            "old_weight": current_weight,
            "new_weight": new_weight,
            "adjust_amount": weight_increase
        }

    def get_current_strategy_status(self) -> Dict[str, Any]:
        """获取当前策略状态（支撑白盒调试/可解释性）"""
        return {
            "current_strategy": self.current_strategy,
            "current_strategy_weight": self.current_strategy_weight,
            "adjust_count": self.strategy_adjust_count,
            "domain_strategies": {k: v["name"] for k, v in self.domain_strategies.items()}
        }

# 自主策略核心验证入口（无冗余，快速测试）
if __name__ == "__main__":
    # 1. 初始化策略模块
    strategy_module = UMCStrategy()
    print("UMC自主决策策略模块初始化完成")
    
    # 2. 模拟代谢因子（来自umc_metabolism.py的输出）
    mock_metabolic_factors = {
        "information": 0.6,
        "energy": 0.8,
        "matter": 0.4
    }
    
    # 3. 选择量子领域最优策略
    print("\n=== 选择量子领域最优策略 ===")
    quantum_strategy = strategy_module.select_optimal_strategy("quantum", mock_metabolic_factors)
    print(f"策略名称：{quantum_strategy['strategy_name']}")
    print(f"策略适配得分：{quantum_strategy['strategy_score']:.2f}")
    print(f"优化目标：{quantum_strategy['optimize_target']}")
    
    # 4. 调整策略权重（模拟自主学习反馈）
    print("\n=== 自主调整策略权重 ===")
    adjust_result = strategy_module.adjust_strategy_weight("qubit_stability", 0.1)
    print(f"调整目标：{adjust_result['target']}")
    print(f"原权重：{adjust_result['old_weight']} → 新权重：{adjust_result['new_weight']}")
    
    # 5. 查看当前策略状态
    print("\n=== 当前策略状态 ===")
    print(strategy_module.get_current_strategy_status())