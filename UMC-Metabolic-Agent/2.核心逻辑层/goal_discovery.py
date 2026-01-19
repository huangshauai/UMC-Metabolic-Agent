# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 自主目标发现模块（AGI L3核心）
核心逻辑：基于数据特征重要性（方差/互信息）无监督识别优化目标，无需人工设定
设计原则：目标发现可配置、结果可解释、与性能监控/代谢循环深度联动，无冗余逻辑
"""
import configparser
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

# 联动性能监控模块（记录目标发现日志，支撑可解释性）
from umc_performance import PerformanceMonitor

class AutonomousGoalDiscovery:
    """自主目标发现核心类（AGI L3无监督目标识别）"""
    def __init__(self):
        # 1. 加载核心配置（复用parameters.ini的AGI_L3段）
        self.param_cfg = self._load_config("parameters.ini")
        # 2. 初始化性能监控（记录目标发现日志）
        self.perf_monitor = PerformanceMonitor()
        # 3. 定义目标模板（可扩展，适配不同领域）
        self.goal_templates = {
            "information": "提升{feature}的信息维度价值（信息因子优化）",
            "energy": "提升{feature}的能量利用效率（能量因子优化）",
            "matter": "提升{feature}的物质产出能力（物质因子优化）",
            "stability": "提升{feature}的代谢稳定性（稳定性优化）"
        }
        # 4. 初始化目标发现状态（支撑可解释性）
        self.goal_history = []
        self.current_goal = None
        self.current_goal_basis = None

    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """统一加载参数配置（读取目标发现阈值）"""
        cfg = configparser.ConfigParser()
        cfg.read(config_file, encoding="utf-8")
        return cfg

    def _calculate_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        核心：计算数据特征重要性（融合方差+互信息，无监督）
        方差：衡量特征离散度（离散度越高，优化价值越大）
        互信息：衡量特征与整体代谢的关联度（关联度越高，优先级越高）
        """
        feature_importance = {}
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

        # 1. 计算各特征的方差（基础重要性）
        var_scores = data[numeric_cols].var().to_dict()
        # 归一化方差得分（0~1）
        max_var = max(var_scores.values()) if var_scores else 1.0
        norm_var_scores = {k: v / max_var for k, v in var_scores.items()}

        # 2. 计算各特征与整体的互信息（增强重要性）
        # 构建整体代谢指标（所有特征的均值）
        data["metabolism_core"] = data[numeric_cols].mean(axis=1)
        mi_scores = {}
        for col in numeric_cols:
            # 离散化特征（互信息计算要求）
            col_bins = pd.cut(data[col], bins=10, labels=False, duplicates="drop")
            core_bins = pd.cut(data["metabolism_core"], bins=10, labels=False, duplicates="drop")
            # 计算互信息（值越大，关联度越高）
            mi = self._mutual_information(col_bins, core_bins)
            mi_scores[col] = mi

        # 归一化互信息得分（0~1）
        max_mi = max(mi_scores.values()) if mi_scores else 1.0
        norm_mi_scores = {k: v / max_mi for k, v in mi_scores.items()}

        # 3. 融合方差+互信息（权重各50%）
        for col in numeric_cols:
            feature_importance[col] = (norm_var_scores[col] + norm_mi_scores[col]) / 2

        # 删除临时列
        data.drop("metabolism_core", axis=1, inplace=True)
        return feature_importance

    def _mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """计算两个离散序列的互信息（简化版，适配无监督场景）"""
        # 计算联合概率分布
        joint_prob = pd.crosstab(x, y, normalize=True).values
        # 计算边缘概率分布
        x_prob = pd.Series(x).value_counts(normalize=True).values
        y_prob = pd.Series(y).value_counts(normalize=True).values

        # 计算互信息
        mi = 0.0
        for i in range(len(x_prob)):
            for j in range(len(y_prob)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
        return mi

    def _map_importance_to_factor(self, feature: str, data: pd.DataFrame) -> str:
        """将高重要性特征映射到代谢因子维度（信息/能量/物质）"""
        # 计算特征与各代谢因子的相关性（简化版：基于特征名关键词+数值分布）
        if "info" in feature.lower() or "特征" in feature or "目标" in feature:
            return "information"
        elif "energy" in feature.lower() or "能量" in feature or "效率" in feature:
            return "energy"
        elif "matter" in feature.lower() or "物质" in feature or "产出" in feature:
            return "matter"
        else:
            # 无关键词时，基于数值分布判断（离散度高→信息，均值高→能量，方差小→物质）
            feature_mean = data[feature].mean()
            feature_std = data[feature].std()
            if feature_std > 0.3:
                return "information"
            elif feature_mean > 0.5:
                return "energy"
            else:
                return "stability"

    def discover_goal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        AGI L3核心：自主发现优化目标（无监督，无需人工输入）
        :param data: 标准化后的数据（来自data_processing.py）
        :return: 目标发现结果（目标文本、优先级、依据、因子维度）
        """
        # 1. 获取目标发现阈值（来自parameters.ini）
        discovery_threshold = float(self.param_cfg["AGI_L3"]["goal_discovery_threshold"])

        # 2. 计算特征重要性
        feature_importance = self._calculate_feature_importance(data)
        if not feature_importance:
            # 无有效特征时，使用默认目标
            default_goal = "提升整体代谢循环稳定性（默认目标）"
            goal_result = {
                "goal": default_goal,
                "priority": 1,
                "basis": "无有效数值特征，启用默认目标",
                "factor_dimension": "stability",
                "status": "default"
            }
            self.current_goal = default_goal
            self.current_goal_basis = goal_result["basis"]
            self.goal_history.append(goal_result)
            # 记录目标发现日志
            self.perf_monitor.log_goal_discovery(default_goal, 1, {"reason": "无有效特征"})
            return goal_result

        # 3. 筛选高重要性特征（超过阈值）
        high_importance_features = {k: v for k, v in feature_importance.items() if v >= discovery_threshold}
        if not high_importance_features:
            # 无特征超过阈值，选重要性最高的特征
            top_feature = max(feature_importance.items(), key=lambda x: x[1])
            high_importance_features = {top_feature[0]: top_feature[1]}

        # 4. 选择最优特征（重要性最高）
        top_feature, top_importance = max(high_importance_features.items(), key=lambda x: x[1])

        # 5. 映射到代谢因子维度，生成目标文本
        factor_dim = self._map_importance_to_factor(top_feature, data)
        goal_text = self.goal_templates[factor_dim].format(feature=top_feature)

        # 6. 计算目标优先级（1~5，重要性越高优先级越高）
        priority = min(int(np.ceil(top_importance * 5)), 5)

        # 7. 构造目标发现结果（可解释性）
        goal_result = {
            "goal": goal_text,
            "priority": priority,
            "basis": f"特征{top_feature}的重要性得分{top_importance:.2f}（阈值{discovery_threshold}），映射到{factor_dim}维度",
            "factor_dimension": factor_dim,
            "feature_importance": feature_importance,
            "status": "success"
        }

        # 8. 更新状态，记录日志
        self.current_goal = goal_text
        self.current_goal_basis = goal_result["basis"]
        self.goal_history.append(goal_result)
        # 写入性能监控日志（支撑可解释性）
        self.perf_monitor.log_goal_discovery(goal_text, priority, {"feature": top_feature, "importance": top_importance})

        return goal_result

    def get_goal_history(self, last_n: int = 5) -> list[Dict[str, Any]]:
        """获取最近n条目标发现历史（支撑可视化/可解释性）"""
        return self.goal_history[-last_n:] if self.goal_history else []

# 自主目标发现验证入口（无冗余，快速测试）
if __name__ == "__main__":
    # 1. 初始化目标发现模块
    goal_discoverer = AutonomousGoalDiscovery()
    print("UMC自主目标发现模块初始化完成")

    # 2. 加载/模拟标准化数据（来自data_processing.py）
    # 模拟量子领域标准化数据（0~1范围，含缺失值处理后的数据）
    test_data = pd.DataFrame({
        "qubit_stability": np.random.rand(100) * 0.9,
        "energy_consumption": np.random.rand(100) * 0.8,
        "matter_output": np.random.rand(100) * 0.7
    })
    # 人为提升qubit_stability的重要性（增大方差）
    test_data["qubit_stability"] = test_data["qubit_stability"] * np.random.choice([0.1, 0.9], size=100)

    print(f"\n=== 输入标准化数据 ===")
    print(f"数据形状：{test_data.shape}")
    print(f"特征均值：\n{test_data.mean()}")

    # 3. 自主发现目标
    goal_result = goal_discoverer.discover_goal(test_data)
    print(f"\n=== 自主目标发现结果 ===")
    print(f"优化目标：{goal_result['goal']}")
    print(f"目标优先级：{goal_result['priority']}")
    print(f"发现依据：{goal_result['basis']}")
    print(f"代谢因子维度：{goal_result['factor_dimension']}")

    # 4. 查看目标发现历史
    print(f"\n=== 目标发现历史 ===")
    print(goal_discoverer.get_goal_history())