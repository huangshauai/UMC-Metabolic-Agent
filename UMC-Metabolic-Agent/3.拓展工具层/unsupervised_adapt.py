# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 无监督自适应模块（领域自主识别+参数自动调优+零标注适配）
核心逻辑：无需人工标注/参数调整，自主识别数据领域、提取特征、适配代谢循环/策略权重
设计原则：无监督、自驱动、领域无关、效果可评估，适配新手零配置使用多领域数据
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# 导入核心工具
from tool_build import UMCAgent
from tool_config import ConfigManager

class UnsupervisedAdaptor:
    """无监督自适应器（核心功能：领域识别、特征提取、参数自适应、效果评估）"""
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        初始化无监督自适应器
        :param config_manager: 配置管理器实例（默认自动初始化）
        """
        # 初始化配置管理器
        self.config_manager = config_manager if config_manager else ConfigManager()
        # 初始化领域特征库（内置量子/原子/物流领域的基准特征，用于匹配）
        self.domain_feature_lib = self._init_domain_feature_lib()
        # 初始化自适应状态
        self.adapt_history = []
        self.current_domain = "unknown"  # 当前识别的领域
        self.current_adapt_params = {}   # 当前自适应调整的参数
        self.current_feature = None      # 当前数据的核心特征

    def _init_domain_feature_lib(self) -> Dict[str, Any]:
        """初始化内置领域特征库（基准特征，用于无监督匹配）"""
        # 每个领域的基准特征：均值、方差、特征相关性、主成分占比（基于大量样本统计）
        domain_feature_lib = {
            "quantum": {
                "desc": "量子领域数据（qubit稳定性、能耗、物质输出）",
                "feature_cols": ["qubit_stability", "energy_consumption", "matter_output"],
                "mean": [0.45, 0.4, 0.35],          # 各特征均值
                "std": [0.2, 0.18, 0.15],           # 各特征方差
                "corr_matrix": [[1.0, -0.7, 0.6],   # 特征相关性矩阵
                                [-0.7, 1.0, -0.5],
                                [0.6, -0.5, 1.0]],
                "pca_var_ratio": [0.75, 0.15, 0.1]  # 主成分方差占比
            },
            "atomic": {
                "desc": "原子领域数据（原子频率、能效、粒子产率）",
                "feature_cols": ["atomic_frequency", "energy_efficiency", "particle_yield"],
                "mean": [0.5, 0.42, 0.38],
                "std": [0.22, 0.19, 0.17],
                "corr_matrix": [[1.0, -0.65, 0.55],
                                [-0.65, 1.0, -0.45],
                                [0.55, -0.45, 1.0]],
                "pca_var_ratio": [0.7, 0.2, 0.1]
            },
            "logistics": {
                "desc": "物流领域数据（物流效率、运输成本、配送速度）",
                "feature_cols": ["logistics_efficiency", "transport_cost", "delivery_speed"],
                "mean": [0.48, 0.45, 0.4],
                "std": [0.18, 0.2, 0.16],
                "corr_matrix": [[1.0, -0.8, 0.7],
                                [-0.8, 1.0, -0.6],
                                [0.7, -0.6, 1.0]],
                "pca_var_ratio": [0.8, 0.12, 0.08]
            }
        }
        # 保存领域特征库到本地（便于扩展）
        lib_path = "./domain_feature_lib.json"
        if not os.path.exists(lib_path):
            with open(lib_path, "w", encoding="utf-8") as f:
                json.dump(domain_feature_lib, f, ensure_ascii=False, indent=2)
        return domain_feature_lib

    def extract_unsupervised_feature(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        无监督特征提取（核心：提取数据的核心统计特征，用于领域匹配）
        :param data: 原始/标准化数据（支持任意数值型数据）
        :return: 数据的核心无监督特征
        """
        print("\n🔍 开始无监督特征提取...")
        # 1. 数据预处理（去重、填充缺失值、标准化）
        data_clean = data.copy().drop_duplicates()
        data_clean = data_clean.fillna(data_clean.mean())
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)
        data_scaled_df = pd.DataFrame(data_scaled, columns=data_clean.columns)

        # 2. 提取基础统计特征
        feature = {
            "feature_cols": data_clean.columns.tolist(),
            "sample_count": len(data_clean),
            "mean": data_scaled.mean(axis=0).tolist(),
            "std": data_scaled.std(axis=0).tolist(),
            "corr_matrix": data_scaled_df.corr().values.tolist(),  # 相关性矩阵
            "pca_var_ratio": []  # 主成分方差占比
        }

        # 3. 提取PCA特征（主成分分析）
        pca_n = min(3, len(data_clean.columns))  # 最多取前3个主成分
        pca = PCA(n_components=pca_n)
        pca.fit(data_scaled)
        feature["pca_var_ratio"] = pca.explained_variance_ratio_.tolist()

        # 4. 补充分布特征（正态性检验）
        feature["normality_pvalue"] = [stats.shapiro(data_scaled[:, i])[1] for i in range(min(3, data_scaled.shape[1]))]

        # 保存当前特征
        self.current_feature = feature
        print(f"✅ 特征提取完成：样本数{feature['sample_count']} | 特征列{feature['feature_cols']}")
        return feature

    def match_domain(self, feature: Dict[str, Any]) -> Tuple[str, float]:
        """
        无监督领域匹配（核心：对比特征库，识别数据所属领域）
        :param feature: 数据的无监督特征
        :return: (匹配的领域名称, 匹配相似度[0~1])
        """
        print("\n🎯 开始无监督领域匹配...")
        similarity_scores = {}

        # 遍历领域特征库，计算相似度
        for domain, domain_feature in self.domain_feature_lib.items():
            # 1. 均值相似度（余弦相似度）
            mean_sim = 1 - pairwise_distances([feature["mean"]], [domain_feature["mean"]], metric="cosine")[0][0]
            # 2. 方差相似度（余弦相似度）
            std_sim = 1 - pairwise_distances([feature["std"]], [domain_feature["std"]], metric="cosine")[0][0]
            # 3. 相关性矩阵相似度（平均余弦相似度）
            corr_sim = 1 - pairwise_distances(feature["corr_matrix"], domain_feature["corr_matrix"], metric="cosine").mean()
            # 4. PCA方差占比相似度（余弦相似度）
            # 对齐PCA维度（不足补0）
            pca_self = feature["pca_var_ratio"] + [0]*(3-len(feature["pca_var_ratio"]))
            pca_domain = domain_feature["pca_var_ratio"] + [0]*(3-len(domain_feature["pca_var_ratio"]))
            pca_sim = 1 - pairwise_distances([pca_self], [pca_domain], metric="cosine")[0][0]

            # 综合相似度（加权平均）
            total_sim = (mean_sim * 0.3) + (std_sim * 0.2) + (corr_sim * 0.3) + (pca_sim * 0.2)
            similarity_scores[domain] = max(0, min(1, total_sim))  # 限制在0~1之间

        # 确定匹配领域（相似度最高且≥阈值0.5，否则为unknown）
        max_sim_domain = max(similarity_scores.items(), key=lambda x: x[1])
        match_domain = max_sim_domain[0] if max_sim_domain[1] >= 0.5 else "unknown"
        match_sim = max_sim_domain[1] if match_domain != "unknown" else 0.0

        # 打印匹配结果
        print("领域匹配得分：")
        for domain, score in similarity_scores.items():
            print(f"  - {domain}：{score:.3f}")
        print(f"✅ 匹配结果：{match_domain}（相似度：{match_sim:.3f}）")

        # 更新当前领域
        self.current_domain = match_domain
        return match_domain, match_sim

    def adapt_params(self, domain: str, feature: Dict[str, Any]) -> Dict[str, Any]:
        """
        无监督参数自适应调整（核心：根据领域自动调整代谢/策略参数）
        :param domain: 匹配的领域
        :param feature: 数据的无监督特征
        :return: 自适应调整后的参数
        """
        print(f"\n⚙️  开始{domain}领域参数自适应调整...")
        # 备份当前配置（调整前）
        self.config_manager.backup_config(backup_name=f"pre_adapt_{domain}")

        # 初始化自适应参数（基于领域和特征）
        adapt_params = {
            "domain": domain,
            "adapt_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metabolism_params": {},
            "strategy_params": {},
            "agi_l3_params": {}
        }

        # 1. 领域专属参数调整
        if domain == "quantum":
            # 量子领域：提高稳定性权重，降低循环速度
            adapt_params["metabolism_params"] = {
                "core_factor_weight": min(1.0, float(self.config_manager.param_cfg["METABOLISM"]["core_factor_weight"]) * 1.1),
                "stability_threshold": min(1.0, float(self.config_manager.param_cfg["METABOLISM"]["stability_threshold"]) * 1.05),
                "cycle_speed": max(0.01, float(self.config_manager.param_cfg["BASIC"]["cycle_speed"]) * 0.9)
            }
            adapt_params["strategy_params"] = {
                "qubit_stability": 0.9,
                "atomic_frequency": 0.5,
                "logistics_efficiency": 0.5,
                "unknown_domain": 0.5
            }
            adapt_params["agi_l3_params"] = {
                "goal_discovery_threshold": max(0.1, float(self.config_manager.param_cfg["AGI_L3"]["goal_discovery_threshold"]) * 0.9)
            }

        elif domain == "atomic":
            # 原子领域：提高能耗上限，调整目标发现阈值
            adapt_params["metabolism_params"] = {
                "energy_consumption_limit": min(1.0, float(self.config_manager.param_cfg["METABOLISM"]["energy_consumption_limit"]) * 1.05),
                "core_factor_weight": min(1.0, float(self.config_manager.param_cfg["METABOLISM"]["core_factor_weight"]) * 1.05)
            }
            adapt_params["strategy_params"] = {
                "qubit_stability": 0.5,
                "atomic_frequency": 0.9,
                "logistics_efficiency": 0.5,
                "unknown_domain": 0.5
            }
            adapt_params["agi_l3_params"] = {
                "self_learning_feedback_rate": min(1.0, float(self.config_manager.param_cfg["AGI_L3"]["self_learning_feedback_rate"]) * 1.1)
            }

        elif domain == "logistics":
            # 物流领域：提高循环速度，降低稳定性阈值
            adapt_params["metabolism_params"] = {
                "cycle_speed": min(1.0, float(self.config_manager.param_cfg["BASIC"]["cycle_speed"]) * 1.1),
                "stability_threshold": max(0.5, float(self.config_manager.param_cfg["METABOLISM"]["stability_threshold"]) * 0.95)
            }
            adapt_params["strategy_params"] = {
                "qubit_stability": 0.5,
                "atomic_frequency": 0.5,
                "logistics_efficiency": 0.9,
                "unknown_domain": 0.5
            }
            adapt_params["agi_l3_params"] = {
                "auto_recovery_fault_threshold": min(10, int(self.config_manager.param_cfg["AGI_L3"]["auto_recovery_fault_threshold"]) + 1)
            }

        else:
            # 未知领域：保守调整，提高容错性
            adapt_params["metabolism_params"] = {
                "core_factor_weight": 0.7,
                "stability_threshold": 0.75,
                "cycle_speed": 0.05
            }
            adapt_params["strategy_params"] = {
                "unknown_domain": 0.8
            }
            adapt_params["agi_l3_params"] = {
                "auto_recovery_fault_threshold": 2,
                "goal_discovery_threshold": 0.4
            }

        # 2. 基于数据特征的动态调整（补充领域通用调整）
        # 根据样本量调整缓存大小
        sample_count = feature["sample_count"]
        adapt_params["metabolism_params"]["data_cache_size"] = min(1000, max(10, int(sample_count * 0.1)))

        # 3. 应用参数调整到配置文件
        # 更新代谢/基础参数
        for param, value in adapt_params["metabolism_params"].items():
            if param in self.config_manager.param_cfg["METABOLISM"]:
                self.config_manager.param_cfg["METABOLISM"][param] = str(value)
            elif param in self.config_manager.param_cfg["BASIC"]:
                self.config_manager.param_cfg["BASIC"][param] = str(value)
        # 更新策略参数
        for param, value in adapt_params["strategy_params"].items():
            if param in self.config_manager.param_cfg["STRATEGY"]:
                self.config_manager.param_cfg["STRATEGY"][param] = str(value)
        # 更新AGI_L3参数
        for param, value in adapt_params["agi_l3_params"].items():
            if param in self.config_manager.param_cfg["AGI_L3"]:
                self.config_manager.param_cfg["AGI_L3"][param] = str(value)

        # 保存配置
        self.config_manager._save_param_config()
        print("✅ 参数自适应调整完成，调整项：")
        for param_type, params in adapt_params.items():
            if param_type in ["metabolism_params", "strategy_params", "agi_l3_params"]:
                for k, v in params.items():
                    print(f"  - {param_type}.{k}：{v}")

        # 更新当前自适应参数
        self.current_adapt_params = adapt_params
        return adapt_params

    def evaluate_adapt_effect(self, umc_agent: UMCAgent, data: pd.DataFrame) -> Dict[str, float]:
        """
        无监督自适应效果评估（核心：评估调整后智能体的运行效果）
        :param umc_agent: 自适应后的UMCAgent实例
        :param data: 测试数据
        :return: 效果评估指标（稳定性、一致性、效率）
        """
        print("\n📊 开始无监督自适应效果评估...")
        # 运行智能体获取结果
        run_result = umc_agent.run(data, domain_name=self.current_domain)

        # 提取评估指标（无监督，无需标注）
        metrics = {
            # 1. 代谢稳定性（核心指标：越高越好）
            "metabolic_stability": run_result["metabolic_result"]["stability_score"],
            # 2. 结果一致性（多次运行的稳定性得分方差：越低越好，转换为0~1）
            "result_consistency": self._calculate_consistency(umc_agent, data),
            # 3. 运行效率（循环次数/样本数：越低越好，转换为0~1）
            "run_efficiency": max(0, 1 - (run_result["metabolic_result"]["cycle_count"] / len(data))),
            # 4. 性能达标率（相对阈值：越高越好）
            "performance_rate": run_result["perf_score"] / float(self.config_manager.param_cfg["VALIDATION"]["blackbox_test_threshold"])
        }
        # 归一化所有指标到0~1
        metrics = {k: max(0, min(1, v)) for k, v in metrics.items()}
        # 综合效果得分（加权平均）
        metrics["comprehensive_score"] = (
            metrics["metabolic_stability"] * 0.4 +
            metrics["result_consistency"] * 0.2 +
            metrics["run_efficiency"] * 0.2 +
            metrics["performance_rate"] * 0.2
        )

        # 打印评估结果
        print("✅ 自适应效果评估结果：")
        for metric, score in metrics.items():
            print(f"  - {metric}：{score:.3f}")
        print(f"  - 综合效果得分：{metrics['comprehensive_score']:.3f}（≥0.7为优秀）")

        return metrics

    def _calculate_consistency(self, umc_agent: UMCAgent, data: pd.DataFrame, run_times: int = 3) -> float:
        """计算多次运行的结果一致性（无监督指标）"""
        stability_scores = []
        for i in range(run_times):
            result = umc_agent.run(data, domain_name=self.current_domain)
            stability_scores.append(result["metabolic_result"]["stability_score"])
        # 计算方差，转换为一致性得分（方差越小，一致性越高）
        var = np.var(stability_scores)
        consistency = max(0, 1 - min(var * 10, 1))  # 方差*10后限制在0~1，取反
        return consistency

    def run_full_adapt(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        一键运行全流程无监督自适应（特征提取→领域匹配→参数调整→效果评估）
        :param data: 任意领域的数值型数据
        :return: 自适应全流程结果
        """
        print("🚀 开始UMC智能体无监督自适应全流程...")
        start_time = time.time()

        # 1. 无监督特征提取
        feature = self.extract_unsupervised_feature(data)

        # 2. 无监督领域匹配
        domain, similarity = self.match_domain(feature)

        # 3. 参数自适应调整
        adapt_params = self.adapt_params(domain, feature)

        # 4. 初始化自适应后的智能体
        umc_agent_adapted = UMCAgent()  # 自动加载调整后的配置

        # 5. 自适应效果评估
        adapt_effect = self.evaluate_adapt_effect(umc_agent_adapted, data)

        # 6. 汇总自适应结果
        full_result = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": f"{time.time() - start_time:.2f}s",
            "data_info": {"sample_count": len(data), "feature_cols": data.columns.tolist()},
            "domain_match": {"domain": domain, "similarity": similarity},
            "adapt_params": adapt_params,
            "adapt_effect": adapt_effect,
            "is_adapt_successful": adapt_effect["comprehensive_score"] >= 0.6  # ≥0.6为成功
        }

        # 记录自适应历史
        self.adapt_history.append(full_result)

        # 保存自适应结果到本地
        result_dir = "./adapt_results"
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"adapt_result_{domain}_{time.strftime('%Y%m%d%H%M%S')}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(full_result, f, ensure_ascii=False, indent=2)

        # 打印最终结果
        print("\n🏁 无监督自适应全流程完成！")
        print(f"  - 数据规模：{len(data)}行 × {len(data.columns)}列")
        print(f"  - 识别领域：{domain}（相似度{similarity:.3f}）")
        print(f"  - 综合效果：{adapt_effect['comprehensive_score']:.3f}")
        print(f"  - 自适应成功：{full_result['is_adapt_successful']}")
        print(f"  - 结果保存：{result_path}")

        return full_result

    def add_custom_domain(self, domain_name: str, domain_data: pd.DataFrame, domain_desc: str = "") -> None:
        """
        扩展自定义领域到特征库（无监督，无需标注）
        :param domain_name: 自定义领域名称
        :param domain_data: 自定义领域的样本数据
        :param domain_desc: 领域描述
        """
        print(f"\n🆕 开始扩展自定义领域：{domain_name}...")
        # 提取自定义领域的特征
        domain_feature = self.extract_unsupervised_feature(domain_data)
        # 构建自定义领域特征（简化版，保留核心）
        custom_domain_feature = {
            "desc": domain_desc if domain_desc else f"自定义领域：{domain_name}",
            "feature_cols": domain_feature["feature_cols"],
            "mean": domain_feature["mean"],
            "std": domain_feature["std"],
            "corr_matrix": domain_feature["corr_matrix"],
            "pca_var_ratio": domain_feature["pca_var_ratio"]
        }
        # 添加到领域特征库
        self.domain_feature_lib[domain_name] = custom_domain_feature
        # 保存更新后的特征库
        with open("./domain_feature_lib.json", "w", encoding="utf-8") as f:
            json.dump(self.domain_feature_lib, f, ensure_ascii=False, indent=2)
        print(f"✅ 自定义领域{domain_name}已添加到特征库，支持自动匹配！")

# 无监督自适应模块验证入口（一站式测试）
if __name__ == "__main__":
    # 1. 初始化无监督自适应器
    adaptor = UnsupervisedAdaptor()
    print("🚀 无监督自适应器初始化完成！")

    # 2. 生成不同领域的测试数据（量子/原子/物流/自定义）
    from tool_build import create_test_data
    # 测试1：量子领域数据
    print("\n=== 测试1：量子领域数据自适应 ===")
    quantum_data = create_test_data(domain_name="quantum", sample_count=200)
    quantum_adapt_result = adaptor.run_full_adapt(quantum_data)

    # 测试2：自定义领域数据（扩展+自适应）
    print("\n=== 测试2：自定义领域数据自适应 ===")
    # 生成自定义数据（比如金融领域）
    custom_data = pd.DataFrame({
        "risk_score": np.random.rand(150) * 0.9,
        "return_rate": np.random.rand(150) * 0.8,
        "liquidity": np.random.rand(150) * 0.7
    })
    # 扩展自定义领域
    adaptor.add_custom_domain("finance", custom_data, "金融领域数据（风险评分、收益率、流动性）")
    # 自适应自定义领域数据
    custom_adapt_result = adaptor.run_full_adapt(custom_data)

    # 3. 查看自适应历史
    print("\n=== 自适应历史汇总 ===")
    print(f"自适应次数：{len(adaptor.adapt_history)}")
    for idx, history in enumerate(adaptor.adapt_history):
        print(f"  {idx+1}. 领域：{history['domain_match']['domain']} | 综合得分：{history['adapt_effect']['comprehensive_score']:.3f}")

    print("\n🎉 无监督自适应模块测试完成！所有结果已保存至 ./adapt_results")