# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 工具封装与构建模块（一站式使用入口）
核心逻辑：整合所有核心模块，提供简洁API，支持一键运行/数据处理/结果可视化
设计原则：易用性优先、配置自动化、结果可解释，适配新手快速上手
"""
import configparser
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文显示
plt.rcParams["axes.unicode_minus"] = False

# 导入所有核心模块
from data_processing import DataProcessor
from goal_discovery import AutonomousGoalDiscovery
from umc_metabolism import MetabolicCycle
from umc_strategy import UMCStrategy
from umc_performance import PerformanceMonitor
from self_learning_feedback import SelfLearningFeedback
from auto_recovery import AutoRecovery
from signal_interpreter import SignalInterpreter

class UMCAgent:
    """UMC智能体工具类（一站式核心功能封装）"""
    def __init__(self, config_dir: str = "./"):
        """
        初始化UMC智能体
        :param config_dir: 配置文件目录（默认当前目录）
        """
        # 1. 自动初始化配置文件（无配置时生成默认配置）
        self._init_default_config(config_dir)
        # 2. 初始化核心模块
        self.data_processor = DataProcessor()
        self.goal_discoverer = AutonomousGoalDiscovery()
        self.metabolic_cycle = MetabolicCycle()
        self.strategy_module = UMCStrategy()
        self.perf_monitor = PerformanceMonitor()
        self.feedback_optimizer = SelfLearningFeedback()
        self.auto_recovery = AutoRecovery()
        self.signal_interpreter = SignalInterpreter()
        # 3. 初始化运行状态
        self.run_history = []
        self.current_summary = None

    def _init_default_config(self, config_dir: str) -> None:
        """自动生成默认配置文件（parameters.ini/paths.ini）"""
        # === 生成parameters.ini ===
        param_path = os.path.join(config_dir, "parameters.ini")
        if not os.path.exists(param_path):
            param_cfg = configparser.ConfigParser()
            # BASIC段
            param_cfg["BASIC"] = {
                "runtime_log_level": "DEBUG",
                "cycle_speed": "0.1",
                "data_cache_size": "100"
            }
            # METABOLISM段
            param_cfg["METABOLISM"] = {
                "core_factor_weight": "0.8",
                "energy_consumption_limit": "0.9",
                "stability_threshold": "0.8"
            }
            # STRATEGY段
            param_cfg["STRATEGY"] = {
                "qubit_stability": "0.8",
                "atomic_frequency": "0.7",
                "logistics_efficiency": "0.75",
                "unknown_domain": "0.6"
            }
            # VALIDATION段
            param_cfg["VALIDATION"] = {
                "blackbox_test_threshold": "0.7"
            }
            # AGI_L3段
            param_cfg["AGI_L3"] = {
                "goal_discovery_threshold": "0.5",
                "self_learning_feedback_rate": "0.5",
                "auto_recovery_fault_threshold": "3"
            }
            # 写入配置文件
            with open(param_path, "w", encoding="utf-8") as f:
                param_cfg.write(f)
        
        # === 生成paths.ini ===
        path_path = os.path.join(config_dir, "paths.ini")
        if not os.path.exists(path_path):
            path_cfg = configparser.ConfigParser()
            path_cfg["PATH"] = {
                "log_dir": "./logs",
                "backup_dir": "./backups",
                "processed_data_dir": "./processed_data",
                "result_dir": "./results"
            }
            # 写入配置文件
            with open(path_path, "w", encoding="utf-8") as f:
                path_cfg.write(f)
        
        # 创建必要目录
        path_cfg = configparser.ConfigParser()
        path_cfg.read(path_path, encoding="utf-8")
        for dir_name in path_cfg["PATH"].values():
            os.makedirs(dir_name, exist_ok=True)

    def load_data(self, data_path: str, domain_name: str = "unknown") -> pd.DataFrame:
        """
        工具函数：加载并标准化数据（支持CSV/Excel）
        :param data_path: 数据文件路径（.csv/.xlsx）
        :param domain_name: 数据领域名称
        :return: 标准化后的数据
        """
        # 1. 加载原始数据
        if data_path.endswith(".csv"):
            raw_data = pd.read_csv(data_path, encoding="utf-8")
        elif data_path.endswith(".xlsx"):
            raw_data = pd.read_excel(data_path)
        else:
            raise ValueError("仅支持CSV/Excel格式数据")
        
        # 2. 标准化数据（带故障恢复）
        def _process_data():
            return self.data_processor.standardize_data(raw_data, domain_name)
        
        standardized_data = self.auto_recovery.run_with_recovery(_process_data)
        print(f"✅ 数据加载完成：{data_path} | 形状：{standardized_data.shape} | 领域：{domain_name}")
        return standardized_data

    def run(self, data: pd.DataFrame, domain_name: str = "unknown") -> Dict[str, Any]:
        """
        核心工具：一键运行UMC智能体全流程
        :param data: 标准化后的数据
        :param domain_name: 数据领域名称
        :return: 全流程运行结果（含目标/策略/代谢/性能/优化）
        """
        # 包装核心运行逻辑（带故障恢复）
        def _core_run_logic():
            # 1. 自主发现目标
            goal_result = self.goal_discoverer.discover_goal(data)
            print(f"🎯 自主发现目标：{goal_result['goal']} | 优先级：{goal_result['priority']}")
            
            # 2. 选择最优策略
            # 先运行一次代谢循环获取因子
            mock_adapt_rules = {"factor_mapping": {"default": "stability"}}
            metabolic_pre = self.metabolic_cycle.run(data, goal_result["goal"], mock_adapt_rules)
            strategy_result = self.strategy_module.select_optimal_strategy(domain_name, metabolic_pre["core_factors"])
            print(f"📋 最优策略：{strategy_result['strategy_name']} | 得分：{strategy_result['strategy_score']:.2f}")
            
            # 3. 正式运行代谢循环
            metabolic_result = self.metabolic_cycle.run(data, goal_result["goal"], {"factor_mapping": strategy_result["factor_weight"]})
            print(f"🔄 代谢循环完成：稳定得分{metabolic_result['stability_score']:.2f} | 循环次数{metabolic_result['cycle_count']}")
            
            # 4. 性能校验
            perf_score = self.perf_monitor.score_result(metabolic_result, goal_result["goal"])
            print(f"📊 性能校验得分：{perf_score:.2f}")
            
            # 5. 自主学习反馈优化
            feedback_result = self.feedback_optimizer.feedback_optimize(data, metabolic_result)
            if feedback_result["optimize_status"] != "no_optimize":
                print(f"🔧 自主优化完成：调整{feedback_result['adjust_target']}权重{feedback_result['adjust_amount']:.3f}")
            else:
                print(f"🔧 无需优化：{feedback_result['reason']}")
            
            # 6. 构造运行结果
            run_result = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "domain_name": domain_name,
                "data_shape": data.shape,
                "goal_result": goal_result,
                "strategy_result": strategy_result,
                "metabolic_result": metabolic_result,
                "perf_score": perf_score,
                "feedback_result": feedback_result
            }
            
            # 7. 保存运行结果
            self.run_history.append(run_result)
            result_dir = "./results"
            result_path = os.path.join(result_dir, f"run_result_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(run_result, f, ensure_ascii=False, indent=2)
            
            # 8. 更新当前摘要
            self.current_summary = self.get_summary(run_result)
            return run_result
        
        # 执行核心逻辑（带故障恢复）
        run_result = self.auto_recovery.run_with_recovery(_core_run_logic)
        print(f"✅ UMC智能体运行完成！")
        return run_result

    def get_summary(self, run_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        工具函数：生成运行结果摘要（简化版，便于查看核心信息）
        :param run_result: 运行结果（默认取最新）
        :return: 结果摘要
        """
        if run_result is None:
            if not self.run_history:
                return {"message": "暂无运行记录"}
            run_result = self.run_history[-1]
        
        summary = {
            "运行时间": run_result["timestamp"],
            "数据领域": run_result["domain_name"],
            "数据规模": f"{run_result['data_shape'][0]}行 × {run_result['data_shape'][1]}列",
            "优化目标": run_result["goal_result"]["goal"],
            "最优策略": run_result["strategy_result"]["strategy_name"],
            "代谢稳定性": f"{run_result['metabolic_result']['stability_score']:.2f}",
            "性能得分": f"{run_result['perf_score']:.2f}",
            "优化状态": run_result["feedback_result"]["optimize_status"]
        }
        return summary

    def visualize_result(self, run_result: Optional[Dict[str, Any]] = None, save_fig: bool = True) -> None:
        """
        工具函数：可视化运行结果（核心因子分布+性能得分）
        :param run_result: 运行结果（默认取最新）
        :param save_fig: 是否保存图片
        """
        if run_result is None:
            if not self.run_history:
                raise ValueError("暂无运行记录，无法可视化")
            run_result = self.run_history[-1]
        
        # 创建2×1子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 子图1：代谢核心因子分布
        factors = run_result["metabolic_result"]["core_factors"]
        ax1.bar(factors.keys(), factors.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax1.set_title(f"代谢核心因子分布 | 领域：{run_result['domain_name']}", fontsize=12, fontweight="bold")
        ax1.set_ylabel("因子得分（0~1）")
        ax1.set_ylim(0, 1)
        # 添加数值标签
        for k, v in factors.items():
            ax1.text(k, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
        
        # 子图2：关键指标汇总
        metrics = {
            "性能得分": run_result["perf_score"],
            "代谢稳定性": run_result["metabolic_result"]["stability_score"],
            "策略得分": run_result["strategy_result"]["strategy_score"]
        }
        ax2.bar(metrics.keys(), metrics.values(), color=["#d62728", "#9467bd", "#8c564b"])
        ax2.set_title("关键指标得分", fontsize=12, fontweight="bold")
        ax2.set_ylabel("得分（0~1）")
        ax2.set_ylim(0, 1)
        # 添加数值标签
        for k, v in metrics.items():
            ax2.text(k, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
        
        # 整体布局
        plt.tight_layout()
        
        # 保存图片
        if save_fig:
            fig_path = os.path.join("./results", f"visual_result_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"📸 可视化结果已保存：{fig_path}")
        
        # 显示图片
        plt.show()

# 工具函数：快速创建测试数据
def create_test_data(domain_name: str = "quantum", sample_count: int = 100) -> pd.DataFrame:
    """
    快速创建测试数据（适配不同领域）
    :param domain_name: 领域名称（quantum/atomic/logistics）
    :param sample_count: 样本数量
    :return: 测试数据
    """
    np.random.seed(42)  # 固定随机种子，保证结果可复现
    if domain_name == "quantum":
        data = pd.DataFrame({
            "qubit_stability": np.random.rand(sample_count) * 0.9,
            "energy_consumption": np.random.rand(sample_count) * 0.8,
            "matter_output": np.random.rand(sample_count) * 0.7
        })
    elif domain_name == "atomic":
        data = pd.DataFrame({
            "atomic_frequency": np.random.rand(sample_count) * 0.9,
            "energy_efficiency": np.random.rand(sample_count) * 0.8,
            "particle_yield": np.random.rand(sample_count) * 0.7
        })
    elif domain_name == "logistics":
        data = pd.DataFrame({
            "logistics_efficiency": np.random.rand(sample_count) * 0.9,
            "transport_cost": np.random.rand(sample_count) * 0.8,
            "delivery_speed": np.random.rand(sample_count) * 0.7
        })
    else:
        data = pd.DataFrame({
            "feature_1": np.random.rand(sample_count) * 0.9,
            "feature_2": np.random.rand(sample_count) * 0.8,
            "feature_3": np.random.rand(sample_count) * 0.7
        })
    
    # 人为添加少量缺失值（模拟真实数据）
    for col in data.columns:
        data.loc[np.random.choice(sample_count, size=int(sample_count*0.05)), col] = np.nan
    
    # 保存测试数据
    data_path = f"./test_data_{domain_name}.csv"
    data.to_csv(data_path, index=False, encoding="utf-8")
    print(f"📄 测试数据已生成：{data_path}")
    return data

# 验证入口（一站式测试UMC智能体）
if __name__ == "__main__":
    # 1. 初始化UMC智能体（自动生成配置）
    umc_agent = UMCAgent()
    print("🚀 UMC智能体初始化完成！")

    # 2. 创建测试数据（量子领域）
    test_data = create_test_data(domain_name="quantum", sample_count=200)

    # 3. 加载并标准化数据（封装故障恢复）
    standardized_data = umc_agent.load_data("./test_data_quantum.csv", domain_name="quantum")

    # 4. 一键运行全流程
    run_result = umc_agent.run(standardized_data, domain_name="quantum")

    # 5. 查看运行结果摘要
    print("\n=== 运行结果摘要 ===")
    summary = umc_agent.get_summary(run_result)
    for k, v in summary.items():
        print(f"{k}：{v}")

    # 6. 可视化运行结果
    umc_agent.visualize_result(run_result)

    # 7. 查看故障/优化历史
    print("\n=== 故障摘要 ===")
    print(umc_agent.auto_recovery.get_fault_summary())
    print("\n=== 优化历史 ===")
    print(umc_agent.feedback_optimizer.get_feedback_history())