# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 性能监控与闭环验证模块（AGI L3核心）
核心逻辑：量化评估代谢循环/策略执行结果，记录性能/错误日志，支撑闭环验证与故障恢复
设计原则：评分规则可配置、日志标准化、与所有核心模块联动，无冗余监控逻辑
"""
import configparser
import os
import time
import json
from typing import Dict, Any, Optional
import pandas as pd

class PerformanceMonitor:
    """性能监控与闭环验证核心类（AGI L3闭环的关键）"""
    def __init__(self):
        # 1. 加载配置（参数+路径，复用统一配置文件）
        self.param_cfg = self._load_config("parameters.ini")
        self.path_cfg = self._load_config("paths.ini")
        # 2. 初始化监控状态（支撑可解释性/故障恢复）
        self.performance_history = []  # 性能评分历史
        self.error_history = []        # 错误日志历史
        self.current_performance_score = 0.0
        self.error_count = 0  # 连续错误计数，支撑auto_recovery的故障阈值
        # 3. 初始化日志目录（自动创建，无需手动操作）
        self._init_log_dir()

    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """统一加载参数/路径配置"""
        cfg = configparser.ConfigParser()
        cfg.read(config_file, encoding="utf-8")
        return cfg

    def _init_log_dir(self) -> None:
        """创建日志目录（从paths.ini读取，跨系统兼容）"""
        log_dir = self.path_cfg["PATH"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        # 初始化性能日志文件
        self.perf_log_path = os.path.join(log_dir, "performance.log")
        self.error_log_path = os.path.join(log_dir, "error.log")

    def score_result(self, metabolic_result: Dict[str, Any], goal: str) -> float:
        """
        AGI L3核心：量化评分代谢循环执行结果（闭环验证的核心依据）
        :param metabolic_result: umc_metabolism.py输出的代谢循环结果
        :param goal: goal_discovery.py自主发现的优化目标
        :return: 归一化性能得分（0~1），低于阈值触发优化/恢复
        """
        # 1. 提取核心评分维度
        stability_score = metabolic_result["stability_score"]  # 代谢稳定性
        is_stable = metabolic_result["is_stable"]              # 是否稳定
        cycle_count = metabolic_result["cycle_count"]          # 循环次数（越少越好）
        target_factor = self._get_target_factor(metabolic_result["core_factors"], goal)  # 目标相关因子

        # 2. 计算基础得分（加权融合各维度）
        stability_weight = float(self.param_cfg["METABOLISM"]["core_factor_weight"])
        target_weight = 0.4  # 目标因子权重（固定，保证核心目标优先级）
        cycle_weight = 0.1   # 循环次数权重（越少得分越高）

        base_score = (
            stability_score * stability_weight +
            target_factor * target_weight +
            (1 / (cycle_count + 1)) * cycle_weight  # 循环次数越少，得分越高
        )

        # 3. 应用验证阈值（来自parameters.ini）
        blackbox_threshold = float(self.param_cfg["VALIDATION"]["blackbox_test_threshold"])
        final_score = base_score if is_stable else base_score * 0.8  # 不稳定则扣分

        # 4. 归一化得分（0~1），避免超出范围
        final_score = max(0.0, min(final_score, 1.0))

        # 5. 更新监控状态（支撑自主反馈/故障恢复）
        self.current_performance_score = final_score
        self.performance_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "stability_score": stability_score,
            "final_score": final_score,
            "is_stable": is_stable,
            "cycle_count": cycle_count
        })

        # 6. 记录性能日志（支撑可解释性）
        self.log_performance(final_score, goal, is_stable)

        # 7. 低于阈值则增加错误计数（支撑故障恢复）
        if final_score < blackbox_threshold:
            self.error_count += 1
        else:
            self.error_count = 0  # 达标则重置错误计数

        return final_score

    def _get_target_factor(self, core_factors: Dict[str, float], goal: str) -> float:
        """提取目标相关的核心因子值（与自主目标发现联动）"""
        if "信息" in goal:
            return core_factors["information"]
        elif "能量" in goal:
            return core_factors["energy"]
        elif "物质" in goal:
            return core_factors["matter"]
        else:
            return (core_factors["information"] + core_factors["energy"] + core_factors["matter"]) / 3

    def log_performance(self, score: float, goal: str, is_stable: bool) -> None:
        """记录性能日志（标准化格式，支撑白盒调试）"""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance_score": score,
            "goal": goal,
            "is_stable": is_stable,
            "error_count": self.error_count
        }
        # 追加写入日志文件（JSON格式，便于解析）
        with open(self.perf_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_error(self, error_msg: str) -> None:
        """记录错误日志（支撑故障恢复/可解释性）"""
        error_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_msg": error_msg,
            "error_count": self.error_count,
            "current_performance_score": self.current_performance_score
        }
        with open(self.error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
        # 更新错误历史
        self.error_history.append(error_entry)

    def log_goal_discovery(self, goal: str, priority: int, basis: Dict[str, float]) -> None:
        """记录自主目标发现日志（与goal_discovery.py联动）"""
        goal_log_path = os.path.join(self.path_cfg["PATH"]["log_dir"], "goal_discovery.log")
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "priority": priority,
            "basis": basis
        }
        with open(goal_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def get_performance_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """获取性能摘要（支撑可视化/可解释性）"""
        if not self.performance_history:
            return {"message": "暂无性能数据"}
        # 取最近n条数据计算统计值
        recent_history = self.performance_history[-last_n:]
        avg_score = sum([item["final_score"] for item in recent_history]) / len(recent_history)
        stable_rate = sum([1 for item in recent_history if item["is_stable"]]) / len(recent_history)

        return {
            "average_score": round(avg_score, 2),
            "stable_rate": round(stable_rate, 2),
            "current_error_count": self.error_count,
            "last_goal": recent_history[-1]["goal"],
            "last_score": recent_history[-1]["final_score"]
        }

# 性能监控核心验证入口（无冗余，快速测试）
if __name__ == "__main__":
    # 1. 初始化性能监控模块
    perf_monitor = PerformanceMonitor()
    print("UMC性能监控模块初始化完成")

    # 2. 模拟代谢循环结果（来自umc_metabolism.py）
    mock_metabolic_result = {
        "core_factors": {"information": 0.6, "energy": 0.8, "matter": 0.4},
        "stability_score": 0.85,
        "is_stable": True,
        "cycle_count": 10,
        "adapt_rules_used": {"qubit_stability": "energy"}
    }

    # 3. 模拟自主发现的目标
    mock_goal = "提升量子比特代谢稳定性（能量维度）"

    # 4. 评分并记录性能
    score = perf_monitor.score_result(mock_metabolic_result, mock_goal)
    print(f"\n=== 性能评分结果 ===")
    print(f"目标：{mock_goal}")
    print(f"性能得分：{score:.2f}")
    print(f"当前错误计数：{perf_monitor.error_count}")

    # 5. 查看性能摘要
    summary = perf_monitor.get_performance_summary()
    print(f"\n=== 性能摘要 ===")
    print(f"平均得分：{summary['average_score']}")
    print(f"稳定率：{summary['stable_rate']}")

    # 6. 模拟记录错误日志
    perf_monitor.log_error("测试错误：代谢循环稳定性低于阈值")
    print(f"\n错误日志已记录，当前错误计数：{perf_monitor.error_count}")