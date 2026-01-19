# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 白盒调试工具（核心逻辑拆解+状态监控+参数分析）
核心逻辑：从白盒视角暴露智能体内部状态、分步追踪运行流程、分析参数影响，辅助调试/学习
设计原则：极致透明、分步拆解、日志详尽、适配新手理解内部逻辑
"""
import configparser
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
from typing import Dict, Any, List, Callable, Optional
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 导入核心工具和模块
from tool_build import UMCAgent, create_test_data
from data_processing import DataProcessor
from goal_discovery import AutonomousGoalDiscovery
from umc_performance import PerformanceMonitor

class WhiteboxDebugger:
    """UMC智能体白盒调试器（核心功能：监控/追踪/分析/调试）"""
    def __init__(self, umc_agent: UMCAgent, debug_log_dir: str = "./whitebox_logs"):
        """
        初始化白盒调试器
        :param umc_agent: 已初始化的UMCAgent实例
        :param debug_log_dir: 白盒调试日志目录
        """
        # 关联UMC智能体实例
        self.umc_agent = umc_agent
        # 初始化调试日志目录
        self.debug_log_dir = debug_log_dir
        os.makedirs(self.debug_log_dir, exist_ok=True)
        # 初始化调试状态
        self.debug_history = []
        self.step_trace_log = []
        self.param_analysis_result = {}

    def monitor_module_states(self, save_log: bool = True) -> Dict[str, Any]:
        """
        白盒核心：实时监控所有核心模块的内部状态
        输出内容：缓存、计数、配置、历史记录等内部变量
        :param save_log: 是否保存状态日志
        :return: 所有模块的状态汇总
        """
        print("🔍 开始监控核心模块内部状态...")
        state_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_processor": {
                "cache_size": len(self.umc_agent.data_processor.data_cache),
                "cache_keys": list(self.umc_agent.data_processor.data_cache.keys()),
                "process_rules": self.umc_agent.data_processor.process_rules
            },
            "goal_discoverer": {
                "goal_history_count": len(self.umc_agent.goal_discoverer.goal_history),
                "current_goal": self.umc_agent.goal_discoverer.current_goal,
                "current_goal_basis": self.umc_agent.goal_discoverer.current_goal_basis
            },
            "perf_monitor": {
                "performance_history_count": len(self.umc_agent.perf_monitor.performance_history),
                "error_count": self.umc_agent.perf_monitor.error_count,
                "current_performance_score": self.umc_agent.perf_monitor.current_performance_score
            },
            "feedback_optimizer": {
                "optimize_count": self.umc_agent.feedback_optimizer.optimize_count,
                "feedback_history_count": len(self.umc_agent.feedback_optimizer.feedback_history),
                "current_optimize_result": self.umc_agent.feedback_optimizer.current_optimize_result
            },
            "auto_recovery": {
                "fault_count": len(self.umc_agent.auto_recovery.fault_history),
                "rollback_count": self.umc_agent.auto_recovery.rollback_count,
                "last_backup_time": time.ctime(self.umc_agent.auto_recovery.last_backup_time)
            },
            "strategy_module": {
                "current_strategy_weights": {k: v for k, v in self.umc_agent.strategy_module.param_cfg["STRATEGY"].items()}
            }
        }

        # 打印状态（结构化，便于阅读）
        print("\n=== 核心模块状态汇总 ===")
        for module, state in state_summary.items():
            if module == "timestamp":
                print(f"📌 监控时间：{state}")
                continue
            print(f"\n📦 模块：{module}")
            for k, v in state.items():
                # 简化长文本输出
                if isinstance(v, str) and len(v) > 100:
                    v = v[:100] + "..."
                print(f"  - {k}：{v}")

        # 保存状态日志
        if save_log:
            log_path = os.path.join(self.debug_log_dir, f"module_state_{time.strftime('%Y%m%d%H%M%S')}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(state_summary, f, ensure_ascii=False, indent=2)
            print(f"\n💾 模块状态日志已保存：{log_path}")

        # 记录调试历史
        self.debug_history.append({"type": "module_state", "data": state_summary})
        return state_summary

    def trace_run_step_by_step(self, data: pd.DataFrame, domain_name: str = "unknown") -> Dict[str, Any]:
        """
        白盒核心：分步追踪UMC智能体run流程，输出每步的输入/输出/中间变量
        拆解步骤：目标发现→策略选择→代谢循环→性能校验→自主优化
        :param data: 标准化后的数据
        :param domain_name: 数据领域
        :return: 分步追踪结果
        """
        print("\n🚶 开始分步追踪UMC智能体运行流程...")
        self.step_trace_log = []  # 重置分步日志
        trace_result = {"domain_name": domain_name, "steps": []}

        try:
            # === 步骤1：自主发现目标（输出特征重要性） ===
            print("\n===== 步骤1：自主目标发现 =====")
            step1_start = time.time()
            goal_result = self.umc_agent.goal_discoverer.discover_goal(data)
            step1_end = time.time()
            # 记录步骤1详情
            step1_data = {
                "step": "goal_discovery",
                "duration": f"{step1_end - step1_start:.3f}s",
                "input_shape": data.shape,
                "output_goal": goal_result["goal"],
                "feature_importance": goal_result["feature_importance"],
                "priority": goal_result["priority"]
            }
            trace_result["steps"].append(step1_data)
            self.step_trace_log.append(step1_data)
            # 打印步骤1关键信息
            print(f"✅ 目标发现耗时：{step1_data['duration']}")
            print(f"🎯 发现目标：{step1_data['output_goal']}")
            print(f"📊 特征重要性TOP1：{max(step1_data['feature_importance'].items(), key=lambda x: x[1])}")

            # === 步骤2：选择最优策略（输出策略得分） ===
            print("\n===== 步骤2：最优策略选择 =====")
            step2_start = time.time()
            # 先运行预代谢循环获取因子
            mock_adapt_rules = {"factor_mapping": {"default": "stability"}}
            metabolic_pre = self.umc_agent.metabolic_cycle.run(data, goal_result["goal"], mock_adapt_rules)
            strategy_result = self.umc_agent.strategy_module.select_optimal_strategy(domain_name, metabolic_pre["core_factors"])
            step2_end = time.time()
            # 记录步骤2详情
            step2_data = {
                "step": "strategy_selection",
                "duration": f"{step2_end - step2_start:.3f}s",
                "input_factors": metabolic_pre["core_factors"],
                "output_strategy": strategy_result["strategy_name"],
                "strategy_score": strategy_result["strategy_score"],
                "strategy_weights": strategy_result["factor_weight"]
            }
            trace_result["steps"].append(step2_data)
            self.step_trace_log.append(step2_data)
            # 打印步骤2关键信息
            print(f"✅ 策略选择耗时：{step2_data['duration']}")
            print(f"📋 最优策略：{step2_data['output_strategy']}（得分：{step2_data['strategy_score']:.2f}）")

            # === 步骤3：运行代谢循环（输出核心因子/稳定性） ===
            print("\n===== 步骤3：代谢循环执行 =====")
            step3_start = time.time()
            metabolic_result = self.umc_agent.metabolic_cycle.run(data, goal_result["goal"], {"factor_mapping": strategy_result["factor_weight"]})
            step3_end = time.time()
            # 记录步骤3详情
            step3_data = {
                "step": "metabolic_cycle",
                "duration": f"{step3_end - step3_start:.3f}s",
                "core_factors": metabolic_result["core_factors"],
                "stability_score": metabolic_result["stability_score"],
                "cycle_count": metabolic_result["cycle_count"],
                "is_stable": metabolic_result["is_stable"]
            }
            trace_result["steps"].append(step3_data)
            self.step_trace_log.append(step3_data)
            # 打印步骤3关键信息
            print(f"✅ 代谢循环耗时：{step3_data['duration']}")
            print(f"🔄 循环次数：{step3_data['cycle_count']} | 稳定性得分：{step3_data['stability_score']:.2f}")
            print(f"📊 核心因子：{step3_data['core_factors']}")

            # === 步骤4：性能校验（输出得分/错误计数） ===
            print("\n===== 步骤4：性能闭环验证 =====")
            step4_start = time.time()
            perf_score = self.umc_agent.perf_monitor.score_result(metabolic_result, goal_result["goal"])
            step4_end = time.time()
            # 记录步骤4详情
            step4_data = {
                "step": "performance_validation",
                "duration": f"{step4_end - step4_start:.3f}s",
                "input_stability": metabolic_result["stability_score"],
                "output_score": perf_score,
                "error_count": self.umc_agent.perf_monitor.error_count,
                "is_passed": perf_score >= float(self.umc_agent.perf_monitor.param_cfg["VALIDATION"]["blackbox_test_threshold"])
            }
            trace_result["steps"].append(step4_data)
            self.step_trace_log.append(step4_data)
            # 打印步骤4关键信息
            print(f"✅ 性能校验耗时：{step4_data['duration']}")
            print(f"📊 性能得分：{step4_data['output_score']:.2f} | 是否达标：{step4_data['is_passed']}")
            print(f"❌ 错误计数：{step4_data['error_count']}")

            # === 步骤5：自主学习反馈（输出优化结果） ===
            print("\n===== 步骤5：自主学习反馈 =====")
            step5_start = time.time()
            feedback_result = self.umc_agent.feedback_optimizer.feedback_optimize(data, metabolic_result)
            step5_end = time.time()
            # 记录步骤5详情
            step5_data = {
                "step": "self_learning_feedback",
                "duration": f"{step5_end - step5_start:.3f}s",
                "input_perf_score": perf_score,
                "optimize_result": feedback_result,
                "optimize_count": self.umc_agent.feedback_optimizer.optimize_count
            }
            trace_result["steps"].append(step5_data)
            self.step_trace_log.append(step5_data)
            # 打印步骤5关键信息
            print(f"✅ 自主优化耗时：{step5_data['duration']}")
            if feedback_result["optimize_status"] != "no_optimize":
                print(f"🔧 优化目标：{feedback_result['adjust_target']} | 调整幅度：{feedback_result['adjust_amount']:.3f}")
                print(f"⚖️  权重变化：{feedback_result['old_weight']:.2f} → {feedback_result['new_weight']:.2f}")
            else:
                print(f"🔧 无需优化：{feedback_result['reason']}")

            # === 汇总分步结果 ===
            trace_result["total_duration"] = f"{sum([float(step['duration'].replace('s','')) for step in trace_result['steps']]):.3f}s"
            trace_result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # 保存分步追踪日志
            log_path = os.path.join(self.debug_log_dir, f"step_trace_{time.strftime('%Y%m%d%H%M%S')}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(trace_result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 分步追踪日志已保存：{log_path}")
            print(f"\n🏁 分步追踪完成！总耗时：{trace_result['total_duration']}")

            # 记录调试历史
            self.debug_history.append({"type": "step_trace", "data": trace_result})
            return trace_result

        except Exception as e:
            error_msg = f"分步追踪失败：{str(e)}\n{traceback.format_exc()}"
            print(f"\n❌ {error_msg}")
            # 记录错误日志
            error_log_path = os.path.join(self.debug_log_dir, f"step_trace_error_{time.strftime('%Y%m%d%H%M%S')}.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
            self.debug_history.append({"type": "step_trace_error", "data": error_msg})
            raise e

    def analyze_param_sensitivity(self, data: pd.DataFrame, param_name: str, param_values: List[float], param_section: str = "AGI_L3") -> Dict[str, Any]:
        """
        白盒核心：参数敏感度分析（测试不同参数值对结果的影响）
        适用场景：调试目标发现阈值、反馈率、故障阈值等关键参数
        :param data: 标准化后的数据
        :param param_name: 要分析的参数名（如goal_discovery_threshold）
        :param param_values: 测试的参数值列表（如[0.3,0.4,0.5,0.6,0.7]）
        :param param_section: 参数所在的配置段（如AGI_L3/METABOLISM）
        :return: 参数敏感度分析结果
        """
        print(f"\n📊 开始参数敏感度分析：{param_section}.{param_name}")
        analysis_result = {
            "param_name": param_name,
            "param_section": param_section,
            "param_values": param_values,
            "metrics": ["goal_priority", "stability_score", "perf_score", "optimize_count"],
            "results": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存原始参数值（分析后恢复）
        original_value = self.umc_agent.perf_monitor.param_cfg[param_section][param_name]

        try:
            for value in param_values:
                print(f"\n--- 测试参数值：{value} ---")
                # 更新参数值
                self.umc_agent.perf_monitor.param_cfg[param_section][param_name] = str(value)
                with open("parameters.ini", "w", encoding="utf-8") as f:
                    self.umc_agent.perf_monitor.param_cfg.write(f)
                
                # 重新初始化受影响的模块
                self.umc_agent.goal_discoverer = AutonomousGoalDiscovery()
                self.umc_agent.feedback_optimizer = SelfLearningFeedback()
                self.umc_agent.auto_recovery = AutoRecovery()

                # 运行核心流程（简化版）
                goal_result = self.umc_agent.goal_discoverer.discover_goal(data)
                mock_adapt_rules = {"factor_mapping": {"default": "stability"}}
                metabolic_pre = self.umc_agent.metabolic_cycle.run(data, goal_result["goal"], mock_adapt_rules)
                strategy_result = self.umc_agent.strategy_module.select_optimal_strategy("unknown", metabolic_pre["core_factors"])
                metabolic_result = self.umc_agent.metabolic_cycle.run(data, goal_result["goal"], {"factor_mapping": strategy_result["factor_weight"]})
                perf_score = self.umc_agent.perf_monitor.score_result(metabolic_result, goal_result["goal"])
                feedback_result = self.umc_agent.feedback_optimizer.feedback_optimize(data, metabolic_result)

                # 记录该参数值的结果
                result_item = {
                    "param_value": value,
                    "goal_priority": goal_result["priority"],
                    "stability_score": metabolic_result["stability_score"],
                    "perf_score": perf_score,
                    "optimize_count": self.umc_agent.feedback_optimizer.optimize_count
                }
                analysis_result["results"].append(result_item)

                # 打印该轮结果
                print(f"  目标优先级：{result_item['goal_priority']}")
                print(f"  稳定性得分：{result_item['stability_score']:.2f}")
                print(f"  性能得分：{result_item['perf_score']:.2f}")
                print(f"  优化次数：{result_item['optimize_count']}")

            # 生成敏感度分析图表
            self._plot_param_sensitivity(analysis_result)

            # 保存分析结果
            log_path = os.path.join(self.debug_log_dir, f"param_sensitivity_{param_name}_{time.strftime('%Y%m%d%H%M%S')}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 参数敏感度分析日志已保存：{log_path}")

            # 记录调试历史
            self.debug_history.append({"type": "param_sensitivity", "data": analysis_result})
            self.param_analysis_result = analysis_result
            return analysis_result

        finally:
            # 恢复原始参数值
            self.umc_agent.perf_monitor.param_cfg[param_section][param_name] = original_value
            with open("parameters.ini", "w", encoding="utf-8") as f:
                self.umc_agent.perf_monitor.param_cfg.write(f)
            print(f"\n🔙 已恢复原始参数值：{original_value}")

    def _plot_param_sensitivity(self, analysis_result: Dict[str, Any]) -> None:
        """生成参数敏感度分析图表"""
        param_values = analysis_result["param_values"]
        metrics = analysis_result["metrics"]
        results = analysis_result["results"]

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            # 提取该指标的所有值
            metric_values = [item[metric] for item in results]
            # 绘制折线图
            axes[idx].plot(param_values, metric_values, marker="o", linewidth=2, markersize=6)
            axes[idx].set_title(f"{metric} 随 {analysis_result['param_name']} 变化", fontsize=10, fontweight="bold")
            axes[idx].set_xlabel(analysis_result['param_name'])
            axes[idx].set_ylabel(metric)
            axes[idx].grid(True, alpha=0.3)
            # 添加数值标签
            for x, y in zip(param_values, metric_values):
                axes[idx].text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        fig_path = os.path.join(self.debug_log_dir, f"param_sensitivity_{analysis_result['param_name']}_{time.strftime('%Y%m%d%H%M%S')}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"📸 参数敏感度图表已保存：{fig_path}")
        plt.show()

    def debug_core_function(self, func: Callable, func_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        白盒核心：单步调试核心函数，输出输入/输出/执行时间/异常
        :param func: 要调试的核心函数（如data_processor._handle_missing_values）
        :param func_name: 函数名称（用于日志）
        :param args/kwargs: 函数参数
        :return: 调试结果
        """
        print(f"\n🔧 开始单步调试函数：{func_name}")
        debug_result = {
            "func_name": func_name,
            "input_args": str(args)[:200] + "..." if len(str(args)) > 200 else str(args),
            "input_kwargs": str(kwargs)[:200] + "..." if len(str(kwargs)) > 200 else str(kwargs),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": 0.0,
            "output": None,
            "error": None
        }

        try:
            # 执行函数并计时
            start = time.time()
            output = func(*args, **kwargs)
            end = time.time()

            # 记录成功结果
            debug_result["duration"] = f"{end - start:.3f}s"
            debug_result["output"] = str(output)[:500] + "..." if len(str(output)) > 500 else str(output)
            print(f"✅ 函数执行成功！耗时：{debug_result['duration']}")
            print(f"📤 函数输出：{debug_result['output']}")

        except Exception as e:
            # 记录异常结果
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            debug_result["error"] = error_msg[:1000] + "..." if len(error_msg) > 1000 else error_msg
            print(f"❌ 函数执行异常：{debug_result['error']}")

        # 保存函数调试日志
        log_path = os.path.join(self.debug_log_dir, f"func_debug_{func_name}_{time.strftime('%Y%m%d%H%M%S')}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(debug_result, f, ensure_ascii=False, indent=2)
        print(f"💾 函数调试日志已保存：{log_path}")

        # 记录调试历史
        self.debug_history.append({"type": "func_debug", "data": debug_result})
        return debug_result

    def compare_config(self, config_path1: str = "./parameters.ini", config_path2: str = "./parameters_default.ini") -> Dict[str, Any]:
        """
        白盒辅助：对比两个配置文件的差异，定位参数问题
        :param config_path1: 当前配置文件
        :param config_path2: 参考配置文件（如默认配置）
        :return: 配置差异结果
        """
        print(f"\n🔍 对比配置文件：{config_path1} vs {config_path2}")
        # 加载两个配置文件
        cfg1 = configparser.ConfigParser()
        cfg1.read(config_path1, encoding="utf-8")
        cfg2 = configparser.ConfigParser()
        cfg2.read(config_path2, encoding="utf-8") if os.path.exists(config_path2) else None

        compare_result = {
            "only_in_cfg1": [],  # 仅在cfg1中存在的配置
            "only_in_cfg2": [],  # 仅在cfg2中存在的配置
            "value_diff": [],    # 键相同但值不同的配置
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 遍历cfg1的所有配置
        for section in cfg1.sections():
            for key, value in cfg1[section].items():
                full_key = f"{section}.{key}"
                # 检查cfg2中是否存在该配置
                if not cfg2.has_section(section) or not cfg2[section].get(key):
                    compare_result["only_in_cfg1"].append(full_key)
                else:
                    # 对比值是否不同
                    if value != cfg2[section][key]:
                        compare_result["value_diff"].append({
                            "key": full_key,
                            "cfg1_value": value,
                            "cfg2_value": cfg2[section][key]
                        })

        # 遍历cfg2的所有配置（检查仅在cfg2中存在的）
        if os.path.exists(config_path2):
            for section in cfg2.sections():
                for key, value in cfg2[section].items():
                    full_key = f"{section}.{key}"
                    if not cfg1.has_section(section) or not cfg1[section].get(key):
                        compare_result["only_in_cfg2"].append(full_key)

        # 打印对比结果
        print("\n=== 配置对比结果 ===")
        if compare_result["only_in_cfg1"]:
            print(f"📌 仅在当前配置中存在：{compare_result['only_in_cfg1']}")
        if compare_result["only_in_cfg2"]:
            print(f"📌 仅在参考配置中存在：{compare_result['only_in_cfg2']}")
        if compare_result["value_diff"]:
            print(f"\n📌 值不同的配置：")
            for diff in compare_result["value_diff"]:
                print(f"  - {diff['key']}：{diff['cfg1_value']} (当前) vs {diff['cfg2_value']} (参考)")
        if not any([compare_result["only_in_cfg1"], compare_result["only_in_cfg2"], compare_result["value_diff"]]):
            print(f"✅ 两个配置文件完全一致")

        # 保存对比日志
        log_path = os.path.join(self.debug_log_dir, f"config_compare_{time.strftime('%Y%m%d%H%M%S')}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(compare_result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 配置对比日志已保存：{log_path}")

        # 记录调试历史
        self.debug_history.append({"type": "config_compare", "data": compare_result})
        return compare_result

# 白盒调试工具验证入口（一站式测试所有白盒功能）
if __name__ == "__main__":
    # 1. 初始化基础UMC智能体
    print("🚀 初始化UMC智能体（基础）...")
    umc_agent = UMCAgent()
    # 生成默认配置文件（用于对比）
    default_config_path = "./parameters_default.ini"
    if not os.path.exists(default_config_path):
        umc_agent._init_default_config("./")
        os.rename("./parameters.ini", default_config_path)
        umc_agent = UMCAgent()  # 重新初始化，生成当前配置

    # 2. 创建测试数据并加载
    print("\n📄 生成测试数据...")
    test_data = create_test_data(domain_name="quantum", sample_count=100)
    standardized_data = umc_agent.load_data("./test_data_quantum.csv", domain_name="quantum")

    # 3. 初始化白盒调试器
    print("\n🔧 初始化白盒调试器...")
    whitebox = WhiteboxDebugger(umc_agent)

    # 4. 功能1：监控模块内部状态
    whitebox.monitor_module_states()

    # 5. 功能2：分步追踪运行流程
    trace_result = whitebox.trace_run_step_by_step(standardized_data, domain_name="quantum")

    # 6. 功能3：参数敏感度分析（测试目标发现阈值）
    analysis_result = whitebox.analyze_param_sensitivity(
        standardized_data,
        param_name="goal_discovery_threshold",
        param_values=[0.3, 0.4, 0.5, 0.6, 0.7],
        param_section="AGI_L3"
    )

    # 7. 功能4：单步调试核心函数（测试缺失值处理）
    debug_result = whitebox.debug_core_function(
        func=umc_agent.data_processor._handle_missing_values,
        func_name="data_processor._handle_missing_values",
        data=standardized_data.copy()
    )

    # 8. 功能5：配置文件对比
    whitebox.compare_config(config_path1="./parameters.ini", config_path2="./parameters_default.ini")

    # 9. 查看调试历史
    print("\n=== 白盒调试历史汇总 ===")
    print(f"调试记录数：{len(whitebox.debug_history)}")
    for idx, record in enumerate(whitebox.debug_history):
        print(f"  {idx+1}. 类型：{record['type']}")

    print("\n🎉 白盒调试工具测试完成！所有日志已保存至 ./whitebox_logs")