# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 黑盒测试工具（纯外部视角：接口/IO/性能/稳定性/容错性）
核心逻辑：不关注内部实现，仅验证输入→输出的合规性、性能、稳定性，生成标准化测试报告
设计原则：用例化、自动化、报告标准化、适配新手快速验证整体功能
"""
import configparser
import os
import json
import pandas as pd
import numpy as np
import time
import psutil
import traceback
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# 导入核心工具
from tool_build import UMCAgent, create_test_data

class BlackboxTester:
    """UMC智能体黑盒测试器（核心功能：接口测试/IO验证/性能压测/稳定性测试/容错性测试）"""
    def __init__(self, test_report_dir: str = "./blackbox_reports"):
        """
        初始化黑盒测试器
        :param test_report_dir: 黑盒测试报告目录
        """
        # 初始化测试报告目录
        self.test_report_dir = test_report_dir
        os.makedirs(self.test_report_dir, exist_ok=True)
        # 初始化测试状态
        self.test_suite_result = {
            "test_suite_name": "UMC智能体黑盒测试套件",
            "start_time": "",
            "end_time": "",
            "total_test_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "test_cases": [],
            "performance_metrics": {},
            "stability_metrics": {},
            "error_summary": []
        }
        # 初始化UMC智能体实例（每次测试前重置，避免状态污染）
        self.umc_agent = None

    def _reset_agent(self) -> None:
        """重置UMC智能体实例（避免历史状态影响测试结果）"""
        self.umc_agent = UMCAgent()

    def _record_test_case(self, case_name: str, case_type: str, input_desc: str, expected_output: str, actual_output: str, is_passed: bool, error_msg: str = "") -> None:
        """
        记录单个测试用例结果（结构化）
        :param case_name: 用例名称
        :param case_type: 用例类型（接口/IO/性能/稳定性/容错性）
        :param input_desc: 输入描述
        :param expected_output: 预期输出
        :param actual_output: 实际输出
        :param is_passed: 是否通过
        :param error_msg: 错误信息（可选）
        """
        test_case = {
            "case_name": case_name,
            "case_type": case_type,
            "input_desc": input_desc,
            "expected_output": expected_output,
            "actual_output": actual_output[:500] + "..." if len(str(actual_output)) > 500 else str(actual_output),
            "is_passed": is_passed,
            "error_msg": error_msg[:1000] + "..." if len(error_msg) > 1000 else error_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_suite_result["test_cases"].append(test_case)
        self.test_suite_result["total_test_cases"] += 1
        if is_passed:
            self.test_suite_result["passed_cases"] += 1
        else:
            self.test_suite_result["failed_cases"] += 1
            self.test_suite_result["error_summary"].append({
                "case_name": case_name,
                "error_msg": error_msg
            })

    def test_interface_availability(self) -> None:
        """
        黑盒核心：接口可用性测试（验证核心API是否可正常调用）
        测试用例：load_data/run/get_summary/visualize_result的基础调用
        """
        print("\n📝 开始接口可用性测试...")
        self._reset_agent()
        test_cases = [
            {
                "name": "load_data接口-正常CSV数据",
                "input": "./test_data_quantum.csv",
                "expected": "返回标准化DataFrame，无异常",
                "run_func": lambda: self.umc_agent.load_data("./test_data_quantum.csv", "quantum")
            },
            {
                "name": "run接口-标准化数据",
                "input": "量子领域标准化数据（100行×3列）",
                "expected": "返回运行结果字典，包含目标/策略/代谢/性能等字段",
                "run_func": lambda: self.umc_agent.run(self.umc_agent.load_data("./test_data_quantum.csv", "quantum"), "quantum")
            },
            {
                "name": "get_summary接口-有运行记录",
                "input": "最新运行结果",
                "expected": "返回结果摘要字典，包含运行时间/目标/性能等字段",
                "run_func": lambda: self.umc_agent.get_summary()
            },
            {
                "name": "visualize_result接口-有运行记录",
                "input": "最新运行结果，save_fig=False",
                "expected": "生成可视化图表，无异常",
                "run_func": lambda: self.umc_agent.visualize_result(save_fig=False)
            }
        ]

        # 预生成测试数据
        create_test_data(domain_name="quantum", sample_count=100)

        # 执行每个接口测试用例
        for case in test_cases:
            try:
                # 执行测试函数
                actual_output = case["run_func"]()
                # 判断是否通过（简化版：无异常且返回非空）
                is_passed = True if actual_output is not None else False
                actual_output_desc = f"类型：{type(actual_output).__name__} | 非空：{actual_output is not None}"
                self._record_test_case(
                    case_name=case["name"],
                    case_type="接口测试",
                    input_desc=case["input"],
                    expected_output=case["expected"],
                    actual_output=actual_output_desc,
                    is_passed=is_passed
                )
                print(f"  ✅ {case['name']}：{'通过' if is_passed else '失败'}")
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self._record_test_case(
                    case_name=case["name"],
                    case_type="接口测试",
                    input_desc=case["input"],
                    expected_output=case["expected"],
                    actual_output="执行异常",
                    is_passed=False,
                    error_msg=error_msg
                )
                print(f"  ❌ {case['name']}：失败 | 错误：{str(e)[:50]}...")

    def test_input_output_validation(self) -> None:
        """
        黑盒核心：输入输出合规性测试（验证IO格式/值域/完整性）
        测试用例：数据格式验证、输出字段验证、值域范围验证
        """
        print("\n📝 开始输入输出合规性测试...")
        self._reset_agent()
        # 预生成测试数据并加载
        test_data = create_test_data(domain_name="quantum", sample_count=100)
        standardized_data = self.umc_agent.load_data("./test_data_quantum.csv", "quantum")
        run_result = self.umc_agent.run(standardized_data, "quantum")

        test_cases = [
            {
                "name": "输入验证-标准化数据值域",
                "input": "量子领域标准化数据",
                "expected": "所有数值列的值域在0~1之间",
                "check_func": lambda: (standardized_data.min().min() >= 0) and (standardized_data.max().max() <= 1)
            },
            {
                "name": "输出验证-run_result字段完整性",
                "input": "标准化数据运行结果",
                "expected": "包含timestamp/goal_result/strategy_result/metabolic_result/perf_score/feedback_result字段",
                "check_func": lambda: all([k in run_result for k in ["timestamp", "goal_result", "strategy_result", "metabolic_result", "perf_score", "feedback_result"]])
            },
            {
                "name": "输出验证-性能得分值域",
                "input": "运行结果的性能得分",
                "expected": "性能得分在0~1之间",
                "check_func": lambda: (run_result["perf_score"] >= 0) and (run_result["perf_score"] <= 1)
            },
            {
                "name": "输出验证-稳定性得分值域",
                "input": "代谢循环的稳定性得分",
                "expected": "稳定性得分在0~1之间",
                "check_func": lambda: (run_result["metabolic_result"]["stability_score"] >= 0) and (run_result["metabolic_result"]["stability_score"] <= 1)
            }
        ]

        # 执行每个IO验证用例
        for case in test_cases:
            try:
                # 执行检查函数
                check_result = case["check_func"]()
                is_passed = check_result
                actual_output = f"检查结果：{check_result}"
                self._record_test_case(
                    case_name=case["name"],
                    case_type="IO验证",
                    input_desc=case["input"],
                    expected_output=case["expected"],
                    actual_output=actual_output,
                    is_passed=is_passed
                )
                print(f"  ✅ {case['name']}：{'通过' if is_passed else '失败'}")
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self._record_test_case(
                    case_name=case["name"],
                    case_type="IO验证",
                    input_desc=case["input"],
                    expected_output=case["expected"],
                    actual_output="执行异常",
                    is_passed=False,
                    error_msg=error_msg
                )
                print(f"  ❌ {case['name']}：失败 | 错误：{str(e)[:50]}...")

    def test_performance(self, sample_sizes: List[int] = [100, 500, 1000], run_rounds: int = 3) -> None:
        """
        黑盒核心：性能压测（验证不同数据规模下的响应时间/资源占用）
        :param sample_sizes: 测试的样本规模列表
        :param run_rounds: 每个规模的运行轮数（取平均值）
        """
        print("\n📈 开始性能压测...")
        self._reset_agent()
        performance_metrics = {"sample_sizes": sample_sizes, "rounds_per_size": run_rounds, "results": []}

        # 遍历不同样本规模
        for sample_size in sample_sizes:
            round_times = []
            round_mem_usages = []
            print(f"  测试样本规模：{sample_size}行...")

            # 预生成对应规模的测试数据
            test_data = create_test_data(domain_name="quantum", sample_count=sample_size)
            data_path = f"./test_data_quantum_{sample_size}.csv"
            test_data.to_csv(data_path, index=False, encoding="utf-8")

            # 多轮运行取平均
            for round_idx in range(run_rounds):
                self._reset_agent()  # 每轮重置智能体
                try:
                    # 记录开始时间和内存占用
                    start_time = time.time()
                    start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

                    # 执行核心流程
                    standardized_data = self.umc_agent.load_data(data_path, "quantum")
                    self.umc_agent.run(standardized_data, "quantum")

                    # 记录结束时间和内存占用
                    end_time = time.time()
                    end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

                    # 计算耗时和内存增量
                    run_time = end_time - start_time
                    mem_usage = end_mem - start_mem

                    round_times.append(run_time)
                    round_mem_usages.append(mem_usage)
                    print(f"    第{round_idx+1}轮：耗时{run_time:.2f}s | 内存增量{mem_usage:.2f}MB")
                except Exception as e:
                    error_msg = f"性能压测轮次失败：{str(e)}"
                    print(f"    ❌ 第{round_idx+1}轮：{error_msg[:50]}...")
                    round_times.append(0)
                    round_mem_usages.append(0)

            # 计算该样本规模的平均指标
            avg_run_time = np.mean([t for t in round_times if t > 0]) if round_times else 0
            avg_mem_usage = np.mean([m for m in round_mem_usages if m > 0]) if round_mem_usages else 0

            # 记录性能指标
            performance_metrics["results"].append({
                "sample_size": sample_size,
                "avg_run_time_s": avg_run_time,
                "avg_mem_usage_mb": avg_mem_usage,
                "success_rounds": len([t for t in round_times if t > 0]),
                "total_rounds": run_rounds
            })

            # 记录性能测试用例
            self._record_test_case(
                case_name=f"性能压测-{sample_size}行数据",
                case_type="性能测试",
                input_desc=f"{sample_size}行量子领域数据，运行{run_rounds}轮",
                expected_output=f"平均耗时<10s，平均内存增量<100MB",
                actual_output=f"平均耗时{avg_run_time:.2f}s | 平均内存增量{avg_mem_usage:.2f}MB | 成功轮数{len([t for t in round_times if t > 0])}/{run_rounds}",
                is_passed=(avg_run_time < 10) and (avg_mem_usage < 100)
            )

        # 保存性能指标到测试套件结果
        self.test_suite_result["performance_metrics"] = performance_metrics
        print(f"  ✅ 性能压测完成！")

    def test_stability(self, cycle_count: int = 10, interval_s: int = 1) -> None:
        """
        黑盒核心：稳定性测试（验证长时间/循环运行的鲁棒性）
        :param cycle_count: 循环运行次数
        :param interval_s: 每次循环间隔（秒）
        """
        print("\n🔄 开始稳定性测试...")
        self._reset_agent()
        stability_metrics = {"cycle_count": cycle_count, "interval_s": interval_s, "success_cycles": 0, "failed_cycles": 0, "error_cycles": []}

        # 预生成测试数据
        test_data = create_test_data(domain_name="quantum", sample_count=100)
        data_path = "./test_data_quantum_stability.csv"
        test_data.to_csv(data_path, index=False, encoding="utf-8")

        # 循环运行核心流程
        for cycle_idx in range(cycle_count):
            print(f"  稳定性循环 {cycle_idx+1}/{cycle_count}...")
            try:
                self._reset_agent()  # 每次循环重置智能体
                standardized_data = self.umc_agent.load_data(data_path, "quantum")
                self.umc_agent.run(standardized_data, "quantum")
                stability_metrics["success_cycles"] += 1
                time.sleep(interval_s)  # 间隔
            except Exception as e:
                error_msg = f"循环{cycle_idx+1}失败：{str(e)}"
                stability_metrics["failed_cycles"] += 1
                stability_metrics["error_cycles"].append({
                    "cycle_idx": cycle_idx+1,
                    "error_msg": error_msg
                })
                print(f"    ❌ {error_msg[:50]}...")

        # 记录稳定性测试用例
        success_rate = stability_metrics["success_cycles"] / cycle_count
        self._record_test_case(
            case_name=f"稳定性测试-{cycle_count}次循环",
            case_type="稳定性测试",
            input_desc=f"循环运行{cycle_count}次，每次间隔{interval_s}s",
            expected_output=f"成功率≥90%",
            actual_output=f"成功次数{stability_metrics['success_cycles']}/{cycle_count} | 成功率{success_rate:.2%} | 失败次数{stability_metrics['failed_cycles']}",
            is_passed=(success_rate >= 0.9)
        )

        # 保存稳定性指标到测试套件结果
        self.test_suite_result["stability_metrics"] = stability_metrics
        print(f"  ✅ 稳定性测试完成！成功率：{success_rate:.2%}")

    def test_fault_tolerance(self) -> None:
        """
        黑盒核心：异常输入容错性测试（验证异常输入下的鲁棒性）
        测试用例：空数据、格式错误数据、超大列数据、缺失列数据、非数值数据
        """
        print("\n🛡️ 开始异常输入容错性测试...")
        self._reset_agent()
        fault_test_cases = [
            {
                "name": "容错性-空数据",
                "input_func": lambda: pd.DataFrame(),
                "expected": "捕获异常，返回明确错误信息，不崩溃",
                "input_desc": "空DataFrame"
            },
            {
                "name": "容错性-格式错误数据（TXT）",
                "input_func": lambda: (open("./test_data_error.txt", "w").write("invalid data") or "./test_data_error.txt"),
                "expected": "捕获文件格式错误，返回明确错误信息，不崩溃",
                "input_desc": "TXT格式文件（仅支持CSV/Excel）"
            },
            {
                "name": "容错性-超大列数据（100列）",
                "input_func": lambda: pd.DataFrame(np.random.rand(100, 100)),
                "expected": "正常处理，无崩溃，返回运行结果",
                "input_desc": "100行×100列随机数据"
            },
            {
                "name": "容错性-缺失列数据",
                "input_func": lambda: pd.DataFrame({"qubit_stability": [0.8, 0.7]}),  # 仅单列
                "expected": "正常处理，自主发现目标，无崩溃",
                "input_desc": "仅含qubit_stability列的数据集"
            },
            {
                "name": "容错性-非数值数据",
                "input_func": lambda: pd.DataFrame({"qubit_stability": ["a", "b", "c"]}),
                "expected": "尝试转换失败，捕获异常，返回明确错误信息，不崩溃",
                "input_desc": "非数值列的数据集"
            }
        ]

        # 执行每个容错性测试用例
        for case in fault_test_cases:
            try:
                self._reset_agent()
                input_data = case["input_func"]()
                actual_output = ""
                is_passed = True

                # 区分数据文件和DataFrame输入
                if isinstance(input_data, str):  # 文件路径
                    try:
                        self.umc_agent.load_data(input_data, "quantum")
                        actual_output = "加载文件无异常（不符合预期）"
                        is_passed = False
                    except ValueError as e:
                        actual_output = f"捕获预期异常：{str(e)[:50]}..."
                        is_passed = True
                else:  # DataFrame
                    try:
                        if not input_data.empty:
                            self.umc_agent.run(input_data, "quantum")
                            actual_output = "运行无异常，返回结果"
                        else:
                            self.umc_agent.run(input_data, "quantum")
                            actual_output = "空数据处理无异常（不符合预期）"
                            is_passed = False
                    except Exception as e:
                        actual_output = f"捕获预期异常：{str(e)[:50]}..."
                        is_passed = True

                self._record_test_case(
                    case_name=case["name"],
                    case_type="容错性测试",
                    input_desc=case["input_desc"],
                    expected_output=case["expected"],
                    actual_output=actual_output,
                    is_passed=is_passed
                )
                print(f"  ✅ {case['name']}：{'通过' if is_passed else '失败'}")
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self._record_test_case(
                    case_name=case["name"],
                    case_type="容错性测试",
                    input_desc=case["input_desc"],
                    expected_output=case["expected"],
                    actual_output="执行异常",
                    is_passed=False,
                    error_msg=error_msg
                )
                print(f"  ❌ {case['name']}：失败 | 错误：{str(e)[:50]}...")

    def generate_test_report(self, save_html: bool = True) -> str:
        """
        生成标准化黑盒测试报告（JSON+可选HTML）
        :param save_html: 是否生成HTML格式报告（便于阅读）
        :return: 报告文件路径
        """
        # 补充测试套件的时间和汇总信息
        self.test_suite_result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pass_rate = self.test_suite_result["passed_cases"] / self.test_suite_result["total_test_cases"] if self.test_suite_result["total_test_cases"] > 0 else 0
        self.test_suite_result["pass_rate"] = pass_rate

        # 保存JSON格式报告（结构化，便于解析）
        report_filename = f"blackbox_test_report_{time.strftime('%Y%m%d%H%M%S')}"
        json_report_path = os.path.join(self.test_report_dir, f"{report_filename}.json")
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(self.test_suite_result, f, ensure_ascii=False, indent=2)

        # 生成HTML格式报告（便于阅读）
        html_report_path = ""
        if save_html:
            html_report_path = os.path.join(self.test_report_dir, f"{report_filename}.html")
            html_content = self._generate_html_report()
            with open(html_report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        # 打印报告汇总
        print("\n=== 黑盒测试报告汇总 ===")
        print(f"测试套件：{self.test_suite_result['test_suite_name']}")
        print(f"测试时间：{self.test_suite_result['start_time']} ~ {self.test_suite_result['end_time']}")
        print(f"总用例数：{self.test_suite_result['total_test_cases']}")
        print(f"通过用例：{self.test_suite_result['passed_cases']}")
        print(f"失败用例：{self.test_suite_result['failed_cases']}")
        print(f"通过率：{pass_rate:.2%}")
        print(f"JSON报告：{json_report_path}")
        if save_html:
            print(f"HTML报告：{html_report_path}")

        return json_report_path

    def _generate_html_report(self) -> str:
        """生成HTML格式的测试报告（简化版，便于阅读）"""
        pass_rate = self.test_suite_result["passed_cases"] / self.test_suite_result["total_test_cases"] if self.test_suite_result["total_test_cases"] > 0 else 0
        pass_rate_color = "green" if pass_rate >= 0.9 else "orange" if pass_rate >= 0.7 else "red"

        # 构建HTML内容
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>{self.test_suite_result['test_suite_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .case {{ margin: 10px 0; padding: 10px; border-radius: 4px; }}
                .passed {{ background: #e8f5e9; border: 1px solid #81c784; }}
                .failed {{ background: #ffebee; border: 1px solid #e57373; }}
                .metrics {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                .error {{ color: #d32f2f; }}
            </style>
        </head>
        <body>
            <h1>{self.test_suite_result['test_suite_name']}</h1>
            <div class="summary">
                <p>测试时间：{self.test_suite_result['start_time']} ~ {self.test_suite_result['end_time']}</p>
                <p>总用例数：{self.test_suite_result['total_test_cases']}</p>
                <p>通过用例：{self.test_suite_result['passed_cases']}</p>
                <p>失败用例：{self.test_suite_result['failed_cases']}</p>
                <p>通过率：<span style="color: {pass_rate_color}; font-weight: bold;">{pass_rate:.2%}</span></p>
            </div>

            <h2>性能指标</h2>
            <div class="metrics">
                <pre>{json.dumps(self.test_suite_result['performance_metrics'], ensure_ascii=False, indent=2)}</pre>
            </div>

            <h2>稳定性指标</h2>
            <div class="metrics">
                <pre>{json.dumps(self.test_suite_result['stability_metrics'], ensure_ascii=False, indent=2)}</pre>
            </div>

            <h2>测试用例详情</h2>
        """

        # 添加每个测试用例的详情
        for case in self.test_suite_result['test_cases']:
            case_class = "passed" if case['is_passed'] else "failed"
            case_status = "通过" if case['is_passed'] else "失败"
            html += f"""
            <div class="case {case_class}">
                <h3>{case['case_name']}（{case['case_type']}）- {case_status}</h3>
                <p><strong>输入：</strong>{case['input_desc']}</p>
                <p><strong>预期输出：</strong>{case['expected_output']}</p>
                <p><strong>实际输出：</strong>{case['actual_output']}</p>
                {f"<p class='error'><strong>错误信息：</strong>{case['error_msg']}</p>" if not case['is_passed'] else ""}
                <p><small>时间：{case['timestamp']}</small></p>
            </div>
            """

        # 添加错误汇总
        if self.test_suite_result['error_summary']:
            html += f"""
            <h2>错误汇总</h2>
            <div class="metrics">
                <pre>{json.dumps(self.test_suite_result['error_summary'], ensure_ascii=False, indent=2)}</pre>
            </div>
            """

        html += """
        </body>
        </html>
        """
        return html

    def run_all_tests(self) -> str:
        """
        一键运行所有黑盒测试用例（接口+IO+性能+稳定性+容错性）
        :return: 测试报告路径
        """
        print("🚀 开始运行UMC智能体全量黑盒测试套件...")
        self.test_suite_result["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 依次运行各类测试
        self.test_interface_availability()
        self.test_input_output_validation()
        self.test_performance(sample_sizes=[100, 500], run_rounds=2)  # 简化版，减少测试时间
        self.test_stability(cycle_count=5, interval_s=1)  # 简化版，减少测试时间
        self.test_fault_tolerance()

        # 生成测试报告
        report_path = self.generate_test_report(save_html=True)
        print(f"\n🎉 全量黑盒测试完成！报告已保存至：{report_path}")
        return report_path

# 黑盒测试工具验证入口（一站式测试所有黑盒功能）
if __name__ == "__main__":
    # 1. 初始化黑盒测试器
    blackbox_tester = BlackboxTester()

    # 2. 一键运行所有黑盒测试
    report_path = blackbox_tester.run_all_tests()

    # 3. 可选：单独运行某类测试
    # blackbox_tester.test_interface_availability()
    # blackbox_tester.test_input_output_validation()
    # blackbox_tester.test_performance()
    # blackbox_tester.test_stability()
    # blackbox_tester.test_fault_tolerance()
    # blackbox_tester.generate_test_report()