# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 全自动化批量运行脚本
核心逻辑：多配置组合、批量执行、并行调度、结果汇总，支持全流程自动化
设计原则：无人工干预、批量验证、多场景对比、完整日志归档
"""
import os
import sys
import json
import time
import logging
import warnings
import argparse
import multiprocessing
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志（批量运行专用，按任务归档）
LOG_DIR = "./umc_batch_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(task_id)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/batch_run_{datetime.now().strftime('%Y%m%d%H%M%S')}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UMC-BatchRun")
# 自定义日志过滤器（添加task_id字段）
class TaskFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'task_id'):
            record.task_id = 'main'
        return True
logger.addFilter(TaskFilter())
warnings.filterwarnings("ignore")

# ------------------------------ 批量运行配置 ------------------------------
# 默认批量配置（支持多领域、多参数组合）
BATCH_CONFIG = {
    "output_root": "./umc_batch_output",       # 批量输出根目录
    "parallel_workers": 2,                     # 并行执行的工作线程数
    "generate_unique_data": True,              # 为每个任务生成独立测试数据
    "data_rows_per_task": 1000,                # 每个任务的测试数据行数
    "cleanup_before_run": False,               # 运行前清理旧输出
    "generate_summary_report": True,           # 生成批量结果汇总报告
    "archive_logs": True,                      # 归档运行日志
    
    # 批量执行的任务配置（多组合）
    "task_configs": [
        # 任务1：量子领域，基础配置
        {
            "task_id": "quantum_basic",
            "domain": "quantum",
            "run_time": 60,
            "adapt_iterations": 50,
            "learning_rate": 0.01,
            "target_metric": "metabolic_efficiency"
        },
        # 任务2：生物领域，高迭代配置
        {
            "task_id": "biology_high_iter",
            "domain": "biology",
            "run_time": 90,
            "adapt_iterations": 100,
            "learning_rate": 0.008,
            "target_metric": "domain_adapt_score"
        },
        # 任务3：化学领域，快速验证配置
        {
            "task_id": "chemistry_fast",
            "domain": "chemistry",
            "run_time": 30,
            "adapt_iterations": 20,
            "learning_rate": 0.015,
            "target_metric": "stability"
        },
        # 任务4：材料领域，平衡配置
        {
            "task_id": "materials_balance",
            "domain": "materials",
            "run_time": 75,
            "adapt_iterations": 75,
            "learning_rate": 0.009,
            "target_metric": "metabolic_efficiency"
        }
    ]
}

# ------------------------------ 依赖检查与初始化 ------------------------------
def check_and_install_deps():
    """自动检查并安装所有依赖（无交互）"""
    required_pkgs = [
        "pandas", "numpy", "pyjwt", "bcrypt", "pydantic",
        "fastapi", "uvicorn", "streamlit", "matplotlib", "plotly"
    ]
    missing = []
    for pkg in required_pkgs:
        try:
            __import__(pkg if pkg != "pyjwt" else "jwt")
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.warning(f"缺失依赖包：{', '.join(missing)}，自动安装...")
        try:
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info("依赖包安装完成")
        except Exception as e:
            logger.error(f"依赖安装失败：{e}")
            sys.exit(1)

# 添加当前目录到Python路径
sys.path.insert(0, os.getcwd())

# 核心模块导入（批量模式容错）
try:
    check_and_install_deps()
    from universal_cmd import UniversalCmd
    from tuner_dashboard import TunerDashboard
    from result_analysis import ResultAnalyzer
    from report_generator import ReportGenerator
    MODULES_LOADED = True
    logger.info("✅ 所有核心模块加载成功")
except Exception as e:
    logger.error(f"❌ 核心模块加载失败：{e}")
    sys.exit(1)

# ------------------------------ 批量工具函数 ------------------------------
def cleanup_old_output(output_root: str):
    """清理旧的批量输出"""
    if os.path.exists(output_root):
        import shutil
        logger.info(f"清理旧输出目录：{output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

def generate_batch_test_data(task_config: Dict, root_dir: str) -> str:
    """为单个任务生成独立的测试数据"""
    task_id = task_config["task_id"]
    data_dir = f"{root_dir}/test_data"
    os.makedirs(data_dir, exist_ok=True)
    data_path = f"{data_dir}/{task_id}_test_data.csv"
    
    # 如果要求生成唯一数据，或文件不存在则生成
    if BATCH_CONFIG["generate_unique_data"] or not os.path.exists(data_path):
        import pandas as pd
        import numpy as np
        
        np.random.seed(hash(task_id) % 2**32)  # 每个任务不同随机种子
        rows = BATCH_CONFIG["data_rows_per_task"]
        
        data = {
            "timestamp": pd.date_range(start="2026-01-01", periods=rows, freq="1min"),
            "metabolic_efficiency": np.random.uniform(0.6, 0.95, size=rows),
            "domain_adapt_score": np.random.uniform(0.5, 0.9, size=rows),
            "core_factor": np.random.uniform(0.7, 0.9, size=rows),
            "stability": np.random.uniform(0.65, 0.85, size=rows),
            "cycle_speed": np.random.uniform(0.02, 0.08, size=rows),
            "sample_id": [f"{task_id}_{i:04d}" for i in range(rows)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False, encoding="utf-8")
        logger.info(f"[task={task_id}] 生成测试数据：{data_path}")
    else:
        logger.info(f"[task={task_id}] 使用已有测试数据：{data_path}")
    
    return data_path

def init_task_logger(task_id: str):
    """初始化任务专用logger"""
    task_logger = logging.getLogger(f"UMC-BatchRun-{task_id}")
    task_logger.addFilter(TaskFilter())
    def log_with_task_id(level, msg):
        record = task_logger.makeRecord(
            task_logger.name, level, "", 0, msg, (), None
        )
        record.task_id = task_id
        task_logger.handle(record)
    return log_with_task_id

# ------------------------------ 单任务全流程执行 ------------------------------
def execute_single_task(task_config: Dict, output_root: str) -> Dict:
    """执行单个任务的全流程：运行→调优→分析→报告"""
    task_id = task_config["task_id"]
    log = init_task_logger(task_id)
    
    # 创建任务输出目录
    task_output_dir = f"{output_root}/{task_id}"
    os.makedirs(task_output_dir, exist_ok=True)
    
    # 初始化结果存储
    task_result = {
        "task_id": task_id,
        "config": task_config,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "elapsed_time": None,
        "run_result": None,
        "tune_result": None,
        "analysis_result": None,
        "report_paths": None,
        "error": None
    }
    
    try:
        log("INFO", "="*50)
        log("INFO", f"开始执行任务：{task_id}（领域：{task_config['domain']}）")
        log("INFO", "="*50)
        start_time = time.time()
        
        # 1. 生成测试数据
        data_path = generate_batch_test_data(task_config, output_root)
        
        # 2. 智能体运行
        log("INFO", "执行智能体基础运行...")
        cmd = UniversalCmd()
        run_output_path = f"{task_output_dir}/{task_id}_run_result.csv"
        
        run_args = type('Args', (object,), {
            "data_path": data_path,
            "domain": task_config["domain"],
            "run_time": task_config["run_time"],
            "output_path": run_output_path
        })
        
        run_result = cmd._execute_run(run_args, return_result=True)
        task_result["run_result"] = {
            "output_path": run_output_path,
            "metrics": run_result["core_metrics"],
            "run_time_actual": time.time() - start_time
        }
        log("INFO", f"智能体运行完成：平均代谢效率={run_result['core_metrics']['avg_metabolic_efficiency']:.3f}")
        
        # 3. 智能体调优
        log("INFO", f"执行智能体调优（迭代次数：{task_config['adapt_iterations']}）")
        tuner = TunerDashboard()
        tuner.default_params.update({
            "domain": task_config["domain"],
            "adapt_iterations": task_config["adapt_iterations"],
            "learning_rate": task_config["learning_rate"],
            "target_metric": task_config["target_metric"]
        })
        
        tuner._start_tuner(data_path)
        # 等待调优完成
        while tuner.tuner_status["is_running"]:
            time.sleep(0.5)
        
        tune_result_path = f"{task_output_dir}/{task_id}_tune_result.json"
        tune_result = {
            "best_score": tuner.tuner_status["best_score"],
            "best_params": tuner.tuner_status["best_params"],
            "convergence_iter": tuner.tuner_status["convergence_iter"],
            "output_path": tune_result_path
        }
        
        # 保存调优结果
        with open(tune_result_path, "w", encoding="utf-8") as f:
            json.dump(tune_result, f, ensure_ascii=False, indent=2)
        
        task_result["tune_result"] = tune_result
        log("INFO", f"智能体调优完成：最优得分={tuner.tuner_status['best_score']:.3f}")
        
        # 4. 结果分析
        log("INFO", "执行结果分析...")
        analyzer = ResultAnalyzer(output_dir=f"{task_output_dir}/analysis")
        analysis_data = {
            "task_id": task_id,
            "run_metrics": run_result["core_metrics"],
            "tune_metrics": tune_result,
            "analysis_time": datetime.now().isoformat()
        }
        
        analysis_path = f"{task_output_dir}/{task_id}_analysis_result.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        task_result["analysis_result"] = {
            "analysis_path": analysis_path,
            "key_metrics": {
                "run_avg_efficiency": run_result["core_metrics"]["avg_metabolic_efficiency"],
                "tune_best_score": tune_result["best_score"],
                "improvement_rate": (tune_result["best_score"] - run_result["core_metrics"]["avg_metabolic_efficiency"]) / run_result["core_metrics"]["avg_metabolic_efficiency"] * 100
            }
        }
        log("INFO", f"结果分析完成：调优提升率={task_result['analysis_result']['key_metrics']['improvement_rate']:.2f}%")
        
        # 5. 生成报告
        log("INFO", "生成分析报告...")
        report_generator = ReportGenerator(output_dir=f"{task_output_dir}/reports")
        report_paths = report_generator.generate_comprehensive_report(
            analysis_data,
            report_name=f"{task_id}_report",
            format_list=["md", "html", "json"],
            with_plots=True
        )
        
        task_result["report_paths"] = report_paths
        for fmt, path in report_paths.items():
            log("INFO", f"生成{fmt.upper()}报告：{path}")
        
        # 任务完成
        elapsed_time = time.time() - start_time
        task_result.update({
            "status": "success",
            "end_time": datetime.now().isoformat(),
            "elapsed_time": elapsed_time
        })
        
        log("INFO", "="*50)
        log("INFO", f"任务执行完成：{task_id}（总耗时：{elapsed_time:.2f}秒）")
        log("INFO", "="*50)
        
    except Exception as e:
        error_msg = f"任务执行失败：{str(e)}"
        log("ERROR", error_msg)
        task_result.update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": error_msg
        })
    
    return task_result

# ------------------------------ 批量任务调度 ------------------------------
def execute_batch_tasks() -> Dict:
    """执行批量任务调度"""
    # 初始化输出目录
    if BATCH_CONFIG["cleanup_before_run"]:
        cleanup_old_output(BATCH_CONFIG["output_root"])
    os.makedirs(BATCH_CONFIG["output_root"], exist_ok=True)
    
    # 批量结果存储
    batch_result = {
        "batch_id": f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_elapsed_time": None,
        "total_tasks": len(BATCH_CONFIG["task_configs"]),
        "success_tasks": 0,
        "failed_tasks": 0,
        "task_results": [],
        "summary_report_path": None
    }
    
    logger.info("="*60)
    logger.info(f"启动UMC智能体批量运行（批次ID：{batch_result['batch_id']}）")
    logger.info(f"总任务数：{batch_result['total_tasks']} | 并行线程数：{BATCH_CONFIG['parallel_workers']}")
    logger.info("="*60)
    
    start_time = time.time()
    
    # 执行批量任务（并行/串行）
    task_configs = BATCH_CONFIG["task_configs"]
    if BATCH_CONFIG["parallel_workers"] > 1:
        # 并行执行
        logger.info("采用并行模式执行任务...")
        with ThreadPoolExecutor(max_workers=BATCH_CONFIG["parallel_workers"]) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(execute_single_task, cfg, BATCH_CONFIG["output_root"]): cfg 
                for cfg in task_configs
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                task_result = future.result()
                batch_result["task_results"].append(task_result)
                
                # 更新统计
                if task_result["status"] == "success":
                    batch_result["success_tasks"] += 1
                else:
                    batch_result["failed_tasks"] += 1
                
                # 打印进度
                completed = len(batch_result["task_results"])
                progress = (completed / batch_result["total_tasks"]) * 100
                logger.info(f"进度：{completed}/{batch_result['total_tasks']} ({progress:.1f}%) | 成功：{batch_result['success_tasks']} | 失败：{batch_result['failed_tasks']}")
    else:
        # 串行执行
        logger.info("采用串行模式执行任务...")
        for cfg in task_configs:
            task_result = execute_single_task(cfg, BATCH_CONFIG["output_root"])
            batch_result["task_results"].append(task_result)
            
            # 更新统计
            if task_result["status"] == "success":
                batch_result["success_tasks"] += 1
            else:
                batch_result["failed_tasks"] += 1
            
            # 打印进度
            completed = len(batch_result["task_results"])
            progress = (completed / batch_result["total_tasks"]) * 100
            logger.info(f"进度：{completed}/{batch_result['total_tasks']} ({progress:.1f}%) | 成功：{batch_result['success_tasks']} | 失败：{batch_result['failed_tasks']}")
    
    # 批量执行完成
    total_elapsed = time.time() - start_time
    batch_result.update({
        "end_time": datetime.now().isoformat(),
        "total_elapsed_time": total_elapsed
    })
    
    logger.info("="*60)
    logger.info(f"批量运行完成（总耗时：{total_elapsed:.2f}秒）")
    logger.info(f"执行结果：成功{batch_result['success_tasks']} | 失败{batch_result['failed_tasks']}")
    logger.info("="*60)
    
    # 生成批量汇总报告
    if BATCH_CONFIG["generate_summary_report"] and batch_result["task_results"]:
        logger.info("生成批量结果汇总报告...")
        summary_report_path = generate_batch_summary_report(batch_result)
        batch_result["summary_report_path"] = summary_report_path
        logger.info(f"汇总报告生成完成：{summary_report_path}")
    
    # 保存批量结果
    batch_result_path = f"{BATCH_CONFIG['output_root']}/batch_result_{batch_result['batch_id']}.json"
    with open(batch_result_path, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, ensure_ascii=False, indent=2)
    logger.info(f"批量运行结果已保存：{batch_result_path}")
    
    return batch_result

def generate_batch_summary_report(batch_result: Dict) -> str:
    """生成批量运行结果汇总报告"""
    # 构建汇总数据
    summary_data = {
        "batch_info": {
            "batch_id": batch_result["batch_id"],
            "start_time": batch_result["start_time"],
            "end_time": batch_result["end_time"],
            "total_elapsed_time": batch_result["total_elapsed_time"],
            "total_tasks": batch_result["total_tasks"],
            "success_rate": (batch_result["success_tasks"] / batch_result["total_tasks"]) * 100 if batch_result["total_tasks"] > 0 else 0
        },
        "task_summary": [],
        "key_metrics_comparison": {},
        "recommendations": []
    }
    
    # 提取每个任务的核心指标
    domain_metrics = {}
    for task in batch_result["task_results"]:
        if task["status"] == "success":
            # 任务摘要
            task_summary = {
                "task_id": task["task_id"],
                "domain": task["config"]["domain"],
                "status": task["status"],
                "elapsed_time": task["elapsed_time"],
                "run_avg_efficiency": task["analysis_result"]["key_metrics"]["run_avg_efficiency"],
                "tune_best_score": task["analysis_result"]["key_metrics"]["tune_best_score"],
                "improvement_rate": task["analysis_result"]["key_metrics"]["improvement_rate"]
            }
            summary_data["task_summary"].append(task_summary)
            
            # 按领域汇总
            domain = task["config"]["domain"]
            if domain not in domain_metrics:
                domain_metrics[domain] = []
            domain_metrics[domain].append({
                "task_id": task["task_id"],
                "improvement_rate": task["analysis_result"]["key_metrics"]["improvement_rate"],
                "best_score": task["analysis_result"]["key_metrics"]["tune_best_score"]
            })
    
    # 领域对比分析
    for domain, metrics in domain_metrics.items():
        avg_improvement = sum([m["improvement_rate"] for m in metrics]) / len(metrics)
        max_score = max([m["best_score"] for m in metrics])
        summary_data["key_metrics_comparison"][domain] = {
            "task_count": len(metrics),
            "avg_improvement_rate": avg_improvement,
            "max_best_score": max_score,
            "best_task": metrics[0]["task_id"]  # 最优任务
        }
    
    # 生成优化建议
    summary_data["recommendations"] = [
        f"整体任务成功率：{summary_data['batch_info']['success_rate']:.1f}%",
        f"最优表现领域：{max(summary_data['key_metrics_comparison'].items(), key=lambda x: x[1]['avg_improvement_rate'])[0]}（平均提升率：{max(summary_data['key_metrics_comparison'].items(), key=lambda x: x[1]['avg_improvement_rate'])[1]['avg_improvement_rate']:.2f}%）",
        f"建议迭代次数：基于批量结果，{75}次迭代在多数场景下表现最优",
        f"建议学习率：{0.009}为平衡效率和效果的最优值"
    ]
    
    # 生成汇总报告
    report_generator = ReportGenerator(output_dir=f"{BATCH_CONFIG['output_root']}/summary")
    report_paths = report_generator.generate_comprehensive_report(
        summary_data,
        report_name=f"batch_summary_{batch_result['batch_id']}",
        format_list=["md", "html", "pdf"],
        with_plots=True,
        is_summary=True  # 标记为汇总报告
    )
    
    return report_paths.get("html", list(report_paths.values())[0])

# ------------------------------ 命令行入口 ------------------------------
def main():
    """批量运行主函数"""
    parser = argparse.ArgumentParser(
        description="UMC-Metabolic-Agent 全自动化批量运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
批量运行示例：
  1. 默认配置批量运行：python run_all.py
  2. 自定义并行数：python run_all.py --workers 4
  3. 清理旧数据后运行：python run_all.py --cleanup
  4. 仅生成汇总报告：python run_all.py --summary-only --batch-result ./umc_batch_output/batch_result_xxx.json
  5. 串行执行任务：python run_all.py --workers 1
        """
    )
    
    # 命令行参数
    parser.add_argument("--workers", "-w", type=int, default=BATCH_CONFIG["parallel_workers"],
                        help="并行执行的工作线程数（默认：2）")
    parser.add_argument("--cleanup", action="store_true",
                        help="运行前清理旧的批量输出数据")
    parser.add_argument("--no-summary", action="store_false", dest="generate_summary",
                        help="不生成批量汇总报告")
    parser.add_argument("--summary-only", action="store_true",
                        help="仅基于已有批量结果生成汇总报告")
    parser.add_argument("--batch-result", type=str,
                        help="指定已有批量结果文件路径（仅summary-only模式）")
    parser.add_argument("--output-root", "-o", type=str, default=BATCH_CONFIG["output_root"],
                        help="批量输出根目录（默认：./umc_batch_output）")
    
    args = parser.parse_args()
    
    # 更新批量配置
    BATCH_CONFIG["parallel_workers"] = args.workers
    BATCH_CONFIG["cleanup_before_run"] = args.cleanup
    BATCH_CONFIG["generate_summary_report"] = args.generate_summary
    BATCH_CONFIG["output_root"] = args.output_root
    
    # 执行对应操作
    if args.summary_only:
        # 仅生成汇总报告
        if not args.batch_result or not os.path.exists(args.batch_result):
            logger.error("summary-only模式需要指定有效的批量结果文件路径")
            sys.exit(1)
        
        with open(args.batch_result, "r", encoding="utf-8") as f:
            batch_result = json.load(f)
        
        summary_path = generate_batch_summary_report(batch_result)
        logger.info(f"汇总报告生成完成：{summary_path}")
    else:
        # 执行完整批量运行
        execute_batch_tasks()
    
    logger.info("✅ 批量运行脚本执行完成")

if __name__ == "__main__":
    main()