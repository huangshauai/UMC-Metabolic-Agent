# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 通用批量处理脚本
核心逻辑：轻量化批量任务处理，支持自定义任务列表、多任务类型、灵活并行，适配通用批量场景
设计原则：配置灵活、易用性强、输出简洁、扩展方便，兼顾新手和进阶用户
"""
import os
import sys
import json
import time
import logging
import warnings
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------ 基础配置与日志 ------------------------------
# 颜色输出工具（增强可读性）
class Color:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    RESET = '\033[0m'

# 日志配置（轻量化，带任务标识）
logging.basicConfig(
    level=logging.INFO,
    format=f"{Color.BLUE}[%(asctime)s]{Color.RESET} [{Color.PURPLE}%(task_id)s{Color.RESET}] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("UMC-BatchProcess")
warnings.filterwarnings("ignore")

# 默认批量配置（可通过命令行/配置文件覆盖）
DEFAULT_CONFIG = {
    "output_dir": "./umc_batch_process_output",  # 批量输出根目录
    "parallel_workers": 2,                       # 并行工作线程数
    "task_type": "tune",                         # 默认任务类型：run/tune/analyze/all
    "generate_data": True,                       # 是否自动生成测试数据
    "data_rows": 1000,                           # 每个任务的测试数据行数
    "save_individual_report": True,              # 是否保存单个任务报告
    "save_batch_summary": True,                  # 是否保存批量汇总
    "overwrite": False,                          # 是否覆盖已有结果
    
    # 批量任务列表（支持多参数/多领域组合）
    "tasks": [
        {"task_id": "task_quantum_001", "domain": "quantum", "iter": 30, "lr": 0.01},
        {"task_id": "task_quantum_002", "domain": "quantum", "iter": 50, "lr": 0.01},
        {"task_id": "task_biology_001", "domain": "biology", "iter": 30, "lr": 0.008},
        {"task_id": "task_chemistry_001", "domain": "chemistry", "iter": 40, "lr": 0.015}
    ]
}

# ------------------------------ 依赖检查与模块导入 ------------------------------
def check_dependencies():
    """检查并自动安装核心依赖"""
    required = ["pandas", "numpy", "matplotlib"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.warning(f"{Color.YELLOW}缺失依赖：{', '.join(missing)}，自动安装...{Color.RESET}")
        try:
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                stdout=subprocess.DEVNULL
            )
            logger.info(f"{Color.GREEN}依赖安装完成{Color.RESET}")
        except Exception as e:
            logger.error(f"{Color.RED}依赖安装失败：{e}{Color.RESET}")
            sys.exit(1)

# 添加当前目录到Python路径
sys.path.insert(0, os.getcwd())

# 核心模块导入
try:
    check_dependencies()
    from universal_cmd import UniversalCmd
    from tuner_dashboard import TunerDashboard
    from result_analysis import ResultAnalyzer
    MODULE_LOADED = True
    logger.info(f"{Color.GREEN}✅ 核心模块导入成功{Color.RESET}")
except Exception as e:
    logger.error(f"{Color.RED}❌ 模块导入失败：{e}{Color.RESET}")
    logger.error(f"{Color.YELLOW}请确保核心文件（universal_cmd.py/tuner_dashboard.py）在当前目录{Color.RESET}")
    sys.exit(1)

# ------------------------------ 通用工具函数 ------------------------------
def load_config(config_path: str = None) -> dict:
    """加载配置文件（JSON格式），无则使用默认配置"""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                custom_config = json.load(f)
            config.update(custom_config)
            logger.info(f"{Color.BLUE}📄 加载自定义配置：{config_path}{Color.RESET}")
        except Exception as e:
            logger.error(f"{Color.RED}加载配置文件失败：{e}，使用默认配置{Color.RESET}")
    return config

def ensure_dir(dir_path: str):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"{Color.BLUE}📁 创建目录：{dir_path}{Color.RESET}")

def generate_task_data(task_id: str, config: dict) -> str:
    """为单个任务生成测试数据"""
    data_dir = f"{config['output_dir']}/test_data"
    ensure_dir(data_dir)
    data_path = f"{data_dir}/{task_id}_data.csv"
    
    # 如果文件已存在且不覆盖，直接返回
    if os.path.exists(data_path) and not config["overwrite"]:
        logger.info(f"{Color.BLUE}📄 任务{task_id}使用已有数据：{data_path}{Color.RESET}")
        return data_path
    
    # 生成测试数据
    np.random.seed(hash(task_id) % 2**32)  # 每个任务独立随机种子
    data = {
        "timestamp": pd.date_range(start="2026-01-01", periods=config["data_rows"], freq="1min"),
        "metabolic_efficiency": np.random.uniform(0.6, 0.95, size=config["data_rows"]),
        "domain_adapt_score": np.random.uniform(0.5, 0.9, size=config["data_rows"]),
        "core_factor": np.random.uniform(0.7, 0.9, size=config["data_rows"]),
        "stability": np.random.uniform(0.65, 0.85, size=config["data_rows"]),
        "sample_id": [f"{task_id}_{i:04d}" for i in range(config["data_rows"])]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False, encoding="utf-8")
    logger.info(f"{Color.GREEN}✅ 任务{task_id}生成数据：{data_path}（{config['data_rows']}行）{Color.RESET}")
    return data_path

def get_task_logger(task_id: str):
    """获取带任务ID的logger"""
    task_logger = logging.getLogger(f"BatchProcess-{task_id}")
    def log(msg, level="info"):
        if level == "info":
            task_logger.info(msg, extra={"task_id": task_id})
        elif level == "error":
            task_logger.error(msg, extra={"task_id": task_id})
        elif level == "warning":
            task_logger.warning(msg, extra={"task_id": task_id})
    return log

# ------------------------------ 单个任务处理函数 ------------------------------
def process_single_task(task: dict, config: dict) -> dict:
    """处理单个任务（支持不同任务类型）"""
    task_id = task["task_id"]
    log = get_task_logger(task_id)
    task_result = {
        "task_id": task_id,
        "config": task,
        "status": "failed",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "elapsed_time": 0,
        "metrics": {},
        "error": None
    }
    
    try:
        # 1. 准备任务目录和数据
        task_dir = f"{config['output_dir']}/{task_id}"
        ensure_dir(task_dir)
        data_path = generate_task_data(task_id, config) if config["generate_data"] else task.get("data_path")
        
        if not data_path or not os.path.exists(data_path):
            raise ValueError(f"任务{task_id}数据文件不存在：{data_path}")
        
        log(f"{Color.CYAN}🚀 开始处理任务（类型：{config['task_type']}）{Color.RESET}")
        start_time = time.time()
        
        # 2. 根据任务类型执行不同操作
        if config["task_type"] == "run" or config["task_type"] == "all":
            # 执行智能体运行
            log(f"{Color.BLUE}🔄 执行智能体运行（领域：{task['domain']}）{Color.RESET}")
            cmd = UniversalCmd()
            run_output = f"{task_dir}/{task_id}_run_result.csv"
            
            run_args = type('Args', (object,), {
                "data_path": data_path,
                "domain": task["domain"],
                "run_time": task.get("run_time", 60),
                "output_path": run_output
            })
            
            run_res = cmd._execute_run(run_args, return_result=True)
            task_result["metrics"]["run"] = {
                "avg_metabolic_efficiency": run_res["core_metrics"]["avg_metabolic_efficiency"],
                "domain_adapt_score": run_res["core_metrics"]["domain_adapt_score"],
                "stability_score": run_res["core_metrics"]["stability_score"],
                "output_path": run_output
            }
            log(f"{Color.GREEN}✅ 运行完成：平均代谢效率={run_res['core_metrics']['avg_metabolic_efficiency']:.3f}{Color.RESET}")
        
        if config["task_type"] == "tune" or config["task_type"] == "all":
            # 执行智能体调优
            log(f"{Color.BLUE}🔧 执行智能体调优（迭代：{task['iter']}，学习率：{task['lr']}）{Color.RESET}")
            tuner = TunerDashboard()
            tuner.default_params.update({
                "domain": task["domain"],
                "adapt_iterations": task["iter"],
                "learning_rate": task["lr"],
                "target_metric": "metabolic_efficiency"
            })
            
            tuner._start_tuner(data_path)
            # 等待调优完成
            while tuner.tuner_status["is_running"]:
                time.sleep(0.5)
            
            # 记录调优结果
            tune_result = {
                "best_score": tuner.tuner_status["best_score"],
                "convergence_iter": tuner.tuner_status["convergence_iter"],
                "stability_score": tuner.tuner_status["stability_score"],
                "best_params": tuner.tuner_status["best_params"]
            }
            task_result["metrics"]["tune"] = tune_result
            log(f"{Color.GREEN}✅ 调优完成：最优得分={tune_result['best_score']:.3f}{Color.RESET}")
            
            # 保存调优结果
            tune_output = f"{task_dir}/{task_id}_tune_result.json"
            with open(tune_output, "w", encoding="utf-8") as f:
                json.dump(tune_result, f, indent=2)
            task_result["metrics"]["tune"]["output_path"] = tune_output
        
        if config["task_type"] == "analyze" or config["task_type"] == "all":
            # 执行结果分析
            log(f"{Color.BLUE}📊 执行结果分析{Color.RESET}")
            analyzer = ResultAnalyzer(output_dir=f"{task_dir}/analysis")
            analysis_data = {
                "task_id": task_id,
                "metrics": task_result["metrics"],
                "analysis_time": datetime.now().isoformat()
            }
            
            # 保存分析结果
            analysis_output = f"{task_dir}/{task_id}_analysis.json"
            with open(analysis_output, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2)
            task_result["metrics"]["analysis"] = {"output_path": analysis_output}
            log(f"{Color.GREEN}✅ 分析完成：结果保存至{analysis_output}{Color.RESET}")
        
        # 3. 任务完成处理
        elapsed_time = time.time() - start_time
        task_result.update({
            "status": "success",
            "end_time": datetime.now().isoformat(),
            "elapsed_time": round(elapsed_time, 2)
        })
        
        log(f"{Color.GREEN}🎉 任务处理完成（耗时：{elapsed_time:.2f}秒）{Color.RESET}")
        
    except Exception as e:
        error_msg = str(e)
        task_result["error"] = error_msg
        log(f"{Color.RED}❌ 任务处理失败：{error_msg}{Color.RESET}", level="error")
    
    return task_result

# ------------------------------ 批量任务调度 ------------------------------
def run_batch_process(config: dict):
    """执行批量处理"""
    # 初始化输出目录
    ensure_dir(config["output_dir"])
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"{Color.PURPLE}📦 启动批量处理（批次ID：{batch_id}）{Color.RESET}")
    logger.info(f"{Color.BLUE}📋 批量配置：任务数={len(config['tasks'])} | 并行数={config['parallel_workers']} | 任务类型={config['task_type']}{Color.RESET}")
    
    # 存储批量结果
    batch_result = {
        "batch_id": batch_id,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_tasks": len(config["tasks"]),
        "success_tasks": 0,
        "failed_tasks": 0,
        "task_results": [],
        "summary": {}
    }
    
    try:
        # 1. 执行批量任务
        if config["parallel_workers"] > 1:
            # 并行执行
            logger.info(f"{Color.BLUE}⚡ 采用并行模式执行任务（{config['parallel_workers']}线程）{Color.RESET}")
            with ThreadPoolExecutor(max_workers=config["parallel_workers"]) as executor:
                futures = {executor.submit(process_single_task, task, config): task for task in config["tasks"]}
                
                for future in as_completed(futures):
                    task_res = future.result()
                    batch_result["task_results"].append(task_res)
                    
                    # 更新统计
                    if task_res["status"] == "failed":
                        batch_result["failed_tasks"] += 1
                    else:
                        batch_result["success_tasks"] += 1
                    
                    # 打印进度
                    completed = len(batch_result["task_results"])
                    progress = (completed / config["tasks"]) * 100
                    logger.info(f"{Color.YELLOW}📊 进度：{completed}/{len(config['tasks'])} ({progress:.1f}%) | 成功：{batch_result['success_tasks']} | 失败：{batch_result['failed_tasks']}{Color.RESET}")
        else:
            # 串行执行
            logger.info(f"{Color.BLUE}📶 采用串行模式执行任务{Color.RESET}")
            for task in config["tasks"]:
                task_res = process_single_task(task, config)
                batch_result["task_results"].append(task_res)
                
                if task_res["status"] == "failed":
                    batch_result["failed_tasks"] += 1
                else:
                    batch_result["success_tasks"] += 1
                
                completed = len(batch_result["task_results"])
                progress = (completed / len(config["tasks"])) * 100
                logger.info(f"{Color.YELLOW}📊 进度：{completed}/{len(config['tasks'])} ({progress:.1f}%) | 成功：{batch_result['success_tasks']} | 失败：{batch_result['failed_tasks']}{Color.RESET}")
        
        # 2. 生成批量汇总
        if config["save_batch_summary"]:
            logger.info(f"{Color.BLUE}📈 生成批量汇总报告{Color.RESET}")
            # 计算汇总指标
            success_rate = (batch_result["success_tasks"] / batch_result["total_tasks"]) * 100 if batch_result["total_tasks"] > 0 else 0
            avg_elapsed = 0
            domain_metrics = {}
            
            for task_res in batch_result["task_results"]:
                if task_res["status"] != "failed":
                    avg_elapsed += task_res["elapsed_time"]
                    domain = task_res["config"]["domain"]
                    if domain not in domain_metrics:
                        domain_metrics[domain] = {"count": 0, "avg_score": 0}
                    
                    # 汇总调优/运行指标
                    if "tune" in task_res["metrics"]:
                        domain_metrics[domain]["avg_score"] += task_res["metrics"]["tune"]["best_score"]
                    elif "run" in task_res["metrics"]:
                        domain_metrics[domain]["avg_score"] += task_res["metrics"]["run"]["avg_metabolic_efficiency"]
                    domain_metrics[domain]["count"] += 1
            
            # 计算领域平均得分
            for domain in domain_metrics:
                if domain_metrics[domain]["count"] > 0:
                    domain_metrics[domain]["avg_score"] /= domain_metrics[domain]["count"]
            
            avg_elapsed = avg_elapsed / batch_result["success_tasks"] if batch_result["success_tasks"] > 0 else 0
            
            # 构建汇总数据
            batch_result["summary"] = {
                "success_rate": round(success_rate, 2),
                "avg_elapsed_time": round(avg_elapsed, 2),
                "domain_metrics": domain_metrics,
                "top_task": None
            }
            
            # 找出最优任务
            top_score = 0
            top_task = None
            for task_res in batch_result["task_results"]:
                if task_res["status"] == "success":
                    if "tune" in task_res["metrics"]:
                        score = task_res["metrics"]["tune"]["best_score"]
                    else:
                        score = task_res["metrics"]["run"]["avg_metabolic_efficiency"]
                    
                    if score > top_score:
                        top_score = score
                        top_task = task_res["task_id"]
            
            batch_result["summary"]["top_task"] = top_task
            batch_result["summary"]["top_score"] = round(top_score, 3) if top_task else 0
            
            # 保存汇总文件
            summary_path = f"{config['output_dir']}/batch_summary_{batch_id}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(batch_result["summary"], f, ensure_ascii=False, indent=2)
            logger.info(f"{Color.GREEN}✅ 批量汇总已保存：{summary_path}{Color.RESET}")
        
        # 3. 完成批量处理
        batch_result["end_time"] = datetime.now().isoformat()
        total_elapsed = (datetime.fromisoformat(batch_result["end_time"]) - 
                         datetime.fromisoformat(batch_result["start_time"])).total_seconds()
        
        # 保存批量结果
        result_path = f"{config['output_dir']}/batch_result_{batch_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2)
        
        # 打印批量结果
        logger.info(f"{Color.PURPLE}========== 批量处理完成 =========={Color.RESET}")
        logger.info(f"{Color.GREEN}📊 批量结果：{Color.RESET}")
        logger.info(f"   总任务数：{batch_result['total_tasks']}")
        logger.info(f"   成功数：{batch_result['success_tasks']} | 失败数：{batch_result['failed_tasks']}")
        logger.info(f"   成功率：{success_rate:.2f}%")
        logger.info(f"   平均耗时：{avg_elapsed:.2f}秒/任务")
        logger.info(f"   总耗时：{total_elapsed:.2f}秒")
        logger.info(f"   最优任务：{top_task}（得分：{top_score:.3f}）")
        logger.info(f"   结果文件：{result_path}")
        logger.info(f"{Color.PURPLE}=================================={Color.RESET}")
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 批量处理失败：{e}{Color.RESET}")
        batch_result["error"] = str(e)
    
    return batch_result

# ------------------------------ 命令行入口 ------------------------------
def main():
    """批量处理主函数"""
    parser = argparse.ArgumentParser(
        description="UMC-Metabolic-Agent 通用批量处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Color.CYAN}使用示例：{Color.RESET}
  1. 默认配置运行：
     python batch_process.py
     
  2. 使用自定义配置文件：
     python batch_process.py --config my_config.json
     
  3. 指定任务类型和并行数：
     python batch_process.py --task-type run --workers 4
     
  4. 仅调优任务，覆盖已有结果：
     python batch_process.py --task-type tune --overwrite --workers 2
     
  5. 自定义输出目录：
     python batch_process.py --output ./my_batch_output --task-type all
     
{Color.CYAN}配置文件格式（JSON）：{Color.RESET}
{{
  "output_dir": "./my_output",
  "parallel_workers": 2,
  "task_type": "tune",
  "tasks": [
    {{"task_id": "task1", "domain": "quantum", "iter": 50, "lr": 0.01}},
    {{"task_id": "task2", "domain": "biology", "iter": 40, "lr": 0.008}}
  ]
}}
        """
    )
    
    # 命令行参数
    parser.add_argument("--config", "-c", type=str, help="自定义配置文件路径（JSON）")
    parser.add_argument("--task-type", "-t", type=str, choices=["run", "tune", "analyze", "all"],
                        help="任务类型：run(仅运行)/tune(仅调优)/analyze(仅分析)/all(全流程)")
    parser.add_argument("--workers", "-w", type=int, help="并行工作线程数")
    parser.add_argument("--output", "-o", type=str, help="输出目录路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有结果文件")
    parser.add_argument("--no-data", action="store_false", dest="generate_data", help="不自动生成测试数据")
    parser.add_argument("--no-summary", action="store_false", dest="save_batch_summary", help="不生成批量汇总")
    
    args = parser.parse_args()
    
    # 加载配置并覆盖命令行参数
    config = load_config(args.config)
    if args.task_type:
        config["task_type"] = args.task_type
    if args.workers:
        config["parallel_workers"] = args.workers
    if args.output:
        config["output_dir"] = args.output
    if args.overwrite:
        config["overwrite"] = args.overwrite
    if args.generate_data is not None:
        config["generate_data"] = args.generate_data
    if args.save_batch_summary is not None:
        config["save_batch_summary"] = args.save_batch_summary
    
    # 执行批量处理
    run_batch_process(config)

if __name__ == "__main__":
    main()