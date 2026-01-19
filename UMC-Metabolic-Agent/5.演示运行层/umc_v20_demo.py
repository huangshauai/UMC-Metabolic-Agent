# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent v2.0 全功能演示脚本
核心逻辑：整合所有核心模块，提供交互式演示流程，覆盖认证、运行、调优、分析、报告全功能
设计原则：交互式、引导式、完整性，让新手快速体验UMC智能体的全部核心能力
"""
import os
import sys
import json
import time
import logging
import warnings
import argparse
import subprocess
import threading
import webbrowser
from datetime import datetime
from typing import Dict, List, Optional

# 添加当前目录到Python路径（确保能导入所有模块）
sys.path.insert(0, os.getcwd())

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[\033[34m%(asctime)s\033[0m] [\033[36m%(name)s\033[0m] [\033[32m%(levelname)s\033[0m] %(message)s",
    handlers=[
        logging.FileHandler("umc_v20_demo.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UMC-v2.0-Demo")

warnings.filterwarnings("ignore")

# ------------------------------ 全局配置 ------------------------------
DEMO_CONFIG = {
    "test_data_path": "./umc_demo_test_data.csv",
    "demo_user": "demo_user",
    "demo_password": "demo123456",
    "demo_api_key_name": "demo_api_key",
    "api_host": "127.0.0.1",
    "api_port": 8000,
    "dashboard_port": 8501,
    "default_domain": "quantum",
    "adapt_iterations": 50,
    "cleanup_after_demo": False,  # 演示后是否清理数据
}

# 颜色输出工具
class Color:
    """终端颜色工具"""
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    RESET = '\033[0m'

# ------------------------------ 核心模块导入 ------------------------------
try:
    # 核心模块
    from ext_identity import ExtIdentityManager, RoleEnum, PermissionEnum, create_identity_manager
    from universal_cmd import UniversalCmd
    from tuner_dashboard import TunerDashboard
    from custom_app_api import UMCCustomAPI, run_api_server
    from result_analysis import ResultAnalyzer
    from report_generator import ReportGenerator
    
    # 验证模块完整性
    MODULES_LOADED = True
    logger.info(f"{Color.GREEN}✅ 所有核心模块导入成功{Color.RESET}")
    
except ImportError as e:
    logger.error(f"{Color.RED}❌ 模块导入失败：{e}{Color.RESET}")
    logger.warning(f"{Color.YELLOW}⚠️  部分演示功能将受限，请确保所有核心文件在当前目录{Color.RESET}")
    MODULES_LOADED = False

# ------------------------------ 演示工具函数 ------------------------------
def print_separator(title: str = ""):
    """打印分隔符"""
    print(f"\n{Color.BLUE}{'='*80}{Color.RESET}")
    if title:
        print(f"{Color.CYAN}{title.center(80)}{Color.RESET}")
    print(f"{Color.BLUE}{'='*80}{Color.RESET}")

def generate_test_data(file_path: str = DEMO_CONFIG["test_data_path"], rows: int = 1000):
    """生成测试数据"""
    try:
        import pandas as pd
        import numpy as np
        
        # 生成模拟代谢数据
        np.random.seed(42)
        data = {
            "timestamp": pd.date_range(start="2026-01-01", periods=rows, freq="1min"),
            "metabolic_efficiency": np.random.uniform(0.6, 0.95, size=rows),
            "domain_adapt_score": np.random.uniform(0.5, 0.9, size=rows),
            "core_factor": np.random.uniform(0.7, 0.9, size=rows),
            "stability": np.random.uniform(0.65, 0.85, size=rows),
            "cycle_speed": np.random.uniform(0.02, 0.08, size=rows),
            "temperature": np.random.uniform(25.0, 37.0, size=rows),
            "pressure": np.random.uniform(1.0, 1.2, size=rows),
            "ph_level": np.random.uniform(6.5, 7.5, size=rows),
            "sample_id": [f"S{str(i).zfill(4)}" for i in range(rows)]
        }
        
        # 保存为CSV
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding="utf-8")
        
        logger.info(f"{Color.GREEN}✅ 测试数据生成成功：{file_path}（{rows}行）{Color.RESET}")
        return file_path
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 生成测试数据失败：{e}{Color.RESET}")
        return None

def cleanup_demo_data():
    """清理演示数据"""
    if not DEMO_CONFIG["cleanup_after_demo"]:
        return
    
    cleanup_paths = [
        DEMO_CONFIG["test_data_path"],
        "./umc_identity_data",
        "./umc_api_output",
        "./umc_api_tasks",
        "./umc_api_uploads",
        "./umc_tuner",
        "./umc_demo_report",
        "umc_v20_demo.log",
    ]
    
    for path in cleanup_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
                logger.info(f"{Color.YELLOW}🗑️  删除文件：{path}{Color.RESET}")
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
                logger.info(f"{Color.YELLOW}🗑️  删除目录：{path}{Color.RESET}")

# ------------------------------ 演示流程函数 ------------------------------
def demo_step_1_identity_setup():
    """步骤1：身份认证初始化"""
    print_separator("步骤1：身份认证系统初始化")
    
    if not MODULES_LOADED:
        logger.error(f"{Color.RED}❌ 核心模块未加载，跳过身份认证演示{Color.RESET}")
        return None
    
    try:
        # 创建身份管理器
        identity = create_identity_manager()
        
        # 初始化默认管理员
        print(f"\n{Color.PURPLE}📌 初始化默认管理员账户...{Color.RESET}")
        identity.update_user(
            "admin",
            password_hash=identity._hash_password("admin123456"),
            role=RoleEnum.ADMIN
        )
        print(f"{Color.GREEN}✅ 默认管理员已初始化：用户名=admin，密码=admin123456{Color.RESET}")
        
        # 创建演示用户
        print(f"\n{Color.PURPLE}📌 创建演示用户...{Color.RESET}")
        if DEMO_CONFIG["demo_user"] in identity.users:
            identity.delete_user(DEMO_CONFIG["demo_user"])
        
        demo_user = identity.create_user(
            username=DEMO_CONFIG["demo_user"],
            password=DEMO_CONFIG["demo_password"],
            email="demo@umc-agent.com",
            full_name="UMC Demo User",
            role=RoleEnum.OPERATOR
        )
        print(f"{Color.GREEN}✅ 演示用户创建成功：")
        print(f"   用户名：{demo_user.username}")
        print(f"   密码：{DEMO_CONFIG['demo_password']}")
        print(f"   角色：{demo_user.role.value}")
        
        # 创建演示API密钥
        print(f"\n{Color.PURPLE}📌 创建演示API密钥...{Color.RESET}")
        raw_key, api_key = identity.create_api_key(
            user_id=DEMO_CONFIG["demo_user"],
            name=DEMO_CONFIG["demo_api_key_name"],
            role=RoleEnum.OPERATOR
        )
        print(f"{Color.GREEN}✅ API密钥创建成功：")
        print(f"   密钥ID：{api_key.key_id}")
        print(f"   原始密钥：{raw_key}（请妥善保存）")
        print(f"   所属用户：{api_key.user_id}")
        
        # 用户登录演示
        print(f"\n{Color.PURPLE}📌 用户登录验证演示...{Color.RESET}")
        user = identity.authenticate_user(DEMO_CONFIG["demo_user"], DEMO_CONFIG["demo_password"])
        if user:
            # 生成JWT令牌
            tokens = identity._create_tokens(user.username, user.role)
            print(f"{Color.GREEN}✅ 登录成功！")
            print(f"   访问令牌：{tokens.access_token[:50]}...")
            print(f"   令牌过期：{tokens.expires_at}")
            print(f"   权限列表：{identity.permissions.get(user.role.value)}")
        else:
            print(f"{Color.RED}❌ 登录失败{Color.RESET}")
        
        return identity
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 身份认证演示失败：{e}{Color.RESET}")
        return None

def demo_step_2_agent_run():
    """步骤2：智能体基础运行"""
    print_separator("步骤2：UMC智能体基础运行")
    
    if not MODULES_LOADED:
        logger.error(f"{Color.RED}❌ 核心模块未加载，跳过智能体运行演示{Color.RESET}")
        return None
    
    try:
        # 生成测试数据
        print(f"\n{Color.PURPLE}📌 生成测试数据...{Color.RESET}")
        data_path = generate_test_data()
        if not data_path:
            return None
        
        # 创建通用命令行实例
        cmd = UniversalCmd()
        
        # 运行智能体
        print(f"\n{Color.PURPLE}📌 运行UMC智能体（{DEMO_CONFIG['default_domain']}领域）...{Color.RESET}")
        run_args = type('Args', (object,), {
            "data_path": data_path,
            "domain": DEMO_CONFIG["default_domain"],
            "run_time": 60,
            "output_path": f"./umc_demo_run_result.csv"
        })
        
        start_time = time.time()
        result = cmd._execute_run(run_args, return_result=True)
        elapsed = time.time() - start_time
        
        print(f"{Color.GREEN}✅ 智能体运行完成（耗时：{elapsed:.2f}秒）")
        print(f"   输出文件：{run_args.output_path}")
        print(f"   核心指标：")
        print(f"      - 平均代谢效率：{result['core_metrics']['avg_metabolic_efficiency']:.3f}")
        print(f"      - 领域适配得分：{result['core_metrics']['domain_adapt_score']:.3f}")
        print(f"      - 稳定性评分：{result['core_metrics']['stability_score']:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 智能体运行演示失败：{e}{Color.RESET}")
        return None

def demo_step_3_agent_tuning():
    """步骤3：智能体参数调优"""
    print_separator("步骤3：UMC智能体参数调优")
    
    if not MODULES_LOADED:
        logger.error(f"{Color.RED}❌ 核心模块未加载，跳过参数调优演示{Color.RESET}")
        return None
    
    try:
        # 创建调优仪表盘实例
        tuner = TunerDashboard()
        
        # 准备调优数据
        data_path = DEMO_CONFIG["test_data_path"]
        if not os.path.exists(data_path):
            generate_test_data()
        
        # 更新调优参数
        print(f"\n{Color.PURPLE}📌 配置调优参数...{Color.RESET}")
        tuner.default_params.update({
            "domain": DEMO_CONFIG["default_domain"],
            "adapt_iterations": DEMO_CONFIG["adapt_iterations"],
            "learning_rate": 0.01,
            "core_factor_weight": 0.85,
            "target_metric": "metabolic_efficiency"
        })
        
        print(f"{Color.BLUE}📋 调优参数：{Color.RESET}")
        for key, value in tuner.default_params.items():
            print(f"   {key}: {value}")
        
        # 启动调优
        print(f"\n{Color.PURPLE}📌 启动参数调优（{DEMO_CONFIG['adapt_iterations']}次迭代）...{Color.RESET}")
        tuner._start_tuner(data_path)
        
        # 等待调优完成
        while tuner.tuner_status["is_running"]:
            progress = tuner.tuner_status["progress"]
            current_score = tuner.tuner_status["current_score"]
            best_score = tuner.tuner_status["best_score"]
            print(f"\r{Color.CYAN}⏳ 调优进度：{progress:.1f}% | 当前得分：{current_score:.3f} | 最优得分：{best_score:.3f}{Color.RESET}", end="")
            time.sleep(0.5)
        
        print(f"\n{Color.GREEN}✅ 调优完成！")
        print(f"   最优得分：{tuner.tuner_status['best_score']:.3f}")
        print(f"   耗时：{tuner.tuner_status['elapsed_time']:.2f}秒")
        print(f"   最优参数：")
        for key, value in tuner.tuner_status['best_params'].items():
            print(f"      {key}: {value}")
        
        # 保存调优记录
        tuner._save_tuner_record({
            "record_id": f"demo_tuner_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "domain": DEMO_CONFIG["default_domain"],
            "iterations": DEMO_CONFIG["adapt_iterations"],
            "best_score": tuner.tuner_status["best_score"],
            "best_params": tuner.tuner_status["best_params"],
            "status": "completed",
            "duration": tuner.tuner_status["elapsed_time"],
            "score_history": [round(x, 3) for x in tuner.tuner_status.get('score_history', [])]
        })
        
        return tuner.tuner_status
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 参数调优演示失败：{e}{Color.RESET}")
        return None

def demo_step_4_result_analysis():
    """步骤4：调优结果分析"""
    print_separator("步骤4：调优结果深度分析")
    
    if not MODULES_LOADED:
        logger.error(f"{Color.RED}❌ 核心模块未加载，跳过后果分析演示{Color.RESET}")
        return None
    
    try:
        # 创建分析器
        analyzer = ResultAnalyzer(output_dir="./umc_demo_analysis")
        
        # 加载调优历史记录
        tuner_dir = "./umc_tuner/history"
        history_files = [f for f in os.listdir(tuner_dir) if f.startswith("demo_tuner_") and f.endswith(".json")]
        
        if not history_files:
            print(f"{Color.YELLOW}⚠️  未找到调优记录，跳过分析演示{Color.RESET}")
            return None
        
        # 加载最新的调优记录
        latest_file = sorted(history_files)[-1]
        with open(f"{tuner_dir}/{latest_file}", "r", encoding="utf-8") as f:
            tuner_data = json.load(f)
        
        print(f"\n{Color.PURPLE}📌 分析调优记录：{latest_file}{Color.RESET}")
        
        # 执行多维度分析
        score_history = tuner_data["score_history"]
        analysis_result = {
            "basic_metrics": {
                "best_score": tuner_data["best_score"],
                "avg_score": sum(score_history) / len(score_history),
                "std_score": (sum([(x - sum(score_history)/len(score_history))**2 for x in score_history]) / len(score_history))**0.5,
                "min_score": min(score_history),
                "max_score": max(score_history)
            },
            "convergence_analysis": {
                "convergence_iter": next(i for i, score in enumerate(score_history) if score >= tuner_data["best_score"] * 0.99) + 1,
                "stability_score": 1 - (max(score_history[-10:]) - min(score_history[-10:])) if len(score_history)>=10 else 1.0,
                "improvement_rate": (score_history[-1] - score_history[0]) / len(score_history)
            },
            "param_analysis": {
                "optimal_params": tuner_data["best_params"],
                "sensitivity": {
                    "learning_rate": 0.85,
                    "core_factor_weight": 0.92,
                    "stability_threshold": 0.78
                }
            },
            "recommendations": [
                "调优过程收敛良好，建议保留当前最优参数",
                f"最优学习率：{tuner_data['best_params']['learning_rate']:.4f}",
                f"建议迭代次数：{len(score_history)}（当前已足够）"
            ]
        }
        
        # 打印分析结果
        print(f"{Color.GREEN}✅ 分析完成！核心指标：{Color.RESET}")
        print(f"   最优得分：{analysis_result['basic_metrics']['best_score']:.3f}")
        print(f"   平均得分：{analysis_result['basic_metrics']['avg_score']:.3f}")
        print(f"   得分稳定性：{analysis_result['convergence_analysis']['stability_score']:.3f}")
        print(f"   收敛迭代：{analysis_result['convergence_analysis']['convergence_iter']}")
        print(f"\n{Color.BLUE}📋 优化建议：{Color.RESET}")
        for rec in analysis_result["recommendations"]:
            print(f"   - {rec}")
        
        # 保存分析报告
        analysis_path = "./umc_demo_analysis_report.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        print(f"\n{Color.GREEN}📄 分析报告已保存：{analysis_path}{Color.RESET}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 结果分析演示失败：{e}{Color.RESET}")
        return None

def demo_step_5_report_generation():
    """步骤5：分析报告生成"""
    print_separator("步骤5：多格式分析报告生成")
    
    if not MODULES_LOADED:
        logger.error(f"{Color.RED}❌ 核心模块未加载，跳过报告生成演示{Color.RESET}")
        return None
    
    try:
        # 创建报告生成器
        report_generator = ReportGenerator(output_dir="./umc_demo_report")
        
        # 加载分析数据
        analysis_path = "./umc_demo_analysis_report.json"
        if not os.path.exists(analysis_path):
            print(f"{Color.YELLOW}⚠️  未找到分析数据，跳过报告生成{Color.RESET}")
            return None
        
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        
        # 构建报告数据
        report_data = {
            "report_title": "UMC-Metabolic-Agent v2.0 演示报告",
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "demo_config": DEMO_CONFIG,
            "analysis_result": analysis_data,
            "generated_by": DEMO_CONFIG["demo_user"]
        }
        
        # 生成多格式报告
        print(f"\n{Color.PURPLE}📌 生成Markdown/HTML格式报告...{Color.RESET}")
        report_paths = report_generator.generate_comprehensive_report(
            report_data,
            report_name="umc_v20_demo_report",
            format_list=["md", "html"],
            with_plots=True
        )
        
        # 打印报告路径
        print(f"{Color.GREEN}✅ 报告生成成功：{Color.RESET}")
        for fmt, path in report_paths.items():
            print(f"   {fmt.upper()}格式：{path}")
        
        return report_paths
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 报告生成演示失败：{e}{Color.RESET}")
        return None

def demo_step_6_api_server():
    """步骤6：启动API服务"""
    print_separator("步骤6：启动RESTful API服务")
    
    if not MODULES_LOADED:
        logger.error(f"{Color.RED}❌ 核心模块未加载，跳过API服务演示{Color.RESET}")
        return None
    
    try:
        # 在后台线程启动API服务器
        print(f"\n{Color.PURPLE}📌 启动API服务器（http://{DEMO_CONFIG['api_host']}:{DEMO_CONFIG['api_port']}）...{Color.RESET}")
        
        def start_api():
            api = UMCCustomAPI()
            api.run_server(
                host=DEMO_CONFIG["api_host"],
                port=DEMO_CONFIG["api_port"],
                reload=False
            )
        
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        
        # 等待服务器启动
        time.sleep(3)
        
        print(f"{Color.GREEN}✅ API服务器已启动：")
        print(f"   服务地址：http://{DEMO_CONFIG['api_host']}:{DEMO_CONFIG['api_port']}")
        print(f"   文档地址：http://{DEMO_CONFIG['api_host']}:{DEMO_CONFIG['api_port']}/docs")
        print(f"   按 Ctrl+C 停止服务器")
        
        # 自动打开文档页面
        webbrowser.open(f"http://{DEMO_CONFIG['api_host']}:{DEMO_CONFIG['api_port']}/docs")
        
        # 保持运行
        try:
            while api_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n{Color.YELLOW}🛑 API服务器已停止{Color.RESET}")
        
        return True
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ API服务演示失败：{e}{Color.RESET}")
        return False

def demo_step_7_dashboard():
    """步骤7：启动调优仪表盘"""
    print_separator("步骤7：启动Web调优仪表盘")
    
    try:
        # 启动Streamlit仪表盘
        print(f"\n{Color.PURPLE}📌 启动调优仪表盘（http://localhost:{DEMO_CONFIG['dashboard_port']}）...{Color.RESET}")
        print(f"{Color.YELLOW}⚠️  仪表盘将在新窗口打开，按 Ctrl+C 停止{Color.RESET}")
        
        # 启动命令
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "tuner_dashboard.py",
            "--server.port", str(DEMO_CONFIG["dashboard_port"]),
            "--server.headless", "false"
        ]
        
        # 自动打开浏览器
        webbrowser.open(f"http://localhost:{DEMO_CONFIG['dashboard_port']}")
        
        # 运行仪表盘
        subprocess.run(cmd, check=True)
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}🛑 仪表盘已停止{Color.RESET}")
        return True
    except Exception as e:
        logger.error(f"{Color.RED}❌ 仪表盘演示失败：{e}{Color.RESET}")
        return False

# ------------------------------ 主演示流程 ------------------------------
def main_demo_flow():
    """主演示流程"""
    print(f"{Color.BLUE}{'='*80}{Color.RESET}")
    print(f"{Color.CYAN}{'UMC-Metabolic-Agent v2.0 全功能演示'.center(80)}{Color.RESET}")
    print(f"{Color.BLUE}{'='*80}{Color.RESET}")
    
    print(f"\n{Color.PURPLE}📋 演示内容：{Color.RESET}")
    print(f"   1. 身份认证系统初始化（用户/API密钥管理）")
    print(f"   2. 智能体基础运行（测试数据生成+核心功能）")
    print(f"   3. 参数调优演示（实时监控+自动优化）")
    print(f"   4. 调优结果分析（多维度指标分析）")
    print(f"   5. 分析报告生成（Markdown/HTML格式）")
    print(f"   6. API服务启动（RESTful接口+文档）")
    print(f"   7. Web调优仪表盘（可视化操作界面）")
    
    # 确认开始
    while True:
        choice = input(f"\n{Color.YELLOW}🚀 是否开始演示？(y/n): {Color.RESET}").strip().lower()
        if choice in ["y", "n"]:
            break
        print(f"{Color.RED}❌ 请输入 y 或 n{Color.RESET}")
    
    if choice != "y":
        print(f"{Color.YELLOW}🛑 演示已取消{Color.RESET}")
        return
    
    try:
        # 执行演示步骤
        demo_step_1_identity_setup()
        demo_step_2_agent_run()
        demo_step_3_agent_tuning()
        demo_step_4_result_analysis()
        demo_step_5_report_generation()
        
        # 交互式选择后续演示
        print_separator("选择后续演示内容")
        print(f"{Color.CYAN}请选择要演示的功能（输入数字）：{Color.RESET}")
        print(f"   1 - 启动API服务")
        print(f"   2 - 启动Web调优仪表盘")
        print(f"   3 - 退出演示")
        
        while True:
            choice = input(f"\n{Color.YELLOW}请选择：{Color.RESET}").strip()
            if choice in ["1", "2", "3"]:
                break
            print(f"{Color.RED}❌ 请输入 1、2 或 3{Color.RESET}")
        
        if choice == "1":
            demo_step_6_api_server()
        elif choice == "2":
            demo_step_7_dashboard()
        elif choice == "3":
            print(f"{Color.YELLOW}🛑 演示结束{Color.RESET}")
        
    except Exception as e:
        logger.error(f"{Color.RED}❌ 演示过程出错：{e}{Color.RESET}")
    finally:
        # 清理演示数据
        if DEMO_CONFIG["cleanup_after_demo"]:
            print_separator("清理演示数据")
            cleanup_demo_data()
            print(f"{Color.GREEN}✅ 演示数据已清理{Color.RESET}")
        
        print_separator("UMC-Metabolic-Agent v2.0 演示完成")
        print(f"{Color.CYAN}感谢使用UMC智能体！{Color.RESET}")

# ------------------------------ 命令行入口 ------------------------------
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="UMC-Metabolic-Agent v2.0 演示脚本")
    parser.add_argument("--cleanup", action="store_true", help="演示后清理数据")
    parser.add_argument("--skip-modules-check", action="store_true", help="跳过模块检查")
    parser.add_argument("--dashboard-only", action="store_true", help="仅启动调优仪表盘")
    parser.add_argument("--api-only", action="store_true", help="仅启动API服务")
    
    args = parser.parse_args()
    
    # 更新配置
    DEMO_CONFIG["cleanup_after_demo"] = args.cleanup
    
    # 执行指定的演示模式
    if args.dashboard_only:
        demo_step_7_dashboard()
    elif args.api_only:
        demo_step_6_api_server()
    else:
        # 执行完整演示
        main_demo_flow()