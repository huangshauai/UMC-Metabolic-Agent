# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 调优器核心操作示例脚本
核心逻辑：分步演示调优器（TunerDashboard）的完整使用流程，适合新手入门学习
设计原则：步骤拆解、注释详尽、输出清晰、可直接运行、聚焦核心操作
"""
import os
import sys
import json
import time
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 配置基础日志（仅输出关键信息）
logging.basicConfig(
    level=logging.INFO,
    format="[\033[34m%(asctime)s\033[0m] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Tuner-Example")
warnings.filterwarnings("ignore")

# ------------------------------ 环境准备与模块导入 ------------------------------
# 添加当前目录到Python路径（确保能导入调优器模块）
sys.path.insert(0, os.getcwd())

# 导入调优器核心模块（带容错提示）
try:
    from tuner_dashboard import TunerDashboard
    logger.info("\033[32m✅ 调优器模块导入成功\033[0m")
except ImportError as e:
    logger.error(f"\033[31m❌ 调优器模块导入失败：{e}\033[0m")
    logger.error("⚠️  请确保 tuner_dashboard.py 文件在当前目录")
    sys.exit(1)

# ------------------------------ 基础配置（新手友好版） ------------------------------
# 示例配置：所有参数都有明确注释，新手可直接修改
EXAMPLE_CONFIG = {
    "test_data_path": "./example_tuner_test_data.csv",  # 测试数据路径
    "output_dir": "./example_tuner_output",             # 调优结果输出目录
    "domain": "quantum",                                # 目标优化领域
    "adapt_iterations": 50,                             # 调优迭代次数（新手建议20-100）
    "learning_rate": 0.01,                              # 学习率（新手建议0.005-0.02）
    "core_factor_weight": 0.85,                         # 核心因子权重
    "target_metric": "metabolic_efficiency",            # 优化目标指标
    "plot_tuning_process": True,                        # 是否可视化调优过程
    "save_tuning_record": True                          # 是否保存调优记录
}

# ------------------------------ 新手友好工具函数 ------------------------------
def print_step(step_num: int, step_desc: str):
    """打印步骤提示（新手友好）"""
    print(f"\n{'='*70}")
    print(f"\033[36m步骤 {step_num}：{step_desc}\033[0m")
    print(f"{'='*70}")

def generate_simple_test_data(data_path: str, rows: int = 1000):
    """生成新手友好的简化测试数据（带详细注释）"""
    if os.path.exists(data_path):
        logger.info(f"📄 使用已有测试数据：{data_path}")
        return data_path
    
    logger.info(f"📊 生成测试数据（{rows}行），模拟代谢效率数据...")
    
    # 设置随机种子，保证结果可复现（新手调试必备）
    np.random.seed(42)
    
    # 构建模拟数据：仅保留核心字段，降低新手理解成本
    data = {
        # 时间戳：按分钟递增
        "timestamp": pd.date_range(start="2026-01-01", periods=rows, freq="1min"),
        # 代谢效率：核心优化指标，范围0.6-0.95
        "metabolic_efficiency": np.random.uniform(0.6, 0.95, size=rows),
        # 领域适配得分：辅助指标，范围0.5-0.9
        "domain_adapt_score": np.random.uniform(0.5, 0.9, size=rows),
        # 核心因子：辅助指标，范围0.7-0.9
        "core_factor": np.random.uniform(0.7, 0.9, size=rows),
        # 稳定性：辅助指标，范围0.65-0.85
        "stability": np.random.uniform(0.65, 0.85, size=rows),
        # 样本ID：便于数据追踪
        "sample_id": [f"S{str(i).zfill(4)}" for i in range(rows)]
    }
    
    # 保存为CSV文件（新手易读取格式）
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False, encoding="utf-8")
    
    # 打印数据预览（新手直观了解数据结构）
    logger.info(f"📈 测试数据预览（前5行）：")
    print(df.head())
    
    logger.info(f"\033[32m✅ 测试数据生成完成：{data_path}\033[0m")
    return data_path

def ensure_output_dir(dir_path: str):
    """确保输出目录存在（新手友好：自动创建目录）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"📁 创建输出目录：{dir_path}")

# ------------------------------ 调优器核心操作分步演示 ------------------------------
def main_tuner_example():
    """调优器核心操作分步演示（新手入门）"""
    logger.info("\033[35m🚀 开始UMC调优器核心操作示例演示\033[0m")
    
    # ===================== 步骤1：环境准备 =====================
    print_step(1, "环境准备：生成测试数据 + 创建输出目录")
    # 生成测试数据
    data_path = generate_simple_test_data(EXAMPLE_CONFIG["test_data_path"])
    # 创建输出目录
    ensure_output_dir(EXAMPLE_CONFIG["output_dir"])
    
    # ===================== 步骤2：初始化调优器 =====================
    print_step(2, "初始化调优器实例（核心对象）")
    # 创建调优器实例：调优器的核心入口
    tuner = TunerDashboard()
    
    # 打印调优器默认配置（新手了解可配置参数）
    logger.info(f"🔧 调优器默认配置：")
    for key, value in tuner.default_params.items():
        print(f"   {key}: {value}")
    
    # ===================== 步骤3：配置调优参数 =====================
    print_step(3, "配置调优参数（自定义优化目标）")
    # 更新调优参数：覆盖默认配置，适配当前示例
    tuner.default_params.update({
        "domain": EXAMPLE_CONFIG["domain"],               # 优化领域
        "adapt_iterations": EXAMPLE_CONFIG["adapt_iterations"],  # 迭代次数
        "learning_rate": EXAMPLE_CONFIG["learning_rate"], # 学习率
        "core_factor_weight": EXAMPLE_CONFIG["core_factor_weight"], # 核心因子权重
        "target_metric": EXAMPLE_CONFIG["target_metric"]  # 优化目标指标
    })
    
    # 打印更新后的配置（新手确认参数是否生效）
    logger.info(f"🔧 更新后的调优配置（仅展示修改项）：")
    modified_params = {k: v for k, v in tuner.default_params.items() if k in EXAMPLE_CONFIG}
    for key, value in modified_params.items():
        print(f"   {key}: {value}")
    
    # ===================== 步骤4：启动调优并监控进度 =====================
    print_step(4, "启动调优 + 实时监控进度（核心操作）")
    logger.info(f"🚀 启动调优（领域：{EXAMPLE_CONFIG['domain']}，迭代次数：{EXAMPLE_CONFIG['adapt_iterations']}）")
    
    # 记录调优开始时间
    start_time = time.time()
    
    # 启动调优：调优器核心方法
    tuner._start_tuner(data_path)
    
    # 实时监控调优进度（新手直观了解调优过程）
    logger.info(f"📊 调优进度监控（按Ctrl+C可中断，但建议等待完成）：")
    while tuner.tuner_status["is_running"]:
        # 获取当前调优状态
        progress = tuner.tuner_status["progress"]       # 进度百分比
        current_score = tuner.tuner_status["current_score"]  # 当前得分
        best_score = tuner.tuner_status["best_score"]    # 最优得分
        current_iter = tuner.tuner_status["current_iter"]    # 当前迭代次数
        
        # 实时打印进度（覆盖当前行，更整洁）
        print(f"\r⏳ 进度：{progress:.1f}% | 迭代：{current_iter}/{EXAMPLE_CONFIG['adapt_iterations']} | 当前得分：{current_score:.3f} | 最优得分：{best_score:.3f}", end="")
        
        # 每0.5秒更新一次，降低资源占用
        time.sleep(0.5)
    
    # 计算调优耗时
    elapsed_time = time.time() - start_time
    
    # 换行，结束进度监控
    print()
    logger.info(f"\033[32m✅ 调优完成！总耗时：{elapsed_time:.2f}秒\033[0m")
    
    # ===================== 步骤5：解析调优结果 =====================
    print_step(5, "解析调优结果（核心输出）")
    # 获取调优状态字典：包含所有核心结果
    tune_status = tuner.tuner_status
    
    # 打印核心结果（新手重点关注）
    logger.info(f"🏆 调优核心结果：")
    print(f"   1. 最优得分：{tune_status['best_score']:.3f}（越高越好）")
    print(f"   2. 收敛迭代次数：{tune_status['convergence_iter']}（越早收敛越好）")
    print(f"   3. 调优稳定性：{tune_status['stability_score']:.3f}（越接近1越稳定）")
    print(f"   4. 调优耗时：{tune_status['elapsed_time']:.2f}秒")
    
    # 打印最优参数（新手了解哪些参数最优）
    logger.info(f"🔧 调优最优参数（建议保存）：")
    best_params = tune_status["best_params"]
    for key, value in best_params.items():
        print(f"   {key}: {value:.4f}")
    
    # ===================== 步骤6：保存调优记录 =====================
    if EXAMPLE_CONFIG["save_tuning_record"]:
        print_step(6, "保存调优记录（便于后续分析）")
        # 构建调优记录（结构化存储）
        tune_record = {
            "record_id": f"example_tuner_{time.strftime('%Y%m%d%H%M%S')}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": EXAMPLE_CONFIG,                  # 调优配置
            "status": tune_status,                     # 调优状态
            "elapsed_time": elapsed_time,              # 总耗时
            "score_history": [round(x, 3) for x in tune_status.get('score_history', [])]  # 得分历史
        }
        
        # 保存为JSON文件（新手易读取）
        record_path = f"{EXAMPLE_CONFIG['output_dir']}/tune_record_{tune_record['record_id']}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(tune_record, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\033[32m✅ 调优记录已保存：{record_path}\033[0m")
    
    # ===================== 步骤7：可视化调优过程（可选） =====================
    if EXAMPLE_CONFIG["plot_tuning_process"]:
        print_step(7, "可视化调优过程（直观分析）")
        try:
            # 获取得分历史
            score_history = tuner.tuner_status.get('score_history', [])
            if not score_history:
                logger.warning("⚠️  无调优得分历史，跳过可视化")
                return
            
            # 创建画布（新手友好的尺寸）
            plt.figure(figsize=(12, 6))
            
            # 绘制得分变化曲线
            plt.plot(score_history, label="调优得分", color="#2E86AB", linewidth=2)
            # 标记最优得分点
            best_idx = np.argmax(score_history)
            plt.scatter(best_idx, score_history[best_idx], color="#E63946", s=100, label=f"最优得分 ({score_history[best_idx]:.3f})")
            # 标记收敛点
            conv_iter = tuner.tuner_status.get('convergence_iter', 0)
            if conv_iter < len(score_history):
                plt.axvline(x=conv_iter, color="#F1FAEE", linestyle="--", label=f"收敛迭代 ({conv_iter})")
            
            # 设置图表属性（新手易读）
            plt.title(f"UMC调优器得分变化（领域：{EXAMPLE_CONFIG['domain']}）", fontsize=14)
            plt.xlabel("迭代次数", fontsize=12)
            plt.ylabel("调优得分", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            # 保存图表
            plot_path = f"{EXAMPLE_CONFIG['output_dir']}/tuning_process_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"\033[32m✅ 调优过程可视化完成：{plot_path}\033[0m")
        except Exception as e:
            logger.error(f"\033[31m❌ 可视化失败：{e}\033[0m")
            logger.warning("⚠️  请确保安装了matplotlib：pip install matplotlib")
    
    # ===================== 演示完成 =====================
    print_step(8, "调优器操作演示完成")
    logger.info("\033[35m🎉 调优器核心操作示例演示完成！\033[0m")
    logger.info("📋 新手后续学习建议：")
    logger.info("   1. 修改EXAMPLE_CONFIG中的参数（如迭代次数、学习率），观察调优效果变化")
    logger.info("   2. 查看保存的调优记录文件，分析得分历史和最优参数")
    logger.info("   3. 尝试更换target_metric（如domain_adapt_score、stability），优化不同指标")
    logger.info("   4. 增加测试数据行数，观察调优耗时和效果的关系")

# ------------------------------ 新手友好的命令行入口 ------------------------------
if __name__ == "__main__":
    # 打印新手提示
    print("""
\033[36m=========================================
UMC调优器操作示例 - 新手入门指南
=========================================\033[0m
📖 本脚本将分步演示调优器的核心操作，包含：
   1. 环境准备（生成测试数据）
   2. 调优器初始化
   3. 调优参数配置
   4. 调优执行与进度监控
   5. 调优结果解析
   6. 调优记录保存
   7. 调优过程可视化

💡 新手提示：
   - 所有参数都在EXAMPLE_CONFIG中，可直接修改
   - 运行过程中会打印详细的步骤说明
   - 运行完成后会在example_tuner_output目录生成结果文件
   - 建议先按默认配置运行，再尝试修改参数

\033[32m按回车键开始演示...\033[0m
""")
    # 等待用户确认（新手友好）
    input()
    
    # 执行核心演示
    main_tuner_example()