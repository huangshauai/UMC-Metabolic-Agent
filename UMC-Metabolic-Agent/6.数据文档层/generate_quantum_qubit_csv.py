# generate_quantum_qubit_csv.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===================== 核心配置（固定参数，保证数据可复现） =====================
np.random.seed(42)  # 固定随机种子，多次生成结果一致
TOTAL_SAMPLES = 1000  # 总样本数：1000条
START_TIME = datetime(2026, 1, 1, 0, 0, 0)  # 数据起始时间
QUBIT_OPTIONS = [2, 4, 8, 16]  # 量子比特数可选值（覆盖典型场景）
GATE_MIN, GATE_MAX = 10, 100  # 量子门操作次数范围

# ===================== 生成完整数据集 =====================
def generate_quantum_qubit_data():
    """生成量子比特领域完整测试数据，写入quantum_qubit.csv"""
    data_rows = []
    
    for sample_idx in range(TOTAL_SAMPLES):
        # 1. 基础时序与标识字段
        current_time = START_TIME + timedelta(minutes=sample_idx)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        qubit_count = np.random.choice(QUBIT_OPTIONS)  # 随机选择比特数
        gate_operations = np.random.randint(GATE_MIN, GATE_MAX + 1)  # 随机门操作次数
        sample_id = f"Q_{qubit_count}_{str(sample_idx + 1).zfill(4)}"  # 唯一样本ID
        
        # 2. 核心量子指标（符合物理规律：比特数越多，保真度越低、错误率越高）
        # 量子态保真度（核心优化目标）：2比特≈0.99，16比特≈0.79，带小幅随机波动
        fidelity_base = 0.99 - (qubit_count / 16) * 0.2
        quantum_fidelity = round(np.clip(np.random.normal(fidelity_base, 0.02), 0.70, 0.99), 3)
        
        # 量子相干时间（μs）：比特数越多，相干时间越短
        coherence_base = 100 - (qubit_count / 16) * 80
        coherence_time = round(np.clip(np.random.normal(coherence_base, 5), 10.0, 100.0), 1)
        
        # 量子门错误率：比特数越多，错误率越高
        error_base = 0.001 + (qubit_count / 16) * 0.049
        error_rate = round(np.clip(np.random.normal(error_base, 0.003), 0.001, 0.05), 3)
        
        # 量子测量准确率：与错误率负相关
        measurement_acc = round(np.clip(np.random.normal(0.95 - (error_rate * 0.5), 0.01), 0.85, 0.99), 3)
        
        # 3. 智能体适配字段（与核心指标关联，保证调优有效性）
        domain_adapt_score = round(np.clip(quantum_fidelity * 0.95, 0.60, 0.95), 3)
        core_factor = round(np.clip(np.random.normal(0.85, 0.05), 0.70, 0.95), 3)
        stability = round(np.clip(quantum_fidelity * 0.9, 0.65, 0.90), 3)
        
        # 4. 组装数据行
        data_rows.append([
            timestamp, sample_id, qubit_count, gate_operations, quantum_fidelity,
            coherence_time, error_rate, measurement_acc, domain_adapt_score,
            core_factor, stability
        ])
    
    # ===================== 写入CSV文件 =====================
    # 定义字段名（与UMC智能体脚本完全兼容）
    columns = [
        "timestamp", "sample_id", "qubit_count", "gate_operations", "quantum_fidelity",
        "coherence_time", "error_rate", "measurement_acc", "domain_adapt_score",
        "core_factor", "stability"
    ]
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv("quantum_qubit.csv", index=False, encoding="utf-8")
    
    # 打印生成结果
    print(f"✅ 已生成完整的量子比特测试数据文件：quantum_qubit.csv")
    print(f"📊 数据规模：{len(df)}条样本 | 字段数：{len(columns)}个")
    print(f"📈 数据预览（前5行）：")
    print(df.head())
    
    # 验证核心指标分布
    print(f"\n🎯 核心指标统计（量子保真度）：")
    print(f"   平均值：{df['quantum_fidelity'].mean():.3f}")
    print(f"   最小值：{df['quantum_fidelity'].min():.3f} | 最大值：{df['quantum_fidelity'].max():.3f}")
    print(f"   按比特数分组统计：")
    for qubit in QUBIT_OPTIONS:
        subset = df[df['qubit_count'] == qubit]['quantum_fidelity']
        print(f"      {qubit}比特：平均值={subset.mean():.3f}")

if __name__ == "__main__":
    generate_quantum_qubit_data()