# generate_atomic_spectra_csv.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===================== 核心配置（固定参数，保证数据可复现） =====================
np.random.seed(42)  # 固定随机种子，多次生成结果一致
TOTAL_SAMPLES = 1000  # 总样本数：1000条
START_TIME = datetime(2026, 1, 1, 0, 0, 0)  # 数据起始时间
ELEMENT_OPTIONS = ["H", "He", "Li", "Na"]  # 被测元素类型
TEMP_COEFF_MIN, TEMP_COEFF_MAX = 0.1, 0.8  # 温度系数范围

# ===================== 元素特性配置（符合物理规律） =====================
ELEMENT_PROPERTIES = {
    "H": {"res_base": 0.002, "wave_base": 0.0002, "snr_base": 400, "intensity_base": 0.85},  # 氢：分辨率高、信噪比高
    "He": {"res_base": 0.003, "wave_base": 0.0003, "snr_base": 350, "intensity_base": 0.80}, # 氦：中等特性
    "Li": {"res_base": 0.005, "wave_base": 0.0005, "snr_base": 250, "intensity_base": 0.75}, # 锂：分辨率一般
    "Na": {"res_base": 0.008, "wave_base": 0.0008, "snr_base": 200, "intensity_base": 0.90}  # 钠：谱线强度高、分辨率低
}

# ===================== 生成完整数据集 =====================
def generate_atomic_spectra_data():
    """生成原子光谱领域完整测试数据，写入atomic_spectra.csv"""
    data_rows = []
    
    for sample_idx in range(TOTAL_SAMPLES):
        # 1. 基础时序与标识字段
        current_time = START_TIME + timedelta(minutes=sample_idx)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        element_type = np.random.choice(ELEMENT_OPTIONS)  # 随机选择被测元素
        sample_id = f"A_{element_type}_{str(sample_idx + 1).zfill(4)}"  # 唯一样本ID
        
        # 2. 核心原子光谱指标（基于元素特性+随机波动）
        elem_prop = ELEMENT_PROPERTIES[element_type]
        
        # 光谱分辨率（核心优化目标：值越小越好，对应原metabolic_efficiency值越大越好）
        # 处理逻辑：取倒数后归一化，保证智能体调优逻辑兼容
        spectral_resolution = round(np.clip(
            np.random.normal(elem_prop["res_base"], 0.001), 0.001, 0.010
        ), 4)
        # 适配智能体的"效率"逻辑：分辨率越小→适配得分越高
        res_efficiency = 1 / spectral_resolution / 1000  # 归一化到0-1范围
        
        # 波长精度（值越小越好）
        wavelength_accuracy = round(np.clip(
            np.random.normal(elem_prop["wave_base"], 0.0001), 0.0001, 0.0010
        ), 5)
        
        # 信噪比（值越大越好）
        snr = round(np.clip(
            np.random.normal(elem_prop["snr_base"], 30), 100, 500
        ), 1)
        
        # 谱线强度（值越大越好）
        spectral_intensity = round(np.clip(
            np.random.normal(elem_prop["intensity_base"], 0.05), 0.60, 0.95
        ), 3)
        
        # 温度系数（值越小越好，环境稳定性越高）
        temperature_coeff = round(np.clip(
            np.random.uniform(TEMP_COEFF_MIN, TEMP_COEFF_MAX), 0.1, 0.8
        ), 2)
        
        # 3. 智能体适配字段（与核心指标关联，保证调优有效性）
        domain_adapt_score = round(np.clip(res_efficiency * 0.95, 0.60, 0.95), 3)
        core_factor = round(np.clip(np.random.normal(0.85, 0.05), 0.70, 0.95), 3)
        stability = round(np.clip(
            (1 - temperature_coeff/10) * 0.9, 0.65, 0.90
        ), 3)
        
        # 4. 组装数据行
        data_rows.append([
            timestamp, sample_id, element_type, spectral_resolution,
            wavelength_accuracy, snr, spectral_intensity, temperature_coeff,
            domain_adapt_score, core_factor, stability
        ])
    
    # ===================== 写入CSV文件 =====================
    # 定义字段名（与UMC智能体脚本完全兼容）
    columns = [
        "timestamp", "sample_id", "element_type", "spectral_resolution",
        "wavelength_accuracy", "snr", "spectral_intensity", "temperature_coeff",
        "domain_adapt_score", "core_factor", "stability"
    ]
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv("atomic_spectra.csv", index=False, encoding="utf-8")
    
    # 打印生成结果
    print(f"✅ 已生成完整的原子光谱测试数据文件：atomic_spectra.csv")
    print(f"📊 数据规模：{len(df)}条样本 | 字段数：{len(df.columns)}个")
    print(f"📈 数据预览（前5行）：")
    print(df.head())
    
    # 验证核心指标分布
    print(f"\n🎯 核心指标统计（光谱分辨率）：")
    print(f"   平均值：{df['spectral_resolution'].mean():.4f} nm")
    print(f"   最小值：{df['spectral_resolution'].min():.4f} | 最大值：{df['spectral_resolution'].max():.4f}")
    print(f"   按元素分组统计：")
    for elem in ELEMENT_OPTIONS:
        subset = df[df['element_type'] == elem]['spectral_resolution']
        print(f"      {elem}：平均值={subset.mean():.4f} nm")

if __name__ == "__main__":
    generate_atomic_spectra_data()