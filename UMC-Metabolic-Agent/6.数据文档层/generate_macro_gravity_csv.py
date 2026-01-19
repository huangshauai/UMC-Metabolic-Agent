# generate_macro_gravity_csv.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===================== 核心配置（固定参数，保证数据可复现） =====================
np.random.seed(42)  # 固定随机种子，多次生成结果一致
TOTAL_SAMPLES = 1000  # 总样本数：1000条
START_TIME = datetime(2026, 1, 1, 0, 0, 0)  # 数据起始时间
CELESTIAL_BODIES = ["Earth", "Moon", "Mars", "Jupiter"]  # 天体类型
DISTANCE_MIN, DISTANCE_MAX = 500, 2000  # 观测距离范围（km）

# ===================== 天体特性配置（符合物理规律） =====================
CELESTIAL_PROPERTIES = {
    "Earth": {
        "gravity_base": 9.81,    # 基准引力场强度（m/s²）
        "accuracy_base": 0.92,   # 基准测地线精度
        "curvature_base": 5.2e-9,# 基准时空曲率（1/m²）
        "mass": 5.97             # 质量（×10²⁴kg）
    },
    "Moon": {
        "gravity_base": 1.62,
        "accuracy_base": 0.88,
        "curvature_base": 0.8e-9,
        "mass": 0.73
    },
    "Mars": {
        "gravity_base": 3.72,
        "accuracy_base": 0.90,
        "curvature_base": 2.1e-9,
        "mass": 6.42
    },
    "Jupiter": {
        "gravity_base": 24.79,
        "accuracy_base": 0.94,
        "curvature_base": 18.5e-9,
        "mass": 1898.0
    }
}

# ===================== 生成完整数据集 =====================
def generate_macro_gravity_data():
    """生成宏观引力领域完整测试数据，写入macro_gravity.csv"""
    data_rows = []
    
    for sample_idx in range(TOTAL_SAMPLES):
        # 1. 基础时序与标识字段
        current_time = START_TIME + timedelta(minutes=sample_idx)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        celestial_body = np.random.choice(CELESTIAL_BODIES)  # 随机选择天体
        distance = np.random.randint(DISTANCE_MIN, DISTANCE_MAX + 1)  # 随机观测距离
        sample_id = f"G_{celestial_body}_{str(sample_idx + 1).zfill(4)}"  # 唯一样本ID
        
        # 2. 核心宏观引力指标（基于天体特性+随机波动）
        body_prop = CELESTIAL_PROPERTIES[celestial_body]
        
        # 引力场强度（小幅随机波动）
        gravitational_field = round(np.clip(
            np.random.normal(body_prop["gravity_base"], 0.02), 
            body_prop["gravity_base"] * 0.98, body_prop["gravity_base"] * 1.02
        ), 2)
        
        # 测地线精度（核心优化目标：值越大越好）
        # 距离越远，精度略有下降（符合观测规律）
        distance_factor = 1 - (distance - DISTANCE_MIN) / (DISTANCE_MAX - DISTANCE_MIN) * 0.02
        geodesic_accuracy = round(np.clip(
            np.random.normal(body_prop["accuracy_base"] * distance_factor, 0.005), 0.85, 0.95
        ), 3)
        
        # 时空曲率（与引力场强度正相关）
        spacetime_curvature = round(np.clip(
            np.random.normal(body_prop["curvature_base"] * (gravitational_field / body_prop["gravity_base"]), body_prop["curvature_base"] * 0.1),
            1e-9, 2e-8
        ), 9)
        
        # 天体质量（固定值，无波动）
        celestial_mass = body_prop["mass"]
        
        # 3. 智能体适配字段（与核心指标关联，保证调优有效性）
        domain_adapt_score = round(np.clip(geodesic_accuracy * 0.97, 0.80, 0.95), 3)
        core_factor = round(np.clip(np.random.normal(0.87, 0.03), 0.80, 0.95), 3)
        # 稳定性：与测地线精度正相关，与距离负相关
        stability = round(np.clip(
            geodesic_accuracy * 0.92 - (distance / DISTANCE_MAX) * 0.05, 0.80, 0.88
        ), 3)
        
        # 4. 组装数据行
        data_rows.append([
            timestamp, sample_id, celestial_body, gravitational_field,
            geodesic_accuracy, spacetime_curvature, celestial_mass, distance,
            domain_adapt_score, core_factor, stability
        ])
    
    # ===================== 写入CSV文件 =====================
    # 定义字段名（与UMC智能体脚本完全兼容）
    columns = [
        "timestamp", "sample_id", "celestial_body", "gravitational_field",
        "geodesic_accuracy", "spacetime_curvature", "celestial_mass", "distance",
        "domain_adapt_score", "core_factor", "stability"
    ]
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv("macro_gravity.csv", index=False, encoding="utf-8")
    
    # 打印生成结果
    print(f"✅ 已生成完整的宏观引力测试数据文件：macro_gravity.csv")
    print(f"📊 数据规模：{len(df)}条样本 | 字段数：{len(df.columns)}个")
    print(f"📈 数据预览（前5行）：")
    print(df.head())
    
    # 验证核心指标分布
    print(f"\n🎯 核心指标统计（测地线精度）：")
    print(f"   平均值：{df['geodesic_accuracy'].mean():.3f}")
    print(f"   最小值：{df['geodesic_accuracy'].min():.3f} | 最大值：{df['geodesic_accuracy'].max():.3f}")
    print(f"   按天体分组统计：")
    for body in CELESTIAL_BODIES:
        subset = df[df['celestial_body'] == body]['geodesic_accuracy']
        print(f"      {body}：平均值={subset.mean():.3f}")

if __name__ == "__main__":
    generate_macro_gravity_data()