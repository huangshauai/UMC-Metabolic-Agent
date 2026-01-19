# UMC-Metabolic-Agent 跨领域适配指南
# cross_domain_adapt.md
## 文档概述
本文档详细介绍 **UMC-Metabolic-Agent** 的跨领域适配能力，包括核心原理、实现步骤、配置示例、性能评估方法及常见问题解决方案。通过本指南，用户可快速实现智能体在**量子计算、原子光谱、宏观引力**等多领域间的无缝切换，无需修改核心代码，仅需配置领域特征映射规则即可完成适配。

## 核心概念
### 1.1 什么是跨领域适配
跨领域适配是指 UMC 智能体通过**通用适配层**和**领域特征映射规则**，将不同领域的核心指标（如量子保真度、光谱分辨率、测地线精度）转换为智能体可识别的通用指标，从而实现同一套核心逻辑在多领域的复用。

**通俗类比**：跨领域适配就像智能体的「领域翻译器」—— 把不同领域的「专业术语」（核心指标）翻译成智能体能理解的「通用语言」（等效效率值），同时保留领域特有规律。

### 1.2 关键术语
| 术语 | 定义 | 作用 |
|------|------|------|
| **通用适配层** | 智能体的核心中间层，负责处理所有领域的通用逻辑（运行、调优、分析） | 隔离领域差异，保证核心代码复用 |
| **领域特征映射** | 将领域特有核心指标转换为通用指标（`metabolic_efficiency` 等效值）的规则 | 让智能体识别不同领域的优化目标 |
| **领域适配得分** | （`domain_adapt_score`）衡量智能体在当前领域的适配程度，取值范围 0.6-0.95 | 评估适配效果，指导调优参数调整 |
| **通用字段集** | 所有领域数据集必须包含的 5 个通用字段（`timestamp`/`sample_id`/`domain_adapt_score`/`core_factor`/`stability`） | 保证数据集与智能体的兼容性 |

### 1.3 跨领域适配的核心优势
1. **零代码修改**：无需修改智能体核心模块（如 `universal_cmd.py`/`tuner_dashboard.py`），仅需配置映射规则；
2. **高复用性**：同一套运行、调优、分析逻辑适配所有领域；
3. **可扩展性**：新增领域仅需添加数据集和映射规则，无需重构核心架构；
4. **结果一致性**：不同领域的调优结果可通过通用指标对比，便于跨领域分析。

## 适配核心原理
UMC 智能体的跨领域适配基于 **「通用适配层 + 领域特征映射」** 双层架构，具体原理如下：

### 2.1 架构设计
```
[领域数据集] → [领域特征映射规则] → [通用适配层] → [智能体核心逻辑] → [领域化结果输出]
```
1. **领域数据集**：符合通用字段规范的领域专用数据（如 `quantum_qubit.csv`/`atomic_spectra.csv`）；
2. **领域特征映射规则**：定义领域核心指标到通用指标的转换公式；
3. **通用适配层**：执行映射计算，输出通用指标（等效效率值），供核心逻辑使用；
4. **智能体核心逻辑**：运行、调优、分析等通用功能，不感知领域差异；
5. **领域化结果输出**：将通用结果转换为领域特有指标（如等效效率值→量子保真度），便于用户理解。

### 2.2 核心映射规则
跨领域适配的关键是 **「领域核心指标 → 通用等效效率值」** 的映射，遵循以下通用公式：
```
通用等效效率值 = 归一化(领域核心指标) × 领域权重系数 × 稳定性系数
```
- **归一化**：将领域核心指标的原始范围映射到 `[0.6, 0.95]`（与通用适配得分范围一致）；
  - 对于**越大越好**的指标（如量子保真度、测地线精度）：
    $$\text{归一化值} = 0.6 + (0.95-0.6) \times \frac{X - X_{min}}{X_{max} - X_{min}}$$
  - 对于**越小越好**的指标（如光谱分辨率、量子错误率）：
    $$\text{归一化值} = 0.6 + (0.95-0.6) \times \frac{X_{max} - X}{X_{max} - X_{min}}$$
- **领域权重系数**：根据领域重要性设置，默认 `1.0`；
- **稳定性系数**：由数据集 `stability` 字段提供，反映领域数据的稳定性。

### 2.3 领域适配流程
以量子领域为例，适配流程如下：
1. **输入**：量子数据集的 `quantum_fidelity`（范围 0.7-0.99）；
2. **归一化计算**：将 `quantum_fidelity` 映射到 `[0.6, 0.95]`；
3. **通用等效效率值计算**：`等效值 = 归一化值 × 1.0 × stability`；
4. **核心逻辑处理**：智能体基于等效值进行调优；
5. **结果输出**：将调优后的等效值反向转换为 `quantum_fidelity`，并输出领域化报告。

## 快速实现步骤
以下是实现 UMC 智能体跨领域适配的 **5 步标准流程**，新手可直接按步骤操作。

### 3.1 步骤 1：准备符合规范的领域数据集
所有领域数据集必须满足 **「通用字段 + 领域特有字段」** 规范：
1. **必选通用字段**（5 个）：`timestamp`/`sample_id`/`domain_adapt_score`/`core_factor`/`stability`；
2. **必选领域特有字段**：1-2 个领域核心指标（如量子领域的 `quantum_fidelity`、原子光谱领域的 `spectral_resolution`）；
3. **数据规范**：
   - 格式为 CSV，编码 UTF-8；
   - 无缺失值、异常值，数值符合领域物理规律；
   - 样本数 ≥ 500 条（保证调优效果）。

**示例**：量子领域数据集的字段结构
| 通用字段 | 领域特有字段 |
|----------|--------------|
| timestamp | quantum_fidelity |
| sample_id | error_rate |
| domain_adapt_score | qubit_count |
| core_factor | gate_operations |
| stability | - |

### 3.2 步骤 2：配置领域特征映射规则
创建领域映射规则配置文件（JSON 格式），命名为 `domain_mapping.json`，放在项目根目录。

**通用配置模板**：
```json
{
    "domain_list": [
        {
            "domain_name": "quantum",
            "core_metric": "quantum_fidelity",
            "metric_type": "higher_better",
            "metric_range": [0.7, 0.99],
            "weight": 1.0,
            "description": "量子领域核心指标：量子态保真度"
        },
        {
            "domain_name": "atomic_spectra",
            "core_metric": "spectral_resolution",
            "metric_type": "lower_better",
            "metric_range": [0.001, 0.010],
            "weight": 1.0,
            "description": "原子光谱领域核心指标：光谱分辨率（nm）"
        },
        {
            "domain_name": "macro_gravity",
            "core_metric": "geodesic_accuracy",
            "metric_type": "higher_better",
            "metric_range": [0.85, 0.95],
            "weight": 1.0,
            "description": "宏观引力领域核心指标：测地线精度"
        }
    ]
}
```
配置参数说明：
| 参数 | 取值 | 说明 |
|------|------|------|
| `domain_name` | 字符串 | 领域名称，需与脚本 `--domain` 参数一致 |
| `core_metric` | 字符串 | 领域核心指标字段名 |
| `metric_type` | `higher_better`/`lower_better` | 指标优化方向 |
| `metric_range` | [最小值, 最大值] | 指标的合理取值范围 |
| `weight` | 0.8-1.2 | 领域权重系数 |
| `description` | 字符串 | 领域指标说明 |

### 3.3 步骤 3：加载映射规则到智能体
修改智能体通用模块（如 `universal_cmd.py`），添加映射规则加载逻辑，示例代码如下：
```python
import json
import os

def load_domain_mapping():
    """加载领域特征映射规则"""
    mapping_path = "./domain_mapping.json"
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("领域映射规则文件 domain_mapping.json 不存在")
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    
    # 构建领域映射字典
    domain_dict = {}
    for domain in mapping_data["domain_list"]:
        domain_dict[domain["domain_name"]] = {
            "core_metric": domain["core_metric"],
            "metric_type": domain["metric_type"],
            "metric_range": domain["metric_range"],
            "weight": domain["weight"]
        }
    return domain_dict

def calculate_universal_efficiency(domain_name, metric_value, stability, domain_dict):
    """计算通用等效效率值"""
    domain_config = domain_dict.get(domain_name)
    if not domain_config:
        raise ValueError(f"未知领域：{domain_name}")
    
    min_val, max_val = domain_config["metric_range"]
    metric_type = domain_config["metric_type"]
    weight = domain_config["weight"]
    
    # 归一化计算
    if metric_type == "higher_better":
        norm_val = 0.6 + (0.95 - 0.6) * (metric_value - min_val) / (max_val - min_val)
    else:
        norm_val = 0.6 + (0.95 - 0.6) * (max_val - metric_value) / (max_val - min_val)
    
    # 计算通用等效效率值
    universal_eff = norm_val * weight * stability
    return round(universal_eff, 3)
```

### 3.4 步骤 4：运行跨领域适配任务
使用 `run_quick.py` 或 `batch_process.py` 运行不同领域的任务，无需修改核心代码，仅需指定 `--domain` 参数：
```bash
# 运行量子领域任务
python run_quick.py run --domain quantum --data-path ./quantum_qubit.csv

# 运行原子光谱领域任务
python run_quick.py run --domain atomic_spectra --data-path ./atomic_spectra.csv

# 批量运行多领域任务
python batch_process.py --config multi_domain_config.json
```
**多领域批量配置文件示例**（`multi_domain_config.json`）：
```json
{
    "output_dir": "./multi_domain_output",
    "parallel_workers": 2,
    "task_type": "tune",
    "tasks": [
        {"task_id": "quantum_001", "domain": "quantum", "iter": 50, "lr": 0.01},
        {"task_id": "atomic_001", "domain": "atomic_spectra", "iter": 60, "lr": 0.012},
        {"task_id": "gravity_001", "domain": "macro_gravity", "iter": 70, "lr": 0.011}
    ]
}
```

### 3.5 步骤 5：验证跨领域适配效果
通过以下 3 个核心指标验证适配效果：
1. **领域适配得分**（`domain_adapt_score`）：
   - 计算方式：`domain_adapt_score = 通用等效效率值 × 0.95`；
   - 合格标准：`domain_adapt_score ≥ 0.75`。
2. **核心指标一致性**：
   - 对比调优前后领域核心指标的变化，合格标准：**调优后核心指标提升率 ≥ 5%**。
3. **跨领域结果可比性**：
   - 不同领域的通用等效效率值可直接对比，评估智能体在各领域的表现。

**验证代码示例**：
```python
import pandas as pd

# 加载多领域调优结果
quantum_result = pd.read_csv("./multi_domain_output/quantum_001/quantum_001_tune_result.csv")
atomic_result = pd.read_csv("./multi_domain_output/atomic_001/atomic_001_tune_result.csv")

# 计算核心指标提升率
quantum_improve = (quantum_result["best_quantum_fidelity"].iloc[0] - quantum_result["initial_quantum_fidelity"].iloc[0]) / quantum_result["initial_quantum_fidelity"].iloc[0] * 100
atomic_improve = (atomic_result["initial_spectral_resolution"].iloc[0] - atomic_result["best_spectral_resolution"].iloc[0]) / atomic_result["initial_spectral_resolution"].iloc[0] * 100

print(f"量子领域保真度提升率：{quantum_improve:.2f}%")
print(f"原子光谱领域分辨率降低率：{atomic_improve:.2f}%")
```

## 适配性能评估
### 4.1 评估指标体系
| 评估维度 | 核心指标 | 合格标准 |
|----------|----------|----------|
| **适配兼容性** | 领域适配得分 | ≥ 0.75 |
| **调优有效性** | 核心指标提升率 | ≥ 5% |
| **结果一致性** | 通用等效效率值波动 | ≤ 2% |
| **运行效率** | 跨领域任务平均耗时 | 与单领域任务耗时差异 ≤ 10% |

### 4.2 评估方法
1. **单领域适配评估**：
   - 运行该领域的调优任务，计算 `domain_adapt_score` 和核心指标提升率；
   - 若两项指标均达标，则判定该领域适配成功。
2. **跨领域对比评估**：
   - 运行多个领域的调优任务，计算各领域的通用等效效率值；
   - 对比不同领域的等效效率值，评估智能体的跨领域表现。
3. **长期稳定性评估**：
   - 重复运行同一领域任务 10 次，计算通用等效效率值的标准差；
   - 标准差 ≤ 2% 则判定适配效果稳定。

## 新增领域适配指南
若需新增自定义领域（如分子动力学、天体物理），按以下步骤完成适配：
### 5.1 步骤 1：创建领域数据集
1. 确定领域核心指标（如分子动力学的 `energy_minimization`）；
2. 生成符合规范的 CSV 数据集，包含 **5 个通用字段 + 领域特有字段**；
3. 编写数据集生成脚本（`generate_xxx_csv.py`），固定随机种子 `42` 保证可复现。

### 5.2 步骤 2：添加领域映射规则
在 `domain_mapping.json` 中新增一条领域配置：
```json
{
    "domain_name": "molecular_dynamics",
    "core_metric": "energy_minimization",
    "metric_type": "higher_better",
    "metric_range": [0.8, 0.98],
    "weight": 1.0,
    "description": "分子动力学领域核心指标：能量最小化效率"
}
```

### 5.3 步骤 3：验证适配效果
运行新增领域的调优任务，验证 `domain_adapt_score` 和核心指标提升率是否达标。

### 5.4 步骤 4：更新文档
在 `README.md` 和 `cross_domain_adapt.md` 中补充新增领域的说明，便于其他用户使用。

## 常见问题与解决方案
| 问题现象 | 原因分析 | 解决方案 |
|----------|----------|----------|
| 领域适配得分低（< 0.75） | 1. 领域核心指标范围配置错误；2. 数据存在异常值；3. 权重系数设置不合理 | 1. 核对指标范围是否与数据集一致；2. 清理数据中的异常值；3. 调整权重系数至 1.1-1.2 |
| 跨领域调优参数不一致 | 不同领域的最优学习率/迭代次数差异大 | 1. 按领域配置调优参数（在 `domain_mapping.json` 中添加 `lr`/`iter` 字段）；2. 批量脚本中为不同任务指定参数 |
| 新增领域无法识别 | 1. `domain_mapping.json` 中未添加该领域；2. `--domain` 参数拼写错误 | 1. 补充领域映射规则；2. 核对 `--domain` 参数与配置文件中的 `domain_name` 一致 |
| 通用等效效率值计算错误 | 指标优化方向配置错误（如 `higher_better` 写成 `lower_better`） | 修正 `metric_type` 参数，重新运行任务 |

## 总结
UMC-Metabolic-Agent 的跨领域适配能力基于 **「通用适配层 + 领域特征映射」** 架构，实现了「一次开发、多领域复用」的目标。用户仅需遵循以下核心原则，即可快速完成跨领域适配：
1. **数据集规范**：必须包含 5 个通用字段，领域核心指标符合物理规律；
2. **映射规则准确**：正确配置核心指标的范围、优化方向和权重；
3. **效果验证充分**：通过领域适配得分和核心指标提升率验证适配效果。

未来，UMC 智能体将进一步优化跨领域适配算法，支持更复杂的领域特征映射，提升多领域协同调优能力。