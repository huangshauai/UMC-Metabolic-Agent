# UMC-Metabolic-Agent
UMC-Metabolic-Agent 是一款面向多领域（量子计算、原子光谱、宏观引力等）的智能调优与分析工具，支持批量处理、参数调优、结果分析等核心功能，具备高兼容性、可复现性和易用性，适配不同领域的专用数据集，开箱即用。

## 🚀 核心功能
| 功能模块 | 核心能力 | 适用场景 |
|----------|----------|----------|
| 智能运行 | 支持多领域数据集的快速运行，输出核心指标 | 单任务快速验证、数据初筛 |
| 参数调优 | 自适应学习率/迭代次数调优，实时收敛监控 | 核心指标优化、参数寻优 |
| 批量处理 | 多任务并行执行，结果自动汇总，失败隔离 | 多参数对比、大规模数据处理 |
| 结果分析 | 结构化分析报告，核心指标统计，可视化输出 | 调优结果评估、领域对比分析 |
| 质量控制 | 完善的异常处理、日志监控、数据校验 | 生产环境部署、结果可靠性保障 |

## ⚡ 快速开始
### 1. 环境准备
- Python 版本：3.8+（推荐3.9/3.10）
- 操作系统：Windows/Linux/macOS（无系统限制）
- 依赖：无需额外编译，仅需Python第三方库

### 2. 安装依赖
```bash
# 克隆项目（若有）或直接创建目录，将所有脚本/数据集放入
mkdir UMC-Metabolic-Agent && cd UMC-Metabolic-Agent

# 创建并激活虚拟环境（可选，推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装核心依赖
pip install pandas numpy matplotlib psutil bcrypt pyjwt flask requests pytest
```

### 3. 快速运行（5分钟上手）
#### 步骤1：生成测试数据集
以量子比特数据集为例，生成1000条完整数据：
```bash
# 将generate_quantum_qubit_csv.py放入当前目录，运行
python generate_quantum_qubit_csv.py
```
生成成功后会得到 `quantum_qubit.csv`，包含量子领域完整测试数据。

#### 步骤2：快速运行智能体
```bash
# 运行量子领域智能体，输出核心指标
python run_quick.py run --domain quantum --data-path ./quantum_qubit.csv
```

#### 步骤3：调优核心指标
```bash
# 调优量子保真度（核心优化目标）
python example_tuner_operation.py
# （运行前修改脚本中test_data_path为./quantum_qubit.csv）
```

#### 步骤4：批量处理多任务
```bash
# 运行通用批量处理脚本
python batch_process.py --task-type tune --workers 2
```

## 📁 核心模块说明
| 模块文件 | 功能说明 | 关键特性 |
|----------|----------|----------|
| `universal_cmd.py` | 通用命令模块 | 支持运行/分析/报告生成，兼容所有数据集 |
| `tuner_dashboard.py` | 调优仪表盘模块 | 实时收敛监控、参数自适应调整、状态可视化 |
| `ext_identity.py` | 身份认证模块 | 密码哈希存储、JWT令牌、RBAC权限控制 |
| `custom_app_api.py` | API接口模块 | RESTful规范、异步任务、跨域支持 |
| `result_analysis.py` | 结果分析模块 | 结构化报告、核心指标统计、可视化图表 |

## 📊 专用数据集说明
### 1. 数据集列表
| 数据集文件 | 适用领域 | 核心优化指标 | 生成脚本 |
|------------|----------|--------------|----------|
| `quantum_qubit.csv` | 量子比特调控 | 量子态保真度（quantum_fidelity） | `generate_quantum_qubit_csv.py` |
| `atomic_spectra.csv` | 原子光谱分析 | 光谱分辨率（spectral_resolution） | `generate_atomic_spectra_csv.py` |
| `macro_gravity.csv` | 宏观引力分析 | 测地线精度（geodesic_accuracy） | `generate_macro_gravity_csv.py` |

### 2. 数据集通用特性
- 所有数据集均包含1000条有效样本，固定随机种子（42）保证可复现；
- 字段兼容UMC智能体所有脚本，无需修改代码即可直接使用；
- 数值符合对应领域物理规律，无异常值/缺失值；
- 包含通用字段：`timestamp`/`sample_id`/`domain_adapt_score`/`core_factor`/`stability`。

### 3. 数据集快速使用
```python
# 示例：读取并验证量子比特数据集
import pandas as pd

df = pd.read_csv("quantum_qubit.csv")
print(f"数据规模：{len(df)}条 | 字段数：{len(df.columns)}")
print(f"核心指标（量子保真度）范围：{df['quantum_fidelity'].min()} ~ {df['quantum_fidelity'].max()}")
```

## 🛠️ 常用脚本使用指南
### 1. 快速运行脚本（run_quick.py）
```bash
# 基本用法
python run_quick.py [run/tune/analyze] --domain <领域名> --data-path <数据文件路径>

# 示例：运行原子光谱数据集
python run_quick.py run --domain atomic_spectra --data-path ./atomic_spectra.csv

# 示例：调优宏观引力数据集
python run_quick.py tune --domain macro_gravity --data-path ./macro_gravity.csv --iter 80 --lr 0.011
```

### 2. 通用批量处理脚本（batch_process.py）
```bash
# 基本用法
python batch_process.py [--config <配置文件>] [--task-type <任务类型>] [--workers <并行数>]

# 示例：使用自定义配置文件
python batch_process.py --config my_config.json

# 示例：仅运行任务，4线程并行
python batch_process.py --task-type run --workers 4 --output ./my_batch_output
```

### 3. 调优示例脚本（example_tuner_operation.py）
```bash
# 直接运行（需先修改脚本内的数据集路径）
python example_tuner_operation.py
```

### 4. 数据集生成脚本
```bash
# 量子比特数据集
python generate_quantum_qubit_csv.py

# 原子光谱数据集
python generate_atomic_spectra_csv.py

# 宏观引力数据集
python generate_macro_gravity_csv.py
```

## ✅ 质量控制
### 1. 质量规范文档
完整的质量控制标准请参考 `quality_control.txt`，涵盖：
- 核心模块质量要求（代码规范、依赖管理、容错机制）；
- 数据集质量标准（字段规范、数据合理性、可复现性）；
- 运行/调优质量控制（输入验证、过程监控、结果验证）；
- 质量检查流程与常见问题解决方案。

### 2. 日常质量检查
```bash
# 验证数据集完整性
python -c "import pandas as pd; df=pd.read_csv('quantum_qubit.csv'); print(f'缺失值：{df.isnull().any().any()}')"

# 验证模块导入
python -c "from tuner_dashboard import TunerDashboard; print('调优模块导入成功')"

# 验证快速运行功能
python run_quick.py run --domain quantum --data-path ./quantum_qubit.csv --test
```

## ❓ 常见问题
| 问题现象 | 解决方案 |
|----------|----------|
| 模块导入失败 | 1. 检查核心文件是否在当前目录；2. 重新安装依赖：`pip install -r requirements.txt` |
| 数据生成失败 | 1. 检查随机种子设置（np.random.seed(42)）；2. 核对字段名拼写 |
| 调优不收敛 | 1. 降低学习率（如0.01→0.008）；2. 增加迭代次数（如50→80） |
| 脚本运行报错 | 1. 检查参数范围（迭代次数≥10，学习率≥0.001）；2. 确认文件路径正确 |
| 输出结果异常 | 1. 清理数据异常值；2. 验证核心指标计算逻辑 |

## 📜 版本更新日志
| 版本 | 日期 | 核心更新 |
|------|------|----------|
| v1.0 | 2026-01-01 | 初始版本，支持量子领域数据集和基础调优功能 |
| v1.5 | 2026-01-10 | 新增原子光谱/宏观引力数据集，完善批量处理功能 |
| v2.0 | 2026-01-18 | 补充质量控制规范，优化异常处理，完善API接口 |

## 🧑‍💻 维护与反馈
- **维护团队**：UMC-Metabolic-Agent 开发团队
- **问题反馈**：提交Issue至项目仓库，或联系维护人员
- **贡献指南**：欢迎提交PR，所有贡献需符合`quality_control.txt`中的质量标准

## 📄 许可证
本项目采用 MIT 许可证，详情请参考 LICENSE 文件。

---
### 备注
- 所有脚本/数据集文件均可直接复制使用，无需额外修改；
- 若需扩展新领域数据集，可参考现有生成脚本的结构，保持字段兼容性；
- 生产环境使用前，请务必通过`quality_control.txt`中的质量检查流程。