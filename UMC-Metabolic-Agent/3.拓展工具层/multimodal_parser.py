# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 多模态数据解析模块（文本/表格/图片/时序/JSON统一解析+标准化）
核心逻辑：适配多源异构数据，统一解析为结构化数值数据，供UMC智能体直接使用
设计原则：模态专属解析、跨模态对齐、智能补全、零配置使用，适配新手处理多模态数据
"""
import pandas as pd
import numpy as np
import json
import os
import re
import cv2
import base64
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# 导入核心工具
from tool_build import create_test_data

class MultimodalParser:
    """多模态数据解析器（核心功能：多类型数据解析、标准化、融合、补全）"""
    def __init__(self, output_dir: str = "./multimodal_processed"):
        """
        初始化多模态解析器
        :param output_dir: 解析后数据的保存目录
        """
        # 基础配置
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 支持的模态类型
        self.supported_modalities = ["table", "text", "image", "timeseries", "json"]
        # 解析历史
        self.parse_history = []
        # 标准化器和补全器（复用避免重复训练）
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)

    def parse_table(self, data_source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        表格数据解析（支持CSV/Excel/JSON文件或DataFrame，核心模态）
        :param data_source: 文件路径或DataFrame
        :param kwargs: 可选参数（header=0, index_col=None, sheet_name=0等）
        :return: 标准化后的表格数据
        """
        print("\n📊 开始表格数据解析...")
        # 1. 加载数据
        if isinstance(data_source, str):
            if data_source.endswith(".csv"):
                df = pd.read_csv(data_source, header=kwargs.get("header", 0), index_col=kwargs.get("index_col", None), encoding="utf-8")
            elif data_source.endswith((".xlsx", ".xls")):
                df = pd.read_excel(data_source, sheet_name=kwargs.get("sheet_name", 0), header=kwargs.get("header", 0), index_col=kwargs.get("index_col", None))
            elif data_source.endswith(".json"):
                df = pd.read_json(data_source, encoding="utf-8")
            else:
                raise ValueError(f"不支持的表格文件格式：{data_source}")
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise TypeError(f"表格数据源类型不支持：{type(data_source)}")

        # 2. 数据清洗（核心：适配UMC智能体输入要求）
        parsed_df = self._clean_and_standardize(df, "table")
        print(f"✅ 表格数据解析完成：{len(parsed_df)}行 × {len(parsed_df.columns)}列")
        return parsed_df

    def parse_text(self, data_source: Union[str, List[str]], text_type: str = "numeric_extract", **kwargs) -> pd.DataFrame:
        """
        文本数据解析（支持文本文件/文本列表，提取数值特征）
        :param data_source: 文本文件路径或文本列表
        :param text_type: 解析类型（numeric_extract：数值提取/keyword_extract：关键词数值化）
        :param kwargs: 可选参数（target_cols：目标列名列表）
        :return: 标准化后的数值表格数据
        """
        print("\n📝 开始文本数据解析...")
        # 1. 加载文本
        if isinstance(data_source, str):
            with open(data_source, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        elif isinstance(data_source, list) and all(isinstance(t, str) for t in data_source):
            texts = data_source
        else:
            raise TypeError(f"文本文源类型不支持：{type(data_source)}")

        # 2. 文本解析（提取数值特征）
        parsed_data = []
        target_cols = kwargs.get("target_cols", ["feature_1", "feature_2", "feature_3"])

        if text_type == "numeric_extract":
            # 提取文本中的所有数值（适配实验报告/监测日志等文本）
            for text in texts:
                # 正则提取浮点数/整数
                nums = re.findall(r'-?\d+\.?\d*', text)
                nums = [float(num) for num in nums] if nums else [0.0]*len(target_cols)
                # 对齐列数（不足补0，超过截断）
                nums = nums[:len(target_cols)] if len(nums) > len(target_cols) else nums + [0.0]*(len(target_cols)-len(nums))
                parsed_data.append(nums)

        elif text_type == "keyword_extract":
            # 关键词数值化（适配领域描述文本）
            # 内置领域关键词库
            domain_keywords = {
                "quantum": ["qubit", "量子", "稳定性", "能耗", "物质输出"],
                "atomic": ["原子", "频率", "能效", "粒子产率"],
                "logistics": ["物流", "效率", "成本", "速度"]
            }
            for text in texts:
                # 计算文本与各领域关键词的匹配度
                keyword_scores = []
                for domain, keywords in domain_keywords.items():
                    score = sum([1 for kw in keywords if kw in text]) / len(keywords)
                    keyword_scores.append(score)
                # 对齐列数
                keyword_scores = keyword_scores[:len(target_cols)] if len(keyword_scores) > len(target_cols) else keyword_scores + [0.0]*(len(target_cols)-len(keyword_scores))
                parsed_data.append(keyword_scores)

        else:
            raise ValueError(f"不支持的文本解析类型：{text_type}")

        # 3. 转为DataFrame并标准化
        df = pd.DataFrame(parsed_data, columns=target_cols)
        parsed_df = self._clean_and_standardize(df, "text")
        print(f"✅ 文本数据解析完成：{len(parsed_df)}行 × {len(parsed_df.columns)}列")
        return parsed_df

    def parse_image(self, data_source: Union[str, List[str]], extract_type: str = "pixel_stat", **kwargs) -> pd.DataFrame:
        """
        图片数据解析（支持图片文件/路径列表，数值化提取视觉特征）
        :param data_source: 图片文件路径或路径列表
        :param extract_type: 特征提取类型（pixel_stat：像素统计/edge_density：边缘密度）
        :param kwargs: 可选参数（target_cols：目标列名列表）
        :return: 标准化后的数值表格数据
        """
        print("\n🖼️ 开始图片数据解析...")
        # 检查OpenCV依赖
        try:
            import cv2
        except ImportError:
            raise ImportError("解析图片需要安装OpenCV：pip install opencv-python")

        # 1. 加载图片路径
        if isinstance(data_source, str):
            if os.path.isdir(data_source):
                # 目录下所有图片
                img_ext = [".jpg", ".jpeg", ".png", ".bmp"]
                img_paths = [os.path.join(data_source, f) for f in os.listdir(data_source) if f.lower().endswith(tuple(img_ext))]
            elif os.path.isfile(data_source) and data_source.lower().endswith(tuple([".jpg", ".jpeg", ".png", ".bmp"])):
                img_paths = [data_source]
            else:
                raise ValueError(f"图片源无效：{data_source}")
        elif isinstance(data_source, list) and all(isinstance(p, str) for p in data_source):
            img_paths = data_source
        else:
            raise TypeError(f"图片源类型不支持：{type(data_source)}")

        # 2. 图片特征提取（数值化）
        parsed_data = []
        target_cols = kwargs.get("target_cols", ["mean_brightness", "contrast", "edge_density"])

        for img_path in img_paths:
            try:
                # 读取图片（灰度图简化处理）
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️  无法读取图片：{img_path}，跳过")
                    continue

                # 特征提取
                if extract_type == "pixel_stat":
                    # 像素统计特征（亮度均值、对比度、熵）
                    mean_bright = np.mean(img) / 255.0  # 归一化到0~1
                    contrast = np.std(img) / 255.0
                    entropy = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
                    entropy = -np.sum(entropy * np.log2(entropy + 1e-8)) / np.log2(256)  # 归一化熵
                    features = [mean_bright, contrast, entropy]

                elif extract_type == "edge_density":
                    # 边缘密度特征（Canny边缘检测）
                    edges = cv2.Canny(img, 100, 200)
                    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])  # 边缘像素占比
                    # 补充其他特征
                    mean_bright = np.mean(img) / 255.0
                    contrast = np.std(img) / 255.0
                    features = [edge_density, mean_bright, contrast]

                else:
                    raise ValueError(f"不支持的图片提取类型：{extract_type}")

                # 对齐列数
                features = features[:len(target_cols)] if len(features) > len(target_cols) else features + [0.0]*(len(target_cols)-len(features))
                parsed_data.append(features)

            except Exception as e:
                print(f"⚠️  处理图片{img_path}失败：{str(e)}，填充默认值")
                parsed_data.append([0.0]*len(target_cols))

        # 3. 转为DataFrame并标准化
        df = pd.DataFrame(parsed_data, columns=target_cols)
        parsed_df = self._clean_and_standardize(df, "image")
        print(f"✅ 图片数据解析完成：{len(parsed_df)}行 × {len(parsed_df.columns)}列")
        return parsed_df

    def parse_timeseries(self, data_source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        时序数据解析（支持CSV/Excel/JSON或DataFrame，提取时序特征）
        :param data_source: 文件路径或DataFrame（需包含时间列和数值列）
        :param kwargs: 可选参数（time_col：时间列名，window_size：滑动窗口大小）
        :return: 标准化后的时序特征表格数据
        """
        print("\n⏱️ 开始时序数据解析...")
        # 1. 加载时序数据
        if isinstance(data_source, str):
            if data_source.endswith(".csv"):
                df = pd.read_csv(data_source, encoding="utf-8")
            elif data_source.endswith((".xlsx", ".xls")):
                df = pd.read_excel(data_source, encoding="utf-8")
            elif data_source.endswith(".json"):
                df = pd.read_json(data_source, encoding="utf-8")
            else:
                raise ValueError(f"不支持的时序文件格式：{data_source}")
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise TypeError(f"时序数据源类型不支持：{type(data_source)}")

        # 2. 时序特征提取
        time_col = kwargs.get("time_col", "timestamp")
        window_size = kwargs.get("window_size", 5)
        # 确保时间列是datetime类型
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        # 按时间排序
        df = df.sort_values(by=time_col)

        # 提取数值列（排除时间列）
        numeric_cols = [col for col in df.columns if col != time_col and pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            raise ValueError("时序数据中无数值列可提取特征")

        # 滑动窗口提取时序特征（均值、方差、趋势、峰值）
        parsed_data = []
        target_cols = []
        for col in numeric_cols:
            target_cols.extend([f"{col}_mean", f"{col}_std", f"{col}_trend", f"{col}_peak"])

        # 滑动窗口计算
        for i in range(window_size-1, len(df)):
            window = df.iloc[i-window_size+1:i+1]
            row_features = []
            for col in numeric_cols:
                window_vals = window[col].values
                # 均值
                mean_val = np.mean(window_vals)
                # 方差
                std_val = np.std(window_vals)
                # 趋势（线性回归斜率）
                x = np.arange(window_size)
                trend = np.polyfit(x, window_vals, 1)[0] if window_size >= 2 else 0.0
                # 峰值（窗口内最大值）
                peak_val = np.max(window_vals)
                # 添加特征
                row_features.extend([mean_val, std_val, trend, peak_val])
            parsed_data.append(row_features)

        # 3. 转为DataFrame并标准化
        df_parsed = pd.DataFrame(parsed_data, columns=target_cols)
        parsed_df = self._clean_and_standardize(df_parsed, "timeseries")
        print(f"✅ 时序数据解析完成：{len(parsed_df)}行 × {len(parsed_df.columns)}列")
        return parsed_df

    def parse_json(self, data_source: Union[str, dict, List[dict]], **kwargs) -> pd.DataFrame:
        """
        半结构化JSON解析（支持JSON文件/字典/字典列表，扁平化为表格）
        :param data_source: JSON文件路径/字典/字典列表
        :param kwargs: 可选参数（flatten_depth：扁平化深度）
        :return: 标准化后的表格数据
        """
        print("\n🔧 开始JSON数据解析...")
        # 1. 加载JSON数据
        if isinstance(data_source, str):
            with open(data_source, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        elif isinstance(data_source, (dict, list)):
            json_data = data_source
        else:
            raise TypeError(f"JSON源类型不支持：{type(data_source)}")

        # 2. 扁平化JSON（处理嵌套结构）
        flatten_depth = kwargs.get("flatten_depth", 2)
        df = self._flatten_json(json_data, flatten_depth)

        # 3. 标准化处理
        parsed_df = self._clean_and_standardize(df, "json")
        print(f"✅ JSON数据解析完成：{len(parsed_df)}行 × {len(parsed_df.columns)}列")
        return parsed_df

    def _flatten_json(self, json_data: Union[dict, List[dict]], depth: int = 2, parent_key: str = "") -> pd.DataFrame:
        """
        扁平化嵌套JSON（核心辅助方法）
        :param json_data: JSON数据
        :param depth: 扁平化深度
        :param parent_key: 父键（递归用）
        :return: 扁平化后的DataFrame
        """
        flat_data = []

        def _flatten(item: Any, current_depth: int, current_key: str):
            if current_depth > depth:
                return {current_key: str(item)} if current_key else {}
            if isinstance(item, dict):
                result = {}
                for k, v in item.items():
                    new_key = f"{current_key}_{k}" if current_key else k
                    result.update(_flatten(v, current_depth+1, new_key))
                return result
            elif isinstance(item, list):
                result_list = []
                for i, elem in enumerate(item):
                    new_key = f"{current_key}_{i}" if current_key else str(i)
                    result_list.append(_flatten(elem, current_depth+1, new_key))
                # 合并列表项为行
                merged = {}
                for res in result_list:
                    merged.update(res)
                return merged
            else:
                return {current_key: item} if current_key else {}

        # 处理单字典或字典列表
        if isinstance(json_data, dict):
            flat_data.append(_flatten(json_data, 1, parent_key))
        elif isinstance(json_data, list):
            for item in json_data:
                flat_item = _flatten(item, 1, parent_key)
                flat_data.append(flat_item)

        # 转为DataFrame
        df = pd.DataFrame(flat_data)
        # 过滤非数值列（保留可转换为数值的列）
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                pass
        # 只保留数值列
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        df = df[numeric_cols] if numeric_cols else pd.DataFrame()
        return df

    def _clean_and_standardize(self, df: pd.DataFrame, modality: str) -> pd.DataFrame:
        """
        通用数据清洗与标准化（所有模态的统一处理逻辑）
        :param df: 原始解析数据
        :param modality: 模态类型
        :return: 标准化后的DataFrame
        """
        # 1. 去重
        df_clean = df.drop_duplicates()
        # 2. 过滤异常值（3σ原则）
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean = df_clean[(df_clean[col] >= mean - 3*std) & (df_clean[col] <= mean + 3*std)]
        # 3. 缺失值补全（KNN）
        if not df_clean.empty and len(numeric_cols) > 0:
            df_clean[numeric_cols] = self.imputer.fit_transform(df_clean[numeric_cols])
        # 4. 标准化（均值0，方差1）
        if not df_clean.empty and len(numeric_cols) > 0:
            df_clean[numeric_cols] = self.scaler.fit_transform(df_clean[numeric_cols])
            # 映射到0~1区间（适配UMC智能体输入要求）
            df_clean[numeric_cols] = (df_clean[numeric_cols] - df_clean[numeric_cols].min()) / (df_clean[numeric_cols].max() - df_clean[numeric_cols].min() + 1e-8)
        # 5. 重置索引
        df_clean = df_clean.reset_index(drop=True)
        # 6. 保存解析后的数据
        save_path = os.path.join(self.output_dir, f"{modality}_parsed_{time.strftime('%Y%m%d%H%M%S')}.csv")
        df_clean.to_csv(save_path, index=False, encoding="utf-8")
        print(f"📁 解析后数据已保存：{save_path}")
        return df_clean

    def fuse_multimodal_data(self, data_dict: Dict[str, pd.DataFrame], align_method: str = "sample_count") -> pd.DataFrame:
        """
        多模态数据融合（将不同模态解析后的数据融合为统一表格）
        :param data_dict: 模态-数据字典（如{"table": df1, "text": df2}）
        :param align_method: 对齐方法（sample_count：按样本数截断/col_merge：列合并）
        :return: 融合后的标准化数据
        """
        print("\n🔗 开始多模态数据融合...")
        if not data_dict:
            raise ValueError("无多模态数据可融合")

        # 1. 数据对齐
        fused_df = None
        if align_method == "sample_count":
            # 按最小样本数截断所有数据
            min_samples = min([len(df) for df in data_dict.values()])
            aligned_dfs = [df.iloc[:min_samples].reset_index(drop=True) for df in data_dict.values()]
            # 列重命名避免冲突
            renamed_dfs = []
            for idx, (modality, df) in enumerate(data_dict.items()):
                df_aligned = df.iloc[:min_samples].reset_index(drop=True)
                df_renamed = df_aligned.rename(columns={col: f"{modality}_{col}" for col in df_aligned.columns})
                renamed_dfs.append(df_renamed)
            # 横向合并
            fused_df = pd.concat(renamed_dfs, axis=1)

        elif align_method == "col_merge":
            # 列合并（要求所有数据样本数相同）
            sample_counts = [len(df) for df in data_dict.values()]
            if len(set(sample_counts)) > 1:
                raise ValueError("col_merge方法要求所有模态数据样本数相同")
            renamed_dfs = []
            for modality, df in data_dict.items():
                df_renamed = df.rename(columns={col: f"{modality}_{col}" for col in df.columns})
                renamed_dfs.append(df_renamed)
            fused_df = pd.concat(renamed_dfs, axis=1)

        else:
            raise ValueError(f"不支持的融合对齐方法：{align_method}")

        # 2. 标准化融合后的数据
        fused_df = self._clean_and_standardize(fused_df, "multimodal_fused")
        print(f"✅ 多模态数据融合完成：{len(fused_df)}行 × {len(fused_df.columns)}列")
        return fused_df

    def run_multimodal_parse(self, parse_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        一键运行多模态解析（支持批量解析不同模态数据）
        :param parse_config: 解析配置字典
        示例：
        {
            "table": {"data_source": "./quantum_data.csv"},
            "text": {"data_source": ["./report.txt"], "text_type": "numeric_extract"},
            "image": {"data_source": "./img_dir", "extract_type": "edge_density"}
        }
        :return: 模态-解析后数据字典
        """
        print("\n🚀 开始多模态批量解析...")
        parse_results = {}
        # 遍历配置解析各模态
        for modality, config in parse_config.items():
            if modality not in self.supported_modalities:
                print(f"⚠️  不支持的模态类型：{modality}，跳过")
                continue
            try:
                if modality == "table":
                    df = self.parse_table(**config)
                elif modality == "text":
                    df = self.parse_text(**config)
                elif modality == "image":
                    df = self.parse_image(**config)
                elif modality == "timeseries":
                    df = self.parse_timeseries(**config)
                elif modality == "json":
                    df = self.parse_json(**config)
                parse_results[modality] = df
                # 记录解析历史
                self.parse_history.append({
                    "modality": modality,
                    "config": config,
                    "sample_count": len(df),
                    "col_count": len(df.columns),
                    "parse_time": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print(f"❌ 解析{modality}模态失败：{str(e)}")

        print(f"✅ 多模态批量解析完成！成功解析{len(parse_results)}种模态")
        return parse_results

# 多模态解析模块验证入口（一站式测试）
if __name__ == "__main__":
    # 1. 初始化多模态解析器
    parser = MultimodalParser()
    print("🚀 多模态解析器初始化完成！")

    # 2. 生成测试数据
    # 表格测试数据
    table_data = create_test_data(domain_name="quantum", sample_count=100)
    table_data_path = "./test_multimodal_table.csv"
    table_data.to_csv(table_data_path, index=False, encoding="utf-8")

    # 文本测试数据（实验报告文本）
    text_data = [
        "量子实验报告：qubit稳定性0.85，能耗0.72，物质输出0.68",
        "原子实验报告：原子频率0.78，能效0.65，粒子产率0.59",
        "物流监测：物流效率0.82，运输成本0.75，配送速度0.69"
    ]
    text_data_path = "./test_multimodal_text.txt"
    with open(text_data_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_data))

    # 时序测试数据
    timeseries_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-01-01", periods=50, freq="H"),
        "qubit_stability": np.random.rand(50)*0.9,
        "energy_consumption": np.random.rand(50)*0.8
    })
    timeseries_data_path = "./test_multimodal_timeseries.csv"
    timeseries_data.to_csv(timeseries_data_path, index=False, encoding="utf-8")

    # JSON测试数据
    json_data = [
        {"quantum": {"qubit_stability": 0.85, "energy_consumption": 0.72}, "timestamp": "2026-01-01 00:00:00"},
        {"quantum": {"qubit_stability": 0.82, "energy_consumption": 0.75}, "timestamp": "2026-01-01 01:00:00"}
    ]
    json_data_path = "./test_multimodal_json.json"
    with open(json_data_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 3. 一键多模态解析
    parse_config = {
        "table": {"data_source": table_data_path},
        "text": {"data_source": text_data_path, "text_type": "numeric_extract", "target_cols": ["qubit_stability", "energy_consumption", "matter_output"]},
        "timeseries": {"data_source": timeseries_data_path, "time_col": "timestamp", "window_size": 5},
        "json": {"data_source": json_data_path, "flatten_depth": 2}
    }
    parse_results = parser.run_multimodal_parse(parse_config)

    # 4. 多模态数据融合
    if len(parse_results) >= 2:
        fused_data = parser.fuse_multimodal_data(parse_results, align_method="sample_count")
        print(f"\n📊 多模态融合后数据维度：{len(fused_data)}行 × {len(fused_data.columns)}列")

    # 5. 查看解析历史
    print("\n📜 解析历史汇总：")
    for idx, history in enumerate(parser.parse_history):
        print(f"  {idx+1}. 模态：{history['modality']} | 样本数：{history['sample_count']} | 列数：{history['col_count']}")

    print("\n🎉 多模态解析模块测试完成！所有解析后数据已保存至 ./multimodal_processed")