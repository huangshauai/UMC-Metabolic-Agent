# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 可视化图表生成模块（多类型图表+自动适配+高清导出）
核心逻辑：将智能体运行结果/自适应效果/多模态数据转为专业可视化图表，支持一键生成/导出
设计原则：自动适配数据、零配置生成、高清可视化、多格式导出，适配新手快速分析数据
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体（避免乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# 设置高清显示
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

class PlotGenerator:
    """可视化图表生成器（核心功能：多类型图表生成、批量绘图、结果导出）"""
    def __init__(self, output_dir: str = "./plots", style: str = "seaborn-v0_8-whitegrid"):
        """
        初始化图表生成器
        :param output_dir: 图表保存目录
        :param style: 绘图风格（matplotlib/seaborn风格）
        """
        # 基础配置
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.style = style
        plt.style.use(style)
        # 支持的图表类型
        self.supported_plots = ["line", "bar", "heatmap", "scatter", "radar", "hist", "box"]
        # 绘图历史
        self.plot_history = []

    def generate_line_plot(self, data: pd.DataFrame, x_col: str, y_cols: List[str], **kwargs) -> str:
        """
        生成折线图（适配时序数据/运行趋势数据）
        :param data: 绘图数据
        :param x_col: X轴列名
        :param y_cols: Y轴列名列表
        :param kwargs: 可选参数（title/ylabel/xlabel/figsize/colors/save_name）
        :return: 图表保存路径
        """
        print("\n📈 开始生成折线图...")
        # 解析参数
        title = kwargs.get("title", f"{x_col} vs {', '.join(y_cols)}")
        xlabel = kwargs.get("xlabel", x_col)
        ylabel = kwargs.get("ylabel", "数值")
        figsize = kwargs.get("figsize", (12, 6))
        colors = kwargs.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        save_name = kwargs.get("save_name", f"line_plot_{time.strftime('%Y%m%d%H%M%S')}")

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制折线
        for idx, y_col in enumerate(y_cols):
            if y_col in data.columns:
                color = colors[idx % len(colors)]
                ax.plot(data[x_col], data[y_col], label=y_col, color=color, linewidth=2, marker="o", markersize=4)

        # 图表美化
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # 记录历史
        self.plot_history.append({
            "plot_type": "line",
            "title": title,
            "save_path": save_path,
            "data_shape": data.shape,
            "plot_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ 折线图生成完成：{save_path}")
        return save_path

    def generate_bar_plot(self, data: pd.DataFrame, x_col: str, y_cols: List[str], **kwargs) -> str:
        """
        生成柱状图（适配对比数据/指标得分数据）
        :param data: 绘图数据
        :param x_col: X轴列名
        :param y_cols: Y轴列名列表
        :param kwargs: 可选参数（title/ylabel/xlabel/figsize/colors/bar_width/save_name）
        :return: 图表保存路径
        """
        print("\n📊 开始生成柱状图...")
        # 解析参数
        title = kwargs.get("title", f"{x_col} vs {', '.join(y_cols)}")
        xlabel = kwargs.get("xlabel", x_col)
        ylabel = kwargs.get("ylabel", "数值")
        figsize = kwargs.get("figsize", (12, 6))
        colors = kwargs.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        bar_width = kwargs.get("bar_width", 0.2)
        save_name = kwargs.get("save_name", f"bar_plot_{time.strftime('%Y%m%d%H%M%S')}")

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制分组柱状图
        x_pos = np.arange(len(data[x_col].unique()))
        for idx, y_col in enumerate(y_cols):
            if y_col in data.columns:
                values = data[y_col].values[:len(x_pos)]
                offset = (idx - len(y_cols)/2 + 0.5) * bar_width
                ax.bar(x_pos + offset, values, width=bar_width, label=y_col, color=colors[idx % len(colors)], alpha=0.8)

        # 设置X轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data[x_col].unique(), rotation=45, ha="right")

        # 图表美化
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # 记录历史
        self.plot_history.append({
            "plot_type": "bar",
            "title": title,
            "save_path": save_path,
            "data_shape": data.shape,
            "plot_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ 柱状图生成完成：{save_path}")
        return save_path

    def generate_heatmap(self, data: pd.DataFrame, **kwargs) -> str:
        """
        生成热力图（适配相关性矩阵/相似度矩阵）
        :param data: 绘图数据（矩阵形式）
        :param kwargs: 可选参数（title/annot/vmin/vmax/figsize/cmap/save_name）
        :return: 图表保存路径
        """
        print("\n🔥 开始生成热力图...")
        # 解析参数
        title = kwargs.get("title", "相关性热力图")
        annot = kwargs.get("annot", True)  # 是否显示数值
        vmin = kwargs.get("vmin", 0)
        vmax = kwargs.get("vmax", 1)
        figsize = kwargs.get("figsize", (10, 8))
        cmap = kwargs.get("cmap", "RdBu_r")
        save_name = kwargs.get("save_name", f"heatmap_{time.strftime('%Y%m%d%H%M%S')}")

        # 只保留数值列
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("热力图数据中无有效数值列")

        # 计算相关性矩阵（如果输入不是矩阵）
        if numeric_data.shape[0] != numeric_data.shape[1]:
            plot_data = numeric_data.corr()
        else:
            plot_data = numeric_data

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        sns.heatmap(
            plot_data,
            ax=ax,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "相关系数"}
        )

        # 图表美化
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # 记录历史
        self.plot_history.append({
            "plot_type": "heatmap",
            "title": title,
            "save_path": save_path,
            "data_shape": data.shape,
            "plot_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ 热力图生成完成：{save_path}")
        return save_path

    def generate_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> str:
        """
        生成散点图（适配特征分布/变量关系分析）
        :param data: 绘图数据
        :param x_col: X轴列名
        :param y_col: Y轴列名
        :param kwargs: 可选参数（title/hue_col/figsize/alpha/size/save_name）
        :return: 图表保存路径
        """
        print("\n🔵 开始生成散点图...")
        # 解析参数
        title = kwargs.get("title", f"{x_col} vs {y_col} 散点图")
        hue_col = kwargs.get("hue_col", None)  # 分组列
        figsize = kwargs.get("figsize", (10, 6))
        alpha = kwargs.get("alpha", 0.7)
        size = kwargs.get("size", 50)
        save_name = kwargs.get("save_name", f"scatter_plot_{time.strftime('%Y%m%d%H%M%S')}")

        # 检查列是否存在
        if x_col not in data.columns or y_col not in data.columns:
            raise ValueError(f"X/Y轴列名不存在：{x_col}/{y_col}")

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制散点图
        if hue_col and hue_col in data.columns:
            # 分组散点图
            unique_hues = data[hue_col].unique()
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(unique_hues)]
            for idx, hue_val in enumerate(unique_hues):
                hue_data = data[data[hue_col] == hue_val]
                ax.scatter(
                    hue_data[x_col], hue_data[y_col],
                    label=str(hue_val),
                    color=colors[idx],
                    alpha=alpha,
                    s=size
                )
            ax.legend(fontsize=10, loc="best")
        else:
            # 普通散点图
            ax.scatter(
                data[x_col], data[y_col],
                color="#1f77b4",
                alpha=alpha,
                s=size,
                edgecolors="black",
                linewidths=0.5
            )

        # 图表美化
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # 记录历史
        self.plot_history.append({
            "plot_type": "scatter",
            "title": title,
            "save_path": save_path,
            "data_shape": data.shape,
            "plot_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ 散点图生成完成：{save_path}")
        return save_path

    def generate_radar_plot(self, data: pd.DataFrame, categories: List[str], **kwargs) -> str:
        """
        生成雷达图（适配多维度指标对比/自适应效果评估）
        :param data: 绘图数据（每行一个样本，每列一个维度）
        :param categories: 维度列表（对应列名）
        :param kwargs: 可选参数（title/figsize/colors/labels/save_name）
        :return: 图表保存路径
        """
        print("\n🎯 开始生成雷达图...")
        # 解析参数
        title = kwargs.get("title", "多维度指标雷达图")
        figsize = kwargs.get("figsize", (8, 8))
        colors = kwargs.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        labels = kwargs.get("labels", [f"样本{i+1}" for i in range(len(data))])
        save_name = kwargs.get("save_name", f"radar_plot_{time.strftime('%Y%m%d%H%M%S')}")

        # 检查维度列是否存在
        missing_cols = [cat for cat in categories if cat not in data.columns]
        if missing_cols:
            raise ValueError(f"维度列不存在：{missing_cols}")

        # 数据标准化到0~1（雷达图适配）
        plot_data = data[categories].copy()
        plot_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min() + 1e-8)

        # 计算角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # 绘制每个样本的雷达图
        for idx in range(len(plot_data)):
            values = plot_data.iloc[idx].values.tolist()
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, linewidth=2, label=labels[idx], color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.2, color=colors[idx % len(colors)])

        # 设置维度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, alpha=0.7)

        # 图表美化
        ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
        ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.2, 1.0))
        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # 记录历史
        self.plot_history.append({
            "plot_type": "radar",
            "title": title,
            "save_path": save_path,
            "data_shape": data.shape,
            "plot_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ 雷达图生成完成：{save_path}")
        return save_path

    def generate_hist_plot(self, data: pd.DataFrame, cols: List[str], **kwargs) -> str:
        """
        生成直方图（适配数据分布分析）
        :param data: 绘图数据
        :param cols: 列名列表
        :param kwargs: 可选参数（title/bins/figsize/colors/save_name）
        :return: 图表保存路径
        """
        print("\n📊 开始生成直方图...")
        # 解析参数
        title = kwargs.get("title", "数据分布直方图")
        bins = kwargs.get("bins", 20)
        figsize = kwargs.get("figsize", (12, 6))
        colors = kwargs.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        save_name = kwargs.get("save_name", f"hist_plot_{time.strftime('%Y%m%d%H%M%S')}")

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制直方图
        for idx, col in enumerate(cols):
            if col in data.columns:
                ax.hist(
                    data[col].dropna(),
                    bins=bins,
                    label=col,
                    color=colors[idx % len(colors)],
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5
                )

        # 图表美化
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("数值", fontsize=12)
        ax.set_ylabel("频数", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # 记录历史
        self.plot_history.append({
            "plot_type": "hist",
            "title": title,
            "save_path": save_path,
            "data_shape": data.shape,
            "plot_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ 直方图生成完成：{save_path}")
        return save_path

    def generate_box_plot(self, data: pd.DataFrame, x_col: str, y_cols: List[str], **kwargs) -> str:
        """
        生成箱线图（适配异常值分析/数据离散程度分析）
        :param data: 绘图数据
        :param x_col: X轴列名（分组）
        :param y_cols: Y轴列名列表
        :param kwargs: 可选参数（title/figsize/colors/save_name）
        :return: 图表保存路径
        """
        print("\n📦 开始生成箱线图...")
        # 解析参数
        title = kwargs.get("title", "数据分布箱线图")
        figsize = kwargs.get("figsize", (12, 6))
        colors = kwargs.get("colors", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        save_name = kwargs.get("save_name", f"box_plot_{time.strftime('%Y%m%d%H%M%S')}")

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)

        # 准备数据
        plot_data = []
        plot_labels = []
        for y_col in y_cols:
            if y_col in data.columns:
                for x_val in data[x_col].unique():
                    plot_data.append(data[data[x_col] == x_val][y_col].dropna())
                    plot_labels.append(f"{y_col}-{x_val}")

        # 绘制箱线图
        bp = ax.boxplot(
            plot_data,
            labels=plot_labels,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 4}
        )

        # 美化箱线图颜色
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[idx % len(colors)])
            patch.set_alpha(0.7)

        # 图表美化
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel("数值", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # 记录历史
        self.plot_history.append({
            "plot_type": "box",
            "title": title,
            "save_path": save_path,
            "data_shape": data.shape,
            "plot_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ 箱线图生成完成：{save_path}")
        return save_path

    def generate_adapt_report_plots(self, adapt_result: Dict[str, Any]) -> List[str]:
        """
        一键生成自适应效果分析报告图表（适配unsupervised_adapt.py的输出）
        :param adapt_result: 自适应结果字典
        :return: 生成的图表路径列表
        """
        print("\n📋 开始生成自适应效果分析报告图表...")
        plot_paths = []

        # 1. 提取自适应效果数据
        effect_data = pd.DataFrame([adapt_result["adapt_effect"]])
        domain = adapt_result["domain_match"]["domain"]
        adapt_time = adapt_result["start_time"]

        # 2. 生成自适应效果柱状图
        bar_path = self.generate_bar_plot(
            data=effect_data,
            x_col=pd.Series([f"{domain}领域"]),
            y_cols=["metabolic_stability", "result_consistency", "run_efficiency", "performance_rate"],
            title=f"{adapt_time} {domain}领域自适应效果指标",
            xlabel="领域",
            ylabel="得分（0~1）",
            save_name=f"adapt_effect_bar_{domain}"
        )
        plot_paths.append(bar_path)

        # 3. 生成自适应效果雷达图
        radar_path = self.generate_radar_plot(
            data=effect_data,
            categories=["metabolic_stability", "result_consistency", "run_efficiency", "performance_rate"],
            title=f"{domain}领域自适应效果雷达图",
            labels=[f"{domain}自适应"],
            save_name=f"adapt_effect_radar_{domain}"
        )
        plot_paths.append(radar_path)

        # 4. 提取参数调整数据并生成折线图（对比调整前后）
        if "adapt_params" in adapt_result:
            adapt_params = adapt_result["adapt_params"]
            param_data = []
            param_names = []
            # 收集代谢参数
            for param, value in adapt_params.get("metabolism_params", {}).items():
                param_data.append(value)
                param_names.append(param)
            # 收集策略参数
            for param, value in adapt_params.get("strategy_params", {}).items():
                param_data.append(value)
                param_names.append(param)
            # 生成参数调整柱状图
            param_df = pd.DataFrame({
                "param_name": param_names,
                "param_value": param_data
            })
            param_path = self.generate_bar_plot(
                data=param_df,
                x_col="param_name",
                y_cols=["param_value"],
                title=f"{domain}领域自适应参数调整",
                xlabel="参数名",
                ylabel="调整后值",
                save_name=f"adapt_params_bar_{domain}"
            )
            plot_paths.append(param_path)

        print(f"✅ 自适应效果分析报告图表生成完成，共{len(plot_paths)}张图表")
        return plot_paths

    def generate_multimodal_analysis_plots(self, multimodal_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        一键生成多模态数据分析图表（适配multimodal_parser.py的输出）
        :param multimodal_data: 多模态数据字典
        :return: 生成的图表路径列表
        """
        print("\n📊 开始生成多模态数据分析图表...")
        plot_paths = []

        for modality, data in multimodal_data.items():
            if data.empty:
                continue

            # 1. 生成数据分布直方图
            hist_path = self.generate_hist_plot(
                data=data,
                cols=data.columns[:4],  # 最多显示4列
                title=f"{modality}模态数据分布直方图",
                save_name=f"multimodal_hist_{modality}"
            )
            plot_paths.append(hist_path)

            # 2. 生成数据相关性热力图
            heatmap_path = self.generate_heatmap(
                data=data,
                title=f"{modality}模态数据相关性热力图",
                save_name=f"multimodal_heatmap_{modality}"
            )
            plot_paths.append(heatmap_path)

            # 3. 生成散点图（前两列）
            if len(data.columns) >= 2:
                scatter_path = self.generate_scatter_plot(
                    data=data,
                    x_col=data.columns[0],
                    y_col=data.columns[1],
                    title=f"{modality}模态{data.columns[0]} vs {data.columns[1]}散点图",
                    save_name=f"multimodal_scatter_{modality}"
                )
                plot_paths.append(scatter_path)

        print(f"✅ 多模态数据分析图表生成完成，共{len(plot_paths)}张图表")
        return plot_paths

    def batch_generate_plots(self, plot_config: List[Dict[str, Any]]) -> List[str]:
        """
        批量生成图表（配置化，新手友好）
        :param plot_config: 绘图配置列表
        示例：
        [
            {"plot_type": "line", "data": df, "x_col": "time", "y_cols": ["value1", "value2"]},
            {"plot_type": "bar", "data": df, "x_col": "category", "y_cols": ["score1", "score2"]}
        ]
        :return: 生成的图表路径列表
        """
        print("\n🚀 开始批量生成图表...")
        plot_paths = []

        for config in plot_config:
            plot_type = config.get("plot_type")
            if plot_type not in self.supported_plots:
                print(f"⚠️  不支持的图表类型：{plot_type}，跳过")
                continue

            try:
                if plot_type == "line":
                    path = self.generate_line_plot(**config)
                elif plot_type == "bar":
                    path = self.generate_bar_plot(**config)
                elif plot_type == "heatmap":
                    path = self.generate_heatmap(**config)
                elif plot_type == "scatter":
                    path = self.generate_scatter_plot(**config)
                elif plot_type == "radar":
                    path = self.generate_radar_plot(**config)
                elif plot_type == "hist":
                    path = self.generate_hist_plot(**config)
                elif plot_type == "box":
                    path = self.generate_box_plot(**config)
                else:
                    path = ""

                if path:
                    plot_paths.append(path)
            except Exception as e:
                print(f"❌ 生成{plot_type}图表失败：{str(e)}")

        print(f"✅ 批量绘图完成，成功生成{len(plot_paths)}张图表")
        return plot_paths

# 可视化图表生成模块验证入口（一站式测试）
if __name__ == "__main__":
    # 1. 初始化图表生成器
    plotter = PlotGenerator()
    print("🚀 可视化图表生成器初始化完成！")

    # 2. 生成测试数据
    # 时序测试数据
    time_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-01-01", periods=20, freq="H"),
        "qubit_stability": np.random.rand(20)*0.9,
        "energy_consumption": np.random.rand(20)*0.8,
        "matter_output": np.random.rand(20)*0.7
    })

    # 自适应效果测试数据
    test_adapt_result = {
        "start_time": "2026-01-01 12:00:00",
        "domain_match": {"domain": "quantum", "similarity": 0.85},
        "adapt_effect": {
            "metabolic_stability": 0.88,
            "result_consistency": 0.92,
            "run_efficiency": 0.85,
            "performance_rate": 0.89,
            "comprehensive_score": 0.88
        },
        "adapt_params": {
            "metabolism_params": {"core_factor_weight": 0.88, "stability_threshold": 0.85, "cycle_speed": 0.09},
            "strategy_params": {"qubit_stability": 0.9, "atomic_frequency": 0.5, "logistics_efficiency": 0.5}
        }
    }

    # 多模态测试数据
    test_multimodal_data = {
        "table": pd.DataFrame({
            "feature1": np.random.rand(50)*0.9,
            "feature2": np.random.rand(50)*0.8,
            "feature3": np.random.rand(50)*0.7
        }),
        "text": pd.DataFrame({
            "qubit_stability": np.random.rand(10)*0.9,
            "energy_consumption": np.random.rand(10)*0.8,
            "matter_output": np.random.rand(10)*0.7
        })
    }

    # 3. 测试各类图表生成
    # 折线图
    plotter.generate_line_plot(
        data=time_data,
        x_col="timestamp",
        y_cols=["qubit_stability", "energy_consumption", "matter_output"],
        title="量子领域时序数据趋势",
        xlabel="时间",
        ylabel="数值"
    )

    # 柱状图
    plotter.generate_bar_plot(
        data=time_data.head(10),
        x_col="timestamp",
        y_cols=["qubit_stability", "energy_consumption"],
        title="量子领域前10小时指标对比",
        xlabel="时间",
        ylabel="数值"
    )

    # 热力图
    plotter.generate_heatmap(
        data=time_data[["qubit_stability", "energy_consumption", "matter_output"]],
        title="量子领域指标相关性热力图"
    )

    # 散点图
    plotter.generate_scatter_plot(
        data=time_data,
        x_col="qubit_stability",
        y_col="energy_consumption",
        title="量子稳定性 vs 能耗散点图"
    )

    # 雷达图
    plotter.generate_radar_plot(
        data=time_data.head(3),
        categories=["qubit_stability", "energy_consumption", "matter_output"],
        title="量子领域前3小时指标雷达图"
    )

    # 直方图
    plotter.generate_hist_plot(
        data=time_data,
        cols=["qubit_stability", "energy_consumption"],
        title="量子领域指标分布直方图"
    )

    # 箱线图
    time_data["hour"] = time_data["timestamp"].dt.hour // 4  # 按4小时分组
    plotter.generate_box_plot(
        data=time_data,
        x_col="hour",
        y_cols=["qubit_stability", "energy_consumption"],
        title="量子领域指标按小时分组箱线图"
    )

    # 4. 测试自适应效果报告生成
    plotter.generate_adapt_report_plots(test_adapt_result)

    # 5. 测试多模态数据分析图表生成
    plotter.generate_multimodal_analysis_plots(test_multimodal_data)

    # 6. 查看绘图历史
    print("\n📜 绘图历史汇总：")
    for idx, history in enumerate(plotter.plot_history):
        print(f"  {idx+1}. 类型：{history['plot_type']} | 标题：{history['title']} | 路径：{history['save_path']}")

    print("\n🎉 可视化图表生成模块测试完成！所有图表已保存至 ./plots")