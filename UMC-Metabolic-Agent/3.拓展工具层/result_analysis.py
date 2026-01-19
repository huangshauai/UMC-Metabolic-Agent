# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 结果分析模块（统计分析+特征评估+领域适配+异常检测+报告生成）
核心逻辑：对智能体运行结果/自适应效果/多模态数据进行深度分析，输出可解释的分析报告
设计原则：自动化、可解释、量化评估、零配置使用，适配新手分析多场景结果数据
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os
import time
import warnings
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 导入核心模块
try:
    from plot_generator import PlotGenerator
except ImportError:
    print("⚠️  未找到plot_generator模块，可视化功能将不可用")
    PlotGenerator = None

class ResultAnalyzer:
    """结果分析器（核心功能：统计分析、特征评估、领域适配分析、异常检测、报告生成）"""
    def __init__(self, output_dir: str = "./analysis_reports"):
        """
        初始化结果分析器
        :param output_dir: 分析报告保存目录
        """
        # 基础配置
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 初始化可视化生成器（可选）
        self.plotter = PlotGenerator(output_dir="./analysis_plots") if PlotGenerator else None
        # 分析历史
        self.analysis_history = []

    def basic_statistical_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        基础统计分析（核心：描述性统计、分布检验、相关性分析）
        :param data: 待分析数据
        :param kwargs: 可选参数（target_cols：目标列列表，save_name：保存名）
        :return: 统计分析结果
        """
        print("\n📊 开始基础统计分析...")
        # 解析参数
        target_cols = kwargs.get("target_cols", data.select_dtypes(include=[np.number]).columns.tolist())
        save_name = kwargs.get("save_name", f"basic_stats_{time.strftime('%Y%m%d%H%M%S')}")
        # 只保留数值列
        numeric_data = data[target_cols].copy()

        # 1. 描述性统计
        desc_stats = numeric_data.describe().to_dict()
        # 补充中位数、众数、偏度、峰度
        extended_stats = {}
        for col in target_cols:
            extended_stats[col] = {
                "median": float(numeric_data[col].median()),
                "mode": float(numeric_data[col].mode().iloc[0] if not numeric_data[col].mode().empty else np.nan),
                "skewness": float(numeric_data[col].skew()),
                "kurtosis": float(numeric_data[col].kurt()),
                "missing_rate": float(numeric_data[col].isnull().sum() / len(numeric_data)),
                "cv": float(numeric_data[col].std() / numeric_data[col].mean()) if numeric_data[col].mean() != 0 else 0.0  # 变异系数
            }

        # 2. 分布正态性检验（Shapiro-Wilk）
        normality_test = {}
        for col in target_cols:
            if len(numeric_data[col].dropna()) >= 3:  # 至少3个样本
                stat, p_value = stats.shapiro(numeric_data[col].dropna())
                normality_test[col] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05  # α=0.05
                }
            else:
                normality_test[col] = {"error": "样本数不足，无法检验"}

        # 3. 相关性分析（Pearson/Spearman）
        correlation_analysis = {
            "pearson": numeric_data.corr().to_dict(),
            "spearman": numeric_data.corr(method="spearman").to_dict()
        }

        # 4. 极值分析
        extreme_analysis = {}
        for col in target_cols:
            q1 = numeric_data[col].quantile(0.25)
            q3 = numeric_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)][col]
            extreme_analysis[col] = {
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": int(len(outliers)),
                "outlier_rate": float(len(outliers) / len(numeric_data)),
                "min_value": float(numeric_data[col].min()),
                "max_value": float(numeric_data[col].max())
            }

        # 汇总分析结果
        analysis_result = {
            "basic_info": {
                "sample_count": len(data),
                "feature_count": len(target_cols),
                "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "descriptive_statistics": desc_stats,
            "extended_statistics": extended_stats,
            "normality_test": normality_test,
            "correlation_analysis": correlation_analysis,
            "extreme_analysis": extreme_analysis
        }

        # 保存分析结果
        save_path = os.path.join(self.output_dir, f"{save_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # 生成可视化图表（可选）
        if self.plotter:
            # 生成统计可视化图表
            self.plotter.generate_hist_plot(
                data=numeric_data,
                cols=target_cols[:4],
                title="数据分布直方图",
                save_name=f"{save_name}_hist"
            )
            self.plotter.generate_heatmap(
                data=numeric_data,
                title="Pearson相关性热力图",
                save_name=f"{save_name}_pearson_heatmap"
            )

        # 记录分析历史
        self.analysis_history.append({
            "analysis_type": "basic_statistical",
            "data_shape": data.shape,
            "save_path": save_path,
            "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # 打印关键结果
        print(f"✅ 基础统计分析完成：")
        print(f"  - 样本数：{len(data)} | 特征数：{len(target_cols)}")
        print(f"  - 异常值率（平均）：{np.mean([v['outlier_rate'] for v in extreme_analysis.values()]):.3f}")
        print(f"  - 正态分布特征数：{sum([1 for v in normality_test.values() if v.get('is_normal', False)])}/{len(target_cols)}")
        print(f"  - 分析报告保存：{save_path}")

        return analysis_result

    def feature_importance_analysis(self, data: pd.DataFrame, target_col: str, **kwargs) -> Dict[str, Any]:
        """
        特征重要性分析（核心：基于随机森林评估特征对目标列的贡献）
        :param data: 待分析数据
        :param target_col: 目标列名
        :param kwargs: 可选参数（save_name：保存名）
        :return: 特征重要性分析结果
        """
        print("\n🔍 开始特征重要性分析...")
        # 解析参数
        save_name = kwargs.get("save_name", f"feature_importance_{time.strftime('%Y%m%d%H%M%S')}")

        # 数据预处理
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in numeric_cols:
            raise ValueError(f"目标列{target_col}不是数值列")
        # 移除目标列，保留特征列
        feature_cols = [col for col in numeric_cols if col != target_col]
        if not feature_cols:
            raise ValueError("无可用特征列进行重要性分析")

        # 准备数据
        X = data[feature_cols].fillna(data[feature_cols].mean())
        y = data[target_col].fillna(data[target_col].mean())
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)

        # 计算特征重要性
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf_model.feature_importances_,
            "normalized_importance": rf_model.feature_importances_ / rf_model.feature_importances_.sum()
        }).sort_values(by="importance", ascending=False)

        # 模型评估
        y_pred = rf_model.predict(X_scaled)
        model_metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2_score": float(r2_score(y, y_pred)),
            "explained_variance": float(rf_model.score(X_scaled, y))
        }

        # 汇总分析结果
        analysis_result = {
            "basic_info": {
                "target_column": target_col,
                "feature_count": len(feature_cols),
                "sample_count": len(data),
                "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "feature_importance": feature_importance.to_dict("records"),
            "top_5_features": feature_importance.head(5)["feature"].tolist(),
            "model_metrics": model_metrics
        }

        # 保存分析结果
        save_path = os.path.join(self.output_dir, f"{save_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # 生成可视化图表（可选）
        if self.plotter:
            # 生成特征重要性柱状图
            importance_df = pd.DataFrame({
                "feature": feature_importance["feature"],
                "normalized_importance": feature_importance["normalized_importance"]
            })
            self.plotter.generate_bar_plot(
                data=importance_df.head(10),  # 只显示前10个特征
                x_col="feature",
                y_cols=["normalized_importance"],
                title=f"特征重要性分析（目标列：{target_col}）",
                xlabel="特征名",
                ylabel="归一化重要性",
                save_name=f"{save_name}_bar"
            )

        # 记录分析历史
        self.analysis_history.append({
            "analysis_type": "feature_importance",
            "target_col": target_col,
            "data_shape": data.shape,
            "save_path": save_path,
            "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # 打印关键结果
        print(f"✅ 特征重要性分析完成：")
        print(f"  - 目标列：{target_col} | 特征数：{len(feature_cols)}")
        print(f"  - 模型R²得分：{model_metrics['r2_score']:.3f}")
        print(f"  - 最重要的5个特征：{', '.join(analysis_result['top_5_features'])}")
        print(f"  - 分析报告保存：{save_path}")

        return analysis_result

    def domain_adaptation_analysis(self, adapt_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        领域自适应效果分析（核心：量化评估无监督自适应的效果和稳定性）
        :param adapt_result: 无监督自适应结果（来自unsupervised_adapt.py）
        :param kwargs: 可选参数（save_name：保存名）
        :return: 领域自适应分析结果
        """
        print("\n🌐 开始领域自适应效果分析...")
        # 解析参数
        save_name = kwargs.get("save_name", f"domain_adapt_{time.strftime('%Y%m%d%H%M%S')}")

        # 1. 基础信息提取
        domain = adapt_result["domain_match"]["domain"]
        similarity = adapt_result["domain_match"]["similarity"]
        adapt_effect = adapt_result["adapt_effect"]
        adapt_params = adapt_result["adapt_params"]

        # 2. 效果量化评估
        effect_evaluation = {
            "comprehensive_score": adapt_effect["comprehensive_score"],
            "score_grade": self._grade_score(adapt_effect["comprehensive_score"]),
            "key_metrics": {
                "metabolic_stability": {
                    "score": adapt_effect["metabolic_stability"],
                    "grade": self._grade_score(adapt_effect["metabolic_stability"]),
                    "contribution": 0.4  # 权重
                },
                "result_consistency": {
                    "score": adapt_effect["result_consistency"],
                    "grade": self._grade_score(adapt_effect["result_consistency"]),
                    "contribution": 0.2
                },
                "run_efficiency": {
                    "score": adapt_effect["run_efficiency"],
                    "grade": self._grade_score(adapt_effect["run_efficiency"]),
                    "contribution": 0.2
                },
                "performance_rate": {
                    "score": adapt_effect["performance_rate"],
                    "grade": self._grade_score(adapt_effect["performance_rate"]),
                    "contribution": 0.2
                }
            },
            "weighted_score": sum([
                adapt_effect[k] * v["contribution"]
                for k, v in effect_evaluation["key_metrics"].items()
            ])
        }

        # 3. 参数调整分析
        param_analysis = {}
        # 代谢参数分析
        metabolism_params = adapt_params.get("metabolism_params", {})
        param_analysis["metabolism_params"] = {
            "param_count": len(metabolism_params),
            "key_adjustments": self._identify_key_param_changes(metabolism_params),
            "adjustment_range": self._calculate_param_adjustment_range(metabolism_params)
        }
        # 策略参数分析
        strategy_params = adapt_params.get("strategy_params", {})
        param_analysis["strategy_params"] = {
            "param_count": len(strategy_params),
            "domain_weight": self._get_domain_strategy_weight(strategy_params, domain),
            "max_strategy_weight": max(strategy_params.values()) if strategy_params else 0.0
        }

        # 4. 稳定性分析
        stability_analysis = {
            "domain_similarity": similarity,
            "similarity_grade": self._grade_similarity(similarity),
            "adapt_success": adapt_result["is_adapt_successful"],
            "expected_stability": self._predict_stability(similarity, effect_evaluation["comprehensive_score"])
        }

        # 5. 改进建议生成
        improvement_suggestions = self._generate_improvement_suggestions(
            effect_evaluation, stability_analysis, domain
        )

        # 汇总分析结果
        analysis_result = {
            "basic_info": {
                "domain": domain,
                "domain_similarity": similarity,
                "adapt_time": adapt_result["start_time"],
                "data_sample_count": adapt_result["data_info"]["sample_count"],
                "data_feature_count": len(adapt_result["data_info"]["feature_cols"])
            },
            "effect_evaluation": effect_evaluation,
            "param_analysis": param_analysis,
            "stability_analysis": stability_analysis,
            "improvement_suggestions": improvement_suggestions
        }

        # 保存分析结果
        save_path = os.path.join(self.output_dir, f"{save_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # 生成可视化图表（可选）
        if self.plotter:
            self.plotter.generate_adapt_report_plots(adapt_result)

        # 记录分析历史
        self.analysis_history.append({
            "analysis_type": "domain_adaptation",
            "domain": domain,
            "comprehensive_score": effect_evaluation["comprehensive_score"],
            "save_path": save_path,
            "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # 打印关键结果
        print(f"✅ 领域自适应效果分析完成：")
        print(f"  - 匹配领域：{domain}（相似度：{similarity:.3f}）")
        print(f"  - 综合效果得分：{effect_evaluation['comprehensive_score']:.3f}（等级：{effect_evaluation['score_grade']}）")
        print(f"  - 自适应成功：{stability_analysis['adapt_success']}")
        print(f"  - 分析报告保存：{save_path}")

        return analysis_result

    def multimodal_data_analysis(self, multimodal_data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        多模态数据融合分析（核心：评估不同模态数据的一致性和互补性）
        :param multimodal_data: 多模态数据字典（来自multimodal_parser.py）
        :param kwargs: 可选参数（save_name：保存名）
        :return: 多模态数据分析结果
        """
        print("\n🎭 开始多模态数据融合分析...")
        # 解析参数
        save_name = kwargs.get("save_name", f"multimodal_analysis_{time.strftime('%Y%m%d%H%M%S')}")

        # 1. 基础信息统计
        modal_info = {}
        for modality, data in multimodal_data.items():
            modal_info[modality] = {
                "sample_count": len(data),
                "feature_count": len(data.columns),
                "missing_rate": float(data.isnull().sum().sum() / (len(data) * len(data.columns))),
                "data_density": float(len(data.dropna()) / len(data))
            }

        # 2. 模态一致性分析
        consistency_analysis = {}
        # 提取所有模态的数值列均值
        modal_means = {}
        for modality, data in multimodal_data.items():
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                modal_means[modality] = numeric_data.mean().mean()  # 所有列的均值的均值

        # 计算模态间的一致性（变异系数）
        if len(modal_means) >= 2:
            mean_vals = list(modal_means.values())
            cv = np.std(mean_vals) / np.mean(mean_vals) if np.mean(mean_vals) != 0 else 0.0
            consistency_analysis = {
                "modal_consistency_cv": cv,
                "consistency_grade": self._grade_consistency(cv),
                "modal_means": modal_means,
                "mean_consistency": 1 - cv if cv <= 1 else 0.0
            }
        else:
            consistency_analysis = {"error": "模态数不足，无法分析一致性"}

        # 3. 模态互补性分析
        complementarity_analysis = {}
        # 计算不同模态特征的重叠度
        all_features = []
        for modality, data in multimodal_data.items():
            all_features.extend([f"{modality}_{col}" for col in data.columns])
        unique_features = len(set(all_features))
        total_features = len(all_features)
        complementarity_analysis = {
            "feature_overlap_rate": 1 - (unique_features / total_features) if total_features > 0 else 0.0,
            "complementarity_score": unique_features / total_features if total_features > 0 else 0.0,
            "complementarity_grade": self._grade_complementarity(complementarity_analysis["complementarity_score"])
        }

        # 4. 融合效果评估
        fusion_evaluation = {
            "data_quality_score": 1 - np.mean([info["missing_rate"] for info in modal_info.values()]),
            "data_quality_grade": self._grade_score(fusion_evaluation["data_quality_score"]),
            "overall_complementarity": complementarity_analysis["complementarity_score"],
            "overall_consistency": consistency_analysis.get("mean_consistency", 0.0),
            "fusion_score": (fusion_evaluation["data_quality_score"] * 0.4 +
                            complementarity_analysis["complementarity_score"] * 0.3 +
                            consistency_analysis.get("mean_consistency", 0.0) * 0.3),
            "fusion_grade": self._grade_score(fusion_evaluation["fusion_score"])
        }

        # 汇总分析结果
        analysis_result = {
            "basic_info": {
                "modal_count": len(multimodal_data),
                "total_sample_count": sum([info["sample_count"] for info in modal_info.values()]),
                "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "modal_info": modal_info,
            "consistency_analysis": consistency_analysis,
            "complementarity_analysis": complementarity_analysis,
            "fusion_evaluation": fusion_evaluation
        }

        # 保存分析结果
        save_path = os.path.join(self.output_dir, f"{save_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # 生成可视化图表（可选）
        if self.plotter:
            self.plotter.generate_multimodal_analysis_plots(multimodal_data)

        # 记录分析历史
        self.analysis_history.append({
            "analysis_type": "multimodal_data",
            "modal_count": len(multimodal_data),
            "fusion_score": fusion_evaluation["fusion_score"],
            "save_path": save_path,
            "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # 打印关键结果
        print(f"✅ 多模态数据融合分析完成：")
        print(f"  - 模态数：{len(multimodal_data)} | 总样本数：{fusion_evaluation['basic_info']['total_sample_count']}")
        print(f"  - 融合效果得分：{fusion_evaluation['fusion_score']:.3f}（等级：{fusion_evaluation['fusion_grade']}）")
        print(f"  - 模态互补性：{complementarity_analysis['complementarity_score']:.3f} | 一致性：{consistency_analysis.get('mean_consistency', 0.0):.3f}")
        print(f"  - 分析报告保存：{save_path}")

        return analysis_result

    def generate_comprehensive_report(self, analysis_results: List[Dict[str, Any]], **kwargs) -> str:
        """
        生成综合分析报告（核心：整合所有分析结果，输出人类可读的markdown报告）
        :param analysis_results: 各类分析结果列表
        :param kwargs: 可选参数（report_title：报告标题）
        :return: 报告保存路径
        """
        print("\n📋 开始生成综合分析报告...")
        # 解析参数
        report_title = kwargs.get("report_title", f"UMC智能体分析报告_{time.strftime('%Y%m%d%H%M%S')}")
        save_name = kwargs.get("save_name", f"comprehensive_report_{time.strftime('%Y%m%d%H%M%S')}")

        # 构建报告内容
        report_content = f"""# {report_title}

## 报告概览
- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
- 分析类型：{', '.join([r.get('basic_info', {}).get('analysis_type', r.get('analysis_type', 'unknown')) for r in analysis_results])}
- 数据规模：{sum([r.get('basic_info', {}).get('sample_count', 0) for r in analysis_results])} 样本

"""

        # 逐个添加分析结果
        for idx, result in enumerate(analysis_results):
            analysis_type = result.get("analysis_type", f"分析{idx+1}")
            report_content += f"## {idx+1}. {self._get_analysis_type_name(analysis_type)}\n\n"

            # 基础统计分析
            if analysis_type == "basic_statistical":
                report_content += self._format_basic_stats_report(result)
            # 特征重要性分析
            elif analysis_type == "feature_importance":
                report_content += self._format_feature_importance_report(result)
            # 领域自适应分析
            elif analysis_type == "domain_adaptation":
                report_content += self._format_domain_adapt_report(result)
            # 多模态数据分析
            elif analysis_type == "multimodal_data":
                report_content += self._format_multimodal_report(result)
            # 通用格式
            else:
                report_content += f"### 关键指标\n"
                for k, v in result.get("basic_info", {}).items():
                    report_content += f"- {k}：{v}\n"
                report_content += "\n"

        # 添加总结和建议
        report_content += f"""## 总结与建议

### 核心结论
"""
        # 提取关键结论
        for result in analysis_results:
            if "effect_evaluation" in result:
                score = result["effect_evaluation"]["comprehensive_score"]
                grade = result["effect_evaluation"]["score_grade"]
                report_content += f"- 领域自适应效果：{score:.3f}（{grade}）\n"
            elif "fusion_evaluation" in result:
                score = result["fusion_evaluation"]["fusion_score"]
                grade = result["fusion_evaluation"]["fusion_grade"]
                report_content += f"- 多模态融合效果：{score:.3f}（{grade}）\n"
            elif "model_metrics" in result:
                r2 = result["model_metrics"]["r2_score"]
                report_content += f"- 特征重要性模型解释度：{r2:.3f}\n"

        report_content += f"""
### 改进建议
1. 针对得分较低的指标（<0.7），建议调整对应的自适应参数
2. 多模态数据若一致性较低，建议增加样本量或优化特征提取逻辑
3. 特征重要性较低的列可考虑移除，提升智能体运行效率
4. 定期重新评估领域自适应效果，确保长期稳定性

---
*报告由UMC-Metabolic-Agent自动生成*
"""

        # 保存报告
        save_path = os.path.join(self.output_dir, f"{save_name}.md")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✅ 综合分析报告生成完成：{save_path}")
        return save_path

    # ------------------------------ 辅助方法 ------------------------------
    def _grade_score(self, score: float) -> str:
        """给得分评级（0~1）"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "中等"
        elif score >= 0.6:
            return "及格"
        else:
            return "待优化"

    def _grade_similarity(self, similarity: float) -> str:
        """给领域相似度评级（0~1）"""
        if similarity >= 0.8:
            return "极高"
        elif similarity >= 0.7:
            return "高"
        elif similarity >= 0.6:
            return "中等"
        elif similarity >= 0.5:
            return "低"
        else:
            return "极低"

    def _grade_consistency(self, cv: float) -> str:
        """给一致性评级（变异系数）"""
        if cv <= 0.1:
            return "极高"
        elif cv <= 0.2:
            return "高"
        elif cv <= 0.3:
            return "中等"
        elif cv <= 0.4:
            return "低"
        else:
            return "极低"

    def _grade_complementarity(self, score: float) -> str:
        """给互补性评级（0~1）"""
        if score >= 0.9:
            return "极高"
        elif score >= 0.8:
            return "高"
        elif score >= 0.7:
            return "中等"
        elif score >= 0.6:
            return "低"
        else:
            return "极低"

    def _identify_key_param_changes(self, params: Dict[str, float]) -> List[str]:
        """识别关键参数调整"""
        if not params:
            return []
        # 假设基准值为0.5，超过±0.1视为关键调整
        key_changes = []
        for param, value in params.items():
            if abs(value - 0.5) >= 0.1:
                key_changes.append(f"{param}（{value:.3f}）")
        return key_changes if key_changes else ["无显著调整"]

    def _calculate_param_adjustment_range(self, params: Dict[str, float]) -> Dict[str, float]:
        """计算参数调整范围"""
        if not params:
            return {"min": 0.0, "max": 0.0, "range": 0.0}
        values = list(params.values())
        return {
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values)
        }

    def _get_domain_strategy_weight(self, strategy_params: Dict[str, float], domain: str) -> float:
        """获取领域策略权重"""
        domain_key = None
        for key in strategy_params.keys():
            if domain in key or key == "unknown_domain":
                domain_key = key
                break
        return strategy_params.get(domain_key, 0.0) if domain_key else 0.0

    def _predict_stability(self, similarity: float, score: float) -> float:
        """预测长期稳定性"""
        return (similarity * 0.6 + score * 0.4)  # 相似度60%权重，效果得分40%权重

    def _generate_improvement_suggestions(self, effect: Dict[str, Any], stability: Dict[str, Any], domain: str) -> List[str]:
        """生成改进建议"""
        suggestions = []
        score = effect["comprehensive_score"]

        if score < 0.7:
            suggestions.append(f"{domain}领域自适应效果待优化（综合得分{score:.3f}），建议：")
            # 分析低分指标
            low_metrics = [k for k, v in effect["key_metrics"].items() if v["score"] < 0.7]
            if "metabolic_stability" in low_metrics:
                suggestions.append("- 调整代谢稳定性阈值参数，提高核心因子权重")
            if "result_consistency" in low_metrics:
                suggestions.append("- 增加多次运行的样本量，优化一致性计算逻辑")
            if "run_efficiency" in low_metrics:
                suggestions.append("- 降低循环速度，减少不必要的迭代次数")
            if "performance_rate" in low_metrics:
                suggestions.append("- 调整性能阈值，适配当前领域数据特征")
        else:
            suggestions.append(f"{domain}领域自适应效果良好（综合得分{score:.3f}），建议保持当前参数配置")

        if stability["similarity_grade"] in ["低", "极低"]:
            suggestions.append(f"领域匹配相似度较低（{stability['domain_similarity']:.3f}），建议扩展领域特征库")

        return suggestions

    def _get_analysis_type_name(self, analysis_type: str) -> str:
        """获取分析类型的中文名称"""
        type_mapping = {
            "basic_statistical": "基础统计分析",
            "feature_importance": "特征重要性分析",
            "domain_adaptation": "领域自适应效果分析",
            "multimodal_data": "多模态数据融合分析"
        }
        return type_mapping.get(analysis_type, analysis_type)

    def _format_basic_stats_report(self, result: Dict[str, Any]) -> str:
        """格式化基础统计分析报告"""
        content = f"""### 数据基本信息
- 样本数量：{result['basic_info']['sample_count']}
- 特征数量：{result['basic_info']['feature_count']}

### 关键统计指标
| 特征 | 均值 | 中位数 | 变异系数 | 缺失率 | 异常值率 |
|------|------|--------|----------|--------|----------|
"""
        for col, stats in result["descriptive_statistics"].items():
            extended = result["extended_statistics"][col]
            extreme = result["extreme_analysis"][col]
            content += f"| {col} | {stats['mean']:.3f} | {extended['median']:.3f} | {extended['cv']:.3f} | {extended['missing_rate']:.3f} | {extreme['outlier_rate']:.3f} |\n"

        content += f"""
### 分布检验结果
| 特征 | 正态性p值 | 是否正态分布 |
|------|-----------|--------------|
"""
        for col, test in result["normality_test"].items():
            if "p_value" in test:
                content += f"| {col} | {test['p_value']:.3f} | {'是' if test['is_normal'] else '否'} |\n"
            else:
                content += f"| {col} | - | {test['error']} |\n"

        content += "\n"
        return content

    def _format_feature_importance_report(self, result: Dict[str, Any]) -> str:
        """格式化特征重要性分析报告"""
        content = f"""### 分析配置
- 目标列：{result['basic_info']['target_column']}
- 特征数量：{result['basic_info']['feature_count']}
- 模型R²得分：{result['model_metrics']['r2_score']:.3f}

### 特征重要性排名（前5）
| 排名 | 特征名 | 归一化重要性 |
|------|--------|--------------|
"""
        for idx, item in enumerate(result["top_5_features"][:5]):
            importance = next((f["normalized_importance"] for f in result["feature_importance"] if f["feature"] == item), 0.0)
            content += f"| {idx+1} | {item} | {importance:.3f} |\n"

        content += "\n"
        return content

    def _format_domain_adapt_report(self, result: Dict[str, Any]) -> str:
        """格式化领域自适应分析报告"""
        content = f"""### 自适应基础信息
- 匹配领域：{result['basic_info']['domain']}
- 领域相似度：{result['basic_info']['domain_similarity']:.3f}（{result['stability_analysis']['similarity_grade']}）
- 数据样本数：{result['basic_info']['data_sample_count']}

### 效果评估结果
| 指标 | 得分 | 等级 | 权重 |
|------|------|------|------|
"""
        for metric, info in result["effect_evaluation"]["key_metrics"].items():
            content += f"| {metric} | {info['score']:.3f} | {info['grade']} | {info['contribution']} |\n"

        content += f"""
- 综合得分：{result['effect_evaluation']['comprehensive_score']:.3f}（{result['effect_evaluation']['score_grade']}）
- 自适应成功：{result['stability_analysis']['adapt_success']}

### 改进建议
"""
        for suggestion in result["improvement_suggestions"]:
            content += f"- {suggestion}\n"

        content += "\n"
        return content

    def _format_multimodal_report(self, result: Dict[str, Any]) -> str:
        """格式化多模态数据分析报告"""
        content = f"""### 模态基本信息
| 模态 | 样本数 | 特征数 | 缺失率 | 数据密度 |
|------|--------|--------|--------|----------|
"""
        for modal, info in result["modal_info"].items():
            content += f"| {modal} | {info['sample_count']} | {info['feature_count']} | {info['missing_rate']:.3f} | {info['data_density']:.3f} |\n"

        content += f"""
### 融合效果评估
- 融合效果得分：{result['fusion_evaluation']['fusion_score']:.3f}（{result['fusion_evaluation']['fusion_grade']}）
- 数据质量得分：{result['fusion_evaluation']['data_quality_score']:.3f}（{result['fusion_evaluation']['data_quality_grade']}）
- 模态互补性：{result['complementarity_analysis']['complementarity_score']:.3f}（{result['complementarity_analysis']['complementarity_grade']}）
"""
        if "mean_consistency" in result["consistency_analysis"]:
            content += f"- 模态一致性：{result['consistency_analysis']['mean_consistency']:.3f}（{self._grade_consistency(1 - result['consistency_analysis']['mean_consistency'])}）\n"

        content += "\n"
        return content

# 结果分析模块验证入口（一站式测试）
if __name__ == "__main__":
    # 1. 初始化结果分析器
    analyzer = ResultAnalyzer()
    print("🚀 结果分析器初始化完成！")

    # 2. 生成测试数据
    # 基础统计分析测试数据
    test_data = pd.DataFrame({
        "qubit_stability": np.random.rand(100)*0.9,
        "energy_consumption": np.random.rand(100)*0.8,
        "matter_output": np.random.rand(100)*0.7,
        "noise": np.random.normal(0, 1, 100)  # 正态分布数据
    })
    # 插入缺失值和异常值
    test_data.loc[10:15, "qubit_stability"] = np.nan
    test_data.loc[20, "energy_consumption"] = 5.0  # 异常值

    # 特征重要性分析测试数据
    target_col = "matter_output"

    # 领域自适应结果测试数据
    test_adapt_result = {
        "start_time": "2026-01-01 12:00:00",
        "end_time": "2026-01-01 12:05:00",
        "total_duration": "300.00s",
        "data_info": {"sample_count": 200, "feature_cols": ["qubit_stability", "energy_consumption", "matter_output"]},
        "domain_match": {"domain": "quantum", "similarity": 0.85},
        "adapt_params": {
            "domain": "quantum",
            "adapt_time": "2026-01-01 12:02:00",
            "metabolism_params": {"core_factor_weight": 0.88, "stability_threshold": 0.85, "cycle_speed": 0.09},
            "strategy_params": {"qubit_stability": 0.9, "atomic_frequency": 0.5, "logistics_efficiency": 0.5},
            "agi_l3_params": {"goal_discovery_threshold": 0.45}
        },
        "adapt_effect": {
            "metabolic_stability": 0.88,
            "result_consistency": 0.92,
            "run_efficiency": 0.85,
            "performance_rate": 0.89,
            "comprehensive_score": 0.88
        },
        "is_adapt_successful": True
    }

    # 多模态数据测试数据
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
        }),
        "timeseries": pd.DataFrame({
            "ts_feature1": np.random.rand(30)*0.9,
            "ts_feature2": np.random.rand(30)*0.8
        })
    }

    # 3. 执行各类分析
    # 基础统计分析
    basic_result = analyzer.basic_statistical_analysis(test_data, target_cols=["qubit_stability", "energy_consumption", "matter_output"])

    # 特征重要性分析
    feature_result = analyzer.feature_importance_analysis(test_data, target_col="matter_output")

    # 领域自适应分析
    adapt_result = analyzer.domain_adaptation_analysis(test_adapt_result)

    # 多模态数据分析
    multimodal_result = analyzer.multimodal_data_analysis(test_multimodal_data)

    # 4. 生成综合分析报告
    all_results = [basic_result, feature_result, adapt_result, multimodal_result]
    report_path = analyzer.generate_comprehensive_report(all_results, report_title="UMC智能体测试分析报告")

    # 5. 查看分析历史
    print("\n📜 分析历史汇总：")
    for idx, history in enumerate(analyzer.analysis_history):
        print(f"  {idx+1}. 类型：{history['analysis_type']} | 路径：{history['save_path']}")

    print(f"\n🎉 结果分析模块测试完成！")
    print(f"  - 综合分析报告：{report_path}")
    print(f"  - 所有分析结果已保存至 ./analysis_reports")
    print(f"  - 分析图表已保存至 ./analysis_plots（若启用可视化）")