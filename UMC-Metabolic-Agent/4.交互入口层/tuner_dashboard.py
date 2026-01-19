# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 调优仪表盘模块（Web可视化+实时监控+交互式调优）
核心逻辑：基于Streamlit构建Web调优仪表盘，提供可视化参数调优、实时监控、结果分析能力
设计原则：可视化、交互式、实时性、新手友好，让调优过程直观可控
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
import warnings
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 导入核心模块（兼容未安装情况）
try:
    from universal_cmd import UniversalCmd
    from result_analysis import ResultAnalyzer
    from report_generator import ReportGenerator
    CORE_MODULES_LOADED = True
except ImportError as e:
    st.warning(f"核心模块导入失败：{e}\n部分功能将受限，建议确保核心文件在当前目录")
    CORE_MODULES_LOADED = False

warnings.filterwarnings("ignore")

class TunerDashboard:
    """调优仪表盘（核心：Web可视化、实时监控、交互式调优）"""
    def __init__(self):
        """初始化调优仪表盘"""
        # 基础配置
        self.base_dir = os.getcwd()
        self.tuner_dir = "./umc_tuner"
        self.data_dir = f"{self.tuner_dir}/data"
        self.config_dir = f"{self.tuner_dir}/configs"
        self.history_dir = f"{self.tuner_dir}/history"
        self.report_dir = f"{self.tuner_dir}/reports"
        
        # 创建目录
        for dir_path in [self.tuner_dir, self.data_dir, self.config_dir, self.history_dir, self.report_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化核心模块
        self.cmd = UniversalCmd() if CORE_MODULES_LOADED else None
        self.analyzer = ResultAnalyzer(output_dir=f"{self.tuner_dir}/analysis") if CORE_MODULES_LOADED else None
        self.report_generator = ReportGenerator(output_dir=self.report_dir) if CORE_MODULES_LOADED else None
        
        # 默认调优参数
        self.default_params = {
            "domain": "general",
            "adapt_iterations": 50,
            "learning_rate": 0.01,
            "core_factor_weight": 0.8,
            "stability_threshold": 0.75,
            "cycle_speed": 0.05,
            "target_metric": "metabolic_efficiency",
            "early_stop_patience": 10,
            "batch_size": 32
        }
        
        # 调优状态
        self.tuner_status = {
            "is_running": False,
            "current_iter": 0,
            "total_iter": 0,
            "current_score": 0.0,
            "best_score": 0.0,
            "best_params": {},
            "progress": 0.0,
            "start_time": "",
            "elapsed_time": 0.0
        }
        
        # 加载历史记录
        self.history_records = self._load_history_records()
        
        # Streamlit页面配置
        st.set_page_config(
            page_title="UMC智能体调优仪表盘",
            page_icon="🔧",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def run(self):
        """运行调优仪表盘（核心入口）"""
        # 页面标题
        st.title("🔧 UMC-Metabolic-Agent 调优仪表盘")
        st.divider()
        
        # 侧边栏导航
        with st.sidebar:
            st.header("📋 导航菜单")
            page = st.radio(
                "选择功能页面",
                [
                    "仪表盘主页",
                    "参数配置",
                    "调优监控",
                    "历史记录",
                    "结果分析",
                    "报告导出"
                ],
                index=0
            )
            
            st.divider()
            st.header("⚙️ 基础设置")
            self.default_params["domain"] = st.selectbox(
                "目标领域",
                ["general", "quantum", "biology", "chemistry", "finance"],
                index=0
            )
            
            # 快速加载数据
            st.subheader("📥 数据加载")
            uploaded_file = st.file_uploader("上传调优数据（CSV/Excel）", type=["csv", "xlsx"], key="main_upload")
            if uploaded_file:
                self._save_uploaded_data(uploaded_file)
                st.success(f"✅ 数据已保存：{uploaded_file.name}")
        
        # 页面路由
        if page == "仪表盘主页":
            self._render_dashboard_home()
        elif page == "参数配置":
            self._render_param_config()
        elif page == "调优监控":
            self._render_tuner_monitor()
        elif page == "历史记录":
            self._render_history_records()
        elif page == "结果分析":
            self._render_result_analysis()
        elif page == "报告导出":
            self._render_report_export()

    # ------------------------------ 页面渲染逻辑 ------------------------------
    def _render_dashboard_home(self):
        """渲染仪表盘主页"""
        st.subheader("📊 调优概览")
        
        # 分栏展示核心信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 历史最优得分", f"{self._get_best_history_score():.3f}")
        with col2:
            st.metric("🔢 调优记录数", len(self.history_records))
        with col3:
            st.metric("⚡ 当前调优状态", "运行中" if self.tuner_status["is_running"] else "空闲")
        with col4:
            st.metric("📂 可用调优数据", len(self._list_data_files()))
        
        st.divider()
        
        # 快速开始调优
        st.subheader("🚀 快速开始调优")
        with st.form("quick_tuner_form"):
            data_file = st.selectbox("选择调优数据", self._list_data_files())
            domain = st.selectbox("目标领域", ["general", "quantum", "biology", "chemistry"], index=0)
            adapt_iter = st.slider("调优迭代次数", 10, 200, 50, 10)
            submit_btn = st.form_submit_button("开始调优", type="primary")
            
            if submit_btn:
                if not data_file:
                    st.error("❌ 请先选择调优数据")
                else:
                    # 更新参数并开始调优
                    self.default_params.update({
                        "domain": domain,
                        "adapt_iterations": adapt_iter
                    })
                    self._start_tuner(f"{self.data_dir}/{data_file}")
        
        st.divider()
        
        # 最近调优记录
        st.subheader("📜 最近调优记录")
        if self.history_records:
            recent_records = self.history_records[-5:]  # 最近5条
            recent_df = pd.DataFrame(recent_records)
            recent_df = recent_df[["timestamp", "domain", "iterations", "best_score", "status"]]
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("暂无调优记录，开始第一次调优吧！")

    def _render_param_config(self):
        """渲染参数配置页面"""
        st.subheader("⚙️ 调优参数配置")
        st.info("📝 配置智能体调优参数，所有参数将保存为配置文件供后续使用")
        
        # 参数配置表单
        with st.form("param_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # 基础参数
                st.subheader("基础参数")
                domain = st.selectbox("目标领域", ["general", "quantum", "biology", "chemistry", "finance"], 
                                     index=["general", "quantum", "biology", "chemistry", "finance"].index(self.default_params["domain"]))
                adapt_iter = st.number_input("调优迭代次数", 10, 500, self.default_params["adapt_iterations"], 10)
                learning_rate = st.slider("学习率", 0.001, 0.1, self.default_params["learning_rate"], 0.001, format="%.3f")
                target_metric = st.selectbox("优化目标指标", 
                                            ["metabolic_efficiency", "domain_adapt_score", "matter_output"],
                                            index=["metabolic_efficiency", "domain_adapt_score", "matter_output"].index(self.default_params["target_metric"]))
            
            with col2:
                # 高级参数
                st.subheader("高级参数")
                core_factor = st.slider("核心因子权重", 0.1, 1.0, self.default_params["core_factor_weight"], 0.05, format="%.2f")
                stability_thresh = st.slider("稳定性阈值", 0.5, 1.0, self.default_params["stability_threshold"], 0.05, format="%.2f")
                cycle_speed = st.slider("循环速度", 0.01, 0.2, self.default_params["cycle_speed"], 0.01, format="%.2f")
                early_stop = st.number_input("早停耐心值", 0, 50, self.default_params["early_stop_patience"], 5)
            
            # 配置文件名称
            config_name = st.text_input("配置文件名称", f"tuner_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # 提交按钮
            save_btn = st.form_submit_button("保存配置", type="primary")
            apply_btn = st.form_submit_button("应用并开始调优")
        
        # 处理表单提交
        if save_btn:
            # 保存配置
            config_data = {
                "config_name": config_name,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "params": {
                    "domain": domain,
                    "adapt_iterations": adapt_iter,
                    "learning_rate": learning_rate,
                    "core_factor_weight": core_factor,
                    "stability_threshold": stability_thresh,
                    "cycle_speed": cycle_speed,
                    "target_metric": target_metric,
                    "early_stop_patience": early_stop,
                    "batch_size": self.default_params["batch_size"]
                }
            }
            
            config_path = f"{self.config_dir}/{config_name}.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            # 更新默认参数
            self.default_params.update(config_data["params"])
            
            st.success(f"✅ 配置已保存：{config_path}")
        
        if apply_btn:
            # 应用参数并开始调优
            self.default_params.update({
                "domain": domain,
                "adapt_iterations": adapt_iter,
                "learning_rate": learning_rate,
                "core_factor_weight": core_factor,
                "stability_threshold": stability_thresh,
                "cycle_speed": cycle_speed,
                "target_metric": target_metric,
                "early_stop_patience": early_stop
            })
            
            # 检查数据文件
            data_files = self._list_data_files()
            if not data_files:
                st.error("❌ 请先上传调优数据文件")
            else:
                # 开始调优
                self._start_tuner(f"{self.data_dir}/{data_files[0]}")
                st.success("🚀 调优已开始，前往【调优监控】页面查看实时进度")

    def _render_tuner_monitor(self):
        """渲染调优监控页面"""
        st.subheader("📈 调优实时监控")
        
        # 调优状态展示
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("当前迭代", f"{self.tuner_status['current_iter']}/{self.tuner_status['total_iter']}")
        with col2:
            st.metric("当前得分", f"{self.tuner_status['current_score']:.3f}")
        with col3:
            st.metric("最优得分", f"{self.tuner_status['best_score']:.3f}")
        with col4:
            st.metric("调优进度", f"{self.tuner_status['progress']:.1f}%")
        
        st.divider()
        
        # 进度条
        st.progress(self.tuner_status["progress"] / 100)
        
        # 调优控制
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            if st.button("开始调优", type="primary", disabled=self.tuner_status["is_running"]):
                data_files = self._list_data_files()
                if not data_files:
                    st.error("❌ 请先上传调优数据文件")
                else:
                    self._start_tuner(f"{self.data_dir}/{data_files[0]}")
        
        with col_ctrl2:
            if st.button("暂停调优", disabled=not self.tuner_status["is_running"]):
                self.tuner_status["is_running"] = False
                st.warning("⚠️ 调优已暂停")
        
        with col_ctrl3:
            if st.button("终止调优", disabled=not self.tuner_status["is_running"]):
                self.tuner_status["is_running"] = False
                self.tuner_status["progress"] = 100.0
                st.warning("⚠️ 调优已终止")
        
        st.divider()
        
        # 实时可视化
        if self.tuner_status["is_running"] or self.tuner_status["current_iter"] > 0:
            # 生成监控数据
            iter_list = list(range(1, self.tuner_status["current_iter"] + 1)) if self.tuner_status["current_iter"] > 0 else [0]
            score_list = [np.random.uniform(0.6, 0.95) for _ in iter_list] if self.tuner_status["current_iter"] > 0 else [0]
            
            if score_list:
                self.tuner_status["current_score"] = score_list[-1]
                self.tuner_status["best_score"] = max(score_list)
            
            # 绘制得分趋势图
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=iter_list,
                y=score_list,
                mode="lines+markers",
                name="调优得分",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4)
            ))
            
            # 添加最优得分线
            fig.add_hline(
                y=self.tuner_status["best_score"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"最优得分: {self.tuner_status['best_score']:.3f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title="调优得分趋势",
                xaxis_title="迭代次数",
                yaxis_title="调优得分",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 实时参数展示
            st.subheader("🔧 当前最优参数")
            if self.tuner_status["best_params"]:
                params_df = pd.DataFrame({
                    "参数名称": list(self.tuner_status["best_params"].keys()),
                    "参数值": list(self.tuner_status["best_params"].values())
                })
                st.dataframe(params_df, use_container_width=True)
            else:
                # 展示默认参数
                params_df = pd.DataFrame({
                    "参数名称": list(self.default_params.keys()),
                    "参数值": list(self.default_params.values())
                })
                st.dataframe(params_df, use_container_width=True)
            
            # 自动刷新
            if self.tuner_status["is_running"]:
                st.empty()
                time.sleep(1)
                st.rerun()
        else:
            st.info("📌 调优未运行，点击【开始调优】按钮启动调优过程")

    def _render_history_records(self):
        """渲染历史记录页面"""
        st.subheader("📜 调优历史记录")
        
        if not self.history_records:
            st.info("暂无调优历史记录")
            return
        
        # 历史记录筛选
        col1, col2 = st.columns(2)
        with col1:
            domain_filter = st.selectbox("按领域筛选", ["全部"] + list(set([r["domain"] for r in self.history_records])))
        with col2:
            status_filter = st.selectbox("按状态筛选", ["全部"] + list(set([r["status"] for r in self.history_records])))
        
        # 应用筛选
        filtered_records = self.history_records
        if domain_filter != "全部":
            filtered_records = [r for r in filtered_records if r["domain"] == domain_filter]
        if status_filter != "全部":
            filtered_records = [r for r in filtered_records if r["status"] == status_filter]
        
        # 展示历史记录
        history_df = pd.DataFrame(filtered_records)
        history_df = history_df[["timestamp", "domain", "iterations", "best_score", "status", "duration"]]
        st.dataframe(history_df, use_container_width=True)
        
        # 选择记录查看详情
        st.subheader("🔍 记录详情")
        record_idx = st.selectbox("选择记录", range(len(filtered_records)), format_func=lambda x: f"记录{x+1} - {filtered_records[x]['timestamp']}")
        
        if record_idx is not None and len(filtered_records) > 0:
            record = filtered_records[record_idx]
            
            # 详情展示
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 基础信息")
                st.write(f"**时间戳**: {record['timestamp']}")
                st.write(f"**领域**: {record['domain']}")
                st.write(f"**迭代次数**: {record['iterations']}")
                st.write(f"**最优得分**: {record['best_score']:.3f}")
                st.write(f"**状态**: {record['status']}")
                st.write(f"**耗时**: {record['duration']:.2f}秒")
            
            with col2:
                st.write("### 最优参数")
                best_params = record.get("best_params", {})
                if best_params:
                    for param, value in best_params.items():
                        st.write(f"**{param}**: {value}")
                else:
                    st.write("暂无参数记录")
            
            # 绘制历史趋势
            if "score_history" in record and record["score_history"]:
                score_history = record["score_history"]
                iter_list = list(range(1, len(score_history)+1))
                
                fig = px.line(
                    x=iter_list,
                    y=score_history,
                    title=f"调优得分趋势（{record['timestamp']}）",
                    labels={"x": "迭代次数", "y": "调优得分"},
                    height=400
                )
                fig.add_hline(y=record["best_score"], line_dash="dash", line_color="red", annotation_text=f"最优: {record['best_score']:.3f}")
                st.plotly_chart(fig, use_container_width=True)
            
            # 操作按钮
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("重新运行此配置"):
                    self.default_params.update(record.get("best_params", self.default_params))
                    data_files = self._list_data_files()
                    if data_files:
                        self._start_tuner(f"{self.data_dir}/{data_files[0]}")
                        st.success("🚀 调优已开始")
            
            with col_btn2:
                if st.button("导出配置"):
                    config_data = {
                        "config_name": f"history_config_{record['timestamp'].replace(' ', '_').replace(':', '')}",
                        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": f"history_record_{record['timestamp']}",
                        "params": record.get("best_params", self.default_params)
                    }
                    config_path = f"{self.config_dir}/{config_data['config_name']}.json"
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, ensure_ascii=False, indent=2)
                    st.success(f"✅ 配置已导出：{config_path}")
            
            with col_btn3:
                if st.button("删除记录"):
                    record_path = f"{self.history_dir}/{record['record_id']}.json"
                    if os.path.exists(record_path):
                        os.remove(record_path)
                    self.history_records = self._load_history_records()
                    st.success("✅ 记录已删除")
                    st.rerun()

    def _render_result_analysis(self):
        """渲染结果分析页面"""
        st.subheader("📊 调优结果分析")
        
        # 选择分析数据
        history_files = [f for f in os.listdir(self.history_dir) if f.endswith(".json")]
        if not history_files:
            st.info("暂无调优数据可分析")
            return
        
        selected_file = st.selectbox("选择调优记录", history_files)
        if selected_file:
            # 加载数据
            with open(f"{self.history_dir}/{selected_file}", "r", encoding="utf-8") as f:
                tuner_data = json.load(f)
            
            st.divider()
            
            # 核心指标分析
            st.subheader("🎯 核心指标分析")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("最优得分", f"{tuner_data['best_score']:.3f}")
            with col2:
                st.metric("平均得分", f"{np.mean(tuner_data['score_history']):.3f}")
            with col3:
                st.metric("得分标准差", f"{np.std(tuner_data['score_history']):.3f}")
            with col4:
                st.metric("收敛迭代", f"{self._get_convergence_iter(tuner_data['score_history'])}")
            
            st.divider()
            
            # 多维度分析图表
            tab1, tab2, tab3 = st.tabs(["得分分布", "参数敏感性", "收敛分析"])
            
            with tab1:
                # 得分分布直方图
                fig = px.histogram(
                    x=tuner_data["score_history"],
                    nbins=20,
                    title="调优得分分布",
                    labels={"x": "得分", "y": "频次"},
                    color_discrete_sequence=["#1f77b4"]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # 参数敏感性分析（模拟）
                params = tuner_data.get("best_params", self.default_params)
                param_names = list(params.keys())[:6]  # 取前6个参数
                sensitivity_scores = [np.random.uniform(0.1, 0.9) for _ in param_names]
                
                fig = px.bar(
                    x=param_names,
                    y=sensitivity_scores,
                    title="参数敏感性分析",
                    labels={"x": "参数名称", "y": "敏感性得分"},
                    color=sensitivity_scores,
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # 收敛分析
                score_history = tuner_data["score_history"]
                iter_list = list(range(1, len(score_history)+1))
                
                # 计算移动平均
                window_size = max(1, int(len(score_history) * 0.1))
                if window_size < len(score_history):
                    moving_avg = pd.Series(score_history).rolling(window=window_size).mean().tolist()
                else:
                    moving_avg = score_history
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=iter_list,
                    y=score_history,
                    mode="lines",
                    name="原始得分",
                    line=dict(color="#1f77b4", width=1, opacity=0.5)
                ))
                fig.add_trace(go.Scatter(
                    x=iter_list,
                    y=moving_avg,
                    mode="lines",
                    name=f"移动平均（窗口{window_size}）",
                    line=dict(color="red", width=2)
                ))
                fig.update_layout(
                    title="收敛趋势分析",
                    xaxis_title="迭代次数",
                    yaxis_title="调优得分",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # 分析报告
            if st.button("生成详细分析报告", type="primary"):
                if self.analyzer:
                    # 执行分析
                    analysis_result = {
                        "tuner_data": tuner_data,
                        "core_metrics": {
                            "best_score": tuner_data["best_score"],
                            "avg_score": np.mean(tuner_data["score_history"]),
                            "std_score": np.std(tuner_data["score_history"]),
                            "convergence_iter": self._get_convergence_iter(tuner_data["score_history"]),
                            "stability_score": 1 - np.std(tuner_data["score_history"][-10:]) if len(score_history)>=10 else 0.9
                        },
                        "param_analysis": {
                            "most_impact_param": param_names[np.argmax(sensitivity_scores)],
                            "param_sensitivity": dict(zip(param_names, sensitivity_scores))
                        },
                        "conclusion": "调优过程收敛良好，参数配置合理，建议保留当前最优参数" if tuner_data["best_score"] >= 0.8 else "调优得分偏低，建议增加迭代次数或调整学习率"
                    }
                    
                    # 保存分析结果
                    analysis_path = f"{self.tuner_dir}/analysis/{selected_file.replace('.json', '_analysis.json')}"
                    with open(analysis_path, "w", encoding="utf-8") as f:
                        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ 分析报告已生成：{analysis_path}")
                    
                    # 展示分析结论
                    st.subheader("📝 分析结论")
                    st.write(analysis_result["conclusion"])
                else:
                    st.error("❌ ResultAnalyzer模块未加载，无法生成分析报告")

    def _render_report_export(self):
        """渲染报告导出页面"""
        st.subheader("📄 调优报告导出")
        
        # 选择要导出的调优记录
        history_files = [f for f in os.listdir(self.history_dir) if f.endswith(".json")]
        if not history_files:
            st.info("暂无调优记录可导出报告")
            return
        
        selected_files = st.multiselect("选择调优记录（可多选）", history_files)
        
        # 报告配置
        col1, col2 = st.columns(2)
        with col1:
            report_format = st.multiselect("报告格式", ["md", "html", "pdf"], default=["md", "html"])
        with col2:
            report_name = st.text_input("报告名称", f"umc_tuner_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 导出选项
        include_plots = st.checkbox("包含可视化图表", value=True)
        include_analysis = st.checkbox("包含深度分析", value=True)
        include_compare = st.checkbox("包含多记录对比", value=len(selected_files)>1)
        
        # 导出按钮
        if st.button("生成并导出报告", type="primary"):
            if not selected_files:
                st.error("❌ 请至少选择一条调优记录")
            else:
                if self.report_generator:
                    # 加载选中的记录
                    selected_records = []
                    for file in selected_files:
                        with open(f"{self.history_dir}/{file}", "r", encoding="utf-8") as f:
                            selected_records.append(json.load(f))
                    
                    # 构建报告数据
                    report_data = {
                        "report_name": report_name,
                        "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "record_count": len(selected_records),
                        "records": selected_records,
                        "include_plots": include_plots,
                        "include_analysis": include_analysis,
                        "include_compare": include_compare
                    }
                    
                    # 生成报告
                    report_paths = self.report_generator.generate_comprehensive_report(
                        report_data,
                        report_name=report_name,
                        format_list=report_format,
                        with_plots=include_plots
                    )
                    
                    # 展示报告路径
                    st.success("✅ 调优报告已生成！")
                    for fmt, path in report_paths.items():
                        st.write(f"📄 {fmt.upper()}格式：{path}")
                        
                        # 提供下载按钮
                        with open(path, "rb") as f:
                            st.download_button(
                                label=f"下载{fmt.upper()}报告",
                                data=f,
                                file_name=os.path.basename(path),
                                mime="text/markdown" if fmt=="md" else "text/html" if fmt=="html" else "application/pdf"
                            )
                else:
                    st.error("❌ ReportGenerator模块未加载，无法生成报告")

    # ------------------------------ 辅助方法 ------------------------------
    def _start_tuner(self, data_path: str):
        """启动调优过程"""
        # 初始化调优状态
        self.tuner_status.update({
            "is_running": True,
            "current_iter": 0,
            "total_iter": self.default_params["adapt_iterations"],
            "current_score": 0.0,
            "best_score": 0.0,
            "best_params": self.default_params.copy(),
            "progress": 0.0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": 0.0
        })
        
        # 模拟调优过程（后台运行）
        def tuner_worker():
            score_history = []
            start_time = time.time()
            
            for i in range(self.default_params["adapt_iterations"]):
                if not self.tuner_status["is_running"]:
                    break
                
                # 模拟调优得分
                current_score = np.random.uniform(0.6, 0.95)
                score_history.append(current_score)
                
                # 更新状态
                self.tuner_status.update({
                    "current_iter": i + 1,
                    "current_score": current_score,
                    "best_score": max(score_history) if score_history else 0.0,
                    "progress": ((i + 1) / self.default_params["adapt_iterations"]) * 100,
                    "elapsed_time": time.time() - start_time
                })
                
                # 模拟参数更新
                if current_score == self.tuner_status["best_score"]:
                    self.tuner_status["best_params"] = {
                        **self.default_params,
                        "learning_rate": self.default_params["learning_rate"] * (0.99 ** i),
                        "core_factor_weight": np.clip(self.default_params["core_factor_weight"] + np.random.uniform(-0.01, 0.01), 0.1, 1.0)
                    }
                
                time.sleep(0.1)  # 模拟调优耗时
            
            # 调优结束
            self.tuner_status["is_running"] = False
            self.tuner_status["progress"] = 100.0
            
            # 保存调优记录
            self._save_tuner_record({
                "record_id": f"tuner_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "domain": self.default_params["domain"],
                "iterations": self.default_params["adapt_iterations"],
                "best_score": self.tuner_status["best_score"],
                "score_history": score_history,
                "best_params": self.tuner_status["best_params"],
                "status": "completed" if self.tuner_status["current_iter"] >= self.default_params["adapt_iterations"] else "interrupted",
                "duration": self.tuner_status["elapsed_time"]
            })
            
            # 刷新历史记录
            self.history_records = self._load_history_records()
        
        # 在新线程中运行调优
        import threading
        tuner_thread = threading.Thread(target=tuner_worker, daemon=True)
        tuner_thread.start()

    def _save_tuner_record(self, record_data: Dict[str, Any]):
        """保存调优记录"""
        record_path = f"{self.history_dir}/{record_data['record_id']}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)

    def _load_history_records(self) -> List[Dict[str, Any]]:
        """加载历史调优记录"""
        history_records = []
        history_files = [f for f in os.listdir(self.history_dir) if f.endswith(".json")]
        
        for file in history_files:
            try:
                with open(f"{self.history_dir}/{file}", "r", encoding="utf-8") as f:
                    record = json.load(f)
                    history_records.append(record)
            except Exception as e:
                st.warning(f"加载历史记录失败：{file} - {e}")
        
        # 按时间戳排序
        history_records.sort(key=lambda x: x["timestamp"], reverse=True)
        return history_records

    def _save_uploaded_data(self, uploaded_file):
        """保存上传的数据文件"""
        save_path = f"{self.data_dir}/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return save_path

    def _list_data_files(self) -> List[str]:
        """列出可用的数据文件"""
        return [f for f in os.listdir(self.data_dir) if f.endswith((".csv", ".xlsx"))]

    def _get_best_history_score(self) -> float:
        """获取历史最优得分"""
        if not self.history_records:
            return 0.0
        return max([r.get("best_score", 0.0) for r in self.history_records])

    def _get_convergence_iter(self, score_history: List[float]) -> int:
        """计算收敛迭代次数"""
        if len(score_history) < 10:
            return len(score_history)
        
        # 找到得分稳定的迭代点
        threshold = 0.01  # 变化阈值
        for i in range(len(score_history)-10, len(score_history)):
            recent_scores = score_history[i-10:i]
            if max(recent_scores) - min(recent_scores) < threshold:
                return i
        return len(score_history)

# 仪表盘入口
def main():
    """主函数：启动调优仪表盘"""
    dashboard = TunerDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()