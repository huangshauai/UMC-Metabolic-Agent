# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 信号解析基础模块（AGI L3通用交互底层）
核心逻辑：标准化解析文本类输入信号，映射到代谢因子（信息/能量/物质）维度，为多模态解析提供统一基础
设计原则：文本清洗通用化、因子映射可扩展、与多模态模块无缝衔接，无冗余解析逻辑
"""
import configparser
import re
import json
import time
from typing import Dict, Any, Optional
import pandas as pd

class SignalInterpreter:
    """信号解析基础类（文本信号统一解析，多模态解析的父类）"""
    def __init__(self):
        # 1. 加载配置（复用参数配置，主要读取日志级别）
        self.param_cfg = self._load_config("parameters.ini")
        # 2. 定义文本-代谢因子映射规则（AGI L3通用映射，可扩展）
        self.factor_mapping_rules = {
            # 关键词 → 代谢因子维度 + 权重
            "信息": ("information", 1.0),
            "特征": ("information", 0.9),
            "目标": ("information", 0.8),
            "能量": ("energy", 1.0),
            "效率": ("energy", 0.9),
            "消耗": ("energy", 0.8),
            "物质": ("matter", 1.0),
            "数据量": ("matter", 0.9),
            "产出": ("matter", 0.8),
            "稳定": ("stability", 0.9),
            "优化": ("optimize", 0.9)
        }
        # 3. 初始化解析状态（支撑可解释性）
        self.parse_history = []
        self.current_parse_result = None

    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """统一加载参数配置（主要读取日志级别，控制解析日志详细度）"""
        cfg = configparser.ConfigParser()
        cfg.read(config_file, encoding="utf-8")
        return cfg

    def _clean_text(self, text: str) -> str:
        """通用文本清洗（去特殊字符、去空格、小写化，保证解析一致性）"""
        # 1. 去除特殊字符（保留中文、数字、字母）
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
        # 2. 去除多余空格，首尾去空格
        text = re.sub(r"\s+", " ", text).strip()
        # 3. 英文小写化（避免大小写影响解析）
        text = text.lower()
        return text

    def _map_text_to_factor(self, clean_text: str) -> Dict[str, float]:
        """核心：将清洗后的文本映射到代谢因子维度，输出因子权重"""
        factor_scores = {
            "information": 0.0,
            "energy": 0.0,
            "matter": 0.0,
            "stability": 0.0,
            "optimize": 0.0
        }

        # 遍历映射规则，匹配关键词并累加权重
        for keyword, (factor, weight) in self.factor_mapping_rules.items():
            if keyword in clean_text:
                factor_scores[factor] += weight

        # 归一化因子得分（0~1），避免权重过大
        total_score = sum(factor_scores.values())
        if total_score == 0:
            total_score = 1.0  # 避免除零

        normalized_scores = {k: v / total_score for k, v in factor_scores.items()}
        return normalized_scores

    def parse(self, input_text: str, return_df: bool = False) -> Dict[str, Any] | pd.DataFrame:
        """
        AGI L3核心：文本信号统一解析入口（被多模态模块/核心智能体调用）
        :param input_text: 原始文本信号（用户输入/语音转文本/日志文本）
        :param return_df: 是否返回DataFrame格式（适配代谢循环的数据输入）
        :return: 结构化解析结果（字典/DF），包含清洗文本、因子映射、解析时间
        """
        # 1. 文本清洗
        clean_text = self._clean_text(input_text)
        
        # 2. 映射到代谢因子维度
        factor_scores = self._map_text_to_factor(clean_text)
        
        # 3. 构造解析结果（结构化，支撑可解释性）
        parse_result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_text": input_text,
            "clean_text": clean_text,
            "factor_scores": factor_scores,
            "parse_status": "success"
        }

        # 4. 更新解析状态（支撑日志/可解释性）
        self.current_parse_result = parse_result
        self.parse_history.append(parse_result)

        # 5. 控制日志详细度（基于parameters.ini的日志级别）
        log_level = self.param_cfg["BASIC"]["runtime_log_level"]
        if log_level == "DEBUG":
            self._log_parse_result(parse_result)

        # 6. 支持返回DataFrame格式（适配代谢循环的数据输入要求）
        if return_df:
            df = pd.DataFrame([factor_scores])
            df["timestamp"] = parse_result["timestamp"]
            df["clean_text"] = clean_text
            return df
        return parse_result

    def _log_parse_result(self, parse_result: Dict[str, Any]) -> None:
        """记录解析日志（仅DEBUG级别输出，避免冗余日志）"""
        log_msg = f"[PARSE DEBUG] {parse_result['timestamp']} - 清洗后文本：{parse_result['clean_text']} - 因子得分：{parse_result['factor_scores']}"
        print(log_msg)  # 核心智能体集成时可写入日志文件

    def get_parse_history(self, last_n: int = 10) -> list[Dict[str, Any]]:
        """获取最近n条解析历史（支撑可解释性/可视化）"""
        return self.parse_history[-last_n:] if self.parse_history else []

# 信号解析基础验证入口（无冗余，快速测试）
if __name__ == "__main__":
    # 1. 初始化信号解析模块
    interpreter = SignalInterpreter()
    print("UMC信号解析基础模块初始化完成")

    # 2. 测试文本解析（模拟用户输入）
    test_text = "提升量子比特的能量效率，优化物质产出和信息稳定性！"
    print(f"\n=== 原始输入文本 ===")
    print(test_text)

    # 3. 执行解析（字典格式）
    parse_result = interpreter.parse(test_text)
    print(f"\n=== 结构化解析结果 ===")
    print(json.dumps(parse_result, ensure_ascii=False, indent=2))

    # 4. 执行解析（DataFrame格式，适配代谢循环）
    parse_df = interpreter.parse(test_text, return_df=True)
    print(f"\n=== DataFrame格式解析结果 ===")
    print(parse_df)

    # 5. 查看解析历史
    print(f"\n=== 解析历史 ===")
    print(interpreter.get_parse_history())