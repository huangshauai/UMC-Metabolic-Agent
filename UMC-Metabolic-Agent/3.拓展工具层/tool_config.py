# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 配置管理工具（可视化编辑+参数校验+版本管理+一键重置）
核心逻辑：自动化管理parameters.ini/paths.ini，避免手动修改出错，适配新手友好的配置管理
设计原则：交互可视化、参数强校验、操作可回滚、配置可分享
"""
import configparser
import os
import json
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

class ConfigManager:
    """UMC智能体配置管理器（核心功能：编辑/校验/备份/回滚/导出/导入）"""
    def __init__(self, config_dir: str = "./", config_version_dir: str = "./config_versions"):
        """
        初始化配置管理器
        :param config_dir: 配置文件所在目录（默认当前目录）
        :param config_version_dir: 配置版本备份目录
        """
        # 基础路径配置
        self.config_dir = config_dir
        self.param_path = os.path.join(config_dir, "parameters.ini")
        self.path_path = os.path.join(config_dir, "paths.ini")
        self.config_version_dir = config_version_dir
        os.makedirs(config_version_dir, exist_ok=True)

        # 加载配置文件（无则生成默认配置）
        self.param_cfg = configparser.ConfigParser()
        self.path_cfg = configparser.ConfigParser()
        self._load_or_init_config()

        # 参数校验规则（核心：定义每个参数的类型、值域、说明，新手友好）
        self.param_validation_rules = {
            "BASIC": {
                "runtime_log_level": {
                    "type": "str",
                    "allowed_values": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    "default": "DEBUG",
                    "desc": "运行日志级别：DEBUG(详细)/INFO(普通)/WARNING(警告)/ERROR(仅错误)"
                },
                "cycle_speed": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "default": 0.1,
                    "desc": "代谢循环速度（0.01~1.0，越小越快，资源消耗越高）"
                },
                "data_cache_size": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "default": 100,
                    "desc": "数据缓存大小（10~1000，缓存标准化后的数据条数）"
                }
            },
            "METABOLISM": {
                "core_factor_weight": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.8,
                    "desc": "核心因子权重（0.1~1.0，权重越高，因子影响越大）"
                },
                "energy_consumption_limit": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "default": 0.9,
                    "desc": "能耗上限（0.5~1.0，超过则触发降级运行）"
                },
                "stability_threshold": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "default": 0.8,
                    "desc": "稳定性阈值（0.5~1.0，达到则认为代谢循环稳定）"
                }
            },
            "STRATEGY": {
                "qubit_stability": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.8,
                    "desc": "量子稳定性策略权重（0.1~1.0）"
                },
                "atomic_frequency": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.7,
                    "desc": "原子频率策略权重（0.1~1.0）"
                },
                "logistics_efficiency": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.75,
                    "desc": "物流效率策略权重（0.1~1.0）"
                },
                "unknown_domain": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.6,
                    "desc": "未知领域默认策略权重（0.1~1.0）"
                }
            },
            "VALIDATION": {
                "blackbox_test_threshold": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "default": 0.7,
                    "desc": "黑盒测试阈值（0.5~1.0，性能得分≥此值则达标）"
                }
            },
            "AGI_L3": {
                "goal_discovery_threshold": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.5,
                    "desc": "目标发现阈值（0.1~1.0，越低越容易发现新目标）"
                },
                "self_learning_feedback_rate": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.5,
                    "desc": "自主学习反馈率（0.1~1.0，越高优化幅度越大）"
                },
                "auto_recovery_fault_threshold": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "default": 3,
                    "desc": "自动恢复故障阈值（1~10，错误次数≥此值触发恢复）"
                }
            }
        }

        # 路径配置校验规则
        self.path_validation_rules = {
            "PATH": {
                "log_dir": {
                    "default": "./logs",
                    "desc": "日志文件存储目录"
                },
                "backup_dir": {
                    "default": "./backups",
                    "desc": "系统状态备份目录"
                },
                "processed_data_dir": {
                    "default": "./processed_data",
                    "desc": "标准化数据存储目录"
                },
                "result_dir": {
                    "default": "./results",
                    "desc": "运行结果存储目录"
                }
            }
        }

    def _load_or_init_config(self) -> None:
        """加载配置文件，无则生成默认配置（带注释，新手友好）"""
        # === 处理parameters.ini ===
        if os.path.exists(self.param_path):
            self.param_cfg.read(self.param_path, encoding="utf-8")
            # 校验并补全缺失的配置项
            self._complete_param_config()
        else:
            # 生成带注释的默认parameters.ini
            self._generate_default_param_config()

        # === 处理paths.ini ===
        if os.path.exists(self.path_path):
            self.path_cfg.read(self.path_path, encoding="utf-8")
            # 校验并补全缺失的配置项
            self._complete_path_config()
        else:
            # 生成默认paths.ini
            self._generate_default_path_config()

    def _complete_param_config(self) -> None:
        """补全parameters.ini中缺失的配置项（避免配置不全）"""
        for section, params in self.param_validation_rules.items():
            if not self.param_cfg.has_section(section):
                self.param_cfg[section] = {}
            for param, rules in params.items():
                if param not in self.param_cfg[section]:
                    self.param_cfg[section][param] = str(rules["default"])
        # 保存补全后的配置
        self._save_param_config()

    def _complete_path_config(self) -> None:
        """补全paths.ini中缺失的配置项"""
        for section, params in self.path_validation_rules.items():
            if not self.path_cfg.has_section(section):
                self.path_cfg[section] = {}
            for param, rules in params.items():
                if param not in self.path_cfg[section]:
                    self.path_cfg[section][param] = rules["default"]
        # 保存补全后的配置
        self._save_path_config()

    def _generate_default_param_config(self) -> None:
        """生成带注释的默认parameters.ini（新手友好，含参数说明）"""
        # 先构建配置内容（带注释）
        param_content = [
            "# UMC智能体核心参数配置文件",
            "# 注释：修改前建议先使用tool_config.py的backup_config备份当前配置",
            "",
            "[BASIC]",
            "# 运行日志级别：DEBUG(详细)/INFO(普通)/WARNING(警告)/ERROR(仅错误)",
            "runtime_log_level = DEBUG",
            "# 代谢循环速度（0.01~1.0，越小越快，资源消耗越高）",
            "cycle_speed = 0.1",
            "# 数据缓存大小（10~1000，缓存标准化后的数据条数）",
            "data_cache_size = 100",
            "",
            "[METABOLISM]",
            "# 核心因子权重（0.1~1.0，权重越高，因子影响越大）",
            "core_factor_weight = 0.8",
            "# 能耗上限（0.5~1.0，超过则触发降级运行）",
            "energy_consumption_limit = 0.9",
            "# 稳定性阈值（0.5~1.0，达到则认为代谢循环稳定）",
            "stability_threshold = 0.8",
            "",
            "[STRATEGY]",
            "# 量子稳定性策略权重（0.1~1.0）",
            "qubit_stability = 0.8",
            "# 原子频率策略权重（0.1~1.0）",
            "atomic_frequency = 0.7",
            "# 物流效率策略权重（0.1~1.0）",
            "logistics_efficiency = 0.75",
            "# 未知领域默认策略权重（0.1~1.0）",
            "unknown_domain = 0.6",
            "",
            "[VALIDATION]",
            "# 黑盒测试阈值（0.5~1.0，性能得分≥此值则达标）",
            "blackbox_test_threshold = 0.7",
            "",
            "[AGI_L3]",
            "# 目标发现阈值（0.1~1.0，越低越容易发现新目标）",
            "goal_discovery_threshold = 0.5",
            "# 自主学习反馈率（0.1~1.0，越高优化幅度越大）",
            "self_learning_feedback_rate = 0.5",
            "# 自动恢复故障阈值（1~10，错误次数≥此值触发恢复）",
            "auto_recovery_fault_threshold = 3",
            ""
        ]
        # 写入文件
        with open(self.param_path, "w", encoding="utf-8") as f:
            f.write("\n".join(param_content))
        # 重新加载
        self.param_cfg.read(self.param_path, encoding="utf-8")
        print(f"📄 生成默认parameters.ini：{self.param_path}")

    def _generate_default_path_config(self) -> None:
        """生成默认paths.ini"""
        self.path_cfg["PATH"] = {
            "log_dir": "./logs",
            "backup_dir": "./backups",
            "processed_data_dir": "./processed_data",
            "result_dir": "./results"
        }
        self._save_path_config()
        print(f"📄 生成默认paths.ini：{self.path_path}")

    def _save_param_config(self) -> None:
        """保存parameters.ini（格式化，便于阅读）"""
        with open(self.param_path, "w", encoding="utf-8") as f:
            self.param_cfg.write(f)

    def _save_path_config(self) -> None:
        """保存paths.ini"""
        with open(self.path_path, "w", encoding="utf-8") as f:
            self.path_cfg.write(f)

    def validate_param(self, section: str, param: str, value: Any) -> Tuple[bool, str]:
        """
        参数合法性校验（核心：避免无效参数值）
        :param section: 配置段（如AGI_L3）
        :param param: 参数名（如goal_discovery_threshold）
        :param value: 待校验的参数值
        :return: (是否合法, 错误信息/成功提示)
        """
        # 检查参数是否在校验规则中
        if section not in self.param_validation_rules or param not in self.param_validation_rules[section]:
            return False, f"参数{section}.{param}无校验规则，可能是无效参数"

        rules = self.param_validation_rules[section][param]
        try:
            # 类型转换
            if rules["type"] == "int":
                val = int(value)
            elif rules["type"] == "float":
                val = float(value)
            elif rules["type"] == "str":
                val = str(value).upper() if param == "runtime_log_level" else str(value)
            else:
                return False, f"不支持的参数类型：{rules['type']}"

            # 值域/允许值校验
            if rules["type"] in ["int", "float"]:
                if "min" in rules and val < rules["min"]:
                    return False, f"参数值{val}小于最小值{rules['min']}"
                if "max" in rules and val > rules["max"]:
                    return False, f"参数值{val}大于最大值{rules['max']}"
            elif rules["type"] == "str" and "allowed_values" in rules:
                if val not in rules["allowed_values"]:
                    return False, f"字符串参数值必须是：{rules['allowed_values']}"

            return True, f"参数{section}.{param}校验通过（值：{val}）"
        except ValueError:
            return False, f"参数{section}.{param}值{value}无法转换为{rules['type']}类型"
        except Exception as e:
            return False, f"参数校验异常：{str(e)}"

    def edit_param_interactive(self) -> None:
        """
        可视化交互编辑parameters.ini（新手友好，带提示+校验）
        操作流程：选择配置段→选择参数→输入新值→校验→保存
        """
        print("\n🎛️  开始交互编辑parameters.ini（输入q退出）")
        print("=== 可选配置段 ===")
        # 列出所有配置段（带说明）
        section_desc = {
            "BASIC": "基础运行参数",
            "METABOLISM": "代谢循环参数",
            "STRATEGY": "策略权重参数",
            "VALIDATION": "性能校验参数",
            "AGI_L3": "AGI-L3自主能力参数"
        }
        for idx, (section, desc) in enumerate(section_desc.items(), 1):
            print(f"  {idx}. {section} - {desc}")

        # 选择配置段
        while True:
            section_choice = input("\n请选择配置段（输入序号/q）：")
            if section_choice.lower() == "q":
                return
            try:
                section_idx = int(section_choice) - 1
                section_list = list(section_desc.keys())
                if 0 <= section_idx < len(section_list):
                    current_section = section_list[section_idx]
                    break
                else:
                    print(f"❌ 无效序号，请输入1~{len(section_list)}")
            except ValueError:
                print("❌ 请输入数字序号或q")

        # 列出该段的所有参数（带说明+当前值）
        print(f"\n=== {current_section}段参数列表 ===")
        params = self.param_validation_rules[current_section]
        for idx, (param, rules) in enumerate(params.items(), 1):
            current_val = self.param_cfg[current_section][param]
            print(f"  {idx}. {param} - {rules['desc']} | 当前值：{current_val}")

        # 选择参数
        while True:
            param_choice = input("\n请选择要修改的参数（输入序号/q）：")
            if param_choice.lower() == "q":
                return
            try:
                param_idx = int(param_choice) - 1
                param_list = list(params.keys())
                if 0 <= param_idx < len(param_list):
                    current_param = param_list[param_idx]
                    break
                else:
                    print(f"❌ 无效序号，请输入1~{len(param_list)}")
            except ValueError:
                print("❌ 请输入数字序号或q")

        # 输入新值（带提示）
        rules = params[current_param]
        current_val = self.param_cfg[current_section][current_param]
        print(f"\n=== 修改{current_section}.{current_param} ===")
        print(f"参数说明：{rules['desc']}")
        if rules["type"] in ["int", "float"]:
            print(f"取值范围：{rules.get('min', '无')} ~ {rules.get('max', '无')}")
        elif rules["type"] == "str" and "allowed_values" in rules:
            print(f"允许值：{rules['allowed_values']}")
        print(f"当前值：{current_val} | 默认值：{rules['default']}")

        while True:
            new_value = input("请输入新值（输入d恢复默认值/q取消）：")
            if new_value.lower() == "q":
                print("取消修改")
                return
            if new_value.lower() == "d":
                new_value = rules["default"]
                print(f"恢复为默认值：{new_value}")
                break
            # 校验新值
            is_valid, msg = self.validate_param(current_section, current_param, new_value)
            if is_valid:
                print(f"✅ {msg}")
                break
            else:
                print(f"❌ {msg}，请重新输入")

        # 备份当前配置（修改前自动备份）
        self.backup_config(backup_name=f"pre_edit_{current_section}_{current_param}")

        # 修改并保存参数
        self.param_cfg[current_section][current_param] = str(new_value)
        self._save_param_config()
        print(f"✅ 已修改{current_section}.{current_param}为：{new_value}，配置已保存")

        # 询问是否继续编辑
        continue_choice = input("是否继续编辑其他参数？(y/n)：")
        if continue_choice.lower() == "y":
            self.edit_param_interactive()

    def edit_path_interactive(self) -> None:
        """交互编辑paths.ini（带目录合法性校验）"""
        print("\n📁 开始交互编辑paths.ini（输入q退出）")
        print("=== 当前路径配置 ===")
        for param, rules in self.path_validation_rules["PATH"].items():
            current_val = self.path_cfg["PATH"][param]
            print(f"  {param} - {rules['desc']} | 当前值：{current_val}")

        # 选择要修改的路径参数
        param_list = list(self.path_validation_rules["PATH"].keys())
        while True:
            param_choice = input("\n请选择要修改的路径参数（输入序号/q）：")
            if param_choice.lower() == "q":
                return
            try:
                param_idx = int(param_choice) - 1
                if 0 <= param_idx < len(param_list):
                    current_param = param_list[param_idx]
                    break
                else:
                    print(f"❌ 无效序号，请输入1~{len(param_list)}")
            except ValueError:
                print("❌ 请输入数字序号或q")

        # 输入新路径
        rules = self.path_validation_rules["PATH"][current_param]
        current_val = self.path_cfg["PATH"][current_param]
        print(f"\n=== 修改{current_param} ===")
        print(f"参数说明：{rules['desc']}")
        print(f"当前值：{current_val} | 默认值：{rules['default']}")

        while True:
            new_path = input("请输入新路径（输入d恢复默认值/q取消）：")
            if new_path.lower() == "q":
                print("取消修改")
                return
            if new_path.lower() == "d":
                new_path = rules["default"]
                print(f"恢复为默认值：{new_path}")
                break
            # 校验路径（是否可创建）
            try:
                os.makedirs(new_path, exist_ok=True)
                print(f"✅ 路径{new_path}合法（已自动创建）")
                break
            except Exception as e:
                print(f"❌ 路径{new_path}不合法：{str(e)}，请重新输入")

        # 备份当前配置
        self.backup_config(backup_name=f"pre_edit_path_{current_param}")

        # 修改并保存
        self.path_cfg["PATH"][current_param] = new_path
        self._save_path_config()
        print(f"✅ 已修改{current_param}为：{new_path}，配置已保存")

        # 继续编辑
        continue_choice = input("是否继续编辑其他路径参数？(y/n)：")
        if continue_choice.lower() == "y":
            self.edit_path_interactive()

    def backup_config(self, backup_name: str = "auto") -> str:
        """
        备份当前配置文件（版本管理核心）
        :param backup_name: 备份名称（便于识别）
        :return: 备份目录路径
        """
        timestamp = time.strftime("%Y%m%d%H%M%S")
        backup_dir = os.path.join(self.config_version_dir, f"config_backup_{backup_name}_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

        # 复制配置文件到备份目录
        shutil.copy2(self.param_path, os.path.join(backup_dir, "parameters.ini"))
        shutil.copy2(self.path_path, os.path.join(backup_dir, "paths.ini"))

        # 生成备份说明文件
        backup_info = {
            "backup_time": timestamp,
            "backup_name": backup_name,
            "param_path": self.param_path,
            "path_path": self.path_path,
            "backup_dir": backup_dir
        }
        with open(os.path.join(backup_dir, "backup_info.json"), "w", encoding="utf-8") as f:
            json.dump(backup_info, f, ensure_ascii=False, indent=2)

        print(f"💾 配置已备份到：{backup_dir}")
        return backup_dir

    def rollback_config(self) -> None:
        """
        回滚配置到指定备份版本（可视化选择）
        """
        # 列出所有备份版本
        backup_dirs = [d for d in os.listdir(self.config_version_dir) if d.startswith("config_backup_")]
        if not backup_dirs:
            print("❌ 无配置备份版本，无法回滚")
            return

        print("\n🔙 配置回滚 - 可选备份版本：")
        backup_dirs.sort(reverse=True)  # 最新的在前
        for idx, backup_dir in enumerate(backup_dirs, 1):
            # 解析备份信息
            backup_info_path = os.path.join(self.config_version_dir, backup_dir, "backup_info.json")
            if os.path.exists(backup_info_path):
                with open(backup_info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                print(f"  {idx}. {backup_dir} - 备份时间：{info['backup_time']} | 名称：{info['backup_name']}")
            else:
                print(f"  {idx}. {backup_dir} - 无备份信息")

        # 选择备份版本
        while True:
            choice = input("\n请选择要回滚的版本序号（输入q取消）：")
            if choice.lower() == "q":
                return
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(backup_dirs):
                    target_backup = backup_dirs[choice_idx]
                    break
                else:
                    print(f"❌ 无效序号，请输入1~{len(backup_dirs)}")
            except ValueError:
                print("❌ 请输入数字序号或q")

        # 确认回滚
        confirm = input(f"确认回滚到版本{target_backup}吗？(y/n)：")
        if confirm.lower() != "y":
            print("取消回滚")
            return

        # 先备份当前配置（防止回滚错误）
        self.backup_config(backup_name=f"pre_rollback_{target_backup}")

        # 复制备份文件覆盖当前配置
        target_backup_dir = os.path.join(self.config_version_dir, target_backup)
        shutil.copy2(os.path.join(target_backup_dir, "parameters.ini"), self.param_path)
        shutil.copy2(os.path.join(target_backup_dir, "paths.ini"), self.path_path)

        # 重新加载配置
        self.param_cfg.read(self.param_path, encoding="utf-8")
        self.path_cfg.read(self.path_path, encoding="utf-8")

        print(f"✅ 已回滚配置到版本：{target_backup}")

    def reset_config_to_default(self) -> None:
        """一键重置配置到默认值（危险操作，需确认）"""
        confirm = input("\n⚠️  确认重置所有配置到默认值吗？(y/n)：")
        if confirm.lower() != "y":
            print("取消重置")
            return

        # 重置前备份
        self.backup_config(backup_name="pre_reset_to_default")

        # 重新生成默认配置
        self._generate_default_param_config()
        self._generate_default_path_config()

        # 重新加载
        self.param_cfg.read(self.param_path, encoding="utf-8")
        self.path_cfg.read(self.path_path, encoding="utf-8")

        print("✅ 已重置所有配置到默认值")

    def export_config(self, export_path: str = "./umc_config_export.json") -> str:
        """
        导出配置为JSON格式（便于分享/迁移）
        :param export_path: 导出文件路径
        :return: 导出文件路径
        """
        # 构建导出数据
        export_data = {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {section: dict(self.param_cfg[section]) for section in self.param_cfg.sections()},
            "paths": {section: dict(self.path_cfg[section]) for section in self.path_cfg.sections()}
        }

        # 写入JSON
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 配置已导出到：{export_path}")
        return export_path

    def import_config(self, import_path: str) -> None:
        """
        从JSON文件导入配置（先校验，后导入）
        :param import_path: 导入文件路径
        """
        if not os.path.exists(import_path):
            print(f"❌ 导入文件不存在：{import_path}")
            return

        # 加载导入数据
        with open(import_path, "r", encoding="utf-8") as f:
            import_data = json.load(f)

        # 校验导入数据格式
        if "parameters" not in import_data or "paths" not in import_data:
            print("❌ 导入文件格式错误，缺少parameters/paths字段")
            return

        # 导入前备份
        self.backup_config(backup_name="pre_import_config")

        # 导入parameters
        for section, params in import_data["parameters"].items():
            if not self.param_cfg.has_section(section):
                self.param_cfg[section] = {}
            for param, value in params.items():
                # 先校验参数
                if section in self.param_validation_rules and param in self.param_validation_rules[section]:
                    is_valid, msg = self.validate_param(section, param, value)
                    if is_valid:
                        self.param_cfg[section][param] = str(value)
                    else:
                        print(f"⚠️  参数{section}.{param}校验失败，跳过导入：{msg}")
                else:
                    self.param_cfg[section][param] = str(value)
                    print(f"⚠️  参数{section}.{param}无校验规则，直接导入")
        self._save_param_config()

        # 导入paths
        for section, params in import_data["paths"].items():
            if not self.path_cfg.has_section(section):
                self.path_cfg[section] = {}
            for param, value in params.items():
                self.path_cfg[section][param] = value
                # 自动创建目录
                if param in self.path_validation_rules["PATH"]:
                    os.makedirs(value, exist_ok=True)
        self._save_path_config()

        print(f"✅ 已从{import_path}导入配置（部分参数可能因校验失败未导入）")

    def show_config_summary(self) -> None:
        """显示当前配置摘要（新手友好，关键参数）"""
        print("\n📋 UMC智能体当前配置摘要")
        print("=== 核心参数 ===")
        key_params = [
            ("BASIC", "runtime_log_level", "日志级别"),
            ("BASIC", "cycle_speed", "循环速度"),
            ("AGI_L3", "goal_discovery_threshold", "目标发现阈值"),
            ("AGI_L3", "self_learning_feedback_rate", "反馈率"),
            ("VALIDATION", "blackbox_test_threshold", "测试阈值"),
            ("PATH", "log_dir", "日志目录"),
            ("PATH", "result_dir", "结果目录")
        ]
        for section, param, desc in key_params:
            try:
                if section in ["PATH"]:
                    val = self.path_cfg[section][param]
                else:
                    val = self.param_cfg[section][param]
                print(f"  {desc}：{val}")
            except:
                print(f"  {desc}：未配置")

# 配置管理工具验证入口（一站式测试所有配置管理功能）
if __name__ == "__main__":
    # 1. 初始化配置管理器
    config_manager = ConfigManager()
    print("🚀 UMC配置管理器初始化完成！")

    # 2. 显示当前配置摘要
    config_manager.show_config_summary()

    # 3. 交互编辑参数（新手核心功能）
    while True:
        print("\n=== 配置管理功能菜单 ===")
        print("1. 编辑核心参数（parameters.ini）")
        print("2. 编辑路径配置（paths.ini）")
        print("3. 备份当前配置")
        print("4. 回滚配置到备份版本")
        print("5. 重置配置到默认值")
        print("6. 导出配置为JSON")
        print("7. 从JSON导入配置")
        print("8. 显示配置摘要")
        print("9. 退出")

        choice = input("\n请选择功能（输入序号）：")
        if choice == "1":
            config_manager.edit_param_interactive()
        elif choice == "2":
            config_manager.edit_path_interactive()
        elif choice == "3":
            config_manager.backup_config(backup_name="manual_backup")
        elif choice == "4":
            config_manager.rollback_config()
        elif choice == "5":
            config_manager.reset_config_to_default()
        elif choice == "6":
            config_manager.export_config()
        elif choice == "7":
            import_path = input("请输入导入JSON文件路径：")
            config_manager.import_config(import_path)
        elif choice == "8":
            config_manager.show_config_summary()
        elif choice == "9":
            print("👋 退出配置管理器")
            break
        else:
            print("❌ 无效选择，请输入1~9")