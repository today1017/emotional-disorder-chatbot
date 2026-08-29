# -*- coding: utf-8 -*-
"""
config.py —— 全局配置中心（路径、参数、常量）
所有脚本统一引用此模块，避免硬编码分散。
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_figures")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError:
    pass  # Streamlit Cloud 只读文件系统，忽略

DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "桌面")
DATA_FILES = [
    os.path.join(DATA_DIR, "weibo.csv"),
    os.path.join(DATA_DIR, "zhuhu.csv"),
]
SOURCE_NAMES = ["微博", "知乎"]
DATA_ENCODING = "gb18030"

EXPECTED_VALUES = {
    "情绪极性": {"积极", "消极", "中性"},
    "求助信号": {"明确求助", "间接求助", "无求助意图"},
    "是否隐喻": {"是", "否"},
    "安全风险": {"是", "否"},
}

SENTIMENT_ORDER = ["消极", "中性", "积极"]
HELP_ORDER = ["明确求助", "间接求助", "无求助意图"]
METAPHOR_ORDER = ["是", "否"]
RISK_ORDER = ["是", "否"]

COLOR_MAP = {
    "消极": "#C65B5B",
    "中性": "#B8B8B8",
    "积极": "#6C9E6C",
    "是": "#4F81BD",
    "否": "#D9D9D9",
    "明确求助": "#C65B5B",
    "间接求助": "#E8A13D",
    "无求助意图": "#B8B8B8",
}
