# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具模块（停用词表、数据加载、文本处理、字体配置）
完整版：含所有工具函数 + 运行时下载字体注册 + 详细调试日志
"""
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter

import jieba
import pandas as pd

import config

BASE_STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们",
    "他们", "她们", "它们", "这", "那", "这些", "那些", "这个", "那个", "这里",
    "那里", "就", "都", "也", "又", "再", "还", "更", "最", "很", "太", "真",
    "挺", "蛮", "不", "没", "有", "没有", "是", "被", "把", "让", "给", "跟",
    "对", "从", "向", "到", "与", "及", "或", "并", "而", "但", "却", "可",
    "等", "以", "因", "为", "所以", "因为", "如果", "虽然", "但是", "然后",
    "后来", "于是", "就是", "那些", "还是", "也是", "一个", "一种",
    "一些", "一下", "一时", "一直", "一定", "一起", "一样", "一边", "自己",
    "本身", "咱们", "大家", "什么", "怎么", "怎样", "哪", "哪边", "多少", "谁",
    "现在", "今天", "明天", "昨天", "时候", "时间", "当时", "之前", "以后",
    "以前", "感觉", "觉得", "知道", "想要", "希望", "可以", "可能", "应该",
    "已经", "正在", "过", "着", "呢", "吗", "吧", "啊", "哦", "喔", "唉",
    "哎", "呀", "嘛", "哈", "哈哈", "真的", "确实", "其实", "反正", "不过",
    "居然", "竟然", "怎么", "这么", "那么", "说", "做", "看", "想", "走",
    "来", "去", "回", "好", "让", "那些", "起来", "出来", "过来", "东西", "事情",
    "问题", "地方", "里面", "外面", "上面", "下面",
}

STOPWORD_FILE_CANDIDATES = [
    os.path.join(config.DATA_DIR, "hit_stopwords.txt"),
    "stopwords.txt",
]


def load_stopwords():
    sw = set(BASE_STOPWORDS)
    for p in STOPWORD_FILE_CANDIDATES:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                sw.update(line.strip() for line in f if line.strip())
            print(f"   [停用词] 已加载外部停用词表：{p}（累计 {len(sw)} 词）")
    return sw


def read_csv_with_encoding(file_path):
    for enc in (config.DATA_ENCODING, "utf-8-sig", "gbk", "utf-8"):
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"   [读取] {os.path.basename(file_path)}  编码={enc}  共 {len(df)} 条")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"无法识别文件编码：{file_path}")


def load_and_merge():
    frames = []
    for path, src in zip(config.DATA_FILES, config.SOURCE_NAMES):
        df = read_csv_with_encoding(path)
        df["数据来源"] = src
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"   [合并] 合并后总条数：{len(data)}")
    return data


def clean_text(text):
    t = str(text)
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"@\S+", "", t)
    t = re.sub(r"#(.+?)#", r"\1", t)
    t = re.sub(r"[^\u4e00-\u9fff]", "", t)
    return t


def tokenize(text, stopwords=None):
    if stopwords is None:
        stopwords = load_stopwords()
    t = clean_text(text)
    tokens = []
    for w in jieba.lcut(t):
        w = w.strip()
        if len(w) < 2 or w in stopwords:
            continue
        tokens.append(w)
    return tokens


def find_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def setup_matplotlib():
    """
    1. 优先下载并注册 Noto Sans SC (TTF)
    2. 失败回退 fc-list 探测
    3. 再失败用常见预装字体名
    4. 终极兜底 DejaVu Sans
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import subprocess
    import tempfile
    import urllib.request

    print("[FONT] === setup_matplotlib START ===")
    print(f"[FONT] Matplotlib version: {matplotlib.__version__}")
    print(f"[FONT] Font cache dir: {matplotlib.get_cachedir()}")

    # 1. 优先：下载并注册 Noto Sans SC (TTF)
    def _download_and_register_font():
        font_url_ttf = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/TTF/SimplifiedChinese/NotoSansSC-Regular.ttf"
        font_url_otf = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"

        for url in (font_url_ttf, font_url_otf):
            try:
                print(f"[FONT] Downloading font from {url} ...")
                with urllib.request.urlopen(url, timeout=15) as resp:
                    font_data = resp.read()
                print(f"[FONT] Downloaded {len(font_data)} bytes")

                with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
                    tmp.write(font_data)
                    tmp_path = tmp.name
                print(f"[FONT] Saved to {tmp_path}, size={os.path.getsize(tmp_path)}")

                fm.fontManager.addfont(tmp_path)
                prop = fm.FontProperties(fname=tmp_path)
                family = prop.get_name()
                print(f"[FONT] ✅ Registered font family: {family}")
                return family
            except Exception as e:
                print(f"[FONT] ⚠️ Failed to download/register from {url}: {e}")
                continue
        return None

    print("[FONT] === setup_matplotlib START ===")
    print(f"[FONT] Matplotlib version: {matplotlib.__version__}")
    print(f"[FONT] Font cache dir: {matplotlib.get_cachedir()}")

    # 1. 优先下载注册
    family = None
    try:
        family = None
        font_url_ttf = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/TTF/SimplifiedChinese/NotoSansSC-Regular.ttf"
        font_url_otf = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"
        for url in (font_url_ttf, font_url_otf):
            try:
                print(f"[FONT] Downloading font from {url} ...")
                with urllib.request.urlopen(url, timeout=15) as resp:
                    font_data = resp.read()
                print(f"[FONT] Downloaded {len(font_data)} bytes")

                with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
                    tmp.write(font_data)
                    tmp_path = tmp.name
                print(f"[FONT] Saved to {tmp_path}, size={os.path.getsize(tmp_path)}")

                fm.fontManager.addfont(tmp_path)
                prop = fm.FontProperties(fname=tmp_path)
                family = prop.get_name()
                print(f"[FONT] ✅ Registered font family: {family}")
                break
            except Exception as e:
                print(f"[FONT] ⚠️ Failed to download/register from {url}: {e}")
                continue
    except Exception as e:
        print(f"[FONT] Download block failed: {e}")

    if family:
        plt.rcParams["font.family"] = family
        print(f"[FONT] ✅ Using downloaded font: {family}")
        plt.rcParams["axes.unicode_minus"] = False
        print("[FONT] === setup_matplotlib END (downloaded) ===")
        return plt

    # 2. 回退：fc-list 探测
    try:
        out = subprocess.check_output(
            ["fc-list", ":lang=zh", "family"],
            stderr=subprocess.DEVNULL, text=True, timeout=3
        )
        for line in out.splitlines():
            fam = line.split(":")[0].strip()
            if fam:
                plt.rcParams["font.family"] = fam
                print(f"[FONT] fc-list detected: {fam}")
                plt.rcParams["axes.unicode_minus"] = False
                print("[FONT] === setup_matplotlib END (fc-list) ===")
                return plt
    except Exception as e:
        print(f"[FONT] fc-list failed: {e}")

    # 3. 常见预装字体名
    for fname in ["Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "WenQuanYi Zen Hei"]:
        try:
            plt.rcParams["font.family"] = fname
            print(f"[FONT] Fallback to preset: {fname}")
            plt.rcParams["axes.unicode_minus"] = False
            print("[FONT] === setup_matplotlib END (preset) ===")
            return plt
        except Exception as e:
            print(f"[FONT] Preset {fname} failed: {e}")

    # 4. 终极兜底
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    print("[FONT] ⚠️ Using DejaVu Sans (no Chinese support)")
    print("[FONT] === setup_matplotlib END (DejaVu) ===")
    return plt
