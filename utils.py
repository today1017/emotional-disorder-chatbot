# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具模块（含全景字体调试）
部署后看日志，我能告诉你确切用哪个字体
"""
import os
import re
import sys
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
    "等", "以", "因", "为", "所以", "因为", "如果", "虽然", "那些", "还是",
    "也是", "一个", "那些", "一些", "一下", "一时", "一直", "一定", "一起",
    "一样", "一边", "自己", "本身", "咱们", "大家", "什么", "怎么", "怎样",
    "哪", "哪边", "多少", "谁", "现在", "今天", "明天", "昨天", "时候", "时间",
    "当时", "之前", "以后", "以前", "感觉", "觉得", "知道", "想要", "希望",
    "可以", "可能", "应该", "已经", "正在", "过", "着", "呢", "吗", "吧", "啊",
    "哦", "喔", "唉", "哎", "呀", "嘛", "哈", "哈哈", "真的", "确实", "其实",
    "反正", "不过", "居然", "竟然", "怎么", "那些", "那么", "说", "做", "看",
    "想", "走", "来", "去", "回", "好", "让", "那些", "起来", "出来", "过来",
    "东西", "事情", "问题", "地方", "里面", "外面", "上面", "下面",
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
    全景调试版：打印容器里所有字体信息，最后用一个能跑通的配置。
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import os
    import glob

    print("[FONT] ===== FULL FONT DEBUG START =====")
    print(f"[FONT] Matplotlib version: {matplotlib.__version__}")
    print(f"[FONT] Font cache dir: {matplotlib.get_cachedir()}")

    # 1. 列出常见字体目录下的所有字体文件
    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/home/adminuser/.fonts",
        "/home/adminuser/.local/share/fonts",
    ]
    print("[FONT] === Scanning font directories ===")
    for d in font_dirs:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "**", "*.tt*"), recursive=True)
            files += glob.glob(os.path.join(d, "**", "*.otf"), recursive=True)
            files += glob.glob(os.path.join(d, "**", "*.ttc"), recursive=True)
            if files:
                print(f"[FONT] Dir {d}: {len(files)} font files")
                for f in files[:10]:
                    print(f"[FONT]   FILE: {f}")
                if len(files) > 10:
                    print(f"[FONT]   ... and {len(files)-10} more")
            else:
                print(f"[FONT] Dir {d}: exists but empty")
        else:
            print(f"[FONT] Dir {d}: NOT EXISTS")

    # 2. 列出 matplotlib 字体管理器里的所有字体家族名
    print("[FONT] === Matplotlib fontManager families (first 50) ===")
    # 修复：先取名字集合再排序，避免 FontEntry 比较错误
    families = sorted({f.name for f in fm.fontManager.ttflist})
    for i, fam in enumerate(families[:50]):
        print(f"[FONT]   FAMILY: {fam}")
    if len(families) > 50:
        print(f"[FONT]   ... and {len(families)-50} more families")

    # 3. 尝试 fc-list
    print("[FONT] === fc-list output ===")
    try:
        import subprocess
        out = subprocess.check_output(
            ["fc-list", ":lang=zh", "family", "file"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        for line in out.splitlines()[:20]:
            print(f"[FONT]   FC: {line}")
    except Exception as e:
        print(f"[FONT] fc-list failed: {e}")

    # 3. 尝试所有已知字体文件路径，找到第一个存在的
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKSC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    print("[FONT] === Checking candidate font files ===")
    working_path = None
    for p in candidate_paths:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        print(f"[FONT]   CHECK: {p} -> {'EXISTS' if exists else 'MISSING'} ({size} bytes)")
        if exists and size > 0:
            working_path = p
            break

    # 4. 尝试用 matplotlib.font_manager 加载第一个存在的字体文件
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import matplotlib

    family_used = None
    if working_path:
        try:
            prop = fm.FontProperties(fname=working_path)
            family = prop.get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            family_used = family
            print(f"[FONT] ✅ SUCCESS: Using font file {working_path} -> family '{family}'")
        except Exception as e:
            print(f"[FONT] ❌ FontProperties failed for {working_path}: {e}")

    # 5. 如果文件加载失败，尝试直接用字体家族名（从 fontManager 里找）
    if not family_used:
        print("[FONT] === Trying known family names from fontManager ===")
        known_families = [
            "Noto Sans CJK SC", "Noto Sans CJK", "Noto Sans SC",
            "Source Han Sans SC", "Source Han Sans",
            "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
            "DejaVu Sans",
        ]
        fm_families = {f.name for f in fm.fontManager.ttflist}
        for fam in known_families:
            if fam in fm_families:
                plt.rcParams["font.family"] = fam
                plt.rcParams["axes.unicode_minus"] = False
                family_used = fam
                print(f"[FONT] ✅ SUCCESS: Using family name '{fam}' (found in fontManager)")
                break

    # 6. 终极兜底
    if not family_used:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
        print("[FONT] ⚠️ FALLBACK: Using DejaVu Sans (no Chinese support)")

    print(f"[FONT] === FINAL: font.family = {plt.rcParams['font.family']} ===")
    print("[FONT] ===== FULL FONT DEBUG END =====")
    return plt
