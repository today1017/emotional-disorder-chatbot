# -*- coding: utf-8 -*-
"""
 rules.py —— 规则模型核心逻辑（与 Streamlit UI 解耦，便于独立测试）
 功能：
   1. 从人工标注语料（csv）中按"词类区分度"自动提取各维度的关键词表
   2. 基于高频词匹配的简单规则分类：情绪极性 / 求助信号 / 安全风险
   3. 按（情绪极性, 求助信号, 安全风险）组合映射预设建议话术
 依赖：pandas, jieba（均为标准库之外需 pip 安装）
"""
import os
import re
import sys
from collections import Counter
from functools import lru_cache

import jieba
import pandas as pd

# 同目录下直接导入 config 和 utils（部署时已复制到同级）
import config
import utils

CLASS_LABEL_MAP = {
    "情绪极性": {"积极": ["积极"], "消极": ["消极"], "中性": ["中性"]},
    "求助信号": {"是": ["明确求助", "间接求助"], "否": ["无求助意图"]},
    "安全风险": {"是": ["是"], "否": ["否"]},
}

MIN_TOTAL = 5
MIN_SHARE = 0.3
TOP_K = {
    "情绪极性": 50,
    "求助信号": 60,  # 提高以捕获更多求助信号
    "安全风险": 50,
}

NOISE_WORDS = {
    "哈哈哈", "哈哈哈哈", "啊啊啊", "呜呜", "呜呜呜", "呵呵", "嘿嘿", "嘻嘻",
    "阿萍", "奶奶", "天心", "富马", "盐酸", "劳拉", "草酸", "喹硫平",
    "嗯", "哦", "哈", "呵呵", "行行行", "知道了", "听你说", "是这样的",
}

STOPWORDS = utils.BASE_STOPWORDS

FALLBACK_KEYWORDS = {
    "情绪极性": {
        "积极": ["开心", "高兴", "快乐", "不错", "喜欢", "舒服", "轻松", "期待",
                 "治愈", "谢谢", "爱", "温暖", "好起来", "加油", "鼓励"],
        "消极": ["痛苦", "难受", "抑郁", "焦虑", "崩溃", "绝望", "想死", "自杀",
                 "难过", "伤心", "哭", "累", "疲惫", "孤独", "害怕", "恐惧",
                 "折磨", "好不起来", "没有希望", "没希望", "看不到希望", "没有出路",
                 "活着没意思", "没有意义", "不想活", "想结束", "受不了", "撑不住",
                 "不想干", "什么都不想做", "好累", "太累了", "心累", "绝望"],
        "中性": [],
    },
    "求助信号": {
        "是": ["怎么办", "帮帮我", "帮帮我吧", "帮帮忙", "帮助我", "救救", "救救我",
               "谁能", "有人吗", "有没有办法", "不知道怎么办", "怎么办才好", "求助",
               "请教", "咨询", "需要帮助", "帮我", "求求", "带我", "怎么办啊",
               "我想找人", "我想问问", "我想聊聊", "我需要帮助", "请求帮助", "紧急",
               "需要有人", "有没有人", "能不能帮我"],
        "否": [],
    },
    "安全风险": {
        "是": ["自杀", "自残", "想死", "结束生命", "活不下去", "伤害自己", "割腕",
               "吞药", "跳楼", "不想活", "安乐死", "结束一切", "结束自己",
               "活着没意思", "撑不下去", "撑不住", "解脱", "不想存在",
               "自杀意念", "自伤", "轻生", "了结", "别死了", "结束了",
               # === 访谈新增高价值风险词 ===
               "划手", "撞墙", "撕指甲", "捶墙", "捶墙流血", "头撞墙", "拳头流血",
               "攒药", "攒够药", "吃药自杀", "准备好跳", "站在天台", "坐在窗边",
               "手里拿着药瓶", "刀片很凉", "看着血流出来", "皮肤上又多了一道",
               "今晚可能就结束了", "活不过今晚", "睡过去就醒不来", "最后一次说这些",
               "告别", "遗言", "再见", "这是最后一次", "照顾好自己", "对不起大家",
               "生日过后就", "等考完试", "还有几天", "时间限定",
               "流血", "破皮", "缝针", "洗胃", "抢救", "住院", "躯体化",
               "安全约定", "安全契约", "紧急联络人", "替代自伤", "舒缓方法",
               "情绪过载", "快要崩溃", "撑不住了", "熬不过今晚", "到极限了",
               "窒息感", "被按在水里", "被淹没", "黑暗中", "深渊边缘",
               "黑狗", "溺水", "坠落", "下坠", "容器隐喻", "战斗隐喻"],
        "否": [],
    },
}

SEED_WORDS = {
    "情感极性": {
        "积极": ["开心", "快乐", "幸福", "高兴", "喜欢", "美好", "希望", "温暖",
                 "享受", "满足", "成功", "胜利", "愉快", "美好", "阳光", "甜美",
                 "精彩", "突破", "进步", "成长", "收获", "感恩", "乐观", "自信",
                 "有趣", "好玩", "棒", "优秀", "平和", "满足",
                 # === 扩充: 生活满意度/社交正向/自我效能/积极体验 ===
                 "顺利", "舒适", "轻松", "自在", "安心", "踏实", "温馨", "甜蜜",
                 "充实", "有意义", "有成就感", "被理解", "被关爱", "被支持", "被接纳",
                 "期待", "向往", "憧憬", "信任", "珍惜", "感恩", "热情", "活力",
                 "盼望", "喜爱", "热爱", "钟爱", "偏爱", "挚爱",
                 "好", "赞", "棒", "妙", "美", "善", "真", "纯",
                 "喜", "乐", "福", "祥", "瑞", "吉", "庆", "贺",
                 "欢", "欣", "悦", "愉", "畅", "怡", "舒", "宁",
                 "强", "稳", "固", "坚", "实", "正", "清", "明"],
        "消极": ["痛苦", "压力", "难过", "伤心", "焦虑", "烦躁", "沮丧", "疲惫",
                 "压抑", "孤独", "寂寞", "委屈", "失望", "愧疚",
                 "厌倦", "厌烦", "颓废", "低落",
                 "难受死了", "生不如死", "活不下去", "想不开",
                 # === 扩充: 抑郁核心症状/焦虑躯体化/认知扭曲/绝望感 ===
                 "抑郁", "崩溃", "绝望", "无助", "无望", "空虚", "麻木", "窒息",
                 "失眠", "嗜睡", "噩梦", "头痛", "胸闷", "心慌", "发抖", "冒汗",
                 "没动力", "没兴趣", "没意义", "没价值", "没希望", "没未来",
                 "自责", "内疚", "悔恨", "自我否定", "自我怀疑", "自我厌恶",
                 "想哭", "眼泪", "哭泣", "落泪", "泪流", "抽泣", "哽咽",
                 "紧张", "恐惧", "害怕", "担心", "惶恐", "惊恐", "不安",
                 "心累", "无力", "疲倦", "倦怠", "慵懒", "消沉", "颓丧",
                 "折磨", "煎熬", "挣扎", "困苦", "艰辛", "艰难", "难熬",
                 "想死", "轻生", "遗书",
                 "活着没意思", "不如死了", "离开世界", "不想活", "结束生命",
                 "恨自己", "讨厌自己", "看不起自己", "没用", "废物", "垃圾",
                 "好累", "好苦", "好难", "受不了", "撑不住", "扛不住",
                 "黑暗", "深渊", "泥潭", "牢笼", "枷锁",
                 "崩溃了", "撑不下去", "受不了了", "坚持不住", "到极限了"],
        "中性": [],
    },
    "求助意图": {
        "显": ["怎么了", "帮帮我", "救救我", "好忙", "怎么办", "感染", "感染了",
               "谁来", "谁能帮", "我没有办法", "我不知道怎么", "怎么办好", "求助",
               "帮忙", "询问", "需要帮助", "可以吗", "能否", "可不可以",
               "怎么搞", "怎么办好", "怎么办好", "怎么回事", "怎么办好",
               # === 知乎求助语料 ===
               "怎么办好呢", "能不能帮", "我没有办法", "怎么办好呢",
               "谁能给我个怎么解决去", "帮帮我的问题怎么办",
               "谁能够给我个怎么解决去", "帮帮我的问题",
               "有人可以帮忙吗", "帮帮忙",
               "谁来帮帮我", "帮帮忙吧", "谁来帮帮我",
               "谁能够帮我解决", "求求你帮帮我",
               "能不能帮帮我", "求求你帮帮我",
               "帮帮我吧", "帮帮我", "帮帮忙", "帮帮忙吧",
               "谁来帮帮我", "帮帮我", "帮帮忙"],
        "无": ["没关系", "不要紧", "行", "好", "无关紧要", "不太严重",
               # === 扩充: 日常闲聊/自给自足/拒绝帮助/不涉及心理话题 ===
               "可以", "没事", "不必", "不用", "不需要", "我能行",
               "我自己", "我没事", "挺好的", "还好", "一般", "还行",
               "差不多", "就这样", "无所谓", "随便", "哈哈", "嘻嘻",
               "谢谢", "感谢", "多谢", "客气", "不客气", "好的",
               "收到", "了解", "明白", "知道了", "今天", "明天",
               "天气", "吃饭", "睡觉", "工作", "学习", "上课",
               "考试", "作业", "放假", "无聊", "一般般", "凑合",
               "看电影", "玩游戏", "听歌", "运动", "旅游",
               "不聊了", "再见", "拜拜", "晚安", "你说得对",
               "有道理", "同意", "支持", "加油", "然后呢",
               "我很好", "我挺好的", "我还好", "别担心",
               "不用管我", "不用担心", "我自己能处理",
               "我能搞定", "没问题", "小事一桩", "不值一提",
               "生活", "日常", "普通", "正常", "刚下班", "刚放学"],
    },
    "风险标记": {
        "有": ["自杀", "跳楼", "割腕", "吃药", "不想活", "活不下去", "想不开", "离开",
               "死了", "死掉", "去死", "找死", "消失", "结束", "结束生命",
               "解脱", "一了百了", "再也不想", "没有意义", "没有未来",
               "没人在乎", "没人关心", "没人爱我", "被抛弃",
               "遗书", "告别", "最后", "最后一次",
               # === 扩充: 间接自杀表达/自伤/放弃治疗/死亡美化/暗示信号 ===
               "撞墙", "磕墙", "磕手指", "撞墙流血", "拳头砸墙",
               "站阳台", "靠窗边", "数药瓶", "药片数量",
               "割手腕", "割皮", "割伤", "伤口", "流血",
               "拒绝治疗", "不配合治疗", "差不多了",
               "睡过去就不想醒", "跟你说这些",
               "展露和自己", "对不住大家",
               "看开", "看淡", "时日无多", "时间不多了",
               "没人需要我", "我是累赘", "拖累别人",
               "不想成为负担", "我走了", "别找我",
               "不用等我", "照顾好自己",
               "刀片", "美工刀", "剪刀", "玻璃碎片",
               "安眠药", "百草枯", "敌敌畏", "老鼠药",
               "来不及了", "太晚了", "离开这里",
               "消失吧", "结束吧", "再见了世界",
               "世界再见", "来生", "下辈子",
               "遗言", "遗愿", "后事", "安排好",
               "服毒", "上吊", "投河", "卧轨", "触电"],
        "无": ["状态好", "挺好的", "没有", "最近可以", "一般般", "没事"],
    },
}


@lru_cache(maxsize=1)
def load_corpus():
    frames = []
    for path in config.DATA_FILES:
        if not os.path.exists(path):
            continue
        for enc in (config.DATA_ENCODING, "utf-8-sig", "gbk", "utf-8"):
            try:
                frames.append(pd.read_csv(path, encoding=enc))
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
    return pd.concat(frames, ignore_index=True) if frames else None


def clean_text(text):
    return utils.clean_text(text)


def tokenize(text):
    return utils.tokenize(text)


def extract_keywords(df):
    keywords = {}
    for dim, cls_map in CLASS_LABEL_MAP.items():
        cls_counters = {}
        for cls, labels in cls_map.items():
            sub = df[df[dim].astype(str).isin(labels)]
            cnt = Counter()
            for t in sub["文本"]:
                cnt.update(tokenize(t))
            cls_counters[cls] = cnt

        total = Counter()
        for cnt in cls_counters.values():
            total.update(cnt)

        dim_words = {}
        for cls, cnt in cls_counters.items():
            cand = []
            for w, c in cnt.items():
                if w in NOISE_WORDS or len(w) < 2:
                    continue
                tot = total[w]
                if tot < MIN_TOTAL or c / tot < MIN_SHARE:
                    continue
                cand.append((c, w))
            cand.sort(reverse=True)
            dim_words[cls] = [w for _, w in cand[:TOP_K[dim]]]
        keywords[dim] = dim_words
    return keywords


def merge_seeds(extracted):
    merged = {}
    for dim, cls_map in extracted.items():
        merged[dim] = {}
        for cls, words in cls_map.items():
            seeds = SEED_WORDS.get(dim, {}).get(cls, [])
            combined = list(seeds) + [w for w in words if w not in seeds]
            merged[dim][cls] = combined[:TOP_K[dim]]
    return merged


@lru_cache(maxsize=1)
def get_keywords():
    df = load_corpus()
    if df is not None and not df.empty and "文本" in df.columns:
        return merge_seeds(extract_keywords(df))
    return FALLBACK_KEYWORDS


DEFAULT_LABELS = {"情绪极性": "中性", "求助信号": "否", "安全风险": "否"}


def classify(cleaned_text, dim, keywords):
    scores = {}
    matched = set()
    for cls, words in keywords[dim].items():
        s = 0
        for w in words:
            n = cleaned_text.count(w)
            if n > 0:
                s += n
                matched.add(w)
        scores[cls] = s

    best_cls = max(scores, key=lambda c: scores[c])
    if scores[best_cls] == 0:
        return DEFAULT_LABELS[dim], []
    return best_cls, sorted(matched)


def compute_sentiment_intensity(cleaned_text, keywords=None):
    if keywords is None:
        keywords = get_keywords()
    pos_set = set(keywords["情绪极性"]["积极"])
    neg_set = set(keywords["情绪极性"]["消极"])
    pos_hits = {w: cleaned_text.count(w) for w in pos_set if w in cleaned_text}
    neg_hits = {w: cleaned_text.count(w) for w in neg_set if w in cleaned_text}
    pos_score = sum(pos_hits.values())
    neg_score = sum(neg_hits.values())
    net = neg_score - pos_score
    intensity = min(100, int(round(abs(net) * 20)))
    return {"净得分": net, "强度": intensity,
            "积极命中": pos_hits, "消极命中": neg_hits}


SUGGESTIONS = {
    ("消极", "是", "是"):
        "你现在描述的情况非常重要，请先确保自己处在安全的环境里。"
        "不要独自承受——请立即联系专业支持（全国心理援助热线 12356），"
        "并让身边信任的人陪伴你。你的安全是最重要的，我在这里陪你。",
    ("消极", "是", "否"):
        "谢谢你愿意把难受说出来，这已经很不容易了。"
        "听起来你现在情绪很低落，可以多和我说说发生了什么吗？"
        "我会认真听。如果需要，我也可以帮你对接专业的心理支持资源。",
    ("消极", "否", "是"):
        "我注意到你的描述里有一些需要重视的信号。"
        "你不需要一个人扛着，我很想陪你聊一聊。"
        "同时我非常建议你联系专业人士（全国心理援助热线 12356），"
        "或者让我帮你找到附近的帮助资源，好吗？",
    ("消极", "否", "否"):
        "我能感受到你现在的疲惫和低落，谢谢你愿意信任我。"
        "你不需要一个人扛着，随时都可以来找我聊聊。"
        "慢慢来，我会一直在。",
    ("中性", "是", "是"):
        "谢谢你的信任。请你先确保自己处在安全的环境，"
        "尽快联系专业支持（全国心理援助热线 12356）或让信任的人陪伴你。"
        "我在这里，会陪你把问题一件一件地理清楚。",
    ("中性", "是", "否"):
        "好的，我听到你的诉求了。具体是遇到了什么情况呢？"
        "我们可以一起想办法，我也会尽力帮你找到合适的支持资源。",
    ("中性", "否", "是"):
        "我注意到你的描述里有一些可能需要关注的信号。"
        "如果最近状态有波动，建议你多加留意，必要时及时寻求专业支持"
        "（如全国心理援助热线 12356）。需要的话，我也很愿意陪你聊聊。",
    ("中性", "否", "否"):
        "收到。如果之后有想聊的、需要帮忙的地方，随时告诉我，我都在。",
    ("积极", "是", "是"):
        "你能主动表达求助，是非常勇敢的一步。"
        "不过你的描述里仍包含一些需要重视的线索，请务必联系专业人士"
        "（全国心理援助热线 12356）确认安全，我也会一直陪着你。",
    ("积极", "是", "否"):
        "为你现在相对平稳的状态感到高兴，也谢谢你愿意求助。"
        "说说具体想解决什么？我可以帮你梳理一下可以用的资源。",
    ("积极", "否", "是"):
        "你的描述里有一些值得留意的信号。即使整体感觉还好，"
        "也建议近期多关注自己的情绪变化，必要时寻求专业支持。"
        "我会一直在你身边。",
    ("积极", "否", "否"):
        "太好了，看起来你最近状态不错。"
        "继续保持这些让你开心的习惯，如果偶尔低落，也可以随时来找我聊聊。",
    ("消极", "咨询师视角", "是"):
        "谢谢你分享这些治疗过程中的感受。我知道有时候很难在专业支持之外找到出口。"
        "我们可以一起探索哪些应对策略对你最有效，我会一直在。",
    ("消极", "咨询师视角", "否"):
        "谢谢你的诚实分享。即使在没有紧急风险的情况下，这些情绪也值得被看见和接纳。"
    ,
    ("中性", "咨询师视角", "是"):
        "谢谢你信任我。安全是第一位的——请确保你处在安全的环境中，并联系专业支持（12356）来获得持续的指导。",
    ("中性", "咨询师视角", "否"):
        "收到。如果你以后想聊聊这些感受或者有任何变化，随时都可以来找我。我会保持空间给你。",
}

DEFAULT_SUGGESTION = (
    "谢谢你愿意分享。我听到了你的表达，很希望能帮你。"
    "如果愿意，可以再多说一些你的具体情况，我会尽力陪你一起想办法。"
)


def get_suggestion(sentiment, help_flag, risk_flag):
    return SUGGESTIONS.get((sentiment, help_flag, risk_flag), DEFAULT_SUGGESTION)


def analyze(text):
    cleaned = clean_text(text)
    keywords = get_keywords()
    result = {}
    for dim in CLASS_LABEL_MAP:
        label, matched = classify(cleaned, dim, keywords)
        result[dim] = {"label": label, "matched": matched}
    result["情感强度"] = compute_sentiment_intensity(cleaned, keywords)
    result["建议话术"] = get_suggestion(
        result["情绪极性"]["label"],
        result["求助信号"]["label"],
        result["安全风险"]["label"],
    )
    return result


# ==========================================================================
# 上下文感知分析（多轮对话支持）
# ==========================================================================
def analyze_with_context(text, context=None):
    """
    结合对话上下文的增强分析：
      - 上下文情绪修正：短促回应（如"嗯""是的"）单独分析会判为中性，
        结合前一轮情绪进行继承修正
      - 风险升级检测：连续多轮低风险 → 本轮高风险时标记升级
      - 情绪波动检测：情感强度大幅跳变时附加波动提示
    返回在 analyze() 结果基础上追加 "上下文" 键：
      {"情绪修正": bool, "继承情绪": str, "风险升级": bool,
       "强度波动": float, "连续消极": int, "上下文提示": [str]}
    """
    result = analyze(text)

    if context is None:
        return result

    ctx = context
    hints = []

    # 1) 情感强度波动检测（与上一轮比较）
    delta = 0.0
    if len(ctx.sentiment_trend) >= 1:
        prev = ctx.sentiment_trend[-1]
        cur = result["情感强度"]["强度"]
        delta = cur - prev
        if abs(delta) >= 15:
            direction = "上升（情绪加剧）" if delta > 0 else "下降（情绪缓和）"
            hints.append(f"情感强度较上轮{direction}（{prev}→{cur}）")

    # 2) 短促回应继承修正（新文本无关键词命中 + 文本极短 + 历史存在主导情绪）
    cleaned = clean_text(text)
    short_reply = len(cleaned) <= 6
    no_hits = all(not result[dim]["matched"] for dim in ("情绪极性", "求助信号", "安全风险"))
    inherited = False
    if short_reply and no_hits and ctx.total_turns > 0:
        # 继承前一轮情绪极性（若存在）
        if ctx._sentiment_labels:
            prev_sent = ctx._sentiment_labels[-1]
            result["情绪极性"]["label"] = prev_sent
            result["情绪极性"]["matched"] = [f"（继承前文{prev_sent}语境）"]
            inherited = True
            hints.append(f"短促回应，按上下文继承前一轮{prev_sent}情绪语境")
        else:
            last_sent = ctx.sentiment_trend[-1] if ctx.sentiment_trend else 50
            if last_sent >= 55:
                result["情绪极性"]["label"] = "消极"
                result["情绪极性"]["matched"] = ["（继承前文消极语境）"]
                inherited = True
                hints.append("短促回应，按上下文继承消极情绪语境")

    # 3) 风险升级检测（此前无风险、本轮有风险）
    escalation = False
    if ctx.risk_history and not ctx.risk_history[-1] and result["安全风险"]["label"] == "是":
        escalation = True
        hints.append("⚠️ 本轮首次出现安全风险信号，建议重点干预")

    # 4) 连续消极提示
    if ctx.negative_streak >= 2 and result["情绪极性"]["label"] == "消极":
        hints.append(f"用户已连续 {ctx.negative_streak + 1} 轮表达消极情绪")

    result["上下文"] = {
        "情绪修正": inherited,
        "继承情绪": result["情绪极性"]["label"] if inherited else None,
        "风险升级": escalation,
        "强度波动": round(delta, 1),
        "连续消极": ctx.negative_streak + 1 if result["情绪极性"]["label"] == "消极" else 0,
        "上下文提示": hints,
    }
    return result


if __name__ == "__main__":
    tests = [
        "今天好难过，活着真没意思，我好想结束这一切，太痛苦了。",
        "今天跟妈妈聊了很久，感觉好多了，心情很放松。",
        "我最近总是睡不好，工作压力很大，该怎么办，谁能帮帮我？",
        "最近事情好多，但是慢慢来吧。",
    ]
    for t in tests:
        print("-" * 60)
        print("文本：", t)
        r = analyze(t)
        for dim in CLASS_LABEL_MAP:
            item = r[dim]
            print(f"  {dim}: {item['label']}  (命中: {'、'.join(item['matched'][:8]) or '无'})")
        print("  建议话术:", r["建议话术"])
