# -*- coding: utf-8 -*-
"""
 情感障碍人群语言特征分析 —— 数据清洗 / 描述统计 / 交叉分析 / 文本挖掘 / 图表导出
 项目阶段：大创项目 · 数据分析阶段
 运行环境：Python 3.9+ ；依赖：pandas, matplotlib, wordcloud, jieba
 运行方式：python text_analysis_pipeline.py
 输出目录：./output_figures/（与脚本同目录自动创建）
"""
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils
plt = utils.setup_matplotlib()


def log_stage(stage_no, stage_name):
    print("=" * 68)
    print(f"【阶段{stage_no}】{stage_name} 执行完成 ✔")
    print("=" * 68)


def quality_report(data):
    print("\n" + "#" * 68)
    print("#  阶段一 · 数据质量报告")
    print("#" * 68)

    print(f"\n[1] 总条数：{len(data)} 条")
    print(f"    来源构成：{data['数据来源'].value_counts().to_dict()}")

    missing = pd.DataFrame({
        "缺失数": data.isna().sum() + data.apply(lambda s: (s.astype(str) == "").sum()),
        "缺失率%": ((data.isna().sum() + data.apply(lambda s: (s.astype(str) == "").sum())) / len(data) * 100).round(2),
    })
    print("\n[2] 缺失值统计：")
    print(missing.to_string())

    print("\n[3] 取值合法性校验（预期范围外为脏数据）：")
    total_dirty = 0
    for col, expected in config.EXPECTED_VALUES.items():
        actual = set(data[col].dropna().astype(str).unique())
        unexpected = actual - expected - {""}
        if unexpected:
            total_dirty += sum(data[col].astype(str).isin(unexpected))
            print(f"    {col}：发现范围外取值 {unexpected}（共 {sum(data[col].astype(str).isin(unexpected))} 条）")
        else:
            print(f"    {col}：取值均在预期范围内 ✔")

    print("\n[4] 各维度分布（频数 / 占比%）：")
    for col in config.EXPECTED_VALUES.keys():
        vc = data[col].value_counts(dropna=False)
        pct = (vc / len(data) * 100).round(2)
        dist = pd.DataFrame({"频数": vc, "占比%": pct})
        print(f"\n    —— {col} ——")
        print(dist.to_string())

    print("\n[5] 文本字段抽查（前 3 条）：")
    for t in data["文本"].head(3):
        print("    ", str(t)[:50])

    print(f"\n>>> 脏数据合计 {total_dirty} 条（后续分析阶段将自动过滤范围外取值）")
    return total_dirty


def calc_freq_pct(data, col, category_order):
    vc = data[col].value_counts()
    order = [c for c in category_order if c in vc.index]
    df = pd.DataFrame({
        "类别": order,
        "频数": [vc[c] for c in order],
        "占比%": [round(vc[c] / vc.sum() * 100, 2) for c in order],
    })
    return df


def plot_single_distribution(freq_df, title, ylabel="频数", filename=None, show_pct=True):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cats = freq_df["类别"]
    vals = freq_df["频数"]
    colors = [config.COLOR_MAP.get(c, "#7F7F7F") for c in cats]

    bars = ax.bar(cats, vals, color=colors, edgecolor="white", width=0.6)
    pcts = freq_df["占比%"].tolist()
    for bar, count, pct in zip(bars, vals, pcts):
        label = str(count) if not show_pct else f"{count}\n({pct}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                label, ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def stage2_descriptive(data):
    print("\n" + "#" * 68)
    print("#  阶段二 · 描述性统计分析")
    print("#" * 68)

    specs = [
        ("情绪极性", config.SENTIMENT_ORDER, "图1 情绪极性分布", "fig1_sentiment_distribution.png"),
        ("求助信号", config.HELP_ORDER, "图2 求助信号分布", "fig2_help_seeking_distribution.png"),
        ("是否隐喻", config.METAPHOR_ORDER, "图3 隐喻使用分布", "fig3_metaphor_distribution.png"),
        ("安全风险", config.RISK_ORDER, "图4 安全风险分布", "fig4_risk_flag_distribution.png"),
    ]
    for col, order, title, fname in specs:
        freq_df = calc_freq_pct(data, col, order)
        print(f"\n[{col}] 频数表：")
        print(freq_df.to_string(index=False))
        fig = plot_single_distribution(freq_df, title)
        plt.close(fig)
    log_stage("二", "描述性统计分析")


def cross_table(data, row_col, col_col, row_order, col_order):
    ct = pd.crosstab(data[row_col], data[col_col])
    ct = ct.reindex(index=[r for r in row_order if r in ct.index],
                    columns=[c for c in col_order if c in ct.columns])
    ct["合计"] = ct.sum(axis=1)
    return ct


def plot_stacked_cross(ct, row_order, col_order, title, xlabel, ylabel, filename, show_legend=True):
    ct_pct = ct.drop(columns=["合计"]) / ct.drop(columns=["合计"]).sum(axis=1).values[:, None] * 100

    fig, ax = plt.subplots(figsize=(8, 4.8))
    categories = list(ct.index)
    value_cols = [c for c in ct.columns if c != "合计"]
    n_cols = len(value_cols)
    bottom = np.zeros(len(categories))

    for c in value_cols:
        vals = ct[c].values
        pcts = ct_pct[c].values
        color = config.COLOR_MAP.get(c, "#7F7F7F")
        bars = ax.bar(categories, vals, bottom=bottom, label=c, color=color,
                      edgecolor="white", width=0.55)
        for b, p in zip(bars, pcts):
            if p > 0:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f"{p:.1f}%", ha="center", va="center",
                        fontsize=9, color="white")
        bottom += vals

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(title="图例", loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def stage3_cross(data):
    print("\n" + "#" * 68)
    print("#  阶段三 · 交叉分析")
    print("#" * 68)

    specs = [
        ("情绪极性", "求助信号", config.SENTIMENT_ORDER, config.HELP_ORDER,
         "图5 情绪极性 × 求助信号 交叉分布", "情绪极性", "fig5_sentiment_x_help.png"),
        ("情绪极性", "是否隐喻", config.SENTIMENT_ORDER, config.METAPHOR_ORDER,
         "图6 情绪极性 × 隐喻使用 交叉分布", "情绪极性", "fig6_sentiment_x_metaphor.png"),
        ("求助信号", "安全风险", config.HELP_ORDER, config.RISK_ORDER,
         "图7 求助信号 × 安全风险 交叉分布", "求助信号", "fig7_help_x_risk.png"),
    ]
    for row_col, col_col, row_order, col_order, title, xlabel, fname in specs:
        ct = cross_table(data, row_col, col_col, row_order, col_order)
        print(f"\n[{row_col} × {col_col}] 交叉频数表：")
        print(ct.to_string())
        plot_stacked_cross(ct, row_order, col_order, title, xlabel, "频数", fname)
        plt.close("all")
    log_stage("三", "交叉分析")


def top_keywords(tokens_list, k=20):
    counter = Counter()
    for toks in tokens_list:
        counter.update(toks)
    return counter.most_common(k)


def plot_wordcloud(freq_counter, title, filename, font_path):
    from wordcloud import WordCloud

    wc = WordCloud(
        font_path=font_path or "simhei.ttf",
        width=1200, height=800,
        background_color="white",
        max_words=200,
        colormap="Greens" if "积极" in title else "Reds",
        random_state=42,
    ).generate_from_frequencies(freq_counter)

    path = os.path.join(config.OUTPUT_DIR, filename)
    wc.to_file(path)
    print(f"   [词云] 已保存：{path}")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=14)
    return fig, path


import os

def stage4_text_mining(data):
    print("\n" + "#" * 68)
    print("#  阶段四 · 文本基础挖掘（高频词 + 词云）")
    print("#" * 68)

    stopwords = utils.load_stopwords()

    pos_data = data[data["情绪极性"] == "积极"]
    neg_data = data[data["情绪极性"] == "消极"]
    print(f"   积极文本 {len(pos_data)} 条 | 消极文本 {len(neg_data)} 条")

    print("   [分词] 正在对积极文本分词...")
    pos_tokens = [utils.tokenize(t, stopwords) for t in pos_data["文本"]]
    print("   [分词] 正在对消极文本分词...")
    neg_tokens = [utils.tokenize(t, stopwords) for t in neg_data["文本"]]

    pos_top = top_keywords(pos_tokens, 20)
    neg_top = top_keywords(neg_tokens, 20)
    pos_dict = dict(pos_top)
    neg_dict = dict(neg_top)

    print("\n[积极-消极高频词对比表]（Top 20）")
    print(f"{'排名':<4}{'积极组词语':<10}{'频次':<6}{'消极组词语':<10}{'频次':<6}")
    print("-" * 46)
    for i in range(20):
        pw, pn = pos_top[i] if i < len(pos_top) else ("", "")
        nw, nn = neg_top[i] if i < len(neg_top) else ("", "")
        print(f"{i + 1:<6}{pw:<10}{pn:<6}{nw:<10}{nn:<6}")

    font_path = utils.find_chinese_font()
    fig_p, _ = plot_wordcloud(pos_dict, "积极文本高频词词云", "fig8_wordcloud_positive.png", font_path)
    plt.close(fig_p)
    fig_n, _ = plot_wordcloud(neg_dict, "消极文本高频词词云", "fig9_wordcloud_negative.png", font_path)
    plt.close(fig_n)

    log_stage("四", "文本基础挖掘")


def stage5_export(data):
    print("\n" + "#" * 68)
    print("#  阶段五 · 一键导出全部图表")
    print("#" * 68)

    fig_specs = []

    dim_specs = [
        ("情绪极性", config.SENTIMENT_ORDER, "图1 情绪极性分布", "fig1_sentiment_distribution.png"),
        ("求助信号", config.HELP_ORDER, "图2 求助信号分布", "fig2_help_seeking_distribution.png"),
        ("是否隐喻", config.METAPHOR_ORDER, "图3 隐喻使用分布", "fig3_metaphor_distribution.png"),
        ("安全风险", config.RISK_ORDER, "图4 安全风险分布", "fig4_risk_flag_distribution.png"),
    ]
    for col, order, title, fname in dim_specs:
        freq_df = calc_freq_pct(data, col, order)
        fig_specs.append((fname, plot_single_distribution(freq_df, title)))

    cross_specs = [
        ("情绪极性", "求助信号", config.SENTIMENT_ORDER, config.HELP_ORDER,
         "图5 情绪极性 × 求助信号 交叉分布", "情绪极性", "fig5_sentiment_x_help.png"),
        ("情绪极性", "是否隐喻", config.SENTIMENT_ORDER, config.METAPHOR_ORDER,
         "图6 情绪极性 × 隐喻使用 交叉分布", "情绪极性", "fig6_sentiment_x_metaphor.png"),
        ("求助信号", "安全风险", config.HELP_ORDER, config.RISK_ORDER,
         "图7 求助信号 × 安全风险 交叉分布", "求助信号", "fig7_help_x_risk.png"),
    ]
    for row_col, col_col, row_order, col_order, title, xlabel, fname in cross_specs:
        ct = cross_table(data, row_col, col_col, row_order, col_order)
        fig_specs.append((fname, plot_stacked_cross(ct, row_order, col_order,
                                                    title, xlabel, "频数", fname)))

    for fname, fig in fig_specs:
        path = os.path.join(config.OUTPUT_DIR, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   [导出] {path}")

    pngs = sorted(f for f in os.listdir(config.OUTPUT_DIR) if f.endswith(".png"))
    print(f"\n>>> 共 {len(pngs)} 张图片已导出到目录：{config.OUTPUT_DIR}")
    for p in pngs:
        print("      -", p)
    log_stage("五", "图表一键导出")


def main():
    print("开始执行情感障碍人群语言特征分析流水线...\n")

    data = utils.load_and_merge()
    quality_report(data)
    log_stage("一", "数据加载与质量报告")

    mask = pd.Series(True, index=data.index)
    for col, expected in config.EXPECTED_VALUES.items():
        mask &= data[col].astype(str).isin(expected)
    data_clean = data[mask].copy()
    print(f"\n[过滤] 保留有效记录 {len(data_clean)} / {len(data)} 条"
          f"（剔除 {len(data) - len(data_clean)} 条范围外/缺失标注）")

    stage2_descriptive(data_clean)
    stage3_cross(data_clean)
    stage4_text_mining(data_clean)
    stage5_export(data_clean)

    print("\n" + "=" * 68)
    print("全流程执行完毕！所有图表已导出至 output_figures/ 目录。")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n[错误] 缺少依赖库：{e}")
        print("请先安装依赖：pip install pandas matplotlib jieba wordcloud")
        sys.exit(1)
