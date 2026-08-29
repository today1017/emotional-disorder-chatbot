# -*- coding: utf-8 -*-
"""
 extended_analysis.py —— 成员 C 未完成交付物的补充分析
 在 text_analysis_pipeline.py 基础上补齐：
   阶段A：规则模型准确率评估（在 2000 条人工标注上计算混淆矩阵 + 分类报表）
   阶段B：不同平台（微博 vs 知乎）语料的情绪对比图
   阶段C：TF-IDF 高频词分析（积极 vs 消极）
   阶段D：LDA 主题建模（提取高频讨论主题）
 所有图表导出到 output_figures/（沿用现有目录与图号，新增 fig10~fig15）
 运行：python extended_analysis.py
 依赖：pandas, matplotlib, jieba, scikit-learn（pip install scikit-learn）
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jieba

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_demo"))
import rules as rule_model

import config
import utils
plt = utils.setup_matplotlib()

EVAL_MAP = {
    "情绪极性": lambda s: s,
    "求助信号": lambda s: "是" if s in ("明确求助", "间接求助") else "否",
    "安全风险": lambda s: s,
}


def load_and_merge():
    data = utils.load_and_merge()
    mask = pd.Series(True, index=data.index)
    for col, expected in config.EXPECTED_VALUES.items():
        mask &= data[col].astype(str).isin(expected)
    clean = data[mask].copy()
    print(f"加载 {len(data)} 条，过滤后有效 {len(clean)} 条")
    return clean


def stage_a_evaluate(data):
    print("\n" + "#" * 68)
    print("#  阶段A · 规则模型准确率评估（基于人工标注）")
    print("#" * 68)

    texts = data["文本"].tolist()
    report_rows = []
    figs = []

    for dim in ["情绪极性", "求助信号", "安全风险"]:
        y_true = data[dim].map(EVAL_MAP[dim])
        print(f"\n[评估] {dim}：对 {len(texts)} 条文本运行规则模型...")
        y_pred = [rule_model.analyze(t)[dim]["label"] for t in texts]

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, zero_division=0,
                                   output_dict=True)
        labels = sorted(set(y_true) | set(y_pred))
        print(f"  准确率: {acc:.4f}")
        print(pd.DataFrame(cm, index=labels, columns=labels))

        print(f"\n  分类报表：")
        print(classification_report(y_true, y_pred, zero_division=0))

        for cls in labels:
            report_rows.append({
                "维度": dim, "类别": cls,
                "精确率": round(cr[cls]["precision"], 4),
                "召回率": round(cr[cls]["recall"], 4),
                "F1": round(cr[cls]["f1-score"], 4),
                "样本数": cr[cls]["support"],
            })
        report_rows.append({
            "维度": dim, "类别": "总体",
            "精确率": "", "召回率": "", "F1": "",
            "样本数": len(y_true),
        })

        fig = plot_confusion(cm, labels, f"混淆矩阵 · {dim}", f"fig1{len(figs)+1}_confusion_{dim}.png")
        figs.append(fig)

    report = pd.DataFrame(report_rows)
    out_csv = os.path.join(config.OUTPUT_DIR, "evaluation_report.csv")
    report.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n>>> 评估报表已保存：{out_csv}")
    return figs


def plot_confusion(cm, classes, title, filename):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cm_show = cm.astype(float)
    im = ax.imshow(cm_show, cmap="Blues")
    threshold = cm_show.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm_show[i, j] > threshold else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(title, pad=10)
    fig.colorbar(im, shrink=0.85)
    fig.tight_layout()
    return fig


def stage_b_platform(data):
    print("\n" + "#" * 68)
    print("#  阶段B · 不同平台语料的情绪对比")
    print("#" * 68)

    ct = pd.crosstab(data["数据来源"], data["情绪极性"])
    ct = ct.reindex(columns=[c for c in config.SENTIMENT_ORDER if c in ct.columns])
    ct["合计"] = ct.sum(axis=1)
    print("\n各平台情绪频数表：")
    print(ct.to_string())

    pct = ct.drop(columns=["合计"]) / ct.drop(columns=["合计"]).sum(axis=1).values[:, None] * 100
    print("\n各平台情绪占比%（行内）：")
    print(pct.round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {"消极": "#C65B5B", "中性": "#B8B8B8", "积极": "#6C9E6C"}
    bottom = np.zeros(len(ct.index))
    for col in ct.columns[:-1]:
        vals = ct[col].values
        bars = ax.bar(list(ct.index), vals, bottom=bottom, label=col,
                      color=colors[col], edgecolor="white", width=0.5)
        for b, p in zip(bars, pct[col].values):
            if p > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2,
                        f"{p:.1f}%", ha="center", va="center", color="white", fontsize=10)
        bottom += vals
    ax.set_ylabel("文本条数")
    ax.set_title("不同平台语料情绪分布对比（微博 vs 知乎）", pad=12)
    ax.legend(title="情绪极性", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def tokenize_list(text):
    t = rule_model.clean_text(text)
    return [w for w in jieba.lcut(t)
            if len(w.strip()) >= 2 and w.strip() not in rule_model.STOPWORDS]


def stage_c_tfidf(data):
    print("\n" + "#" * 68)
    print("#  阶段C · TF-IDF 高频词分析（积极 vs 消极）")
    print("#" * 68)

    vectorizer = TfidfVectorizer(tokenizer=tokenize_list, lowercase=False,
                                 min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(data["文本"])
    names = np.array(vectorizer.get_feature_names_out())

    pos_mask = (data["情绪极性"] == "积极").values
    neg_mask = (data["情绪极性"] == "消极").values

    pos_mean = X[pos_mask].mean(axis=0).A1 if pos_mask.sum() else np.zeros(X.shape[1])
    neg_mean = X[neg_mask].mean(axis=0).A1 if neg_mask.sum() else np.zeros(X.shape[1])

    pos_idx = np.argsort(pos_mean)[::-1][:20]
    neg_idx = np.argsort(neg_mean)[::-1][:20]

    print(f"\n[积极组 Top20（TF-IDF均值）]：")
    print("   " + "  ".join(f"{names[i]}={pos_mean[i]:.3f}" for i in pos_idx))
    print(f"\n[消极组 Top20（TF-IDF均值）]：")
    print("   " + "  ".join(f"{names[i]}={neg_mean[i]:.3f}" for i in neg_idx))

    k = 15
    p_i, n_i = pos_idx[:k][::-1], neg_idx[:k][::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.barh([names[i] for i in p_i], [pos_mean[i] for i in p_i], color="#6C9E6C")
    ax1.set_title("积极文本 TF-IDF Top15")
    ax1.invert_yaxis()
    ax2.barh([names[i] for i in n_i], [neg_mean[i] for i in n_i], color="#C65B5B")
    ax2.set_title("消极文本 TF-IDF Top15")
    ax2.invert_yaxis()
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("积极 / 消极文本 TF-IDF 高频词对比", fontsize=14)
    fig.tight_layout()
    return fig


def stage_d_lda(data, n_topics=5, n_words=12):
    print("\n" + "#" * 68)
    print(f"#  阶段D · LDA 主题建模（{n_topics} 个主题）")
    print("#" * 68)

    cv = CountVectorizer(tokenizer=tokenize_list, lowercase=False,
                         min_df=5, max_df=0.5, max_features=2000)
    X = cv.fit_transform(data["文本"])
    names = np.array(cv.get_feature_names_out())

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                    max_iter=20)
    lda.fit(X)

    topic_words = []
    print("\n各主题关键词：")
    for t_idx, topic in enumerate(lda.components_):
        top = topic.argsort()[::-1][:n_words]
        words = [names[i] for i in top]
        topic_words.append(words)
        print(f"  主题{t_idx + 1}：{'、'.join(words)}")

    doc_topic = lda.transform(X)
    print("\n主题强度占比（所有文档平均）：")
    for t_idx in range(n_topics):
        print(f"  主题{t_idx + 1}：{doc_topic[:, t_idx].mean() * 100:.1f}%")

    fig, axes = plt.subplots(1, n_topics, figsize=(3.2 * n_topics, 4.5))
    if n_topics == 1:
        axes = [axes]
    for t_idx, ax in enumerate(axes):
        top = lda.components_[t_idx].argsort()[::-1][:n_words]
        ax.barh([names[i] for i in top][::-1],
                [lda.components_[t_idx][i] for i in top][::-1],
                color=plt.cm.tab10(t_idx))
        ax.set_title(f"主题{t_idx + 1}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("LDA 主题关键词分布", fontsize=14)
    fig.tight_layout()
    return fig


def stage_f_intensity(data):
    print("\n" + "#" * 68)
    print("#  阶段F · 情感强度评分模型（基于积极/消极关键词词典加权）")
    print("#" * 68)

    scores = []
    for _, row in data.iterrows():
        cleaned = rule_model.clean_text(str(row["文本"]))
        info = rule_model.compute_sentiment_intensity(cleaned)
        scores.append({
            "情绪极性": row["情绪极性"],
            "净得分": info["净得分"],
            "强度": info["强度"],
        })

    df = pd.DataFrame(scores)
    print("\n各情绪组情感强度描述统计：")
    print(df.groupby("情绪极性")["强度"].describe().round(1).to_string())
    print("\n各情绪组净得分描述统计：")
    print(df.groupby("情绪极性")["净得分"].describe().round(2).to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    order = ["消极", "中性", "积极"]
    colors = {"消极": "#C65B5B", "中性": "#B8B8B8", "积极": "#6C9E6C"}
    data_box = [df.loc[df["情绪极性"] == g, "强度"].values for g in order]
    bp = ax1.boxplot(data_box, tick_labels=order, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=1.5))
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(colors[g])
    ax1.set_ylabel("情感强度（0-100）")
    ax1.set_xlabel("情绪极性")
    ax1.set_title("情感强度分布箱线图", pad=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bins = np.arange(-6, 7, 1)
    for g, c in colors.items():
        vals = df.loc[df["情绪极性"] == g, "净得分"]
        ax2.hist(vals, bins=bins, alpha=0.6, label=g, color=c, edgecolor="white")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("净得分（正值=偏消极）")
    ax2.set_ylabel("文本数量")
    ax2.set_title("净得分分布直方图", pad=10)
    ax2.legend(title="情绪极性", frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("情感强度评分模型分析", fontsize=14)
    fig.tight_layout()
    return fig


def stage_e_export(fig_specs):
    print("\n" + "#" * 68)
    print("#  阶段E · 统一导出图表")
    print("#" * 68)
    for fname, fig in fig_specs:
        path = os.path.join(config.OUTPUT_DIR, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   [导出] {path}")


def main():
    data = load_and_merge()

    figs_a = stage_a_evaluate(data)
    fig_b = stage_b_platform(data)
    fig_c = stage_c_tfidf(data)
    fig_d = stage_d_lda(data, n_topics=5)
    fig_f = stage_f_intensity(data)

    fig_specs = [
        ("fig10_platform_sentiment.png", fig_b),
        ("fig11_tfidf_top_words.png", fig_c),
        ("fig12_lda_topics.png", fig_d),
        ("fig16_sentiment_intensity.png", fig_f),
    ]
    for i, f in enumerate(figs_a):
        fig_specs.append((f"fig1{3 + i}_confusion.png", f))

    stage_e_export(fig_specs)

    print("\n" + "=" * 68)
    print("补充分析全部完成！")
    print("=" * 68)


if __name__ == "__main__":
    main()
