# -*- coding: utf-8 -*-
"""
system_diagrams.py —— 生成系统逻辑结构图与交互流程图（成员 C 产出物）
生成两张高清 PNG，直接用于文档或答辩 PPT：
  fig17_system_logic.png   —— 系统逻辑结构图（分层架构：输入→分析→策略→输出）
  fig18_interaction_flow.png —— 交互流程图（端到端处理路径，含风险分支）
运行：python system_diagrams.py
依赖：matplotlib（已安装）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = r"C:\Users\12745\OneDrive\桌面\output_figures"


# ------------------------------ 公共绘制工具 ------------------------------ #
def _round_box(ax, xy, w, h, text, fc, ec="#666666", lw=1.5,
               fontsize=11, color="black", bold=False):
    """绘制圆角矩形 + 居中文字"""
    box = mpatches.FancyBboxPatch(xy, w, h,
                                  boxstyle="round,pad=0.05",
                                  fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=color, fontweight=weight, zorder=3,
            path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])


def _arrow(ax, start, end, color="#888888", lw=1.5, style="->", zorder=1):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)


# ==========================================================================
# 图17：系统逻辑结构图（分层）
# ==========================================================================
def draw_system_logic():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_aspect("equal")

    # 背景层
    layer_cfg = [
        (0.4, 6.0, 11.2, 1.2, "用户输入层",     "#E3F2FD", "#1565C0"),
        (0.4, 4.4, 11.2, 1.2, "分析与识别层",   "#FFF8E1", "#F57F17"),
        (0.4, 2.8, 11.2, 1.2, "策略匹配层",     "#F3E5F5", "#6A1B9A"),
        (0.4, 1.2, 11.2, 1.2, "输出与反馈层",   "#E8F5E9", "#2E7D32"),
    ]
    for x, y, w, h, title, fc, tc in layer_cfg:
        layer_box = mpatches.FancyBboxPatch((x, y), w, h,
                                            boxstyle="round,pad=0.08",
                                            fc=fc, ec=tc, lw=2, alpha=0.95, zorder=1)
        ax.add_patch(layer_box)
        ax.text(x + 0.25, y + h - 0.18, title, fontsize=12, color=tc,
                fontweight="bold", va="top")

    # 输入层内容
    inputs = [
        (1.8, 6.2, 2.2, 0.55, "用户输入文本",   "#BBDEFB", "#0D47A1"),
        (5.0, 6.2, 2.2, 0.55, "历史对话上下文", "#BBDEFB", "#0D47A1"),
        (8.2, 6.2, 2.2, 0.55, "会话状态",       "#BBDEFB", "#0D47A1"),
    ]
    for bx, by, bw, bh, t, fc, ec in inputs:
        _round_box(ax, (bx, by), bw, bh, t, fc, ec, fontsize=10)

    # 分析层内容
    analyses = [
        (1.2, 4.6, 2.4, 0.6, "情绪极性分类",   "#FFE082", "#E65100"),
        (4.2, 4.6, 2.4, 0.6, "求助信号识别",   "#FFE082", "#E65100"),
        (7.2, 4.6, 2.4, 0.6, "安全风险检测",   "#FFE082", "#E65100"),
    ]
    for bx, by, bw, bh, t, fc, ec in analyses:
        _round_box(ax, (bx, by), bw, bh, t, fc, ec, fontsize=10)

    # 情感强度标注
    _round_box(ax, (10.2, 4.6), 1.2, 0.6, "情感强度\n评分", "#FFE082", "#E65100", fontsize=9)

    # 策略层内容
    strategies = [
        (1.2, 3.0, 2.4, 0.6, "规则库策略匹配", "#E1BEE7", "#4A148C"),
        (4.2, 3.0, 2.4, 0.6, "关键词种子库",   "#E1BEE7", "#4A148C"),
        (7.2, 3.0, 2.4, 0.6, "安全危机转介",   "#E1BEE7", "#4A148C"),
    ]
    for bx, by, bw, bh, t, fc, ec in strategies:
        _round_box(ax, (bx, by), bw, bh, t, fc, ec, fontsize=10)

    # 输出层内容
    outputs = [
        (1.2, 1.4, 2.4, 0.6, "预设共情话术",   "#C8E6C9", "#1B5E20"),
        (4.2, 1.4, 2.4, 0.6, "LLM 共情回应",  "#C8E6C9", "#1B5E20"),
        (7.2, 1.4, 2.4, 0.6, "风险警告提示",   "#C8E6C9", "#1B5E20"),
    ]
    for bx, by, bw, bh, t, fc, ec in outputs:
        _round_box(ax, (bx, by), bw, bh, t, fc, ec, fontsize=10)

    # 层间箭头
    for cx in [2.9, 6.1, 9.4]:
        _arrow(ax, (cx, 6.2), (cx, 5.5), "#1565C0")
        _arrow(ax, (cx, 4.6), (cx, 3.8), "#F57F17")
        _arrow(ax, (cx, 3.0), (cx, 2.2), "#6A1B9A")

    # 标题
    ax.text(6.0, 7.15, "系统逻辑结构图（分层架构）",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#212121")

    fig.tight_layout()
    return fig


# ==========================================================================
# 图18：交互流程图（端到端路径）
# ==========================================================================
def draw_interaction_flow():
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_aspect("equal")

    # 节点定义 (x, y, w, h, text, fill, edge)
    # 统一高度 0.6
    nodes = {
        "start":   (4.5, 8.0, 2.0, 0.6, "用户输入文本",       "#E3F2FD", "#1565C0"),
        "clean":   (4.5, 6.8, 2.0, 0.6, "文本清洗（去噪/分词）", "#E3F2FD", "#1565C0"),
        "kw":      (4.5, 5.6, 2.0, 0.6, "关键词提取与统计",   "#FFF8E1", "#F57F17"),
        "sent":    (1.8, 4.4, 2.0, 0.6, "情绪极性分类",       "#FFF8E1", "#F57F17"),
        "help":    (4.5, 4.4, 2.0, 0.6, "求助信号识别",       "#FFF8E1", "#F57F17"),
        "risk":    (7.2, 4.4, 2.0, 0.6, "安全风险检测",       "#FFF8E1", "#F57F17"),
        "intent":  (4.5, 3.4, 2.0, 0.6, "综合意图判定",       "#F3E5F5", "#6A1B9A"),
        # 风险分支
        "risk_yes": (8.4, 3.4, 2.0, 0.6, "风险信号触发",    "#FFCDD2", "#C62828"),
        "risk_warn": (8.4, 2.4, 2.0, 0.6, "输出风险警告\n+ 转介热线", "#FFCDD2", "#C62828"),
        # 策略分支
        "rule":    (2.0, 2.4, 2.0, 0.6, "规则库策略匹配",     "#F3E5F5", "#6A1B9A"),
        "llm":     (4.5, 2.4, 2.0, 0.6, "LLM 共情生成",      "#F3E5F5", "#6A1B9A"),
        "fallback": (7.0, 2.4, 2.0, 0.6, "预设话术回退",       "#F3E5F5", "#6A1B9A"),
        # 输出
        "out":     (4.5, 1.2, 2.0, 0.6, "输出建议话术",       "#E8F5E9", "#2E7D32"),
        "end":     (4.5, 0.0, 2.0, 0.6, "记录历史 / 结束",    "#E8F5E9", "#2E7D32"),
    }
    for k, (x, y, w, h, t, fc, ec) in nodes.items():
        _round_box(ax, (x, y), w, h, t, fc, ec, fontsize=9, bold=(k == "end"))

    # 主路径箭头
    _arrow(ax, (5.5, 8.0), (5.5, 7.7))
    _arrow(ax, (5.5, 6.8), (5.5, 6.5))
    _arrow(ax, (5.5, 5.6), (5.5, 5.3))
    # 分叉三路
    _arrow(ax, (5.5, 5.6), (2.8, 5.3), color="#F57F17")
    _arrow(ax, (5.5, 5.6), (5.5, 5.3), color="#F57F17")
    _arrow(ax, (5.5, 5.6), (8.2, 5.3), color="#F57F17")
    _arrow(ax, (2.8, 4.4), (5.5, 3.9), color="#F57F17")
    _arrow(ax, (5.5, 4.4), (5.5, 3.9), color="#F57F17")
    _arrow(ax, (8.2, 4.4), (5.5, 3.9), color="#F57F17")

    # 风险分支
    _arrow(ax, (8.2, 4.4), (9.4, 4.1), color="#C62828", lw=2)
    _arrow(ax, (9.4, 3.4), (9.4, 3.1), color="#C62828", lw=2)

    # 策略分支（风险/非风险）
    _arrow(ax, (4.5, 3.4), (3.0, 3.0), color="#6A1B9A")  # → 规则
    _arrow(ax, (4.5, 3.4), (5.5, 3.0), color="#6A1B9A")  # → LLM
    _arrow(ax, (4.5, 3.4), (8.0, 3.0), color="#6A1B9A")  # → 回退

    # 策略→输出
    _arrow(ax, (3.0, 2.4), (5.5, 1.9), color="#2E7D32")
    _arrow(ax, (5.5, 2.4), (5.5, 1.9), color="#2E7D32")
    _arrow(ax, (9.4, 2.4), (5.5, 1.9), color="#2E7D32", style="-|>", lw=2)
    _arrow(ax, (5.5, 1.2), (5.5, 0.7))

    # 条件标注
    ax.text(9.6, 3.95, "风险=是", fontsize=8, color="#C62828",
            ha="left", style="italic")
    ax.text(3.0, 3.05, "规则命中", fontsize=8, color="#6A1B9A",
            ha="center", style="italic")
    ax.text(5.5, 3.05, "LLM 增强", fontsize=8, color="#6A1B9A",
            ha="center", style="italic")
    ax.text(8.0, 3.05, "规则为空", fontsize=8, color="#6A1B9A",
            ha="center", style="italic")
    ax.text(9.6, 2.95, "安全警告", fontsize=8, color="#C62828",
            ha="left", style="italic")

    # 标题
    ax.text(5.5, 8.65, "交互流程图（端到端处理路径）",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#212121")

    fig.tight_layout()
    return fig


def main():
    import os
    for fname, draw_fn in [
        ("fig17_system_logic.png",    draw_system_logic),
        ("fig18_interaction_flow.png", draw_interaction_flow),
    ]:
        fig = draw_fn()
        path = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[导出] {path}")
    print("系统逻辑图 & 交互流程图生成完毕")


if __name__ == "__main__":
    main()
