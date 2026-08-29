# -*- coding: utf-8 -*-
"""
app.py —— 情感障碍人群语言特征分析 · Streamlit 多轮对话演示原型
运行：  streamlit run app.py
功能：
  1. 多轮对话：连续输入多条消息，AI 基于历史上下文生成连贯共情回应
  2. 每轮实时规则分析：情绪极性 / 求助信号 / 安全风险 / 情感强度
  3. 上下文感知：情感趋势折线图、风险升级检测、连续消极提示、短回应情绪继承
  4. 历史会话管理：可新建会话 / 重置当前会话
  5. LLM 多轮增强：拼接最近对话历史 + 上下文摘要进 prompt（失败回退预设话术）
"""
import datetime as dt
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st

import conversation
import llm
import rules
import utils

# ------------------------------ 页面配置 ------------------------------ #
st.set_page_config(
    page_title="情感语言特征分析 · 多轮对话原型",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 统一使用 utils.setup_matplotlib() 配置中文字体（含云端自带字体）
plt = utils.setup_matplotlib()

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      .chat-user { background:#D6EAF8; border-radius:14px; padding:10px 14px;
                   margin:6px 0; text-align:right; border:1px solid #AED6F1;
                   color:#1B2631; }
      .chat-ai   { background:#EAECEE; border-radius:14px; padding:10px 14px;
                   margin:6px 0; border:1px solid #D5D8DC;
                   color:#1B2631; }
      .chat-meta { font-size:11px; color:#7F8C8D; margin-top:4px; }
      .chat-tags { margin-top:6px; }
      .badge { display:inline-block; border-radius:999px; padding:2px 10px;
               font-size:12px; margin-right:6px; border:1px solid; }
      .risk-flag { background:#FFEBEE; color:#C62828; border-color:#EF9A9A; }
      .ok-flag   { background:#E8F5E9; color:#2E7D32; border-color:#A5D6A7; }
      .warn-flag { background:#FFF3E0; color:#E65100; border-color:#FFCC80; }
      .ctx-hint  { font-size:12px; color:#6A1B9A; background:#F3E5F5;
                   border-radius:8px; padding:6px 10px; margin:4px 0; }
      .card-sentiment { border-radius:10px; padding:10px 14px; margin-bottom:8px;
                        box-shadow:0 1px 3px rgba(0,0,0,.06); }
      @media (prefers-color-scheme: dark) {
        .chat-user { background:#1B3A5C; border-color:#2E6DA4; color:#ECF0F1; }
        .chat-ai   { background:#2C3E50; border-color:#34495E; color:#ECF0F1; }
        .chat-meta { color:#ABB2B9; }
        .ctx-hint  { background:#4A235A; color:#D2B4DE; }
        .badge { color:#ECF0F1; }
      }
      [data-theme="dark"] .chat-user { background:#1B3A5C; border-color:#2E6DA4; color:#ECF0F1; }
      [data-theme="dark"] .chat-ai   { background:#2C3E50; border-color:#34495E; color:#ECF0F1; }
      [data-theme="dark"] .chat-meta { color:#ABB2B9; }
      [data-theme="dark"] .ctx-hint { background:#4A235A; color:#D2B4DE; }
      [data-theme="dark"] .badge { color:#ECF0F1; }
    </style>
    """,
    unsafe_allow_html=True,
)

SENTIMENT_STYLE = {
    "积极": ("#E7F5E8", "#2E7D32", "😊"),
    "中性": ("#F2F2F2", "#5F6368", "😐"),
    "消极": ("#FDEBEC", "#C62828", "😔"),
}

# ------------------------------ 会话状态初始化 ------------------------------ #
if "session" not in st.session_state:
    st.session_state.session = conversation.create_session()
if "enable_llm" not in st.session_state:
    st.session_state.enable_llm = True
if "api_key" not in st.session_state:
    st.session_state.api_key = llm.get_api_key() or "50010b60f8f34be7873400365e680f0c.wD9dxw4eYyIM4kLc"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = llm.DEFAULT_MODEL
if "llm_base_url" not in st.session_state:
    st.session_state.llm_base_url = llm.DEFAULT_BASE_URL


def new_session():
    """新建会话（清空上下文）"""
    st.session_state.session = conversation.create_session()


def process_input(text):
    """核心分析处理：规则分析 + LLM 回应 + 写入会话（供发送按钮和示例按钮共用）"""
    sess = st.session_state.session
    text = (text or "").strip()
    if not text:
        return

    with st.spinner("正在运行规则模型..."):
        result = rules.analyze_with_context(text, sess.context)
    sess.add_user_message(text, result)

    ai_reply = None
    if st.session_state.enable_llm and st.session_state.api_key:
        with st.spinner("AI 正在基于对话历史生成回应..."):
            history = [(m.role, m.content)
                       for m in sess.messages
                       if m.role in ("user", "assistant") and m.content != text]
            history = history[-6:]
            context_summary = sess.context.get_context_summary()
            ai_reply = llm.generate_multi_turn_reply(
                text, result,
                api_key=st.session_state.api_key,
                conversation_history=history,
                context_summary=context_summary,
                base_url=st.session_state.llm_base_url,
                model=st.session_state.llm_model,
            )
    sess.add_assistant_message(ai_reply or result["建议话术"])
    st.session_state.user_input = ""


def on_send_click():
    """发送按钮回调"""
    text = st.session_state.get("user_input") or ""
    process_input(text)


def fill_demo(text):
    """示例场景按钮回调：直接触发分析"""
    st.session_state.user_input = text
    process_input(text)


# ==========================================================================
# 侧边栏：LLM 设置 + 当前轮分析卡片
# ==========================================================================
with st.sidebar:
    st.header("⚙️ 控制面板")

    with st.expander("🤖 LLM 设置（智谱 GLM）", expanded=False):
        st.checkbox("启用 AI 回应", key="enable_llm")
        st.text_input("API Key", type="password", key="api_key", value=st.session_state.api_key,
                      help=f"环境变量 {llm.ENV_KEY_NAME} 或此处填写")
        st.text_input("模型", key="llm_model")
        st.text_input("接口地址", key="llm_base_url")
        if not st.session_state.api_key:
            st.caption("未配置 Key，将回退预设话术。")

    st.button("🔄 新建会话", on_click=new_session, use_container_width=True)

    # ---- 会话导出/情绪报告 ----
    st.divider()
    st.subheader("📤 导出会话记录")
    
    def export_session_json():
        """导出会话为 JSON"""
        return st.session_state.session.to_dict()
    
    def export_emotion_report():
        """生成情绪分析报告（Markdown 格式）"""
        sess = st.session_state.session
        ctx = sess.context
        user_msgs = [m for m in sess.messages if m.role == "user"]
        
        lines = [
            f"# 情感障碍对话情绪分析报告",
            f"",
            f"## 基本信息",
            f"- 生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 会话 ID：{sess.session_id}",
            f"- 对话轮次：{ctx.total_turns}",
            f"- 用户发言数：{len(user_msgs)}",
            f"- 求助信号触发：{ctx.help_seeking_count} 次",
            f"- 主导情绪：{ctx.dominant_sentiment}",
            f"- 连续消极轮次：{ctx.negative_streak}",
            f"- 风险升级标记：{'是' if ctx.escalation_flag else '否'}",
            f"",
            f"## 情感强度趋势",
        ]
        
        if ctx.sentiment_trend:
            lines.append(f"- 情感强度序列：{ctx.sentiment_trend}")
            lines.append(f"- 平均强度：{np.mean(ctx.sentiment_trend):.1f}")
            lines.append(f"- 最高强度：{max(ctx.sentiment_trend)}")
            lines.append(f"- 最低强度：{min(ctx.sentiment_trend)}")
            risk_turns = [i+1 for i, r in enumerate(ctx.risk_history) if r]
            if risk_turns:
                lines.append(f"- ⚠️ 风险轮次：{risk_turns}")
        
        lines.append(f"")
        lines.append(f"## 逐轮分析明细")
        
        for i, msg in enumerate(user_msgs, 1):
            r = msg.analysis
            lines.append(f"")
            lines.append(f"### 第 {i} 轮 ({msg.timestamp.strftime('%H:%M:%S')})")
            lines.append(f"**用户输入**：{msg.content}")
            lines.append(f"- 情绪极性：{r['情绪极性']['label']} (命中: {', '.join(r['情绪极性']['matched'][:5]) or '无'})")
            lines.append(f"- 求助信号：{r['求助信号']['label']} (命中: {', '.join(r['求助信号']['matched'][:5]) or '无'})")
            lines.append(f"- 安全风险：{r['安全风险']['label']} (命中: {', '.join(r['安全风险']['matched'][:5]) or '无'})")
            lines.append(f"- 情感强度：{r['情感强度']['强度']} (净得分: {r['情感强度']['净得分']})")
            
            ctx_info = r.get("上下文", {})
            hints = ctx_info.get("上下文提示", [])
            if hints:
                lines.append(f"- 上下文提示：{'; '.join(hints)}")
        
        lines.append(f"")
        lines.append(f"## 总结与建议")
        
        if ctx.escalation_flag:
            lines.append(f"⚠️ **检测到风险升级**，建议立即关注并转介专业支持。")
        if ctx.negative_streak >= 3:
            lines.append(f"⚠️ **连续 {ctx.negative_streak + 1} 轮消极情绪**，建议加强干预频次。")
        if ctx.help_seeking_count > 0:
            lines.append(f"✅ 用户表达了 {ctx.help_seeking_count} 次求助信号，回应及时性至关重要。")
        
        lines.append(f"")
        lines.append(f"> 报告仅供参考，不构成医疗诊断。紧急情况请拨打全国心理援助热线 12356 或 120。")
        
        return "\n".join(lines)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 下载 JSON", use_container_width=True):
            json_data = export_session_json()
            st.download_button(
                label="确认下载 JSON",
                data=json.dumps(json_data, ensure_ascii=False, indent=2),
                file_name=f"session_{st.session_state.session.session_id}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
    with col2:
        if st.button("📄 下载情绪报告", use_container_width=True):
            report_md = export_emotion_report()
            st.download_button(
                label="确认下载报告",
                data=report_md,
                file_name=f"emotion_report_{st.session_state.session.session_id}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ---- 会话概览 ----
    st.divider()
    st.subheader("📊 会话概览")
    sess = st.session_state.session
    ctx = sess.context

    c1, c2, c3 = st.columns(3)
    c1.metric("对话轮次", ctx.total_turns)
    c2.metric("求助信号", f"{ctx.help_seeking_count} 次")
    c3.metric("主导情绪", ctx.dominant_sentiment)

    if ctx.negative_streak >= 2:
        st.warning(f"⚠️ 连续 {ctx.negative_streak + 1} 轮消极情绪")
    if ctx.escalation_flag:
        st.error("🔴 检测到风险升级（由无风险转入风险）")

    # ---- 最新一轮分析卡片 ----
    st.divider()
    st.subheader("🩺 最新规则分析")
    user_msgs = [m for m in sess.messages if m.role == "user"]
    if user_msgs:
        last = user_msgs[-1]
        r = last.analysis
        sent = r["情绪极性"]["label"]
        bg, fg, emoji = SENTIMENT_STYLE.get(sent, SENTIMENT_STYLE["中性"])
        risk_flag = r["安全风险"]["label"]
        help_flag = r["求助信号"]["label"]
        intensity = r["情感强度"]["强度"]

        st.markdown(
            f"""
            <div class="card-sentiment" style="background:{bg};border-left:6px solid {fg};">
              <div style="color:{fg};font-size:12px;font-weight:600;">情绪极性</div>
              <div style="color:{fg};font-size:24px;font-weight:700;">{emoji} {sent}</div>
              <div style="margin-top:4px;">
                <span class="badge {'ok-flag' if risk_flag=='否' else 'risk-flag'}">⚠ 风险：{'是' if risk_flag=='是' else '未检出'}</span>
                <span class="badge {'ok-flag' if help_flag=='否' else 'warn-flag'}">求助：{'是' if help_flag=='是' else '未检出'}</span>
                <span class="badge warn-flag">强度 {intensity}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 上下文提示（情绪修正/风险升级/波动等）
        ctx_info = r.get("上下文", {})
        for hint in ctx_info.get("上下文提示", []):
            st.markdown(f'<div class="ctx-hint">🔍 {hint}</div>', unsafe_allow_html=True)

        # 命中关键词
        with st.expander("🔍 规则命中关键词"):
            for dim in ["情绪极性", "求助信号", "安全风险"]:
                words = r[dim]["matched"]
                st.markdown(f"**{dim}**：{'、'.join(words) if words else '（无命中）'}")
            pos_hits = r["情感强度"]["积极命中"]
            neg_hits = r["情感强度"]["消极命中"]
            net = r["情感强度"]["净得分"]
            st.markdown(f"**情感强度**：净得分 {net}（积极 {sum(pos_hits.values())} / 消极 {sum(neg_hits.values())}）")
    else:
        st.info("暂无分析。输入第一条消息开始对话。")


# ==========================================================================
# 主区域：多轮对话界面
# ==========================================================================
st.title("💬 情感障碍人群语言特征分析 · 多轮对话原型")
st.caption(
    "多轮对话演示：支持上下文记忆、情感趋势追踪、风险升级检测。"
    "规则判断与 AI 回应仅供参考，不构成医疗建议。"
)

# ---- 情感趋势图（置于对话上方） ----
sess = st.session_state.session
ctx = sess.context
if len(ctx.sentiment_trend) >= 2:
    trend_fig, trend_ax = plt.subplots(figsize=(10, 2.6))
    turns = list(range(1, len(ctx.sentiment_trend) + 1))
    vals = ctx.sentiment_trend
    colors = ["#C62828" if r else "#43A047" for r in ctx.risk_history]
    trend_ax.plot(turns, vals, color="#1565C0", marker="o", linewidth=2, zorder=3)
    trend_ax.fill_between(turns, vals, 50, where=[v >= 50 for v in vals],
                          color="#C65B5B", alpha=0.15, zorder=1)
    trend_ax.fill_between(turns, vals, 50, where=[v < 50 for v in vals],
                          color="#6C9E6C", alpha=0.15, zorder=1)
    trend_ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    trend_ax.set_xticks(turns)
    trend_ax.set_ylim(0, 100)
    trend_ax.set_ylabel("情感强度")
    trend_ax.set_title("对话情感强度趋势（红色 = 存在安全风险轮次）", fontsize=12)
    trend_ax.spines["top"].set_visible(False)
    trend_ax.spines["right"].set_visible(False)
    st.pyplot(trend_fig)
    plt.close(trend_fig)

# ---- 对话消息流 ----
chat_container = st.container(border=True, height=460)
with chat_container:
    if not sess.messages:
        st.markdown(
            '<div style="color:#7F8C8D;text-align:center;padding:40px 0;">'
            "👋 你好，我是演示用共情助手。可以和我聊聊你的感受（多轮对话已启用）。"
            "</div>",
            unsafe_allow_html=True,
        )

    for msg in sess.messages:
        if msg.role == "user":
            st.markdown(
                f"""
                <div class="chat-user">
                  <div>{msg.content}</div>
                  <div class="chat-meta">你 · {msg.timestamp.strftime('%H:%M:%S')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # 用户消息下方展示规则标签
            if msg.analysis:
                r = msg.analysis
                sent = r["情绪极性"]["label"]
                risk_flag = r["安全风险"]["label"]
                help_flag = r["求助信号"]["label"]
                tag_risk = f'<span class="badge {"risk-flag" if risk_flag=="是" else "ok-flag"}">风险{"是" if risk_flag=="是" else "否"}</span>'
                tag_help = f'<span class="badge {"warn-flag" if help_flag=="是" else "ok-flag"}">求助{"是" if help_flag=="是" else "否"}</span>'
                tag_sent = f'<span class="badge warn-flag">{sent} · 强度{r["情感强度"]["强度"]}</span>'
                st.markdown(
                    f'<div class="chat-tags">{tag_sent}{tag_help}{tag_risk}</div>',
                    unsafe_allow_html=True,
                )
        elif msg.role == "assistant":
            st.markdown(
                f"""
                <div class="chat-ai">
                  <div>🤖 {msg.content}</div>
                  <div class="chat-meta">AI 助手 · {msg.timestamp.strftime('%H:%M:%S')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---- 底部输入区 ----
st.markdown("---")
input_col, send_col = st.columns([5, 1])
with input_col:
    st.text_area(
        "输入消息",
        key="user_input",
        height=80,
        placeholder="继续输入下一句话，例如：其实我爸妈根本不理解我……",
        label_visibility="collapsed",
    )
with send_col:
    st.button("发送", key="send_btn", type="primary",
              use_container_width=True, on_click=on_send_click)
    st.caption("Enter 换行 / 点击发送")

# 示例场景按钮（演示多轮对话 - 包含成员B访谈场景）
st.markdown("**多轮演示场景**（按顺序点击模拟一段连续对话）：")
demo_cols = st.columns(3)
scenarios = [
    ("① 初始倾诉", "最近总觉得活着好累，什么都不想干，特别绝望……"),
    ("② 情绪加深", "嗯…工作也辞了，天天在家躺着，感觉人生没有希望了。"),
    ("③ 求助信号", "有时候甚至想过结束一切，我该怎么办，能救救我吗？"),
    ("④ 就医就诊", "医生告诉我需要长期服药，每次复诊都很焦虑等待结果，真的好难熬"),
    ("⑤ 家庭压力", "工作压力加上家人的期待让我喘不过气，觉得自己在两头都难顾"),
    ("⑥ 咨询师视角", "患者在治疗第6周出现改善，情绪更稳定但仍有间歇性焦虑发作"),
]
for col, (label, text) in zip(demo_cols, scenarios):
    col.button(label, on_click=fill_demo, args=(text,), use_container_width=True)

st.divider()
st.markdown(
    """
    <div style="color:#9e9e9e;font-size:12px;">
      本应用为大创项目学术研究演示原型，多轮对话由 LLM 生成，仅供参考，
      <strong>不构成任何诊断或医疗建议</strong>。紧急情况请拨打
      <strong>全国心理援助热线 12356</strong> 或 <strong>120</strong>。
    </div>
    """,
    unsafe_allow_html=True,
)
