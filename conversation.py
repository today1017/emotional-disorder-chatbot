# -*- coding: utf-8 -*-
"""
conversation.py —— 多轮对话状态管理模块
功能：
  1. 对话会话管理（创建/续接/重置）
  2. 上下文感知分析（情感趋势、风险升级、求助信号累积）
  3. 对话历史序列化（支持持久化存储）
数据结构：
  ConversationSession:
    - session_id: str (唯一标识)
    - created_at: datetime
    - messages: List[Message] (对话消息列表)
    - context: DialogContext (上下文聚合信息)
  Message:
    - role: "user" | "assistant" | "system"
    - content: str (原始文本)
    - analysis: dict (规则分析结果，仅user消息)
    - timestamp: datetime
  DialogContext:
    - sentiment_trend: List[float] (情感强度时间序列)
    - risk_history: List[bool] (风险标记历史)
    - help_seeking_count: int (累计求助信号次数)
    - dominant_sentiment: str (主导情绪)
    - escalation_flag: bool (是否触发风险升级)
"""
import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Message:
    """单条对话消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    analysis: Optional[Dict[str, Any]] = None  # 用户消息的分析结果
    timestamp: dt.datetime = field(default_factory=dt.datetime.now)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "analysis": self.analysis,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DialogContext:
    """对话上下文聚合信息（用于上下文感知分析）"""
    sentiment_trend: List[float] = field(default_factory=list)
    risk_history: List[bool] = field(default_factory=list)
    help_seeking_count: int = 0
    negative_streak: int = 0  # 连续消极轮次
    positive_streak: int = 0  # 连续积极轮次
    dominant_sentiment: str = "中性"
    escalation_flag: bool = False  # 风险升级标志
    total_turns: int = 0
    _sentiment_labels: List[str] = field(default_factory=list, repr=False)

    def update(self, analysis: Dict[str, Any]):
        """根据最新分析结果更新上下文"""
        self.total_turns += 1

        sentiment = analysis.get("情绪极性", {}).get("label", "中性")
        intensity = analysis.get("情感强度", {}).get("强度", 50)
        risk = analysis.get("安全风险", {}).get("label") == "是"
        help_flag = analysis.get("求助信号", {}).get("label") == "是"

        self.sentiment_trend.append(intensity)
        self.risk_history.append(risk)

        if help_flag:
            self.help_seeking_count += 1

        if sentiment == "消极":
            self.negative_streak += 1
            self.positive_streak = 0
        elif sentiment == "积极":
            self.positive_streak += 1
            self.negative_streak = 0
        else:
            self.negative_streak = 0
            self.positive_streak = 0

        # 主导情绪：基于最近5轮情绪标签加权投票（近几轮权重大）
        self._sentiment_labels.append(sentiment)
        recent_labels = self._sentiment_labels[-5:]
        weights = {label: sum(0.6 ** (len(recent_labels) - 1 - i)
                              for i, l in enumerate(recent_labels) if l == label)
                   for label in set(recent_labels)}
        self.dominant_sentiment = max(weights, key=weights.get)

        if len(self.risk_history) >= 2:
            if not self.risk_history[-2] and self.risk_history[-1]:
                self.escalation_flag = True

    def get_context_summary(self) -> str:
        """生成上下文摘要文本（用于拼接到LLM prompt）"""
        parts = []
        parts.append(f"当前为第 {self.total_turns} 轮对话")
        parts.append(f"用户主导情绪：{self.dominant_sentiment}")

        if self.negative_streak >= 3:
            parts.append(f"⚠️ 用户已连续 {self.negative_streak} 轮表达消极情绪")
        elif self.positive_streak >= 2:
            parts.append(f"✓ 用户近 {self.positive_streak} 轮情绪趋于积极")

        if self.help_seeking_count > 0:
            parts.append(f"累计 {self.help_seeking_count} 次求助信号")

        if self.escalation_flag:
            parts.append("🔴 本轮首次检测到安全风险，需重点关注")

        if len(self.sentiment_trend) >= 2:
            trend = self.sentiment_trend[-1] - self.sentiment_trend[-2]
            if trend > 10:
                parts.append("情感强度明显上升（情绪加剧）")
            elif trend < -10:
                parts.append("情感强度明显下降（情绪缓和）")

        return "；".join(parts)


@dataclass
class ConversationSession:
    """完整对话会话"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: dt.datetime = field(default_factory=dt.datetime.now)
    messages: List[Message] = field(default_factory=list)
    context: DialogContext = field(default_factory=DialogContext)

    def add_user_message(self, content: str, analysis: Dict[str, Any]) -> Message:
        msg = Message(role="user", content=content, analysis=analysis)
        self.messages.append(msg)
        self.context.update(analysis)
        return msg

    def add_assistant_message(self, content: str) -> Message:
        msg = Message(role="assistant", content=content)
        self.messages.append(msg)
        return msg

    def add_system_message(self, content: str) -> Message:
        msg = Message(role="system", content=content)
        self.messages.append(msg)
        return msg

    def get_recent_messages(self, n: int = 6) -> List[Message]:
        """获取最近n条消息（用于LLM上下文窗口）"""
        return self.messages[-n:] if len(self.messages) > n else self.messages

    def get_conversation_history_for_llm(self) -> List[Dict[str, str]]:
        """转换为OpenAI格式的messages列表"""
        result = []
        for msg in self.get_recent_messages(8):
            if msg.role in ("user", "assistant"):
                result.append({"role": msg.role, "content": msg.content})
        return result

    def reset(self):
        """重置会话（保留session_id）"""
        self.messages.clear()
        self.context = DialogContext()
        self.created_at = dt.datetime.now()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "context": {
                "sentiment_trend": self.context.sentiment_trend,
                "risk_history": self.context.risk_history,
                "help_seeking_count": self.context.help_seeking_count,
                "negative_streak": self.context.negative_streak,
                "positive_streak": self.context.positive_streak,
                "dominant_sentiment": self.context.dominant_sentiment,
                "escalation_flag": self.context.escalation_flag,
                "total_turns": self.context.total_turns,
            },
        }


def create_session() -> ConversationSession:
    """创建新会话工厂函数"""
    return ConversationSession()


def format_chat_display(messages: List[Message], max_preview_len: int = 80) -> List[dict]:
    """格式化消息用于聊天界面展示"""
    display_msgs = []
    for msg in messages:
        preview = msg.content[:max_preview_len]
        if len(msg.content) > max_preview_len:
            preview += "..."
        display_msgs.append({
            "role": msg.role,
            "content": preview,
            "full_content": msg.content,
            "analysis": msg.analysis,
            "timestamp": msg.timestamp.strftime("%H:%M:%S"),
        })
    return display_msgs
