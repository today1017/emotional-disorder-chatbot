# -*- coding: utf-8 -*-
"""
llm.py —— LLM 共情回应生成（与 Streamlit UI 解耦，便于独立测试）
对接智谱开放平台 GLM（OpenAI 兼容接口 /api/paas/v4/chat/completions），
也可在侧边栏修改接口地址/模型，切换任意 OpenAI 兼容服务。
说明：仅使用标准库 urllib，无需额外安装 requests/openai SDK。
"""
import json
import os
import urllib.error
import urllib.request

# ------------------------------ 默认配置 ------------------------------ #
# 默认使用智谱开放平台（GLM-4-Flash 长期免费）
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4-flash"
ENV_KEY_NAME = "ZHIPU_API_KEY"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2


def get_api_key():
    """\u4ece\u73af\u5883\u53d8\u91cf\u6216 Streamlit Secrets \u83b7\u53d6 API Key
       \u4f18\u5148\u987a\u5e8f: st.secrets > ZHIPU_API_KEY > SILICONFLOW_API_KEY > OPENAI_API_KEY
    """
    # 1. \u5c1d\u8bd5\u4ece Streamlit Secrets \u83b7\u53d6\uff08Streamlit Cloud\uff09
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets:
            for key_name in (ENV_KEY_NAME, "SILICONFLOW_API_KEY", "OPENAI_API_KEY"):
                if key_name in st.secrets:
                    val = st.secrets[key_name]
                    if val and val.strip():
                        return val.strip()
    except Exception:
        pass
    
    # 2. \u5c1d\u8bd5\u4ece\u73af\u5883\u53d8\u91cf\u83b7\u53d6
    key = (os.environ.get(ENV_KEY_NAME) or
           os.environ.get("SILICONFLOW_API_KEY") or
           os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    
    # 3. \u4f7f\u7528\u786c\u7f16\u7801\u7684\u9ed8\u8ba4Key
    default_key = "50010b60f8f34be7873400365e680f0c.wD9dxw4eYyIM4kLc"
    if default_key:
        return default_key
    return ""


def build_prompt(user_text, rule_result):
    """
    将规则模型判定结果 + 用户原始文本拼接为 LLM 提示词。
    要求 LLM 生成一段 100-150 字的中文共情回应。
    rule_result 结构见 rules.analyze() 返回值。
    """
    sent = rule_result["情绪极性"]["label"]
    help_flag = rule_result["求助信号"]["label"]
    risk_flag = rule_result["安全风险"]["label"]

    # 安全风险提示词动态追加（避免 AI 忽略危机情况）
    risk_note = (
        "用户当前可能存在自伤/自杀风险，回应的开头和结尾都要传递关怀，"
        "并明确建议其联系专业支持（全国心理援助热线 12356）或身边信任的人。"
        if risk_flag == "是" else
        "无需特别强调危机资源，专注共情与陪伴。"
    )

    prompt = (
        "你是用户的好朋友，像微信上聊天一样自然、真诚、温暖，服务于「情感障碍人群语言特征分析」"
        "科研演示系统。请根据用户的发言和系统判定结果，生成一段真诚的回应。\n\n"
        "【用户发言】\n"
        f'"{user_text}"\n\n'
        "【系统规则模型判定结果】\n"
        f"- 情绪极性：{sent}\n"
        f"- 求助信号：{help_flag}（是否主动求助/隐含求助）\n"
        f"- 安全风险：{risk_flag}（是否存在自伤/自杀风险信号）\n\n"
        "【生成要求】\n"
        "1. 以温暖、真诚、不评判的语气共情，并具体呼应发言中提到的内容；\n"
        "2. 长度控制在 100-150 个汉字；\n"
        "3. 若求助信号为“是”，主动询问困扰细节或提供可落地的支持；\n"
        "4. " + risk_note + "\n"
        "5. 不要提及“系统判定/规则模型”等字样，不要下医学诊断，不要说教；\n"
        "6. 直接输出回应正文，不要任何前缀、解释、引号或编号。"
    )
    return prompt


def build_multi_turn_prompt(user_text, rule_result, conversation_history=None,
                            context_summary=None):
    sent = rule_result["情绪极性"]["label"]
    help_flag = rule_result["求助信号"]["label"]
    risk_flag = rule_result["安全风险"]["label"]

    risk_note = (
        "用户当前可能存在自伤/自杀风险，回复的开头和结尾要表达关切，"
        "明确引导求助专业支持，全国心理救援热线 12356，或建议联系身边的人。"
        if risk_flag == "是" else
        "没有特别强调危险，可以专注偶聊和闪开转移。"
    )

    history_block = ""
    if conversation_history:
        history_lines = []
        for role, msg_content in conversation_history[-12:]:
            speaker = "用户" if role == "user" else "助手"
            history_lines.append(f"{speaker}：{msg_content}")
        history_block = "以下是之前的对话历史：" + chr(10) + chr(10).join(history_lines) + chr(10) + chr(10)

    context_block = ""
    if context_summary:
        context_block = "对话上下文摘要：" + chr(10) + context_summary + chr(10) + chr(10)

    user_line = chr(10) + "【用户最新发言】" + chr(10) + '"' + user_text + '"' + chr(10) + chr(10)
    sys_line = "【系统分析（内部参考）】" + chr(10) + "- 情绪：" + sent + chr(10) + "- 求助：" + help_flag + chr(10) + "- 风险：" + risk_flag + chr(10) + chr(10)

    rules = "【回复要求】" + chr(10)
    rules += "1. 先共情再引导：先回应用户的情绪，再给实用建议；" + chr(10)
    rules += "2. 必须承接用户之前说的内容，不要转移话题；" + chr(10)
    rules += "3. 60-120字，不要太长；" + chr(10)
    rules += "4. 口语词偶尔用，但每次回复开头不要一样，不要总用同一个语气词开头，正常说话就行；" + chr(10)
    rules += "5. " + risk_note + chr(10)
    rules += "6. 不提系统判断，不像医生问诊，不说教；" + chr(10)
    rules += "7. 直接输出回复，不要任何前缀、编号、引号。"

    prompt = (
        "你是一位温暖、真诚的朋友，正在微信上和用户聊天。"
        "用户可能正在经历情绪困扰，你需要先偶聊、再引导，像真正的朋友一样陪伴。"
        "现在正在进行一场多轮对话，你必须仔细参考之前的对话历史，记住用户说过的每一句话。"
        + chr(10) + chr(10)
        + history_block
        + context_block
        + user_line
        + sys_line
        + rules
    )
    return prompt


def call_llm(prompt, api_key, base_url=DEFAULT_BASE_URL,
             model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT):
    import time

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是温暖专业的心理援助共情回应助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "max_tokens": 300,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            return text or None
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return None
    return None


def generate_empathic_reply(user_text, rule_result, api_key=None,
                            base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                            timeout=DEFAULT_TIMEOUT):
    """一站式入口：拼接 prompt → 调用 LLM → 返回回应文本；失败返回 None"""
    if not api_key:
        return None
    prompt = build_prompt(user_text, rule_result)
    return call_llm(prompt, api_key=api_key, base_url=base_url,
                    model=model, timeout=timeout)


def generate_multi_turn_reply(user_text, rule_result, api_key=None,
                              conversation_history=None, context_summary=None,
                              base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                              timeout=DEFAULT_TIMEOUT):
    """多轮对话版一站式入口：支持历史对话与上下文摘要"""
    if not api_key:
        return None
    prompt = build_multi_turn_prompt(user_text, rule_result,
                                     conversation_history, context_summary)
    return call_llm(prompt, api_key=api_key, base_url=base_url,
                    model=model, timeout=timeout)


# ------------------------------ 自测 ------------------------------ #
if __name__ == "__main__":
    import rules

    demo_text = "最近真的好崩溃，什么都不想干，有时候甚至想死，活着太没意思了。"
    result = rules.analyze(demo_text)
    print(build_prompt(demo_text, result))

    key = get_api_key()
    if not key:
        print("\n[自测] 未检测到 API Key（环境变量），跳过真实调用，验证失败回退路径...")
        print("generate_empathic_reply 返回:", generate_empathic_reply(demo_text, result))
    else:
        print("\n[自测] 检测到 API Key，正在调用 SiliconFlow...")
        reply = generate_empathic_reply(demo_text, result, api_key=key)
        print("AI 回应:", reply)
