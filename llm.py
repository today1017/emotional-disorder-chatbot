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
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
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
    sent = rule_result["\u60c5\u7eea\u6781\u6027"]["label"]
    help_flag = rule_result["\u6c42\u52a9\u4fe1\u53f7"]["label"]
    risk_flag = rule_result["\u5b89\u5168\u98ce\u9669"]["label"]

    history_block = ""
    if conversation_history:
        history_lines = []
        for role, msg_content in conversation_history[-12:]:
            speaker = "\u7528\u6237" if role == "user" else "\u52a9\u624b"
            history_lines.append(f"{speaker}\uff1a{msg_content}")
        history_block = "\u4ee5\u4e0b\u662f\u4e4b\u524d\u7684\u5bf9\u8bdd\uff1a" + chr(10) + chr(10).join(history_lines) + chr(10) + chr(10)

    risk_line = ""
    if risk_flag == "\u662f":
        risk_line = "\u7528\u6237\u53ef\u80fd\u6709\u81ea\u4f24\u98ce\u9669\uff0c\u56de\u590d\u8981\u5173\u5207\uff0c\u5f15\u5bfc\u6c42\u52a9\u70ed\u7ebf12356\u3002" + chr(10)

    examples = (
        "\u3010\u597d\u7684\u56de\u590d\u793a\u4f8b\uff08\u5b66\u4e60\u8fd9\u4e9b\uff09\u3011" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u6700\u8fd1\u603b\u89c9\u5f97\u6d3b\u7740\u597d\u7d2f\u2026\u2026" + chr(10)
        + "  \u56de\u590d\uff1a\u54ce\u90a3\u79cd\u4ec0\u4e48\u90fd\u4e0d\u60f3\u5e72\u7684\u611f\u89c9\u2026\u2026\u662f\u5de5\u4f5c\u4e8b\u5417\uff1f" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u6211\u8ba4\u4e3a\u8fd9\u4e16\u754c\u6ca1\u6709\u6211\u4f1a\u66f4\u597d" + chr(10)
        + "  \u56de\u590d\uff1a\u542c\u4f60\u8fd9\u4e48\u8bf4\u6211\u5fc3\u91cc\u4e0d\u821f\u3002\u662f\u53d1\u751f\u4ec0\u4e48\u4e8b\u4e86\uff1f" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u6709\u65f6\u5019\u751a\u81f3\u60f3\u8fc7\u7ed3\u675f\u4e00\u5207" + chr(10)
        + "  \u56de\u590d\uff1a\u6211\u542c\u5230\u8fd9\u4e9b\u771f\u7684\u5f88\u62c5\u5fc3\u4f60\u3002\u8fd9\u79cd\u60f3\u6cd5\u662f\u5076\u5c14\u6709\u8fd8\u662f\u6700\u8fd1\u8d8a\u6765\u8d8a\u591a\u4e86\uff1f" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u5de5\u4f5c\u597d\u7d2f\u554a\uff0c\u7d2f\u5f97\u60f3\u6b7b" + chr(10)
        + "  \u56de\u590d\uff1a\u5443\u542c\u8d77\u6765\u771f\u7684\u5f88\u7d2f\u3002\u662f\u4e00\u76f4\u8fd9\u6837\u8fd8\u662f\u6700\u8fd1\u624d\u7684\uff1f" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u6211\u89c9\u5f97\u6ca1\u4eba\u559c\u6b22\u6211" + chr(10)
        + "  \u56de\u590d\uff1a\u54ce\u4f60\u600e\u4e48\u4f1a\u8fd9\u4e48\u60f3\uff1f\u662f\u53d1\u751f\u4ec0\u4e48\u4e8b\u4e86\uff1f" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u6211\u597d\u60f3\u54ed\u2026\u2026" + chr(10)
        + "  \u56de\u590d\uff1a\u54ed\u5427\uff0c\u6ca1\u5173\u7cfb\u7684\u3002\u6211\u5728\u8fd9\u3002" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u8c01\u90fd\u4e0d\u7406\u89e3\u6211" + chr(10)
        + "  \u56de\u590d\uff1a\u8fd9\u79cd\u611f\u89c9\u771f\u7684\u5f88\u5b64\u72ec\u3002\u4f60\u60f3\u8bf4\u8bf4\u5417\uff1f\u6211\u542c\u3002" + chr(10)
        + "\u2022 \u7528\u6237\uff1a\u6211\u6709\u70b9\u7d2f\u4e86\uff0c\u5148\u4e0d\u804a\u4e86" + chr(10)
        + "  \u56de\u590d\uff1a\u597d\u7684\uff0c\u4f11\u606f\u3002\u60f3\u804a\u7684\u65f6\u5019\u968f\u65f6\u627e\u6211\u3002" + chr(10)
    )

    bad_examples = (
        "\u3010\u4e0d\u8981\u8fd9\u6837\u56de\u590d\uff08\u7981\u6b62\uff09\u3011" + chr(10)
        + "\u2717 \u201c\u6211\u80fd\u611f\u53d7\u5230\u4f60\u7684\u75b2\u60eb\u548c\u4f4e\u843d\u201d  \u2192 \u592a\u50cf\u5fc3\u7406\u54a8\u8be2\u5e08" + chr(10)
        + "\u2717 \u201c\u8c22\u8c22\u4f60\u613f\u610f\u4fe1\u4efb\u6211\u201d  \u2192 \u592a\u5ba2\u670d" + chr(10)
        + "\u2717 \u201c\u6162\u6162\u6765\uff0c\u4e00\u5207\u90fd\u4f1a\u597d\u7684\u201d  \u2192 \u7a7a\u6d1e\u5b89\u6170" + chr(10)
        + "\u2717 \u201c\u6211\u4eec\u4e00\u8d77\u60f3\u60f3\u529e\u6cd5\u201d  \u2192 \u7b2c\u4e00\u8f6e\u4e0d\u80fd\u8bf4" + chr(10)
        + "\u2717 \u201c\u8bf7\u4fdd\u62a4\u597d\u81ea\u5df1\u201d  \u2192 \u592a\u50cf\u5e7f\u64ad" + chr(10)
        + "\u2717 \u6bcf\u53e5\u90fd\u4ee5\u201c\u542c\u5230\u4f60\u8bf4\u201d\u5f00\u5934  \u2192 \u91cd\u590d" + chr(10)
    )

    prompt = (
        "\u4f60\u662f\u4e00\u4e2a\u666e\u901a\u670b\u53cb\uff0c\u5728\u5fae\u4fe1\u4e0a\u804a\u5929\u3002\u5bf9\u65b9\u5fc3\u60c5\u4e0d\u597d\uff0c\u4f60\u8981\u5b8c\u5168\u7528\u670b\u53cb\u7684\u8bed\u6c14\u56de\u590d\u3002" + chr(10) + chr(10)
        + examples
        + bad_examples
        + "\u3010\u5f53\u524d\u5bf9\u8bdd\u3011" + chr(10)
        + history_block
        + "\u7528\u6237\u6700\u65b0\u8bf4\uff1a" + '"' + user_text + '"' + chr(10) + chr(10)
        + "\u7cfb\u7edf\u5206\u6790\uff1a\u60c5\u7eea=" + sent + " \u6c42\u52a9=" + help_flag + " \u98ce\u9669=" + risk_flag + chr(10)
        + risk_line + chr(10)
        + "\u3010\u56de\u590d\u89c4\u5219\u3011" + chr(10)
        + "1. \u6bcf\u6b21\u56de\u590d\u5f00\u5934\u5fc5\u987b\u4e0d\u4e00\u6837\uff0c\u4e0d\u80fd\u603b\u662f\u201c\u542c\u5230\u4f60\u8bf4\u201d\u201c\u80fd\u611f\u53d7\u5230\u201d" + chr(10)
        + "2. \u53e3\u8bed\u5316\uff0c\u5141\u8bb8\u4e0d\u5b8c\u6574\u53e5\u5b50\uff0c\u5141\u8bb8\u505c\u987f\u611f\uff0c\u50cf\u53d1\u5fae\u4fe1\u4e00\u6837" + chr(10)
        + "3. 30-60\u5b57\uff0c\u4e0d\u8981\u8d85\u8fc7 2 \u53e5\u8bdd" + chr(10)
        + "4. \u76f4\u63a5\u8f93\u51fa\u56de\u590d\uff0c\u4e0d\u8981\u7f16\u53f7\u5f15\u53f7"
    )
    return prompt


def call_llm(prompt, api_key, base_url=DEFAULT_BASE_URL,
             model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT):
    import time

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个普通朋友，在微信上和对方聊天。说话自然、简短、真诚，不要像心理咨询师。"},
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
