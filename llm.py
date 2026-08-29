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
    sent = rule_result["\u60c5\u7eea\u6781\u6027"]["label"]
    help_flag = rule_result["\u6c42\u52a9\u4fe1\u53f7"]["label"]
    risk_flag = rule_result["\u5b89\u5168\u98ce\u9669"]["label"]

    risk_note = ""
    if risk_flag == "\u662f":
        risk_note = "\u7528\u6237\u5f53\u524d\u53ef\u80fd\u5b58\u5728\u81ea\u4f24/\u81ea\u6740\u98ce\u9669\uff0c\u56de\u590d\u7684\u5f00\u5934\u548c\u7ed3\u5c3e\u8981\u8868\u8fbe\u5173\u5207\uff0c" \
            "\u660e\u786e\u5f15\u5bfc\u6c42\u52a9\u4e13\u4e1a\u652f\u6301\uff0c\u5168\u56fd\u5fc3\u7406\u6551\u63f4\u70ed\u7ebf 12356\uff0c\u6216\u5efa\u8bae\u8054\u7cfb\u8eab\u8fb9\u7684\u4eba\u3002"
    else:
        risk_note = "\u6ca1\u6709\u7279\u522b\u5f3a\u8c03\u5371\u9669\uff0c\u53ef\u4ee5\u4e13\u6ce8\u5076\u804a\u548c\u95ea\u5f00\u8f6c\u79fb\u3002"

    history_block = ""
    if conversation_history:
        history_lines = []
        for role, msg_content in conversation_history[-12:]:
            speaker = "\u7528\u6237" if role == "user" else "\u52a9\u624b"
            history_lines.append(f"{speaker}\uff1a{msg_content}")
        history_block = "\u4ee5\u4e0b\u662f\u4e4b\u524d\u7684\u5bf9\u8bdd\u5386\u53f2\uff1a" + chr(10) + chr(10).join(history_lines) + chr(10) + chr(10)

    context_block = ""
    if context_summary:
        context_block = "\u5bf9\u8bdd\u4e0a\u4e0b\u6587\u6458\u8981\uff1a" + chr(10) + context_summary + chr(10) + chr(10)

    user_line = chr(10) + "\u3010\u7528\u6237\u6700\u65b0\u53d1\u8a00\u3011" + chr(10) + '"' + user_text + '"' + chr(10) + chr(10)
    sys_line = "\u3010\u7cfb\u7edf\u5206\u6790\uff08\u5185\u90e8\u53c2\u8003\uff09\u3011" + chr(10) + "- \u60c5\u7eea\uff1a" + sent + chr(10) + "- \u6c42\u52a9\uff1a" + help_flag + chr(10) + "- \u98ce\u9669\uff1a" + risk_flag + chr(10) + chr(10)

    rules = "\u3010\u56de\u590d\u8981\u6c42\uff08\u5fc5\u987b\u9075\u5b88\uff09\u3011" + chr(10)
    rules += "1. \u60c5\u7eea\u4f18\u5148\uff1a\u7528\u6237\u7684\u7b2c\u4e00\u53e5\u8bdd\u901a\u5e38\u662f\u60c5\u7eea\u53d1\u6cc4\uff0c\u4e0d\u662f\u95ee\u9898\u6c42\u52a9\uff0c\u5148\u63a5\u4f4f\u60c5\u7eea\uff0c\u4e0d\u8981\u6025\u7740\u89e3\u51b3\u95ee\u9898\uff1b" + chr(10)
    rules += "2. \u52a8\u6001\u5171\u60c5\uff1a\u5148\u63cf\u8ff0\u4f60\u611f\u53d7\u5230\u7684\u60c5\u7eea\uff08\u201c\u542c\u8d77\u6765\u4f60\u73b0\u5728\u2026\u2026\u201d\uff09\uff0c\u518d\u8bf4\u660e\u4f60\u7406\u89e3\u7684\u539f\u56e0\uff08\u201c\u56e0\u4e3a\u2026\u2026\u201d\uff09\uff0c\u6700\u540e\u66ff\u7528\u6237\u8bf4\u51fa\u8bf4\u4e0d\u51fa\u53e3\u7684\u77db\u76fe\uff1b" + chr(10)
    rules += "3. \u53e3\u8bed\u5316\uff1a\u7528\u201c\u54ce\u5440\u201d\u201c\u786e\u5b9e\u201d\u201c\u771f\u7684\u5417\u201d\u201c\u5929\u554a\u201d\u7b49\u53e3\u8bed\u8bcd\uff0c\u5141\u8bb8\u4e0d\u5b8c\u6574\u7684\u53e5\u5b50\uff0c\u5141\u8bb8\u505c\u987f\u611f\uff1b" + chr(10)
    rules += "4. \u7981\u6b62\u7a7a\u6d1e\u5b89\u6170\uff1a\u201c\u4e00\u8d77\u6162\u6162\u6765\u201d\u201c\u522b\u6025\u201d\u201c\u4e00\u5207\u90fd\u4f1a\u597d\u7684\u201d\u201c\u7ed9\u81ea\u5df1\u4e00\u70b9\u65f6\u95f4\u201d\uff0c\u8fd9\u4e9b\u592a\u50cf\u5fc3\u7406\u54a8\u8be2\u5e08\u8bf4\u6cd5\uff1b" + chr(10)
    rules += "5. \u7981\u6b62\u8bf4\u201c\u6211\u5b8c\u5168\u7406\u89e3\u4f60\u201d\uff0c\u53ef\u4ee5\u8bf4\u201c\u6211\u597d\u50cf\u80fd\u611f\u53d7\u5230\u4e00\u90e8\u5206\u4f60\u7684\u611f\u53d7\u201d\uff1b" + chr(10)
    rules += "6. \u63a7\u5236\u5728 40-80 \u5b57\uff0c\u6bcf\u53e5\u8bdd\u4e0d\u8d85\u8fc7 3 \u53e5\uff1b" + chr(10)
    rules += "7. " + risk_note + chr(10)
    rules += "8. \u4e0d\u63d0\u7cfb\u7edf\u5224\u65ad\uff0c\u4e0d\u50cf\u533b\u751f\u95ee\u8bca\uff0c\u4e0d\u8bf4\u6559\uff1b" + chr(10)
    rules += "9. \u76f4\u63a5\u8f93\u51fa\u56de\u590d\uff0c\u4e0d\u8981\u4efb\u4f55\u524d\u7f00\u3001\u7f16\u53f7\u3001\u5f15\u53f7\u3002"

    prompt = (
        "\u4f60\u662f\u7528\u6237\u7684\u670b\u53cb\uff0c\u5728\u5fae\u4fe1\u4e0a\u804a\u5929\u3002"
        "\u4f60\u7684\u98ce\u683c\u662f\u6709\u6e29\u5ea6\u4f46\u4e0d\u8fc7\u5206\u6e29\u6696\u7684\u966a\u4f34\u8005\uff0c"
        "\u53c2\u8003\u771f\u5b9e\u54a8\u8be2\u5e08\u7684\u5171\u60c5\u65b9\u5f0f\uff1a\u66ff\u7528\u6237\u8bf4\u51fa\u5fc3\u91cc\u7684\u611f\u53d7\u3001\u539f\u56e0\u3001\u60f3\u6cd5\u3001\u77db\u76fe\u3002"
        "\u7528\u6237\u53ef\u80fd\u5728\u7ecf\u5386\u60c5\u7eea\u56f0\u6270\uff0c\u56de\u590d\u8981\u50cf\u670b\u53cb\u53d1\u5fae\u4fe1\u4e00\u6837\u81ea\u7136\uff0c\u6709\u5177\u4f53\u5185\u5bb9\uff0c\u4e0d\u8981\u5957\u8bdd\u548c\u5e94\u916c\u8bed\u3002"
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
