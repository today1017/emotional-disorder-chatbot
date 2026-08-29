# 在线演示

**访问地址**: [https://emotional-disorder-chatbot-llf.streamlit.app/](https://emotional-disorder-chatbot-llf.streamlit.app/)

---

# Streamlit Cloud 部署指南

## 一键部署步骤

1. **Fork/上传代码到 GitHub**
   - 将 `streamlit_demo/` 文件夹下所有文件上传到 GitHub 仓库根目录
   - 必须包含：`app.py`、`requirements.txt`、`.streamlit/config.toml`、`conversation.py`、`llm.py`、`rules.py`

2. **在 Streamlit Cloud 部署**
   - 访问 https://share.streamlit.io
   - 点击 "New app" → 选择你的 GitHub 仓库
   - Main file path: `app.py`
   - 点击 "Deploy!"

3. **配置 Secrets（可选，用于 LLM 功能）**
   - 在 Streamlit Cloud 应用设置中添加 Secrets：
   ```toml
   ZHIPU_API_KEY = "你的智谱API密钥"
   ```
   - 或在应用侧边栏手动填入

## 注意事项

- **数据文件**：云端无 `weibo.csv`/`zhuhu.csv`，规则模型会自动回退到 `FALLBACK_KEYWORDS`/`SEED_WORDS`（已内置访谈扩充词表），功能不受影响
- **中文字体**：`config.toml` 已配置无头模式，matplotlib 使用默认字体
- **端口**：固定 8501，由平台自动映射

## 本地运行

```bash
cd streamlit_demo
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

## 功能清单

- ✅ 多轮对话 + 上下文感知
- ✅ 规则模型：情绪极性/求助信号/安全风险/情感强度
- ✅ LLM 增强回应（智谱 GLM-4-Flash 免费 / GLM-4-Plus 付费 / DeepSeek-V3 付费）
- ✅ 侧边栏模型选择器 + API 错误透明化显示
- ✅ 朋友风格 prompt + few-shot 示例（8个正面 + 7个禁止）
- ✅ 情感趋势折线图 + 风险标记
- ✅ 风险升级检测 + 连续消极提示 + 短回应情绪继承
- ✅ 会话导出（JSON + Markdown 情绪报告）
- ✅ 6 个预设演示场景（含访谈真实语料）
- ✅ 侧边栏：LLM 配置 / 会话概览 / 最新分析卡片
- ✅ 深色模式适配

## 更新日志

- **2026-08-30**：prompt 重写为朋友风格 + few-shot 示例；API 错误透明化显示；DeepSeek-V3 支持；关键词库扩充
- **2026-08-23**：注入成员B访谈高价值风险词（50+ 条），显著提升安全风险/求助召回
- **2026-08-23**：新增会话导出（JSON/Markdown 情绪报告）功能
- **2026-08-23**：Streamlit Cloud 部署就绪配置