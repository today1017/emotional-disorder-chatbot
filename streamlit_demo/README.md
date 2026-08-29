# 情感障碍语料分析与对话干预原型 —— 运行说明

## 在线演示

**访问地址**: [https://emotional-disorder-chatbot-llf.streamlit.app/](https://emotional-disorder-chatbot-llf.streamlit.app/)

---

## 项目简介
本应用为「大学生创新创业训练计划」的程序演示原型，用于展示基于关键词规则 + 大语言模型的情感分析与共情回应生成能力。

### 核心功能
1. **情感极性分类**（积极 / 中性 / 消极）
2. **求助信号识别**（是 / 否）
3. **安全风险检测**（是 / 否）
4. **情感强度评分**（0-100，基于词典加权）
5. **AI 共情回应生成**（调用智谱 GLM / DeepSeek，支持开关）
6. **多轮对话**（连续输入，基于历史上下文生成连贯回应）
7. **多模型支持**（侧边栏切换：GLM-4-Flash 免费 / GLM-4-Plus 付费 / DeepSeek-V3 付费）
8. **API 错误透明化**（调用失败时侧边栏显示具体错误原因）
9. **上下文感知分析**：
   - 情感趋势折线图（随时间变化可视化）
   - 风险升级检测（由无风险→有风险的自动警示）
   - 短回应情绪继承（"嗯""是的"等短句继承前文语境）
   - 连续消极情绪提示
10. **历史会话管理**（一键新建会话，重置上下文）

---

## 环境准备

### 依赖安装（Python 3.10+）
```bash
pip install streamlit pandas jieba
```
> 若需使用 LLM 增强功能，还需设置智谱 API Key（见下方"LLM 配置"）。

### 可选：仅运行规则模型（无需 LLM）
无需任何 API Key，直接启动即可：
```bash
cd streamlit_demo
python -m streamlit run app.py
```

---

## LLM 配置（可选）

本应用支持多种 LLM 模型，通过侧边栏切换：

| 模型 | 接口地址 | 费用 | 说明 |
|------|----------|------|------|
| GLM-4-Flash | `open.bigmodel.cn` | **免费** | 推荐日常使用 |
| GLM-4-Plus | `open.bigmodel.cn` | 付费 | 效果更好 |
| DeepSeek-V3 | `api.deepseek.com` | 付费（500万免费 token） | 另一选择 |

### 设置 API Key（任选一种）

**方式一：环境变量（推荐）**
```bash
# Windows PowerShell
setx ZHIPU_API_KEY "你的智谱API密钥"

# macOS / Linux
export ZHIPU_API_KEY="你的智谱API密钥"
```
> API Key 优先级：`ZHIPU_API_KEY` > `SILICONFLOW_API_KEY` > `OPENAI_API_KEY`

**方式二：应用内侧边栏设置**
启动后点击左侧「设置 → 对话设置」，直接填写 API Key 并选择模型，无需重启。

**方式三：Streamlit Cloud Secrets**
在 Streamlit Cloud 应用设置中添加 Secrets：
```toml
ZHIPU_API_KEY = "你的智谱API密钥"
```

### API 错误处理
- 若 API Key 无效或余额不足，侧边栏会显示红色错误提示（如 `HTTP 402: Insufficient Balance`）
- 调用失败时自动回退到规则模型预设话术

---

## 启动应用

```bash
cd streamlit_demo
python -m streamlit run app.py --server.port 8501
```
浏览器打开 `http://localhost:8501` 即可使用。

### 无头模式（服务器部署）
```bash
python -m streamlit run app.py --server.headless true --server.port 8501
```

---

## 目录结构
```
streamlit_demo/
├── app.py           # Streamlit 主界面（多轮对话 + 模型选择器 + API 错误显示）
├── conversation.py  # 多轮对话状态管理（上下文聚合/情感趋势/风险升级）
├── rules.py         # 规则模型（关键词提取、分类、建议话术、情感强度、上下文感知）
├── llm.py           # LLM 调用封装（智谱 GLM / DeepSeek，多轮 prompt + few-shot 示例）
├── config.py        # 全局配置
├── utils.py         # 工具函数（中文字体配置）
├── requirements.txt # 依赖声明
├── NotoSansSC-Regular.otf  # 中文字体
├── README.md        # 本文件
└── DEPLOY_README.md # 部署说明
```

---

## 关键数据文件

分析所需语料文件位于上级目录（桌面）：
- `weibo.csv` —— 微博情感障碍语料（1489 条）
- `zhuhu.csv` —— 知乎情感障碍语料（518 条）
- 文件编码：GB18030

---

## 项目文件说明

| 文件 | 用途 |
|------|------|
| `text_analysis_pipeline.py` | 五阶段描述统计分析流水线（fig1-fig9） |
| `extended_analysis.py` | 规则评估 + 平台对比 + TF-IDF + LDA + 情感强度（fig10-fig16） |
| `system_diagrams.py` | 系统逻辑结构图 + 交互流程图（fig17-fig18） |
| `output_figures/` | 所有导出图表（300dpi PNG） |

---

## 规则模型说明

### 分类策略
基于从语料库中提取的高频词作为关键词种子库，采用子串匹配计分：
1. 对每条文本进行清洗（去 URL / @ / 话题标签 / 特殊符号）和 jieba 分词
2. 统计各类别（情绪极性 / 求助信号 / 安全风险）种子词命中次数
3. 按加权得分最高的类别输出判断结果

### 评估指标（在 1998 条人工标注上）
| 维度 | 准确率 | 说明 |
|------|--------|------|
| 情绪极性 | 65.97% | 消极识别较好（F1=0.79），积极/中性受限于样本量 |
| 求助信号 | 83.73% | "是"类召回率偏低（0.17），倾向保守判断 |
| 安全风险 | 82.98% | 同上，"是"类召回 0.22 |

> 规则模型局限：关键词覆盖有限，对隐喻、反讽等复杂表达识别不足。LLM 增强可弥补此短板。

### 关键词库扩展
- 消极关键词：170→195 个（新增"没有希望""结束一切""去死"等变体）
- 安全风险关键词：74→89 个（新增"去死""死了算了""不如去死"等口语化表达）

---

## 多轮对话功能说明

升级后的原型支持**连续多轮对话**，核心机制：

1. **上下文感知分析**（`rules.analyze_with_context`）：
   - 短回应继承：用户回复"嗯""好吧"等无关键词短句时，自动继承前一轮情绪极性
   - 风险升级检测：对比历史风险标记，首次出现风险时提示"本轮首次出现安全风险信号"
   - 情感波动检测：与上一轮情感强度比较，>15 分时提示"情绪加剧/缓和"

2. **对话上下文聚合**（`conversation.py`）：
   - 维护情感强度时间序列、风险历史、求助信号计数、连续消极/积极轮次
   - 主导情绪采用最近 5 轮加权投票计算
   - 上下文摘要自动拼接进 LLM prompt

3. **LLM 多轮增强**（`llm.build_multi_turn_prompt`）：
   - 自动携带最近 12 条历史对话（用户+助手）
   - 附带上下文摘要（轮次、主导情绪、风险升级、连续消极等）
   - 朋友风格 prompt：system message 为"普通朋友微信聊天"，包含 8 个正面示例 + 7 个禁止示例
   - 强制每次回复开头不同，避免套话；30-60 字，不超过 2 句话

4. **情感趋势可视化**：对话上方实时折线图展示情感强度变化，红色标记存在安全风险的轮次

---

## 已知限制
- 语料来源为社交媒体文本，非正式对话语料，可能与实际干预场景存在偏差
- 规则模型基于统计高频词，无法覆盖长尾表达和语义理解
- LLM 回应仅供参考，**不可替代专业心理咨询或临床诊断**

---

## 联系方式
本应用为学术研究演示原型，如有问题请联系项目组成员。
