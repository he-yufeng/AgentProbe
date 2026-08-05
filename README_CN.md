<div align="center">

<img src="docs/banner.png" alt="AgentProbe — AI Agent 回归测试框架" width="100%">

一个装饰器捕获 Agent 输出，存为基线，CI 中自动检测回归。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/he-yufeng/AgentProbe/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/AgentProbe/actions)

**[English](README.md) · [中文](README_CN.md)** &nbsp;·&nbsp; [快速开始](#快速开始) · [工作原理](#工作原理) · [与其他工具对比](#与其他工具对比)

</div>

---

## 解决什么问题

你上线了一个 AI Agent，运行良好。两周后，你改了个 prompt，换了个模型，或者升级了依赖，然后某个功能悄悄坏了。但你不知道，因为**没有测试能捕获 Agent 行为的回归**。

传统单元测试不适用于 Agent。输出是非确定性的自然语言，不能简单地 `assertEqual`，手写 fixture 的成本比写 Agent 本身还高。

**AgentProbe** 解决这个问题。一个装饰器捕获你的 Agent 输出并保存为基线快照。下次运行时，它把新输出和基线对比：精确匹配或语义相似度。有变化，测试就失败。在 CI 中运行，就能在上线前捕获回归。

## 工作原理

![AgentProbe 快照流程](docs/architecture.png)

## 快速开始

```bash
pip install agentpoke
```

### 1. 快照测试

```python
from agentprobe import snapshot

@snapshot("summarize_article")
def test_summarize():
    result = my_agent.summarize("文章内容...")
    return result
```

首次运行：创建基线到 `.agentprobe/snapshots/summarize_article.json`。后续运行：对比输出和基线，不一致则失败。`async def` 异步 Agent 用法相同。快照文件应该提交进仓库，CI 才能对比。

非确定字段和凭据在比较前处理掉：

```python
# 屏蔽任意层级的易变字段，避免误报不匹配
@snapshot("summarize_article", redact=["timestamp", "request_id"])

# 把 API key、token、JWT、邮箱从长文本里抠出来打码，
# 自定义形状用 redact_patterns 传正则（也可全局开 --agentprobe-redact-secrets）
@snapshot("summarize_article", redact_secrets=True, redact_patterns=[r"internal-\d{4}"])
```

### 2. Mock LLM

`MockLLM` 是 `openai.Client` 的直接替代品，返回剧本化响应，测试 Agent 逻辑不打任何 API：

```python
from agentprobe import MockLLM

mock = MockLLM(responses=[
    "文档讨论了三个主要话题。",
    {"tool_calls": [{"id": "1", "function": {"name": "search", "arguments": '{"q": "test"}'}}]},
])

result = mock.chat.completions.create(messages=[{"role": "user", "content": "总结这篇文档"}])
assert "三个主要话题" in result.choices[0].message.content
assert mock.call_count == 1  # mock.calls 记录全部调用；mock.reset() 复用
```

剧本响应按序消费，用完后由 `default_response=` 兜底。

### 3. 工具调用断言

验证 Agent 以正确的姿势调用了正确的工具：

```python
from agentprobe import assert_no_tool_called, assert_tool_called, assert_tool_sequence

assert_tool_called(tool_calls, "web_search", times=1)
assert_tool_called(tool_calls, "web_search", with_args={"query": "最新新闻"})
assert_tool_sequence(tool_calls, ["web_search", "summarize"])
assert_no_tool_called(tool_calls, "delete_file")
```

各种变体覆盖真实场景里的麻烦情况：

- `assert_tool_sequence(..., contiguous=True)` 检查两个工具必须相邻调用，防止 planner 重排悄悄破坏流程。
- 次数不确定时用 `min_times`/`max_times` 替代 `times`：`assert_tool_called(tool_calls, "api_call", max_times=3)` 给不稳定的重试加上限。
- `assert_max_tool_calls(tool_calls, 10)` 约束整段运行的总调用数，不只是单个工具（一次不调也算过）。
- `with_args` 是嵌套 subset 匹配，也能处理 OpenAI function call 的 JSON 字符串参数。
- `assert_tool_not_called_with(tool_calls, "run", {"sudo": True})` 允许用工具，但带了危险参数子集就失败。

### 4. Schema 验证

```python
from pydantic import BaseModel
from agentprobe import assert_schema

class AgentResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

def test_output_structure():
    output = my_agent.run("法国的首都是什么？")
    result = assert_schema(output, AgentResponse)
    assert result.confidence > 0.8
```

### 5. 多步骤追踪

逐步记录 Agent 做了什么，再对这条 trace 做断言或快照：

```python
from agentprobe import Trace, assert_tool_sequence

trace = Trace()
trace.record_llm("planning the search")
trace.record_tool_call("search", {"query": "rainfall 2023"})
trace.record_event("retry", attempt=2)
trace.record_tool_call("fetch", {"url": "https://example.com"})

assert_tool_sequence(trace.tool_calls, ["search", "fetch"])
assert trace.names == ["llm", "search", "retry", "fetch"]
# trace.to_dict() 对快照友好，适合做整段运行的回归测试
```

### 6. 成本追踪

在 trace 上记录 token 用量，断言整段运行没超出美元预算，把悄悄烧更多钱的回归（更长的 prompt、多出来的轮次、更贵的模型）挡在合并之前。价格可以来自 dict、callable，或者 [TokenTracker](https://github.com/he-yufeng/TokenTracker) 的价格表（`pip install toktally`）：

```python
from agentprobe import assert_cost_under

# 价格 dict：{模型: (每 1k 输入价 usd, 每 1k 输出价 usd)}
assert_cost_under(trace, 0.05, pricing={"gpt-4o": (0.005, 0.015)})
```

## Pytest 集成

AgentProbe 自动注册为 pytest 插件，提供 `agentprobe` fixture：

```python
def test_with_fixture(agentprobe):
    output = my_agent.run("你好")
    result = agentprobe.capture("greeting_test", output)
    assert result.passed
```

```bash
pytest tests/                                        # 正常运行测试
pytest tests/ --agentprobe-update                    # 有意变更后重建基线
pytest tests/ --agentprobe-mode=semantic --agentprobe-threshold=0.85
```

快照变化时，AgentProbe 会在失败信息里输出存储 JSON 与当前输出的 unified diff，CI 日志直接看到哪个字段或哪句话漂移了。独立 CLI 与参数一一对应：`agentprobe run`、`agentprobe run --mode semantic --threshold 0.9`、`agentprobe update`。

### 本地复盘失败快照

比较失败时还会把本次实际输出存到 `.agentprobe/last_run/`，不用重跑测试就能检查漂移并决定是否接受：

```bash
agentprobe diff              # 基线 vs 最近失败输出，带相似度
agentprobe diff summarize    # 只看某一个快照
agentprobe diff --stat       # 每个快照一行：+新增 -删除 行数，先过一遍再逐个深挖
agentprobe diff --html report.html  # 自包含 HTML 报告，适合分享或挂 CI 构件
agentprobe accept            # 把全部 last_run 提升为新基线
agentprobe accept summarize  # 只提升某一个
```

日常闭环就是这样：CI 红了，`agentprobe diff` 看清是哪句话变了，`agentprobe accept` 确认新基线。不用手改 JSON，也不用闭眼全量 update。

## 对比模式

| 模式 | 工作原理 | 适用场景 |
|------|---------|---------|
| `exact`（默认） | 序列化后字符串相等 | 确定性 Agent、结构化输出 |
| `semantic` | 通过 sentence-transformers 计算余弦相似度（`pip install agentpoke[semantic]`） | 非确定性 LLM 输出 |

## 与其他工具对比

| 功能 | AgentProbe | DeepEval | Promptfoo |
|------|-----------|----------|-----------|
| pytest 原生 | 是（插件） | 独立运行器 | 仅 CLI |
| 快照基线 | 是 | 否 | 否 |
| 语义比较 | 是 | 是 | 是 |
| Mock LLM | 是（内置） | 否 | 部分 |
| 工具调用断言 | 是 | 否 | 否 |
| Schema 验证 | 是（Pydantic） | 部分 | 否 |
| 需要云服务 | 否 | 可选 | 否 |
| 配置格式 | Python 代码 | Python 代码 | YAML |

## GitHub Actions

```yaml
- name: Run agent tests
  run: |
    pip install agentpoke
    pytest tests/ -v
```

## 常见问题

**需要 API Key 吗？**
不需要。`MockLLM` 做确定性测试，完全不调用任何 API。要打真实 LLM 时用的是厂商的 key，那是你 Agent 的依赖，不是 AgentProbe 的。

**非确定性输出导致测试不稳定怎么办？**
用语义模式并设置合适的阈值，或者用 `MockLLM` 让底层 LLM 变成确定性的。

**能和 LangChain / CrewAI / AutoGen 一起用吗？**
可以。AgentProbe 测试的是 Agent 的输出，不是内部实现。在测试函数里调用你的 Agent 并返回结果就行。

## 路线图

**已完成**：异步 Agent 测试、工具调用断言（存在性、次数上下界、调用顺序、禁用参数检查）、多步追踪、与 TokenTracker 联动的成本追踪、快照不一致时的终端可视化 diff 与 `--html` 自包含报告、带原子写入的 `pytest-xdist` 并行支持、按模式脱敏的快照密钥清洗。

**规划中**：

- **交互式快照评审**：`--agentprobe-review` 模式，逐个走查变化的快照、一条条接受或拒绝。
- **框架适配器**：对 LangChain、LlamaIndex、OpenAI Assistants API 的一等步骤捕获，多步运行追踪不再需要手写胶水。
- **离线语义模式**：本地 embedding 后端，阈值检查不必每次断言都打 API。

## 贡献

欢迎贡献。如果你在生产环境中测试 AI Agent，有什么想法或需求，请开 issue。

## 相关项目

- **[CoreCoder](https://github.com/he-yufeng/CoreCoder)**：想搞懂一个 coding agent 到底怎么运作？把整套约 1000 行引擎从头读到尾，而不是当黑箱。
- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)**：被丢进一个陌生代码库？它给你一份带「从哪读起」路径的 wiki，可自托管的 DeepWiki 替代。
- **[LiteBench](https://github.com/he-yufeng/LiteBench)**：一条命令给任意 LLM 跑基准：内置 HumanEval、GSM8K、MMLU，也能加你自己的任务。
- **[agentcikit](https://github.com/he-yufeng/agentcikit)**：LLM agent 的 CI 安全层：回放运行、给工具调用上围栏、上线前分诊失败。

## 许可证

[MIT](LICENSE)

---

<div align="center">

**别再上线没测试的 Agent 了。**

[报告 Bug](https://github.com/he-yufeng/AgentProbe/issues) · [功能请求](https://github.com/he-yufeng/AgentProbe/issues)

</div>
