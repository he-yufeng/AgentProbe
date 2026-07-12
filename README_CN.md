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

你上线了一个 AI Agent，运行良好。两周后，你改了个 prompt，换了个模型，或者升级了依赖——然后某个功能悄悄坏了。但你不知道，因为**没有测试能捕获 Agent 行为的回归**。

传统单元测试不适用于 Agent。输出是非确定性的，是自然语言而不是精确值。你不能简单地 `assertEqual`。就算能，写测试 fixture 的时间比写 Agent 还长。

**AgentProbe** 解决这个问题。一个装饰器捕获你的 Agent 输出并保存为基线快照。下次运行时，它会把新输出和基线对比——支持精确匹配和语义相似度。如果有变化，测试失败。在 CI 中运行，就能在上线前捕获回归。

## 工作原理

![AgentProbe 快照流程](docs/architecture.png)

## 快速开始

```bash
pip install agentprobe
```

### 1. 快照测试

捕获 Agent 输出并跨运行对比：

```python
from agentprobe import snapshot

@snapshot("summarize_article")
def test_summarize():
    result = my_agent.summarize("文章内容...")
    return result
```

首次运行：创建基线到 `.agentprobe/snapshots/summarize_article.json`。
后续运行：对比输出和基线，不一致则失败。

异步 Agent 也可以直接使用同一个 decorator：

```python
@snapshot("async_summarize")
async def test_async_summarize():
    result = await my_agent.summarize_async("文章内容...")
    return result
```

当输出里带有时间戳、request id 这类非确定字段时，把它们列进 `redact`，比较前会被屏蔽，避免误报不匹配：

```python
@snapshot("summarize_article", redact=["timestamp", "request_id"])
def test_summarize():
    return my_agent.summarize("...")  # {"summary": "...", "timestamp": 1718...}
```

列出的 key 会在任意层级被替换为 `"<redacted>"` 再保存和比较；其他字段的真实变化仍会让快照失败。

### 2. Mock LLM

不调用任何 API 测试 Agent 逻辑：

```python
from agentprobe import MockLLM

def test_agent_with_mock():
    mock = MockLLM(responses=[
        "文档讨论了三个主要话题。",
        "根据我的分析，情感是正面的。"
    ])
    result = mock.chat.completions.create(
        messages=[{"role": "user", "content": "总结这篇文档"}]
    )
    assert "三个主要话题" in result.choices[0].message.content
```

### 3. 工具调用断言

验证 Agent 调用了正确的工具：

```python
from agentprobe import assert_no_tool_called, assert_tool_called, assert_tool_sequence

def test_agent_uses_search():
    tool_calls = [
        {"name": "web_search", "arguments": {"query": "最新新闻"}},
        {"name": "summarize", "arguments": {"text": "..."}},
    ]
    assert_tool_called(tool_calls, "web_search", times=1)
    assert_tool_called(tool_calls, "web_search", with_args={"query": "最新新闻"})
    assert_tool_sequence(tool_calls, ["web_search", "summarize"])
    assert_no_tool_called(tool_calls, "delete_file")
```

对于多步骤 Agent，可以用 `assert_tool_sequence(..., contiguous=True)` 检查两个工具调用必须相邻，避免 planner 重排后悄悄破坏流程。

当调用次数不确定时，用 `min_times`/`max_times` 替代精确的 `times`——比如断言不稳定的 API 最多重试三次，或搜索至少跑了两次：

```python
assert_tool_called(tool_calls, "api_call", max_times=3)   # 有重试，但有上限
assert_tool_called(tool_calls, "web_search", min_times=2)  # 至少两次搜索
```

`with_args` 支持嵌套 subset 匹配，也能处理 OpenAI function call 常见的 JSON 字符串参数：

```python
assert_tool_called(
    tool_calls,
    "write_file",
    with_args={"metadata": {"mode": "safe"}},
)
```

### 4. Schema 验证

断言 Agent 输出符合预期结构：

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

逐步记录 Agent 做了什么，再对这条 trace 做断言或快照。`trace.tool_calls` 可以直接喂给断言辅助函数：

```python
from agentprobe import Trace, assert_tool_sequence

def test_research_flow():
    trace = Trace()
    # 随 Agent 运行记录每一步（工具调用、LLM 轮次、自定义事件）
    trace.record_llm("planning the search")
    trace.record_tool_call("search", {"query": "rainfall 2023"})
    trace.record_event("retry", attempt=2)
    trace.record_tool_call("fetch", {"url": "https://example.com"})

    assert_tool_sequence(trace.tool_calls, ["search", "fetch"])
    assert trace.names == ["llm", "search", "retry", "fetch"]
    # trace.to_dict() 对快照友好，适合做整段运行的回归测试
```

### 6. 成本追踪

在 trace 上记录 token 用量，断言整段运行没超出美元预算，把那些悄悄烧更多钱的回归（更长的 prompt、多出来的轮次、换了更贵的模型）挡在合并之前。价格可以来自一个 dict、一个 callable，或者在装了 [TokenTracker](https://github.com/he-yufeng/TokenTracker) 时用它的价格表：

```python
from agentprobe import Trace, assert_cost_under

def test_run_stays_under_budget():
    trace = Trace()
    trace.record_llm("plan", model="gpt-4o", input_tokens=1200, output_tokens=300)
    trace.record_llm("answer", model="gpt-4o", input_tokens=800, output_tokens=500)

    # 价格 dict：{模型: (每 1k 输入价 usd, 每 1k 输出价 usd)}
    assert_cost_under(trace, 0.05, pricing={"gpt-4o": (0.005, 0.015)})
    # 或 pricing=None，改用 TokenTracker 的价格表（pip install toktally）
```

## Pytest 集成

AgentProbe 自动注册为 pytest 插件，提供 `agentprobe` fixture：

```python
def test_with_fixture(agentprobe):
    output = my_agent.run("你好")
    result = agentprobe.capture("greeting_test", output)
    assert result.passed
```

### 命令行参数

```bash
# 正常运行测试
pytest tests/

# 更新所有快照（重新生成基线）
pytest tests/ --agentprobe-update

# 使用语义比较
pytest tests/ --agentprobe-mode=semantic --agentprobe-threshold=0.85
```

快照变化时，AgentProbe 会在失败信息里输出存储 JSON 快照与当前输出的 unified diff。
CI 日志里可以直接看到哪个字段或哪句话发生了漂移。

### AgentProbe CLI

```bash
# 运行测试
agentprobe run

# 语义比较模式
agentprobe run --mode semantic --threshold 0.9

# 更新所有快照
agentprobe update
```

## 对比模式

| 模式 | 工作原理 | 适用场景 |
|------|---------|---------|
| `exact`（默认） | 序列化后字符串相等 | 确定性 Agent、结构化输出 |
| `semantic` | 通过 sentence-transformers 计算余弦相似度 | 非确定性 LLM 输出 |

语义模式需安装可选依赖：

```bash
pip install agentprobe[semantic]
```

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

## GitHub Actions

```yaml
- name: Run agent tests
  run: |
    pip install agentprobe
    pytest tests/ -v
```

快照文件（`.agentprobe/snapshots/`）应该提交到你的仓库，这样 CI 才能对比。

## 常见问题

**需要 API Key 吗？**
不需要。用 `MockLLM` 做确定性测试，完全不调用任何 API。

**语义比较怎么工作？**
用 sentence-transformers 把基线和当前输出编码为向量，计算余弦相似度。分数高于阈值（默认 0.85）则通过。

**能和 LangChain / CrewAI / AutoGen 一起用吗？**
可以。AgentProbe 不关心你用什么框架，它测试的是 Agent 的输出。

**非确定性输出导致测试不稳定怎么办？**
用语义模式并设置合适的阈值。也可以用 `MockLLM` 让底层 LLM 变成确定性的。

## 路线图

**已完成**：异步 Agent 测试（`async def`）、工具调用断言（存在性、次数上下界、调用顺序、禁用参数检查）、多步追踪（记录中间步骤）、与 TokenTracker 联动的成本追踪、快照不一致时的终端可视化 diff，以及带原子写入的 `pytest-xdist` 并行支持。

**规划中**：

- **交互式快照评审**：一个 `--agentprobe-review` 模式，逐个走查发生变化的快照、一条条接受或拒绝，而不是一次性重建所有基线。
- **框架适配器**：对 LangChain、LlamaIndex、OpenAI Assistants API 的一等步骤捕获，让多步运行的追踪不再需要手写胶水代码。
- **离线语义模式**：给语义比对加一个本地 embedding 后端，让阈值检查不必每次断言都打一次 API。

## 贡献

欢迎贡献。如果你在生产环境中测试 AI Agent，有什么想法或需求，请开 issue。

## 相关项目

AgentProbe 属于我维护的一小套 agent 测试工具，下面是几个相关的：

- **[CoreCoder](https://github.com/he-yufeng/CoreCoder)** — 想搞懂一个 coding agent 到底怎么运作？把整套约 1000 行引擎从头读到尾，而不是当黑箱。
- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)** — 被丢进一个陌生代码库？它给你一份带「从哪读起」路径的 wiki，一个可自托管的 DeepWiki 替代。
- **[LiteBench](https://github.com/he-yufeng/LiteBench)** — 一条命令给任意 LLM 跑基准：内置 HumanEval、GSM8K、MMLU，也能加你自己的任务。
- **[agentcikit](https://github.com/he-yufeng/agentcikit)** — LLM agent 的 CI 安全层：回放运行、给工具调用上围栏、上线前分诊失败。

## 许可证

[MIT](LICENSE)

---

<div align="center">

**别再上线没测试的 Agent 了。**

[报告 Bug](https://github.com/he-yufeng/AgentProbe/issues) · [功能请求](https://github.com/he-yufeng/AgentProbe/issues)

</div>
