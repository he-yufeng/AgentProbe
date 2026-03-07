<div align="center">

# AgentProbe

**AI Agent 回归测试框架。像 Jest 快照测试，但专为 LLM 设计。**

一个装饰器捕获 Agent 输出，存为基线，CI 中自动检测回归。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/he-yufeng/AgentProbe/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/AgentProbe/actions)

**[English](README.md) | [中文](README_CN.md)**

</div>

---

## 解决什么问题

你上线了一个 AI Agent，运行良好。两周后，你改了个 prompt，换了个模型，或者升级了依赖——然后某个功能悄悄坏了。但你不知道，因为**没有测试能捕获 Agent 行为的回归**。

传统单元测试不适用于 Agent。输出是非确定性的，是自然语言而不是精确值。你不能简单地 `assertEqual`。就算能，写测试 fixture 的时间比写 Agent 还长。

**AgentProbe** 解决这个问题。一个装饰器捕获你的 Agent 输出并保存为基线快照。下次运行时，它会把新输出和基线对比——支持精确匹配和语义相似度。如果有变化，测试失败。在 CI 中运行，就能在上线前捕获回归。

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
from agentprobe import assert_tool_called

def test_agent_uses_search():
    tool_calls = [
        {"name": "web_search", "arguments": {"query": "最新新闻"}},
        {"name": "summarize", "arguments": {"text": "..."}},
    ]
    assert_tool_called(tool_calls, "web_search", times=1)
    assert_tool_called(tool_calls, "web_search", with_args={"query": "最新新闻"})
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

- [ ] 异步 Agent 支持
- [ ] 多步 Agent 追踪（记录中间步骤）
- [ ] 成本追踪集成（与 TokenTracker 联动）
- [ ] 终端中的可视化 diff
- [ ] `pytest-xdist` 并行支持

## 贡献

欢迎贡献。如果你在生产环境中测试 AI Agent，有什么想法或需求，请开 issue。

## 许可证

[MIT](LICENSE)

---

<div align="center">

**别再上线没测试的 Agent 了。**

[报告 Bug](https://github.com/he-yufeng/AgentProbe/issues) · [功能请求](https://github.com/he-yufeng/AgentProbe/issues)

</div>
