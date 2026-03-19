# llmperf

`llmperf` 是一个面向 OpenAI 兼容接口的轻量命令行工具，用于快速发起聊天请求、观察流式输出，并为后续性能测试或接口联调提供基础能力。

## 安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -e .
```

安装完成后可使用以下命令：

```bash
llmperf --help
```

## `request` 命令

`request` 用于向 OpenAI 兼容的 `/v1/chat/completions` 接口发送一次请求。


### 查看帮助

```bash
llmperf request --help
```

### 请求方式（三选一）：

1. `--messages` 直接传 JSON 字符串
2. `--file` 从文件读取 JSON 数据
3. 用 `--user` 搭配可选的 `--system` 快速构造消息


### 用法

```bash
# 三种请求模式
llmperf request --messages '[{"role":"user","content":"hello"}]'
llmperf request --file messages.json
llmperf request --user "请介绍一下这个项目"

# 详细示例
llmperf request \
  --user "请介绍一下这个项目" \
  --system "你是一个简洁的技术助手" \
  --url "http://localhost:8000/v1/chat/completions" \
  --model "qwen" \
  --temperature 0.6 \
  --max-tokens 512 \
  --enable-thinking \
  --stream
```

## 重构 TODO

### 目标

- 将 `request` 与 `bench` 的公共能力下沉为通用模型与 backend 抽象。
- 让 `bench` 只负责调度与统计，不直接依赖 OpenAI 协议细节。
- 为后续接入非 OpenAI 规范接口预留清晰扩展点。

### 分步重构

1. 新建 `llmperf/core/models.py`
   - 定义 `ChatRequest`、`ChatResult`、`BenchConfig`。
   - 高层模块后续统一依赖这些中性模型。

2. 新建 `llmperf/backends/base.py`
   - 定义 `ChatBackend` 抽象接口。
   - 让 `request` / `bench` 依赖抽象而不是具体实现。

3. 新建 `llmperf/backends/openai.py`
   - 把当前 OpenAI 协议相关逻辑收敛到这里。
   - 负责请求构造、HTTP 调用、流式与非流式响应解析。

4. 重构 `llmperf/commands/request.py`
   - 保留现有消息解析逻辑。
   - 改为先组装 `ChatRequest`，再调用 `ChatBackend`。

5. 重构 `llmperf/commands/bench.py`
   - 改为接收 `backend`、`BenchConfig`、`list[ChatRequest]`。
   - 只负责调度、限流、收集结果，不处理协议细节。

6. 重构 `llmperf/cli.py`
   - 保留 Typer 参数声明。
   - 在命令函数内部组装配置对象并选择 backend。

7. 拆分数据集读取逻辑
   - 新增 `llmperf/datasets/` 目录。
   - 将现有 `read_zss_file()` 移到独立 reader 中。

8. 增加数据集选择入口
   - 按格式选择 reader。
   - 避免 bench 主流程依赖某一种样本格式。

9. 新增汇总统计模块
   - 新建 `llmperf/core/summary.py`。
   - 输出成功率、失败率、P50/P95/P99 latency、TTFT、吞吐等指标。

10. 清理命名与依赖方向
    - 将高层通用代码中的 OpenAI 命名逐步替换为中性命名。
    - 保持协议特定逻辑只存在于 `backends/openai.py`。

11. 最后增加第二种 backend 做验证
    - 用一个非 OpenAI 接口验证抽象是否合理。
    - 理想状态下只需新增 backend 文件并注册工厂。

### 建议顺序

1. `core/models.py`
2. `backends/base.py`
3. `backends/openai.py`
4. `commands/request.py`
5. `commands/bench.py`
6. `core/summary.py`
7. `datasets/`
8. `cli.py`
9. 第二个 backend 验证

