# llmperf

`llmperf` 是一个面向 OpenAI 兼容接口的轻量命令行工具，支持：

- 发送单次聊天请求并查看流式或非流式输出
- 基于 JSONL 数据集批量压测聊天接口
- 汇总延迟、TTFT、TPOT 等关键基准指标

## 安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -e .
```

安装完成后可使用以下命令查看入口帮助：

```bash
llmperf --help
```

## 命令概览

```bash
llmperf request --help
llmperf bench --help
```

## `request` 命令

`request` 用于向 OpenAI 兼容的 `/v1/chat/completions` 接口或 `/generate` 接口发送一次请求。

### 输入方式

当 URL 不以 `/generate` 结尾时，使用聊天请求模式。以下三种输入方式必须且只能选择一种：

1. `--messages`：直接传入 JSON 数组字符串
2. `--file`：从文件读取 JSON 消息数组
3. `--user`：直接提供用户输入，可搭配 `--system`

当 URL 以 `/generate` 结尾时，使用文本生成模式，必须提供非空 `--text`，且不能同时使用 `--messages`、`--file`、`--user` 或 `--system`。

### 示例

```bash
# 方式 1：直接传消息 JSON
llmperf request \
  --messages '[{"role":"user","content":"hello"}]'

# 方式 2：从文件读取消息
llmperf request \
  --file messages.json

# 方式 3：使用 user/system 快速构造消息
llmperf request \
  --user "请介绍一下这个项目" \
  --system "你是一个简洁的技术助手"
```

```bash
# /generate 文本生成模式
llmperf request \
  --url "http://localhost:8000/generate" \
  --text "请介绍一下这个项目"
```

```bash
# 带采样参数的完整示例
llmperf request \
  --url "http://localhost:8000/v1/chat/completions" \
  --user "请介绍一下这个项目" \
  --system "你是一个简洁的技术助手" \
  --model "qwen" \
  --temperature 0.6 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --max-completion-tokens 512 \
  --enable-thinking \
  --stream
```

### 常用参数

- `--url`：目标 OpenAI 兼容接口地址
- `--text`：`/generate` 文本生成模式的输入文本
- `--model`：请求中携带的模型名
- `--tokenizer-path`：本地 prompt token 估算优先使用的 tokenizer 路径
- `--rid`：可选请求标识
- `--temperature`：采样温度
- `--presence-penalty`：presence penalty
- `--frequency-penalty`：frequency penalty
- `--repetition-penalty`：repetition penalty
- `--max-completion-tokens`：最大生成 token 数
- `--ignore-eos`：忽略 EOS
- `--seed`：随机种子
- `--enable-thinking/--disable-thinking`：是否启用 thinking
- `--stream/--no-stream`：是否使用流式输出
- `--timeout`：请求超时时间

当后端未返回 `prompt_tokens` 时，`request` 和 `bench` 会优先使用本地 tokenizer 对输入进行估算。若同时提供 `--tokenizer-path` 和 `--model`，会优先使用 `--tokenizer-path`。

## `bench` 命令

`bench` 用于从数据集文件中读取请求并批量发送到 OpenAI 兼容接口，输出压测摘要。

### 支持的数据模式

- `openai-jsonl`
- `zss-jsonl`
- `random`

当前实现主要支持 JSONL 文件输入，使用 `--mode` 指定记录解析方式。

### 示例

```bash
llmperf bench \
  --url "http://localhost:8000/v1/chat/completions" \
  --file dataset.jsonl \
  --mode openai-jsonl \
  --num-requests 100 \
  --qps 5 \
  --max-concurrency 20
```

```bash
# 使用覆盖参数批量改写数据集中的请求配置
llmperf bench \
  --file dataset.jsonl \
  --mode zss-jsonl \
  --model "qwen" \
  --temperature 0.7 \
  --max-completion-tokens 256 \
  --enable-thinking \
  --ignore-eos
```

```bash
# 使用随机数据集生成压测请求
llmperf bench \
  --mode random \
  --url "http://localhost:8000/v1/chat/completions" \
  --num-requests 100 \
  --tokenizer-path "/path/to/tokenizer" \
  --model "qwen" \
  --min-input-tokens 128 \
  --max-input-tokens 512 \
  --min-output-tokens 64 \
  --max-output-tokens 256 \
  --seed 1
```

### `random` 模式说明

- `random` 模式不读取 `--file`，而是动态生成请求
- 必须提供 `--num-requests`
- tokenizer 加载优先使用 `--tokenizer-path`，未提供时退回到 `--model`
- 随机模式会为每条请求生成一条随机 `user` 消息
- 每条请求的 `max_completion_tokens` 会在给定输出区间内随机生成
- 随机模式生成的数据默认设置 `ignore_eos=True`

### `random` 模式常用参数

- `--tokenizer-path`：优先使用的 tokenizer 路径或名称
- `--model`：当未提供 `--tokenizer-path` 时，用于加载 tokenizer；同时也会作为请求里的模型名
- `--seed`：随机数据生成种子
- `--min-input-tokens`：单条请求的最小输入 token 数
- `--max-input-tokens`：单条请求的最大输入 token 数
- `--min-output-tokens`：单条请求的最小输出 token 数
- `--max-output-tokens`：单条请求的最大输出 token 数

### 输出内容

`bench` 会输出一份终端摘要，包含：

- 总请求数、成功数、失败数
- 成功率
- 总耗时
- 请求吞吐
- 平均并发度
- `latency`、`ttft`、`tpot` 的均值及分位数

## 数据文件说明

### `openai-jsonl`

每行一个 JSON 对象，至少包含：

```json
{"conversations": [{"role": "user", "content": "hello"}]}
```

### `zss-jsonl`

每行一个 JSON 对象，典型字段包括：

```json
{
  "messages": [{"role": "user", "content": "hello"}],
  "model": "qwen",
  "stream": true,
  "max_tokens": 128,
  "temperature": 0.7,
  "seed": 1,
  "frequency_penalty": 0.0,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "chat_template_kwargs": {"enable_thinking": true}
}
```

## 说明

- `request` 适合联调、单次验证和观察模型输出
- `bench` 适合压测、回归对比和吞吐/延迟分析
- 共享的数据模型和 tokenizer 相关辅助逻辑位于 `llmperf/common/`
- 如需最新参数说明，以 `llmperf request --help` 和 `llmperf bench --help` 为准
