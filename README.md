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


