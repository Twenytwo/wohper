# Using Wohper as an agent's local brain

Wohper exposes an OpenAI-compatible API on `http://127.0.0.1:8114/v1`, so
any agent that speaks the OpenAI protocol can drive DeepSeek-V4-Flash
locally - nothing leaves the machine. The shim translates OpenAI
function/tool calls to and from DeepSeek's native DSML tool format, so the
model can actually *act* (call tools), not just chat.

## What works

- **Plain chat**: streaming and non-streaming, with the no-think fast path
  (`"reasoning": false`) and language-adaptive default system prompt.
- **Tool calling**: pass OpenAI `tools` in the request; the shim renders
  them into DeepSeek's DSML schema, and parses the model's DSML tool calls
  back into OpenAI `tool_calls` (with `finish_reason: "tool_calls"`,
  `content: null`). Tolerant parser: an imperfect closing tag from the
  model does not lose the call.
- **Tool results**: send the tool output back as a `{"role": "tool",
  "tool_call_id": "...", "content": "..."}` message; DeepSeek's official
  encoder folds it into the conversation as a `<tool_result>` block, so
  the multi-step agent loop closes.

Tool-calling requests are always answered as one buffered result (a DSML
block is only parseable when complete); if the client asked to stream, the
result is delivered as a single SSE burst. Plain chat still streams live.

## Reality check: speed

The engine serves one request at a time at seconds per token. An agent
makes many calls per task, so a task takes minutes, not seconds. This is a
good fit for **asynchronous, background work** - message it, it works while
you do something else, it replies when done - and a poor fit for snappy
real-time loops. The agent layer adds *capability* (actions), not speed.
Speed is the GPU milestone.

## Example: OpenClaw

[OpenClaw](https://openclaw.ai) is an open-source local agent that can use
a custom OpenAI-compatible provider. Point it at Wohper by adding a custom
provider to its models config (`~/.openclaw/models.json`), with the base
URL set to the local shim:

```json
{
  "providers": {
    "wohper": {
      "baseUrl": "http://127.0.0.1:8114/v1",
      "apiKey": "local",
      "models": [
        { "id": "wohper-deepseek-v4-flash", "name": "Wohper (local DeepSeek-V4)" }
      ]
    }
  }
}
```

Then select `wohper/wohper-deepseek-v4-flash` as the agent model. Start the
Wohper server first (`py -X utf8 tools/wohper_cli.py` boots it), keep it
running, and expect background-paced task times. Check OpenClaw's current
config reference for the exact key names, which can change between
versions.

## Raw API examples

Tool call:

```bash
curl http://127.0.0.1:8114/v1/chat/completions -d '{
  "model": "wohper-deepseek-v4-flash",
  "messages": [{"role": "user", "content": "What is the weather in Rome?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object",
      "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "reasoning": false
}'
# -> finish_reason "tool_calls", tool_calls[0].function = get_weather({"city":"Rome"})
```

The engine and the OpenAI shim are entirely local: `127.0.0.1` is your own
machine, and no request leaves it.
