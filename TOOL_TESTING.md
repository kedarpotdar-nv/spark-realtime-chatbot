# Testing Tool Calling

## Quick Test Guide

### 1. Start the Server

Make sure your LLM server (gpt-oss-120b) is running:
```bash
# Start LLM server (adjust command based on your setup)
./llama-server --model gpt-oss-120b.gguf -ngl 99 --port 8080
```

Then start the voice server:
```bash
# For HTTPS (recommended for microphone access)
./launch-https.sh

# OR for HTTP (if using SSH port forwarding)
./launch.sh
```

### 2. Open the Web Interface

- HTTPS: `https://localhost:8443` (or your Spark hostname)
- HTTP: `http://localhost:8001` (if using SSH port forwarding)

### 3. Test Weather Tool

**Try these voice commands:**

1. **"What's the weather?"**
   - Should trigger `get_weather` tool
   - You should hear: "Looking that up for you..."
   - Then get weather response

2. **"Check the weather in Santa Clara"**
   - Should call tool with location parameter
   - Should return: "55 degrees and sunny"

3. **"What's the weather like in New York?"**
   - Should call tool with different location
   - Should return: "65 degrees and partly cloudy"

### 4. What to Look For in Logs

**Successful tool call should show:**

```
[Voice Session] Starting LLM stream with X messages
[LLM] Request status: 200
[LLM] Stream finished with reason: tool_calls
[Voice Session] ✅ TOOL CALLS DETECTED: 1 tools
  Tool 1: get_weather with args: {"location": "Santa Clara, CA"}
[Voice Session] Executing tool: get_weather with args: {'location': 'Santa Clara, CA'}
[Voice Session] Getting final response after tool execution
[Voice Session] LLM stream completed: X chunks, response length: Y
```

**If tool calling is NOT working, you might see:**

```
[Voice Session] LLM stream completed: X chunks, response length: 0
[Voice Session] WARNING: Empty final_response after X chunks
```

This means the LLM didn't call the tool - check:
- Is the LLM model (gpt-oss-120b) configured correctly?
- Does the model support tool calling?
- Are tools being sent in the request? (check `[LLM] Request status: 200` logs)

### 5. Debugging Tips

**Enable verbose logging** (temporarily) by uncommenting debug lines in `server.py`:

1. In `stream_complete` method (around line 397-400):
   ```python
   # Uncomment these lines:
   if chunk_count <= 3 or finish_reason:
       print(f"[LLM] Chunk {chunk_count} full JSON: {json.dumps(data, indent=2)}")
   ```

2. Check if tools are being sent:
   - Look for `[LLM] Request status: 200` - this means request was sent
   - Check browser console (F12) for any errors

**Common Issues:**

1. **LLM doesn't call tools**
   - Verify model supports OpenAI function calling format
   - Check that `tools` parameter is being sent (look for `[LLM] Request status: 200`)
   - Try asking more explicitly: "Use the weather tool to check the weather"

2. **Tool called but no response**
   - Check `[Voice Session] Executing tool:` logs
   - Verify `execute_tool()` function is working
   - Check for errors in final response generation

3. **"Looking that up" plays but no final answer**
   - Tool executed but LLM didn't generate follow-up response
   - Check `[Voice Session] Getting final response after tool execution` logs
   - May need to adjust LLM temperature or max_tokens

### 6. Test with curl (Advanced)

You can test the LLM directly to verify tool calling:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "What is the weather in Santa Clara?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "The city and state"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "stream": false
  }'
```

Expected response should include `"finish_reason": "tool_calls"` and a `tool_calls` array.

