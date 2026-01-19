"""LLM client with streaming support for llama.cpp and TensorRT-LLM backends."""

import asyncio
import json
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from config import LLMConfig
from .http_session import get_http_manager


class LlamaCppClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.is_trtllm = cfg.backend.lower() == "trtllm"
        if self.is_trtllm:
            print(f"[LLM] Using TensorRT-LLM backend (trtllm-serve)")
        else:
            print(f"[LLM] Using standard OpenAI-compatible backend")

    def _extract_final_channel(self, content: str) -> str:
        """Extract final channel content from reasoning-style outputs."""
        if "<|channel|>final<|message|>" in content:
            content = content.split("<|channel|>final<|message|>", 1)[1]
        if "<|end|>" in content:
            content = content.split("<|end|>", 1)[0]
        return content.strip()

    async def complete(self, messages: List[Dict[str, Any]]) -> str:
        """Non-streaming completion."""
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        if self.cfg.reasoning_effort and self.cfg.reasoning_effort.lower() != "off":
            payload["reasoning_effort"] = self.cfg.reasoning_effort.lower()

        http_manager = get_http_manager()
        session = await http_manager.get_session()
        async with session.post(self.cfg.base_url, json=payload) as resp:
            data = await resp.json()

        raw = data["choices"][0]["message"]["content"]
        return self._extract_final_channel(raw)

    async def stream_complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """Streaming completion - yields text chunks as they arrive.
        Filters out analysis/reasoning chunks and only streams final channel content.
        
        Args:
            messages: List of message dicts
            tools: Optional list of tool definitions in OpenAI format
        """
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
        }
        # Only add reasoning_effort if it's not "off"
        if self.cfg.reasoning_effort and self.cfg.reasoning_effort.lower() != "off":
            payload["reasoning_effort"] = self.cfg.reasoning_effort.lower()
        
        # Add tools if provided (OpenAI format)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"  # Let model decide when to use tools
            print(f"[LLM] Sending request with {len(tools)} tool(s): {[t['function']['name'] for t in tools]}")

        import time
        request_start = time.perf_counter()
        ttft_recorded = False
        ttft_ms = 0

        http_manager = get_http_manager()
        session = await http_manager.get_session()
        try:
            async with session.post(self.cfg.base_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                print(f"[LLM] Request status: {resp.status}")
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[LLM] Server error response: {error_text[:500]}")
                    yield f"data: {json.dumps({'error': f'LLM server error {resp.status}: {error_text[:200]}'})}\n\n"
                    return

                buffer = ""
                accumulated_content = ""  # Accumulate full response for final extraction
                in_final_channel = False  # Track if we're in final channel
                final_marker = "<|channel|>final<|message|>"
                end_marker = "<|end|>"
                line_count = 0
                chunk_count = 0
                # Accumulate tool calls (OpenAI streaming format)
                accumulated_tool_calls = {}  # Map from tool_call index to tool_call dict

                print(f"[LLM] Starting to read stream...")
                try:
                    async for line in resp.content:
                        line_count += 1
                        if line_count <= 5:
                            print(f"[LLM] Line {line_count}: {repr(line[:200]) if line else 'empty'}")
                        if not line:
                            continue
                        
                        line_decoded = line.decode('utf-8', errors='ignore')
                        buffer += line_decoded
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line_str, buffer = buffer.split('\n', 1)
                            line_str = line_str.strip()
                            
                            # Handle different streaming formats
                            if not line_str:
                                continue
                            
                            # Check for SSE format (data: prefix) - standard OpenAI format
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                            elif self.is_trtllm:
                                # TensorRT-LLM might not use 'data: ' prefix
                                # Try parsing as JSON directly (trtllm-serve may stream raw JSON)
                                data_str = line_str
                                if chunk_count == 0:
                                    print(f"[LLM] TRTLLM: First chunk format: {repr(line_str[:200])}")
                            else:
                                continue  # Skip non-SSE lines for standard backend
                            
                            if data_str == '[DONE]':
                                return
                            
                            try:
                                data = json.loads(data_str)
                                chunk_count += 1
                                
                                # Log first few chunks for debugging trtllm format
                                if self.is_trtllm and chunk_count <= 3:
                                    print(f"[LLM] TRTLLM chunk {chunk_count} keys: {list(data.keys())}")
                                    if "choices" in data and data.get("choices"):
                                        print(f"[LLM] TRTLLM choices structure: {json.dumps(data.get('choices', [{}])[0] if data.get('choices') else {}, indent=2)[:300]}")
                                
                                # Handle empty choices array
                                choices = data.get("choices", [])
                                if not choices:
                                    # Skip chunks with empty choices (can happen with some backends)
                                    continue
                                
                                choice = choices[0]
                                finish_reason = choice.get("finish_reason")
                                delta = choice.get("delta", {})
                                
                                # TensorRT-LLM might use different delta structure
                                # Check for alternative field names
                                if self.is_trtllm and not delta and "content" in data:
                                    # trtllm might put content directly in data
                                    delta = {"content": data.get("content", "")}
                                elif self.is_trtllm and not delta and "text" in data:
                                    # trtllm might use "text" instead of "content"
                                    delta = {"content": data.get("text", "")}
                                
                                # Debug logging disabled - uncomment for debugging
                                # if chunk_count <= 3 or finish_reason:
                                #     print(f"[LLM] Chunk {chunk_count} full JSON: {json.dumps(data, indent=2)}")
                                # if chunk_count <= 3:
                                #     print(f"[LLM] Chunk {chunk_count} finish_reason: {finish_reason}, delta keys: {list(delta.keys())}")
                                
                                # Separate handling for content vs reasoning_content
                                actual_content = delta.get("content")  # Actual final content
                                # trtllm uses "reasoning" instead of "reasoning_content"
                                reasoning_content = delta.get("reasoning_content") or delta.get("reasoning", "")
                                tool_calls = delta.get("tool_calls")  # Tool calls (OpenAI format)
                                
                                # TRTLLM debug logging (first 3 chunks only)
                                if self.is_trtllm and chunk_count <= 3:
                                    print(f"[TRTLLM] Chunk {chunk_count}: content={bool(actual_content)}, reasoning={bool(reasoning_content)}, tool_calls={bool(tool_calls)}")
                                
                                # Check if stream is done
                                if finish_reason:
                                    print(f"[LLM] Stream finished with reason: {finish_reason}, accumulated: {len(accumulated_content)} chars")
                                    # TRTLLM: Log accumulated content summary
                                    if self.is_trtllm and accumulated_content:
                                        print(f"[TRTLLM] Accumulated {len(accumulated_content)} chars of reasoning")
                                        # Note: trtllm-serve doesn't support OpenAI-style tool calls
                                        # Tool calls only work with llama.cpp backend
                                    
                                    # If finish_reason is "tool_calls", we have complete tool calls to execute
                                    if finish_reason == "tool_calls" and accumulated_tool_calls:
                                        # Yield tool calls for execution
                                        tool_calls_list = [accumulated_tool_calls[idx] for idx in sorted(accumulated_tool_calls.keys())]
                                        yield f"data: {json.dumps({'tool_calls_complete': tool_calls_list})}\n\n"
                                        return  # Return early - tool execution will happen in caller
                                    # Process current chunk first, then let loop exit naturally
                                    # The end-of-stream logic will run after the loop exits
                                
                                # Debug logging disabled - uncomment for debugging
                                # if chunk_count <= 3:
                                #     print(f"[LLM] Chunk {chunk_count} delta: {delta}")
                                #     print(f"[LLM]   - actual_content: {repr(actual_content[:50]) if actual_content else 'None'}")
                                #     print(f"[LLM]   - reasoning_content: {repr(reasoning_content[:50]) if reasoning_content else 'None'}")
                                #     print(f"[LLM]   - tool_calls: {tool_calls}")
                                
                                # Handle tool calls in delta (OpenAI streaming format)
                                # Tool calls come in chunks: delta.tool_calls is an array
                                # Each element has: index, id, type, function (with name and arguments)
                                # Arguments come as partial JSON strings that need to be accumulated
                                if tool_calls:
                                    for tool_call_delta in tool_calls:
                                        idx = tool_call_delta.get("index")
                                        if idx is not None:
                                            if idx not in accumulated_tool_calls:
                                                accumulated_tool_calls[idx] = {
                                                    "id": tool_call_delta.get("id", ""),
                                                    "type": tool_call_delta.get("type", "function"),
                                                    "function": {"name": "", "arguments": ""}
                                                }
                                            # Accumulate function name and arguments
                                            if "function" in tool_call_delta:
                                                func_delta = tool_call_delta["function"]
                                                if "name" in func_delta:
                                                    accumulated_tool_calls[idx]["function"]["name"] = func_delta["name"]
                                                if "arguments" in func_delta:
                                                    accumulated_tool_calls[idx]["function"]["arguments"] += func_delta["arguments"]
                                
                                # Only accumulate actual content, not reasoning
                                if actual_content:
                                    # Record TTFT on first content
                                    if not ttft_recorded:
                                        ttft_ms = (time.perf_counter() - request_start) * 1000
                                        ttft_recorded = True
                                        print(f"[LLM] ⏱️ TTFT: {ttft_ms:.0f}ms")
                                    accumulated_content += actual_content
                                    
                                    # Harmony API (trtllm-serve) tool call detection during streaming
                                    if self.is_trtllm:
                                        harmony_commentary_marker = "<|channel|>commentary<|message|>"
                                        if harmony_commentary_marker in accumulated_content:
                                            # Extract JSON after the marker
                                            marker_pos = accumulated_content.find(harmony_commentary_marker)
                                            json_start = marker_pos + len(harmony_commentary_marker)
                                            json_str = accumulated_content[json_start:].strip()
                                            
                                            # Try to extract complete JSON
                                            try:
                                                # Find complete JSON object
                                                brace_count = 0
                                                json_end = 0
                                                in_string = False
                                                escape_next = False
                                                for i, char in enumerate(json_str):
                                                    if escape_next:
                                                        escape_next = False
                                                        continue
                                                    if char == '\\':
                                                        escape_next = True
                                                        continue
                                                    if char == '"' and not escape_next:
                                                        in_string = not in_string
                                                    if not in_string:
                                                        if char == '{':
                                                            brace_count += 1
                                                        elif char == '}':
                                                            brace_count -= 1
                                                            if brace_count == 0:
                                                                json_end = i + 1
                                                                break
                                                
                                                if json_end > 0:
                                                    tool_call_json = json_str[:json_end]
                                                    tool_data = json.loads(tool_call_json)
                                                    print(f"[LLM] Harmony API tool call detected during streaming: {tool_data}")
                                                    
                                                    # Convert Harmony format to OpenAI tool call format
                                                    # Detect which tool based on JSON keys and available tools
                                                    tool_name = None
                                                    
                                                    # Method 1: Check if tool_data explicitly contains tool name
                                                    if "name" in tool_data:
                                                        tool_name = tool_data.pop("name")
                                                    elif "function" in tool_data:
                                                        tool_name = tool_data.pop("function")
                                                    
                                                    # Method 2: Infer from parameter keys
                                                    if not tool_name:
                                                        if "context" in tool_data:
                                                            tool_name = "markdown_assistant"

                                                    # Method 3: Check which tools are available and match
                                                    if not tool_name and tools:
                                                        available_tool_names = [t.get("function", {}).get("name", "") for t in tools]
                                                        # Prefer markdown_assistant if it's available and task mentions doc/readme/markdown
                                                        task_lower = tool_data.get("task", "").lower()
                                                        if "markdown_assistant" in available_tool_names:
                                                            if any(kw in task_lower for kw in ["readme", "documentation", "markdown", "document", "guide", "wiki"]):
                                                                tool_name = "markdown_assistant"

                                                    # Default fallback
                                                    if not tool_name:
                                                        tool_name = "markdown_assistant"
                                                    
                                                    print(f"[LLM] Harmony API detected tool: {tool_name}")
                                                    
                                                    openai_tool_call = {
                                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_name,
                                                            "arguments": json.dumps(tool_data)
                                                        }
                                                    }
                                                    
                                                    # Yield as tool_calls_complete and return
                                                    yield f"data: {json.dumps({'tool_calls_complete': [openai_tool_call]})}\n\n"
                                                    return  # Return early - tool execution will happen in caller
                                            except (json.JSONDecodeError, ValueError) as e:
                                                # JSON might be incomplete, continue accumulating
                                                if chunk_count <= 5:
                                                    print(f"[LLM] Harmony API tool call JSON incomplete (will retry): {json_str[:100]}")
                                    
                                    # Check if we've entered the final channel
                                    if final_marker in accumulated_content:
                                        if not in_final_channel:
                                            # Just entered final channel
                                            marker_pos = accumulated_content.rfind(final_marker)
                                            if marker_pos != -1:
                                                in_final_channel = True
                                                # Extract content after marker
                                                content_after_marker = accumulated_content[marker_pos + len(final_marker):]
                                                # Check for end marker
                                                if end_marker in content_after_marker:
                                                    content_after_marker = content_after_marker.split(end_marker, 1)[0]
                                                    in_final_channel = False
                                                # Yield what we have
                                                if content_after_marker:
                                                    yield f"data: {json.dumps({'content': content_after_marker})}\n\n"
                                        elif in_final_channel:
                                            # Already in final channel - check for end marker in this chunk
                                            if actual_content and end_marker in actual_content:
                                                # End marker found - extract final part
                                                final_part = actual_content.split(end_marker, 1)[0]
                                                if final_part:
                                                    yield f"data: {json.dumps({'content': final_part})}\n\n"
                                                in_final_channel = False
                                            elif actual_content:
                                                # Continue streaming final channel content
                                                yield f"data: {json.dumps({'content': actual_content})}\n\n"
                                    else:
                                        # No channel markers found - this LLM doesn't use channel markers
                                        # Only yield actual content, not reasoning_content
                                        if actual_content:  # Only yield if it's actual content, not reasoning
                                            yield f"data: {json.dumps({'content': actual_content})}\n\n"
                                        # Don't yield reasoning_content chunks - wait for final content
                                elif reasoning_content:
                                    # Reasoning content - accumulate for potential analysis
                                    accumulated_content += reasoning_content
                                    # Don't yield reasoning content yet - wait for actual content or end of stream
                                    
                            except json.JSONDecodeError as e:
                                if line_count <= 10:
                                    print(f"[LLM] JSON decode error on line {line_count}: {repr(line_str[:100])}, error: {e}")
                                continue
                
                    print(f"[LLM] Stream ended: {chunk_count} chunks processed, {len(accumulated_content)} chars accumulated")
                    # Debug logging disabled - uncomment for debugging
                    # print(f"[LLM] Full accumulated content: {repr(accumulated_content)}")
                    
                    # Harmony API (trtllm-serve) uses <|channel|>commentary<|message|> markers with JSON tool calls
                    if self.is_trtllm and accumulated_content:
                        harmony_commentary_marker = "<|channel|>commentary<|message|>"
                        if harmony_commentary_marker in accumulated_content:
                            # Extract JSON after the marker
                            marker_pos = accumulated_content.find(harmony_commentary_marker)
                            json_start = marker_pos + len(harmony_commentary_marker)
                            json_str = accumulated_content[json_start:].strip()
                            
                            # Try to extract complete JSON (might be truncated)
                            # Look for complete JSON object
                            try:
                                # Try to find the end of JSON (could be incomplete)
                                brace_count = 0
                                json_end = 0
                                in_string = False
                                escape_next = False
                                for i, char in enumerate(json_str):
                                    if escape_next:
                                        escape_next = False
                                        continue
                                    if char == '\\':
                                        escape_next = True
                                        continue
                                    if char == '"' and not escape_next:
                                        in_string = not in_string
                                    if not in_string:
                                        if char == '{':
                                            brace_count += 1
                                        elif char == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                json_end = i + 1
                                                break
                                
                                if json_end > 0:
                                    tool_call_json = json_str[:json_end]
                                    tool_data = json.loads(tool_call_json)
                                    print(f"[LLM] Harmony API tool call detected: {tool_data}")
                                    
                                    # Convert Harmony format to OpenAI tool call format
                                    # Harmony format: {"task": "...", "codebase_path": "..."}
                                    # OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
                                    
                                    # Determine tool name from the context
                                    tool_name = "markdown_assistant"  # Default fallback
                                    if tools:
                                        # Check which tool matches the structure
                                        for tool_def in tools:
                                            func_name = tool_def.get("function", {}).get("name", "")
                                            if func_name == "markdown_assistant":
                                                tool_name = func_name
                                                break
                                    
                                    # Create OpenAI-format tool call
                                    openai_tool_call = {
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(tool_data)
                                        }
                                    }
                                    
                                    # Yield as tool_calls_complete
                                    yield f"data: {json.dumps({'tool_calls_complete': [openai_tool_call]})}\n\n"
                                    return  # Return early - tool execution will happen in caller
                            except (json.JSONDecodeError, ValueError) as e:
                                print(f"[LLM] Failed to parse Harmony API tool call JSON: {e}")
                                print(f"[LLM] JSON string: {json_str[:200]}")
                    
                    # If we accumulated content but never found channel markers, check if it's reasoning or actual content
                    # Reasoning models output reasoning first, then final content
                    # But tool calls might be embedded in reasoning content
                    if accumulated_content and not in_final_channel and final_marker not in accumulated_content:
                        # Check if accumulated_content contains a tool call (JSON with "tool" key)
                        stripped = accumulated_content.strip()
                        if stripped.startswith("{") and "tool" in stripped:
                            # Looks like a tool call JSON - yield it
                            # Debug logging disabled
                            # print(f"[LLM] Accumulated content looks like tool call: {accumulated_content[:200]}")
                            yield f"data: {json.dumps({'content': accumulated_content})}\n\n"
                        else:
                            # Might be reasoning with embedded tool call - try to extract JSON
                            import re
                            # Look for JSON objects with "tool" key in the reasoning
                            # Try multiple patterns to catch different JSON formats
                            json_patterns = [
                                r'\{[^{}]*"tool"[^{}]*\}',  # Simple pattern
                                r'\{[^{}]*"tool"[^{}]*"[^"]*"[^{}]*\}',  # With quoted values
                                r'\{"tool"\s*:\s*"[^"]+"[^}]*\}',  # More specific
                            ]
                            json_matches = []
                            for pattern in json_patterns:
                                matches = re.findall(pattern, accumulated_content)
                                if matches:
                                    json_matches.extend(matches)
                            
                            if json_matches:
                                # Found tool call JSON in reasoning
                                tool_json = json_matches[-1]  # Take the last match (most complete)
                                print(f"[LLM] Extracted tool call from reasoning: {tool_json}")
                                yield f"data: {json.dumps({'content': tool_json})}\n\n"
                            else:
                                # Check if reasoning mentions tools/filesystem operations
                                # Use reasoning as response for voice-to-voice conversations
                                # This allows normal conversations to work even when model only outputs reasoning
                                # Debug logging disabled
                                # print(f"[LLM] Using reasoning as response: {accumulated_content[:200]}")
                                yield f"data: {json.dumps({'content': accumulated_content})}\n\n"
                    elif not accumulated_content:
                        print(f"[LLM] No content accumulated at all - stream ended with no content")

                    # Log total time
                    total_ms = (time.perf_counter() - request_start) * 1000
                    tokens = len(accumulated_content.split()) if accumulated_content else 0
                    print(f"[LLM] ⏱️ Total: {total_ms:.0f}ms, TTFT: {ttft_ms:.0f}ms, ~{tokens} tokens")
                except Exception as stream_error:
                    print(f"[LLM] Error reading stream: {stream_error}")
                    import traceback
                    traceback.print_exc()
                    if 'accumulated_content' in locals() and accumulated_content:
                        yield f"data: {json.dumps({'content': accumulated_content})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': f'Stream error: {stream_error}'})}\n\n"
        except asyncio.TimeoutError:
            print(f"[LLM] Request timeout after 60 seconds")
            yield f"data: {json.dumps({'error': 'LLM request timeout'})}\n\n"
        except Exception as e:
            print(f"[LLM] Exception in stream_complete: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

