"""Vision Language Model client for Qwen3-VL via llama.cpp."""

import json
import re
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

from config import VLMConfig
from .http_session import get_http_manager


class VLMClient:
    """Client for Vision Language Model (Qwen3-VL via llama.cpp)"""
    
    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        print(f"[VLM] Using {cfg.model} at {cfg.base_url}")
    
    async def analyze_image(self, image_base64: str, prompt: str, system_prompt: str = None, tools: list = None, history: list = None) -> dict:
        """Analyze an image with the VLM and return response with potential tool calls.

        Args:
            image_base64: Base64-encoded image
            prompt: User's text prompt for this turn
            system_prompt: System prompt for the VLM
            tools: Optional list of tool definitions
            history: Optional list of previous conversation messages (text-only, no images)
        """

        if system_prompt is None:
            system_prompt = "You are a helpful visual assistant. Describe what you see and answer questions about images."

        # Build messages: system + history + current (with image)
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (text-only)
        if history:
            for msg in history:
                # Skip system messages and ensure text-only content
                if msg.get("role") == "system":
                    continue
                content = msg.get("content", "")
                # Handle multimodal content (extract text only)
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    content = " ".join(text_parts)
                if content:
                    messages.append({"role": msg["role"], "content": content})
            print(f"[VLM] Including {len(messages) - 1} history messages")

        # Add current user message with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        })
        
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            http_manager = get_http_manager()
            session = await http_manager.get_session()
            async with session.post(
                self.cfg.base_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[VLM] Error {resp.status}: {error_text[:200]}")
                    return {"content": f"VLM error: {resp.status}", "tool_calls": []}

                result = await resp.json()

                # Extract response
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                # Check for text-based tool calls in content (Qwen format)
                if content and ("<tool_call>" in content or '"name":' in content):
                    try:
                        # Try to parse tool call from content
                        import re
                        tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                        if tool_match:
                            tool_json = json.loads(tool_match.group(1))
                            tool_calls = [{
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": tool_json.get("name", ""),
                                    "arguments": json.dumps(tool_json.get("arguments", tool_json))
                                }
                            }]
                            # Remove tool call from content
                            content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    except Exception as e:
                        print(f"[VLM] Error parsing text tool call: {e}")

                print(f"[VLM] Response: {len(content) if content else 0} chars, {len(tool_calls) if tool_calls else 0} tool calls")
                return {"content": content or "", "tool_calls": tool_calls or []}

        except Exception as e:
            print(f"[VLM] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"content": f"VLM error: {str(e)}", "tool_calls": []}

    async def stream_analyze_image(self, image_base64: str, prompt: str, system_prompt: str = None, history: list = None):
        """Stream VLM response for an image analysis request.

        Yields text chunks as they arrive. Does not support tool calls in streaming mode.

        Args:
            image_base64: Base64-encoded image
            prompt: User's text prompt for this turn
            system_prompt: System prompt for the VLM
            history: Optional list of previous conversation messages (text-only)

        Yields:
            str: Text chunks as they arrive
        """
        import time

        if system_prompt is None:
            system_prompt = "You are a helpful visual assistant. Describe what you see and answer questions about images."

        # Build messages: system + history + current (with image)
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (text-only)
        if history:
            for msg in history:
                if msg.get("role") == "system":
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    content = " ".join(text_parts)
                if content:
                    messages.append({"role": msg["role"], "content": content})
            print(f"[VLM Stream] Including {len(messages) - 1} history messages")

        # Add current user message with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        })

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": True
        }

        request_start = time.perf_counter()
        ttft_recorded = False

        try:
            http_manager = get_http_manager()
            session = await http_manager.get_session()
            async with session.post(
                self.cfg.base_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[VLM Stream] Error {resp.status}: {error_text[:200]}")
                    yield f"VLM error: {resp.status}"
                    return

                # Parse SSE stream
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if not line or not line.startswith('data: '):
                        continue

                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')

                        if content:
                            if not ttft_recorded:
                                ttft_ms = (time.perf_counter() - request_start) * 1000
                                print(f"[VLM Stream] TTFT: {ttft_ms:.0f}ms")
                                ttft_recorded = True
                            yield content

                    except json.JSONDecodeError:
                        continue

                total_ms = (time.perf_counter() - request_start) * 1000
                print(f"[VLM Stream] Complete in {total_ms:.0f}ms")

        except Exception as e:
            print(f"[VLM Stream] Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"VLM error: {str(e)}"

