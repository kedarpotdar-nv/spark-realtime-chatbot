"""Nemotron client for deep reasoning tasks."""

import asyncio
import json
from typing import AsyncGenerator

import aiohttp

from config import NemotronConfig
from .http_session import get_http_manager


class NemotronClient:
    """Client for Nemotron 3 Nano deep reasoning tasks.
    
    Nemotron excels at:
    - Multi-step logical reasoning
    - Complex analysis and comparison
    - Architectural review
    - Risk assessment
    - Planning and prioritization
    """
    
    def __init__(self, cfg: NemotronConfig = None):
        self.cfg = cfg or NemotronConfig()
        print(f"[Nemotron] Initialized at {self.cfg.base_url}")
    
    async def stream_reasoning(
        self, 
        problem: str, 
        context: str = "", 
        analysis_type: str = "general",
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """Stream a reasoning response from Nemotron.
        
        Yields chunks in SSE format: "data: {json}\n\n"
        The JSON contains either:
        - {"thinking": "..."} for reasoning steps
        - {"content": "..."} for final response content
        - {"done": true} when complete
        """
        from prompts import (
            NEMOTRON_REASONING_PROMPT, 
            NEMOTRON_ANALYSIS_PROMPT,
            NEMOTRON_PLANNING_PROMPT,
            NEMOTRON_PRIORITIZATION_PROMPT
        )
        
        # Select appropriate system prompt based on analysis type
        if system_prompt is None:
            if analysis_type == "planning":
                system_prompt = NEMOTRON_PLANNING_PROMPT
            elif analysis_type == "prioritization":
                system_prompt = NEMOTRON_PRIORITIZATION_PROMPT
            elif analysis_type in ["comparison", "risk_assessment", "architecture_review"]:
                system_prompt = NEMOTRON_ANALYSIS_PROMPT
            else:
                system_prompt = NEMOTRON_REASONING_PROMPT
        
        # Build the user message with context
        user_content = f"Problem: {problem}"
        if context:
            user_content = f"{user_content}\n\nContext:\n{context}"
        if analysis_type and analysis_type != "general":
            user_content = f"{user_content}\n\nAnalysis Type: {analysis_type}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "stream": True
        }
        
        print(f"[Nemotron] Starting reasoning for: {problem[:100]}...")
        print(f"[Nemotron] Analysis type: {analysis_type}")
        print(f"[Nemotron] Context length: {len(context)} chars")
        if context:
            print(f"[Nemotron] Context preview: {context[:200]}...")
        
        try:
            http_manager = get_http_manager()
            session = await http_manager.get_session()
            async with session.post(
                self.cfg.base_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # Longer timeout for reasoning
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[Nemotron] Error {resp.status}: {error_text[:200]}")
                    yield f'data: {{"error": "Nemotron error: {resp.status}"}}\n\n'
                    return

                # Track if we're in thinking mode
                in_thinking = False
                thinking_buffer = ""
                content_buffer = ""
                chunk_num = 0

                async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue
                        
                        if line == "data: [DONE]":
                            # Flush any remaining content
                            if thinking_buffer:
                                yield f'data: {{"thinking": {json.dumps(thinking_buffer)}}}\n\n'
                            if content_buffer:
                                yield f'data: {{"content": {json.dumps(content_buffer)}}}\n\n'
                            yield 'data: {"done": true}\n\n'
                            break
                        
                        try:
                            data = json.loads(line[6:])
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            
                            # Debug: log first few chunks to see format
                            chunk_num += 1
                            if chunk_num <= 3:
                                print(f"[Nemotron] Raw chunk {chunk_num}: {json.dumps(delta)[:200]}")
                            
                            # Nemotron uses separate reasoning_content field for thinking
                            # Check both delta and message (some servers use different formats)
                            reasoning_chunk = delta.get("reasoning_content", "")
                            content_chunk = delta.get("content", "")
                            
                            if chunk_num <= 3:
                                print(f"[Nemotron] Parsed: reasoning='{reasoning_chunk[:50] if reasoning_chunk else ''}', content='{content_chunk[:50] if content_chunk else ''}'...")
                            
                            # Handle reasoning/thinking content
                            if reasoning_chunk:
                                thinking_buffer += reasoning_chunk
                                # Yield thinking in reasonable chunks for streaming UI
                                if len(thinking_buffer) > 100 or "\n" in thinking_buffer:
                                    yield f'data: {{"thinking": {json.dumps(thinking_buffer)}}}\n\n'
                                    thinking_buffer = ""
                            
                            # Handle regular content (conclusion)
                            if content_chunk:
                                # Also check for <think> tags as fallback for other models
                                if "<think>" in content_chunk or "<thinking>" in content_chunk:
                                    in_thinking = True
                                    content_chunk = content_chunk.replace("<think>", "").replace("<thinking>", "")
                                
                                if "</think>" in content_chunk or "</thinking>" in content_chunk:
                                    in_thinking = False
                                    content_chunk = content_chunk.replace("</think>", "").replace("</thinking>", "")
                                    if thinking_buffer:
                                        yield f'data: {{"thinking": {json.dumps(thinking_buffer)}}}\n\n'
                                        thinking_buffer = ""
                                
                                if in_thinking:
                                    thinking_buffer += content_chunk
                                    if len(thinking_buffer) > 100 or "\n" in thinking_buffer:
                                        yield f'data: {{"thinking": {json.dumps(thinking_buffer)}}}\n\n'
                                        thinking_buffer = ""
                                elif content_chunk:
                                    # Regular content (conclusion)
                                    yield f'data: {{"content": {json.dumps(content_chunk)}}}\n\n'
                                
                        except json.JSONDecodeError:
                            continue
                            
        except asyncio.TimeoutError:
            print("[Nemotron] Request timed out")
            yield 'data: {"error": "Reasoning timed out"}\n\n'
        except Exception as e:
            print(f"[Nemotron] Error: {e}")
            import traceback
            traceback.print_exc()
            yield f'data: {{"error": {json.dumps(str(e))}}}\n\n'
    
    async def reason(self, problem: str, context: str = "", analysis_type: str = "general") -> dict:
        """Non-streaming reasoning - returns complete result."""
        thinking = []
        content = []
        
        async for chunk in self.stream_reasoning(problem, context, analysis_type):
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if "thinking" in data:
                        thinking.append(data["thinking"])
                    elif "content" in data:
                        content.append(data["content"])
                except json.JSONDecodeError:
                    pass
        
        return {
            "thinking": "".join(thinking),
            "content": "".join(content)
        }

