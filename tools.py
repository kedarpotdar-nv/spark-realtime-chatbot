"""Tool definitions and execution for OpenAI-compatible tool calling."""

import json
from typing import Any, Dict, List


# Available tools (OpenAI format) - registry of all possible tools
ALL_TOOLS = {
    # Hermes Agent background delegation tool
    "delegate_task": {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Delegate a complex background task to Hermes Agent (Nous Research). "
                "Use this when the user asks to build real projects or apps, write or refactor code, "
                "perform multi-step research, run workspace/terminal operations, or generate multi-file artifacts. "
                "Hermes runs asynchronously in the background while you continue the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear, detailed description of the coding, project-building, or research task to execute."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context from the conversation or camera scene."
                    },
                    "category": {
                        "type": "string",
                        "enum": ["coding", "project", "research", "general"],
                        "description": "Category of the background task."
                    }
                },
                "required": ["task"]
            }
        }
    },

    # Markdown documentation assistant
    "markdown_assistant": {
        "type": "function",
        "function": {
            "name": "markdown_assistant",
            "description": "A markdown documentation assistant that can write README files, documentation, guides, and other markdown documents. Use this when the user asks to write documentation, create a README, write guides, or produce any markdown content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The documentation task description, e.g. 'Write a README for my project' or 'Create API documentation for the user service'"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context about the project or topic to document"
                    }
                },
                "required": ["task"]
            }
        }
    },

    # HTML live preview assistant
    "html_assistant": {
        "type": "function",
        "function": {
            "name": "html_assistant",
            "description": "An HTML/CSS/JS assistant that generates complete, self-contained interactive web pages and UI components with live preview. Use this when the user asks to build a webpage, design a UI, or create an interactive HTML preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Description of the webpage or component to generate, e.g. 'Build a pricing page with a toggle' or 'Create a dashboard widget'"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional visual or conversation context"
                    }
                },
                "required": ["task"]
            }
        }
    },

    # Nemotron-powered reasoning agent
    "reasoning_assistant": {
        "type": "function",
        "function": {
            "name": "reasoning_assistant",
            "description": "ONLY use for customer data and feature prioritization questions. Has LOCAL DATA FILES with customer feedback and feature requests. Use ONLY when user asks about: customer feedback, feature requests, what to build, prioritization, or roadmap vs customer data. DO NOT use for architecture, system design, caching, performance, or technical questions - answer those directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "The customer data question - e.g. 'What features should we prioritize?' or 'What are customers asking for?'"
                    },
                    "context": {
                        "type": "string",
                        "description": "Any roadmap or plan visible (whiteboard) to compare against customer data. Leave empty if just asking about customer data."
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["general", "comparison", "prioritization", "planning"],
                        "description": "Type: 'prioritization' for feature questions, 'comparison' for roadmap vs customer data"
                    }
                },
                "required": ["problem"]
            }
        }
    },
}


def get_enabled_tools(enabled_tool_ids: List[str]) -> List[Dict[str, Any]]:
    """Get list of tool definitions for enabled tool IDs."""
    tools = []
    seen = set()
    for tool_id in enabled_tool_ids:
        # Support both 'hermes_agent' and 'delegate_task' aliases
        canonical_id = "delegate_task" if tool_id in ("hermes_agent", "delegate_task") else tool_id
        if canonical_id in seen:
            continue
        if canonical_id in ALL_TOOLS and ALL_TOOLS[canonical_id] is not None:
            # If delegate_task is requested, ensure Hermes is configured
            if canonical_id == "delegate_task":
                try:
                    from agents import get_hermes_bridge
                    if not get_hermes_bridge().is_configured():
                        continue
                except Exception:
                    pass
            tools.append(ALL_TOOLS[canonical_id])
            seen.add(canonical_id)
    return tools


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool and return the result as a string."""
    if tool_name in ("delegate_task", "hermes_agent"):
        task = arguments.get("task", "")
        context = arguments.get("context", "")
        category = arguments.get("category", "general")
        return json.dumps({
            "agent_type": "hermes_agent",
            "tool": "delegate_task",
            "task": task,
            "context": context,
            "category": category,
            "status": "dispatched"
        })

    elif tool_name == "markdown_assistant":
        task = arguments.get("task", "")
        context = arguments.get("context", "")
        return json.dumps({
            "agent_type": "markdown_assistant",
            "task": task,
            "context": context,
            "status": "initiated"
        })

    elif tool_name == "html_assistant":
        task = arguments.get("task", "")
        context = arguments.get("context", "")
        return json.dumps({
            "agent_type": "html_assistant",
            "task": task,
            "context": context,
            "status": "initiated"
        })

    elif tool_name == "reasoning_assistant":
        # Nemotron-powered reasoning agent
        problem = arguments.get("problem", "")
        context = arguments.get("context", "")
        analysis_type = arguments.get("analysis_type", "general")
        return json.dumps({
            "agent_type": "reasoning_assistant",
            "problem": problem,
            "context": context,
            "analysis_type": analysis_type,
            "status": "initiated"
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})
