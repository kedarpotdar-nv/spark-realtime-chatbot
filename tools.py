"""Tool definitions and execution for OpenAI-compatible tool calling."""

import json
from typing import Any, Dict, List


# Available tools (OpenAI format) - registry of all possible tools
ALL_TOOLS = {
    # Agents (complex workflows)
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
    for tool_id in enabled_tool_ids:
        if tool_id in ALL_TOOLS and ALL_TOOLS[tool_id] is not None:
            tools.append(ALL_TOOLS[tool_id])
    return tools


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool and return the result as a string."""
    if tool_name == "markdown_assistant":
        task = arguments.get("task", "")
        context = arguments.get("context", "")
        return json.dumps({
            "agent_type": "markdown_assistant",
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
