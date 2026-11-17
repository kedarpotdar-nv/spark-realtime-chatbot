"""Tool definitions and execution for OpenAI-compatible tool calling."""

import json
from typing import Any, Dict, List, Set

from config import PersonalFSConfig


# Available tools (OpenAI format) - registry of all possible tools
ALL_TOOLS = {
    # Hermes Agent background delegation tool
    "delegate_task": {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Delegate a background task to Hermes Agent (Nous Research). "
                "ALWAYS use this when the user wants to generate or create an image, illustration, logo, "
                "poster, photo, artwork, music, audio, song, or other creative media — Hermes already has "
                "skills for those. Also use for building apps, coding/refactoring, multi-step research, "
                "or multi-file artifacts. Hermes runs asynchronously while you keep talking. "
                "Do NOT try to generate images or music yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Clear, detailed task for Hermes. For images include subject, style, "
                            "composition, and any output path preferences."
                        )
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context from the conversation or camera scene."
                    },
                    "category": {
                        "type": "string",
                        "enum": ["coding", "project", "research", "creative", "general"],
                        "description": (
                            "Use 'creative' for image/music/media generation. "
                            "Use coding/project/research for those domains; otherwise general."
                        )
                    }
                },
                "required": ["task"]
            }
        }
    },

    # Personal filesystem / shell tools (Jarvis local mode on DGX Spark)
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List files and folders on the DGX Spark host. "
                "Use for questions like what is in Downloads, Desktop, Documents, or home."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (absolute or relative to home). Defaults to home. Examples: ~/Downloads, /home/user/Projects"
                    }
                },
                "required": []
            }
        }
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the DGX Spark host filesystem and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read (absolute or under home)."
                    }
                },
                "required": ["path"]
            }
        }
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a text file on the DGX Spark host within allowed personal roots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    "search_files": {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files or folders by name under a root directory on the DGX Spark host.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive substring to match in file or folder names."
                    },
                    "root": {
                        "type": "string",
                        "description": "Search root path. Defaults to home."
                    }
                },
                "required": ["query"]
            }
        }
    },
    "run_command": {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command on the DGX Spark host (no Docker sandbox). "
                "Use for quick system checks, listing, scripts, and light automation. "
                "Do not run destructive commands unless the user explicitly asks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute."
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (must be under allowed personal roots). Defaults to home."
                    }
                },
                "required": ["command"]
            }
        }
    },
    "open_path": {
        "type": "function",
        "function": {
            "name": "open_path",
            "description": (
                "Open a file or folder on the DGX Spark desktop with the system opener "
                "(xdg-open). Use when the user asks to open, show, or reveal a file or result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or folder path to open."
                    },
                    "reveal": {
                        "type": "boolean",
                        "description": "If true and path is a file, open the containing folder instead."
                    }
                },
                "required": ["path"]
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


PERSONAL_TOOL_IDS: Set[str] = {
    "list_directory",
    "read_file",
    "write_file",
    "search_files",
    "run_command",
    "open_path",
}

AGENT_TOOL_IDS: Set[str] = {
    "delegate_task",
    "hermes_agent",
    "markdown_assistant",
    "html_assistant",
    "reasoning_assistant",
}


def _personal_tool_allowed(tool_id: str, cfg: PersonalFSConfig) -> bool:
    if not cfg.enabled:
        return False
    if tool_id == "run_command" and not cfg.shell_enabled:
        return False
    return tool_id in PERSONAL_TOOL_IDS


def get_enabled_tools(enabled_tool_ids: List[str]) -> List[Dict[str, Any]]:
    """Get list of tool definitions for enabled tool IDs."""
    tools = []
    seen = set()
    personal_cfg = PersonalFSConfig()
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
            if canonical_id in PERSONAL_TOOL_IDS and not _personal_tool_allowed(canonical_id, personal_cfg):
                continue
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

    elif tool_name in PERSONAL_TOOL_IDS:
        from clients.personal_fs import get_personal_fs

        fs = get_personal_fs()
        if tool_name == "list_directory":
            result = fs.list_directory(arguments.get("path") or str(__import__("pathlib").Path.home()))
        elif tool_name == "read_file":
            result = fs.read_file(arguments.get("path", ""))
        elif tool_name == "write_file":
            result = fs.write_file(arguments.get("path", ""), arguments.get("content", ""))
        elif tool_name == "search_files":
            result = fs.search_files(
                arguments.get("query", ""),
                root=arguments.get("root") or str(__import__("pathlib").Path.home()),
            )
        elif tool_name == "run_command":
            result = await fs.run_command(
                arguments.get("command", ""),
                cwd=arguments.get("cwd"),
            )
        elif tool_name == "open_path":
            result = fs.open_path(
                arguments.get("path", ""),
                reveal=bool(arguments.get("reveal", False)),
            )
        else:
            result = {"success": False, "error": f"Unknown personal tool: {tool_name}"}

        # Return LLM-friendly summary; keep structured payload for UI via wrapper
        summary = fs.format_for_llm(result, tool_name)
        return json.dumps({
            "tool": tool_name,
            "status": "completed" if result.get("success") else "failed",
            "result": result,
            "summary": summary,
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})
