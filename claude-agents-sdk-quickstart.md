# Claude Agents SDK Quickstart & Cheat Sheet

> A comprehensive guide for building AI agents with Python using the Claude Agent SDK.

---

## Table of Contents

- [[#Installation]]
- [[#Authentication]]
- [[#Core Concepts]]
- [[#Basic Agent Creation]]
- [[#Configuration Options]]
- [[#Built-in Tools]]
- [[#Custom Tools]]
- [[#Hooks System]]
- [[#Multi-Agent Orchestration]]
- [[#Model Configuration]]
- [[#Error Handling]]
- [[#Session Management]]
- [[#Best Practices]]
- [[#Quick Reference]]

---

## Installation

```bash
# Install via pip (requires Python 3.10+)
pip install claude-agent-sdk

# Verify installation
python -c "from claude_agent_sdk import query; print('SDK installed successfully')"
```

---

## Authentication

### Option 1: API Key (Recommended for Development)

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

### Option 2: AWS Bedrock

```bash
export CLAUDE_CODE_USE_BEDROCK=1
# Configure AWS credentials via AWS CLI
```

### Option 3: Google Vertex AI

```bash
export CLAUDE_CODE_USE_VERTEX=1
# Configure Google Cloud credentials
```

---

## Core Concepts

### Architecture Overview

```
┌─────────────────────────────────────────┐
│        Claude Agent SDK                 │
├─────────────────────────────────────────┤
│  ┌───────────────┐  ┌─────────────────┐ │
│  │   query()     │  │ ClaudeSDKClient │ │
│  │  (Stateless)  │  │   (Stateful)    │ │
│  └───────────────┘  └─────────────────┘ │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│     Agent Loop & Tools                  │
├─────────────────────────────────────────┤
│  • File Operations (Read, Write, Edit)  │
│  • Code Execution (Bash)                │
│  • Search (Grep, Glob, WebSearch)       │
│  • Custom Tools (MCP Servers)           │
│  • Hooks (Pre/Post Tool Execution)      │
└─────────────────────────────────────────┘
```

### Two Interaction Patterns

| Pattern | Use Case | Features |
|---------|----------|----------|
| `query()` | One-off tasks, simple automation | Stateless, no session persistence |
| `ClaudeSDKClient` | Multi-turn conversations, complex workflows | Stateful, full hook/tool support |

---

## Basic Agent Creation

### Simple Query (Stateless)

```python
import anyio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        model="claude-opus-4-5-20251101"
    )

    async for message in query(prompt="Create a hello world script", options=options):
        print(message)

anyio.run(main())
```

### Interactive Session (Stateful)

```python
import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        system_prompt="You are an expert Python developer",
        permission_mode="acceptEdits",
    )

    async with ClaudeSDKClient(options=options) as client:
        # First interaction
        await client.query("Create a calculator module")
        async for msg in client.receive_response():
            print(msg)

        # Follow-up (maintains context)
        await client.query("Add unit tests for it")
        async for msg in client.receive_response():
            print(msg)

anyio.run(main())
```

---

## Configuration Options

```python
from claude_agent_sdk import ClaudeAgentOptions
from pathlib import Path

options = ClaudeAgentOptions(
    # === Tools ===
    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    disallowed_tools=["WebSearch"],  # Explicitly block

    # === Behavior ===
    system_prompt="Custom instructions for your agent",
    permission_mode="acceptEdits",  # Auto-approve file edits

    # === Model ===
    model="claude-opus-4-5-20251101",  # or "opus" alias

    # === Session ===
    max_turns=20,
    continue_conversation=False,
    resume=None,  # Session ID to resume

    # === Working Directory ===
    cwd=Path("/path/to/project"),

    # === Budget ===
    max_budget_usd=10.0,

    # === Configuration Sources ===
    setting_sources=["project"],  # Load .claude/settings.json

    # === Hooks ===
    hooks={},  # See Hooks section

    # === MCP Servers ===
    mcp_servers={},  # See Custom Tools section
)
```

---

## Built-in Tools

### File Operations

| Tool | Description | Example Use |
|------|-------------|-------------|
| `Read` | Read file contents | Reading source code |
| `Write` | Create new files | Creating new modules |
| `Edit` | Modify existing files | Refactoring code |

### Shell & Search

| Tool | Description | Example Use |
|------|-------------|-------------|
| `Bash` | Execute shell commands | Running tests, builds |
| `Glob` | Find files by pattern | `**/*.py` |
| `Grep` | Search file contents | Finding function definitions |

### Web & Research

| Tool | Description | Example Use |
|------|-------------|-------------|
| `WebSearch` | Search the internet | Looking up documentation |
| `WebFetch` | Fetch web page content | Reading API docs |

### Special

| Tool | Description | Example Use |
|------|-------------|-------------|
| `Task` | Spawn subagents | Delegating specialized tasks |
| `NotebookEdit` | Edit Jupyter notebooks | Data science workflows |
| `TodoRead` / `TodoWrite` | Manage task lists | Tracking work items |

---

## Custom Tools

### Creating Tools with `@tool` Decorator

```python
from claude_agent_sdk import tool, create_sdk_mcp_server
from typing import Any

@tool("greet", "Greet a user by name", {"name": str})
async def greet_user(args: dict[str, Any]) -> dict:
    name = args.get("name", "Guest")
    return {
        "content": [
            {"type": "text", "text": f"Hello, {name}!"}
        ]
    }

@tool("calculate", "Perform math operations", {
    "operation": str,  # add, subtract, multiply, divide
    "a": float,
    "b": float
})
async def calculator(args: dict[str, Any]) -> dict:
    op = args.get("operation")
    a, b = args.get("a", 0), args.get("b", 0)

    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }

    result = operations.get(op, "Unknown operation")
    return {"content": [{"type": "text", "text": f"Result: {result}"}]}
```

### Creating an MCP Server

```python
from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    ClaudeAgentOptions,
    ClaudeSDKClient
)

# Define tools
@tool("fetch_weather", "Get weather for a city", {"city": str})
async def fetch_weather(args):
    city = args.get("city")
    # In reality, call weather API
    return {"content": [{"type": "text", "text": f"Weather in {city}: Sunny, 72°F"}]}

@tool("get_time", "Get current time", {})
async def get_time(args):
    from datetime import datetime
    return {"content": [{"type": "text", "text": f"Current time: {datetime.now()}"}]}

# Create MCP server
weather_server = create_sdk_mcp_server(
    name="weather-service",
    version="1.0.0",
    tools=[fetch_weather, get_time]
)

# Use with agent
async def main():
    options = ClaudeAgentOptions(
        mcp_servers={"weather": weather_server},
        allowed_tools=[
            "mcp__weather__fetch_weather",
            "mcp__weather__get_time",
            "Read", "Write"
        ]
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What's the weather in San Francisco?")
        async for msg in client.receive_response():
            print(msg)
```

### External MCP Servers

```python
options = ClaudeAgentOptions(
    mcp_servers={
        # Internal SDK server
        "internal": internal_server,

        # External stdio server
        "external": {
            "type": "stdio",
            "command": "npx",
            "args": ["my-mcp-tool"]
        },

        # HTTP server
        "http": {
            "type": "http",
            "url": "http://localhost:8000/mcp"
        }
    }
)
```

---

## Hooks System

### Available Hook Events

| Event | Trigger | Use Case |
|-------|---------|----------|
| `PreToolUse` | Before tool execution | Security validation |
| `PostToolUse` | After tool execution | Logging, audit |
| `UserPromptSubmit` | User submits prompt | Input validation |
| `Stop` | Agent stops | Cleanup |
| `SubagentStop` | Subagent stops | Coordination |
| `PreCompact` | Before message compaction | Context management |

### Creating Hooks

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from typing import Any

# Security hook - block dangerous commands
async def security_hook(input_data: dict[str, Any], tool_use_id: str | None, context) -> dict:
    if input_data.get("tool_name") != "Bash":
        return {}

    command = input_data.get("tool_input", {}).get("command", "")
    dangerous = ["rm -rf", "sudo", "> /dev/"]

    for pattern in dangerous:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": input_data.get("hook_event_name"),
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked: {pattern}"
                }
            }
    return {}

# Audit hook - log all tool usage
async def audit_hook(input_data: dict[str, Any], tool_use_id: str | None, context) -> dict:
    print(f"[AUDIT] Tool: {input_data.get('tool_name')}, ID: {tool_use_id}")
    return {}

# File protection hook
async def protect_files_hook(input_data: dict[str, Any], tool_use_id: str | None, context) -> dict:
    if input_data.get("tool_name") not in ["Write", "Edit"]:
        return {}

    path = input_data.get("tool_input", {}).get("file_path", "")
    protected = [".env", "secrets.json", "private_key.pem"]

    for p in protected:
        if p in path:
            return {
                "hookSpecificOutput": {
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Protected file: {p}"
                }
            }
    return {}
```

### Registering Hooks

```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Edit", "Bash"],
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[security_hook]),
            HookMatcher(matcher="Write|Edit", hooks=[protect_files_hook]),
            HookMatcher(matcher=None, hooks=[audit_hook]),  # All tools
        ]
    }
)
```

---

## Multi-Agent Orchestration

### Defining Subagents

```python
from claude_agent_sdk import SubagentDefinition

code_reviewer = SubagentDefinition(
    name="code-reviewer",
    description="Reviews code for quality and best practices",
    system_prompt="""You are an expert code reviewer. Check:
    1. Code quality and style
    2. Security vulnerabilities
    3. Best practices
    4. Performance issues
    """,
    allowed_tools=["Read", "Glob", "Grep"],
    model="claude-opus-4-5-20251101"
)

test_runner = SubagentDefinition(
    name="test-runner",
    description="Runs tests and reports coverage",
    system_prompt="You are a testing specialist.",
    allowed_tools=["Bash", "Read"],
    model="claude-sonnet-4-5-20250929"
)

documentarian = SubagentDefinition(
    name="documentarian",
    description="Creates and updates documentation",
    system_prompt="You are a technical writer.",
    allowed_tools=["Read", "Write", "Edit"],
    model="claude-haiku-4-5-20251001"
)
```

### Orchestrating Multiple Agents

```python
async def orchestrated_workflow():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Task"],
        system_prompt="""You are the orchestrator. Delegate tasks:
        - code-reviewer: for code review
        - test-runner: for testing
        - documentarian: for documentation
        """,
        subagents=[code_reviewer, test_runner, documentarian],
        max_turns=20,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("""
        Review my new feature:
        1. Have code-reviewer check the implementation
        2. Have test-runner verify tests pass
        3. Have documentarian update the README
        4. Provide a summary of all findings
        """)

        async for msg in client.receive_response():
            print(msg)
```

---

## Model Configuration

### Available Models

| Model | Alias | Best For |
|-------|-------|----------|
| `claude-opus-4-5-20251101` | `opus` | Complex reasoning, highest quality |
| `claude-sonnet-4-5-20250929` | `sonnet` | Balanced speed and capability |
| `claude-haiku-4-5-20251001` | `haiku` | Fast, simple tasks, cost-effective |

### System Prompt Templates

```python
SYSTEM_PROMPTS = {
    "python_expert": """You are an expert Python developer.
- Use idiomatic Python and follow PEP 8
- Write type hints for all functions
- Include comprehensive docstrings
- Handle edge cases and errors
""",

    "security_reviewer": """You are a security engineer.
- Check for SQL injection, XSS, CSRF
- Identify auth/authz flaws
- Ensure proper input validation
- Rate severity: CRITICAL, HIGH, MEDIUM, LOW
""",

    "technical_writer": """You are a technical writer.
- Use clear, concise language
- Include practical examples
- Add step-by-step instructions
- Explain the "why" not just the "how"
""",
}

options = ClaudeAgentOptions(
    model="opus",
    system_prompt=SYSTEM_PROMPTS["python_expert"],
    allowed_tools=["Read", "Write", "Edit", "Bash"]
)
```

---

## Error Handling

### Exception Hierarchy

```python
from claude_agent_sdk import (
    ClaudeSDKError,        # Base exception
    CLINotFoundError,      # CLI not installed
    CLIConnectionError,    # Connection issues
    ProcessError,          # Process failed
    CLIJSONDecodeError,    # JSON parsing error
)
```

### Comprehensive Error Handling

```python
async def safe_agent_run():
    try:
        async for msg in query(prompt="...", options=options):
            print(msg)

    except CLINotFoundError:
        print("Error: Claude Code CLI not found")
        print("Run: pip install claude-agent-sdk")

    except CLIConnectionError as e:
        print(f"Connection error: {e}")

    except ProcessError as e:
        print(f"Process failed: {e}")
        if hasattr(e, 'exit_code'):
            print(f"Exit code: {e.exit_code}")

    except CLIJSONDecodeError as e:
        print(f"JSON parse error: {e}")

    except ClaudeSDKError as e:
        print(f"SDK error: {e}")
```

### Retry with Exponential Backoff

```python
import anyio

async def query_with_retry(prompt: str, max_retries: int = 3):
    delay = 1.0

    for attempt in range(max_retries):
        try:
            async for msg in query(prompt=prompt, options=options):
                return msg

        except ProcessError as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} in {delay}s")
                await anyio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
```

---

## Session Management

### Capturing Session ID

```python
async def capture_session():
    session_id = None

    async for msg in query(prompt="Analyze codebase", options=options):
        # Session ID may be in message metadata
        print(msg)

    return session_id
```

### Resuming a Session

```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],
    resume="session-abc123",  # Resume previous session
)

async for msg in query(prompt="Continue analysis", options=options):
    print(msg)
```

### Multi-Turn Conversation

```python
async with ClaudeSDKClient(options=options) as client:
    # Turn 1: Understand
    await client.query("Analyze the project architecture")
    async for msg in client.receive_response():
        print(f"Turn 1: {msg}")

    # Turn 2: Context preserved
    await client.query("What are the main scalability issues?")
    async for msg in client.receive_response():
        print(f"Turn 2: {msg}")

    # Turn 3: Build on previous analysis
    await client.query("How would you refactor to fix those?")
    async for msg in client.receive_response():
        print(f"Turn 3: {msg}")
```

---

## Best Practices

### 1. Minimal Tool Set

```python
# GOOD: Only what's needed
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Grep", "Glob"],  # Read-only
)

# AVOID: All tools unnecessarily
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Edit", "Bash", "WebSearch", ...],
)
```

### 2. Use Hooks for Security

```python
# Always validate dangerous operations
hooks={
    "PreToolUse": [
        HookMatcher(matcher="Bash", hooks=[security_hook]),
        HookMatcher(matcher="Write|Edit", hooks=[protect_files_hook]),
    ]
}
```

### 3. Use CLAUDE.md for Context

```markdown
# CLAUDE.md in your project root

## Architecture
- Backend: Python FastAPI
- Database: PostgreSQL

## Key Files
- src/main.py - Entry point
- src/api/ - API endpoints

## Development
\`\`\`bash
pip install -r requirements.txt
pytest
\`\`\`
```

### 4. Delegate to Specialists

```python
system_prompt="""Orchestrate tasks:
- code-reviewer: code quality
- test-runner: testing
- documenter: documentation
"""
```

### 5. Budget Management

```python
options = ClaudeAgentOptions(
    model="haiku",  # Cost-effective
    max_budget_usd=5.0,
    max_turns=20,
)
```

### 6. Logging and Monitoring

```python
async def logging_hook(input_data, tool_use_id, context):
    import logging
    logging.info(f"Tool: {input_data.get('tool_name')}")
    return {}
```

---

## Quick Reference

### Common Patterns

```python
# === Simple Query ===
async for msg in query(prompt="...", options=options):
    print(msg)

# === Interactive Session ===
async with ClaudeSDKClient(options=options) as client:
    await client.query("...")
    async for msg in client.receive_response():
        print(msg)

# === Custom Tool ===
@tool("name", "description", {"param": type})
async def my_tool(args):
    return {"content": [{"type": "text", "text": "result"}]}

# === Hook ===
async def my_hook(input_data, tool_use_id, context):
    return {"hookSpecificOutput": {...}}
```

### Configuration Cheat Sheet

| Option | Type | Description |
|--------|------|-------------|
| `allowed_tools` | `list[str]` | Tools agent can use |
| `system_prompt` | `str` | Custom instructions |
| `model` | `str` | Model ID or alias |
| `permission_mode` | `str` | `"acceptEdits"` for auto-approve |
| `cwd` | `Path` | Working directory |
| `max_turns` | `int` | Max conversation turns |
| `max_budget_usd` | `float` | Spending limit |
| `hooks` | `dict` | Pre/post tool hooks |
| `mcp_servers` | `dict` | Custom tool servers |
| `subagents` | `list` | Subagent definitions |
| `resume` | `str` | Session ID to resume |

### Tool Naming Convention

```
Built-in:     Read, Write, Bash
MCP tools:    mcp__<server>__<tool>
Example:      mcp__weather__fetch_weather
```

---

## Complete Example: Production Agent

```python
import anyio
import logging
from pathlib import Path
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    tool,
    create_sdk_mcp_server,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom tool
@tool("analyze_complexity", "Analyze code complexity", {"file_path": str})
async def analyze_complexity(args):
    path = args.get("file_path")
    # In reality, run complexity analysis
    return {"content": [{"type": "text", "text": f"Complexity of {path}: Low"}]}

# Security hook
async def security_hook(input_data, tool_use_id, context):
    if input_data.get("tool_name") == "Bash":
        cmd = input_data.get("tool_input", {}).get("command", "")
        if "rm -rf" in cmd or "sudo" in cmd:
            return {
                "hookSpecificOutput": {
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Dangerous command blocked"
                }
            }
    return {}

async def main():
    # Create MCP server
    analyzer_server = create_sdk_mcp_server(
        name="analyzer",
        version="1.0.0",
        tools=[analyze_complexity]
    )

    # Configure agent
    options = ClaudeAgentOptions(
        model="claude-opus-4-5-20251101",
        allowed_tools=[
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            "mcp__analyzer__analyze_complexity"
        ],
        mcp_servers={"analyzer": analyzer_server},
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[security_hook])
            ]
        },
        system_prompt="You are an expert Python developer.",
        permission_mode="acceptEdits",
        cwd=Path.cwd(),
        max_turns=30,
        max_budget_usd=10.0,
    )

    # Run agent
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Review the codebase and suggest improvements")

            async for msg in client.receive_response():
                logger.info(f"Agent: {msg}")

        logger.info("Agent completed successfully")

    except Exception as e:
        logger.error(f"Agent failed: {e}")

if __name__ == "__main__":
    anyio.run(main())
```

---

## Resources

- [Claude Agent SDK - GitHub](https://github.com/anthropics/claude-agent-sdk-python)
- [Official Documentation](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Python SDK Reference](https://platform.claude.com/docs/en/agent-sdk/python)
- [Hooks Documentation](https://platform.claude.com/docs/en/agent-sdk/hooks)
- [Subagents Guide](https://platform.claude.com/docs/en/agent-sdk/subagents)

---

*Last updated: 2026-01-06*
