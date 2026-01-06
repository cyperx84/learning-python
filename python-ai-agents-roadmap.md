# Python AI Agents Learning Roadmap

> A comprehensive guide to mastering AI agent development with Python frameworks.

---

## Learning Path Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PYTHON AI AGENTS MASTERY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 1: FOUNDATIONS                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Python    │→ │  Pydantic   │→ │   FastAPI   │                 │
│  │   Core      │  │  Validation │  │   APIs      │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
│  PHASE 2: LLM FUNDAMENTALS                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  Anthropic  │→ │   OpenAI    │→ │  LiteLLM    │                 │
│  │  SDK        │  │   SDK       │  │  Universal  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
│  PHASE 3: AGENT FRAMEWORKS                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ LangChain   │→ │  LangGraph  │→ │Claude Agent │                 │
│  │  Basics     │  │  Workflows  │  │    SDK      │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
│  PHASE 4: ADVANCED ORCHESTRATION                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  CrewAI     │→ │  AutoGen    │→ │   Custom    │                 │
│  │  Teams      │  │  Multi-Agent│  │   Agents    │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
│  PHASE 5: PRODUCTION                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  RAG/Vector │→ │ Observtic   │→ │  Deploy     │                 │
│  │  DBs        │  │ LangSmith   │  │  Scale      │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Python Foundations

### 1.1 Python Core Concepts

> **Goal:** Master Python fundamentals essential for AI development

```python
# Key concepts to master:

# 1. Async/Await (critical for agents)
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def main():
    result = await fetch_data()
    tasks = [fetch_data() for _ in range(3)]
    results = await asyncio.gather(*tasks)  # Parallel execution

# 2. Type Hints (required for Pydantic)
from typing import Optional, List, Dict, Union, Callable

def process(items: List[str], callback: Callable[[str], None]) -> Dict[str, int]:
    return {item: len(item) for item in items}

# 3. Context Managers
class ResourceManager:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        await self.cleanup()

# 4. Decorators
def retry(times: int):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for _ in range(times):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    continue
        return wrapper
    return decorator

# 5. Generators & Iterators
async def stream_responses():
    for chunk in ["Hello", " ", "World"]:
        yield chunk
        await asyncio.sleep(0.1)
```

**Resources:**
- [ ] [Real Python - Async IO](https://realpython.com/async-io-python/)
- [ ] [Python Type Hints Guide](https://docs.python.org/3/library/typing.html)

---

### 1.2 Pydantic

> **Goal:** Master data validation and settings management

```bash
pip install pydantic pydantic-settings
```

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from typing import Optional, List
from enum import Enum

# Basic Models
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    tags: List[str] = Field(default_factory=list)

    @field_validator('title')
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

# Nested Models
class Agent(BaseModel):
    name: str
    model: str = "claude-opus-4-5-20251101"
    temperature: float = Field(default=0.7, ge=0, le=2)
    tools: List[str] = Field(default_factory=list)

class Workflow(BaseModel):
    agents: List[Agent]
    max_iterations: int = 10

    @model_validator(mode='after')
    def validate_workflow(self):
        if not self.agents:
            raise ValueError('At least one agent required')
        return self

# Settings Management
class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: Optional[str] = None
    debug: bool = False
    max_tokens: int = 4096

    class Config:
        env_file = ".env"

# Usage
settings = Settings()
task = Task(title="Build agent", priority=Priority.HIGH)
agent = Agent(name="coder", tools=["Read", "Write"])
```

**Key Concepts:**
| Concept | Use Case |
|---------|----------|
| `BaseModel` | Define data structures |
| `Field` | Add validation constraints |
| `@field_validator` | Custom field validation |
| `@model_validator` | Cross-field validation |
| `BaseSettings` | Environment configuration |

---

### 1.3 FastAPI

> **Goal:** Build APIs to serve your agents

```bash
pip install fastapi uvicorn python-multipart
```

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import asyncio

app = FastAPI(title="Agent API", version="1.0.0")

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class AgentRequest(BaseModel):
    prompt: str
    model: str = "claude-opus-4-5-20251101"
    tools: List[str] = []

class AgentResponse(BaseModel):
    response: str
    tokens_used: int
    tools_called: List[str]

# Dependency Injection
async def get_agent_client():
    # Initialize your agent client here
    return {"client": "agent_client"}

# Endpoints
@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    client = Depends(get_agent_client)
):
    try:
        # Run your agent here
        return AgentResponse(
            response="Agent response",
            tokens_used=100,
            tools_called=["Read"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streaming Response
from fastapi.responses import StreamingResponse

@app.post("/agent/stream")
async def stream_agent(request: AgentRequest):
    async def generate():
        for chunk in ["Processing", "...", "Done"]:
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="text/event-stream")

# Background Tasks
@app.post("/agent/async")
async def async_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks
):
    task_id = "task_123"
    background_tasks.add_task(run_long_agent_task, task_id, request)
    return {"task_id": task_id, "status": "processing"}

async def run_long_agent_task(task_id: str, request: AgentRequest):
    # Long-running agent task
    await asyncio.sleep(10)

# WebSocket for real-time
from fastapi import WebSocket

@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Agent: {data}")

# Run: uvicorn main:app --reload
```

**Patterns to Master:**
- [ ] Dependency injection
- [ ] Background tasks
- [ ] WebSocket connections
- [ ] Streaming responses (SSE)
- [ ] Error handling middleware

---

## Phase 2: LLM Fundamentals

### 2.1 Anthropic SDK

> **Goal:** Direct Claude API integration

```bash
pip install anthropic
```

```python
import anthropic
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

# Basic Completion
message = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain async/await in Python"}
    ]
)
print(message.content[0].text)

# With System Prompt
message = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    system="You are an expert Python developer. Be concise.",
    messages=[
        {"role": "user", "content": "Best practices for error handling?"}
    ]
)

# Streaming
with client.messages.stream(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a poem"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Tool Use (Function Calling)
tools = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
]

message = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

# Handle tool use response
for block in message.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}")

# Vision (Images)
import base64

with open("image.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
            {"type": "text", "text": "Describe this image"}
        ]
    }]
)
```

---

### 2.2 OpenAI SDK

> **Goal:** OpenAI GPT integration (similar patterns)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY env var

# Basic Completion
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Function Calling
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools
)
```

---

### 2.3 LiteLLM (Universal Interface)

> **Goal:** Single interface for 100+ LLM providers

```bash
pip install litellm
```

```python
from litellm import completion, acompletion
import asyncio

# Works with ANY provider - same interface!
# Claude
response = completion(
    model="claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": "Hello"}]
)

# OpenAI
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)

# Ollama (Local)
response = completion(
    model="ollama/llama2",
    messages=[{"role": "user", "content": "Hello"}]
)

# Async
async def async_call():
    response = await acompletion(
        model="claude-opus-4-5-20251101",
        messages=[{"role": "user", "content": "Hello"}]
    )
    return response

# Streaming
response = completion(
    model="claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")

# Fallbacks
from litellm import Router

router = Router(
    model_list=[
        {"model_name": "primary", "litellm_params": {"model": "claude-opus-4-5-20251101"}},
        {"model_name": "fallback", "litellm_params": {"model": "gpt-4o"}},
    ],
    fallbacks=[{"primary": ["fallback"]}]
)

response = router.completion(
    model="primary",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Provider Mapping:**
| Provider | Model Format |
|----------|--------------|
| Anthropic | `claude-opus-4-5-20251101` |
| OpenAI | `gpt-4o` |
| AWS Bedrock | `bedrock/anthropic.claude-v2` |
| Google Vertex | `vertex_ai/gemini-pro` |
| Ollama | `ollama/llama2` |
| Azure | `azure/gpt-4` |

---

## Phase 3: Agent Frameworks

### 3.1 LangChain

> **Goal:** Build composable LLM applications

```bash
pip install langchain langchain-anthropic langchain-openai langchain-community
```

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Initialize LLM
llm = ChatAnthropic(model="claude-opus-4-5-20251101")

# Basic Chain (LCEL - LangChain Expression Language)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Be concise."),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"role": "Python expert", "input": "Explain decorators"})

# With Memory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain_with_history = RunnableWithMessageHistory(
    prompt_with_history | llm | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Invoke with session
result = chain_with_history.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "user123"}}
)

# Custom Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Create Agent
tools = [calculator, search_web]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with tools."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is 25 * 4?"})
```

**Key LCEL Operators:**
| Operator | Purpose |
|----------|---------|
| `\|` | Chain components (pipe) |
| `RunnableParallel` | Run in parallel |
| `RunnableLambda` | Custom function |
| `RunnablePassthrough` | Pass data through |

---

### 3.2 LangGraph

> **Goal:** Build stateful, multi-actor agent workflows

```bash
pip install langgraph
```

```python
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.tools import tool
from typing import TypedDict, Annotated, List
import operator

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str

# Define Tools
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Found: {query} results"

@tool
def calculate(expr: str) -> str:
    """Calculate math expression."""
    return str(eval(expr))

tools = [search, calculate]

# Initialize LLM with tools
llm = ChatAnthropic(model="claude-opus-4-5-20251101")
llm_with_tools = llm.bind_tools(tools)

# Define Nodes
def agent_node(state: AgentState) -> AgentState:
    """Main agent reasoning."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Decide next step."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Build Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Run
config = {"configurable": {"thread_id": "session1"}}

result = app.invoke(
    {"messages": [HumanMessage(content="What is 10 + 20?")]},
    config=config
)

# Continue conversation (has memory)
result = app.invoke(
    {"messages": [HumanMessage(content="Now multiply that by 2")]},
    config=config
)
```

#### Multi-Agent Workflow

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Literal

class MultiAgentState(TypedDict):
    messages: list
    current_agent: str
    task_complete: bool

# Define specialized agents
def researcher_agent(state: MultiAgentState):
    """Research agent - gathers information."""
    # Use Claude for research
    return {
        "messages": state["messages"] + ["Research complete"],
        "current_agent": "writer"
    }

def writer_agent(state: MultiAgentState):
    """Writer agent - creates content."""
    return {
        "messages": state["messages"] + ["Content written"],
        "current_agent": "reviewer"
    }

def reviewer_agent(state: MultiAgentState):
    """Reviewer agent - reviews and approves."""
    return {
        "messages": state["messages"] + ["Review complete"],
        "task_complete": True
    }

def route_agents(state: MultiAgentState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    if state.get("task_complete"):
        return END
    return state.get("current_agent", "researcher")

# Build multi-agent graph
workflow = StateGraph(MultiAgentState)

workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.add_conditional_edges(START, route_agents)
workflow.add_conditional_edges("researcher", route_agents)
workflow.add_conditional_edges("writer", route_agents)
workflow.add_conditional_edges("reviewer", route_agents)

app = workflow.compile()

# Run
result = app.invoke({
    "messages": ["Write an article about AI"],
    "current_agent": "researcher",
    "task_complete": False
})
```

**LangGraph Concepts:**
| Concept | Description |
|---------|-------------|
| `StateGraph` | Define workflow structure |
| `add_node` | Add processing step |
| `add_edge` | Connect nodes |
| `add_conditional_edges` | Dynamic routing |
| `MemorySaver` | Persist state across runs |
| `ToolNode` | Pre-built tool executor |

---

### 3.3 Claude Agent SDK

> **Goal:** Build production agents with Claude

See [[claude-agents-sdk-quickstart]] for comprehensive guide.

```python
import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        model="claude-opus-4-5-20251101",
        permission_mode="acceptEdits",
        max_turns=20,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Build a REST API for a todo app")
        async for msg in client.receive_response():
            print(msg)

anyio.run(main())
```

---

## Phase 4: Advanced Orchestration

### 4.1 CrewAI

> **Goal:** Build collaborative AI agent teams

```bash
pip install crewai crewai-tools
```

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool
from langchain_anthropic import ChatAnthropic

# Initialize LLM
llm = ChatAnthropic(model="claude-opus-4-5-20251101")

# Define Agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.""",
    verbose=True,
    allow_delegation=False,
    tools=[SerperDevTool()],
    llm=llm
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content about AI discoveries",
    backstory="""You are a renowned Content Strategist,
    known for insightful and engaging articles.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

editor = Agent(
    role="Senior Editor",
    goal="Ensure content is polished and publication-ready",
    backstory="""You have decades of experience editing
    technical content for major publications.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Tasks
research_task = Task(
    description="""Conduct comprehensive analysis of latest AI trends.
    Focus on: LLMs, agents, and multi-modal models.
    Your final answer should be a detailed report.""",
    expected_output="Detailed research report with key findings",
    agent=researcher
)

write_task = Task(
    description="""Using the research, write an engaging blog post.
    Make it informative yet accessible to tech enthusiasts.""",
    expected_output="Blog post draft (800-1000 words)",
    agent=writer,
    context=[research_task]  # Depends on research
)

edit_task = Task(
    description="""Review and polish the blog post.
    Fix grammar, improve flow, ensure accuracy.""",
    expected_output="Final polished blog post",
    agent=editor,
    context=[write_task]
)

# Create Crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True
)

# Run
result = crew.kickoff()
print(result)
```

**CrewAI Concepts:**
| Concept | Purpose |
|---------|---------|
| `Agent` | Individual AI worker with role/goal |
| `Task` | Specific job with expected output |
| `Crew` | Team of agents working together |
| `Process.sequential` | Tasks run one after another |
| `Process.hierarchical` | Manager agent delegates tasks |

---

### 4.2 AutoGen

> **Goal:** Microsoft's multi-agent conversation framework

```bash
pip install pyautogen
```

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os

# Configuration
config_list = [
    {"model": "claude-opus-4-5-20251101", "api_key": os.environ["ANTHROPIC_API_KEY"]}
]

llm_config = {"config_list": config_list}

# Define Agents
assistant = AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant."
)

coder = AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""You are an expert programmer.
    Write clean, efficient code with comments."""
)

reviewer = AssistantAgent(
    name="Reviewer",
    llm_config=llm_config,
    system_message="""You review code for bugs, security issues,
    and best practices. Be thorough but constructive."""
)

# User Proxy (executes code)
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",  # or "ALWAYS", "TERMINATE"
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False,
    }
)

# Two-Agent Chat
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to find prime numbers"
)

# Group Chat
groupchat = GroupChat(
    agents=[user_proxy, coder, reviewer],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="Build a web scraper and review the code"
)
```

---

### 4.3 Instructor (Structured Outputs)

> **Goal:** Guaranteed structured outputs from LLMs

```bash
pip install instructor
```

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field
from typing import List, Optional

# Patch client
client = instructor.from_anthropic(Anthropic())

# Define Output Schema
class CodeReview(BaseModel):
    file_path: str
    issues: List[str] = Field(default_factory=list)
    severity: str = Field(description="low, medium, high, critical")
    suggestions: List[str]
    score: int = Field(ge=0, le=100)

class ReviewReport(BaseModel):
    summary: str
    reviews: List[CodeReview]
    overall_score: int
    approved: bool

# Get Structured Output
review = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": """Review this code:
        def add(a, b):
            return a + b
        """
    }],
    response_model=CodeReview  # Guarantees this structure!
)

print(review.severity)  # Typed access
print(review.score)

# Complex Nested Structures
report = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Review the entire codebase"}],
    response_model=ReviewReport
)

for review in report.reviews:
    print(f"{review.file_path}: {review.severity}")
```

---

## Phase 5: Production Infrastructure

### 5.1 Vector Databases & RAG

```bash
pip install chromadb pinecone-client qdrant-client langchain-chroma
```

#### ChromaDB (Local/Embedded)

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize
client = chromadb.PersistentClient(path="./chroma_db")

# Use OpenAI embeddings (or any other)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

# Create collection
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=openai_ef
)

# Add documents
collection.add(
    documents=["Doc 1 content", "Doc 2 content"],
    metadatas=[{"source": "file1"}, {"source": "file2"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

#### RAG with LangChain

```python
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load and split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Create vector store
vectorstore = Chroma.from_texts(
    texts=["chunk1", "chunk2"],
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# RAG Chain
template = """Answer based on context:
Context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatAnthropic(model="claude-opus-4-5-20251101")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the main topic?")
```

---

### 5.2 Observability & Tracing

#### LangSmith

```bash
pip install langsmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key
```

```python
from langsmith import Client
from langchain_anthropic import ChatAnthropic

# Automatic tracing when env vars set
llm = ChatAnthropic(model="claude-opus-4-5-20251101")
response = llm.invoke("Hello")  # Auto-traced!

# Manual tracing
from langsmith.run_helpers import traceable

@traceable(name="my_agent")
def my_agent(input_text: str) -> str:
    llm = ChatAnthropic(model="claude-opus-4-5-20251101")
    return llm.invoke(input_text).content
```

#### Weights & Biases (wandb)

```bash
pip install wandb
```

```python
import wandb

wandb.init(project="my-agent")

# Log metrics
wandb.log({
    "tokens_used": 1500,
    "response_time": 2.3,
    "success": True
})

# Log artifacts
artifact = wandb.Artifact("model-outputs", type="dataset")
artifact.add_file("outputs.json")
wandb.log_artifact(artifact)
```

---

### 5.3 Deployment

#### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Modal (Serverless)

```bash
pip install modal
```

```python
import modal

app = modal.App("my-agent")

@app.function(
    image=modal.Image.debian_slim().pip_install("anthropic", "langchain"),
    secrets=[modal.Secret.from_name("anthropic-secret")]
)
def run_agent(prompt: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Deploy: modal deploy agent.py
# Run: modal run agent.py::run_agent --prompt "Hello"
```

---

## Framework Comparison

| Framework | Best For | Complexity | Production Ready |
|-----------|----------|------------|------------------|
| **Anthropic SDK** | Direct Claude access | Low | Yes |
| **LangChain** | Composable chains | Medium | Yes |
| **LangGraph** | Stateful workflows | Medium-High | Yes |
| **Claude Agent SDK** | Full agent automation | Medium | Yes |
| **CrewAI** | Team collaboration | Medium | Yes |
| **AutoGen** | Multi-agent chat | Medium | Yes |
| **Instructor** | Structured outputs | Low | Yes |

---

## Learning Checklist

### Phase 1: Foundations (Week 1-2)
- [ ] Master async/await patterns
- [ ] Complete Pydantic models practice
- [ ] Build a FastAPI CRUD app
- [ ] Implement WebSocket endpoint

### Phase 2: LLM Basics (Week 3-4)
- [ ] Call Claude API directly
- [ ] Implement streaming responses
- [ ] Use tool/function calling
- [ ] Try LiteLLM with multiple providers

### Phase 3: Agent Frameworks (Week 5-8)
- [ ] Build LangChain chains with LCEL
- [ ] Create LangGraph stateful workflow
- [ ] Deploy Claude Agent SDK agent
- [ ] Implement memory/persistence

### Phase 4: Orchestration (Week 9-12)
- [ ] Build CrewAI team workflow
- [ ] Create AutoGen group chat
- [ ] Use Instructor for structured outputs
- [ ] Combine frameworks for complex tasks

### Phase 5: Production (Week 13-16)
- [ ] Implement RAG with vector DB
- [ ] Add observability (LangSmith/wandb)
- [ ] Deploy with Docker/Modal
- [ ] Build end-to-end production agent

---

## Quick Install Commands

```bash
# All core dependencies
pip install \
    pydantic pydantic-settings \
    fastapi uvicorn \
    anthropic openai litellm \
    langchain langchain-anthropic langchain-openai \
    langgraph \
    crewai crewai-tools \
    pyautogen \
    instructor \
    chromadb \
    langsmith wandb

# Claude Agent SDK
pip install claude-agent-sdk
```

---

## Resources

### Documentation
- [Pydantic Docs](https://docs.pydantic.dev/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [CrewAI Docs](https://docs.crewai.com/)
- [AutoGen Docs](https://microsoft.github.io/autogen/)
- [Anthropic Docs](https://docs.anthropic.com/)

### Tutorials
- [DeepLearning.AI - LangChain Courses](https://www.deeplearning.ai/)
- [LangChain YouTube](https://www.youtube.com/@LangChain)
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)

---

*Last updated: 2026-01-06*
