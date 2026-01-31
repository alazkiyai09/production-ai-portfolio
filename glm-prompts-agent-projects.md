# GLM-4.7 Implementation Guide: Agent Projects
## 2 Production-Ready AI Agent Systems

---

# PROJECT 2A: AgenticFlow
## Multi-Agent Workflow System with LangGraph

### Project Overview

**What You'll Build**: A multi-agent system with:
- LangGraph state machine architecture
- Specialized agents (Planner, Researcher, Analyzer, Writer, Reviewer)
- Tool calling and integration
- Human-in-the-loop checkpoints
- Workflow orchestration API

**Why This Matters for Jobs**:
- EY: "Design, develop and optimize autonomous and semi-autonomous agents"
- Harnham: "Create automation tools that handle structured tasks"
- Multiple: "Experience with agent frameworks (LangChain, LangGraph)"

**Time Estimate**: 10-14 days

---

## SESSION SETUP PROMPT

Copy and paste this to start your GLM-4.7 session:

```
You are an expert Python developer helping me build a multi-agent workflow system using LangGraph.

PROJECT: AgenticFlow
PURPOSE: A production-ready multi-agent system for my AI Engineer portfolio demonstrating:
- LangGraph state machine architecture with TypedDict states
- Specialized agents with different roles and capabilities
- Tool calling (web search, file I/O, code execution)
- Human-in-the-loop checkpoints for review
- Workflow orchestration with conditional routing
- Production API with status tracking

TECH STACK:
- LangGraph 0.2+ for agent orchestration
- LangChain 0.3+ for tools and LLM interfaces
- Tavily for web search (or DuckDuckGo as fallback)
- FastAPI for workflow API
- Pydantic for state validation
- Python 3.11+

AGENT ROLES:
1. Planner: Breaks down tasks into steps
2. Researcher: Gathers information from web/documents
3. Analyzer: Analyzes data and identifies patterns
4. Writer: Creates written content
5. Reviewer: Evaluates output and provides feedback

QUALITY REQUIREMENTS:
- Type hints on all functions
- Comprehensive docstrings
- Error handling with retries
- Logging throughout
- Unit tests for each component
- Production-ready code

USER CONTEXT:
- I'm transitioning from fraud detection to AI Engineering
- Targeting remote AI Engineer roles
- This must demonstrate I can build production agent systems

RULES:
1. Generate complete, runnable code (no placeholders)
2. Include all imports at the top
3. Add comments explaining key decisions
4. Follow LangGraph best practices

Please confirm you understand, then we'll build this file by file.
```

---

## PROMPT 2.1: Project Structure & Configuration

```
Create the complete project structure for AgenticFlow.

Generate these files:

1. Directory structure (show as tree)
2. requirements.txt with pinned versions
3. pyproject.toml with project metadata
4. .env.example with all environment variables
5. src/__init__.py
6. src/config.py with Pydantic settings

For requirements.txt, include:
- langgraph>=0.2.0
- langchain>=0.3.0
- langchain-openai>=0.2.0
- langchain-anthropic>=0.2.0
- langchain-community>=0.3.0
- tavily-python>=0.3.0
- duckduckgo-search>=4.0.0
- fastapi>=0.109.0
- uvicorn[standard]>=0.27.0
- pydantic>=2.5.0
- pydantic-settings>=2.1.0
- python-dotenv>=1.0.0
- httpx>=0.26.0
- pytest>=7.4.0
- pytest-asyncio>=0.23.0

For config.py, use Pydantic BaseSettings with:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY (optional)
- TAVILY_API_KEY
- DEFAULT_MODEL
- MAX_ITERATIONS
- TIMEOUT_SECONDS
- LOG_LEVEL

Structure:
src/
├── agents/          # Agent definitions
├── tools/           # Tool implementations
├── workflows/       # LangGraph workflows
├── state/           # State definitions
├── api/             # FastAPI endpoints
└── utils/           # Utilities

Output all files completely with no placeholders.
```

---

## PROMPT 2.2: State Definitions

```
Create the state definitions for AgenticFlow.

File: src/state/workflow_state.py

Requirements:
1. Main workflow state using TypedDict
2. Per-agent state for tracking
3. Message history with proper typing
4. State validation helpers

Define these classes and functions:

class AgentOutput(TypedDict):
    """Output from a single agent step."""
    agent_name: str
    output: str
    tools_used: List[str]
    timestamp: str
    
class WorkflowState(TypedDict):
    """
    Main state for the multi-agent workflow.
    """
    # Input
    task: str
    task_type: Literal["research", "analysis", "content_creation", "general"]
    
    # Planning
    plan: List[str]
    current_step: int
    
    # Research
    research_queries: List[str]
    research_results: List[dict]
    
    # Analysis
    analysis: str
    key_findings: List[str]
    
    # Writing
    draft: str
    revision_count: int
    
    # Review
    feedback: List[str]
    approval_status: Literal["pending", "approved", "needs_revision", "rejected"]
    
    # Output
    final_output: str
    
    # Metadata
    status: Literal["planning", "researching", "analyzing", "writing", "reviewing", "complete", "error"]
    error: Optional[str]
    start_time: str
    
    # Message history (for LangGraph)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Agent outputs for tracking
    agent_outputs: List[AgentOutput]

Include helper functions:
- create_initial_state(task: str, task_type: str) -> WorkflowState
- update_state(state: WorkflowState, updates: dict) -> WorkflowState
- is_complete(state: WorkflowState) -> bool
- get_progress(state: WorkflowState) -> dict

Output the complete file with all implementations.
```

---

## PROMPT 2.3: Tool Implementations

```
Create the tool implementations for AgenticFlow.

File: src/tools/agent_tools.py

Requirements:
1. Web search tool (Tavily with DuckDuckGo fallback)
2. File read/write tools (with security)
3. Code execution tool (sandboxed with timeout)
4. Calculator tool (safe evaluation)
5. Current time tool

Use @tool decorator from langchain_core.tools.

Implement these tools:

@tool
def web_search(query: str, num_results: int = 5) -> List[dict]:
    """Search the web for information."""
    # Try Tavily, fallback to DuckDuckGo
    ...

@tool
def read_file(file_path: str) -> str:
    """Read contents of a text file (from workspace only)."""
    # Security: prevent directory traversal
    ...

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file (in workspace only)."""
    ...

@tool
def run_python_code(code: str) -> str:
    """Execute Python code in sandboxed environment (30s timeout)."""
    ...

@tool
def calculator(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    # Use AST for safe evaluation
    ...

@tool
def get_current_time() -> str:
    """Get current date and time."""
    ...

Also create:
- AGENT_TOOLS dict mapping agent types to their tools
- get_tools_for_agent(agent_type: str) -> List[Tool]

Output the complete file with all security measures.
```

---

## PROMPT 2.4: Specialized Agents

```
Create the specialized agent definitions for AgenticFlow.

File: src/agents/specialized_agents.py

Requirements:
1. Base agent class with common functionality
2. 5 specialized agents with unique prompts
3. LLM integration with tool binding
4. Output parsing and state updates

Create these classes:

class BaseAgent(ABC):
    """Base class for all specialized agents."""
    
    def __init__(self, name: str, llm: Optional[BaseChatModel] = None, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        ...
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...
    
    def invoke(self, state: WorkflowState) -> dict:
        """Execute the agent's task."""
        ...
    
    @abstractmethod
    def _build_input(self, state: WorkflowState) -> str:
        ...
    
    @abstractmethod
    def _process_output(self, output: str, state: WorkflowState) -> dict:
        ...

class PlannerAgent(BaseAgent):
    """Creates step-by-step plans for tasks."""
    # System prompt: Break down tasks, create actionable steps
    ...

class ResearcherAgent(BaseAgent):
    """Gathers information using web search and files."""
    # Tools: web_search, read_file
    # System prompt: Find reliable info, synthesize findings
    ...

class AnalyzerAgent(BaseAgent):
    """Analyzes research and extracts insights."""
    # Tools: calculator, read_file
    # System prompt: Identify patterns, draw conclusions
    ...

class WriterAgent(BaseAgent):
    """Creates written content based on analysis."""
    # Tools: read_file, write_file
    # System prompt: Clear, professional writing
    ...

class ReviewerAgent(BaseAgent):
    """Reviews content and provides feedback."""
    # Output: APPROVED, NEEDS REVISION, or REJECTED
    ...

def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """Factory function to create agents."""
    ...

Output the complete file with all agent implementations.
```

---

## PROMPT 2.5: LangGraph Workflow

```
Create the main LangGraph workflow for AgenticFlow.

File: src/workflows/research_workflow.py

Requirements:
1. StateGraph with all agent nodes
2. Conditional routing based on state
3. Checkpointing for persistence
4. Streaming support
5. Error handling

Implement:

class ResearchWorkflow:
    """
    Multi-agent research workflow using LangGraph.
    
    Flow:
    START → Planner → Researcher → Analyzer → Writer → Reviewer → END
                                                          ↓
                                                    (if needs revision)
                                                          ↓
                                                       Writer
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1, max_iterations: int = 10):
        ...
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph with WorkflowState schema
        # Add nodes for each agent
        # Add edges with conditional routing
        # Set entry point and end conditions
        ...
    
    # Node functions
    def _plan_node(self, state: WorkflowState) -> dict:
        ...
    
    def _research_node(self, state: WorkflowState) -> dict:
        ...
    
    def _analyze_node(self, state: WorkflowState) -> dict:
        ...
    
    def _write_node(self, state: WorkflowState) -> dict:
        ...
    
    def _review_node(self, state: WorkflowState) -> dict:
        ...
    
    # Routing functions
    def _should_continue_research(self, state: WorkflowState) -> Literal["continue", "analyze"]:
        ...
    
    def _review_decision(self, state: WorkflowState) -> Literal["approved", "revise", "rejected"]:
        ...
    
    # Execution
    def run(self, task: str, task_type: str = "general") -> WorkflowState:
        """Run the complete workflow."""
        ...
    
    def run_with_streaming(self, task: str, task_type: str = "general") -> Generator[dict, None, None]:
        """Run workflow with streaming updates."""
        ...
    
    def get_state(self, thread_id: str) -> Optional[WorkflowState]:
        """Get current state of a workflow."""
        ...

def create_workflow(**kwargs) -> ResearchWorkflow:
    """Factory function."""
    ...

Output the complete file with full LangGraph implementation.
```

---

## PROMPT 2.6: FastAPI Application

```
Create the FastAPI application for AgenticFlow.

File: src/api/main.py

Requirements:
1. Workflow management endpoints
2. Status tracking and streaming
3. Human-in-the-loop feedback
4. Background task execution

Endpoints:

POST /workflow/start
- Input: WorkflowRequest (task, task_type)
- Output: workflow_id, status, message
- Runs workflow in background

GET /workflow/{workflow_id}/status
- Output: StatusResponse with progress info

GET /workflow/{workflow_id}/stream
- Server-Sent Events for real-time updates

POST /workflow/{workflow_id}/feedback
- Input: FeedbackRequest (feedback, action)
- For human-in-the-loop review

GET /workflow/{workflow_id}/result
- Output: ResultResponse with final output

GET /workflows
- List all workflows

DELETE /workflow/{workflow_id}
- Delete a workflow

GET /health
- Health check

Include:
- Pydantic models for all request/response types
- CORS middleware
- In-memory storage (note: use Redis in production)
- Background task handling
- Error handling

Output the complete file.
```

---

## PROMPT 2.7: Streamlit Demo UI

```
Create a Streamlit demo interface for AgenticFlow.

File: src/ui/app.py

Requirements:
1. Task input form
2. Real-time progress display
3. Agent output visualization
4. Final result display

Features:
- Task type selector
- Progress bar showing workflow status
- Expandable sections for each agent's output
- Timeline view of agent execution
- Download final result as file

Sections:
1. Sidebar: Task input, settings
2. Main: Progress visualization, results
3. Expanders: Agent outputs, research findings, analysis

Output the complete file with professional styling.
```

---

## PROMPT 2.8: Tests

```
Create comprehensive tests for AgenticFlow.

File: tests/test_agenticflow.py

Test categories:

1. State Tests
   - create_initial_state
   - update_state
   - is_complete
   - get_progress

2. Tool Tests
   - web_search (mock API)
   - read_file (security tests)
   - write_file (security tests)
   - calculator (various expressions)
   - run_python_code (timeout, security)

3. Agent Tests
   - Each agent with mock LLM
   - Output parsing
   - State updates

4. Workflow Tests
   - Graph construction
   - Node execution
   - Conditional routing
   - Complete flow (with mocks)

5. API Tests
   - All endpoints
   - Error handling
   - Background tasks

Use pytest fixtures for:
- Mock LLM
- Mock tools
- Sample workflow states

Include at least 30 test cases.
Output the complete file.
```

---

## PROMPT 2.9: Docker & README

```
Create Docker configuration and README for AgenticFlow.

Files:
1. Dockerfile
2. docker-compose.yml
3. README.md

Dockerfile:
- Multi-stage build
- Python 3.11-slim
- Non-root user
- Health check

docker-compose.yml:
- agenticflow-api (port 8001)
- agenticflow-ui (port 8501)
- Optional Redis for production

README.md sections:
1. Title with badges
2. Overview & architecture diagram
3. Agent descriptions with examples
4. Workflow visualization (ASCII/Mermaid)
5. Quick start guide
6. API documentation
7. Configuration options
8. Example usage
9. Testing
10. Deployment

Output all files completely.
```

---

# PROJECT 2B: CustomerSupport-Agent
## Conversational AI Agent with Memory & Integrations

### Project Overview

**What You'll Build**: A customer support agent with:
- Long-term conversation memory (per user)
- FAQ/Knowledge base retrieval (RAG)
- Ticket creation and lookup
- Sentiment analysis for routing
- Human handoff capability
- Multi-turn conversation handling

**Why This Matters for Jobs**:
- Shows practical business application
- Demonstrates memory/context management
- Shows integration patterns
- Common interview project

**Time Estimate**: 7-10 days (after completing 2A)

---

## SESSION SETUP PROMPT

```
You are an expert Python developer helping me build a customer support AI agent.

PROJECT: CustomerSupport-Agent
PURPOSE: A production-ready support agent demonstrating:
- Long-term conversation memory (per user)
- Knowledge base retrieval for FAQs
- Ticket/case management tools
- Sentiment detection and routing
- Escalation to human support

TECH STACK:
- LangGraph for conversation flow
- LangChain for memory and tools
- ChromaDB for knowledge base
- FastAPI + WebSocket for real-time chat
- Pydantic for data models
- Python 3.11+

KEY FEATURES:
1. Memory: Remember user context across sessions
2. RAG: Search FAQ/knowledge base for answers
3. Tickets: Create, lookup, update support tickets
4. Sentiment: Detect frustration, route appropriately
5. Handoff: Escalate to human when needed

QUALITY REQUIREMENTS:
- Production-ready code
- Type hints and docstrings
- Error handling
- Unit tests

Please confirm you understand, then we'll build this file by file.
```

---

## PROMPT 2B.1: Project Structure

```
Create the project structure for CustomerSupport-Agent.

Generate:
1. Directory tree
2. requirements.txt
3. .env.example
4. src/config.py

Additional requirements beyond 2A:
- websockets>=12.0
- chromadb>=0.4.0
- textblob>=0.17.0 (for sentiment)

Structure:
src/
├── memory/          # Conversation memory
├── knowledge/       # Knowledge base/RAG
├── tools/           # Support tools (tickets, etc.)
├── sentiment/       # Sentiment analysis
├── conversation/    # Conversation flow
├── api/             # REST + WebSocket
└── utils/

Output all files.
```

---

## PROMPT 2B.2: Conversation Memory

```
Create the conversation memory system.

File: src/memory/conversation_memory.py

Requirements:
1. Short-term memory (current conversation)
2. Long-term memory (user history, preferences)
3. Summary generation for long conversations
4. Memory search for relevant context

Classes:

class ConversationMemory:
    """Manages conversation context for a user."""
    
    def __init__(self, user_id: str, max_messages: int = 20):
        ...
    
    def add_message(self, role: str, content: str):
        """Add a message to current conversation."""
        ...
    
    def get_context(self, max_tokens: int = 2000) -> str:
        """Get conversation context for LLM."""
        ...
    
    def summarize_if_needed(self):
        """Summarize if conversation gets too long."""
        ...

class UserMemoryStore:
    """Long-term memory storage for users."""
    
    def __init__(self, storage_path: str = "./data/user_memory"):
        ...
    
    def get_user_profile(self, user_id: str) -> dict:
        """Get user's stored profile/preferences."""
        ...
    
    def update_user_profile(self, user_id: str, updates: dict):
        """Update user information."""
        ...
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[dict]:
        """Get recent conversation summaries."""
        ...
    
    def search_history(self, user_id: str, query: str) -> List[dict]:
        """Search user's conversation history."""
        ...

Output the complete file.
```

---

## PROMPT 2B.3: Knowledge Base

```
Create the knowledge base (FAQ) system.

File: src/knowledge/faq_store.py

Requirements:
1. Load FAQs from file/database
2. Vector search for relevant answers
3. Category filtering
4. Confidence scoring

Classes:

class FAQStore:
    """Knowledge base for customer support FAQs."""
    
    def __init__(self, chroma_path: str = "./data/faq_db"):
        ...
    
    def add_faq(self, question: str, answer: str, category: str, metadata: dict = None):
        """Add a FAQ entry."""
        ...
    
    def load_faqs_from_file(self, file_path: str):
        """Bulk load FAQs from JSON/CSV."""
        ...
    
    def search(self, query: str, category: str = None, top_k: int = 3) -> List[FAQResult]:
        """Search for relevant FAQs."""
        ...
    
    def get_categories(self) -> List[str]:
        """Get all FAQ categories."""
        ...

@dataclass
class FAQResult:
    question: str
    answer: str
    category: str
    confidence: float
    metadata: dict

Include sample FAQs for a generic SaaS product.
Output the complete file.
```

---

## PROMPT 2B.4: Support Tools

```
Create the support tools (tickets, account lookup, etc.).

File: src/tools/support_tools.py

Implement these tools:

@tool
def search_faq(query: str, category: str = None) -> str:
    """Search the FAQ knowledge base."""
    ...

@tool
def create_ticket(user_id: str, subject: str, description: str, priority: str = "medium") -> dict:
    """Create a new support ticket."""
    ...

@tool
def get_ticket_status(ticket_id: str) -> dict:
    """Get status of an existing ticket."""
    ...

@tool
def update_ticket(ticket_id: str, status: str = None, notes: str = None) -> dict:
    """Update a support ticket."""
    ...

@tool
def get_user_tickets(user_id: str, status: str = None) -> List[dict]:
    """Get all tickets for a user."""
    ...

@tool
def lookup_account(user_id: str) -> dict:
    """Look up user account information."""
    ...

@tool
def escalate_to_human(user_id: str, reason: str, conversation_summary: str) -> dict:
    """Escalate conversation to human support."""
    ...

Also include:
- TicketStore class for persistence
- Mock account data for demo

Output the complete file.
```

---

## PROMPT 2B.5: Sentiment Analysis

```
Create the sentiment analysis module.

File: src/sentiment/analyzer.py

Requirements:
1. Detect sentiment (positive, negative, neutral)
2. Detect frustration level
3. Suggest routing based on sentiment
4. Track sentiment over conversation

Classes:

class SentimentAnalyzer:
    """Analyze customer sentiment in messages."""
    
    def __init__(self):
        ...
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single message."""
        ...
    
    def analyze_conversation(self, messages: List[str]) -> ConversationSentiment:
        """Analyze sentiment trend over conversation."""
        ...
    
    def should_escalate(self, sentiment_history: List[SentimentResult]) -> bool:
        """Determine if conversation should escalate to human."""
        ...

@dataclass
class SentimentResult:
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    label: Literal["positive", "negative", "neutral"]
    frustration_score: float  # 0 to 1
    keywords: List[str]

@dataclass
class ConversationSentiment:
    average_polarity: float
    trend: Literal["improving", "stable", "declining"]
    escalation_recommended: bool
    reason: str

Use TextBlob for basic sentiment, with keyword detection for frustration.
Output the complete file.
```

---

## PROMPT 2B.6: Conversation Agent

```
Create the main conversation agent using LangGraph.

File: src/conversation/support_agent.py

Requirements:
1. Multi-turn conversation handling
2. Tool selection based on intent
3. Memory integration
4. Sentiment-aware responses
5. Escalation logic

Implement:

class SupportAgent:
    """Customer support conversational agent."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.memory = {}  # user_id -> ConversationMemory
        self.faq_store = FAQStore()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.graph = self._build_graph()
        ...
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the conversation flow graph."""
        # Nodes: understand_intent, search_knowledge, use_tool, generate_response, check_escalation
        # Edges: conditional routing based on intent and sentiment
        ...
    
    def chat(self, user_id: str, message: str) -> SupportResponse:
        """Process a user message and return response."""
        ...
    
    def _understand_intent(self, state: ConversationState) -> dict:
        """Classify user intent."""
        # Intents: question, complaint, request, feedback, greeting, other
        ...
    
    def _search_knowledge(self, state: ConversationState) -> dict:
        """Search FAQ if it's a question."""
        ...
    
    def _use_tool(self, state: ConversationState) -> dict:
        """Use appropriate tool based on intent."""
        ...
    
    def _generate_response(self, state: ConversationState) -> dict:
        """Generate final response."""
        ...
    
    def _check_escalation(self, state: ConversationState) -> Literal["continue", "escalate"]:
        """Check if should escalate to human."""
        ...

@dataclass
class SupportResponse:
    message: str
    intent: str
    sentiment: SentimentResult
    sources: List[str]
    escalated: bool
    ticket_created: Optional[str]

Output the complete file.
```

---

## PROMPT 2B.7: WebSocket API

```
Create the FastAPI application with WebSocket support.

File: src/api/main.py

Requirements:
1. WebSocket endpoint for real-time chat
2. REST endpoints for ticket management
3. Session management
4. Connection handling

Endpoints:

WebSocket /ws/chat/{user_id}
- Real-time bidirectional chat
- Message format: {"type": "message", "content": "..."}
- Response format: {"type": "response", "content": "...", "metadata": {...}}

REST:
POST /chat (alternative to WebSocket)
GET /users/{user_id}/tickets
GET /users/{user_id}/history
POST /feedback (rate response)
GET /health

Include:
- Connection manager for WebSocket
- Session timeout handling
- Error messages
- Typing indicators

Output the complete file.
```

---

## PROMPT 2B.8: Tests & Documentation

```
Create tests and documentation for CustomerSupport-Agent.

Files:
1. tests/test_support_agent.py
2. README.md
3. Dockerfile
4. docker-compose.yml

Tests:
- Memory system
- FAQ search
- Tools (all operations)
- Sentiment analysis
- Conversation flow
- API endpoints (REST + WebSocket)

README:
- Overview with use case
- Architecture diagram
- Features list
- Setup guide
- API documentation
- Example conversations
- Customization guide

Output all files.
```

---

# END OF AGENT PROJECTS

---

## Summary: Agent Projects

| Project | Focus | Complexity | Time |
|---------|-------|------------|------|
| **2A: AgenticFlow** | Core agent architecture | High | 10-14 days |
| **2B: CustomerSupport-Agent** | Practical business app | Medium | 7-10 days |

**Recommendation**: 
- **2A is essential** - demonstrates LangGraph, multi-agent, tools
- **2B is optional** - adds memory/conversation skills if time permits
- For most jobs, 2A alone is sufficient to demonstrate agent capabilities
