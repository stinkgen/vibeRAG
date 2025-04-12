# VibeRAG Agent Architecture & Implementation Plan

**Version:** 1.0 (April 2025)
**Status:** Proposed

## 1. Vision: Real Fucking Agents

This document outlines the architectural vision and implementation plan for integrating **truly autonomous, collaborative, and persistent agents** into the VibeRAG ecosystem. The goal is to move beyond simplistic, deterministic "agent-like" workflows (e.g., basic RAG chains) towards systems where agents possess:

*   **Autonomy:** Define goals, plan multi-step actions, select and use tools dynamically, learn from interactions, and operate with minimal human intervention.
*   **Collaboration:** Engage in complex interactions with users and other agents, forming hierarchies, working towards shared goals, reaching consensus, or performing specialized roles within a team.
*   **Persistence:** Maintain internal state, memory, and a traceable "thought process" over time. Their existence and evolution should be observable and understandable.
*   **Tool Mastery:** Leverage a rich set of tools, including internal VibeRAG capabilities (search, document access), external APIs, and potentially code execution, facilitated by the **Model Context Protocol (MCP)** for standardized data access.
*   **Deep Integration:** Seamlessly participate in core VibeRAG functions like chat, research, presentation generation, and data ingestion, enhancing existing features and enabling entirely new workflows.

We aim to build agents capable of deep analysis, self-improvement, and emergent behavior, providing users with powerful AI partners rather than just reactive tools.

## 2. Core Concepts & Philosophy

*   **Agent Definition:** An agent is a persistent entity defined by its **Personality/Persona**, **Goals**, **Capabilities (Tools)**, **Memory Structure**, **Communication Protocols**, and potentially **Roles**. Agents are instantiated and managed by a dedicated runtime.
*   **Agent Runtime:** A new core service responsible for managing the lifecycle, execution loop (e.g., Observe-Plan-Act), communication, and state of all active agents.
*   **Multi-Agent Systems (MAS):** We will leverage concepts from modern MAS frameworks (like inspirations from AutoGen, CrewAI, AgentVerse, Camel) to facilitate structured collaboration, role assignment, and complex task decomposition among agents.
*   **Memory Architecture:** Agents will possess multi-layered memory:
    *   **Short-Term:** Working memory, context window (managed by the LLM interaction layer).
    *   **Medium-Term:** Scratchpad for current task reasoning, plans, intermediate results (potentially stored in Redis or Postgres).
    *   **Long-Term:** Episodic memory (significant events, conversations, tool uses), semantic memory (learned facts, summaries), procedural memory (successful workflows). Stored in Milvus (vectorized) and/or Postgres (structured summaries), linked to the agent's identity. MemGPT concepts will inform this design.
*   **Model Context Protocol (MCP):** MCP will be central to tool use and data access. Instead of ad-hoc data fetching for tools, we will implement MCP connectors for accessing VibeRAG data (documents, user profiles, chat history, other agent states) and potentially external resources. This standardizes how agents perceive and interact with their environment.
*   **Observability ("Internal Lives"):** Agent actions, decisions, tool calls, inter-agent communication, and internal state changes will be meticulously logged to a persistent, queryable store (e.g., dedicated Postgres tables or an event stream). This provides the foundation for understanding agent behavior, debugging, and tracing their "thoughts."

## 3. Proposed Architecture

We propose introducing a new **Agent Service** and significantly modifying existing components.

```mermaid
graph TD
    subgraph VibeRAG Ecosystem
        User[User Browser] --> FE[Frontend (React UI + Node Proxy)]

        subgraph Backend Services
            FE -- HTTP/WS --> API[FastAPI Backend]
            API -- Manages/Interacts --> AgentSvc[**New: Agent Service / Runtime**]
            AgentSvc -- Schedules/Executes --> AgentInstances(Agent Instances)
            AgentInstances -- Uses --> ToolRegistry[**New: Tool Registry & Executor**]
            ToolRegistry -- Accesses Data via --> MCP[**New: MCP Connector Layer**]
            MCP -- Reads/Writes --> PostgresDB[(PostgreSQL: Users, Chat, **Agents**, **Agent Logs**)]
            MCP -- Reads/Writes --> MilvusDB[(Milvus: Docs, **Agent Memories**)]
            MCP -- Calls --> ExternalAPIs[External APIs / Web]
            API -- Standard Ops --> PostgresDB
            API -- Standard Ops --> MilvusDB
            API -- Standard Ops --> ExternalAPIs
            AgentSvc -- Reads/Writes --> PostgresDB
            AgentSvc -- Reads/Writes --> MilvusDB
            AgentSvc -- Logs --> AgentLogStore[**New/Dedicated: Agent Event/Log Store**]
        end
    end

    style AgentSvc fill:#f9d,stroke:#333,stroke-width:2px
    style ToolRegistry fill:#f9d,stroke:#333,stroke-width:2px
    style MCP fill:#f9d,stroke:#333,stroke-width:2px
    style AgentLogStore fill:#f9d,stroke:#333,stroke-width:2px

```

### 3.1. New Components

1.  **Agent Service / Runtime (`agent_service`):**
    *   **Technology:** Python (likely FastAPI or a dedicated service framework like Dramatiq/Celery for background tasks).
    *   **Responsibilities:**
        *   Manages the lifecycle of agent definitions (CRUD via API).
        *   Instantiates and runs agent processes/threads.
        *   Implements the core agent execution loop (perception, planning, action).
        *   Handles inter-agent communication (message bus, direct calls).
        *   Coordinates multi-agent tasks (hierarchies, workflows).
        *   Manages agent state persistence (short/medium term).
        *   Interfaces with the Tool Registry and MCP Layer.
        *   Writes detailed logs to the Agent Log Store.
    *   **Considerations:** Needs to be highly scalable and resilient. Asynchronous processing is essential.

2.  **Tool Registry & Executor:**
    *   **Technology:** Python module within the backend/agent service.
    *   **Responsibilities:**
        *   Maintains a registry of available tools (defined via code/config).
        *   Provides metadata about tools (description, parameters, MCP requirements) for agent planning.
        *   Securely executes tool calls requested by agents.
        *   Interfaces with the MCP Connector Layer to provide necessary context/data to tools.
        *   Handles permissions/capabilities for which agents can use which tools.

3.  **Model Context Protocol (MCP) Connector Layer:**
    *   **Technology:** Python module within the backend.
    *   **Responsibilities:**
        *   Implements MCP connectors for VibeRAG-specific data sources (Postgres tables for users/chat/agents, Milvus collections for docs/memories, current session state).
        *   Potentially implements connectors for common external APIs (Google Search already exists, could be wrapped).
        *   Provides a standardized interface (`provide_context`, `update_context`) for tools and the Agent Runtime to interact with data.

4.  **Agent Database Models (Postgres):**
    *   New tables: `agents` (definition: id, name, persona, goals, base_prompt, owner_user_id), `agent_capabilities` (link agent to allowed tools), `agent_state` (runtime state, status), potentially tables for medium-term memory/scratchpad.

5.  **Agent Memory Storage (Milvus):**
    *   Extend Milvus usage: Create dedicated collections per agent (e.g., `agent_<agent_id>_memory`) or add agent identifiers to existing collections to store vectorized long-term memories (episodic, semantic). Schema needs to accommodate memory type, timestamps, associated context.

6.  **Agent Event/Log Store:**
    *   **Technology:** Dedicated Postgres tables optimized for append/query, or potentially a dedicated event streaming platform (like Kafka) if scale demands it.
    *   **Responsibilities:** Store detailed, structured logs of agent activities: decisions, plans, tool calls (request, params, result), communications, state changes, errors. Each log entry tagged with agent ID, timestamp, task ID, etc.

### 3.2. Modifications to Existing Components

1.  **FastAPI Backend (`api/app.py` & modules):**
    *   Add new API endpoints for managing agent definitions (CRUD).
    *   Add endpoints/WebSocket channels for user-agent interaction.
    *   Integrate calls to the Agent Service to trigger agent tasks (e.g., assist in chat, run research).
    *   Refactor `research.py` and `slides.py` to delegate tasks to the Agent Service instead of performing monolithic generation.
    *   Expose necessary internal functions/data access patterns via the MCP Layer.
    *   Update authentication/authorization to handle agent identities and permissions.

2.  **Frontend (React UI):**
    *   **New Views:**
        *   `Agent Management View`: UI for users/admins to define, configure, monitor, and manage agents they own/oversee. View agent logs/status.
        *   `Agent Chat Room (Future)`: A dedicated space for observing and interacting with autonomous agents.
    *   **Component Updates:**
        *   `Chat.tsx`: Add options to invoke agents for assistance, display agent messages differently, potentially allow direct @mentions of agents.
        *   `ResearchReport.tsx` / `PresentationViewer.tsx`: Update UI to reflect agent-driven workflows (show progress, agent steps, final output).
        *   `DocumentManager.tsx`: Potentially add options to assign agents to watch folders or process specific documents.
        *   `AdminPanel.tsx`: Add controls for system-wide agent settings, monitoring, and potentially managing global agents.

3.  **Postgres Database:** Add new tables as defined above. Add foreign keys linking agents to users (owners).
4.  **Milvus:** Configure new collections/schemas for agent memories.

### 3.3. Data Flow Examples

*   **Agent-Assisted Chat:**
    1.  User invokes an agent (e.g., `@research_agent find recent papers on X`).
    2.  Frontend sends request to Backend API.
    3.  Backend API validates, identifies the agent, and forwards the task request (query, chat context, user ID) to the Agent Service.
    4.  Agent Service activates the `research_agent` instance.
    5.  Agent plans steps (e.g., [1] Use Google Search tool, [2] Use Semantic Search tool on internal docs, [3] Synthesize results).
    6.  Agent requests `google_search` tool via Tool Registry.
    7.  Tool Registry uses MCP to get API keys, executes the search, returns results.
    8.  Agent requests `semantic_search` tool via Tool Registry.
    9.  Tool Registry uses MCP to get user context (which collections to search), calls Milvus, returns results.
    10. Agent synthesizes the information using its LLM.
    11. Agent sends response back through Agent Service -> Backend API -> Frontend (potentially streaming).
    12. All steps (plan, tool calls, results, synthesis) are logged to the Agent Log Store.

*   **Multi-Agent Research Workflow:**
    1.  User requests a deep research report via the UI.
    2.  Frontend sends request to Backend API.
    3.  Backend API triggers a high-level "Research Manager" agent in the Agent Service.
    4.  Research Manager agent plans the workflow (e.g., [1] Delegate web scraping to `WebScraperAgent`, [2] Delegate internal doc analysis to `DocAnalyzerAgent`, [3] Synthesize findings itself).
    5.  Manager agent sends tasks to worker agents via the Agent Service's communication bus.
    6.  Worker agents execute their tasks (using tools via Tool Registry/MCP) and report results back to the Manager.
    7.  Manager agent synthesizes the final report.
    8.  Manager sends final report back through Agent Service -> Backend API -> Frontend.
    9.  All inter-agent communication and individual agent steps are logged.

## 4. Key Feature Implementation Details

*   **Autonomy & Planning:** Implement a planning module within the Agent Runtime. This could range from simple ReAct prompting to more sophisticated techniques like Chain-of-Thought or Tree-of-Thoughts, allowing agents to dynamically select tools and actions based on their goals and observations.
*   **Tool Use & MCP:** Define tools with clear schemas (inputs, outputs) and MCP requirements. The MCP layer becomes critical for grounding tools in the VibeRAG context. Secure execution (sandboxing for code execution tools) is vital.
*   **Collaboration & MAS:** Define communication protocols (standard message formats) and implement orchestration patterns (e.g., hierarchical delegation, broadcast/subscribe) within the Agent Service. Inspirations: AutoGen's `ConversableAgent`.
*   **Hierarchies:** Implement by defining agent roles and relationships (e.g., `manager`, `worker`) and allowing agents to delegate tasks based on these roles.
*   **Persistence & Memory:** Integrate agent state saving/loading with Postgres (definitions, runtime state) and Milvus (long-term vectorized memories). Ensure the Agent Log Store captures sufficient detail for traceability.
*   **Agent Chat Room:** Requires the Agent Service to support persistent, asynchronous agent instances that can interact proactively based on triggers or scheduled intervals, not just user requests. Frontend needs a dedicated WS connection or polling mechanism to display ongoing agent activity.
*   **Integration (Research/Presentations):** Replace the existing Python scripts (`research.py`, `slides.py`) with calls to trigger pre-defined agent workflows managed by the Agent Service.

## 5. User Experience (UX)

*   **Agent Interaction:** Clear affordances for invoking agents in chat, understanding which agent is responding, and viewing agent-generated content.
*   **Agent Management:** Intuitive UI for creating agents (defining persona, goals, capabilities), monitoring their status, viewing their logs/thought processes, and managing their lifecycle.
*   **Transparency:** Users should be able to understand *why* an agent took certain actions by inspecting its logs (simplified view for users, detailed view for admins/developers).

## 6. Admin Experience

*   System-level monitoring of agent performance, resource usage, and costs.
*   Management of global agents, tool availability, and system-wide agent policies.
*   Access to detailed logs for debugging and auditing.
*   User management integration (who can create/use agents).

## 7. Development Plan (Phased Approach)

This is a major undertaking. A phased approach is crucial:

**Phase 1: Core Infrastructure (Foundation)**

*   **Goal:** Establish the basic building blocks.
*   **Tasks:**
    *   Design and implement Postgres schemas for `agents`, `agent_capabilities`, `agent_state`.
    *   Build the initial Agent Service skeleton (lifecycle management API - CRUD for definitions).
    *   Implement the basic Agent Runtime loop (without complex planning or memory yet).
    *   Set up the Agent Log Store (basic Postgres tables).
    *   Implement the Tool Registry (registering existing tools like semantic search, web search).
    *   Build the initial MCP Connector Layer for basic context (e.g., user ID, API keys).
    *   Backend API endpoints for agent definition CRUD.
    *   Basic Admin UI for agent definition CRUD.
*   **Outcome:** Ability to define agents and see basic logs, but agents don't *do* much yet.

**Phase 2: Basic Agent Execution & Tool Use**

*   **Goal:** Enable agents to perform simple, single-step tasks using existing tools.
*   **Tasks:**
    *   Enhance Agent Runtime with basic planning (e.g., simple ReAct prompting).
    *   Integrate tool execution via Tool Registry/MCP.
    *   Connect Agent Service to existing tools (semantic search, web search) via MCP.
    *   Modify Chat backend/frontend to allow invoking an agent for a single query/response using one tool.
    *   Enhance logging to capture tool calls and basic decisions.
*   **Outcome:** A user can `@agent use_web_search X` and get a result. Agent logs show the tool call.

**Phase 3: Memory & Multi-Step Tasks**

*   **Goal:** Give agents basic memory and the ability to perform multi-step sequences.
*   **Tasks:**
    *   Implement medium-term memory (scratchpad in Postgres/Redis).
    *   Implement basic long-term memory (vectorized summaries/events in Milvus via MCP).
    *   Enhance planning module to handle simple sequential tasks.
    *   Refactor Research/Presentation generation to use a *single* agent performing multiple steps (search -> synthesize).
    *   Update UI to show multi-step progress.
*   **Outcome:** Agents can perform tasks like "search web, then search docs, then summarize" and remember context within a single task. Research/Presentation paths are now agent-driven (single agent).

**Phase 4: Multi-Agent Systems (MAS) & Collaboration**

*   **Goal:** Enable multiple agents to collaborate on tasks.
*   **Tasks:**
    *   Implement inter-agent communication bus within Agent Service.
    *   Define agent roles and implement hierarchical task delegation (e.g., Manager/Worker pattern).
    *   Refactor Research/Presentation generation again to use a *team* of specialized agents (e.g., WebSearchAgent, DocAnalysisAgent, SynthesisAgent) orchestrated by a ManagerAgent.
    *   Enhance logging for inter-agent communication.
    *   Update UI to visualize multi-agent workflows.
*   **Outcome:** Complex tasks like research are performed by collaborating agent teams, yielding potentially deeper results.

**Phase 5: Advanced Features & UI**

*   **Goal:** Polish UX, add advanced agent capabilities, and build the Agent Chat Room.
*   **Tasks:**
    *   Develop the dedicated Agent Management UI (viewing logs, detailed config).
    *   Implement more sophisticated planning and memory techniques.
    *   Add more tools (code execution?, external APIs via MCP).
    *   Build the Agent Chat Room UI and backend support for persistent, autonomous agent interaction.
    *   Refine security, monitoring, and administration features.
*   **Outcome:** A fully-featured agent system with rich interaction possibilities and management tools.

## 8. Technology Choices & Justification

*   **Python/FastAPI:** Leverage existing backend stack for consistency and developer familiarity. Python has a rich ecosystem for AI/ML.
*   **Postgres:** Continue using for structured data (users, chat, agent definitions, logs) due to its robustness and ACID compliance. Upgrade justified the move from SQLite.
*   **Milvus:** Continue using for vector storage (docs, agent memories) due to its specialization.
*   **MCP:** Adopt as a standard for internal data access to promote modularity and future-proofing tool development.
*   **MAS Frameworks (Inspiration):** Draw inspiration from AutoGen, CrewAI, etc., for collaboration patterns, but build the core runtime tailored to VibeRAG to avoid black boxes and ensure deep integration. Avoid direct dependency on overly simplistic frameworks.

## 9. Risks & Considerations

*   **Complexity:** This is a significant increase in system complexity. Requires careful design and testing.
*   **Performance & Scalability:** Running many agents concurrently will require substantial compute resources (CPU, potentially GPU for agent LLM calls) and efficient asynchronous processing. Need performance testing and optimization.
*   **Cost:** LLM calls for agent planning, execution, and communication can become expensive. Need monitoring and controls.
*   **Alignment & Safety:** Ensuring agents act according to user intent and within safe boundaries is paramount, especially with tool use like code execution. Requires careful prompt engineering, capability restrictions, and monitoring. Sandboxing is critical for risky tools.
*   **Observability:** Building effective logging and monitoring is crucial for debugging and understanding emergent behavior.
*   **Development Time:** This is a large project requiring significant development effort across multiple phases.

## 10. Future Work (Post-MVP)

*   Self-improving agents (learning from feedback, refining plans).
*   More sophisticated memory architectures.
*   Wider range of tools and MCP connectors.
*   Multimodal agents (integrating image/audio processing).
*   Agent marketplaces or sharing mechanisms.

---

This architecture provides a roadmap for building a powerful, flexible, and truly *agentic* layer within VibeRAG, moving far beyond basic RAG and unlocking new possibilities for human-AI collaboration and knowledge generation. WAGMI. 