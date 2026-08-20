---
name: diagrams
description: This skill provides guidance for creating architecture diagrams, flowcharts, sequence diagrams, or any visual documentation in specs or markdown files. Always use Mermaid syntax - never use ASCII box art.
---

## Steps

### 1. Choose Diagram Type

Select the appropriate Mermaid diagram type:

| Diagram Type      | Use Case                                              |
| ----------------- | ----------------------------------------------------- |
| `flowchart`       | Architecture, data flow, component relationships      |
| `sequenceDiagram` | API calls, message passing, time-ordered interactions |
| `classDiagram`    | Object relationships, inheritance, interfaces         |
| `stateDiagram`    | State machines, workflow states                       |
| `erDiagram`       | Database schemas, entity relationships                |

### 2. Architecture/Flow Diagram

```mermaid
flowchart TD
    A[ChatWindow] --> B[HttpAgent]
    B --> C["/api/agent"]
    C --> D[run_chat_agent]
    D --> E[LiteLLM]
```

### 3. Sequence Diagram

Use this type for an SSE stream, an HTTP call, or any time-ordered exchange.

```mermaid
sequenceDiagram
    participant Browser
    participant AgentRoute as /api/agent
    participant Agent as run_chat_agent
    participant LiteLLM

    Browser->>AgentRoute: POST RunAgentInput
    AgentRoute->>Agent: run(agent_input)
    Agent-->>Browser: RunStartedEvent
    Agent->>LiteLLM: chat.completions (stream)
    loop Each token
        LiteLLM-->>Agent: delta
        Agent-->>Browser: TextMessageContentEvent
    end
    Agent-->>Browser: RunFinishedEvent
```

### 4. Component Diagram

Show the layer boundaries. Never draw an arrow that points up a layer.

```mermaid
flowchart LR
    subgraph Routes
        R["agent.py"]
    end
    subgraph Core
        P[RagPipeline]
    end
    subgraph Repositories
        DR[PostgresDocumentRepository]
        EI[QdrantEmbeddingIndex]
    end
    subgraph Infrastructure
        PG[(Postgres)]
        QD[(Qdrant)]
    end

    R --> P
    P --> DR
    P --> EI
    DR --> PG
    EI --> QD
```

## Why Mermaid?

- Renders properly in GitHub, VS Code, and most markdown viewers
- Maintainable and editable (text-based)
- Consistent styling across specs
- Supports flowcharts, sequence diagrams, class diagrams, and more

## Never Use ASCII Art

```
BAD - Do not use:
+------------------+     +------------------+
|   RagPipeline    | --> |   Repository     |
+------------------+     +------------------+
```

ASCII box art (`+-+`, `|`, `+--+`) does not render well and is harder to maintain. Always use Mermaid instead.
