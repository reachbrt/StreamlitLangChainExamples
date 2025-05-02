# LangChain Streamlit Chatbot: Diagrams

This document contains diagrams that illustrate the architecture and flow of the LangChain Streamlit Chatbot application.

## Author

**Bharat Kumar Subramanian**
Email: reachbrt@gmail.com

## Architecture Diagram

```mermaid
graph TD
    User[User] <--> StreamlitUI[Streamlit UI]
    StreamlitUI <--> LangChain[LangChain Framework]
    LangChain <--> OpenAI[OpenAI API]

    subgraph Tracing
        StreamlitUI --> InBrowserTracing[In-Browser Tracing]
        LangChain --> LangSmithTracing[LangSmith Tracing]
    end

    subgraph State Management
        StreamlitUI --> SessionState[Session State]
        SessionState --> Messages[Chat Messages]
        SessionState --> Memory[Conversation Memory]
        SessionState --> TraceData[Trace Data]
    end

    style StreamlitUI fill:#f9f,stroke:#333,stroke-width:2px
    style LangChain fill:#bbf,stroke:#333,stroke-width:2px
    style OpenAI fill:#bfb,stroke:#333,stroke-width:2px
    style InBrowserTracing fill:#fbb,stroke:#333,stroke-width:2px
    style LangSmithTracing fill:#fbf,stroke:#333,stroke-width:2px
```

## Component Diagram

```mermaid
classDiagram
    class StreamlitUI {
        +display_chat_history()
        +display_trace_info()
        +handle_user_input()
        +toggle_trace_display()
    }

    class ChatOpenAI {
        +temperature: float
        +api_key: string
        +generate_response()
    }

    class ConversationChain {
        +llm: LLM
        +memory: Memory
        +predict(input: string)
    }

    class ConversationBufferMemory {
        +chat_memory: list
        +clear()
        +save_context()
        +load_memory_variables()
    }

    class StreamlitTracer {
        +trace_data: dict
        +start_time: float
        +reset()
        +on_llm_start()
        +on_llm_end()
        +on_chain_start()
        +on_chain_end()
        +get_trace_data()
    }

    class LangChainTracer {
        +project_name: string
        +api_key: string
    }

    StreamlitUI --> ConversationChain: uses
    ConversationChain --> ChatOpenAI: uses
    ConversationChain --> ConversationBufferMemory: uses
    StreamlitUI --> StreamlitTracer: displays data from
    ConversationChain --> StreamlitTracer: sends events to
    ConversationChain --> LangChainTracer: sends events to
```

## Sequence Diagram: Basic Conversation Flow

```mermaid
sequenceDiagram
    participant User
    participant StreamlitUI
    participant ConversationChain
    participant ChatOpenAI

    User->>StreamlitUI: Enter message
    StreamlitUI->>StreamlitUI: Add message to history
    StreamlitUI->>ConversationChain: predict(input=message)
    ConversationChain->>ConversationBufferMemory: load_memory_variables()
    ConversationChain->>ChatOpenAI: generate_response()
    ChatOpenAI-->>ConversationChain: response
    ConversationChain->>ConversationBufferMemory: save_context()
    ConversationChain-->>StreamlitUI: response
    StreamlitUI->>StreamlitUI: Add response to history
    StreamlitUI-->>User: Display response
```

## Sequence Diagram: Tracing Flow

```mermaid
sequenceDiagram
    participant User
    participant StreamlitUI
    participant ConversationChain
    participant StreamlitTracer
    participant LangSmithTracer

    User->>StreamlitUI: Toggle "Show Trace Information"
    User->>StreamlitUI: Enter message

    StreamlitUI->>StreamlitTracer: reset()
    StreamlitUI->>ConversationChain: predict(callbacks=[tracers])

    ConversationChain->>StreamlitTracer: on_chain_start()
    ConversationChain->>LangSmithTracer: on_chain_start()

    ConversationChain->>StreamlitTracer: on_llm_start()
    ConversationChain->>LangSmithTracer: on_llm_start()

    ConversationChain->>StreamlitTracer: on_llm_end()
    ConversationChain->>LangSmithTracer: on_llm_end()

    ConversationChain->>StreamlitTracer: on_chain_end()
    ConversationChain->>LangSmithTracer: on_chain_end()

    ConversationChain-->>StreamlitUI: response

    StreamlitUI->>StreamlitTracer: get_trace_data()
    StreamlitTracer-->>StreamlitUI: trace_data

    StreamlitUI->>StreamlitUI: Display trace information
    StreamlitUI-->>User: Display response and trace
```

## Data Flow Diagram

```mermaid
graph TD
    UserInput[User Input] --> StreamlitState[Streamlit Session State]
    StreamlitState --> ConversationMemory[Conversation Memory]

    UserInput --> ConversationChain[Conversation Chain]
    ConversationMemory --> ConversationChain

    ConversationChain --> Tracers[Tracers]
    Tracers --> StreamlitTracer[Streamlit Tracer]
    Tracers --> LangSmithTracer[LangSmith Tracer]

    ConversationChain --> OpenAIAPI[OpenAI API]
    OpenAIAPI --> LLMResponse[LLM Response]

    LLMResponse --> ConversationChain
    ConversationChain --> StreamlitState

    StreamlitTracer --> TraceData[Trace Data]
    TraceData --> TraceDisplay[Trace Display]

    LangSmithTracer --> LangSmithAPI[LangSmith API]
    LangSmithAPI --> LangSmithDashboard[LangSmith Dashboard]

    style UserInput fill:#f9f,stroke:#333,stroke-width:2px
    style LLMResponse fill:#bfb,stroke:#333,stroke-width:2px
    style TraceDisplay fill:#fbb,stroke:#333,stroke-width:2px
    style LangSmithDashboard fill:#fbf,stroke:#333,stroke-width:2px
```

## State Transition Diagram

```mermaid
stateDiagram-v2
    [*] --> Initial

    state Initial {
        [*] --> Ready
    }

    state Ready {
        [*] --> WaitingForInput
    }

    WaitingForInput --> ProcessingInput: User enters message
    ProcessingInput --> GeneratingResponse: Send to LLM
    GeneratingResponse --> DisplayingResponse: Response received
    DisplayingResponse --> WaitingForInput: Display complete

    WaitingForInput --> ClearingHistory: User clicks "Clear History"
    ClearingHistory --> WaitingForInput: History cleared

    WaitingForInput --> TogglingTrace: User toggles trace display
    TogglingTrace --> WaitingForInput: Display updated

    state ProcessingInput {
        [*] --> CapturingTraceStart
        CapturingTraceStart --> SendingToLLM
    }

    state DisplayingResponse {
        [*] --> UpdatingChatHistory
        UpdatingChatHistory --> DisplayingTraceInfo: If trace enabled
        DisplayingTraceInfo --> [*]
        UpdatingChatHistory --> [*]: If trace disabled
    }
```

## Component Interaction Diagram

```mermaid
graph TD
    subgraph "User Interface"
        ChatInput[Chat Input]
        ChatHistory[Chat History]
        TraceToggle[Trace Toggle]
        TraceDisplay[Trace Display]
        ClearButton[Clear History Button]
    end

    subgraph "State Management"
        SessionState[Session State]
        Messages[Messages]
        Memory[Memory]
        Tracer[Tracer]
    end

    subgraph "LangChain Components"
        ConversationChain[Conversation Chain]
        ChatOpenAI[ChatOpenAI]
        ConversationMemory[Conversation Memory]
    end

    subgraph "Tracing System"
        StreamlitTracer[Streamlit Tracer]
        LangSmithTracer[LangSmith Tracer]
        TraceTabs[Trace Tabs]
    end

    ChatInput --> SessionState
    SessionState --> Messages
    TraceToggle --> SessionState

    SessionState --> ConversationChain
    ConversationChain --> ChatOpenAI
    ConversationChain --> ConversationMemory
    Memory --> ConversationMemory

    ConversationChain --> StreamlitTracer
    ConversationChain --> LangSmithTracer

    StreamlitTracer --> Tracer
    Tracer --> TraceDisplay
    TraceDisplay --> TraceTabs

    ClearButton --> SessionState
    ClearButton --> Memory
    ClearButton --> Tracer

    Messages --> ChatHistory

    style ChatInput fill:#f9f,stroke:#333,stroke-width:2px
    style TraceToggle fill:#f9f,stroke:#333,stroke-width:2px
    style TraceDisplay fill:#fbb,stroke:#333,stroke-width:2px
    style ChatOpenAI fill:#bfb,stroke:#333,stroke-width:2px
    style StreamlitTracer fill:#bbf,stroke:#333,stroke-width:2px
    style LangSmithTracer fill:#fbf,stroke:#333,stroke-width:2px
```

## Trace Data Structure

```mermaid
classDiagram
    class TraceData {
        +llm_calls: List
        +chain_starts: List
        +chain_ends: List
        +tool_starts: List
        +tool_ends: List
        +errors: List
        +text: List
    }

    class LLMCallEvent {
        +type: string
        +time: string
        +name: string
        +prompts: List[string]
    }

    class LLMEndEvent {
        +type: string
        +time: string
        +response: string
    }

    class ChainStartEvent {
        +type: string
        +time: string
        +name: string
        +inputs: string
    }

    class ChainEndEvent {
        +type: string
        +time: string
        +outputs: string
    }

    class ErrorEvent {
        +type: string
        +time: string
        +error: string
    }

    class TextEvent {
        +type: string
        +time: string
        +text: string
    }

    TraceData --> LLMCallEvent: contains
    TraceData --> LLMEndEvent: contains
    TraceData --> ChainStartEvent: contains
    TraceData --> ChainEndEvent: contains
    TraceData --> ErrorEvent: contains
    TraceData --> TextEvent: contains
```

These diagrams can be rendered using Mermaid-compatible tools or viewers. You can use the [Mermaid Live Editor](https://mermaid.live/) to view and edit these diagrams interactively.
