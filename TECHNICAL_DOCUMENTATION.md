# LangChain Streamlit Chatbot: Technical Documentation

This document provides a detailed technical explanation of the LangChain Streamlit Chatbot implementation, with a focus on the tracing functionality.

## Author

**Bharat Kumar Subramanian**
Email: reachbrt@gmail.com

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Breakdown](#component-breakdown)
3. [Data Flow](#data-flow)
4. [Tracing Implementation](#tracing-implementation)
   - [In-Browser Tracing](#in-browser-tracing)
   - [LangSmith Tracing](#langsmith-tracing)
5. [Code Walkthrough](#code-walkthrough)
6. [Sequence Diagrams](#sequence-diagrams)
7. [Troubleshooting](#troubleshooting)

## Architecture Overview

The chatbot application is built using the following key technologies:

- **Streamlit**: Provides the web interface and state management
- **LangChain**: Orchestrates the conversation flow and LLM interactions
- **OpenAI API**: Powers the language model responses
- **LangSmith**: Offers remote tracing and debugging capabilities

The application follows a simple architecture:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  LangChain  │────▶│  OpenAI API │
│  Interface  │◀────│  Framework  │◀────│             │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ In-Browser  │     │  LangSmith  │
│   Tracing   │     │   Tracing   │
└─────────────┘     └─────────────┘
```

## Component Breakdown

### 1. Streamlit Interface

The Streamlit interface handles:
- User input collection
- Display of conversation history
- Visualization of trace information
- State management across interactions

### 2. LangChain Components

The application uses several LangChain components:
- `ChatOpenAI`: Interface to the OpenAI chat models
- `ConversationChain`: Manages the conversation flow
- `ConversationBufferMemory`: Stores conversation history
- `LangChainTracer`: Sends trace information to LangSmith

### 3. Custom Tracing

The application implements a custom tracing solution:
- `StreamlitTracer`: A custom callback handler that captures detailed trace information
- Tabbed interface for displaying different types of trace data

### 4. State Management

The application uses Streamlit's session state to manage:
- Conversation history
- Memory for the LangChain conversation
- Trace data
- UI state (like whether to show trace information)

## Data Flow

The data flow through the application follows this sequence:

1. User enters a message in the chat input
2. The message is added to the conversation history
3. The message is sent to the LangChain conversation chain
4. Tracers capture information at each step of processing
5. The LLM generates a response
6. The response is added to the conversation history and displayed
7. Trace information is displayed if enabled

## Tracing Implementation

### In-Browser Tracing

The in-browser tracing is implemented using a custom callback handler that captures detailed information about each step of the LangChain execution process.

#### StreamlitTracer Class

The `StreamlitTracer` class extends LangChain's `BaseCallbackHandler` and implements methods to capture:

- LLM calls (start and end)
- Chain operations (start and end)
- Errors
- Text generation

Each event is timestamped and stored in a structured format for display.

#### Trace Display

The trace information is displayed in a tabbed interface with sections for:

1. **LLM Calls**: Shows the prompts sent to the LLM and the responses received
2. **Chain Operations**: Displays the inputs and outputs of each chain
3. **Errors**: Shows any errors that occurred during processing
4. **Text Events**: Displays text generation events

### LangSmith Tracing

LangSmith tracing is implemented using:

1. Environment variables to configure the LangSmith connection
2. The `LangChainTracer` to send trace information to LangSmith
3. A sidebar section that displays the LangSmith status and project information

## Code Walkthrough

### Imports and Setup

```python
import streamlit as st
import os
import uuid
import json
import time
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.base import BaseCallbackHandler
```

These imports provide:
- Streamlit components for the UI
- Standard libraries for utilities
- LangChain components for conversation management
- LangSmith components for remote tracing
- Callback handlers for custom tracing

### Custom Tracer Implementation

```python
class StreamlitTracer(BaseCallbackHandler):
    """Callback handler for tracing in Streamlit."""

    def __init__(self):
        """Initialize the tracer with empty trace data."""
        super().__init__()
        self.reset()

    def reset(self):
        """Reset the trace data."""
        self.trace_data = {
            "llm_calls": [],
            "chain_starts": [],
            "chain_ends": [],
            "tool_starts": [],
            "tool_ends": [],
            "errors": [],
            "text": []
        }
        self.start_time = time.time()
```

The `StreamlitTracer` class:
- Extends `BaseCallbackHandler` to receive callbacks from LangChain
- Maintains a dictionary of trace data categorized by event type
- Provides a `reset()` method to clear trace data between conversations
- Records the start time to calculate elapsed time for each event

### Event Handlers

```python
def on_llm_start(self, serialized, prompts, **kwargs):
    """Record when LLM starts."""
    self.trace_data["llm_calls"].append({
        "type": "llm_start",
        "time": self._get_timestamp(),
        "name": serialized.get("name", "LLM"),
        "prompts": prompts
    })
```

Each event handler:
- Captures relevant information about the event
- Adds a timestamp to show when the event occurred
- Stores the data in the appropriate category

### Tracing Function

```python
def get_response_with_tracing(prompt, conversation_id=None):
    # Reset the Streamlit tracer
    st.session_state["tracer"].reset()

    # Create a list of tracers
    tracers = [st.session_state["tracer"]]

    # Add LangSmith tracer if enabled
    if st.secrets["LANGCHAIN_TRACING_V2"].lower() == "true":
        langsmith_tracer = LangChainTracer(
            project_name=st.secrets["LANGCHAIN_PROJECT"],
        )
        tracers.append(langsmith_tracer)

    # Get response from LangChain with tracing
    response = st.session_state["conversation"].predict(
        input=prompt,
        callbacks=tracers
    )

    # Return the response
    return response
```

The tracing function:
- Resets the Streamlit tracer to clear previous trace data
- Creates a list of tracers to use (always includes the Streamlit tracer)
- Conditionally adds the LangSmith tracer if enabled
- Calls the conversation chain with the tracers attached
- Returns the response from the LLM

### Trace Display

```python
# --- Display Trace Information ---
if st.session_state["show_trace"] and "tracer" in st.session_state:
    st.markdown("---")
    st.subheader("Trace Information")

    trace_data = st.session_state["tracer"].get_trace_data()

    # Create tabs for different trace types
    trace_tabs = st.tabs(["LLM Calls", "Chain Operations", "Errors", "Text Events"])
```

The trace display:
- Only shows if the user has enabled trace display
- Uses tabs to organize different types of trace information
- Formats the trace data for readability
- Uses appropriate Streamlit components for different data types

## Sequence Diagrams

### Basic Conversation Flow

```
┌──────┐          ┌─────────┐          ┌───────────┐          ┌─────┐
│ User │          │Streamlit│          │ LangChain │          │ LLM │
└──┬───┘          └────┬────┘          └─────┬─────┘          └──┬──┘
   │                    │                     │                   │
   │ Enter Message      │                     │                   │
   │ ─────────────────> │                     │                   │
   │                    │                     │                   │
   │                    │ Send Prompt         │                   │
   │                    │ ──────────────────> │                   │
   │                    │                     │                   │
   │                    │                     │ Generate Response │
   │                    │                     │ ─────────────────>│
   │                    │                     │                   │
   │                    │                     │ <─────────────────│
   │                    │ <────────────────── │                   │
   │                    │                     │                   │
   │ <───────────────── │                     │                   │
   │                    │                     │                   │
```

### Tracing Flow

```
┌──────┐    ┌─────────┐    ┌───────────┐    ┌─────────────┐    ┌─────────┐
│ User │    │Streamlit│    │ LangChain │    │StreamlitTrace│    │LangSmith│
└──┬───┘    └────┬────┘    └─────┬─────┘    └──────┬──────┘    └────┬────┘
   │              │               │                 │                │
   │ Enable Trace │               │                 │                │
   │ ────────────>│               │                 │                │
   │              │               │                 │                │
   │ Send Message │               │                 │                │
   │ ────────────>│               │                 │                │
   │              │               │                 │                │
   │              │ Call with     │                 │                │
   │              │ Tracers       │                 │                │
   │              │ ─────────────>│                 │                │
   │              │               │                 │                │
   │              │               │ on_llm_start    │                │
   │              │               │ ───────────────>│                │
   │              │               │                 │                │
   │              │               │ on_llm_start    │                │
   │              │               │ ───────────────────────────────> │
   │              │               │                 │                │
   │              │               │ on_llm_end      │                │
   │              │               │ ───────────────>│                │
   │              │               │                 │                │
   │              │               │ on_llm_end      │                │
   │              │               │ ───────────────────────────────> │
   │              │               │                 │                │
   │              │ <─────────────│                 │                │
   │              │               │                 │                │
   │              │ Display Trace │                 │                │
   │              │ <─────────────────────────────> │                │
   │              │               │                 │                │
   │ <────────────│               │                 │                │
   │              │               │                 │                │
```

## Troubleshooting

### Common Issues

1. **Nested Expanders Error**
   - Streamlit does not allow expanders to be nested inside other expanders
   - Solution: Use tabs or other UI components instead of nested expanders

2. **LangSmith Connection Issues**
   - Check that your LangSmith API key is correct
   - Verify that `LANGCHAIN_TRACING_V2` is set to "true"
   - Ensure you have internet connectivity

3. **OpenAI API Issues**
   - Verify your OpenAI API key is correct
   - Check for rate limiting or quota issues
   - Ensure you have internet connectivity

4. **Trace Display Issues**
   - Make sure the "Show Trace Information" checkbox is checked
   - Verify that the tracer is properly initialized in session state
   - Check for any JavaScript console errors in the browser

### Debugging Tips

1. **Check the Streamlit Logs**
   - Run Streamlit with the `--log_level=debug` flag
   - Look for any error messages or warnings

2. **Inspect Session State**
   - Add a section to display the current session state for debugging
   - Use `st.write(st.session_state)` to see all state variables

3. **Test Tracers Individually**
   - Try using only the Streamlit tracer first
   - Then add the LangSmith tracer to isolate issues

4. **Verify Callback Registration**
   - Add print statements in callback methods to verify they're being called
   - Check that the tracers are properly added to the callbacks list
