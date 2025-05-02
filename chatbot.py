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

# RAG-specific imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Setup ---
# The title will be set based on the current page

# Create a custom callback handler to capture trace information
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

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Record when LLM starts."""
        self.trace_data["llm_calls"].append({
            "type": "llm_start",
            "time": self._get_timestamp(),
            "name": serialized.get("name", "LLM"),
            "prompts": prompts
        })

    def on_llm_end(self, response, **kwargs):
        """Record when LLM ends."""
        self.trace_data["llm_calls"].append({
            "type": "llm_end",
            "time": self._get_timestamp(),
            "response": str(response)
        })

    def on_llm_error(self, error, **kwargs):
        """Record LLM errors."""
        self.trace_data["errors"].append({
            "type": "llm_error",
            "time": self._get_timestamp(),
            "error": str(error)
        })

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Record when chain starts."""
        self.trace_data["chain_starts"].append({
            "type": "chain_start",
            "time": self._get_timestamp(),
            "name": serialized.get("name", "Chain"),
            "inputs": str(inputs)
        })

    def on_chain_end(self, outputs, **kwargs):
        """Record when chain ends."""
        self.trace_data["chain_ends"].append({
            "type": "chain_end",
            "time": self._get_timestamp(),
            "outputs": str(outputs)
        })

    def on_chain_error(self, error, **kwargs):
        """Record chain errors."""
        self.trace_data["errors"].append({
            "type": "chain_error",
            "time": self._get_timestamp(),
            "error": str(error)
        })

    def on_text(self, text, **kwargs):
        """Record text."""
        self.trace_data["text"].append({
            "type": "text",
            "time": self._get_timestamp(),
            "text": text
        })

    def _get_timestamp(self):
        """Get a formatted timestamp."""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.2f}s"

    def get_trace_data(self):
        """Get the trace data."""
        return self.trace_data

# Initialize the tracer
if "tracer" not in st.session_state:
    st.session_state["tracer"] = StreamlitTracer()

# Set up LangSmith tracing environment variables
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# Initialize LangSmith client (for API access if needed)
langsmith_client = Client(
    api_key=st.secrets["LANGCHAIN_API_KEY"],
    api_url=st.secrets["LANGCHAIN_ENDPOINT"]
)

# --- Sidebar Navigation Menu ---
st.sidebar.title("Navigation")

# Define the available pages
pages = {
    "Chat": "chat",
    "RAG Chat": "rag",
    "About": "about",
    "Documentation": "docs"
}

# Initialize the current page in session state if not already set
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "chat"

# Create the navigation menu
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Update the current page in session state
if selected_page:
    st.session_state["current_page"] = pages[selected_page]

# --- Tracing Settings ---
st.sidebar.markdown("---")
st.sidebar.title("Tracing Settings")

# Add a toggle for showing/hiding trace information
if "show_trace" not in st.session_state:
    st.session_state["show_trace"] = False

show_trace = st.sidebar.checkbox("Show Trace Information", value=st.session_state["show_trace"])
st.session_state["show_trace"] = show_trace

# Display LangSmith information
st.sidebar.markdown("---")
st.sidebar.subheader("LangSmith Integration")
st.sidebar.markdown("""
This chatbot also uses LangSmith for remote tracing and debugging.
""")

langsmith_url = f"https://smith.langchain.com/projects/{st.secrets['LANGCHAIN_PROJECT']}"
st.sidebar.markdown(f"[View traces in LangSmith]({langsmith_url})")

# Display tracing status
tracing_status = "Enabled" if st.secrets["LANGCHAIN_TRACING_V2"].lower() == "true" else "Disabled"
st.sidebar.info(f"LangSmith Tracing: {tracing_status}")
st.sidebar.info(f"Project: {st.secrets['LANGCHAIN_PROJECT']}")

# Initialize LLM (replace with your API key or preferred LLM)
llm = ChatOpenAI(temperature=0.7, openai_api_key=st.secrets["OPENAI_API_KEY"])

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you today?"}]

# Initialize RAG chat history in session state
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = [{"role": "assistant", "content": "Ask me anything about our research team, RAG, or AI development. I'll provide answers based on our knowledge base."}]

# Initialize memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory()

# Initialize conversation chain
if "conversation" not in st.session_state:
    st.session_state["conversation"] = ConversationChain(
        llm=llm,
        memory=st.session_state["memory"]
    )

# Function to process uploaded file for RAG
def process_file_for_rag(file_content, file_type):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(file_content)

    # Create embeddings and store in vector database
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Create the RAG prompt template based on file type
    if file_type == "research":
        template = """
        You are an AI assistant for an AI research team. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    elif file_type == "technical":
        template = """
        You are a technical documentation assistant. Use the following pieces of context to answer the question at the end.
        Provide code examples when relevant. If you don't know the answer, just say that you don't know.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    else:  # general
        template = """
        You are a helpful assistant. Use the following pieces of context to answer the question at the end.
        Be concise and clear in your response. If you don't know the answer, just say that you don't know.

        Context:
        {context}

        Question: {question}

        Answer:
        """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain

# Initialize default RAG chain with sample knowledge if not already in session state
if "rag_chain" not in st.session_state:
    # Load the default knowledge base
    try:
        with open("knowledge_base.txt", "r") as f:
            default_knowledge = f.read()

        with st.spinner("Initializing RAG system..."):
            st.session_state["rag_chain"] = process_file_for_rag(default_knowledge, "research")
            st.session_state["current_rag_file"] = "Default Knowledge Base"
    except FileNotFoundError:
        # If the file doesn't exist, create an empty RAG chain
        st.session_state["rag_chain"] = None
        st.session_state["current_rag_file"] = None

# Function to create a trace for regular conversation
def get_response_with_tracing(prompt):
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

# --- Page Content Based on Navigation ---
if st.session_state["current_page"] == "chat":
    # Chat page
    st.title("LangChain Chatbot")

    # Display chat interface
    display_chat = True
    chat_messages = "messages"
    get_response = get_response_with_tracing

elif st.session_state["current_page"] == "rag":
    # RAG chat page
    st.title("Knowledge Base Chat")

    st.markdown("""
    This chat uses Retrieval-Augmented Generation (RAG) to provide answers based on your uploaded documents.
    Upload a text file to create a knowledge base, then ask questions about the content.
    """)

    # File uploader
    with st.sidebar:
        st.subheader("Upload Knowledge Base")
        uploaded_file = st.file_uploader("Choose a text file", type=["txt", "md", "py", "js", "html", "css", "json", "csv"])

        # Document type selector
        doc_type = st.selectbox(
            "Document Type",
            options=["general", "research", "technical"],
            help="Select the type of document you're uploading to optimize the AI's responses"
        )

        # Process the uploaded file
        if uploaded_file is not None:
            # Check if this is a new file
            if "current_rag_file" not in st.session_state or st.session_state["current_rag_file"] != uploaded_file.name:
                try:
                    # Read the file
                    file_content = uploaded_file.read().decode("utf-8")

                    with st.spinner("Processing your document..."):
                        # Process the file for RAG
                        st.session_state["rag_chain"] = process_file_for_rag(file_content, doc_type)
                        st.session_state["current_rag_file"] = uploaded_file.name

                        # Clear the chat history when a new file is uploaded
                        st.session_state["rag_messages"] = [{"role": "assistant", "content": f"I've processed '{uploaded_file.name}'. Ask me anything about it!"}]

                    st.success(f"Successfully processed '{uploaded_file.name}'")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        # Display current knowledge base
        if "current_rag_file" in st.session_state and st.session_state["current_rag_file"]:
            st.info(f"Current Knowledge Base: {st.session_state['current_rag_file']}")

    # Function to get response from RAG
    def get_rag_response(prompt):
        # Check if RAG chain is available
        if "rag_chain" not in st.session_state or st.session_state["rag_chain"] is None:
            return "Please upload a document first to create a knowledge base."

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

        try:
            # Get response from RAG chain with tracing
            response = st.session_state["rag_chain"](
                {"query": prompt},
                callbacks=tracers
            )

            # Return the response text
            return response["result"]
        except Exception as e:
            return f"Error generating response: {str(e)}"

    # Display chat interface
    display_chat = True
    chat_messages = "rag_messages"
    get_response = get_rag_response

elif st.session_state["current_page"] == "about":
    # About page
    st.title("About Us")

    st.markdown("""
    ## AI Research Team

    We are a team of AI researchers and engineers focused on developing advanced conversational AI systems. Our mission is to create intelligent, helpful, and transparent AI assistants that can understand and respond to human needs effectively.

    ### Our Team

    Our team consists of experts in:

    - **AI Research** - Specializing in conversational AI systems
    - **Natural Language Processing** - Focusing on language understanding
    - **Software Development** - Building robust AI applications
    - **User Experience Design** - Creating intuitive user interfaces

    ### Our Approach

    We believe in building AI systems that are:

    - **Transparent**: Users should understand how the AI works
    - **Accountable**: Clear tracing and logging of all interactions
    - **Helpful**: Focused on solving real user problems
    - **Ethical**: Designed with privacy and fairness in mind

    ### This Project

    This chatbot demonstrates our approach to building transparent AI systems. It features:

    - Interactive chat interface with a clean, user-friendly design
    - Conversation memory to maintain context across interactions
    - In-browser tracing for real-time visibility into the AI's reasoning
    - LangSmith integration for detailed analysis and debugging
    - Retrieval-Augmented Generation (RAG) for grounding responses in factual information

    ### Contact Us

    For more information about our research or to collaborate with us, please contact us at research@example.com.
    """)

    # Don't display chat interface on this page
    display_chat = False

elif st.session_state["current_page"] == "docs":
    # Documentation page
    st.title("Documentation")

    st.markdown("""
    ## Technical Documentation

    This page provides a brief overview of the technical aspects of the chatbot. For more detailed documentation, please refer to the `TECHNICAL_DOCUMENTATION.md` and `DIAGRAMS.md` files in the project repository.

    ### Architecture

    The chatbot follows a simple architecture:

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

    ### Tracing Implementation

    The chatbot implements two types of tracing:

    1. **In-Browser Tracing**: Uses a custom callback handler to capture and display trace information directly in the browser.
    2. **LangSmith Tracing**: Sends trace information to LangSmith for remote analysis and debugging.

    ### Usage Instructions

    1. Navigate to the "Chat" page to interact with the chatbot
    2. Toggle "Show Trace Information" in the sidebar to view tracing details
    3. Use the "Clear Chat History" button to reset the conversation
    4. Click the LangSmith link to view remote traces
    """)

    # Don't display chat interface on this page
    display_chat = False
else:
    # Default to chat page if something goes wrong
    st.title("LangChain Chatbot")
    display_chat = True



# Only display chat interface on the chat or RAG pages
if display_chat:
    # --- Display Chat History ---
    for msg in st.session_state[chat_messages]:
        st.chat_message(msg["role"]).write(msg["content"])

    # --- Display Trace Information ---
    if st.session_state["show_trace"] and "tracer" in st.session_state:
        st.markdown("---")
        st.subheader("Trace Information")

        trace_data = st.session_state["tracer"].get_trace_data()

        # Create tabs for different trace types
        trace_tabs = st.tabs(["LLM Calls", "Chain Operations", "Errors", "Text Events"])

        # Tab 1: LLM Calls
        with trace_tabs[0]:
            if not trace_data["llm_calls"]:
                st.info("No LLM calls recorded yet.")
            else:
                for i, call in enumerate(trace_data["llm_calls"]):
                    if call["type"] == "llm_start":
                        st.markdown(f"### LLM Start ({call['time']})")
                        st.markdown(f"**Name:** {call['name']}")
                        for j, prompt in enumerate(call['prompts']):
                            st.markdown(f"**Prompt {j+1}:**")
                            st.code(prompt, language="text")
                    elif call["type"] == "llm_end":
                        st.markdown(f"### LLM End ({call['time']})")
                        st.markdown("**Response:**")
                        st.code(call['response'], language="text")
                    st.markdown("---")

        # Tab 2: Chain Operations
        with trace_tabs[1]:
            if not trace_data["chain_starts"] and not trace_data["chain_ends"]:
                st.info("No chain operations recorded yet.")
            else:
                # Display chain starts
                if trace_data["chain_starts"]:
                    st.markdown("### Chain Starts")
                    for start in trace_data["chain_starts"]:
                        st.markdown(f"**{start['name']} ({start['time']})**")
                        st.markdown("**Inputs:**")
                        st.code(start['inputs'], language="text")
                        st.markdown("---")

                # Display chain ends
                if trace_data["chain_ends"]:
                    st.markdown("### Chain Ends")
                    for end in trace_data["chain_ends"]:
                        st.markdown(f"**End ({end['time']})**")
                        st.markdown("**Outputs:**")
                        st.code(end['outputs'], language="text")
                        st.markdown("---")

        # Tab 3: Errors
        with trace_tabs[2]:
            if not trace_data["errors"]:
                st.info("No errors recorded.")
            else:
                for error in trace_data["errors"]:
                    st.error(f"{error['type']} at {error['time']}: {error['error']}")
                    st.markdown("---")

        # Tab 4: Text Events
        with trace_tabs[3]:
            if not trace_data["text"]:
                st.info("No text events recorded yet.")
            else:
                for text_event in trace_data["text"]:
                    st.markdown(f"**Text ({text_event['time']})**")
                    st.code(text_event['text'], language="text")
                    st.markdown("---")

    # --- Chat Input ---
    if prompt := st.chat_input("Say something"):
        # Generate a conversation ID if not already in session state
        if "conversation_id" not in st.session_state:
            st.session_state["conversation_id"] = f"conv_{str(uuid.uuid4())[:8]}"

        # Add user message to chat history
        st.session_state[chat_messages].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get response using the appropriate function (regular or RAG)
        with st.spinner("Thinking..."):
            ai_response = get_response(prompt)

        # Store and display the response
        st.session_state[chat_messages].append({"role": "assistant", "content": ai_response})
        st.chat_message("assistant").write(ai_response)

        # Force a rerun to update the trace display
        if st.session_state["show_trace"]:
            st.rerun()

# --- Optional: Clear Chat History ---
if st.sidebar.button("Clear Chat History"):
    # Clear the appropriate chat history based on current page
    if st.session_state["current_page"] == "chat":
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you today?"}]
        st.session_state["memory"].clear()
        st.session_state["conversation"] = ConversationChain(
            llm=llm,
            memory=st.session_state["memory"]
        )
    elif st.session_state["current_page"] == "rag":
        st.session_state["rag_messages"] = [{"role": "assistant", "content": "Ask me anything about our research team, RAG, or AI development. I'll provide answers based on our knowledge base."}]

    # Generate a new conversation ID
    st.session_state["conversation_id"] = f"conv_{str(uuid.uuid4())[:8]}"

    # Reset the tracer
    if "tracer" in st.session_state:
        st.session_state["tracer"].reset()

    st.rerun()