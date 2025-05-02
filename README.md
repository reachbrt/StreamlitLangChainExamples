# LangChain Streamlit Chatbot

A conversational AI chatbot with RAG capabilities, built with Streamlit and LangChain.

## Features

- Chat with AI using OpenAI's language models
- Upload documents to create custom knowledge bases (RAG)
- View detailed tracing information for debugging
- LangSmith integration for advanced analysis

## Quick Start

1. **Clone and set up:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add your API key:**
   ```bash
   mkdir -p .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Edit `.streamlit/secrets.toml` and add your OpenAI API key.

3. **Run the app:**
   ```bash
   streamlit run chatbot.py
   ```

4. **Open in browser:**
   The app will be available at http://localhost:8501

## Using the Chatbot

- **Regular Chat:** Ask any question and get AI responses
- **RAG Chat:** Upload documents and ask questions about them
- **Tracing:** Toggle "Show Trace Information" to see how the AI processes your questions
- **Navigation:** Use the sidebar menu to switch between different pages

## Uploading Documents

1. Go to the "RAG Chat" page
2. Use the file uploader in the sidebar
3. Select document type (general, research, technical)
4. Ask questions about your document

## Troubleshooting

- Make sure your API keys are correctly set in `.streamlit/secrets.toml`
- If you see "ModuleNotFoundError", run `pip install tiktoken faiss-cpu`
- For detailed logs, toggle "Show Trace Information" in the sidebar

## Author

**Bharat Kumar Subramanian**
Email: reachbrt@gmail.com

## License

[MIT License](LICENSE)
