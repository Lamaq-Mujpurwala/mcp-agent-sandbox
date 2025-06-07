# MCP Agent Python Starter (Built on MCP SDK)

A practical starter project using the [Model Context Protocol (MCP)] to demonstrate how to build local tool-using AI agents with Claude, LangChain, and Groq.

## 📚 Reference

This project is built on top of the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and follows the [Model Context Protocol specification](https://modelcontextprotocol.io).


# Model Context Protocol (MCP) – Python Implementation
---

## 🧠 What is MCP?

**Model Context Protocol (MCP)** is a standard developed by Anthropic to allow large language models (LLMs) to interact with external tools and environments. It introduces a communication method where the LLM acts as a "client" and interacts with tools via a "server" using a standardized transport layer like `STDIO` or `SSE`.

---

## 🛠️ Prerequisites

Make sure you have the following installed:

- Python 3.11 or higher
- `uv` (package manager): [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
- [Claude Desktop](https://www.anthropic.com/index/claude-desktop) (for testing with MCP tools)
- [Cursor](https://cursor.sh/) (recommended editor for debugging)
- Basic understanding of:
  - JSON-based communication
  - LangChain
  - Python or TypeScript
  - MCP documentation: [modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## 📦 Dependencies

- mcp[cli] (via uv)
- langchain
- langchain_openai
- langchain_groq
- python-dotenv


## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/model-context-protocol.git
cd model-context-protocol

# Initialize environment using uv
uv init

# Create folders
mkdir server docs testcode

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## 📄 Documentation Setup

- Download the MCP Python SDK README
- Save it in your docs/ folder.
- Load it into Claude Desktop and prompt it to generate a server and client using that reference.

✍️ Tip: Be specific in your prompts.

## ▶️ Running the Client
In a new terminal window:
```bash
source .venv/bin/activate  # Reactivate env
uv add mcp-use langchain langchain_groq python-dotenv
```
Create a .env file:
```bash
env
GROQ_API_KEY=your_groq_api_key  # Get from groq.com
```
Run the client:
```bash
uv run path/to/your_client_file.py
```

🧪 Testing with Claude
You can also install your tool config to Claude:
```bash
uv install mcp path/to/your_server_file.py
```
This creates a config file that lets Claude Desktop access your tool natively for use in prompts.

> ⚠️ Note: This project uses the open-source MCP SDK (MIT Licensed). This repository itself is for demonstration purposes and retains custom licensing. See LICENSE for terms.

