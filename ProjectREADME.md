# Model Context Protocol

A Python project that implements the Model Context Protocol for enhanced language model interactions.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- A virtual environment manager (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/model-context-protocol.git
cd model-context-protocol
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Environment Setup

1. Create a `.env` file in the root directory with your API keys:
```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

- `codeserver/`: Contains the code server implementation
- `server/`: Server-related code
- `docs/`: Project documentation
- `main.py`: Main entry point of the application

## Running the Project

To run the project:

```bash
python main.py
```

## Development

The project uses modern Python tooling:
- `pyproject.toml` for dependency management
- `uv.lock` for dependency locking
- `.python-version` for Python version specification

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
