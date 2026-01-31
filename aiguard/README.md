# AIGuard

Security guardrails system for LLM applications - protecting against prompt injection, jailbreaking, PII leakage, and encoding attacks.

## Features

- **Multi-Layer Prompt Injection Detection**
  - Pattern-based detection (heuristics, regex)
  - Semantic similarity detection (sentence-transformers)
  - Optional LLM-based detection

- **Jailbreak Detection**
  - DAN and role-playing attack detection
  - Adversarial suffix detection
  - Character-level attack detection

- **PII Detection & Redaction**
  - Email addresses, phone numbers, SSN, credit cards
  - Named entities (people, organizations, locations)
  - Custom pattern matching

- **Encoding Attack Detection**
  - Base64, hex, unicode tricks
  - Rot13, caesar ciphers
  - Multi-layer encoding

- **Output Filtering**
  - Data leakage prevention
  - Malicious content filtering
  - Response sanitization

- **FastAPI Integration**
  - Easy middleware integration
  - Configurable detection layers
  - Comprehensive logging

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/aiguard.git
cd aiguard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spacy model
python -m spacy download en_core_web_lg

# Copy environment file
cp .env.example .env
```

## Quick Start

```python
from fastapi import FastAPI
from src.middleware.aiguard_middleware import AIGuardMiddleware, AIGuardConfig

app = FastAPI()

# Configure AIGuard
config = AIGuardConfig(
    enable_prompt_injection_detection=True,
    enable_jailbreak_detection=True,
    enable_pii_detection=True,
    enable_encoding_detection=True,
)

# Add middleware
app.add_middleware(AIGuardMiddleware, config=config)

@app.post("/chat")
async def chat(message: str):
    # Your LLM logic here
    return {"response": "Hello!"}
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest src/tests/test_prompt_injection.py
pytest src/tests/test_jailbreak.py
pytest src/tests/test_pii.py

# Run with coverage
pytest --cov=src --cov-report=html
```

## Demo

```bash
# Run interactive demo
python -m src.demo.cli_demo

# Run API server with demo endpoints
uvicorn src.demo.api_demo:app --reload
```

## Configuration

See `.env.example` for all configuration options including:
- Detection thresholds
- Model selection
- Feature flags
- Rate limiting
- Security settings

## Project Structure

```
aiguard/
├── src/
│   ├── guardrails/           # Detection modules
│   │   ├── prompt_injection/
│   │   ├── jailbreak/
│   │   ├── pii/
│   │   ├── encoding/
│   │   └── output_filter/
│   ├── middleware/           # FastAPI integration
│   ├── tests/               # Adversarial test cases
│   └── demo/                # Interactive demos
├── config/                  # Configuration files
├── logs/                    # Application logs
└── pyproject.toml
```

## License

MIT License - see LICENSE file for details
