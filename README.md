# Local First LLM Harness

A minimal routing harness that tries a local model (via [Ollama](https://ollama.com))
first, and falls back to a free cloud model (via [OpenRouter](https://openrouter.ai))
when the prompt looks complex or the local model isn't available.

Built as a learning project to understand local vs cloud LLM routing patterns
applicable to data engineering workflows (log triage, schema docs, SQL review, etc.).

## Setup

```bash
# 1. Install Ollama and pull a local model
brew install ollama
ollama serve &
ollama pull qwen3.5:9b

# 2. Install Python dependencies
pip install  r requirements.txt

# 3. Set your OpenRouter API key
cp .env.example .env
# then edit .env and paste in your real key from openrouter.ai
```

## Run

```bash
python harness.py
```

Type a question at the prompt. The harness prints which tier (local/cloud)
answered.

## How it works

Both Ollama and OpenRouter expose an OpenAI compatible API, so the same
`openai` Python client works against either — only the `base_url` and
`api_key` change. Routing logic (`looks_complex()`) decides which one to
call for a given prompt.
