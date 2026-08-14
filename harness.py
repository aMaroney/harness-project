"""
MINIMAL local-first LLM harness.

See README.md for full setup instructions.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from a local .env file

LOCAL_MODEL = "qwen3.5:9b"
CLOUD_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
cloud_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "missing-key"),
)


# h
def looks_complex(prompt: str) -> bool:
    long_prompt = len(prompt) > 500
    complex_keywords = [
        "refactor",
        "multi-step",
        "step by step",
        "across multiple",
        "compare and contrast",
    ]
    mentions_complex_task = any(word in prompt.lower() for word in complex_keywords)
    return long_prompt or mentions_complex_task


def ask(prompt: str) -> tuple[str, str]:
    """Returns (answer_text, which_tier_answered)."""

    if looks_complex(prompt):
        response = cloud_client.chat.completions.create(
            model=CLOUD_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, "cloud"

    try:
        response = local_client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, "local"
    except Exception as e:
        print(f"(local failed: {e} — falling back to cloud)")
        response = cloud_client.chat.completions.create(
            model=CLOUD_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, "cloud"


def main():
    print("Harness ready. Type a question, or 'quit' to exit.\n")
    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, tier = ask(question)
        print(f"\n[answered by: {tier}]")
        print(answer)
        print()


if __name__ == "__main__":
    main()
