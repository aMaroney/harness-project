"""
MINIMAL local-first LLM harness.

See README.md for full setup instructions.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from a local .env file

LOCAL_MODEL = "qwen3.5:9b"
CLOUD_MODEL = os.environ.get("CLOUD_MODEL", "no model specified")

local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
cloud_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "missing-key"),
)

# ---------------------------------------------------------------------------
# Filesystem tools — the model can request these, your code executes them.
# The model NEVER touches your disk directly; it can only ask, and this
# code decides whether/how to actually do it.
# ---------------------------------------------------------------------------


def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()[
                :4000
            ]  # cap size so a huge file doesn't blow the context window
    except Exception as e:  # noqa: BLE001 — intentionally broad
        return f"error reading file: {e}"


def list_directory(path: str = ".") -> str:
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:  # noqa: BLE001 — intentionally broad
        return f"error listing directory: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a local directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path, defaults to current dir",
                    }
                },
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {"read_file": read_file, "list_directory": list_directory}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(PROJECT_DIR, ".harness_history.json")


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


conversation_history = (
    load_history()
)  # loaded once at startup — this is last session's memory


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


def ask(prompt: str, private_chat: bool, use_cloud: bool) -> tuple[str, str]:
    """Returns (answer_text, which_tier_answered). Handles tool calls in a loop:
    if the model asks to read a file or list a directory, we run it locally
    and hand the result back, until the model gives a final text answer."""

    client, model, tier = (
        (cloud_client, CLOUD_MODEL, "cloud")
        if looks_complex(prompt) or use_cloud
        else (local_client, LOCAL_MODEL, "local")
    )

    messages = conversation_history + [{"role": "user", "content": prompt}]

    try:
        for _ in range(5):  # safety cap so a bad tool call can't loop forever
            response = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                if not private_chat:
                    # Persist only the clean user/assistant exchange — not the tool-call
                    # plumbing, which is single-turn scaffolding, not memory worth keeping.
                    conversation_history.append({"role": "user", "content": prompt})
                    conversation_history.append(
                        {"role": "assistant", "content": msg.content}
                    )
                    save_history(conversation_history)
                return msg.content, tier

            messages.append(msg)
            for call in msg.tool_calls:
                func = AVAILABLE_FUNCTIONS[call.function.name]
                args = json.loads(call.function.arguments or "{}")
                result = func(**args)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        return "gave up after too many tool calls", tier

    except Exception as e:
        if tier == "local":
            print(f"(local failed: {e} — falling back to cloud)")
            fallback_messages = conversation_history + [
                {"role": "user", "content": prompt}
            ]
            response = cloud_client.chat.completions.create(
                model=CLOUD_MODEL, messages=fallback_messages
            )
            answer = response.choices[0].message.content
            if not private_chat:
                conversation_history.append({"role": "user", "content": prompt})
                conversation_history.append({"role": "assistant", "content": answer})
                save_history(conversation_history)
            return answer, "cloud"
        raise


def main():
    print(
        "Harness ready. Type a question, 'reset' to clear memory, or 'quit' to exit.\n"
    )
    while True:
        private_chat = False
        use_cloud = False
        question = input("> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if question.lower() == "reset":
            conversation_history.clear()
            save_history(conversation_history)
            print("Memory cleared.\n")
            continue
        if "--no-save" in question:
            question = question.replace("--no-save", "")
            private_chat = True
        if "--use-cloud" in question:
            question = question.replace("--use-cloud", "")
            use_cloud = True
        if not question:
            continue

        answer, tier = ask(question, private_chat, use_cloud)
        print(f"\n[answered by: {tier}]")
        print(answer)
        print()


if __name__ == "__main__":
    main()
