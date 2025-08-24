# chatbot.py
"""
Level 1: LLM-Only Smart Assistant
--------------------------------
- Answers user questions step-by-step using LLM.
- Refuses to perform math calculations (hints calculator tool).
- Logs interactions into logs/level1.txt
"""

import os
import time
from pathlib import Path
import google.generativeai as genai

# ========= Config =========
LOG_FILE = Path("logs/level1.txt")
MODEL_NAME = "gemini-1.5-pro"

# ========= Setup =========
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class LLMClient:
    """Wrapper around Gemini API with retry handling."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = genai.GenerativeModel(model_name)

    def ask(self, prompt: str) -> str:
        """Ask the LLM with exponential backoff retry on 429 errors."""
        for attempt in range(5):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e):  # quota or rate limit
                    wait_time = (2 ** attempt)
                    print(f"[Rate Limit] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        return "⚠️ Failed after multiple retries."


def build_prompt(user_input: str) -> str:
    """Prompt engineering to force step-by-step and refusal for math."""
    math_keywords = ["+", "-", "*", "/", "add", "subtract", "multiply", "divide", "times"]
    if any(k in user_input.lower() for k in math_keywords):
        return (
            f"The user asked: '{user_input}'.\n"
            "⚠️ IMPORTANT: You are NOT allowed to solve math problems.\n"
            "Instead, politely refuse and hint at using a calculator tool.\n"
            "Example: 'I cannot solve math directly. Please use the calculator tool.'"
        )
    else:
        return (
            f"The user asked: '{user_input}'.\n"
            "You must always answer step-by-step, structured clearly like a teacher.\n"
            "Make sure the explanation is simple and easy to follow."
        )


def log_interaction(user_input: str, response: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\nBot: {response}\n{'-'*40}\n")


def main():
    print("🤖 LLM Smart Assistant (Level 1)")
    print("Type 'exit' to quit.\n")

    client = LLMClient()

    while True:
        user_input = input("> ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        prompt = build_prompt(user_input)
        response = client.ask(prompt)

        print(response)
        log_interaction(user_input, response)


if __name__ == "__main__":
    main()
