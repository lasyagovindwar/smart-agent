# full_agent.py
import os, re, json, datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# -------------------------
# LLM Client (Gemini-only for your setup)
# -------------------------
class LLMClient:
    """
    Minimal Gemini wrapper with:
      - ask(prompt: str) -> str
      - chat(messages: List[Dict[role, content]]) -> str
    Looks for GOOGLE_API_KEY and GEMINI_MODEL in env.
    """
    def __init__(self):
        self.enabled = True
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY.")
            genai.configure(api_key=api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print("Gemini init error:", e)
            self.enabled = False
            self._genai = None
            self._model = None

    def ask(self, prompt: str) -> str:
        if not self.enabled or not self._model:
            return "[LLM unavailable]"
        try:
            resp = self._model.generate_content(prompt)
            # Prefer .text; fallback to concatenating parts if needed
            if getattr(resp, "text", None):
                return resp.text.strip()
            if getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
                return "".join(getattr(p, "text", "") for p in parts).strip()
            return str(resp)
        except Exception as e:
            return f"[LLM error: {e}]"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Compose a single prompt from {role, content} messages.
        (Gemini chat sessions are optional; this keeps it simple)
        """
        sys_msgs = [m["content"] for m in messages if m.get("role") == "system"]
        user_msgs = [m["content"] for m in messages if m.get("role") != "system"]
        sys_block = "\n".join(f"[SYSTEM] {m}" for m in sys_msgs)
        user_block = "\n".join(f"[MESSAGE] {m}" for m in user_msgs)
        prompt = (sys_block + "\n" + user_block).strip()
        return self.ask(prompt)

# -------------------------
# Tools
# -------------------------
from calculator_tool import CalculatorTool
from translator_tool import TranslatorTool

# -------------------------
# Memory
# -------------------------
class Memory:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def as_messages(self) -> List[Dict[str, str]]:
        return self.history[-8:]

    def add_turn(self, user: str, assistant: str):
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})

# -------------------------
# Planner (step extractor)
# -------------------------
class Planner:
    """
    Extracts executable steps from a free-form query.
    Supports:
      - translate "<text>" to/into <lang[, lang2 ...]>
      - solve <expression>   OR   what is <expression>
      - add A and B / A + B
      - subtract A and B / A - B
      - multiply A and B / A * B
      - divide A and B / A / B
      - capital of <country>
      - fallback to ask LLM
    Multi-steps split by 'then', 'and then', 'and also', 'also', ';'
    """
    SPLIT_RE = re.compile(r"\b(?:then|and then|and also|also)\b|;", flags=re.IGNORECASE)

    def _split_segments(self, query: str) -> List[str]:
        chunks = self.SPLIT_RE.split(query.strip())
        out = []
        for c in chunks:
            parts = [p.strip() for p in re.split(r"[.]", c) if p.strip()]
            out.extend(parts)
        return [s for s in out if s]

    # ---- Parsers ----
    def _parse_translate(self, text: str):
        # translate 'hello' into telugu,hindi and german
        m = re.search(
            r"translate\s+['\"]?(.+?)['\"]?\s+(?:into|to)\s+([a-zA-Z ,\-]+)$",
            text, re.IGNORECASE)
        if m:
            raw_text = m.group(1).strip()
            langs = [s.strip() for s in re.split(r",| and ", m.group(2)) if s.strip()]
            return {"type": "translate", "text": raw_text, "to": langs}
        # translate 'hello'
        m2 = re.search(r"translate\s+['\"](.+?)['\"]\s*$", text, re.IGNORECASE)
        if m2:
            return {"type": "translate", "text": m2.group(1).strip(), "to": ["German"]}
        return None

    def _parse_calc_expr(self, text: str):
        # solve 2*3+7-9
        m = re.search(r"^\s*solve\s+([0-9\.\+\-\*/%\^\(\)\s]+)\s*$", text, re.IGNORECASE)
        if m:
            return {"type": "calc_expr", "expr": m.group(1).strip()}
        # what is 2*3+7-9
        m2 = re.search(r"^\s*what\s+is\s+([0-9\.\+\-\*/%\^\(\)\s]+)\s*\?*$", text, re.IGNORECASE)
        if m2:
            return {"type": "calc_expr", "expr": m2.group(1).strip()}
        return None

    def _parse_add(self, text: str):
        m = re.search(r"\badd\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if m: return {"type": "add", "a": float(m.group(1)), "b": float(m.group(2))}
        m2 = re.search(r"\b(-?\d+(?:\.\d+)?)\s*\+\s*(-?\d+(?:\.\d+)?)\b", text)
        if m2: return {"type": "add", "a": float(m2.group(1)), "b": float(m2.group(2))}
        return None

    def _parse_subtract(self, text: str):
        m = re.search(r"\bsubtract\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if m: return {"type": "subtract", "a": float(m.group(1)), "b": float(m.group(2))}
        m2 = re.search(r"\b(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\b", text)
        if m2: return {"type": "subtract", "a": float(m2.group(1)), "b": float(m2.group(2))}
        return None

    def _parse_multiply(self, text: str):
        m = re.search(r"\bmultiply\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if m: return {"type": "multiply", "a": float(m.group(1)), "b": float(m.group(2))}
        m2 = re.search(r"\b(-?\d+(?:\.\d+)?)\s*\*\s*(-?\d+(?:\.\d+)?)\b", text)
        if m2: return {"type": "multiply", "a": float(m2.group(1)), "b": float(m2.group(2))}
        return None

    def _parse_divide(self, text: str):
        m = re.search(r"\bdivide\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if m: return {"type": "divide", "a": float(m.group(1)), "b": float(m.group(2))}
        m2 = re.search(r"\b(-?\d+(?:\.\d+)?)\s*\/\s*(-?\d+(?:\.\d+)?)\b", text)
        if m2: return {"type": "divide", "a": float(m2.group(1)), "b": float(m2.group(2))}
        return None

    def _parse_capital(self, text: str):
        m = re.search(r"\bcapital of\s+([a-zA-Z \-]+)\b", text, re.IGNORECASE)
        if m:
            return {"type": "capital", "country": m.group(1).strip()}
        return None

    def plan(self, query: str) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for seg in self._split_segments(query):
            s = seg.strip()
            for parser in (
                self._parse_translate,
                self._parse_calc_expr,
                self._parse_add,
                self._parse_subtract,
                self._parse_multiply,
                self._parse_divide,
                self._parse_capital,
            ):
                out = parser(s)
                if out:
                    steps.append(out)
                    break
            else:
                steps.append({"type": "ask", "question": s})
        return steps

# -------------------------
# Agent
# -------------------------
class Agent:
    def __init__(self, llm: LLMClient, memory: Memory, log_path: str = "logs/level3.jsonl"):
        self.llm = llm
        self.memory = memory
        self.log_path = log_path
        self.calculator = CalculatorTool()
        self.translator = TranslatorTool(self.llm)
        self.system_general = "You are a precise, helpful assistant. Be concise and correct."
        self.system_capital = "You answer ONLY with the official capital city. No extra words."

    def _llm_answer(self, question: str, system_prompt: str | None = None) -> str:
        system_msg = {"role": "system", "content": system_prompt or self.system_general}
        msgs = [system_msg] + self.memory.as_messages() + [{"role": "user", "content": question}]
        return self.llm.chat(msgs)

    def execute(self, query: str) -> str:
        planner = Planner()
        plan = planner.plan(query)
        steps_meta, outputs = [], []

        for i, step in enumerate(plan, start=1):
            stype = step["type"]
            meta = {"index": i, "type": stype, "input": step}
            try:
                if stype == "translate":
                    text = step["text"]
                    targets = step.get("to", ["German"])
                    result = self.translator.translate(text, targets)

                elif stype == "calc_expr":
                    expr = step["expr"]
                    result = self.calculator.evaluate(expr)

                elif stype == "add":
                    result = self.calculator.add(step["a"], step["b"])

                elif stype == "subtract":
                    result = self.calculator.subtract(step["a"], step["b"])

                elif stype == "multiply":
                    result = self.calculator.multiply(step["a"], step["b"])

                elif stype == "divide":
                    result = self.calculator.divide(step["a"], step["b"])

                elif stype == "capital":
                    country = step["country"]
                    result = self._llm_answer(
                        f"Provide ONLY the capital city of {country}.",
                        system_prompt=self.system_capital
                    )

                elif stype == "ask":
                    result = self._llm_answer(step["question"], system_prompt=self.system_general)

                else:
                    result = f"[Unknown step type: {stype}]"

            except Exception as e:
                result = f"[Error executing step: {e}]"

            meta["output"] = result
            steps_meta.append(meta)
            outputs.append(str(result))

        final_answer = self._compose(outputs)
        self._log(query=query, plan=plan, steps=steps_meta, final_answer=final_answer)
        self.memory.add_turn(user=query, assistant=final_answer)
        return final_answer

    def _compose(self, outputs: List[str]) -> str:
        return outputs[0] if len(outputs) == 1 else "\n".join(f"{i}) {o}" for i, o in enumerate(outputs, 1))

    def _log(self, query: str, plan: List[Dict[str, Any]], steps: List[Dict[str, Any]], final_answer: str):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": query,
            "plan": plan,
            "steps": steps,
            "final_answer": final_answer,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    llm = LLMClient()
    memory = Memory()
    agent = Agent(llm=llm, memory=memory, log_path="logs/level3.jsonl")
    print("Full Agentic AI (Level 3). Type 'exit' to quit.")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user:
            continue
        answer = agent.execute(user)
        # Printing in UTF-8 context (Windows tip: run `chcp 65001` first)
        try:
            print(answer)
        except UnicodeEncodeError:
            print(answer.encode("utf-8", errors="ignore").decode("utf-8"))

if __name__ == "__main__":
    main()
