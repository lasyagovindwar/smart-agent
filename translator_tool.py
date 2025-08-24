# translator_tool.py
from typing import List, Union

class TranslatorTool:
    """
    Multi-language translator tool using your LLM client.
    - Supports single or multiple target languages in one call.
    - Includes aliases and a wide set of Indian + common global languages.
    - Expects an LLM client with an .ask(prompt: str) -> str method.
    """

    SUPPORTED_LANGS = {
        # Indian languages
        "hindi": "Hindi",
        "telugu": "Telugu",
        "tamil": "Tamil",
        "kannada": "Kannada",
        "malayalam": "Malayalam",
        "marathi": "Marathi",
        "gujarati": "Gujarati",
        "punjabi": "Punjabi",
        "bengali": "Bengali",
        "odia": "Odia",
        # Common global
        "german": "German",
        "french": "French",
        "spanish": "Spanish",
        "italian": "Italian",
        "russian": "Russian",
        "arabic": "Arabic",
        "japanese": "Japanese",
        "korean": "Korean",
        "chinese": "Chinese",
        "portuguese": "Portuguese",
    }

    ALIASES = {
        "bangla": "bengali",
        "kanada": "kannada",
        "kanadian": "kannada",
        "malayanam": "malayalam",
        "oriya": "odia",
        "mandarin": "chinese",
    }

    def __init__(self, llm_client):
        """
        llm_client: an object exposing llm_client.ask(prompt: str) -> str
        """
        self.llm = llm_client

    def _normalize_lang(self, name: str) -> str:
        key = (name or "").strip().lower()
        key = self.ALIASES.get(key, key)
        return self.SUPPORTED_LANGS.get(key, name.strip().title())

    def _call(self, prompt: str) -> str:
        # Single adapter point so you can swap LLMs if needed
        return (self.llm.ask(prompt) or "").strip()

    def translate_multi(self, text: str, to_langs: List[str]) -> dict:
        """
        Translate 'text' to multiple languages.
        Returns a dict {DisplayLanguage: Translation}
        """
        out = {}
        for raw in to_langs:
            display = self._normalize_lang(raw)
            prompt = (
                f"Translate the following text into {display}. "
                f"Return ONLY the translation with correct script.\n\n"
                f"Text: {text}"
            )
            out[display] = self._call(prompt)
        return out

    def translate(self, text: str, to: Union[str, List[str]]) -> str:
        """
        Translate to one or many languages.
        - If string: return translation string.
        - If list/CSV-like: return pretty multiline string with each translation.
        """
        targets = to if isinstance(to, list) else [s.strip() for s in str(to).split(",") if s.strip()]
        results = self.translate_multi(text, targets)
        if len(results) == 1:
            return next(iter(results.values()))
        return "\n".join(f"{lang}: {val}" for lang, val in results.items())
