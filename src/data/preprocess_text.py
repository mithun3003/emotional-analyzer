from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Preprocess and clean raw transcript utterance text."""
    if not isinstance(text, str):
        return ""
    # Normalize whitespaces and quotation marks
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text
