"""Checkpoint 5: put only matching saved facts into the model request."""

import re
from pathlib import Path

from step_01 import run_app
from step_03 import persistent_chat
from step_04 import show_saved_memories


def select_saved_facts(prompt, facts):
    words = set(re.findall(r"\b\w+\b", prompt.casefold()))
    selected = {}

    if "framework" in words and "project_framework" in facts:
        selected["project_framework"] = facts["project_framework"]

    if "style" in words and words & {"answer", "response", "preferred"} and "answer_style" in facts:
        selected["answer_style"] = facts["answer_style"]

    return selected


def step_05_chat():
    persistent_chat(
        Path(__file__).with_name("step_05_memory.json"),
        memory_panel=show_saved_memories,
        select_facts=select_saved_facts,
    )


if __name__ == "__main__":
    run_app(5, "Recall relevant saved facts and inspect what each request sent to Ollama.", step_05_chat)
