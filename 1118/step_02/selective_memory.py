"""Checkpoint 5: put only matching saved facts into the model request."""

import re


def select_saved_facts(prompt, facts):
    words = set(re.findall(r"\b\w+\b", prompt.casefold()))
    selected = {}

    if "framework" in words and "project_framework" in facts:
        selected["project_framework"] = facts["project_framework"]

    if "style" in words and words & {"answer", "response", "preferred"} and "answer_style" in facts:
        selected["answer_style"] = facts["answer_style"]

    return selected

