"""Highlight source text that also appears in a cited answer."""

from difflib import SequenceMatcher
from html import escape
import re

COLORS = ("#dbeafe", "#dcfce7", "#ffedd5")
CITATIONS = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
WORDS = re.compile(r"\w+", re.UNICODE)


def source_style(number):
    color = COLORS[(number - 1) % len(COLORS)]
    return f"background:{color};color:#182d37;border-radius:4px;padding:2px 4px"


def highlighted_passage(passage, answer, source_number):
    """Mark substantial word matches; keep the original source wording intact."""
    source_words = list(WORDS.finditer(passage))
    source_tokens = [word.group().casefold() for word in source_words]
    spans = []
    for sentence in re.split(r'(?<=[.!?。！？])\s+(?=[A-Z가-힣"“])|\n', answer):
        citations = {
            int(number)
            for match in CITATIONS.finditer(sentence)
            for number in re.findall(r"\d+", match.group())
        }
        if source_number not in citations:
            continue
        answer_tokens = WORDS.findall(CITATIONS.sub("", sentence).casefold())
        for block in SequenceMatcher(None, answer_tokens, source_tokens, autojunk=False).get_matching_blocks():
            if block.size < 4:
                continue
            start = source_words[block.b].start()
            end = source_words[block.b + block.size - 1].end()
            if end - start >= 20:
                spans.append((start, end))

    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    pieces = []
    cursor = 0
    for start, end in merged:
        pieces.append(escape(passage[cursor:start]))
        pieces.append(
            f'<mark style="{source_style(source_number)}">'
            f'{escape(passage[start:end])}</mark>'
        )
        cursor = end
    pieces.append(escape(passage[cursor:]))
    return "".join(pieces), bool(merged)
