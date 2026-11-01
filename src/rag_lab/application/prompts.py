"""Prompt for grounded answers. Kept in one place so it can be read, versioned and tested.

The prompt is in English (models follow English instructions most reliably) and asks
for the answer in the question's language.
"""

from collections.abc import Sequence

from rag_lab.application.ports.generation import ChatMessage
from rag_lab.domain.answers import ABSTENTION
from rag_lab.domain.chunks import Chunk

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided.

Rules:
1. Support every factual statement with the number of the source(s) it comes from, in \
square brackets right after the statement, e.g. [1] or [2, 3].
2. Use only the sources. Do not use prior knowledge, even if you know the answer.
3. If the sources do not contain the answer, reply with exactly {ABSTENTION} followed by \
one short sentence saying what is missing. Do not guess.
4. Answer in the same language as the question. Be concise."""


def _label(number: int, chunk: Chunk) -> str:
    location = []
    if chunk.page_numbers:
        location.append("page " + ", ".join(str(p) for p in chunk.page_numbers))
    if chunk.section_path:
        location.append(" > ".join(chunk.section_path))
    return f"[{number}]" + (f" ({'; '.join(location)})" if location else "")


def answer_messages(question: str, sources: Sequence[Chunk]) -> list[ChatMessage]:
    """System rules plus a user message with numbered sources and the question."""
    blocks = [f"{_label(n, chunk)}\n{chunk.text}" for n, chunk in enumerate(sources, start=1)]
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]


SKIP = "SKIP"

QUESTION_PROMPT = f"""You write evaluation questions for a document search engine.

Given a passage, write ONE question that the passage answers.

Rules:
1. The passage alone must be enough to answer it.
2. Be specific: ask about names, numbers, conditions or steps stated in the passage.
3. Ask it the way a real user would. Do not copy long phrases from the passage; \
paraphrase, so the question does not simply repeat its words.
4. Write the question in the same language as the passage.
5. If the passage has no factual content (table of contents, boilerplate, a lone \
heading), reply exactly {SKIP}.

Reply with the question only."""


def question_messages(passage: str) -> list[ChatMessage]:
    return [ChatMessage("system", QUESTION_PROMPT), ChatMessage("user", f"Passage:\n{passage}")]


SUPPORTED, UNSUPPORTED = "SUPPORTED", "UNSUPPORTED"

JUDGE_PROMPT = f"""You check whether an answer is supported by the sources it was written from.

Reply with exactly one word first, {SUPPORTED} or {UNSUPPORTED}, then one short sentence.

{SUPPORTED}: every factual statement in the answer is stated or directly implied by the \
numbered sources.
{UNSUPPORTED}: at least one statement is missing from, or contradicts, the sources."""


def judge_messages(question: str, answer: str, sources: Sequence[Chunk]) -> list[ChatMessage]:
    blocks = [f"[{n}]\n{chunk.text}" for n, chunk in enumerate(sources, start=1)]
    user = (
        "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\n"
        f"Answer to check:\n{answer}"
    )
    return [ChatMessage("system", JUDGE_PROMPT), ChatMessage("user", user)]


def parse_verdict(reply: str) -> bool | None:
    """True for SUPPORTED, False for UNSUPPORTED, None when the reply is neither."""
    first = reply.strip().lstrip("*_#>`\"' ").upper()
    if first.startswith(UNSUPPORTED):
        return False
    if first.startswith(SUPPORTED):
        return True
    return None
