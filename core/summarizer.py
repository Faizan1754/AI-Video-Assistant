import os
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm

# How many chunk summaries to request in parallel. Default 1 (one at a time) keeps
# you inside free-tier rate limits; raise SUMMARY_CONCURRENCY on a paid plan.
def _map_concurrency() -> int:
    return max(1, int(os.getenv("SUMMARY_CONCURRENCY", "1")))


# Wording per kind of input. The "meeting" texts are the original prompts, unchanged.
_WORDING = {
    "meeting": {
        "part": "meeting transcript",
        "role": "meeting summarizer",
        "result": "meeting summary",
        "title_source": "the meeting transcript",
        "title_kind": "meeting title",
    },
    "document": {
        "part": "document (class slides or notes)",
        "role": "summarizer of academic and lecture material",
        "result": "summary of the document",
        "title_source": "the document",
        "title_kind": "title",
    },
}


def _wording(source_kind: str) -> dict:
    return _WORDING.get(source_kind, _WORDING["meeting"])


def split_transcript(transcript: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str, source_kind: str = "meeting") -> str:
    """source_kind: "meeting" (default) or "document" (slides / PDF text)."""
    if not transcript or not transcript.strip():
        return "No transcript content to summarize."

    w = _wording(source_kind)

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"Summarize this portion of a {w['part']} concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # Map step: all chunks summarized in parallel (order is preserved).
    chunk_summaries = map_chain.batch(
        [{"text": chunk} for chunk in chunks],
        config={"max_concurrency": _map_concurrency()},
    )

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are an expert {w['role']}. Combine these partial summaries "
                f"into one final professional {w['result']} in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str, source_kind: str = "meeting") -> str:
    w = _wording(source_kind)
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Based on {w['title_source']}, generate a short professional {w['title_kind']} "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({"text": transcript[:2000]})