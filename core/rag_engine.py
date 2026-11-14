from typing import Optional, Sequence, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from core.documents import DocPage
from core.llm import get_llm
from core.segments import Segment, format_timestamp
from core.vector_store import build_vector_store, get_retriever, load_vector_store

SYSTEM_PROMPT = """You are an expert meeting and lecture assistant. Answer the user's
question based ONLY on the context provided below. The context comes from a meeting
or lecture transcript and/or from documents such as slides or PDFs.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.
When a context passage starts with a location such as [02:10 - 02:45] or
[Slide 3 (lecture.pptx)], cite the location of the passage(s) you used at the end
of the relevant sentence.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _location_label(doc) -> str:
    """'[02:10 - 02:45]' for video passages, '[Slide 3 (file.pptx)]' for documents."""
    meta = doc.metadata or {}
    if "start" in meta and "end" in meta:
        return f"[{format_timestamp(meta['start'])} - {format_timestamp(meta['end'])}]"
    if meta.get("label"):
        return f"[{meta['label']}]"
    return ""


def format_docs(docs) -> str:
    return "\n\n".join(f"{_location_label(d)} {d.page_content}".strip() for d in docs)


def _sources_from_docs(docs) -> list:
    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append(
            {
                "timestamp": meta.get("timestamp"),
                "label": meta.get("label"),
                "source_type": meta.get("source_type"),
                "start": meta.get("start"),
                "end": meta.get("end"),
                "text": d.page_content,
            }
        )
    return sources


def _make_chain(retriever, with_sources: bool):
    llm = get_llm()

    if not with_sources:
        # invoke(question) -> str   (same interface as before)
        return (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )

    # invoke(question) -> {"answer": str, "sources": [ {timestamp,start,end,text}, ... ]}
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(answer=answer_chain)
        | RunnableLambda(
            lambda x: {"answer": x["answer"], "sources": _sources_from_docs(x["docs"])}
        )
    )


def build_rag_chain(
    transcripts: Union[None, str, Sequence[Segment]] = None,
    video_id: Optional[str] = None,
    with_sources: bool = False,
    k: int = 4,
    pages: Optional[Sequence[DocPage]] = None,
):
    """
    Index the transcript (a string, or a list of Segment objects to get
    timestamps) and/or document pages (PDF / PPTX) and return a
    question-answering chain over all of it.
    """
    vector_store = build_vector_store(transcripts, video_id=video_id, pages=pages)
    retriever = get_retriever(vector_store, k=k)
    return _make_chain(retriever, with_sources)


def load_rag_chain(video_id: str, with_sources: bool = False, k: int = 4):
    """Rebuild the chain from an index that was saved earlier for video_id."""
    vector_store = load_vector_store(video_id)
    retriever = get_retriever(vector_store, k=k)  # previously missing vector_store
    return _make_chain(retriever, with_sources)


def ask_question(rag_chain, question: str):
    print(f"Question: {question}")
    result = rag_chain.invoke(question)
    answer = result["answer"] if isinstance(result, dict) else result
    print(f"Answer: {answer}")
    return result