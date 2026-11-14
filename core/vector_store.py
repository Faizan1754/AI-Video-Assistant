import hashlib
import re
from functools import lru_cache
from typing import List, Optional, Sequence, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # fall back to the deprecated location
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        raise ImportError(
            "The LangChain packages in this environment are out of sync. This usually "
            "happens after installing a langchain-* package without a pinned version, "
            "which upgrades langchain-core. Fix: pip install -r requirements.txt"
        ) from e

from core.documents import DocPage, pages_to_text
from core.segments import Segment, format_timestamp, segments_to_text

CHROMA_DIR = "vector_db"
COLLECTION_PREFIX = "meetings_transcripts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_CHARS = 500
CHUNK_OVERLAP_CHARS = 50


@lru_cache(maxsize=1)
def get_embedding():
    """Loaded once per process (the model load is slow)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def make_video_id(source: str) -> str:
    """Stable short id from a URL, filename or transcript text."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def _collection_name(video_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]", "", video_id)[:40] or "default"
    return f"{COLLECTION_PREFIX}_{clean}"


def _open_store(video_id: str) -> Chroma:
    return Chroma(
        collection_name=_collection_name(video_id),
        embedding_function=get_embedding(),
        persist_directory=CHROMA_DIR,
    )


def chunk_segments(
    segments: Sequence[Segment],
    max_chars: int = CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> List[List[Segment]]:
    """
    Group whole transcript segments into chunks of roughly max_chars, carrying a
    little overlap into the next chunk. Chunks therefore always start and end on
    segment boundaries, so each one has an exact start/end time.
    """
    chunks: List[List[Segment]] = []
    current: List[Segment] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg.text) + 1
        if current and current_len + seg_len > max_chars:
            chunks.append(current)

            carry: List[Segment] = []
            carry_len = 0
            for prev in reversed(current):
                prev_len = len(prev.text) + 1
                if carry_len + prev_len > max_chars // 2:
                    break
                carry.insert(0, prev)
                carry_len += prev_len
                if carry_len >= overlap_chars:
                    break

            current, current_len = carry, carry_len

        current.append(seg)
        current_len += seg_len

    if current:
        chunks.append(current)
    return chunks


def _docs_from_segments(segments: Sequence[Segment], video_id: str) -> List[Document]:
    docs = []
    for i, group in enumerate(chunk_segments(segments)):
        start, end = group[0].start, group[-1].end
        docs.append(
            Document(
                page_content=segments_to_text(group),
                metadata={
                    "chunk_index": i,
                    "video_id": video_id,
                    "source_type": "video",
                    "start": float(start),
                    "end": float(end),
                    "timestamp": f"{format_timestamp(start)} - {format_timestamp(end)}",
                },
            )
        )
    return docs


def _docs_from_text(text: str, video_id: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS
    )
    return [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i, "video_id": video_id, "source_type": "transcript"},
        )
        for i, chunk in enumerate(splitter.split_text(text))
    ]


def _docs_from_pages(pages: Sequence[DocPage], video_id: str) -> List[Document]:
    """One or more chunks per page/slide; a chunk never spans two pages, so every
    citation points at an exact page or slide."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS
    )
    docs = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "video_id": video_id,
                        "source_type": page.kind,
                        "page": page.number,
                        "label": page.label,
                    },
                )
            )
    return docs


def build_vector_store(
    transcript: Union[None, str, Sequence[Segment]] = None,
    video_id: Optional[str] = None,
    pages: Optional[Sequence[DocPage]] = None,
) -> Chroma:
    """
    Build (or rebuild) the vector store for ONE study session.

    - transcript: a plain string, or a list of Segment objects (timestamps are
      stored as chunk metadata). May be None when only documents are given.
    - pages: pages/slides from core.documents.load_document (PDF / PPTX). They are
      indexed together with the transcript, so one question can be answered from
      the video and the slides.
    - video_id: use make_video_id(url_or_filename) so you can reload it later.
      If omitted, it is derived from the content.

    Each session gets its own Chroma collection, so answers never mix content
    from different sessions, and rebuilding does not duplicate chunks.
    """
    print("Building vector store...")

    plain_text = ""
    if isinstance(transcript, str):
        plain_text = transcript
    elif transcript is not None:
        transcript = list(transcript)
        plain_text = segments_to_text(transcript)

    pages = list(pages or [])
    doc_text = pages_to_text(pages)

    if not plain_text.strip() and not doc_text.strip():
        raise ValueError("Transcript and documents are empty - nothing to index.")

    video_id = video_id or make_video_id(plain_text + "\n" + doc_text)

    docs: List[Document] = []
    if plain_text.strip():
        if isinstance(transcript, str):
            docs.extend(_docs_from_text(transcript, video_id))
        else:
            docs.extend(_docs_from_segments(transcript, video_id))
    docs.extend(_docs_from_pages(pages, video_id))

    # unique, stable index across transcript chunks and document chunks
    for i, d in enumerate(docs):
        d.metadata["chunk_index"] = i

    # Drop any previous index for this id, then create it fresh.
    try:
        _open_store(video_id).delete_collection()
    except Exception:
        pass

    store = _open_store(video_id)
    store.add_documents(docs, ids=[f"{video_id}-{d.metadata['chunk_index']}" for d in docs])
    return store


def load_vector_store(video_id: str) -> Chroma:
    """Reopen the index previously built for this video_id."""
    return _open_store(video_id)


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )