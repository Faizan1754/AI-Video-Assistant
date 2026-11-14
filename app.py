# app.py
"""AI Meeting Assistant — Streamlit front end.

Analyzes YouTube videos and local recordings: transcription, translation,
summarization, and RAG-based Q&A over the transcript.
"""

import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from main import run_pipeline
from core.rag_engine import ask_question
from core.segments import segments_to_timestamped_text, youtube_link

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

APP_TITLE = "AI Meeting Assistant"
APP_ICON = "🎙️"
MAX_UPLOAD_MB = 500
YOUTUBE_URL_PATTERN = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+"
)
SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv", "m4a", "mp3", "wav"]
DOC_FORMATS = ["pdf", "pptx"]
DOC_ONLY = "Slides / PDF only"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1200px; }

        /* Header */
        .app-header { display: flex; align-items: center; gap: .6rem; margin-bottom: 0; }
        .app-subtitle { color: var(--text-color, #8a8f98); font-size: 0.95rem; margin-top: -0.4rem; }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(127, 127, 127, 0.06);
            border: 1px solid rgba(127, 127, 127, 0.15);
            border-radius: 10px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }

        /* Buttons */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 8px;
            font-weight: 500;
        }

        /* Chat */
        .stChatMessage { border-radius: 10px; }

        /* Sidebar section labels */
        .sidebar-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(127,127,127,0.8);
            margin-top: 1.2rem;
            margin-bottom: 0.3rem;
        }

        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


@dataclass
class SessionState:
    result: dict | None = None
    chat_history: list[dict] = field(default_factory=list)  # {"role", "content", "sources"}
    temp_path: str | None = None
    source_url: str | None = None
    processing: bool = False


def get_state() -> SessionState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = SessionState()
    return st.session_state.app_state


state = get_state()


def render_sources(sources: list[dict] | None) -> None:
    """Show the transcript passages an answer was based on, with jump links."""
    if not sources:
        return
    can_link = bool(state.source_url and YOUTUBE_URL_PATTERN.match(state.source_url))
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            stamp = src.get("timestamp") or src.get("label")
            if stamp and can_link and src.get("start") is not None:
                st.markdown(f"**[{stamp}]({youtube_link(state.source_url, src['start'])})**")
            else:
                st.markdown(f"**{stamp or 'Passage'}**")
            st.caption(src.get("text", ""))

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown(
    f'<div class="app-header"><h1>{APP_ICON} {APP_TITLE}</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">Transcribe, translate, and query meeting recordings '
    "or YouTube videos with local speech-to-text and RAG-powered chat.</p>",
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.caption("Local-first meeting analysis")

    st.markdown('<p class="sidebar-label">Source</p>', unsafe_allow_html=True)
    input_type = st.radio(
        "Input source", ["YouTube URL", "Upload File", DOC_ONLY],
        label_visibility="collapsed",
    )

    source = None
    uploaded_file = None
    doc_files = []

    if input_type == "YouTube URL":
        url = st.text_input(
            "YouTube URL", placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        if url and not YOUTUBE_URL_PATTERN.match(url.strip()):
            st.caption("⚠️ That doesn't look like a valid YouTube URL.")
        source = url.strip() if url else None
    elif input_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a recording",
            type=SUPPORTED_FORMATS,
            label_visibility="collapsed",
            help=f"Max {MAX_UPLOAD_MB} MB. Video or audio files.",
        )
        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.error(f"File is {size_mb:.0f} MB — exceeds {MAX_UPLOAD_MB} MB limit.")
                uploaded_file = None
    else:
        doc_files = st.file_uploader(
            "Upload slides or PDFs",
            type=DOC_FORMATS,
            accept_multiple_files=True,
            key="doc_only_files",
            label_visibility="collapsed",
            help="PowerPoint (.pptx) or PDF. Old .ppt files: save as .pptx first.",
        )

    if input_type != DOC_ONLY:
        st.markdown(
            '<p class="sidebar-label">Class material (optional)</p>',
            unsafe_allow_html=True,
        )
        doc_files = st.file_uploader(
            "Attach slides or PDFs",
            type=DOC_FORMATS,
            accept_multiple_files=True,
            key="attached_docs",
            label_visibility="collapsed",
            help="Slides/notes are searched together with the recording in chat.",
        )

    valid_docs = []
    for doc in doc_files or []:
        if doc.size / (1024 * 1024) > MAX_UPLOAD_MB:
            st.error(f"{doc.name} exceeds the {MAX_UPLOAD_MB} MB limit.")
        else:
            valid_docs.append(doc)
    doc_files = valid_docs

    st.markdown('<p class="sidebar-label">Options</p>', unsafe_allow_html=True)
    language = st.selectbox("Output language", ["english", "hinglish"])

    st.write("")
    analyze = st.button(
        "🚀 Analyze", use_container_width=True, type="primary",
        disabled=state.processing,
    )

    st.markdown('<p class="sidebar-label">Session</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear chat", use_container_width=True):
            state.chat_history = []
            st.rerun()
    with col_b:
        if st.button("♻️ Reset", use_container_width=True):
            state.result = None
            state.chat_history = []
            st.rerun()

    with st.expander("ℹ️ About this app"):
        st.markdown(
            "- **Transcription:** faster-whisper (local)\n"
            "- **Translation:** Hindi → English\n"
            "- **Summarization:** LangChain + your chosen LLM (Gemini, Groq, Mistral or local Ollama)\n"
            "- **RAG:** ChromaDB + HuggingFace embeddings\n"
            "- **Documents:** PDF and PowerPoint text extraction\n"
        )

# --------------------------------------------------------------------------- #
# Pipeline execution
# --------------------------------------------------------------------------- #

if analyze:
    if not source and not uploaded_file and not doc_files:
        st.warning("Please provide a YouTube URL, upload a recording, or upload slides/PDFs.")
        st.stop()

    temp_path = None
    temp_dir = None
    try:
        state.processing = True
        # Remember the original URL (for timestamp links) before `source` may
        # be replaced by a temp file path.
        state.source_url = source if (source and not uploaded_file) else None

        doc_paths = []
        if uploaded_file or doc_files:
            # Private temp dir + basename only: no name collisions between
            # sessions and no path tricks from the uploaded file names.
            temp_dir = tempfile.mkdtemp()

        if uploaded_file:
            temp_path = os.path.join(temp_dir, Path(uploaded_file.name).name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = temp_path

        for i, doc in enumerate(doc_files):
            doc_dir = os.path.join(temp_dir, f"doc{i}")  # same-named files can't clash
            os.makedirs(doc_dir)
            doc_path = os.path.join(doc_dir, Path(doc.name).name)
            with open(doc_path, "wb") as f:
                f.write(doc.getbuffer())
            doc_paths.append(doc_path)

        progress = st.progress(0, text="Starting analysis...")

        # If run_pipeline supports a progress callback, wire it up here.
        # Otherwise this gives coarse-grained feedback around the call.
        progress.progress(15, text="Downloading / loading source...")
        result = run_pipeline(source, language, documents=doc_paths)
        progress.progress(100, text="Done.")
        progress.empty()

        state.result = result
        state.chat_history = []  # a new recording starts a fresh conversation
        st.toast("Analysis complete", icon="✅")

    except Exception as e:
        logger.exception("Pipeline failed")
        st.error(f"Analysis failed: {e}")

    finally:
        state.processing = False
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

if state.result:
    result = state.result

    st.subheader(result.get("title", "Untitled"))
    st.write("")

    mode = result.get("mode", "video")
    is_doc_only = mode == "document"
    pages = result.get("pages") or []  # DocPage list (PDF/PPTX), may be empty
    pages_with_labels = "\n\n".join(f"[{p.label}]\n{p.text}" for p in pages)

    transcript = result.get("transcript", "")
    segments = result.get("segments")  # list[Segment] when main.py provides them
    transcript_with_times = (
        segments_to_timestamped_text(segments) if segments else transcript
    )
    if is_doc_only and pages:
        transcript_with_times = pages_with_labels
    summary = result.get("summary", "")
    actions = result.get("action_items", "")
    decisions = result.get("key_decisions", "")
    questions = result.get("open_questions", "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Document text" if is_doc_only else "Transcript",
              f"{len(transcript.split()):,} words")
    c2.metric("Summary", f"{len(summary.split()):,} words")
    if is_doc_only:
        c3.metric("Pages / slides", len(pages))
        c4.metric("Files", len({p.source_name for p in pages}))
    else:
        c3.metric("Action items", len([x for x in actions.splitlines() if x.strip()]))
        c4.metric("Open questions", len([x for x in questions.splitlines() if x.strip()]))

    na_note = "_Not applicable for slides/PDFs - this applies to meeting recordings._"
    actions_md = na_note if is_doc_only else (actions or "_No action items detected._")
    decisions_md = na_note if is_doc_only else (decisions or "_No decisions detected._")
    questions_md = na_note if is_doc_only else (questions or "_No open questions detected._")

    st.write("")

    tabs = st.tabs(
        ["📄 Summary", "📝 Document text" if is_doc_only else "📝 Transcript",
         "✅ Actions", "🎯 Decisions",
         "❓ Questions", "💬 Chat", "⬇️ Export"]
    )

    with tabs[0]:
        st.markdown(summary or "_No summary generated._")

    with tabs[1]:
        shown = transcript_with_times if is_doc_only else transcript
        if segments:
            view = st.radio(
                "Transcript view", ["Plain", "With timestamps"],
                horizontal=True, label_visibility="collapsed",
            )
            if view == "With timestamps":
                shown = transcript_with_times
        st.text_area("Full transcript", shown, height=400, label_visibility="collapsed")

    with tabs[2]:
        st.markdown(actions_md)

    with tabs[3]:
        st.markdown(decisions_md)

    with tabs[4]:
        st.markdown(questions_md)

    with tabs[5]:
        for msg in state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                render_sources(msg.get("sources"))

        prompt = st.chat_input("Ask anything about this recording...")

        if prompt:
            state.chat_history.append({"role": "user", "content": prompt, "sources": []})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    sources = []
                    try:
                        reply = ask_question(result["rag_chain"], prompt)
                        if isinstance(reply, dict):  # chain built with_sources=True
                            answer = reply.get("answer", "")
                            sources = reply.get("sources", [])
                        else:
                            answer = reply
                    except Exception as e:
                        logger.exception("RAG query failed")
                        answer = f"Sorry, I couldn't answer that: {e}"
                st.markdown(answer)
                render_sources(sources)

            state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

    with tabs[6]:
        st.caption("Download individual sections or a combined report.")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "📥 Text (.txt)" if is_doc_only else "📥 Transcript (.txt)",
                transcript_with_times,
                file_name="transcript.txt", mime="text/plain",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "📥 Summary (.txt)", summary,
                file_name="summary.txt", mime="text/plain",
                use_container_width=True,
            )
        with d3:
            combined = (
                f"# {result.get('title', 'Untitled')}\n\n"
                f"## Summary\n{summary}\n\n"
                f"## Action Items\n{actions}\n\n"
                f"## Key Decisions\n{decisions}\n\n"
                f"## Open Questions\n{questions}\n\n"
                f"## Transcript\n{transcript_with_times}\n"
            )
            if is_doc_only:
                combined = (
                    f"# {result.get('title', 'Untitled')}\n\n"
                    f"## Summary\n{summary}\n\n"
                    f"## Document text\n{transcript_with_times}\n"
                )
            elif pages:
                combined += f"\n## Attached documents\n{pages_with_labels}\n"
            st.download_button(
                "📥 Full report (.md)", combined,
                file_name="meeting_report.md", mime="text/markdown",
                use_container_width=True,
            )
else:
    st.info("Paste a YouTube URL, upload a recording, or upload class slides/PDFs in the sidebar to get started.")

st.divider()
st.caption("faster-whisper · LangChain · ChromaDB · LLM API")