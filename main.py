from dotenv import load_dotenv
load_dotenv()  # keep first: load .env before the other imports

from utils.audio_processor import process_input, cleanup_audio, is_url
from core.transcriber import transcribe_all_segments
from core.segments import segments_to_text
from core.documents import is_document, load_document, pages_to_text
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.vector_store import make_video_id
from core.rag_engine import build_rag_chain, ask_question

def run_pipeline(source: str = None, language: str = "english", documents: list = None) -> dict:
    """
    Three modes:
      - video/audio only      run_pipeline("https://youtu.be/...", "english")
      - video/audio + slides  run_pipeline(source, "english", documents=["slides.pptx", "notes.pdf"])
      - documents only        run_pipeline(None, documents=["slides.pptx"])
                              (or pass a .pdf/.pptx path as `source`)
    """
    print("Starting the meeting summarization pipeline...")

    doc_paths = list(documents or [])

    # A PDF/PPTX given as the main source is a documents-only session.
    if source and not is_url(source) and is_document(source):
        doc_paths.insert(0, source)
        source = None

    if not source and not doc_paths:
        raise ValueError("Provide a video/audio source or at least one PDF/PPTX document.")

    # Load documents first: it is fast, and a bad file should fail before the
    # slow download/transcription starts.
    pages = []
    for path in doc_paths:
        pages.extend(load_document(path))

    # ---------------- documents only ----------------
    if not source:
        text = pages_to_text(pages)
        print(f"Extracted {len(pages)} page(s)/slide(s) from {len(doc_paths)} document(s).")

        return {
            "mode": "document",
            "title": generate_title(text, "document"),
            "transcript": text,
            "summary": summarize(text, "document"),
            # Action items / decisions / open questions only make sense for meetings.
            "action_items": "",
            "key_decisions": "",
            "open_questions": "",
            "rag_chain": build_rag_chain(None, with_sources=True, pages=pages),
            "segments": [],
            "pages": pages,
        }

    # ---------------- video / audio (+ optional documents) ----------------
    chunks = process_input(source)

    # Segments carry start/end times (seconds from the start of the video).
    try:
        segments = transcribe_all_segments(chunks, language)
    finally:
        cleanup_audio(chunks)  # chunk files + intermediate wav are no longer needed
    transcript = segments_to_text(segments)
    if not transcript.strip():
        raise ValueError("No speech was detected in this recording.")
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    # One index per session (video + its attached documents); answers come back
    # with timestamped / slide-numbered sources.
    rag_chain = build_rag_chain(
        segments,
        video_id=make_video_id(source + pages_to_text(pages)),
        with_sources=True,
        pages=pages,
    )

    return {
        "mode": "video",
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "segments": segments,
        "pages": pages,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path (or a PDF/PPTX): ").strip() or None
    language = input("Language (english/hinglish): ").strip() or "english"
    extra = input("Slides/PDF paths to add, comma-separated (optional): ").strip()
    documents = [p.strip() for p in extra.split(",") if p.strip()]
    result = run_pipeline(source, language, documents=documents)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    if result["action_items"]:
        print(f"\n✅ Action Items:\n{result['action_items']}")
    if result["key_decisions"]:
        print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    if result["open_questions"]:
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your material (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        reply = ask_question(rag_chain, question)
        answer = reply["answer"] if isinstance(reply, dict) else reply
        print(f"\n🤖 Assistant: {answer}")
        if isinstance(reply, dict):
            for src in reply.get("sources", []):
                where = src.get("timestamp") or src.get("label")
                if where:
                    print(f"   🕒 {where}")
        print()