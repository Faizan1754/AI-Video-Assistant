import os
from dataclasses import dataclass
from typing import Iterable, List

SUPPORTED_DOC_EXTS = (".pdf", ".pptx")


@dataclass
class DocPage:
    """Text of one PDF page or one PowerPoint slide."""
    number: int          # 1-based page / slide number
    text: str
    kind: str            # "pdf" or "pptx"
    source_name: str     # original file name, shown in citations

    @property
    def label(self) -> str:
        unit = "Slide" if self.kind == "pptx" else "Page"
        return f"{unit} {self.number} ({self.source_name})"


def is_document(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_DOC_EXTS


def pages_to_text(pages: Iterable[DocPage]) -> str:
    """All page texts joined (no labels) - what the summarizer consumes."""
    return "\n\n".join(p.text.strip() for p in pages if p.text.strip())


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def _load_pdf(path: str) -> List[DocPage]:
    from pypdf import PdfReader

    name = os.path.basename(path)
    reader = PdfReader(path)

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty user password.
        try:
            ok = reader.decrypt("")
        except Exception:
            ok = 0
        if not ok:
            raise ValueError(f"{name} is password-protected and cannot be read.")

    pages: List[DocPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(DocPage(i, text, "pdf", name))

    if not pages:
        raise ValueError(
            f"No text could be extracted from {name}. It is probably a scanned "
            "document (images only), which needs OCR - not supported yet."
        )
    return pages


# --------------------------------------------------------------------------
# PPTX
# --------------------------------------------------------------------------
def _shape_texts(shapes) -> List[str]:
    from pptx.shapes.group import GroupShape

    out: List[str] = []
    for shape in shapes:
        if isinstance(shape, GroupShape):
            out.extend(_shape_texts(shape.shapes))
            continue
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if text:
                out.append(text)
        if shape.has_table:
            for row in shape.table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    out.append(" | ".join(cells))
    return out


def _load_pptx(path: str) -> List[DocPage]:
    from pptx import Presentation

    name = os.path.basename(path)
    prs = Presentation(path)

    pages: List[DocPage] = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = _shape_texts(slide.shapes)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes is not None and notes.text.strip():
                parts.append("Speaker notes: " + notes.text.strip())
        text = "\n".join(parts).strip()
        if text:
            pages.append(DocPage(i, text, "pptx", name))

    if not pages:
        raise ValueError(
            f"No text found in {name}. The slides may contain only images "
            "(text inside pictures is not extracted)."
        )
    return pages


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load_document(path: str) -> List[DocPage]:
    """Extract text per page/slide from a .pdf or .pptx file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".pptx":
        return _load_pptx(path)
    if ext == ".ppt":
        raise ValueError(
            "Old .ppt files are not supported. Save the file as .pptx (or export "
            "it to PDF) and upload that instead."
        )
    raise ValueError(f"Unsupported document type: {ext or 'unknown'} (use PDF or PPTX).")