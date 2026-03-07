# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp[cli]",
#     "pymupdf",
#     "reportlab",
#     "markdown"
# ]
# ///

import logging
import os
import re
import tempfile
import zipfile
import sys
from functools import lru_cache
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import markdown as md_lib
from mcp.server.fastmcp import FastMCP
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pdf_server")

MAX_PDF_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB (Adjusted for better capacity)
FONTS_DIR_NAME = "fonts"

# Character replacements for unsupported PDF encodings
CHAR_REPLACEMENTS = str.maketrans({
    "•": "-", "●": "-", "▪": "-", "—": "-", "–": "-",
    "│": "|", "─": "-", "┌": "",  "┐": "",  "└": "",  "┘": "",
    "«": '"', "»": '"', "…": "...", "№": "N",
})

TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")
HORIZONTAL_RULE_RE = re.compile(r"^(-{3,}|\*{3,})$")
MARKDOWN_IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)$")
MARKDOWN_TAG_REPLACEMENTS = (
    ("<strong>", "<b>"), ("</strong>", "</b>"),
    ("<em>", "<i>"), ("</em>", "</i>"),
    ("<p>", ""), ("</p>", ""),
    ("<code>", '<font name="Courier" backColor="#f0f0f0">'),
    ("</code>", "</font>"),
)

HEADING_RULES: Tuple[Tuple[str, str, int], ...] = (
    ("#### ", "h4", 5),
    ("### ", "h3", 4),
    ("## ", "h2", 3),
    ("# ", "title", 2),
)

DEFAULT_DOC_MARGIN = 50
DEFAULT_IMAGE_MAX_WIDTH = 6 * inch
DEFAULT_FONT_NAME = "Roboto-Regular"
DEFAULT_EDIT_SUFFIX = "_edited.pdf"
DEFAULT_ROTATE_SUFFIX = "_rot.pdf"
LINE_GROUP_TOLERANCE = 5.0

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

# ---------------------------------------------------------------------------
# Font Management
# ---------------------------------------------------------------------------

class FontManager:
    """Handles font registration and path management."""
    
    _fonts_cache: Optional[Dict[str, str]] = None

    @classmethod
    def get_fonts_dir(cls) -> Path:
        return Path(__file__).parent / FONTS_DIR_NAME

    @classmethod
    def ensure_fonts(cls) -> Dict[str, str]:
        """
        Registers Roboto fonts from the local './fonts' directory.
        Call once; results are cached for the process lifetime.
        """
        if cls._fonts_cache is not None:
            return cls._fonts_cache

        fonts_dir = cls.get_fonts_dir()
        if not fonts_dir.exists():
            # Fallback to system fonts or warning if critical
            # For now, we raise error as per requirements, but in prod we might fallback
            raise PDFProcessingError(
                f"Fonts directory not found: {fonts_dir}\n"
                f"Create a '{FONTS_DIR_NAME}' folder next to the script and place "
                f"Roboto-Regular.ttf, Roboto-Bold.ttf, Roboto-Italic.ttf, "
                f"Roboto-BoldItalic.ttf inside it."
            )

        variants = {
            "Roboto-Regular":    "Roboto-Regular.ttf",
            "Roboto-Bold":       "Roboto-Bold.ttf",
            "Roboto-Italic":     "Roboto-Italic.ttf",
            "Roboto-BoldItalic": "Roboto-BoldItalic.ttf",
        }

        paths: Dict[str, str] = {}
        for font_name, filename in variants.items():
            file_path = fonts_dir / filename
            if not file_path.exists():
                raise PDFProcessingError(
                    f"Required font file missing: {filename}\n"
                    f"Expected location: {file_path}"
                )
            
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(file_path)))
            paths[font_name] = str(file_path)

        # Register family mapping so <b> and <i> tags work in ReportLab paragraphs
        try:
            addMapping("Roboto-Regular", 0, 0, "Roboto-Regular")
            addMapping("Roboto-Regular", 1, 0, "Roboto-Bold")
            addMapping("Roboto-Regular", 0, 1, "Roboto-Italic")
            addMapping("Roboto-Regular", 1, 1, "Roboto-BoldItalic")
        except Exception as e:
            logger.warning(f"Font mapping registration failed: {e}")

        cls._fonts_cache = {
            "regular":      "Roboto-Regular",
            "bold":         "Roboto-Bold",
            "italic":       "Roboto-Italic",
            "mono":         "Roboto-Regular",  # Fallback to Roboto-Regular for Cyrillic support
            "path_regular": paths["Roboto-Regular"],
        }
        #logger.info(f"Fonts registered from {fonts_dir}")
        return cls._fonts_cache

# ---------------------------------------------------------------------------
# ReportLab Components
# ---------------------------------------------------------------------------

class SeparatorLine(Flowable):
    """Draws a horizontal separator line."""
    
    def __init__(self, width_percent=100, thickness=0.5, color=colors.grey, space_after=10):
        super().__init__()
        self.width_percent = width_percent
        self.thickness = thickness
        self.color = color
        self.space_after = space_after
        self.width = 0

    def wrap(self, availWidth, availHeight):
        self.width = availWidth * self.width_percent / 100
        return self.width, self.thickness + self.space_after

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space_after, self.width, self.space_after)

class PDFGenerator:
    """Handles PDF creation from Markdown content using ReportLab."""

    def __init__(self):
        self.fonts = FontManager.ensure_fonts()
        self.styles = self._create_custom_styles()

    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Builds ReportLab paragraph styles."""
        base = getSampleStyleSheet()
        return {
            "title":  ParagraphStyle("CustomTitle",  parent=base["Heading1"], fontName=self.fonts["bold"],    fontSize=24, alignment=TA_CENTER,  spaceAfter=20, textColor=colors.HexColor("#2c3e50"), leading=28),
            "h2":     ParagraphStyle("CustomH2",      parent=base["Heading2"], fontName=self.fonts["bold"],    fontSize=18, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor("#2c3e50"), leading=22),
            "h3":     ParagraphStyle("CustomH3",      parent=base["Heading3"], fontName=self.fonts["bold"],    fontSize=14, spaceBefore=10, spaceAfter=5,  textColor=colors.HexColor("#34495e")),
            "h4":     ParagraphStyle("CustomH4",      parent=base["Heading3"], fontName=self.fonts["bold"],    fontSize=11, spaceBefore=8,  spaceAfter=4,  textColor=colors.HexColor("#7f8c8d")),
            "body":   ParagraphStyle("CustomBody",    parent=base["Normal"],   fontName=self.fonts["regular"], fontSize=10, leading=14, alignment=TA_LEFT, spaceAfter=6),
            "quote":  ParagraphStyle("CustomQuote",   parent=base["Normal"],   fontName=self.fonts["italic"],  fontSize=10, leftIndent=20, rightIndent=20, spaceBefore=10, spaceAfter=10, textColor=colors.HexColor("#555555"), backColor=colors.HexColor("#f5f5f5"), borderPadding=10),
            "code":   ParagraphStyle("CodeBlock",     parent=base["Normal"],   fontName=self.fonts["mono"],    fontSize=9,  leading=11, leftIndent=10, rightIndent=10, spaceBefore=5, spaceAfter=5, backColor=colors.HexColor("#f4f4f4"), borderPadding=5),
            "bullet": ParagraphStyle("CustomBullet",  parent=base["Normal"],   fontName=self.fonts["regular"], fontSize=10, leading=14, leftIndent=20, firstLineIndent=0, spaceAfter=4),
            "center": ParagraphStyle("Center",        parent=base["Normal"],   alignment=TA_CENTER),
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalizes text for reliable PDF rendering."""
        if not text:
            return ""
        text = text.translate(CHAR_REPLACEMENTS)
        if text.isascii():
            return text
        if any(ord(c) > 0xFFFF for c in text):
            return "".join(c if ord(c) <= 0xFFFF else " " for c in text)
        return text

    @staticmethod
    @lru_cache(maxsize=4096)
    def _markdown_to_reportlab(text: str) -> str:
        """Parses inline Markdown and converts to ReportLab XML tags."""
        if not text:
            return ""

        # markdown() handles escaping — do NOT unescape &amp; afterwards
        html = md_lib.markdown(text)
        for old, new in MARKDOWN_TAG_REPLACEMENTS:
            html = html.replace(old, new)

        return html.strip()

    def _inline(self, text: str) -> str:
        """Helper to clean and convert text for inline display."""
        return self._markdown_to_reportlab(self._clean_text(text))

    def _normalize_for_compare(self, text: str) -> str:
        return " ".join(self._clean_text(text).split()).lower()

    @staticmethod
    def _is_title_duplicate(title_norm: str, heading_norm: str) -> bool:
        return (
            heading_norm == title_norm
            or title_norm.startswith(heading_norm + " ")
            or title_norm.endswith(" " + heading_norm)
        )

    @staticmethod
    def _render_code_block(lines: List[str]) -> str:
        return "<br/>".join(
            html_escape(line, quote=False).replace(" ", "&nbsp;")
            for line in lines
        )

    def _append_image_block(self, story: List[Flowable], alt: str, image_path: Path) -> None:
        styles = self.styles
        if not image_path.exists():
            story.append(Paragraph(f"[Image not found: {image_path.name}]", styles["body"]))
            return

        try:
            img = RLImage(str(image_path))
            if img.drawWidth > DEFAULT_IMAGE_MAX_WIDTH:
                ratio = DEFAULT_IMAGE_MAX_WIDTH / img.drawWidth
                img.drawHeight *= ratio
                img.drawWidth = DEFAULT_IMAGE_MAX_WIDTH
            story.append(img)
            story.append(Paragraph(f"<i>{alt}</i>", styles["center"]))
            story.append(Spacer(1, 10))
        except Exception:
            story.append(Paragraph(f"[Error loading image: {image_path.name}]", styles["body"]))

    def _create_table(self, buf: List[str]) -> Optional[Table]:
        """Creates a ReportLab Table from a buffer of markdown table rows."""
        if not buf:
            return None

        rows = []
        body_style = self.styles["body"]
        inline = self._inline
        for row_str in buf:
            cells = [c.strip() for c in row_str.strip().strip("|").split("|")]
            # Skip separator lines like |---|---|
            if cells and all(TABLE_SEPARATOR_CELL_RE.match(c) for c in cells if c):
                continue
            rows.append([Paragraph(inline(c), body_style) for c in cells])

        if not rows:
            return None

        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#ecf0f1")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
            ("FONTNAME",      (0, 0), (-1, 0),  self.fonts["bold"]),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
            ("BACKGROUND",    (0, 1), (-1, -1), colors.white),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        return t

    def create_pdf(self, output_path: Path, title: str, content: str) -> str:
        """Generates a PDF file from markdown content."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        styles = self.styles
        inline = self._inline
        story: List[Flowable] = [Paragraph(inline(title), styles["title"]), Spacer(1, 10)]

        in_code = False
        code_lines: List[str] = []
        table_buf: List[str] = []

        def flush_table_buffer() -> None:
            if not table_buf:
                return
            table = self._create_table(table_buf)
            if table:
                story.append(table)
                story.append(Spacer(1, 10))
            table_buf.clear()

        title_norm = self._normalize_for_compare(title)
        title_deduped = False

        for line in content.splitlines():
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code:
                    story.append(Paragraph(self._render_code_block(code_lines), styles["code"]))
                    code_lines.clear()
                    in_code = False
                else:
                    flush_table_buffer()
                    in_code = True
                continue

            if in_code:
                code_lines.append(line)
                continue

            if stripped.startswith("|"):
                table_buf.append(stripped)
                continue
            flush_table_buffer()

            if not stripped:
                continue

            if HORIZONTAL_RULE_RE.match(stripped):
                story.append(SeparatorLine())
                continue

            img_match = MARKDOWN_IMAGE_RE.match(stripped)
            if img_match:
                alt, image_path_str = img_match.groups()
                self._append_image_block(story, alt, Path(image_path_str).resolve())
                continue

            matched_heading = False
            for prefix, style_key, skip in HEADING_RULES:
                if not stripped.startswith(prefix):
                    continue

                heading_text = stripped[skip:].strip()
                heading_norm = self._normalize_for_compare(heading_text)
                if prefix == "# " and not title_deduped and self._is_title_duplicate(title_norm, heading_norm):
                    title_deduped = True
                    matched_heading = True
                    break

                story.append(Paragraph(inline(heading_text), styles[style_key]))
                matched_heading = True
                break

            if matched_heading:
                continue

            if stripped.startswith("> "):
                story.append(Paragraph(inline(stripped[2:]), styles["quote"]))
                continue

            if stripped.startswith(("- ", "* ")):
                story.append(Paragraph(f"• {inline(stripped[2:])}", styles["bullet"]))
                continue

            story.append(Paragraph(inline(stripped), styles["body"]))

        flush_table_buffer()

        SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=DEFAULT_DOC_MARGIN,
            leftMargin=DEFAULT_DOC_MARGIN,
            topMargin=DEFAULT_DOC_MARGIN,
            bottomMargin=DEFAULT_DOC_MARGIN,
        ).build(story)

        return f"PDF created: {output_path} (Font: {self.fonts['regular']})"

# ---------------------------------------------------------------------------
# PDF Processor Logic
# ---------------------------------------------------------------------------

class PDFProcessor:
    """Core logic for PDF manipulation tasks."""

    @staticmethod
    def validate_file(pdf_path: str) -> Path:
        """Validates existence, extension and size of a PDF file."""
        # Performance: Use resolve() directly which handles both absolute and relative paths
        # relative to CWD. Avoid iterating multiple base paths if possible.
        try:
            path = Path(pdf_path).resolve(strict=True)
        except (FileNotFoundError, OSError):
            # Fallback for some edge cases where CWD isn't implicit (rare)
            path = (Path(os.getcwd()) / pdf_path).resolve()
            if not path.exists():
                raise PDFProcessingError(f"PDF file not found: {pdf_path}")

        if not path.is_file():
            raise PDFProcessingError(f"Path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFProcessingError("File must have .pdf extension")
        if path.stat().st_size > MAX_PDF_SIZE_BYTES:
            raise PDFProcessingError(f"PDF too large (max {MAX_PDF_SIZE_BYTES/1024/1024:.0f} MB)")
        return path

    @staticmethod
    def _default_output_path(path: Path, suffix: str) -> str:
        return str(path.parent / f"{path.stem}{suffix}")

    @staticmethod
    def _save_optimized(doc: fitz.Document, output_path: str) -> None:
        doc.save(output_path, garbage=4, deflate=True)

    @staticmethod
    def _collect_search_hits(page: fitz.Page, query: str, case_sensitive: bool) -> List[fitz.Rect]:
        if case_sensitive:
            return page.search_for(query)

        hits: List[fitz.Rect] = []
        seen = set()
        variants = tuple(dict.fromkeys((query, query.lower(), query.upper(), query.title())))
        for variant in variants:
            for rect in page.search_for(variant):
                key = (round(rect.x0), round(rect.y0))
                if key in seen:
                    continue
                seen.add(key)
                hits.append(rect)
        hits.sort(key=lambda r: (r.y0, r.x0))
        return hits

    @staticmethod
    def _group_hits_by_line(hits: List[fitz.Rect], tolerance: float = LINE_GROUP_TOLERANCE) -> List[List[fitz.Rect]]:
        groups: List[List[fitz.Rect]] = []
        for rect in sorted(hits, key=lambda r: (r.y0, r.x0)):
            if groups and abs(rect.y0 - groups[-1][-1].y1) < tolerance:
                groups[-1].append(rect)
            else:
                groups.append([rect])
        return groups

    @staticmethod
    def _rect_union(rects: List[fitz.Rect]) -> fitz.Rect:
        return fitz.Rect(
            min(r.x0 for r in rects),
            min(r.y0 for r in rects),
            max(r.x1 for r in rects),
            max(r.y1 for r in rects),
        )

    @staticmethod
    def _split_range_token(token: str) -> Optional[Tuple[int, int]]:
        try:
            parts = token.split("-")
            start = int(parts[0])
            end = int(parts[-1]) if len(parts) > 1 else start
            return start, end
        except ValueError:
            return None

    @staticmethod
    def extract_text(pdf_path: str, page_start: int = 1, page_end: Optional[int] = None, sort: bool = True) -> str:
        """
        Extracts text from a page range.
        
        Args:
            sort: If True, sorts text blocks by vertical/horizontal position.
                  Set to False for faster extraction if layout order doesn't matter.
        """
        path = PDFProcessor.validate_file(pdf_path)
        with fitz.open(path) as doc:
            total = len(doc)
            start = max(1, page_start)
            end = min(total, page_end if page_end is not None else total)

            if start > end:
                return f"Error: invalid page range {start}-{end} (total pages: {total})"

            pages = []
            # Optimization: Pre-allocate list size if possible, but python lists are dynamic.
            # Using list comprehension or generator is faster than append in loop for simple cases,
            # but here logic is complex.
            for i in range(start - 1, end):
                # Optimization: 'blocks' is structure-aware. 'text' is faster if structure not needed.
                # We stick to blocks for consistency but allow disabling sort.
                blocks = doc[i].get_text("blocks", sort=sort)
                # Block format: (x0, y0, x1, y1, text, block_no, block_type)
                text = "\n".join(b[4] for b in blocks if b[6] == 0).strip()
                pages.append(f"[Page {i + 1}]\n{text}")

            header = f"[PDF: {path.name} | Pages {start}-{end} of {total}]\n\n"
            return header + "\n\n".join(pages)

    @staticmethod
    def search_text(pdf_path: str, query: str, case_sensitive: bool = False) -> str:
        """Searches for text in PDF and returns locations with context."""
        path = PDFProcessor.validate_file(pdf_path)
        if not query:
            return "Not found"

        results: List[str] = []
        with fitz.open(path) as doc:
            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                hits = PDFProcessor._collect_search_hits(page, query, case_sensitive)
                if not hits:
                    continue

                # Heuristic: clipped extraction is cheaper for a few hits, full page for many.
                if len(hits) < 5:
                    for occ_idx, rect in enumerate(hits, start=1):
                        clip_rect = fitz.Rect(rect.x0 - 50, rect.y0 - 20, rect.x1 + 50, rect.y1 + 20)
                        clip_rect &= page.rect
                        context = page.get_text("text", clip=clip_rect).replace("\n", " ").strip()
                        results.append(
                            f"Page {page_num} occurrence #{occ_idx}: "
                            f"rect=({rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f}) "
                            f'| context: "...{context}..."'
                        )
                    continue

                page_text = page.get_text()
                lower_page = page_text.lower()
                lower_q = query.lower()
                search_off = 0

                for occ_idx, rect in enumerate(hits, start=1):
                    pos = lower_page.find(lower_q, search_off)
                    ctx_str = ""
                    if pos != -1:
                        ctx_s = max(0, pos - 30)
                        ctx_e = min(len(page_text), pos + len(query) + 30)
                        context = page_text[ctx_s:ctx_e].replace("\n", " ").strip()
                        ctx_str = f' | context: "...{context}..."'
                        search_off = pos + 1

                    results.append(
                        f"Page {page_num} occurrence #{occ_idx}: "
                        f"rect=({rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f})"
                        f"{ctx_str}"
                    )

        if not results:
            return "Not found"
        return f"Found {len(results)} occurrence(s):\n" + "\n".join(results)

    @staticmethod
    def _detect_text_style(page: fitz.Page, rect: fitz.Rect) -> Dict[str, Any]:
        """Detects dominant font size and color in a rect."""
        defaults = {"size": 11.0, "color": (0, 0, 0)}
        try:
            # Optimization: limit clip to exact rect to reduce parsing
            blocks = page.get_text("dict", clip=rect)["blocks"]
            spans = [
                span
                for b in blocks if b.get("type") == 0
                for line in b["lines"]
                for span in line["spans"]
                if span["text"].strip()
            ]
            if not spans:
                return defaults

            span = max(spans, key=lambda s: s["size"])
            raw = span.get("color", 0)
            return {
                "size": span["size"],
                "color": (((raw >> 16) & 0xFF) / 255, ((raw >> 8) & 0xFF) / 255, (raw & 0xFF) / 255),
            }
        except Exception:
            return defaults

    @staticmethod
    def _insert_textbox_with_font_fallback(
        page: fitz.Page,
        rect: fitz.Rect,
        text: str,
        fontsize: float,
        color: tuple[float, float, float],
        font_path: str,
    ) -> float:
        """Inserts text with a lightweight fallback when the font isn't already present."""
        try:
            return page.insert_textbox(
                rect,
                text,
                fontsize=fontsize,
                fontname=DEFAULT_FONT_NAME,
                fontfile=None,
                color=color,
                align=0,
            )
        except Exception:
            return page.insert_textbox(
                rect,
                text,
                fontsize=fontsize,
                fontname=DEFAULT_FONT_NAME,
                fontfile=font_path,
                color=color,
                align=0,
            )

    @classmethod
    def edit_text(
        cls,
        pdf_path: str,
        page_num: int,
        old_text: str,
        new_text: str,
        output_path: Optional[str] = None,
        occurrence: int = 0,
    ) -> str:
        """Replaces or deletes text on a page."""
        path = cls.validate_file(pdf_path)
        fonts = FontManager.ensure_fonts()

        with fitz.open(path) as doc:
            if not (1 <= page_num <= len(doc)):
                raise PDFProcessingError(f"Page {page_num} out of range (1-{len(doc)})")

            page = doc[page_num - 1]
            groups = cls._group_hits_by_line(page.search_for(old_text))
            if not groups:
                return f"Text not found: {old_text!r}"

            total = len(groups)
            if occurrence == 0:
                target_groups = groups
            elif 1 <= occurrence <= total:
                target_groups = [groups[occurrence - 1]]
            else:
                return f"Occurrence {occurrence} not found. Total: {total}"

            for group in target_groups:
                for rect in group:
                    page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions()

            warnings: List[str] = []
            failed_inserts = 0
            replacement_lines = max(1, len(new_text.splitlines()))

            if new_text.strip():
                for group in target_groups:
                    bbox = cls._rect_union(group)
                    style = cls._detect_text_style(page, bbox)
                    fontsize = style["size"]
                    line_height = fontsize * 1.4

                    needed_h = replacement_lines * line_height + fontsize
                    orig_h = bbox.y1 - bbox.y0
                    insert_bbox = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y0 + max(orig_h, needed_h))

                    result = cls._insert_textbox_with_font_fallback(
                        page,
                        insert_bbox,
                        new_text,
                        fontsize,
                        style["color"],
                        fonts["path_regular"],
                    )

                    if result < 0:
                        failed_inserts += 1
                        warnings.append(
                            f"Could not fit text into rect {insert_bbox} "
                            f"(overflow {abs(result):.1f} pt). Try shorter replacement text."
                        )

            if failed_inserts and failed_inserts == len(target_groups):
                return "ERROR: text insertion failed — file NOT saved to avoid data loss.\n" + "\n".join(warnings)

            out = output_path or cls._default_output_path(path, DEFAULT_EDIT_SUFFIX)
            cls._save_optimized(doc, out)

        msg = f"Saved to {out} (replaced {len(target_groups)} occurrences)"
        if warnings:
            msg += "\n" + "\n".join(warnings)
        return msg

    @classmethod
    def add_text(
        cls,
        pdf_path: str,
        page_num: int,
        text: str,
        x: float = 56.0,
        y: Optional[float] = None,
        fontsize: float = 10.0,
        output_path: Optional[str] = None,
    ) -> str:
        """Adds new text to a page."""
        path = cls.validate_file(pdf_path)
        fonts = FontManager.ensure_fonts()

        with fitz.open(path) as doc:
            if not (1 <= page_num <= len(doc)):
                raise PDFProcessingError(f"Page {page_num} out of range (1-{len(doc)})")

            page = doc[page_num - 1]
            page_height = page.rect.height
            page_width = page.rect.width

            if y is None:
                blocks = page.get_text("blocks", sort=True)
                text_y1s = [b[3] for b in blocks if b[6] == 0 and str(b[4]).strip()]
                y = (max(text_y1s) + 20) if text_y1s else (page_height - 60)

            right_margin = page_width - 56
            line_height = fontsize * 1.4
            n_lines = max(1, len(text.splitlines()))
            rect = fitz.Rect(x, y, right_margin, y + n_lines * line_height + fontsize)
            if rect.y1 > page_height - 20:
                rect = fitz.Rect(rect.x0, rect.y0, rect.x1, page_height - 20)

            result = cls._insert_textbox_with_font_fallback(
                page,
                rect,
                text,
                fontsize,
                (0, 0, 0),
                fonts["path_regular"],
            )

            out = output_path or cls._default_output_path(path, DEFAULT_EDIT_SUFFIX)
            cls._save_optimized(doc, out)

        msg = f"Saved to {out} (text added at y={y:.1f})"
        if result < 0:
            msg += f"\nWarning: text may be clipped (overflow {abs(result):.1f} pt)."
        return msg

    @staticmethod
    def merge_pdfs(pdf_paths: List[str], output_path: str) -> str:
        """Merges multiple PDF files."""
        with fitz.open() as doc:
            for pdf in pdf_paths:
                with fitz.open(PDFProcessor.validate_file(pdf)) as src:
                    doc.insert_pdf(src)
            doc.save(output_path)
        return f"Merged {len(pdf_paths)} files to {output_path}"

    @staticmethod
    def split_pdf(pdf_path: str, ranges: List[str], output_dir: str) -> str:
        """Splits PDF by ranges."""
        path = PDFProcessor.validate_file(pdf_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files: List[str] = []

        with fitz.open(path) as doc:
            for idx, token in enumerate(ranges, start=1):
                parsed = PDFProcessor._split_range_token(token)
                if parsed is None:
                    logger.warning(f"Invalid range: {token}")
                    continue

                start, end = parsed
                start, end = max(1, start), min(len(doc), end)
                if start > end:
                    continue

                fname = out_dir / f"split_{idx}_{start}-{end}.pdf"
                with fitz.open() as new_doc:
                    new_doc.insert_pdf(doc, from_page=start - 1, to_page=end - 1)
                    new_doc.save(fname)
                files.append(str(fname))

        return f"Created {len(files)} files in {out_dir}"

    @staticmethod
    def rotate_pages(pdf_path: str, rotation: int, pages: Optional[List[int]] = None) -> str:
        """Rotates pages by 90/180/270."""
        path = PDFProcessor.validate_file(pdf_path)
        with fitz.open(path) as doc:
            indices = [page_num - 1 for page_num in (pages or range(1, len(doc) + 1))]
            for idx in indices:
                if 0 <= idx < len(doc):
                    doc[idx].set_rotation(rotation)
            out = PDFProcessor._default_output_path(path, DEFAULT_ROTATE_SUFFIX)
            PDFProcessor._save_optimized(doc, out)
        return f"Rotated pages saved to {out}"

    @staticmethod
    def extract_images(pdf_path: str) -> str:
        """Extracts images to a ZIP file."""
        path = PDFProcessor.validate_file(pdf_path)
        zip_path = Path(tempfile.gettempdir()) / f"{path.stem}_images.zip"
        count = 0
        
        with fitz.open(path) as doc, zipfile.ZipFile(zip_path, "w") as zf:
            for i, page in enumerate(doc):
                for img in page.get_images(full=True):
                    xref = img[0]
                    try:
                        base_img = doc.extract_image(xref)
                        zf.writestr(f"p{i + 1}_img{xref}.{base_img['ext']}", base_img["image"])
                        count += 1
                    except Exception as e:
                        logger.warning(f"Image extract error: {e}")
        
        return f"Extracted {count} images to {zip_path}"

    @staticmethod
    def images_to_pdf(image_paths: List[str], output_path: str) -> str:
        """Creates a PDF from a list of images."""
        if not image_paths:
            raise PDFProcessingError("No images provided")

        with fitz.open() as doc:
            for image in image_paths:
                path = Path(image).resolve()
                if not path.exists():
                    continue
                with fitz.open(path) as img:
                    rect = img[0].rect
                    pdf_bytes = img.convert_to_pdf()
                    with fitz.open("pdf", pdf_bytes) as img_pdf:
                        doc.new_page(width=rect.width, height=rect.height).show_pdf_page(rect, img_pdf, 0)
            PDFProcessor._save_optimized(doc, output_path)

        return f"PDF created from {len(image_paths)} images at {output_path}"

# ---------------------------------------------------------------------------
# MCP Server & Tool Definitions
# ---------------------------------------------------------------------------

mcp = FastMCP("PDFMcp")


def _run_tool(
    fn: Callable[..., str],
    error_prefix: str,
    *args: Any,
    log_error: bool = False,
    **kwargs: Any,
) -> str:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"{error_prefix}: {e}", exc_info=True)
        raise PDFProcessingError(f"{error_prefix}: {e}")


@mcp.tool()
def pdf_extract_text(pdf_path: str, page_start: int = 1, page_end: Optional[int] = None, sort: bool = True) -> str:
    """[PDF ONLY] Extracts text from a .pdf file."""
    return _run_tool(PDFProcessor.extract_text, "Extraction failed", pdf_path, page_start, page_end, sort)


@mcp.tool()
def pdf_create(output_path: str, title: str, content: str) -> str:
    """[PDF ONLY] Creates a new .pdf file from Markdown text."""
    def _create() -> str:
        generator = PDFGenerator()
        return generator.create_pdf(Path(output_path).resolve(), title, content)

    return _run_tool(_create, "Failed to create PDF", log_error=True)


@mcp.tool()
def pdf_search(pdf_path: str, query: str, case_sensitive: bool = False) -> str:
    """[PDF ONLY] Searches for text. Returns page number, occurrence, rect, context."""
    return _run_tool(PDFProcessor.search_text, "Search failed", pdf_path, query, case_sensitive)


@mcp.tool()
def pdf_edit_text(
    pdf_path: str,
    page_num: int,
    old_text: str,
    new_text: str,
    output_path: Optional[str] = None,
    occurrence: int = 0,
) -> str:
    """
    [PDF ONLY] Replaces or deletes EXISTING text on a specific page.
    """
    return _run_tool(
        PDFProcessor.edit_text,
        "Edit failed",
        pdf_path,
        page_num,
        old_text,
        new_text,
        output_path,
        occurrence,
    )


@mcp.tool()
def pdf_add_text(
    pdf_path: str,
    page_num: int,
    text: str,
    x: float = 56.0,
    y: Optional[float] = None,
    fontsize: float = 10.0,
    output_path: Optional[str] = None,
) -> str:
    """[PDF ONLY] Adds NEW text at a specific position on a page."""
    return _run_tool(PDFProcessor.add_text, "Add text failed", pdf_path, page_num, text, x, y, fontsize, output_path)


@mcp.tool()
def pdf_merge(pdf_paths: List[str], output_path: str) -> str:
    """[PDF ONLY] Merges multiple .pdf files."""
    return _run_tool(PDFProcessor.merge_pdfs, "Merge failed", pdf_paths, output_path)


@mcp.tool()
def pdf_split(pdf_path: str, ranges: List[str], output_dir: str) -> str:
    """[PDF ONLY] Splits PDF by ranges (e.g. ["1-3", "5"])."""
    return _run_tool(PDFProcessor.split_pdf, "Split failed", pdf_path, ranges, output_dir)


@mcp.tool()
def pdf_rotate_pages(pdf_path: str, rotation: int, pages: Optional[List[int]] = None) -> str:
    """[PDF ONLY] Rotates pages by 90/180/270."""
    return _run_tool(PDFProcessor.rotate_pages, "Rotation failed", pdf_path, rotation, pages)


@mcp.tool()
def pdf_extract_images(pdf_path: str) -> str:
    """[PDF ONLY] Extracts images to ZIP."""
    return _run_tool(PDFProcessor.extract_images, "Image extraction failed", pdf_path)


@mcp.tool()
def pdf_from_images(image_paths: List[str], output_path: str) -> str:
    """[PDF ONLY] Creates PDF from images."""
    return _run_tool(PDFProcessor.images_to_pdf, "Conversion failed", image_paths, output_path)


if __name__ == "__main__":
    # Пишем информацию о запуске в stderr
    #logger.info("=== Starting Local PDF MCP Server ===")
    #logger.info(f"Максимальный размер PDF: {MAX_PDF_SIZE_BYTES / 1024 / 1024:.0f} MB")
    #logger.info(f"Ожидаемая папка со шрифтами: {FontManager.get_fonts_dir()}")
    
    # Можно сразу проверить наличие шрифтов при запуске, 
    # чтобы сервер упал с понятной ошибкой до того, как к нему обратится ИИ
    try:
        FontManager.ensure_fonts()
        logger.info("Fonts loaded successfully.")
    except Exception as e:
        logger.error(f"Font initialization error: {e}")
        sys.exit(1)

    logger.info("Server started successfully and listening on stdio...")
    
    try:
        # Запускаем сервер
        mcp.run()
    except KeyboardInterrupt:
        # Перехватываем нажатие Ctrl+C
        logger.info("Received shutdown signal (Ctrl+C). Shutting down gracefully...")
        sys.exit(0)

