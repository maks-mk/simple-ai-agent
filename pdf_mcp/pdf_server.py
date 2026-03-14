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
import urllib.error
import urllib.request
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

MAX_PDF_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
FONTS_DIR_NAME = "fonts"

# Download destination for Roboto fallback
ROBOTO_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "mcp_fonts_roboto"

# System font candidates: (display_name, regular, bold, italic, bold_italic)
# italic/bold_italic may duplicate regular/bold if no dedicated variant exists
_SYSTEM_FONT_CANDIDATES: Tuple[Tuple[str, str, str, str, str], ...] = (
    # Windows
    ("Arial",
     r"C:\Windows\Fonts\arial.ttf",
     r"C:\Windows\Fonts\arialbd.ttf",
     r"C:\Windows\Fonts\ariali.ttf",
     r"C:\Windows\Fonts\arialbi.ttf"),
    ("Verdana",
     r"C:\Windows\Fonts\verdana.ttf",
     r"C:\Windows\Fonts\verdanab.ttf",
     r"C:\Windows\Fonts\verdanai.ttf",
     r"C:\Windows\Fonts\verdanaz.ttf"),
    ("Tahoma",
     r"C:\Windows\Fonts\tahoma.ttf",
     r"C:\Windows\Fonts\tahomabd.ttf",
     r"C:\Windows\Fonts\tahoma.ttf",
     r"C:\Windows\Fonts\tahomabd.ttf"),
    ("SegoeUI",
     r"C:\Windows\Fonts\segoeui.ttf",
     r"C:\Windows\Fonts\segoeuib.ttf",
     r"C:\Windows\Fonts\segoeuii.ttf",
     r"C:\Windows\Fonts\segoeuiz.ttf"),
    # Linux
    ("DejaVuSans",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
    ("LiberationSans",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"),
    ("FreeSans",
     "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
     "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
     "/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf",
     "/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf"),
)

# Roboto download URLs (Google Fonts GitHub mirror)
_ROBOTO_URLS: Dict[str, str] = {
    "Roboto-Regular":    "https://github.com/googlefonts/roboto-2/raw/main/src/hinted/Roboto-Regular.ttf",
    "Roboto-Bold":       "https://github.com/googlefonts/roboto-2/raw/main/src/hinted/Roboto-Bold.ttf",
    "Roboto-Italic":     "https://github.com/googlefonts/roboto-2/raw/main/src/hinted/Roboto-Italic.ttf",
    "Roboto-BoldItalic": "https://github.com/googlefonts/roboto-2/raw/main/src/hinted/Roboto-BoldItalic.ttf",
}

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
    """
    Handles font registration with a three-level fallback chain:
    1. Local ./fonts directory (Roboto TTF files placed manually)
    2. System fonts with Cyrillic support (Windows: Arial/Verdana/Tahoma/Segoe UI,
       Linux: DejaVu Sans / Liberation Sans / Free Sans)
    3. Auto-download Roboto from Google Fonts GitHub mirror into a temp directory
       (requires internet access, only on first run)
    """

    _fonts_cache: Optional[Dict[str, str]] = None

    @classmethod
    def get_fonts_dir(cls) -> Path:
        return Path(__file__).parent / FONTS_DIR_NAME

    @classmethod
    def ensure_fonts(cls) -> Dict[str, str]:
        """
        Returns a font dict with keys: regular, bold, italic, mono, path_regular.
        Results are cached for the process lifetime.
        """
        if cls._fonts_cache is not None:
            return cls._fonts_cache

        # Strategy 1: local ./fonts directory
        result = cls._try_local_fonts()
        if result:
            logger.info("Fonts loaded from local directory: %s", cls.get_fonts_dir())
            cls._fonts_cache = result
            return result

        # Strategy 2: system fonts
        result = cls._try_system_fonts()
        if result:
            cls._fonts_cache = result
            return result

        # Strategy 3: download Roboto
        logger.info(
            "No local or system fonts found. Downloading Roboto to %s (internet required)...",
            ROBOTO_DOWNLOAD_DIR,
        )
        result = cls._download_roboto()
        cls._fonts_cache = result
        return result

    @classmethod
    def _try_local_fonts(cls) -> Optional[Dict[str, str]]:
        fonts_dir = cls.get_fonts_dir()
        if not fonts_dir.exists():
            return None

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
                logger.debug("Local font missing: %s", file_path)
                return None
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(file_path)))
            paths[font_name] = str(file_path)

        cls._register_roboto_family_mapping()
        return {
            "regular":      "Roboto-Regular",
            "bold":         "Roboto-Bold",
            "italic":       "Roboto-Italic",
            "mono":         "Roboto-Regular",
            "path_regular": paths["Roboto-Regular"],
        }

    @classmethod
    def _try_system_fonts(cls) -> Optional[Dict[str, str]]:
        for display_name, regular, bold, italic, bold_italic in _SYSTEM_FONT_CANDIDATES:
            reg_path = Path(regular)
            bold_path = Path(bold)
            if not reg_path.exists() or not bold_path.exists():
                continue

            ital_path = Path(italic)
            bi_path = Path(bold_italic)
            # Gracefully degrade if italic variants are missing
            if not ital_path.exists():
                ital_path = reg_path
            if not bi_path.exists():
                bi_path = bold_path

            reg_name  = f"{display_name}-Regular"
            bold_name = f"{display_name}-Bold"
            ital_name = f"{display_name}-Italic"
            bi_name   = f"{display_name}-BoldItalic"

            try:
                for font_name, path in (
                    (reg_name,  reg_path),
                    (bold_name, bold_path),
                    (ital_name, ital_path),
                    (bi_name,   bi_path),
                ):
                    if font_name not in pdfmetrics.getRegisteredFontNames():
                        pdfmetrics.registerFont(TTFont(font_name, str(path)))

                try:
                    addMapping(reg_name, 0, 0, reg_name)
                    addMapping(reg_name, 1, 0, bold_name)
                    addMapping(reg_name, 0, 1, ital_name)
                    addMapping(reg_name, 1, 1, bi_name)
                except Exception as e:
                    logger.warning("Font mapping for %s failed: %s", display_name, e)

                logger.info("Using system font: %s (%s)", display_name, reg_path)
                return {
                    "regular":      reg_name,
                    "bold":         bold_name,
                    "italic":       ital_name,
                    "mono":         reg_name,
                    "path_regular": str(reg_path),
                }
            except Exception as e:
                logger.debug("System font %s failed to register: %s", display_name, e)
                continue

        return None

    @classmethod
    def _download_roboto(cls) -> Dict[str, str]:
        ROBOTO_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}

        for font_name, url in _ROBOTO_URLS.items():
            dest = ROBOTO_DOWNLOAD_DIR / f"{font_name}.ttf"
            if not dest.exists():
                try:
                    urllib.request.urlretrieve(url, str(dest))
                    logger.info("Downloaded %s", font_name)
                except urllib.error.URLError as e:
                    raise PDFProcessingError(
                        f"Failed to download font '{font_name}': {e}\n"
                        f"Options:\n"
                        f"  1. Create a '{FONTS_DIR_NAME}' folder next to the script and place "
                        f"Roboto TTF files inside it.\n"
                        f"  2. Ensure internet access for automatic download.\n"
                        f"  3. Install a supported system font (Windows: Arial/Verdana/Tahoma/Segoe UI; "
                        f"Linux: DejaVu Sans / Liberation Sans)."
                    )
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(dest)))
            paths[font_name] = str(dest)

        cls._register_roboto_family_mapping()
        logger.info("Roboto fonts ready at %s", ROBOTO_DOWNLOAD_DIR)
        return {
            "regular":      "Roboto-Regular",
            "bold":         "Roboto-Bold",
            "italic":       "Roboto-Italic",
            "mono":         "Roboto-Regular",
            "path_regular": paths["Roboto-Regular"],
        }

    @staticmethod
    def _register_roboto_family_mapping() -> None:
        try:
            addMapping("Roboto-Regular", 0, 0, "Roboto-Regular")
            addMapping("Roboto-Regular", 1, 0, "Roboto-Bold")
            addMapping("Roboto-Regular", 0, 1, "Roboto-Italic")
            addMapping("Roboto-Regular", 1, 1, "Roboto-BoldItalic")
        except Exception as e:
            logger.warning("Roboto family mapping failed: %s", e)

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
        html = md_lib.markdown(text)
        for old, new in MARKDOWN_TAG_REPLACEMENTS:
            html = html.replace(old, new)
        return html.strip()

    def _inline(self, text: str) -> str:
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
        if not buf:
            return None
        rows = []
        body_style = self.styles["body"]
        inline = self._inline
        for row_str in buf:
            cells = [c.strip() for c in row_str.strip().strip("|").split("|")]
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

    def _flush_table_buffer(self, story: List[Flowable], table_buf: List[str]) -> None:
        if not table_buf:
            return
        table = self._create_table(table_buf)
        if table:
            story.append(table)
            story.append(Spacer(1, 10))
        table_buf.clear()

    def _handle_heading(
        self,
        story: List[Flowable],
        stripped: str,
        title_norm: str,
        title_deduped: bool,
    ) -> tuple[bool, bool]:
        inline = self._inline
        styles = self.styles
        for prefix, style_key, skip in HEADING_RULES:
            if not stripped.startswith(prefix):
                continue
            heading_text = stripped[skip:].strip()
            heading_norm = self._normalize_for_compare(heading_text)
            if prefix == "# " and not title_deduped and self._is_title_duplicate(title_norm, heading_norm):
                return True, True
            story.append(Paragraph(inline(heading_text), styles[style_key]))
            return True, title_deduped
        return False, title_deduped

    def _handle_special_line(self, story: List[Flowable], stripped: str) -> bool:
        styles = self.styles
        inline = self._inline
        if HORIZONTAL_RULE_RE.match(stripped):
            story.append(SeparatorLine())
            return True
        img_match = MARKDOWN_IMAGE_RE.match(stripped)
        if img_match:
            alt, image_path_str = img_match.groups()
            self._append_image_block(story, alt, Path(image_path_str).resolve())
            return True
        if stripped.startswith("> "):
            story.append(Paragraph(inline(stripped[2:]), styles["quote"]))
            return True
        if stripped.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {inline(stripped[2:])}", styles["bullet"]))
            return True
        return False

    def create_pdf(self, output_path: Path, title: str, content: str) -> str:
        """Generates a PDF file from markdown content."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        styles = self.styles
        inline = self._inline
        story: List[Flowable] = [Paragraph(inline(title), styles["title"]), Spacer(1, 10)]
        in_code = False
        code_lines: List[str] = []
        table_buf: List[str] = []
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
                    self._flush_table_buffer(story, table_buf)
                    in_code = True
                continue
            if in_code:
                code_lines.append(line)
                continue
            if stripped.startswith("|"):
                table_buf.append(stripped)
                continue
            self._flush_table_buffer(story, table_buf)
            if not stripped:
                continue
            if self._handle_special_line(story, stripped):
                continue
            matched_heading, title_deduped = self._handle_heading(story, stripped, title_norm, title_deduped)
            if matched_heading:
                continue
            story.append(Paragraph(inline(stripped), styles["body"]))

        self._flush_table_buffer(story, table_buf)
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
        try:
            path = Path(pdf_path).resolve(strict=True)
        except (FileNotFoundError, OSError):
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
            for i in range(start - 1, end):
                blocks = doc[i].get_text("blocks", sort=sort)
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
    def _insert_textbox(
        page: fitz.Page,
        rect: fitz.Rect,
        text: str,
        fontsize: float,
        color: Tuple[float, float, float],
        fontname: str,
        font_path: str,
    ) -> float:
        """
        Inserts text into a rect using the specified font.
        Always loads the font from file to guarantee Cyrillic support
        regardless of prior registration state.
        Returns remaining space (negative = overflow).
        """
        return page.insert_textbox(
            rect,
            text,
            fontsize=fontsize,
            fontname=fontname,
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
                occurrence_desc = f"all {total} occurrence(s)"
            elif 1 <= occurrence <= total:
                target_groups = [groups[occurrence - 1]]
                occurrence_desc = f"occurrence #{occurrence} of {total}"
            else:
                return f"Occurrence {occurrence} not found. Total occurrences on this page: {total}"

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

                    result = cls._insert_textbox(
                        page,
                        insert_bbox,
                        new_text,
                        fontsize,
                        style["color"],
                        fonts["regular"],
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

        action = "deleted" if not new_text.strip() else "replaced"
        msg = f"Saved to {out} ({action} {occurrence_desc})"
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

            result = cls._insert_textbox(
                page,
                rect,
                text,
                fontsize,
                (0, 0, 0),
                fonts["regular"],
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
    def extract_images(pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Extracts images to a ZIP file.

        Args:
            output_path: Path for the ZIP archive. Defaults to <pdf_name>_images.zip
                         in the same directory as the source PDF.
        """
        path = PDFProcessor.validate_file(pdf_path)

        if output_path:
            zip_path = Path(output_path)
        else:
            zip_path = path.parent / f"{path.stem}_images.zip"

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

        added_count = 0
        skipped: List[str] = []

        with fitz.open() as doc:
            for image in image_paths:
                img_path = Path(image).resolve()
                if not img_path.exists():
                    skipped.append(image)
                    continue
                try:
                    with fitz.open(img_path) as img:
                        rect = img[0].rect
                        pdf_bytes = img.convert_to_pdf()
                        with fitz.open("pdf", pdf_bytes) as img_pdf:
                            doc.new_page(width=rect.width, height=rect.height).show_pdf_page(
                                rect, img_pdf, 0
                            )
                    added_count += 1
                except Exception as e:
                    logger.warning("Failed to add image %s: %s", image, e)
                    skipped.append(image)

            PDFProcessor._save_optimized(doc, output_path)

        msg = f"PDF created from {added_count} image(s) at {output_path}"
        if skipped:
            msg += f"\nSkipped {len(skipped)} file(s) (not found or unreadable): {', '.join(skipped)}"
        return msg

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


def _register_mcp_tool(name: str, doc: str, runner: Callable[..., str]) -> None:
    runner.__name__ = name
    runner.__doc__ = doc
    mcp.tool()(runner)


def _register_pdf_tools() -> None:
    tool_specs = [
        (
            "pdf_extract_text",
            "[PDF ONLY] Extracts text from a .pdf file.",
            lambda pdf_path, page_start=1, page_end=None, sort=True: _run_tool(
                PDFProcessor.extract_text,
                "Extraction failed",
                pdf_path,
                page_start,
                page_end,
                sort,
            ),
        ),
        (
            "pdf_create",
            "[PDF ONLY] Creates a new .pdf file from Markdown text.",
            lambda output_path, title, content: _run_tool(
                lambda: PDFGenerator().create_pdf(Path(output_path).resolve(), title, content),
                "Failed to create PDF",
                log_error=True,
            ),
        ),
        (
            "pdf_search",
            "[PDF ONLY] Searches for text. Returns page number, occurrence, rect, context.",
            lambda pdf_path, query, case_sensitive=False: _run_tool(
                PDFProcessor.search_text,
                "Search failed",
                pdf_path,
                query,
                case_sensitive,
            ),
        ),
        (
            "pdf_edit_text",
            "[PDF ONLY] Replaces or deletes EXISTING text on a specific page.",
            lambda pdf_path, page_num, old_text, new_text, output_path=None, occurrence=0: _run_tool(
                PDFProcessor.edit_text,
                "Edit failed",
                pdf_path,
                page_num,
                old_text,
                new_text,
                output_path,
                occurrence,
            ),
        ),
        (
            "pdf_add_text",
            "[PDF ONLY] Adds NEW text at a specific position on a page.",
            lambda pdf_path, page_num, text, x=56.0, y=None, fontsize=10.0, output_path=None: _run_tool(
                PDFProcessor.add_text,
                "Add text failed",
                pdf_path,
                page_num,
                text,
                x,
                y,
                fontsize,
                output_path,
            ),
        ),
        (
            "pdf_merge",
            "[PDF ONLY] Merges multiple .pdf files.",
            lambda pdf_paths, output_path: _run_tool(PDFProcessor.merge_pdfs, "Merge failed", pdf_paths, output_path),
        ),
        (
            "pdf_split",
            "[PDF ONLY] Splits PDF by ranges (e.g. [\"1-3\", \"5\"]).",
            lambda pdf_path, ranges, output_dir: _run_tool(PDFProcessor.split_pdf, "Split failed", pdf_path, ranges, output_dir),
        ),
        (
            "pdf_rotate_pages",
            "[PDF ONLY] Rotates pages by 90/180/270.",
            lambda pdf_path, rotation, pages=None: _run_tool(PDFProcessor.rotate_pages, "Rotation failed", pdf_path, rotation, pages),
        ),
        (
            "pdf_extract_images",
            "[PDF ONLY] Extracts images to ZIP. output_path is optional; defaults to <pdf_name>_images.zip next to the source file.",
            lambda pdf_path, output_path=None: _run_tool(PDFProcessor.extract_images, "Image extraction failed", pdf_path, output_path),
        ),
        (
            "pdf_from_images",
            "[PDF ONLY] Creates PDF from images. Reports how many images were actually added.",
            lambda image_paths, output_path: _run_tool(PDFProcessor.images_to_pdf, "Conversion failed", image_paths, output_path),
        ),
    ]

    for name, doc, runner in tool_specs:
        _register_mcp_tool(name, doc, runner)


_register_pdf_tools()


if __name__ == "__main__":
    try:
        FontManager.ensure_fonts()
        logger.info("Fonts loaded successfully.")
    except Exception as e:
        logger.error(f"Font initialization error: {e}")
        sys.exit(1)

    logger.info("Server started successfully and listening on stdio...")

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal (Ctrl+C). Shutting down gracefully...")
        sys.exit(0)
