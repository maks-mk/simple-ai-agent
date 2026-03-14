import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pdf_mcp" / "pdf_server.py"

_SPEC = importlib.util.spec_from_file_location("pdf_mcp_pdf_server", MODULE_PATH)
pdf_server = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(pdf_server)


class PDFMcpTests(unittest.TestCase):
    def test_tool_schema_uses_real_types(self):
        tools = pdf_server.mcp._tool_manager._tools

        extract_props = tools["pdf_extract_text"].parameters["properties"]
        self.assertEqual(extract_props["page_start"]["type"], "integer")
        self.assertEqual(extract_props["sort"]["type"], "boolean")

        merge_props = tools["pdf_merge"].parameters["properties"]
        self.assertEqual(merge_props["pdf_paths"]["type"], "array")
        self.assertEqual(merge_props["pdf_paths"]["items"]["type"], "string")

        split_props = tools["pdf_split"].parameters["properties"]
        self.assertEqual(split_props["ranges"]["type"], "array")
        self.assertEqual(split_props["ranges"]["items"]["type"], "string")

        rotate_props = tools["pdf_rotate_pages"].parameters["properties"]
        pages_schema = rotate_props["pages"]["anyOf"][0]
        self.assertEqual(pages_schema["type"], "array")
        self.assertEqual(pages_schema["items"]["type"], "integer")

    def test_inline_code_uses_selected_font_without_courier(self):
        generator = pdf_server.PDFGenerator()

        rendered = generator._markdown_to_reportlab("`код` и текст")

        self.assertIn('name="', rendered)
        self.assertIn(generator.fonts["mono"], rendered)
        self.assertIn("код", rendered)
        self.assertNotIn("Courier", rendered)

    def test_clean_text_preserves_typography(self):
        text = "«текст» №1 — тест…"

        self.assertEqual(pdf_server.PDFGenerator._clean_text(text), text)

    def test_create_pdf_smoke(self):
        generator = pdf_server.PDFGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"
            result = generator.create_pdf(
                output_path,
                "Отчёт",
                "# Отчёт\n\nТекст с `кодом` и «кавычками».",
            )

            self.assertIn(str(output_path), result)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

            with pdf_server.fitz.open(output_path) as doc:
                extracted = doc[0].get_text()

            self.assertIn("Отчёт", extracted)
            self.assertIn("Текст", extracted)
            self.assertIn("кодом", extracted)


if __name__ == "__main__":
    unittest.main()
