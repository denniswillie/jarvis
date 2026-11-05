"""The native artifact evaluator must reject incorrect files, not just find receipts."""
import importlib.util
import io
import unittest

from evals.second_brain.sandbox_artifact_probe import document_evidence


@unittest.skipUnless(importlib.util.find_spec("openpyxl"), "optional document evaluation dependency")
class SpreadsheetEvidenceTests(unittest.TestCase):
    @staticmethod
    def original(alpha=5, total="=SUM(B2:B3)"):
        from openpyxl import Workbook
        workbook, output = Workbook(), io.BytesIO()
        workbook.active.title = "Summary"
        for row in (("project", "hours"), ("alpha", alpha), ("beta", 5), ("total_hours", total)):
            workbook.active.append(row)
        workbook.save(output)
        workbook.close()
        return output.getvalue()

    def test_formula_and_source_values_are_distinct_from_recalculation_proof(self):
        evidence = document_evidence("summary.xlsx", self.original())
        self.assertEqual(evidence["expected_total"], 10)
        self.assertIsNone(evidence["cached_total"])
        self.assertFalse(evidence["formula_recalculation_verified"])
        self.assertFalse(evidence["layout_reviewed"])

    def test_wrong_values_static_total_and_wrong_formula_do_not_pass(self):
        for arguments in ({"alpha": 7}, {"total": 10}, {"total": "=B2*B3"}):
            with self.subTest(arguments=arguments), self.assertRaises(AssertionError):
                document_evidence("summary.xlsx", self.original(**arguments))


@unittest.skipUnless(importlib.util.find_spec("pypdf"), "optional document evaluation dependency")
class PdfEvidenceTests(unittest.TestCase):
    def test_openable_but_blank_pdf_is_not_content_acceptance(self):
        from pypdf import PdfWriter
        writer, output = PdfWriter(), io.BytesIO()
        writer.add_blank_page(width=200, height=200)
        writer.write(output)
        with self.assertRaises(AssertionError):
            document_evidence("summary.pdf", output.getvalue())


class DocumentEvidenceScopeTests(unittest.TestCase):
    def test_unknown_document_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            document_evidence("other.bin", b"not document evidence")
