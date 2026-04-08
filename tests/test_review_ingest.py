import unittest

from scripts.review.ingest_review_approved import _resolve_paper_doi


class ReviewIngestTests(unittest.TestCase):
    def test_prefers_source_document_doi_over_record_id(self):
        extracted = {
            "record_id": "internal-record-1",
            "source_document": {"doi": "10.1000/correct-doi"},
        }
        self.assertEqual(_resolve_paper_doi(extracted, "10.1000_fallback"), "10.1000/correct-doi")


if __name__ == "__main__":
    unittest.main()
