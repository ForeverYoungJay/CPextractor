import unittest

from bs4 import BeautifulSoup

from elsevier.fulltext_parser import extract_tables_from_xml, safe_table_label


class FulltextParserTableLabelTests(unittest.TestCase):
    def test_appendix_table_label_can_be_recovered_from_caption(self):
        soup = BeautifulSoup(
            """
            <root>
              <ce:table>
                <ce:caption>
                  <ce:simple-para>Table A1. Crystal plasticity parameters.</ce:simple-para>
                </ce:caption>
                <tgroup>
                  <tbody>
                    <row><entry>Symbol</entry><entry>Value</entry></row>
                    <row><entry>tau0</entry><entry>85 MPa</entry></row>
                  </tbody>
                </tgroup>
              </ce:table>
            </root>
            """,
            "xml",
        )

        tables = extract_tables_from_xml(soup)

        self.assertEqual(tables[0]["table_label"], "Table A1")
        self.assertEqual(safe_table_label(tables[0]["table_label"], tables[0]["table_index"]), "A1")

    def test_numeric_caption_without_label_keeps_index_fallback(self):
        soup = BeautifulSoup(
            """
            <root>
              <ce:table>
                <ce:caption>
                  <ce:simple-para>Table 1. Main-text parameters.</ce:simple-para>
                </ce:caption>
                <tgroup>
                  <tbody>
                    <row><entry>Symbol</entry><entry>Value</entry></row>
                  </tbody>
                </tgroup>
              </ce:table>
            </root>
            """,
            "xml",
        )

        tables = extract_tables_from_xml(soup)

        self.assertIsNone(tables[0]["table_label"])
        self.assertEqual(safe_table_label(tables[0]["table_label"], tables[0]["table_index"]), "001")


if __name__ == "__main__":
    unittest.main()
