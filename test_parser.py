from bs4 import BeautifulSoup
from elsevier.fulltext_parser import (
    build_refid_to_number,
    replace_crossrefs_with_numbers,
    compress_numeric_citation_groups,
    normalize_text,
)

xml_path = "data/fulltext/10.1016_j.mechmat.2025.105574/paper.xml"   # <-- your XML file

with open(xml_path, "r", encoding="utf-8") as f:
    xml = f.read()

soup = BeautifulSoup(xml, "xml")

# Build mapping from bibliography
refid_to_num = build_refid_to_number(soup)

# Test paragraphs
for p in soup.find_all(["ce:para", "p"]):
    p_copy = BeautifulSoup(str(p), "xml")
    p_tag = p_copy.find(["ce:para", "p"]) or p_copy

    replace_crossrefs_with_numbers(p_tag, refid_to_num)

    txt = normalize_text(p_tag.get_text(" ", strip=True))
    txt = compress_numeric_citation_groups(txt)

    if txt:
        print("----")
        print(txt)
