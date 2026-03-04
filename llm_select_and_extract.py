import os, re, json, glob
from typing import List, Dict, Any

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----------------------------
# Config
# ----------------------------
MAX_SNIPPET_CHARS = 600       # show a small snippet for selection only
MAX_CONTEXT_CHARS = 24000     # extraction prompt context limit

# ----------------------------
# Helpers
# ----------------------------
def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    return text[:max_chars] + ("...[TRUNCATED]..." if len(text) > max_chars else "")

def load_md_files(folder: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    out = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        out.append({
            "name": os.path.basename(path),
            "path": path,
            "text": txt,
            "length": len(txt)
        })
    return out

# ----------------------------
# Stage 1: LLM selection
# ----------------------------
SELECTION_SYSTEM_PROMPT = """
You are an expert scientific document triage assistant.
Your job is to decide which sections and tables are needed to extract structured information about:
- material name, crystal structure
- composition (wt%)
- processing
- simulation (software, method, model type, parameters)

Return ONLY JSON with:
{
  "selected_sections": [...],
  "selected_tables": [...],
  "notes": "short explanation"
}

Rules:
- Choose the minimum needed files, but ensure all relevant info is covered.
- Output file names exactly as given.
- If nothing relevant exists, return empty arrays.
"""

SELECTION_USER_PROMPT_TEMPLATE = """
You are given a paper split into files.

Your task:
Select the minimum set of section files and table files needed to extract:
- materials, crystal structure, composition (wt%), processing
- simulation info: software, method, model type, parameters, conditions

Here are the available sections:
{sections_catalog}

Here are the available tables:
{tables_catalog}

Return only JSON.
"""

def build_catalog(files: List[Dict[str, Any]], kind="SECTION") -> str:
    """
    Build a short catalog listing: filename, length, snippet.
    """
    parts = []
    for f in files:
        snippet = trim_text(f["text"], MAX_SNIPPET_CHARS).replace("\n", " ")
        parts.append(
            f"- {f['name']} | len={f['length']} | snippet=\"{snippet}\""
        )
    return "\n".join(parts)

def llm_select_files(sections: List[Dict[str, Any]], tables: List[Dict[str, Any]], model="gpt-4.1-mini") -> Dict[str, Any]:
    sections_catalog = build_catalog(sections, "SECTION")
    tables_catalog = build_catalog(tables, "TABLE")

    prompt = SELECTION_USER_PROMPT_TEMPLATE.format(
        sections_catalog=sections_catalog,
        tables_catalog=tables_catalog
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )

    return json.loads(resp.choices[0].message.content)

# ----------------------------
# Stage 2: LLM Extraction
# ----------------------------
EXTRACT_SYSTEM_PROMPT = """
You are a precise scientific information extraction engine.
Extract structured data about materials, composition, processing and simulation from a paper excerpt.

Rules:
- Return ONLY valid JSON.
- Use the exact schema given.
- If information is missing, use null or [].
- materials must always be an array.
- composition entries: wt_percent must be a number OR the string "balance".
- simulation.parameters: only include explicitly stated parameters.
"""

EXTRACT_USER_PROMPT_TEMPLATE = """
Extract the following schema from the paper excerpt:

{{
  "materials": [
    {{
      "name": "string or null",
      "crystal_structure": "string or null",
      "composition": [
        {{"element": "string", "wt_percent": "number or 'balance'"}}
      ],
      "processing": "string or null",
      "simulation": {{
        "software": "string or null",
        "method": "string or null",
        "model_type": "string or null",
        "numerical_method": "string or null",
        "deformation_mode": "string or null",
        "strain_rate": "string or null",
        "total_strain": "string or null",
        "temperature": "string or null",
        "parameters": [
          {{
            "variable": "string",
            "description": "string or null",
            "unit": "string or null",
            "value": "number or string"
          }}
        ]
      }}
    }}
  ]
}}

Paper excerpt:
----------------
{context}
----------------

Return JSON only.
"""

def build_context(selected_sections, selected_tables):
    """
    Build context by concatenating selected section/table full content (truncated if needed).
    """
    parts = []
    total = 0

    for s in selected_sections:
        content = trim_text(s["text"], MAX_CONTEXT_CHARS)
        chunk = f"\n\n=== SECTION: {s['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        if total > MAX_CONTEXT_CHARS:
            break

    for t in selected_tables:
        content = trim_text(t["text"], MAX_CONTEXT_CHARS)
        chunk = f"\n\n=== TABLE: {t['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        if total > MAX_CONTEXT_CHARS:
            break

    return "\n".join(parts).strip()

def llm_extract(context: str, model="gpt-4.1-mini") -> Dict[str, Any]:
    prompt = EXTRACT_USER_PROMPT_TEMPLATE.format(context=context)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )

    return json.loads(resp.choices[0].message.content)

# ----------------------------
# Main runner
# ----------------------------
def run_pipeline(paper_dir: str, model_select="gpt-4.1-mini", model_extract="gpt-4.1-mini"):
    sections_dir = os.path.join(paper_dir, "sections")
    tables_dir = os.path.join(paper_dir, "tables")

    sections = load_md_files(sections_dir) if os.path.exists(sections_dir) else []
    tables = load_md_files(tables_dir) if os.path.exists(tables_dir) else []

    # Stage 1: select files
    selection = llm_select_files(sections, tables, model=model_select)

    selected_section_names = set(selection.get("selected_sections", []))
    selected_table_names = set(selection.get("selected_tables", []))

    selected_sections = [s for s in sections if s["name"] in selected_section_names]
    selected_tables = [t for t in tables if t["name"] in selected_table_names]

    # Save selection log
    with open(os.path.join(paper_dir, "llm_selected_files.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    # Stage 2: extraction
    context = build_context(selected_sections, selected_tables)

    extracted = llm_extract(context, model=model_extract)

    with open(os.path.join(paper_dir, "materials_extracted.json"), "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    return selection, extracted

# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    paper_dir = "output/10.1016_j.jmrt.2025.11.003"
    selection, extracted = run_pipeline(paper_dir)
    print("✅ LLM selection:", json.dumps(selection, indent=2, ensure_ascii=False))
    print("✅ Extracted JSON:", json.dumps(extracted, indent=2, ensure_ascii=False))
