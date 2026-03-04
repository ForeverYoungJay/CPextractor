import os, re, json, glob
from typing import List, Dict, Any

# ---- OpenAI ----
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----------------------------
# Config
# ----------------------------
SECTION_KEYWORDS = [
    "material", "experimental", "method", "simulation", "model",
    "finite element", "fem", "numerical", "constitutive",
    "processing", "heat treatment", "rolling", "forging",
    "anneal", "solution", "aging", "quench", "cold", "hot",
    "setup", "procedure", "testing"
]

TABLE_KEYWORDS = [
    "composition", "chemical", "alloy", "wt", "weight percent",
    "parameter", "constant", "johnson", "cook", "jc",
    "model", "simulation", "boundary", "mesh", "element",
    "strain rate", "temperature"
]

MAX_SECTION_CHARS = 3500      # per section file
MAX_TABLE_CHARS = 2500        # per table file
MAX_TOTAL_CONTEXT_CHARS = 24000  # for minimal cost; adjust based on model

# ----------------------------
# Helpers
# ----------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

def matches_keywords(text: str, keywords: List[str]) -> int:
    """Return keyword hit score for relevance ranking."""
    t = normalize(text)
    return sum(1 for kw in keywords if kw in t)

def load_markdown_files(folder: str) -> List[Dict[str, Any]]:
    """Load markdown files with metadata."""
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    items = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        items.append({
            "path": path,
            "name": os.path.basename(path),
            "text": txt
        })
    return items

def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    return text[:max_chars] + ("\n...[TRUNCATED]..." if len(text) > max_chars else "")

def build_minimal_context(sections, tables, debug=True):
    """
    Select the most relevant sections/tables and build a compact context string.
    Also returns which files were selected.
    """
    ranked_sections = []
    for s in sections:
        score = matches_keywords(s["text"], SECTION_KEYWORDS)
        # always include abstract if present
        if score > 0 or "abstract" in s["name"].lower():
            ranked_sections.append((score, s))

    ranked_tables = []
    for t in tables:
        score = matches_keywords(t["text"], TABLE_KEYWORDS)
        if score > 0:
            ranked_tables.append((score, t))

    ranked_sections.sort(key=lambda x: x[0], reverse=True)
    ranked_tables.sort(key=lambda x: x[0], reverse=True)

    context_parts = []
    total_chars = 0

    selected_sections = []
    selected_tables = []

    # --- Add best sections ---
    for score, s in ranked_sections:
        part = f"\n\n=== SECTION: {s['name']} (score={score}) ===\n" + trim_text(s["text"], MAX_SECTION_CHARS)
        if total_chars + len(part) > MAX_TOTAL_CONTEXT_CHARS:
            break
        context_parts.append(part)
        total_chars += len(part)
        selected_sections.append({"name": s["name"], "score": score, "path": s["path"]})

    # --- Add best tables ---
    for score, t in ranked_tables:
        part = f"\n\n=== TABLE: {t['name']} (score={score}) ===\n" + trim_text(t["text"], MAX_TABLE_CHARS)
        if total_chars + len(part) > MAX_TOTAL_CONTEXT_CHARS:
            break
        context_parts.append(part)
        total_chars += len(part)
        selected_tables.append({"name": t["name"], "score": score, "path": t["path"]})

    selected = {
        "selected_sections": selected_sections,
        "selected_tables": selected_tables,
        "total_context_chars": total_chars
    }

    if debug:
        print("\n✅ Selected sections:")
        for item in selected_sections:
            print(f"  - {item['name']} (score={item['score']})")

        print("\n✅ Selected tables:")
        for item in selected_tables:
            print(f"  - {item['name']} (score={item['score']})")

        print(f"\n📦 Total context size: {total_chars} chars\n")

    return "\n".join(context_parts).strip(), selected


# ----------------------------
# LLM Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are a precise scientific information extraction engine.
Extract structured data about materials, composition, processing and simulation from a paper excerpt.

Rules:
- Return ONLY valid JSON.
- Use the exact schema given.
- If information is missing, use null or [].
- materials must always be an array (even if empty).
- composition entries: wt_percent must be a number OR the string "balance".
- For simulation.parameters, include only explicitly stated parameters (constants, coefficients, model inputs).
"""

USER_PROMPT_TEMPLATE = """
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

# ----------------------------
# LLM call
# ----------------------------
def extract_with_llm(context: str, model="gpt-4.1-mini") -> Dict[str, Any]:
    prompt = USER_PROMPT_TEMPLATE.format(context=context)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    return json.loads(resp.choices[0].message.content)

# ----------------------------
# Run extraction for one paper
# ----------------------------
def extract_from_paper_folder(paper_dir: str, out_json_path: str = None):
    sections_dir = os.path.join(paper_dir, "sections")
    tables_dir = os.path.join(paper_dir, "tables")

    sections = load_markdown_files(sections_dir) if os.path.exists(sections_dir) else []
    tables = load_markdown_files(tables_dir) if os.path.exists(tables_dir) else []

    context, selected = build_minimal_context(sections, tables, debug=True)

    # ✅ Save selected file list for transparency
    selected_log_path = os.path.join(paper_dir, "selected_context_files.json")
    with open(selected_log_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    if not context:
        print("⚠️ No relevant context found.")
        result = {"materials": []}
    else:
        result = extract_with_llm(context)

    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result, selected


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    paper_dir = "output/10.1016_j.jmrt.2025.11.003"   # <-- your safe_id folder
    out_json = os.path.join(paper_dir, "materials_extracted.json")

    data = extract_from_paper_folder(paper_dir, out_json_path=out_json)
    print(json.dumps(data, indent=2, ensure_ascii=False))
