import os
from typing import Any

import requests
from openai import OpenAI

# === Crystal Plasticity Parameter Extraction Prompt ===
PROMPT_CRYSTAL_PLASTICITY = """
You are a materials science assistant specialized in extracting structured data from academic papers about crystal plasticity simulations.

Your goal is to extract only the information relevant for building a **parameter-level materials database**.  
Ignore any text not related to material composition, processing, crystal plasticity modeling, or model parameters.

Extract exactly the following categories:

1. **Material information**
   - name  
   - crystal structure (e.g., BCC/FCC/HCP)  
   - chemical composition (element + wt%)  
   - processing / initial condition (e.g., annealed, cold-rolled)

2. **Simulation setup**
   - software or code (e.g., DAMASK, Abaqus, custom code)
   - method (e.g., crystal plasticity FEM, FFT-based solver)
   - model type (e.g., dislocation-density-based, viscoplastic)
   - numerical method (e.g., FFT, FEM)
   - deformation mode (e.g., tension, plane-strain compression, rolling)
   - strain rate
   - total strain
   - temperature (if given)

3. **Model parameters**
   Extract all parameter tables or parameter-like quantities including:
   - variable name  
   - description  
   - unit  
   - numerical value  

4. **If multiple materials appear**, output them as separate entries in "materials".

5. If a field is not present, return null or an empty list.

Follow this exact JSON structure:

{
  "materials": [
    {
      "name": "string or null",
      "crystal_structure": "string or null",
      "composition": [
        {"element": "string", "wt_percent": "number or 'balance'"}
      ],
      "processing": "string or null",
      "simulation": {
        "software": "string or null",
        "method": "string or null",
        "model_type": "string or null",
        "numerical_method": "string or null",
        "deformation_mode": "string or null",
        "strain_rate": "string or null",
        "total_strain": "string or null",
        "temperature": "string or null",
        "parameters": [
          {
            "variable": "string",
            "description": "string or null",
            "unit": "string or null",
            "value": "number or string"
          }
        ]
      }
    }
  ]
}

Output only valid JSON — no Markdown, no code fences, no extra commentary.
"""

BASE_URL = "https://litellm.yibozhang.me/"
MODEL_NAME = "gpt-5"
MARKDOWN_ENDPOINT = "http://tiger:7778/el"


def fetch_paper_markdown(doi: str) -> str:
    """Retrieve the markdown representation of the paper for the given DOI."""
    params = {
        "doi": doi,
    }
    response = requests.get(MARKDOWN_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()

    payload: Any
    try:
        payload = response.json()
    except ValueError:
        return response.text

    # The endpoint is expected to return markdown content; fall back to raw text if needed.
    if isinstance(payload, dict):
        if "markdown" in payload and payload["markdown"]:
            return payload["markdown"]
        if "content" in payload and payload["content"]:
            return payload["content"]

    return str(payload)


def run_prompt(paper_text: str) -> str:
    """Send the prompt and paper text to the model and return the JSON response."""
    api_key = os.environ.get("LITELLM_API_KEY")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": PROMPT_CRYSTAL_PLASTICITY},
            {
                "role": "user",
                "content": "Here is the text from a crystal plasticity simulation paper: "
                f"{paper_text}",
            },
        ],
    )
    return completion.choices[0].message.content


def main() -> None:
    doi = os.environ.get("CRYSTAL_PLASTICITY_DOI", "10.1016/j.actamat.2022.118167")
    paper_text = fetch_paper_markdown(doi)
    result = run_prompt(paper_text)
    print(result)


if __name__ == "__main__":
    main()