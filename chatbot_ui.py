#streamlit run chatbot_ui.py --server.port 8501 --server.address 127.0.0.1
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from chatbot import connect_db, load_config, run_one
from openai import OpenAI


PAGE_TITLE = "CPextractor Chatbot"
EMBEDDING_CANDIDATES = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]
LLM_CANDIDATES = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
]


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    sanitize_state()


def sanitize_state() -> None:
    cleaned: List[Dict[str, Any]] = []
    for item in st.session_state.get("messages", []):
        role = item.get("role")
        if role == "user":
            content = (item.get("content") or "").strip()
            if not content:
                continue
            cleaned.append({"role": "user", "content": content})
            continue
        if role == "assistant":
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            cleaned.append({"role": "assistant", "result": result})
    st.session_state.messages = cleaned


def app_style() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-a: #f6efe6;
  --bg-b: #dbe8f4;
  --ink: #1f2933;
  --muted: #5f6c7a;
  --card: rgba(255, 255, 255, 0.72);
  --line: rgba(31, 41, 51, 0.12);
  --ok: #1f8f63;
  --mid: #d99021;
  --low: #b64a42;
}

.stApp {
  background:
    radial-gradient(900px 360px at 8% -10%, rgba(210, 229, 244, 0.9), rgba(210, 229, 244, 0)),
    radial-gradient(700px 280px at 100% 0%, rgba(248, 226, 202, 0.85), rgba(248, 226, 202, 0)),
    linear-gradient(145deg, var(--bg-a), var(--bg-b));
  color: var(--ink);
}

html, body, [class*="css"] {
  font-family: "Space Grotesk", sans-serif;
}

.hero {
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 18px;
  padding: 1.1rem 1.2rem;
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
  animation: rise 0.45s ease-out;
}

.hero h1 {
  margin: 0 0 0.3rem 0;
  font-size: 1.55rem;
  letter-spacing: 0.2px;
}

.hero p {
  margin: 0;
  color: var(--muted);
  font-size: 0.95rem;
}

.answer-card {
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.75);
  border-radius: 14px;
  padding: 0.9rem;
}

.mono {
  font-family: "IBM Plex Mono", monospace;
}

.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 0.2rem 0.6rem;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.high {
  color: var(--ok);
  background: rgba(31, 143, 99, 0.12);
  border-color: rgba(31, 143, 99, 0.26);
}

.medium {
  color: var(--mid);
  background: rgba(217, 144, 33, 0.14);
  border-color: rgba(217, 144, 33, 0.3);
}

.low {
  color: var(--low);
  background: rgba(182, 74, 66, 0.14);
  border-color: rgba(182, 74, 66, 0.3);
}

@keyframes rise {
  from { transform: translateY(8px); opacity: 0; }
  to   { transform: translateY(0); opacity: 1; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def confidence_badge(conf: str) -> str:
    c = (conf or "low").strip().lower()
    if c not in {"high", "medium", "low"}:
        c = "low"
    return f'<span class="badge {c}">confidence: {c}</span>'


def render_markdown_with_latex(text: str) -> None:
    if not text:
        return

    pattern = r"(\$\$.*?\$\$|\\\[.*?\\\])"
    parts = re.split(pattern, text, flags=re.DOTALL)
    for part in parts:
        if not part:
            continue
        block = part.strip()
        if not block:
            continue
        if block.startswith("$$") and block.endswith("$$"):
            st.latex(block[2:-2].strip())
            continue
        if block.startswith(r"\[") and block.endswith(r"\]"):
            st.latex(block[2:-2].strip())
            continue
        st.markdown(part)


def render_answer(result: Dict[str, Any]) -> None:
    answer = (
        result.get("answer")
        or result.get("final_answer")
        or result.get("response")
        or result.get("output")
        or ""
    )
    conf = result.get("confidence", "low")
    ev = result.get("evidence_ids", [])
    claims = result.get("claims", [])
    gaps = result.get("gaps", [])
    retrieval = result.get("retrieval", {})

    st.markdown(confidence_badge(conf), unsafe_allow_html=True)
    if answer:
        render_markdown_with_latex(answer)
    else:
        st.warning("No `answer` field returned by model. Raw payload shown below.")
        st.json(result)
    if ev:
        st.caption("Evidence IDs: " + ", ".join(ev))
    st.markdown("</div>", unsafe_allow_html=True)

    if claims:
        st.subheader("Claims")
        st.dataframe(claims, width="stretch")

    if gaps:
        st.subheader("Gaps")
        for g in gaps:
            st.write(f"- {g}")

    with st.expander("Retrieval Evidence", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Chunks", "Parameter Vectors", "Structured", "Equations"])
        with tab1:
            chunk_hits = retrieval.get("chunk_hits", [])
            if not chunk_hits:
                st.info("No chunk hits.")
            else:
                for i, h in enumerate(chunk_hits, start=1):
                    st.markdown(
                        f"**C{i}** | DOI: `{h.get('doi')}` | source: `{h.get('source_type')}:{h.get('source_name')}` | score: `{h.get('score')}`"
                    )
                    st.code(h.get("snippet", ""), language="text")
        with tab2:
            parameter_hits = retrieval.get("parameter_hits", [])
            if not parameter_hits:
                st.info("No parameter vector hits.")
            else:
                st.dataframe(parameter_hits, width="stretch")
        with tab3:
            structured_hits = retrieval.get("structured_hits", [])
            if not structured_hits:
                st.info("No structured hits.")
            else:
                st.dataframe(structured_hits, width="stretch")
        with tab4:
            equation_hits = retrieval.get("equation_hits", [])
            if not equation_hits:
                st.info("No equation hits.")
            else:
                for i, h in enumerate(equation_hits, start=1):
                    st.markdown(
                        f"**Q{i}** | DOI: `{h.get('doi')}` | label: `{h.get('label')}` | "
                        f"kind: `{h.get('kind')}` | section: `{h.get('section_title')}` | score: `{h.get('score')}`"
                    )
                    if h.get("latex"):
                        st.latex(h.get("latex"))
                    if h.get("text"):
                        st.caption(h.get("text"))


def run_query(
    config_path: str,
    query: str,
    embedding_model: str,
    llm_model: str,
    k_chunks: int,
    k_params: int,
    k_struct: int,
    retrieve_only: bool,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    db_cfg = cfg.get("db", {})
    rag_cfg = cfg.get("rag", {})
    llm_cfg = cfg.get("llm", {})

    emb_model = embedding_model or rag_cfg.get("embedding_model", "text-embedding-3-small")
    gen_model = llm_model or llm_cfg.get("model_extract", "gpt-4.1-mini")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = None
    if not retrieve_only:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
    elif api_key:
        client = OpenAI(api_key=api_key)

    conn = connect_db(db_cfg)
    try:
        return run_one(
            conn=conn,
            cfg=cfg,
            client=client,
            query=query,
            embedding_model=emb_model,
            llm_model=gen_model,
            k_chunks=k_chunks,
            k_params=k_params,
            k_struct=k_struct,
            retrieve_only=retrieve_only,
        )
    finally:
        conn.close()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=":material/science:", layout="wide")
    init_state()
    app_style()

    st.markdown(
        """
<div class="hero">
  <h1>CP Chatbot Workspace</h1>
  <p>Hybrid retrieval over evidence chunks and structured CP parameters, with explicit evidence tracking.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    cfg = load_config("config.yaml")
    cfg_rag = cfg.get("rag", {})
    cfg_llm = cfg.get("llm", {})
    default_embedding = cfg_rag.get("embedding_model", "text-embedding-3-small")
    default_llm = cfg_llm.get("model_extract", "gpt-4.1-mini")

    embedding_options = list(dict.fromkeys([default_embedding] + EMBEDDING_CANDIDATES + ["Custom..."]))
    llm_options = list(dict.fromkeys([default_llm] + LLM_CANDIDATES + ["Custom..."]))

    with st.sidebar:
        st.header("Runtime")
        config_path = st.text_input("Config path", value="config.yaml")

        emb_choice = st.selectbox("Embedding model", options=embedding_options, index=0)
        if emb_choice == "Custom...":
            embedding_model = st.text_input("Custom embedding model", value=default_embedding)
        else:
            embedding_model = emb_choice

        llm_choice = st.selectbox("LLM model", options=llm_options, index=0)
        if llm_choice == "Custom...":
            llm_model = st.text_input("Custom LLM model", value=default_llm)
        else:
            llm_model = llm_choice

        k_chunks = st.slider("Top-K chunks", min_value=3, max_value=30, value=8)
        k_params = st.slider("Top-K parameter vectors", min_value=3, max_value=40, value=12)
        k_struct = st.slider("Top-K structured", min_value=3, max_value=30, value=8)
        retrieve_only = st.toggle("Retrieve only (no synthesis)", value=False)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear chat", width="stretch"):
                st.session_state.messages = []
                st.session_state.history = []
                st.rerun()
        with col_b:
            payload = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Export JSON",
                data=payload.encode("utf-8"),
                file_name=f"chat_history_{ts}.json",
                mime="application/json",
                width="stretch",
            )

        st.caption("Requires PostgreSQL + pgvector and OPENAI_API_KEY for synthesis mode.")

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            if item["role"] == "assistant":
                render_answer(item["result"])
            else:
                content = (item.get("content") or "").strip()
                if content:
                    st.markdown(content)

    prompt = st.chat_input("Ask about CP parameters, calibration, slip systems, or paper evidence...")
    if prompt:
        prompt = prompt.strip()
        if not prompt:
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running hybrid retrieval and synthesis..."):
                try:
                    result = run_query(
                        config_path=config_path,
                        query=prompt,
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        k_chunks=k_chunks,
                        k_params=k_params,
                        k_struct=k_struct,
                        retrieve_only=retrieve_only,
                    )
                except Exception as e:
                    st.error(str(e))
                    return

            render_answer(result)
            st.session_state.messages.append({"role": "assistant", "result": result})
            st.session_state.history.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "query": prompt,
                    "result": result,
                }
            )


if __name__ == "__main__":
    main()
