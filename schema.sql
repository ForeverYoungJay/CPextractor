CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS papers (
  doi TEXT PRIMARY KEY,
  title TEXT,
  year INT,
  journal TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extractions (
  doi TEXT PRIMARY KEY REFERENCES papers(doi) ON DELETE CASCADE,
  extracted_json JSONB NOT NULL,
  model_select TEXT,
  model_extract TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id BIGSERIAL PRIMARY KEY,
  doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,
  source_type TEXT,
  source_name TEXT,
  text TEXT NOT NULL,
  metadata JSONB,
  embedding VECTOR(1536)   -- use 3072 if you pick embedding-3-large
);

CREATE INDEX IF NOT EXISTS idx_extractions_jsonb
ON extractions USING GIN (extracted_json jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
ON chunks USING GIN (text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id BIGSERIAL PRIMARY KEY,
  doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,

  model_select TEXT,
  model_extract TEXT,
  prompt_version TEXT,
  schema_version TEXT,
  extractor_version TEXT,

  llm_select_input_tokens INT,
  llm_select_output_tokens INT,
  llm_select_total_tokens INT,

  llm_extract_input_tokens INT,
  llm_extract_output_tokens INT,
  llm_extract_total_tokens INT,

  time_select_seconds DOUBLE PRECISION,
  time_extract_seconds DOUBLE PRECISION,
  time_total_seconds DOUBLE PRECISION,

  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_doi
ON pipeline_runs (doi);


CREATE TABLE IF NOT EXISTS "references" (
  ref_pk BIGSERIAL PRIMARY KEY,
  doi TEXT UNIQUE,
  title TEXT,
  journal TEXT,
  year INT
);

CREATE TABLE IF NOT EXISTS paper_references (
  paper_doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,
  ref_label TEXT,
  reference_doi TEXT REFERENCES "references"(doi),
  PRIMARY KEY (paper_doi, ref_label)
);

CREATE TABLE parameter_references (
  param_pk BIGSERIAL PRIMARY KEY,

  paper_doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,
  parameter_symbol TEXT,
  parameter_block TEXT,        -- "plastic" / "elastic"

  ref_label TEXT,
  reference_doi TEXT REFERENCES "references"(doi),

  source_type TEXT,            -- adopted / calibrated / original
  calibration_method TEXT,
  validation_targets TEXT
);

CREATE TABLE parameter_lineage (
  lineage_pk BIGSERIAL PRIMARY KEY,

  parameter_symbol TEXT,
  material TEXT,

  from_paper_doi TEXT,
  to_paper_doi TEXT,

  reference_doi TEXT,
  relationship TEXT,           -- adopted / recalibrated / modified
  notes TEXT
);
