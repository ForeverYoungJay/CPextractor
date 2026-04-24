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
  content_hash TEXT,
  embedding_model TEXT,
  embedding VECTOR(1536)   -- use 3072 if you pick embedding-3-large
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doi_source_hash
ON chunks (doi, source_type, source_name, content_hash);

CREATE INDEX IF NOT EXISTS idx_extractions_jsonb
ON extractions USING GIN (extracted_json jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
ON chunks USING GIN (text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS parameter_vectors (
  vector_id BIGSERIAL PRIMARY KEY,
  doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,
  claim_id TEXT NOT NULL,
  material_id TEXT,
  material_name TEXT,
  process_state_id TEXT,
  process_state_name TEXT,
  sample_id TEXT,
  sample_label TEXT,
  condition_id TEXT,
  condition_label TEXT,
  canonical_name TEXT,
  symbol TEXT,
  domain TEXT,
  constituent_id TEXT,
  constituent_name TEXT,
  phase_id TEXT,
  phase_name TEXT,
  mechanism TEXT,
  family_id TEXT,
  family_name TEXT,
  model_id TEXT,
  branch_id TEXT,
  system_ids JSONB DEFAULT '[]'::jsonb,
  value_text TEXT,
  unit TEXT,
  origin_type TEXT,
  evidence_file TEXT,
  evidence_kind TEXT,
  evidence_snippet TEXT,
  retrieval_text TEXT NOT NULL,
  metadata JSONB DEFAULT '{}'::jsonb,
  content_hash TEXT,
  embedding_model TEXT,
  embedding VECTOR(1536)
);

ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS material_id TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS material_name TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS process_state_id TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS process_state_name TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS sample_id TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS sample_label TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS condition_id TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS condition_label TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS constituent_id TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS constituent_name TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS phase_name TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS model_id TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS branch_id TEXT;
ALTER TABLE parameter_vectors ADD COLUMN IF NOT EXISTS system_ids JSONB DEFAULT '[]'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS idx_parameter_vectors_doi_claim
ON parameter_vectors (doi, claim_id);

CREATE INDEX IF NOT EXISTS idx_parameter_vectors_embedding_hnsw
ON parameter_vectors USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_parameter_vectors_meta
ON parameter_vectors USING GIN (metadata jsonb_path_ops);

CREATE TABLE IF NOT EXISTS table_row_vectors (
  row_vector_id BIGSERIAL PRIMARY KEY,
  doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,
  table_file TEXT,
  row_key TEXT,
  row_text TEXT NOT NULL,
  metadata JSONB DEFAULT '{}'::jsonb,
  content_hash TEXT,
  embedding_model TEXT,
  embedding VECTOR(1536)
);

ALTER TABLE table_row_vectors ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE table_row_vectors ADD COLUMN IF NOT EXISTS embedding_model TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_table_row_vectors_doi_row
ON table_row_vectors (doi, table_file, row_key);

CREATE INDEX IF NOT EXISTS idx_table_row_vectors_embedding_hnsw
ON table_row_vectors USING hnsw (embedding vector_cosine_ops);

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

  llm_evaluate_input_tokens INT,
  llm_evaluate_output_tokens INT,
  llm_evaluate_total_tokens INT,

  time_select_seconds DOUBLE PRECISION,
  time_extract_seconds DOUBLE PRECISION,
  time_evaluate_seconds DOUBLE PRECISION,
  time_total_seconds DOUBLE PRECISION,

  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS prompt_version TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS schema_version TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS extractor_version TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_select_input_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_select_output_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_select_total_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_extract_input_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_extract_output_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_extract_total_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_evaluate_input_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_evaluate_output_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS llm_evaluate_total_tokens INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS time_select_seconds DOUBLE PRECISION;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS time_extract_seconds DOUBLE PRECISION;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS time_evaluate_seconds DOUBLE PRECISION;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS time_total_seconds DOUBLE PRECISION;

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

CREATE TABLE IF NOT EXISTS parameter_references (
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

CREATE TABLE IF NOT EXISTS parameter_lineage (
  lineage_pk BIGSERIAL PRIMARY KEY,

  parameter_symbol TEXT,
  material TEXT,

  from_paper_doi TEXT,
  to_paper_doi TEXT,

  reference_doi TEXT,
  relationship TEXT,           -- adopted / recalibrated / modified
  notes TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
  eval_run_pk BIGSERIAL PRIMARY KEY,
  doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,
  model_evaluate TEXT,
  verdict TEXT,
  document_confidence TEXT,
  document_confidence_score DOUBLE PRECISION,
  quality_tier TEXT,
  review_recommended BOOLEAN,
  llm_evaluate_input_tokens INT,
  llm_evaluate_output_tokens INT,
  llm_evaluate_total_tokens INT,
  time_evaluate_seconds DOUBLE PRECISION,
  evidence_judge_input_tokens INT,
  evidence_judge_output_tokens INT,
  evidence_judge_total_tokens INT,
  evidence_judge_time_seconds DOUBLE PRECISION,
  normalization_judge_input_tokens INT,
  normalization_judge_output_tokens INT,
  normalization_judge_total_tokens INT,
  normalization_judge_time_seconds DOUBLE PRECISION,
  consistency_judge_input_tokens INT,
  consistency_judge_output_tokens INT,
  consistency_judge_total_tokens INT,
  consistency_judge_time_seconds DOUBLE PRECISION,
  meta_judge_input_tokens INT,
  meta_judge_output_tokens INT,
  meta_judge_total_tokens INT,
  meta_judge_time_seconds DOUBLE PRECISION,
  evaluation_json JSONB,
  confidence_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS model_evaluate TEXT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS verdict TEXT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS document_confidence TEXT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS document_confidence_score DOUBLE PRECISION;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS quality_tier TEXT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS review_recommended BOOLEAN;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS llm_evaluate_input_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS llm_evaluate_output_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS llm_evaluate_total_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS time_evaluate_seconds DOUBLE PRECISION;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS evidence_judge_input_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS evidence_judge_output_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS evidence_judge_total_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS evidence_judge_time_seconds DOUBLE PRECISION;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS normalization_judge_input_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS normalization_judge_output_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS normalization_judge_total_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS normalization_judge_time_seconds DOUBLE PRECISION;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS consistency_judge_input_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS consistency_judge_output_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS consistency_judge_total_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS consistency_judge_time_seconds DOUBLE PRECISION;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS meta_judge_input_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS meta_judge_output_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS meta_judge_total_tokens INT;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS meta_judge_time_seconds DOUBLE PRECISION;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS evaluation_json JSONB;
ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS confidence_json JSONB;

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_doi
ON evaluation_runs (doi);

CREATE TABLE IF NOT EXISTS parameter_audits (
  audit_pk BIGSERIAL PRIMARY KEY,
  doi TEXT REFERENCES papers(doi) ON DELETE CASCADE,
  location TEXT,
  canonical_name TEXT,
  symbol TEXT,
  verdict TEXT,
  supportiveness TEXT,
  exactness TEXT,
  normalization_correctness TEXT,
  completeness TEXT,
  provenance_quality TEXT,
  confidence TEXT,
  error_types TEXT,
  uncertainty_types TEXT,
  audit_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_parameter_audits_doi
ON parameter_audits (doi);

CREATE OR REPLACE VIEW evaluation_paper_summary AS
SELECT
  p.doi,
  p.title,
  p.year,
  p.journal,
  e.model_select,
  e.model_extract,
  e.llm_select_total_tokens AS pipeline_llm_select_total_tokens,
  e.llm_extract_total_tokens AS pipeline_llm_extract_total_tokens,
  e.llm_evaluate_total_tokens AS pipeline_llm_evaluate_total_tokens,
  e.time_select_seconds AS pipeline_time_select_seconds,
  e.time_extract_seconds AS pipeline_time_extract_seconds,
  e.time_evaluate_seconds AS pipeline_time_evaluate_seconds,
  e.time_total_seconds AS pipeline_time_total_seconds,
  er.model_evaluate,
  er.verdict,
  er.document_confidence,
  er.document_confidence_score,
  er.quality_tier,
  er.review_recommended,
  er.llm_evaluate_input_tokens AS evaluation_llm_evaluate_input_tokens,
  er.llm_evaluate_output_tokens AS evaluation_llm_evaluate_output_tokens,
  er.llm_evaluate_total_tokens AS evaluation_llm_evaluate_total_tokens,
  er.time_evaluate_seconds AS evaluation_time_evaluate_seconds,
  COALESCE(
    (
      SELECT m.item->>'name'
      FROM jsonb_array_elements(COALESCE(ex.extracted_json->'materials', '[]'::jsonb)) AS m(item)
      LIMIT 1
    ),
    ex.extracted_json->'material'->>'name'
  ) AS material_name,
  COALESCE(
    (
      SELECT m.item->>'framework'
      FROM jsonb_array_elements(COALESCE(ex.extracted_json->'models', '[]'::jsonb)) AS m(item)
      LIMIT 1
    ),
    ex.extracted_json->'constitutive_model'->>'framework'
  ) AS framework
FROM papers p
LEFT JOIN extractions ex ON ex.doi = p.doi
LEFT JOIN pipeline_runs e ON e.doi = p.doi
LEFT JOIN evaluation_runs er ON er.doi = p.doi;
