CREATE TABLE IF NOT EXISTS kg_nodes (
  node_id TEXT PRIMARY KEY,
  node_type TEXT NOT NULL,
  label TEXT,
  properties JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kg_edges (
  edge_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
  target_id TEXT NOT NULL REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
  edge_type TEXT NOT NULL,
  doi TEXT,
  claim_id TEXT,
  confidence_score DOUBLE PRECISION,
  evidence_ids JSONB DEFAULT '[]'::jsonb,
  properties JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_type
ON kg_nodes (node_type);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_properties
ON kg_nodes USING GIN (properties jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_kg_edges_type
ON kg_edges (edge_type);

CREATE INDEX IF NOT EXISTS idx_kg_edges_source
ON kg_edges (source_id);

CREATE INDEX IF NOT EXISTS idx_kg_edges_target
ON kg_edges (target_id);

CREATE INDEX IF NOT EXISTS idx_kg_edges_doi_claim
ON kg_edges (doi, claim_id);

CREATE INDEX IF NOT EXISTS idx_kg_edges_properties
ON kg_edges USING GIN (properties jsonb_path_ops);
