# CPextractor Presentation Draft

## Slide 1. Title
- From Crystal Plasticity Papers to an Evidence-Grounded Database
- CPextractor is a literature-to-database pipeline for extracting, validating, and serving crystal plasticity parameters
- Core message: extraction, trust, and reuse are handled in one system

## Slide 2. Motivation
- Crystal plasticity parameters are spread across prose, tables, and supplementary material
- Scientific reuse needs normalization, provenance, and uncertainty, not only extraction
- Manual curation does not scale for database construction and retrieval workflows

## Slide 3. System Overview
- Inputs: Scopus, DOI lists, or local full-text folders
- Parsing: Elsevier XML into sections, tables, and references
- Extraction: two-stage LLM pipeline
- Postprocessing: deterministic normalization and condition binding
- Trust stack: evidence grounding plus multi-agent evaluation
- Outputs: database tables, review queue, chatbot, analytics

## Slide 4. Pipeline
- Acquire papers
- Parse full text
- Extract candidate parameters
- Refine with normalization and evidence grounding
- Gate and ingest based on quality signals

## Slide 5. Trust Stack
- Layer 1: rule validation
- Layer 2: evidence grounding
- Layer 3: evidence, normalization, consistency, and meta judges
- Layer 4: confidence fusion and quality tiering
- Key point: low-quality papers can be blocked without losing audit trace

## Slide 6. Data Model
- `parameters.registry` is the initial structured extraction target
- `evidence_objects` store reusable evidence spans and table-cell references
- `parameter_claims` are the minimal trusted unit for review and database use

## Slide 7. Database and Retrieval
- Structured tables support curated scientific storage
- Vector tables support semantic retrieval over both chunks and claims
- Hybrid chatbot retrieval combines SQL, chunk vectors, and parameter vectors
- Answers are constrained to cited evidence

## Slide 8. Evaluation
- Extraction correctness
- Judge correctness and calibration
- Database utility
- Key point: success is measured at the curation and retrieval level, not only field extraction

## Slide 9. Impact
- Searchable crystal plasticity parameter database
- Evidence-grounded RAG for scientific question answering
- Analyst workflows for comparing materials, phases, mechanisms, and conditions
- Two publication angles: database paper and information extraction paper

## Slide 10. Close
- CPextractor turns papers into trustworthy scientific infrastructure
- Best live demo: one DOI through parse, extract, ground, audit, ingest, and retrieve
- Best near-term benchmark: annotated claim-level evaluation with confidence calibration
