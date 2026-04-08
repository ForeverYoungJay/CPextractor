from postprocess.param_iter import iter_parameter_items
from postprocess.record_links import resolve_provenance_record


def ingest_references(conn, paper_doi, extracted_json):
    # Keep inserts idempotent across reruns.
    conn.execute("DELETE FROM parameter_references WHERE paper_doi = %s;", (paper_doi,))
    conn.execute("DELETE FROM paper_references WHERE paper_doi = %s;", (paper_doi,))

    for block, p in iter_parameter_items(extracted_json):
            src = p.get("source", {})
            provenance = resolve_provenance_record(extracted_json, src if isinstance(src, dict) else {})
            source_type = provenance.get("origin_type") or (src.get("type") if isinstance(src, dict) else None)
            validation_targets = None

            refs = []
            refs.extend(provenance.get("references", []) or [])
            refs.extend(provenance.get("adopted_from_references", []) or [])
            refs.extend(provenance.get("calibration_based_on_references", []) or [])

            seen_ids = set()
            dedup_refs = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                label = str(ref.get("reference_id") or ref.get("label") or "").strip()
                if not label or label in seen_ids:
                    continue
                seen_ids.add(label)
                dedup_refs.append((label, ref))

            for label, c in dedup_refs:
                ref_doi = str(c.get("doi") or "").strip()
                if not ref_doi:
                    continue

                # references
                conn.execute(
                    """
                    INSERT INTO "references" (doi, title)
                    VALUES (%s, %s)
                    ON CONFLICT (doi) DO UPDATE
                    SET title = COALESCE(EXCLUDED.title, "references".title);
                    """,
                    (ref_doi, c.get("title") or c.get("citation")),
                )

                # paper_references
                conn.execute(
                    """
                    INSERT INTO paper_references (paper_doi, ref_label, reference_doi)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING;
                    """,
                    (paper_doi, label, ref_doi),
                )

                # parameter_references
                conn.execute(
                    """
                    INSERT INTO parameter_references (
                      paper_doi,
                      parameter_symbol,
                      parameter_block,
                      ref_label,
                      reference_doi,
                      source_type,
                      calibration_method,
                      validation_targets
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
                    """,
                    (
                        paper_doi,
                        p.get("symbol"),
                        block,
                        label,
                        ref_doi,
                        source_type,
                        provenance.get("calibration_method"),
                        validation_targets,
                    ),
                )
