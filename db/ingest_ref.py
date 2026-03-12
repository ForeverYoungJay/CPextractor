from postprocess.param_iter import iter_parameter_items


def ingest_references(conn, paper_doi, extracted_json):
    # Keep inserts idempotent across reruns.
    conn.execute("DELETE FROM parameter_references WHERE paper_doi = %s;", (paper_doi,))

    # Canonical reference dictionary comes from top-level references.
    top_level_refs = extracted_json.get("references", []) or []
    ref_by_label = {}
    for r in top_level_refs:
        if not isinstance(r, dict):
            continue
        label = str(r.get("reference_id") or r.get("label") or "").strip()
        if not label:
            continue
        ref_by_label[label] = r

    for block, p in iter_parameter_items(extracted_json):
            src = p.get("source", {})
            source_type = src.get("origin_type") or src.get("type")
            validation_targets = None

            ref_ids = []
            ref_ids.extend(src.get("reference_ids", []) or [])
            ref_ids.extend(src.get("adopted_from_reference_ids", []) or [])
            ref_ids.extend(src.get("calibration_based_on_reference_ids", []) or [])

            seen_ids = set()
            dedup_ids = []
            for rid in ref_ids:
                label = str(rid or "").strip()
                if not label or label in seen_ids:
                    continue
                seen_ids.add(label)
                dedup_ids.append(label)

            for label in dedup_ids:
                c = ref_by_label.get(label, {})
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
                        src.get("calibration_method"),
                        validation_targets,
                    ),
                )
