def ingest_references(conn, paper_doi, extracted_json):
    # Keep inserts idempotent across reruns.
    conn.execute("DELETE FROM parameter_references WHERE paper_doi = %s;", (paper_doi,))

    for block, items_key in [
        ("plastic", "parameters"),
        ("elastic", "constants"),
    ]:
        items = extracted_json.get(
            f"{block}_parameters", {}
        ).get(items_key, [])

        for p in items:
            src = p.get("source", {})
            source_type = src.get("origin_type") or src.get("type")
            validation_targets = src.get("validation_targets")
            if isinstance(validation_targets, list):
                validation_targets = ",".join(str(x) for x in validation_targets)
            for c in src.get("citations", []):
                ref_doi = c.get("doi")
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
                    (ref_doi, c.get("title")),
                )

                # paper_references
                conn.execute(
                    """
                    INSERT INTO paper_references (paper_doi, ref_label, reference_doi)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING;
                    """,
                    (paper_doi, c.get("label"), ref_doi),
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
                        c.get("label"),
                        ref_doi,
                        source_type,
                        src.get("calibration_method"),
                        validation_targets,
                    ),
                )
