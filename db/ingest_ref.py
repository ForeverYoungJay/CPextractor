def ingest_references(conn, paper_doi, extracted_json):
    for block, items_key in [
        ("plastic", "parameters"),
        ("elastic", "constants"),
    ]:
        items = extracted_json.get(
            f"{block}_parameters", {}
        ).get(items_key, [])

        for p in items:
            src = p.get("source", {})
            for c in src.get("citations", []):
                # references
                conn.execute(
                    """
                    INSERT INTO references (doi, title)
                    VALUES (%s, %s)
                    ON CONFLICT (doi) DO UPDATE
                    SET title = COALESCE(EXCLUDED.title, references.title);
                    """,
                    (c["doi"], c.get("title")),
                )

                # paper_references
                conn.execute(
                    """
                    INSERT INTO paper_references (paper_doi, ref_label, reference_doi)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING;
                    """,
                    (paper_doi, c["label"], c["doi"]),
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
                        p["symbol"],
                        block,
                        c["label"],
                        c["doi"],
                        src.get("type"),
                        src.get("calibration_method"),
                        src.get("validation_targets"),
                    ),
                )
