from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from db.pg import connect_pg
from kg.builder import build_graph_from_root
from kg.db import apply_schema, sync_graph_to_postgres


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and sync the CP knowledge graph to PostgreSQL.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default="", help="Defaults to paths.fulltext from config.")
    ap.add_argument("--source-name", default="materials_extracted.json")
    ap.add_argument("--clear-existing", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    db = cfg["db"]
    root = args.root or cfg["paths"]["fulltext"]
    graph = build_graph_from_root(root, args.source_name)

    with connect_pg(
        host=db["host"],
        port=int(db["port"]),
        dbname=db["name"],
        user=db["user"],
        password=db["password"],
    ) as conn:
        apply_schema(conn)
        sync_graph_to_postgres(conn, graph, clear_existing=args.clear_existing)
        conn.commit()

    print(f"Synced {len(graph['nodes'])} KG nodes and {len(graph['edges'])} KG edges from {root}")


if __name__ == "__main__":
    main()
