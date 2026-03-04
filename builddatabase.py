import psycopg2, json

conn = psycopg2.connect("dbname=cpdb user=postgres password=xxx")
cur = conn.cursor()

with open("materials_extracted.json") as f:
    data = json.load(f)

# Example: insert material
mat = data["material"]
cur.execute("""
INSERT INTO materials (name, crystal_structure, phase)
VALUES (%s, %s, %s)
RETURNING material_id
""", (mat["name"], mat["crystal_structure"], mat["phase"]))
material_id = cur.fetchone()[0]

conn.commit()
