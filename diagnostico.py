"""Comprobacion rapida: que modo usa el programa y que apuntes estan indexados en Qdrant.

Uso (con el servidor PARADO):  python diagnostico.py
"""
import os
from collections import Counter

from app import llm, storage, vector_store  # importar llm carga tambien tu archivo .env

print("MODO_BUSQUEDA que ve el programa:", os.getenv("MODO_BUSQUEDA") or "(sin definir: usa 'significado')")
print("Qdrant:", "en la nube" if os.getenv("QDRANT_URL") else f"local, carpeta {vector_store._ruta_local()}")

documentos = storage.cargar()
vector_store.inicializar_coleccion()
puntos, _ = vector_store.obtener_cliente().scroll(
    collection_name=vector_store.COLLECTION_NAME, limit=100000, with_payload=True, with_vectors=False
)
por_documento = Counter(p.payload["documento_id"] for p in puntos)

print(f"\nApuntes en data/documentos.json: {len(documentos)}   |   Fragmentos en Qdrant: {len(puntos)}\n")
print(f"{'id':>4}  {'caracteres':>10}  {'en Qdrant':>9}  titulo")
sin_indexar = 0
for d in documentos:
    n = por_documento.get(d["id"], 0)
    sin_indexar += n == 0
    print(f"{d['id']:>4}  {len(d['contenido']):>10}  {n:>9}  {d['titulo'][:50]}{'   <-- SIN INDEXAR' if n == 0 else ''}")

huerfanos = set(por_documento) - {d["id"] for d in documentos}
if huerfanos:
    print(f"\nFragmentos en Qdrant de ids que ya no estan en el JSON: {sorted(huerfanos)}")
print(f"\nApuntes sin indexar: {sin_indexar}" + ("  ->  ejecuta: python -m app.indexar" if sin_indexar else "  (todo indexado)"))