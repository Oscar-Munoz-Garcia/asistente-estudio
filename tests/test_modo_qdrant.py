import pytest
from fastapi.testclient import TestClient

from app import embeddings, indexar, llm, qa, storage
from app.main import app

DOCS = [
    {"id": 1, "titulo": "Redes", "asignatura": "Redes", "contenido": "Una red conecta dispositivos."},
    {"id": 2, "titulo": "Matrices", "asignatura": "Matematicas", "contenido": "Una matriz es una tabla de numeros."},
]


def test_modo_qdrant_busca_en_la_base_vectorial(qdrant_memoria, embeber_384):
    for d in DOCS:
        indexar.indexar_documento(d, embeber_384)
    # Pasamos documentos=[]: en modo qdrant no se lee el JSON
    r = qa.responder_pregunta("ordenadores", [], embeber_384, lambda s, m: "Respuesta [1]", modo="qdrant")
    assert r["respuesta"] == "Respuesta [1]"
    assert r["fuentes"][0]["documento_id"] == 1


def test_modo_qdrant_sin_fragmentos_relevantes_no_llama_al_modelo(qdrant_memoria, embeber_384):
    for d in DOCS:
        indexar.indexar_documento(d, embeber_384)
    llamadas = []
    r = qa.responder_pregunta("tortilla", [], embeber_384, lambda s, m: llamadas.append(1), modo="qdrant")
    assert r["respuesta"] == qa.SIN_RESPUESTA
    assert llamadas == []


def test_endpoint_preguntar_en_modo_qdrant(qdrant_memoria, embeber_384, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUTA", tmp_path / "documentos.json")
    monkeypatch.setattr(embeddings, "embeber", embeber_384)
    monkeypatch.setattr(llm, "responder", lambda sistema, mensaje: "Respuesta simulada")
    monkeypatch.setenv("MODO_BUSQUEDA", "qdrant")
    for d in DOCS:
        indexar.indexar_documento(d, embeber_384)
    r = TestClient(app).post("/preguntar", json={"pregunta": "ordenadores"})
    assert r.status_code == 200
    assert r.json()["fuentes"][0]["documento_id"] == 1