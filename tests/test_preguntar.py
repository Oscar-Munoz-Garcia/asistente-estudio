import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import embeddings, llm, qa, rag, storage
from app.main import app

# Modelo de embeddings "de mentira" (igual que en test_semantica.py)
CONCEPTOS = {"red": 0, "redes": 0, "dispositivos": 1, "ordenadores": 1,
             "matriz": 2, "matrices": 2, "tabla": 2}


def embeber_falso(textos):
    vectores = np.zeros((len(textos), 4))
    for i, texto in enumerate(textos):
        for palabra in rag.normalizar(texto):
            if palabra in CONCEPTOS:
                vectores[i, CONCEPTOS[palabra]] += 1
        vectores[i, 3] = 0.01
    return vectores


DOCS = [
    {"id": 1, "titulo": "Redes", "asignatura": "Redes", "contenido": "Una red conecta dispositivos."},
    {"id": 2, "titulo": "Matrices", "asignatura": "Matematicas", "contenido": "Una matriz es una tabla de numeros."},
]


def test_la_pregunta_lleva_el_contexto_relevante_al_modelo():
    recibido = {}

    def generar_falso(sistema, mensaje):
        recibido["sistema"], recibido["mensaje"] = sistema, mensaje
        return "Respuesta de prueba [1]"

    r = qa.responder_pregunta("ordenadores", DOCS, embeber_falso, generar_falso)
    assert r["respuesta"] == "Respuesta de prueba [1]"
    assert r["fuentes"][0]["documento_id"] == 1
    assert "Una red conecta dispositivos" in recibido["mensaje"]
    assert "no inventes" in recibido["sistema"]


def test_sin_fragmentos_relevantes_no_se_llama_al_modelo():
    llamadas = []

    def generar_falso(sistema, mensaje):
        llamadas.append(1)
        return "no deberia llegar aqui"

    r = qa.responder_pregunta("receta de tortilla", DOCS, embeber_falso, generar_falso)
    assert r["respuesta"] == qa.SIN_RESPUESTA
    assert r["fuentes"] == []
    assert llamadas == []


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUTA", tmp_path / "documentos.json")
    monkeypatch.setattr(embeddings, "embeber", embeber_falso)
    monkeypatch.setattr(llm, "responder", lambda sistema, mensaje: "Respuesta simulada")
    return TestClient(app)


def test_endpoint_preguntar(client):
    r = client.post("/preguntar", json={"pregunta": "ordenadores"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["respuesta"] == "Respuesta simulada"
    assert cuerpo["fuentes"][0]["documento_id"] == 1
    assert "texto" not in cuerpo["fuentes"][0]  # el modelo de respuesta filtra campos


def test_endpoint_pregunta_demasiado_corta_da_422(client):
    assert client.post("/preguntar", json={"pregunta": "a"}).status_code == 422


def test_endpoint_da_503_si_falla_el_modelo(client, monkeypatch):
    def falla(sistema, mensaje):
        raise llm.ErrorLLM("sin conexion")

    monkeypatch.setattr(llm, "responder", falla)
    r = client.post("/preguntar", json={"pregunta": "ordenadores"})
    assert r.status_code == 503
    assert "sin conexion" in r.json()["detail"]


def test_modo_palabras_no_necesita_embeddings():
    def embeber_prohibido(textos):
        raise AssertionError("En modo palabras no se debe cargar el modelo de embeddings")

    r = qa.responder_pregunta(
        "dispositivos", DOCS, embeber_prohibido, lambda s, m: "Respuesta [1]", modo="palabras"
    )
    assert r["respuesta"] == "Respuesta [1]"
    assert r["fuentes"][0]["documento_id"] == 1


def test_modo_palabras_sin_coincidencias_no_llama_al_modelo():
    llamadas = []
    r = qa.responder_pregunta(
        "tortilla", DOCS, embeber_falso, lambda s, m: llamadas.append(1), modo="palabras"
    )
    assert r["respuesta"] == qa.SIN_RESPUESTA
    assert llamadas == []


def test_modo_desconocido_da_error_claro():
    with pytest.raises(ValueError):
        qa.responder_pregunta("hola", DOCS, embeber_falso, lambda s, m: "x", modo="magia")


def test_endpoint_usa_modo_palabras_si_se_configura(client, monkeypatch):
    monkeypatch.setenv("MODO_BUSQUEDA", "palabras")
    r = client.post("/preguntar", json={"pregunta": "dispositivos"})
    assert r.status_code == 200
    assert r.json()["fuentes"][0]["documento_id"] == 1