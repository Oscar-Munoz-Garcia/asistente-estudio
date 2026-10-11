import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import embeddings, rag, semantica, storage
from app.main import app

# Modelo "de mentira": palabras con significado parecido comparten dimension.
# Asi probamos la logica sin descargar el modelo real.
CONCEPTOS = {"red": 0, "redes": 0, "dispositivos": 1, "ordenadores": 1,
             "matriz": 2, "matrices": 2, "tabla": 2}


def embeber_falso(textos):
    vectores = np.zeros((len(textos), 4))
    for i, texto in enumerate(textos):
        for palabra in rag.normalizar(texto):
            if palabra in CONCEPTOS:
                vectores[i, CONCEPTOS[palabra]] += 1
        vectores[i, 3] = 0.01  # evita vectores nulos
    return vectores


DOCS = [
    {"id": 1, "titulo": "Redes", "asignatura": "Redes", "contenido": "Una red conecta dispositivos."},
    {"id": 2, "titulo": "Matrices", "asignatura": "Matematicas", "contenido": "Una matriz es una tabla de numeros."},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUTA", tmp_path / "documentos.json")
    monkeypatch.setattr(embeddings, "embeber", embeber_falso)
    return TestClient(app)


def test_busqueda_semantica_encuentra_sinonimos():
    # "ordenadores" no aparece en el texto, pero significa casi lo mismo que "dispositivos"
    resultados = semantica.buscar_semantico("ordenadores", DOCS, embeber_falso)
    assert resultados[0]["documento_id"] == 1


def test_busqueda_semantica_sin_documentos():
    assert semantica.buscar_semantico("hola", [], embeber_falso) == []


def test_busqueda_semantica_respeta_el_maximo():
    assert len(semantica.buscar_semantico("tabla", DOCS, embeber_falso, max_resultados=1)) == 1


def test_endpoint_modo_significado(client):
    r = client.get("/buscar", params={"q": "ordenadores", "modo": "significado"})
    assert r.status_code == 200
    assert r.json()[0]["documento_id"] == 1


def test_endpoint_modo_invalido_da_422(client):
    assert client.get("/buscar", params={"q": "redes", "modo": "magia"}).status_code == 422