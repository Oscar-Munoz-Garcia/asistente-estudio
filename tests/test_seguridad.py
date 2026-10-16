import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUTA", tmp_path / "documentos.json")
    monkeypatch.setenv("API_TOKEN", "secreto-de-prueba")
    return TestClient(app)


def test_sin_clave_se_rechaza(client):
    assert client.get("/documentos").status_code == 401


def test_con_clave_incorrecta_se_rechaza(client):
    r = client.get("/documentos", headers={"X-API-Key": "otra-cosa"})
    assert r.status_code == 401


def test_con_clave_correcta_funciona(client):
    r = client.get("/documentos", headers={"X-API-Key": "secreto-de-prueba"})
    assert r.status_code == 200


def test_health_no_necesita_clave(client):
    assert client.get("/health").status_code == 200


def test_tambien_protege_las_escrituras(client):
    nuevo = {"titulo": "Apunte nuevo", "asignatura": "Redes", "contenido": "Texto."}
    assert client.post("/documentos", json=nuevo).status_code == 401
    assert client.post("/preguntar", json={"pregunta": "hola"}).status_code == 401


def test_sin_token_configurado_la_api_queda_abierta(client, monkeypatch):
    monkeypatch.delenv("API_TOKEN")
    assert client.get("/documentos").status_code == 200

def test_la_pagina_principal_no_necesita_clave(client):
    assert client.get("/").status_code == 200