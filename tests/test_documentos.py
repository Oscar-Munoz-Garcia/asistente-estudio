import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Cada test usa un archivo temporal: nunca toca tus apuntes reales.
    monkeypatch.setattr(storage, "RUTA", tmp_path / "documentos.json")
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok"}


def test_listar_devuelve_los_ejemplos(client):
    r = client.get("/documentos")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_filtrar_por_asignatura(client):
    r = client.get("/documentos", params={"asignatura": "redes"})
    assert [d["id"] for d in r.json()] == [1]


def test_obtener_inexistente_da_404(client):
    assert client.get("/documentos/99").status_code == 404


def test_crear_documento_valido(client):
    nuevo = {"titulo": "Docker basico", "asignatura": "Programacion", "contenido": "Contenedores."}
    r = client.post("/documentos", json=nuevo)
    assert r.status_code == 201
    assert r.json()["id"] == 4
    assert len(client.get("/documentos").json()) == 4


def test_crear_con_titulo_corto_da_422(client):
    malo = {"titulo": "ab", "asignatura": "Programacion", "contenido": "x"}
    assert client.post("/documentos", json=malo).status_code == 422


def test_crear_sin_contenido_da_422(client):
    malo = {"titulo": "Titulo valido", "asignatura": "Programacion"}
    assert client.post("/documentos", json=malo).status_code == 422