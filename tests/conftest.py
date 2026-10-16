import numpy as np
import pytest
from qdrant_client import QdrantClient

from app import rag, vector_store


@pytest.fixture(autouse=True)
def aislar_entorno(monkeypatch, tmp_path):
    # Los tests parten siempre sin token, en el modo por defecto y sin tocar ningun Qdrant real
    # (ni el local ni uno en la nube), aunque esas variables esten en tu .env.
    for variable in ("API_TOKEN", "MODO_BUSQUEDA", "QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant_db"))
    vector_store.usar_cliente(None)
    yield
    vector_store.usar_cliente(None)


@pytest.fixture
def qdrant_memoria():
    """Una base Qdrant que vive solo en memoria y desaparece al acabar el test."""
    cliente = QdrantClient(":memory:")
    vector_store.usar_cliente(cliente)
    return cliente


CONCEPTOS = {"red": 0, "redes": 0, "dispositivos": 1, "ordenadores": 1,
             "matriz": 2, "matrices": 2, "tabla": 2}


@pytest.fixture
def embeber_384():
    """Modelo de embeddings 'de mentira' con 384 dimensiones: palabras parecidas comparten una."""

    def embeber(textos):
        assert not isinstance(textos, str), "embeber debe recibir una lista"
        vectores = np.zeros((len(textos), vector_store.VECTOR_SIZE))
        for i, texto in enumerate(textos):
            for palabra in rag.normalizar(texto):
                if palabra in CONCEPTOS:
                    vectores[i, CONCEPTOS[palabra]] += 1
            vectores[i, -1] = 0.01  # evita vectores nulos
        return vectores

    return embeber