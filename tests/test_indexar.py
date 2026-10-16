import pytest

from app import embeddings, indexar, rag, vector_store


def test_indexar_trocea_y_guarda_un_vector_por_fragmento(qdrant_memoria, embeber_384):
    texto = " ".join(f"palabra{i}" for i in range(200))
    doc = {"id": 1, "titulo": "Largo", "asignatura": "Prueba", "contenido": texto}
    esperados = len(rag.trocear(texto))
    assert esperados > 1
    assert indexar.indexar_documento(doc, embeber_384) == esperados
    assert vector_store.contar() == esperados


def test_un_documento_vacio_no_guarda_nada(qdrant_memoria, embeber_384):
    doc = {"id": 1, "titulo": "Vacio", "asignatura": "Prueba", "contenido": "   "}
    assert indexar.indexar_documento(doc, embeber_384) == 0
    assert vector_store.contar() == 0


def test_embeber_rechaza_un_texto_suelto():
    # Este era el origen del error "expected dim: 384, got 12650"
    with pytest.raises(TypeError):
        embeddings.embeber("un texto suelto")