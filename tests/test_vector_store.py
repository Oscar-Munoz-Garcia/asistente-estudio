import numpy as np
import pytest

from app import vector_store


def vector(*posiciones):
    """Un vector de 384 numeros con unos en las posiciones indicadas y ceros en el resto."""
    v = np.zeros(vector_store.VECTOR_SIZE)
    for posicion in posiciones:
        v[posicion] = 1.0
    return v


DOC = {"id": 7, "titulo": "Redes", "asignatura": "Redes"}


def test_guarda_y_busca_el_fragmento_mas_parecido(qdrant_memoria):
    vector_store.guardar_fragmentos(DOC, ["fragmento A", "fragmento B"], [vector(0), vector(1)])
    resultados = vector_store.buscar_similares(vector(1), limite=2)
    assert resultados[0]["texto"] == "fragmento B"
    assert resultados[0]["documento_id"] == 7
    assert resultados[0]["puntuacion"] > resultados[1]["puntuacion"]


def test_buscar_en_una_coleccion_vacia_devuelve_lista_vacia(qdrant_memoria):
    assert vector_store.buscar_similares(vector(0)) == []


def test_guardar_dos_veces_el_mismo_documento_no_duplica(qdrant_memoria):
    for _ in range(2):
        vector_store.guardar_fragmentos(DOC, ["a", "b"], [vector(0), vector(1)])
    assert vector_store.contar() == 2


def test_un_vector_de_tamano_incorrecto_da_un_error_claro(qdrant_memoria):
    with pytest.raises(ValueError, match="384"):
        vector_store.guardar_fragmentos(DOC, ["a"], [np.zeros(12650)])


def test_una_matriz_en_vez_de_un_vector_da_un_error_claro(qdrant_memoria):
    with pytest.raises(ValueError, match="matriz"):
        vector_store.buscar_similares(np.zeros((12650, 384)))