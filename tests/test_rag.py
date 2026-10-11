import pytest

from app import rag


def test_normalizar_quita_tildes_signos_y_palabras_vacias():
    assert rag.normalizar("¿Qué es una Matriz?") == ["matriz"]


def test_trocear_texto_corto_da_un_fragmento():
    assert rag.trocear("hola mundo", 80, 15) == ["hola mundo"]


def test_trocear_texto_vacio_da_lista_vacia():
    assert rag.trocear("") == []


def test_trocear_texto_largo_con_solape():
    texto = " ".join(str(i) for i in range(100))
    fragmentos = rag.trocear(texto, palabras_por_fragmento=40, solape=10)
    assert len(fragmentos) == 3
    # las ultimas 10 palabras del primero son las primeras 10 del segundo
    assert fragmentos[0].split()[-10:] == fragmentos[1].split()[:10]


def test_trocear_con_solape_invalido_falla():
    with pytest.raises(ValueError):
        rag.trocear("a b c", 10, 10)


DOCS = [
    {"id": 1, "titulo": "Redes", "asignatura": "Redes", "contenido": "Una red conecta dispositivos."},
    {"id": 2, "titulo": "Matrices", "asignatura": "Matematicas", "contenido": "Una matriz es una tabla de numeros."},
]


def test_buscar_encuentra_el_documento_correcto():
    resultados = rag.buscar("tabla de numeros", DOCS)
    assert resultados[0]["documento_id"] == 2


def test_buscar_sin_coincidencias_devuelve_lista_vacia():
    assert rag.buscar("xyzzy", DOCS) == []