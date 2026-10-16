import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "apuntes_estudio"
VECTOR_SIZE = 384  # dimension de los embeddings del modelo multilingue

# Una unica conexion para todo el programa: la carpeta local no admite dos a la vez.
_cliente_instance = None
_coleccion_lista = False


def _ruta_local() -> Path:
    por_defecto = Path(__file__).resolve().parent.parent / "data" / "qdrant_db"
    return Path(os.getenv("QDRANT_PATH", por_defecto))


def obtener_cliente() -> QdrantClient:
    """Reutiliza una unica conexion: en Qdrant Cloud si hay URL y clave, y si no, en local."""
    global _cliente_instance
    if _cliente_instance is None:
        url, clave = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
        if url and clave:
            _cliente_instance = QdrantClient(url=url, api_key=clave)
        else:
            ruta = _ruta_local()
            ruta.mkdir(parents=True, exist_ok=True)
            _cliente_instance = QdrantClient(path=str(ruta))
    return _cliente_instance


def usar_cliente(cliente) -> None:
    """Sustituye la conexion (los tests la usan para trabajar en memoria)."""
    global _cliente_instance, _coleccion_lista
    _cliente_instance, _coleccion_lista = cliente, False


def inicializar_coleccion() -> None:
    """Crea la coleccion si no existe. Solo consulta a Qdrant la primera vez."""
    global _coleccion_lista
    if _coleccion_lista:
        return
    cliente = obtener_cliente()
    if not cliente.collection_exists(COLLECTION_NAME):
        cliente.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    _coleccion_lista = True


def _comprobar_vector(vector) -> list[float]:
    """Devuelve el vector como lista de floats, o explica claramente por que no vale."""
    forma = getattr(vector, "shape", None)
    if forma is not None and len(forma) != 1:
        raise ValueError(f"Se esperaba UN vector (1 dimension) y llego una matriz de forma {forma}")
    valores = [float(x) for x in vector]
    if len(valores) != VECTOR_SIZE:
        raise ValueError(f"Se esperaba un vector de {VECTOR_SIZE} numeros y llego uno de {len(valores)}")
    return valores


def _id_punto(documento_id: int, fragmento: int) -> str:
    """Id valido para Qdrant (entero o UUID) y estable: el mismo fragmento siempre tiene el mismo."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"apuntes/{documento_id}/{fragmento}"))


def guardar_fragmentos(documento: dict, fragmentos: list[str], vectores) -> int:
    """Guarda un punto por fragmento: su vector y su texto. Si ya existian, los actualiza."""
    puntos = [
        PointStruct(
            id=_id_punto(documento["id"], i),
            vector=_comprobar_vector(vector),
            payload={
                "documento_id": documento["id"],
                "titulo": documento["titulo"],
                "asignatura": documento["asignatura"],
                "fragmento": i,
                "texto": texto,
            },
        )
        for i, (texto, vector) in enumerate(zip(fragmentos, vectores))
    ]
    if not puntos:
        return 0
    inicializar_coleccion()
    obtener_cliente().upsert(collection_name=COLLECTION_NAME, points=puntos)
    return len(puntos)


def buscar_similares(vector_pregunta, limite: int = 3) -> list[dict]:
    """Devuelve los fragmentos cuyo vector esta mas cerca del de la pregunta."""
    consulta = _comprobar_vector(vector_pregunta)
    inicializar_coleccion()
    respuesta = obtener_cliente().query_points(
        collection_name=COLLECTION_NAME, query=consulta, limit=limite
    )
    return [
        {
            "documento_id": p.payload["documento_id"],
            "titulo": p.payload["titulo"],
            "asignatura": p.payload["asignatura"],
            "fragmento": p.payload["fragmento"],
            "texto": p.payload["texto"],
            "puntuacion": round(p.score, 3),
        }
        for p in respuesta.points
    ]


def contar() -> int:
    """Cuantos fragmentos hay guardados."""
    inicializar_coleccion()
    return obtener_cliente().count(collection_name=COLLECTION_NAME, exact=True).count