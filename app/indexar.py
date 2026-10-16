from app import rag, vector_store


def indexar_documento(documento: dict, embeber) -> int:
    """Trocea el documento, calcula UN embedding por fragmento y los guarda en Qdrant."""
    fragmentos = rag.trocear(documento["contenido"])
    if not fragmentos:
        return 0
    # Igual que en la busqueda en memoria: se anade el titulo para dar contexto.
    textos = [f"{documento['titulo']}. {texto}" for texto in fragmentos]
    return vector_store.guardar_fragmentos(documento, fragmentos, embeber(textos))


if __name__ == "__main__":
    # Uso: python -m app.indexar   (indexa todos los apuntes que ya tienes en data/documentos.json)
    # Para el servidor antes de ejecutarlo: la base local no admite dos programas a la vez.
    from app import embeddings, storage

    total = sum(indexar_documento(d, embeddings.embeber) for d in storage.cargar())
    print(f"Indexados {total} fragmentos en Qdrant")