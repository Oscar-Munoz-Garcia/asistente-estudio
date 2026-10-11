import numpy as np

from app.rag import trocear


def buscar_semantico(consulta: str, documentos: list[dict], embeber, max_resultados: int = 3) -> list[dict]:
    """Devuelve los fragmentos cuyo SIGNIFICADO es mas parecido al de la consulta."""
    fragmentos = []
    for d in documentos:
        for i, texto in enumerate(trocear(d["contenido"])):
            fragmentos.append({
                "documento_id": d["id"],
                "titulo": d["titulo"],
                "fragmento": i,
                "texto": texto,
            })
    if not fragmentos:
        return []

    textos = [f"{f['titulo']}. {f['texto']}" for f in fragmentos]
    matriz = embeber(textos)
    vector = embeber([consulta])[0]

    # Similitud coseno: 1 = mismo significado, 0 = nada que ver.
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1
    norma_consulta = np.linalg.norm(vector) or 1
    similitudes = (matriz / normas) @ (vector / norma_consulta)

    mejores = np.argsort(-similitudes)[:max_resultados]
    return [{**fragmentos[i], "puntuacion": round(float(similitudes[i]), 3)} for i in mejores]