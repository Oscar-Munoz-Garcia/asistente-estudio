from functools import lru_cache
from pathlib import Path

import numpy as np

# Modelo multilingue (incluye espanol), pequeno (~220 MB) y gratuito. Se ejecuta en tu PC.
MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Carpeta donde se guarda el modelo descargado, fuera del repositorio.
CARPETA_MODELOS = Path.home() / ".cache" / "asistente-estudio" / "modelos"

# Memoria de vectores ya calculados, para no repetir trabajo con los mismos textos.
_cache: dict[str, np.ndarray] = {}


@lru_cache(maxsize=1)
def _cargar_modelo():
    # Import "perezoso": el servidor arranca rapido y los tests no necesitan el modelo.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODELO, cache_dir=str(CARPETA_MODELOS))


def embeber(textos: list[str]) -> np.ndarray:
    """Convierte cada texto en un vector de numeros que representa su significado."""
    if isinstance(textos, str):
        raise TypeError("embeber espera una LISTA de textos, no un texto suelto (se recorreria letra a letra)")
    pendientes = [t for t in dict.fromkeys(textos) if t not in _cache]
    if pendientes:
        modelo = _cargar_modelo()
        for texto, vector in zip(pendientes, modelo.embed(pendientes)):
            _cache[texto] = np.asarray(vector, dtype=float)
    return np.array([_cache[t] for t in textos])