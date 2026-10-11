import json
from pathlib import Path

# Archivo donde se guardan los apuntes: <raiz del proyecto>/data/documentos.json
RUTA = Path(__file__).resolve().parent.parent / "data" / "documentos.json"

# Apuntes de ejemplo con los que arranca la API la primera vez.
EJEMPLOS = [
    {"id": 1, "titulo": "Introduccion a las redes", "asignatura": "Redes",
     "contenido": "Una red conecta dispositivos para compartir datos."},
    {"id": 2, "titulo": "Algebra lineal: matrices", "asignatura": "Matematicas",
     "contenido": "Una matriz es una tabla de numeros organizada en filas y columnas."},
    {"id": 3, "titulo": "Git y control de versiones", "asignatura": "Programacion",
     "contenido": "Git guarda el historial de cambios de un proyecto."},
]


def guardar(documentos: list[dict]) -> None:
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    with RUTA.open("w", encoding="utf-8") as f:
        json.dump(documentos, f, ensure_ascii=False, indent=2)


def cargar() -> list[dict]:
    if not RUTA.exists():
        guardar(EJEMPLOS)
        return [dict(d) for d in EJEMPLOS]
    with RUTA.open(encoding="utf-8") as f:
        return json.load(f)