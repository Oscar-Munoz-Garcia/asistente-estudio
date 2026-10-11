from fastapi import FastAPI, HTTPException

app = FastAPI(title="Asistente de estudio")

# Datos de ejemplo en memoria. Más adelante se guardarán de verdad.
DOCUMENTOS = [
    {"id": 1, "titulo": "Introduccion a las redes", "asignatura": "Redes"},
    {"id": 2, "titulo": "Algebra lineal: matrices", "asignatura": "Matematicas"},
    {"id": 3, "titulo": "Git y control de versiones", "asignatura": "Programacion"},
]


@app.get("/health")
def health():
    return {"estado": "ok"}


@app.get("/documentos")
def listar_documentos(asignatura: str | None = None):
    if asignatura is None:
        return DOCUMENTOS
    return [d for d in DOCUMENTOS if d["asignatura"].lower() == asignatura.lower()]


@app.get("/documentos/{documento_id}")
def obtener_documento(documento_id: int):
    for d in DOCUMENTOS:
        if d["id"] == documento_id:
            return d
    raise HTTPException(status_code=404, detail="Documento no encontrado")