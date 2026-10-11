from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Asistente de estudio")


class DocumentoNuevo(BaseModel):
    titulo: str = Field(min_length=3, max_length=100)
    asignatura: str = Field(min_length=2, max_length=50)
    contenido: str = Field(min_length=1)


class Documento(DocumentoNuevo):
    id: int


# Datos de ejemplo en memoria. Más adelante se guardarán de verdad.
DOCUMENTOS = [
    {"id": 1, "titulo": "Introduccion a las redes", "asignatura": "Redes",
     "contenido": "Una red conecta dispositivos para compartir datos."},
    {"id": 2, "titulo": "Algebra lineal: matrices", "asignatura": "Matematicas",
     "contenido": "Una matriz es una tabla de numeros organizada en filas y columnas."},
    {"id": 3, "titulo": "Git y control de versiones", "asignatura": "Programacion",
     "contenido": "Git guarda el historial de cambios de un proyecto."},
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


@app.post("/documentos", response_model=Documento, status_code=201)
def crear_documento(nuevo: DocumentoNuevo):
    nuevo_id = max((d["id"] for d in DOCUMENTOS), default=0) + 1
    documento = {"id": nuevo_id, **nuevo.model_dump()}
    DOCUMENTOS.append(documento)
    return documento