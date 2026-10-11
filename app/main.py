from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app import embeddings, rag, semantica, storage

app = FastAPI(title="Asistente de estudio")


class DocumentoNuevo(BaseModel):
    titulo: str = Field(min_length=3, max_length=100)
    asignatura: str = Field(min_length=2, max_length=50)
    contenido: str = Field(min_length=1)


class Documento(DocumentoNuevo):
    id: int


@app.get("/health")
def health():
    return {"estado": "ok"}


@app.get("/documentos")
def listar_documentos(asignatura: str | None = None):
    documentos = storage.cargar()
    if asignatura is None:
        return documentos
    return [d for d in documentos if d["asignatura"].lower() == asignatura.lower()]


@app.get("/documentos/{documento_id}")
def obtener_documento(documento_id: int):
    for d in storage.cargar():
        if d["id"] == documento_id:
            return d
    raise HTTPException(status_code=404, detail="Documento no encontrado")


@app.post("/documentos", response_model=Documento, status_code=201)
def crear_documento(nuevo: DocumentoNuevo):
    documentos = storage.cargar()
    nuevo_id = max((d["id"] for d in documentos), default=0) + 1
    documento = {"id": nuevo_id, **nuevo.model_dump()}
    documentos.append(documento)
    storage.guardar(documentos)
    return documento

@app.get("/buscar")
def buscar(
    q: str = Query(min_length=2),
    max_resultados: int = Query(3, ge=1, le=10),
    modo: Literal["palabras", "significado"] = "palabras",
):
    documentos = storage.cargar()
    if modo == "significado":
        return semantica.buscar_semantico(q, documentos, embeddings.embeber, max_resultados)
    return rag.buscar(q, documentos, max_resultados)