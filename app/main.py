import os
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app import embeddings, llm, qa, rag, semantica, seguridad, storage

app = FastAPI(title="Asistente de estudio", dependencies=[Depends(seguridad.exigir_token)])


class DocumentoNuevo(BaseModel):
    titulo: str = Field(min_length=3, max_length=100)
    asignatura: str = Field(min_length=2, max_length=50)
    contenido: str = Field(min_length=1)


class Documento(DocumentoNuevo):
    id: int

class Pregunta(BaseModel):
    pregunta: str = Field(min_length=3, max_length=500)


class Fuente(BaseModel):
    documento_id: int
    titulo: str
    fragmento: int
    puntuacion: float


class Respuesta(BaseModel):
    respuesta: str
    fuentes: list[Fuente]    


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

@app.post("/preguntar", response_model=Respuesta)


@app.post("/preguntar", response_model=Respuesta)
def preguntar(datos: Pregunta):
    umbral = float(os.getenv("UMBRAL_SIMILITUD", "0.3"))
    modo = os.getenv("MODO_BUSQUEDA", "significado")
    try:
        return qa.responder_pregunta(
            datos.pregunta, storage.cargar(), embeddings.embeber, llm.responder,
            umbral=umbral, modo=modo,
        )
    except llm.ErrorLLM as e:
        raise HTTPException(status_code=503, detail=str(e))