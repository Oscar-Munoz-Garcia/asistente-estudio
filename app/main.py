import os
import logging
import io
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app import embeddings, indexar, llm, qa, rag, semantica, seguridad, storage
app = FastAPI(title="Asistente de estudio", dependencies=[Depends(seguridad.exigir_token)])
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB

logger = logging.getLogger(__name__)


def _indexar(documento: dict) -> int:
    """Indexa el documento en Qdrant, solo si el modo de busqueda lo usa."""
    if os.getenv("MODO_BUSQUEDA", "significado") != "qdrant":
        return 0
    return indexar.indexar_documento(documento, embeddings.embeber)

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
    _indexar(documento)
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

@app.post("/documentos/pdf", status_code=201)
def subir_pdf(file: UploadFile = File(...), asignatura: str = Form("General")):
    nombre = file.filename or ""
    if not nombre.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")

    contenido = file.file.read(MAX_PDF_BYTES + 1)  # leemos como mucho el limite + 1 byte
    if len(contenido) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="El PDF supera el limite de 10 MB")

    try:
        lector = PdfReader(io.BytesIO(contenido))
        paginas = (pagina.extract_text() for pagina in lector.pages)
        texto = "\n".join(p for p in paginas if p).strip()
    except Exception:
        logger.exception("No se pudo leer el PDF %s", nombre)
        raise HTTPException(status_code=400, detail="No se pudo leer el PDF (archivo danado o protegido)")

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="No se pudo extraer texto del PDF (imagen escaneada o formato no compatible)",
        )

    titulo = nombre[:-4].strip()[:100] or "Documento"
    asignatura = asignatura.strip()[:50] or "General"

    documentos = storage.cargar()
    nuevo_id = max((d["id"] for d in documentos), default=0) + 1
    documento = {"id": nuevo_id, "titulo": titulo, "asignatura": asignatura, "contenido": texto}
    documentos.append(documento)
    storage.guardar(documentos)

    try:
        fragmentos = _indexar(documento)
    except Exception:
        logger.exception("El documento %s se guardo pero no se pudo indexar", nuevo_id)
        raise HTTPException(
            status_code=500,
            detail="El PDF se guardo, pero no se pudo indexar. Ejecuta 'python -m app.indexar' para reintentarlo.",
        )

    mensaje = f"PDF procesado. Titulo: '{titulo}'"
    if fragmentos:
        mensaje += f" ({fragmentos} fragmentos indexados en Qdrant)"
    return {
        "mensaje": mensaje,
        "documento_id": nuevo_id,
        "caracteres_extraidos": len(texto),
        "fragmentos_indexados": fragmentos,
    }
@app.get("/", include_in_schema=False)
async def inicio():
    with open("app/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)