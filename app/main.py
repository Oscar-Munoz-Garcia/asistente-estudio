import os
import io
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

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
async def subir_pdf(
    file: UploadFile = File(...),
    asignatura: str = Form("General") # Permite clasificar el PDF
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
    
    try:
        # Leemos el archivo en memoria
        contenido = await file.read()
        lector = PdfReader(io.BytesIO(contenido))
        
        # Extraemos el texto de todas las páginas
        texto_extraido = ""
        for pagina in lector.pages:
            texto = pagina.extract_text()
            if texto:
                texto_extraido += texto + "\n"
                
        if not texto_extraido.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF (podría ser una imagen escaneada)")

        # Guardamos el documento usando la misma lógica de "crear_documento"
        documentos = storage.cargar()
        nuevo_id = max((d["id"] for d in documentos), default=0) + 1
        
        # Usamos el nombre del archivo (quitándole el .pdf) como título
        titulo_limpio = file.filename[:-4] if file.filename.endswith('.pdf') else file.filename
        
        nuevo_doc = {
            "id": nuevo_id,
            "titulo": titulo_limpio,
            "asignatura": asignatura,
            "contenido": texto_extraido.strip()
        }
        
        documentos.append(nuevo_doc)
        storage.guardar(documentos)
        
        return {
            "mensaje": f"PDF procesado. Título: '{titulo_limpio}'", 
            "documento_id": nuevo_id,
            "caracteres_extraidos": len(texto_extraido)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el PDF: {str(e)}")


@app.get("/", include_in_schema=False)
async def inicio():
    with open("app/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)