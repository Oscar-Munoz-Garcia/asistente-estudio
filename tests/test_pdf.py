import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import embeddings, main, storage, vector_store
from app.main import app


def pdf_con_texto(texto: str) -> bytes:
    """Construye a mano un PDF minimo de una pagina con ese texto (sin librerias extra)."""
    flujo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(flujo)).encode() + b" >>\nstream\n" + flujo + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    salida = b"%PDF-1.4\n"
    posiciones = []
    for numero, objeto in enumerate(objetos, start=1):
        posiciones.append(len(salida))
        salida += f"{numero} 0 obj\n".encode() + objeto + b"\nendobj\n"
    inicio_xref = len(salida)
    salida += f"xref\n0 {len(objetos) + 1}\n".encode() + b"0000000000 65535 f \n"
    for posicion in posiciones:
        salida += f"{posicion:010d} 00000 n \n".encode()
    salida += (
        f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{inicio_xref}\n%%EOF"
    ).encode()
    return salida


def pdf_en_blanco() -> bytes:
    escritor = PdfWriter()
    escritor.add_blank_page(width=200, height=200)
    salida = io.BytesIO()
    escritor.write(salida)
    return salida.getvalue()


def subir(client, nombre, contenido, **datos):
    return client.post("/documentos/pdf", files={"file": (nombre, contenido, "application/pdf")}, data=datos)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUTA", tmp_path / "documentos.json")

    def embeber_prohibido(textos):
        raise AssertionError("En el modo por defecto no debe cargarse el modelo de embeddings")

    monkeypatch.setattr(embeddings, "embeber", embeber_prohibido)
    return TestClient(app)


def test_subir_un_pdf_valido_guarda_el_documento(client):
    r = subir(client, "Apuntes redes.pdf", pdf_con_texto("Una red conecta dispositivos"), asignatura="Redes")
    assert r.status_code == 201
    assert r.json()["caracteres_extraidos"] > 0
    assert r.json()["fragmentos_indexados"] == 0  # modo por defecto: no se usa Qdrant
    nuevo = client.get("/documentos").json()[-1]
    assert nuevo["titulo"] == "Apuntes redes"
    assert nuevo["asignatura"] == "Redes"
    assert "red conecta" in nuevo["contenido"]


def test_la_extension_en_mayusculas_tambien_vale(client):
    assert subir(client, "NOTAS.PDF", pdf_con_texto("Hola mundo")).status_code == 201


def test_un_archivo_que_no_es_pdf_da_400(client):
    r = client.post("/documentos/pdf", files={"file": ("notas.txt", b"hola", "text/plain")})
    assert r.status_code == 400


def test_un_pdf_sin_texto_da_400_y_no_500(client):
    r = subir(client, "escaneado.pdf", pdf_en_blanco())
    assert r.status_code == 400
    assert "texto" in r.json()["detail"]


def test_un_pdf_corrupto_da_400(client):
    assert subir(client, "roto.pdf", b"esto no es un pdf").status_code == 400


def test_un_pdf_demasiado_grande_da_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_PDF_BYTES", 10)
    assert subir(client, "grande.pdf", pdf_con_texto("Hola mundo")).status_code == 413


def test_una_asignatura_vacia_se_convierte_en_general(client):
    subir(client, "sin_asignatura.pdf", pdf_con_texto("Hola mundo"), asignatura="  ")
    assert client.get("/documentos").json()[-1]["asignatura"] == "General"


def test_en_modo_qdrant_indexa_un_vector_por_fragmento(client, qdrant_memoria, embeber_384, monkeypatch):
    monkeypatch.setenv("MODO_BUSQUEDA", "qdrant")
    monkeypatch.setattr(embeddings, "embeber", embeber_384)
    texto = " ".join(f"palabra{i}" for i in range(200))
    r = subir(client, "largo.pdf", pdf_con_texto(texto))
    assert r.status_code == 201
    indexados = r.json()["fragmentos_indexados"]
    assert indexados > 1
    assert vector_store.contar() == indexados