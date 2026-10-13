import os
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# Aparece en /docs como un boton "Authorize" para pegar la clave.
cabecera_clave = APIKeyHeader(name="X-API-Key", auto_error=False)


def exigir_token(request: Request, clave: str | None = Security(cabecera_clave)):
    """Si existe la variable API_TOKEN, exige que cada peticion lleve ese mismo valor."""
    token = os.getenv("API_TOKEN")
    if not token:
        return  # sin token configurado (uso local) la API queda abierta
    if request.url.path == "/health":
        return  # el chequeo de salud debe poder consultarse sin clave
    if not clave or not secrets.compare_digest(clave, token):
        raise HTTPException(status_code=401, detail="Falta la cabecera X-API-Key o es incorrecta.")