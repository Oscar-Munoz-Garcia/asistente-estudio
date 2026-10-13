import pytest


@pytest.fixture(autouse=True)
def sin_token_por_defecto(monkeypatch):
    # Aunque tengas API_TOKEN en tu entorno, los tests parten siempre sin proteccion.
    monkeypatch.delenv("API_TOKEN", raising=False)