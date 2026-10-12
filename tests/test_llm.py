import types

import httpx
import pytest

from app import llm


def _instalar_cliente_falso(monkeypatch, create):
    """Sustituye el cliente de Groq por uno falso cuyo metodo create es el que le pasemos."""
    import groq

    monkeypatch.setenv("GROQ_API_KEY", "clave-de-prueba")

    class ClienteFalso:
        def __init__(self, *args, **kwargs):
            pass

        class chat:
            class completions:
                pass

    ClienteFalso.chat.completions.create = staticmethod(create)
    monkeypatch.setattr(groq, "Groq", ClienteFalso)


def _respuesta(texto):
    mensaje = types.SimpleNamespace(content=texto)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=mensaje)])


def test_sin_clave_da_error_claro(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm.ErrorLLM):
        llm.responder("sistema", "pregunta")


def test_devuelve_el_texto_que_genera_el_modelo(monkeypatch):
    def create(**kwargs):
        assert kwargs["messages"][0] == {"role": "system", "content": "sistema"}
        assert kwargs["messages"][1] == {"role": "user", "content": "pregunta"}
        return _respuesta("Hola desde el modelo")

    _instalar_cliente_falso(monkeypatch, create)
    assert llm.responder("sistema", "pregunta") == "Hola desde el modelo"


def test_los_modelos_gpt_oss_usan_razonamiento_bajo(monkeypatch):
    recibido = {}

    def create(**kwargs):
        recibido.update(kwargs)
        return _respuesta("ok")

    _instalar_cliente_falso(monkeypatch, create)
    monkeypatch.setenv("MODELO_LLM", "openai/gpt-oss-20b")
    llm.responder("sistema", "pregunta")
    assert recibido["reasoning_effort"] == "low"


def test_otros_modelos_no_llevan_parametros_de_razonamiento(monkeypatch):
    recibido = {}

    def create(**kwargs):
        recibido.update(kwargs)
        return _respuesta("ok")

    _instalar_cliente_falso(monkeypatch, create)
    monkeypatch.setenv("MODELO_LLM", "otro/modelo")
    llm.responder("sistema", "pregunta")
    assert "reasoning_effort" not in recibido


def test_respuesta_vacia_da_un_error_claro(monkeypatch):
    _instalar_cliente_falso(monkeypatch, lambda **kwargs: _respuesta(""))
    with pytest.raises(llm.ErrorLLM):
        llm.responder("sistema", "pregunta")


def test_limite_gratuito_da_un_mensaje_claro(monkeypatch):
    import groq

    def create(**kwargs):
        peticion = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        raise groq.RateLimitError("limite", response=httpx.Response(429, request=peticion), body=None)

    _instalar_cliente_falso(monkeypatch, create)
    with pytest.raises(llm.ErrorLLM) as error:
        llm.responder("sistema", "pregunta")
    assert "espera" in str(error.value).lower()