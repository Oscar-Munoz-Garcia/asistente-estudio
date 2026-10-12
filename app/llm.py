import os
from pathlib import Path

from dotenv import load_dotenv

# Lee las variables del archivo .env de la raiz del proyecto (si existe),
# lo ejecutes desde la carpeta que lo ejecutes.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Modelo por defecto: uno de los que aparecen en tu pagina Limits de Groq.
# Se puede cambiar con MODELO_LLM en el .env.
MODELO = os.getenv("MODELO_LLM", "openai/gpt-oss-20b")


class ErrorLLM(Exception):
    """Algo fallo al hablar con el modelo de lenguaje."""


def responder(sistema: str, pregunta: str) -> str:
    """Envia las instrucciones y la pregunta al modelo y devuelve su respuesta en texto."""
    if not os.getenv("GROQ_API_KEY"):
        raise ErrorLLM("Falta GROQ_API_KEY: crea un archivo .env con tu clave (mira el README).")

    import groq  # import perezoso: los tests y el arranque no lo necesitan

    modelo = os.getenv("MODELO_LLM", MODELO)
    opciones = {}
    if "gpt-oss" in modelo:
        # Estos modelos "piensan" antes de responder. Esfuerzo bajo = menos tokens gastados.
        opciones["reasoning_effort"] = "low"

    try:
        cliente = groq.Groq()  # lee GROQ_API_KEY del entorno
        respuesta = cliente.chat.completions.create(
            model=modelo,
            # Margen amplio: en los modelos de razonamiento, "pensar" tambien gasta tokens.
            max_tokens=1500,
            temperature=0.2,  # poca creatividad: queremos respuestas fieles a los apuntes
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": pregunta},
            ],
            **opciones,
        )
    except groq.RateLimitError as e:
        raise ErrorLLM("Has alcanzado el limite de uso gratuito: espera un minuto y vuelve a probar.") from e
    except groq.APIError as e:
        raise ErrorLLM(f"Error al llamar al modelo: {e}") from e

    texto = respuesta.choices[0].message.content
    if not texto:
        raise ErrorLLM("El modelo no devolvio texto (puede que se agotase el limite de tokens). Prueba otra vez.")
    return texto