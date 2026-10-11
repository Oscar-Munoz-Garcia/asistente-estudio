import math
import re
import unicodedata
from collections import Counter

# Palabras muy comunes que no ayudan a distinguir un texto de otro.
PALABRAS_VACIAS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a",
    "en", "y", "o", "u", "e", "que", "es", "son", "se", "por", "para", "con", "sin",
    "su", "sus", "lo", "como", "mas", "pero", "si", "no", "ni",
}


def normalizar(texto: str) -> list[str]:
    """Pasa a minusculas, quita tildes y signos, y descarta palabras vacias."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    palabras = re.findall(r"[a-z0-9]+", texto)
    return [p for p in palabras if p not in PALABRAS_VACIAS]


def trocear(texto: str, palabras_por_fragmento: int = 80, solape: int = 15) -> list[str]:
    """Divide un texto en fragmentos de N palabras que se solapan un poco."""
    if solape >= palabras_por_fragmento:
        raise ValueError("El solape debe ser menor que el tamano del fragmento")
    palabras = texto.split()
    paso = palabras_por_fragmento - solape
    fragmentos = []
    for inicio in range(0, len(palabras), paso):
        fragmentos.append(" ".join(palabras[inicio:inicio + palabras_por_fragmento]))
        if inicio + palabras_por_fragmento >= len(palabras):
            break
    return fragmentos


def buscar(consulta: str, documentos: list[dict], max_resultados: int = 3) -> list[dict]:
    """Devuelve los fragmentos de los apuntes mas relacionados con la consulta."""
    fragmentos = []
    for d in documentos:
        for i, texto in enumerate(trocear(d["contenido"])):
            fragmentos.append({
                "documento_id": d["id"],
                "titulo": d["titulo"],
                "fragmento": i,
                "texto": texto,
                "tokens": normalizar(d["titulo"] + " " + texto),
            })
    if not fragmentos:
        return []

    total = len(fragmentos)
    en_cuantos = Counter()  # en cuantos fragmentos aparece cada palabra
    for f in fragmentos:
        en_cuantos.update(set(f["tokens"]))

    palabras_consulta = set(normalizar(consulta))
    resultados = []
    for f in fragmentos:
        veces = Counter(f["tokens"])
        # Las palabras raras pesan mas que las comunes (idea de TF-IDF).
        puntos = sum(
            veces[p] * math.log(1 + total / en_cuantos[p])
            for p in palabras_consulta if p in veces
        )
        if puntos > 0:
            resultados.append({
                "documento_id": f["documento_id"],
                "titulo": f["titulo"],
                "fragmento": f["fragmento"],
                "texto": f["texto"],
                "puntuacion": round(puntos, 3),
            })
    resultados.sort(key=lambda r: r["puntuacion"], reverse=True)
    return resultados[:max_resultados]