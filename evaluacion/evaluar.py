import sys
import pathlib
import json
import os

# Aseguramos que la raíz del proyecto esté en el camino de búsqueda de Python
RAIZ = pathlib.Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from app import storage, embeddings, llm, qa

# Configuración del prompt para el LLM Juez
PROMPT_JUEZ = """Eres un juez estricto evaluando un sistema de IA (RAG).
Pregunta del usuario: {pregunta}
Contexto recuperado de los apuntes:
{contexto}

Respuesta generada por el asistente:
{respuesta}

Evalúa la respuesta del asistente del 1 al 5 en estas dos métricas:
1. Fidelidad (Faithfulness): ¿La respuesta se basa ÚNICAMENTE en el contexto proporcionado sin inventar nada? (1=Se inventa todo, 5=Totalmente fiel al contexto).
2. Relevancia: ¿La respuesta contesta directamente a la pregunta del usuario? (1=No responde a la pregunta, 5=Respuesta perfecta y directa).

Devuelve SOLO un JSON válido con este formato exacto, sin markdown ni texto adicional:
{{"fidelidad": nota_numero, "relevancia": nota_numero}}
"""

def llamar_llm_juez(prompt_juez, texto_contexto):
    """
    Llama a llm.responder gestionando las posibles firmas de la función en tu proyecto.
    """
    try:
        # Intenta pasar (contexto, pregunta)
        return llm.responder(texto_contexto, prompt_juez)
    except TypeError:
        try:
            # Intenta invertir los argumentos (pregunta, contexto)
            return llm.responder(prompt_juez, texto_contexto)
        except TypeError:
            # Fallback en caso de aceptar un solo parámetro
            return llm.responder(prompt_juez)

def ejecutar_evaluacion():
    ruta_dataset = RAIZ / "evaluacion" / "dataset.json"
    if not ruta_dataset.exists():
        print(f"❌ No se encontró el archivo {ruta_dataset}")
        return

    with open(ruta_dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documentos = storage.cargar()
    resultados = []
    
    print(f"Iniciando evaluación de {len(dataset)} preguntas...\n")

    for item in dataset:
        pregunta = item["pregunta"]
        print(f"Evaluando: '{pregunta}'")
        
        # 1. Hacer la pregunta al RAG
        try:
            respuesta_rag = qa.responder_pregunta(
                pregunta, documentos, embeddings.embeber, llm.responder, umbral=0.3
            )
        except Exception as e:
            print(f"  ❌ Error al procesar la pregunta: {e}")
            continue

        # Extraer texto de la respuesta
        if isinstance(respuesta_rag, dict):
            texto_respuesta = respuesta_rag.get("respuesta", "")
            fuentes_lista = respuesta_rag.get("fuentes", [])
        else:
            texto_respuesta = getattr(respuesta_rag, "respuesta", "")
            fuentes_lista = getattr(respuesta_rag, "fuentes", [])

        # Reconstruir el texto del contexto para que el juez pueda leer el contenido real
        contexto_lineas = []
        for f in fuentes_lista:
            doc_id = f.get("documento_id") if isinstance(f, dict) else getattr(f, "documento_id", None)
            titulo = f.get("titulo") if isinstance(f, dict) else getattr(f, "titulo", "Apunte")
            frag = f.get("fragmento") if isinstance(f, dict) else getattr(f, "fragmento", "")
            
            # Buscar el texto correspondiente en los documentos guardados
            doc = next((d for d in documentos if d.get("id") == doc_id), None)
            if doc:
                contenido_doc = doc.get("contenido", "")
                contexto_lineas.append(f"--- Documento: {titulo} (Fragmento {frag}) ---\n{contenido_doc[:1200]}")
            else:
                contexto_lineas.append(f"--- Documento: {titulo} (Fragmento {frag}) ---")

        texto_contexto = "\n\n".join(contexto_lineas) if contexto_lineas else "Sin contexto recuperado."
        
        # 2. Formatear prompt del juez y evaluar
        prompt_final = PROMPT_JUEZ.format(
            pregunta=pregunta, 
            contexto=texto_contexto, 
            respuesta=texto_respuesta
        )
        
        try:
            evaluacion_cruda = llamar_llm_juez(prompt_final, texto_contexto)
            limpio = evaluacion_cruda.replace("```json", "").replace("```", "").strip()
            notas = json.loads(limpio)
        except Exception as e:
            print(f"  ⚠️ Error al evaluar con el juez: {e}")
            notas = {"fidelidad": 0, "relevancia": 0}

        fidelidad = notas.get("fidelidad", 0)
        relevancia = notas.get("relevancia", 0)

        print(f"  -> Fidelidad: {fidelidad}/5 | Relevancia: {relevancia}/5")

        resultados.append({
            "pregunta": pregunta,
            "fidelidad": fidelidad,
            "relevancia": relevancia
        })

    # 3. Calcular medias
    if resultados:
        media_fidelidad = sum(r["fidelidad"] for r in resultados) / len(resultados)
        media_relevancia = sum(r["relevancia"] for r in resultados) / len(resultados)
    else:
        media_fidelidad, media_relevancia = 0, 0

    print("\n" + "="*45)
    print("🎯 RESULTADOS FINALES DEL RAG")
    print("="*45)
    print(f"Fidelidad media (Ausencia de alucinaciones): {media_fidelidad:.2f}/5")
    print(f"Relevancia media (Responde a la pregunta):   {media_relevancia:.2f}/5")
    
    # Guardar reporte
    ruta_reporte = RAIZ / "evaluacion" / "reporte.json"
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        json.dump({
            "media_fidelidad": media_fidelidad, 
            "media_relevancia": media_relevancia, 
            "detalle": resultados
        }, f, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en: {ruta_reporte}")

if __name__ == "__main__":
    ejecutar_evaluacion()