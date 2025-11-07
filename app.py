"""
Taller IA: Aplicación Multimodal con OCR y LLMs
Curso: Inteligencia Artificial - Universidad EAFIT
Profesor: Jorge Padilla
"""

import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from groq import Groq
from transformers import pipeline
import os
from dotenv import load_dotenv


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
load_dotenv(dotenv_path=".env")

st.set_page_config(
    page_title="Taller IA: OCR + LLM",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Taller IA: OCR + LLM")
st.markdown("### Aplicación Multimodal con Visión Artificial y Procesamiento de Lenguaje Natural")
st.markdown("---")

# =============================================================================
# MÓDULO 1: OCR
# =============================================================================
st.header("📸 Módulo 1: Extracción de Texto (OCR)")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["es", "en"])

with st.spinner("Cargando modelo OCR..."):
    reader = load_ocr_reader()

uploaded_file = st.file_uploader(
    "Sube una imagen con texto",
    type=["png", "jpg", "jpeg"],
    help="Formatos soportados: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    image_np = np.array(image)

    if st.button("Extraer Texto", type="primary"):
        with st.spinner("Extrayendo texto de la imagen..."):
            result = reader.readtext(image_np)
            extracted_text = "\n".join([str(d[1]) for d in result])
            st.session_state["extracted_text"] = extracted_text

    if "extracted_text" in st.session_state:
        st.success("✅ Texto extraído exitosamente")
        st.text_area("Texto extraído:", value=st.session_state["extracted_text"], height=200)

st.markdown("---")

# =============================================================================
# MÓDULO 2 Y 3: LLMs (GROQ y HUGGING FACE)
# =============================================================================
st.header("🧩 Módulo 2 y 3: Análisis con Modelos de Lenguaje")

if "extracted_text" not in st.session_state or not st.session_state["extracted_text"]:
    st.info("👆 Primero extrae texto de una imagen en la sección superior.")
else:
    text_input = st.session_state["extracted_text"]
    provider = st.radio("Proveedor:", ["GROQ", "Hugging Face"])

    temperature = st.slider("Creatividad (temperature):", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Máx. tokens (longitud):", 50, 2000, 500, 50)
    st.markdown("---")

    # -------------------------------------------------------------------------
    # GROQ
    # -------------------------------------------------------------------------
    if provider == "GROQ":
        st.subheader("💬 Análisis con GROQ (llama-3.1-8b-instant)")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("❌ No se encontró GROQ_API_KEY en .env")
        else:
            task = st.selectbox(
                "Tarea a realizar:",
                ["Resumir texto", "Identificar entidades", "Traducir al inglés"]
            )

            if st.button("Ejecutar análisis", type="primary"):
                system_prompts = {
                    "Resumir texto": "Resume el siguiente texto en 3 puntos clave concisos:",
                    "Identificar entidades": "Extrae las entidades principales (personas, lugares, organizaciones, fechas):",
                    "Traducir al inglés": "Traduce el siguiente texto al inglés:"
                }

                try:
                    client = Groq(api_key=groq_api_key)
                    with st.spinner("Analizando con GROQ..."):
                        chat = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": system_prompts[task]},
                                {"role": "user", "content": text_input}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        response = chat.choices[0].message.content
                        st.subheader("🧠 Respuesta del modelo:")
                        st.markdown(response)
                        st.info(f"Modelo: llama-3.1-8b-instant | Tarea: {task}")
                except Exception as e:
                    st.error(f"Error al conectar con GROQ: {e}")

    # -------------------------------------------------------------------------
    # HUGGING FACE
    # -------------------------------------------------------------------------
    elif provider == "Hugging Face":
        st.subheader("🤗 Análisis con Hugging Face")

        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            st.error("❌ No se encontró HUGGINGFACE_API_KEY en .env")
        else:
            import requests
            import re
            import json

            def limpiar_salida(texto):
                """Limpia estructuras técnicas como TranslationOutput(...) o SummarizationOutput(...)."""
                if not texto:
                    return ""
                texto = re.sub(r"[A-Za-z]+Output\([^']*'([^']+)'\)", r"\1", texto)
                texto = texto.replace("\\n", "\n").replace("\\xa0", " ").strip("'\" ")
                return texto.strip()

            def hf_infer(model: str, inputs: str):
                """Llamada segura al nuevo endpoint de inferencia de Hugging Face."""
                url = f"https://router.huggingface.co/hf-inference/models/{model}"
                headers = {"Authorization": f"Bearer {hf_api_key}"}
                payload = {"inputs": inputs}
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=60)
                    if resp.status_code != 200:
                        return f"⚠️ Error {resp.status_code}: {resp.text}"

                    data = resp.json()
                    # Caso 1: lista vacía
                    if isinstance(data, list) and not data:
                        return "No se detectaron resultados."

                    # Caso 2: lista con elementos
                    if isinstance(data, list):
                        item = data[0]
                        if isinstance(item, dict):
                            # Para resumen o traducción
                            if "summary_text" in item:
                                return item["summary_text"]
                            if "translation_text" in item:
                                return item["translation_text"]
                            # Para NER o formato libre
                            if "entity_group" in item:
                                entidades = [
                                    f"{ent.get('word','')} → {ent.get('entity_group',ent.get('entity',''))}"
                                    for ent in data
                                ]
                                return "\n".join(entidades)
                            return str(item)
                        return str(item)

                    # Caso 3: diccionario
                    if isinstance(data, dict):
                        return (
                            data.get("summary_text")
                            or data.get("translation_text")
                            or str(data)
                        )

                    # Caso 4: otro tipo (string)
                    return str(data)

                except Exception as e:
                    return f"⚠️ Error en la solicitud: {e}"

            task = st.selectbox(
                "Tarea a realizar:",
                ["Resumir texto", "Identificar entidades", "Traducir al inglés"]
            )

            if st.button("Ejecutar análisis", type="primary"):
                with st.spinner("Analizando con Hugging Face..."):
                    model = ""
                    output = ""
                    if task == "Resumir texto":
                        model = "sshleifer/distilbart-cnn-12-6"
                    elif task == "Identificar entidades":
                        model = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
                    elif task == "Traducir al inglés":
                        model = "Helsinki-NLP/opus-mt-es-en"

                    output = hf_infer(model, text_input)
                    st.subheader("🧠 Resultado del análisis:")
                    st.markdown(limpiar_salida(output))


# =============================================================================
# SIDEBAR: Información
# =============================================================================
with st.sidebar:
    st.header(" Información del Proyecto")
    st.markdown("""
    **Taller IA: Aplicación Multimodal con OCR y LLMs**
    
    1. Sube una imagen con texto.  
    2. Extrae el texto con OCR.  
    3. Analiza con GROQ o Hugging Face.  

    **Modelos:**
    - GROQ → `llama-3.1-8b-instant`
    - Hugging Face →  
        🧾 `facebook/bart-large-cnn` (resumen)  
        🧍 `Davlan/distilbert-base-multilingual-cased-ner-hrl` (entidades)  
        🌍 `Helsinki-NLP/opus-mt-es-en` (traducción)
    """)

    st.markdown("---")
    groq_key = os.getenv("GROQ_API_KEY")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")

    st.success("GROQ configurado" if groq_key else "GROQ no configurado")
    st.success("Hugging Face configurado" if hf_key else "Hugging Face no configurado")
