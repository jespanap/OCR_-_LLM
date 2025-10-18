"""
Taller IA: Aplicación Multimodal con OCR y LLMs
Curso: Inteligencia Artificial
Universidad: EAFIT
Profesor: Jorge Padilla
"""

import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from groq import Groq
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Taller IA: OCR + LLM",
    page_icon="",
    layout="wide"
)

st.title("Taller IA: OCR + LLM")
st.markdown("### Aplicación Multimodal con Visión Artificial y Procesamiento de Lenguaje Natural")
st.markdown("---")

# =============================================================================
# MÓDULO 1: El Lector de Imágenes (OCR)
# =============================================================================

st.header(" Módulo 1: Extracción de Texto (OCR)")

# Función para cargar el modelo OCR con caché
@st.cache_resource
def load_ocr_reader():
    """
    Carga el modelo EasyOCR en memoria.
    Usa @st.cache_resource para cargar solo una vez y no en cada interacción.
    """
    reader = easyocr.Reader(['es', 'en'])  # Español e Inglés
    return reader

# Cargar el lector OCR
with st.spinner("Cargando modelo OCR..."):
    reader = load_ocr_reader()

# Widget para subir imagen
uploaded_file = st.file_uploader(
    "Sube una imagen con texto",
    type=['png', 'jpg', 'jpeg'],
    help="Formatos soportados: PNG, JPG, JPEG"
)

# Procesar imagen si fue subida
if uploaded_file is not None:
    # Mostrar la imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Convertir imagen a formato numpy array para EasyOCR
    image_np = np.array(image)

    # Botón para extraer texto
    if st.button("Extraer Texto", type="primary"):
        with st.spinner("Extrayendo texto de la imagen..."):
            # Ejecutar OCR
            result = reader.readtext(image_np)

            # Extraer solo el texto de los resultados
            extracted_text = "\n".join([detection[1] for detection in result])

            # Guardar en session_state para persistencia
            st.session_state['extracted_text'] = extracted_text

    # Mostrar texto extraído si existe en session_state
    if 'extracted_text' in st.session_state:
        st.success("Texto extraído exitosamente")
        st.text_area(
            "Texto extraído:",
            value=st.session_state['extracted_text'],
            height=200,
            help="Puedes copiar este texto"
        )

st.markdown("---")

# =============================================================================
# MÓDULO 2 y 3: Conexión con LLMs (GROQ y Hugging Face)
# =============================================================================

st.header("Módulo 2 y 3: Análisis con Modelos de Lenguaje")

# Verificar que hay texto extraído
if 'extracted_text' in st.session_state and st.session_state['extracted_text']:

    # Crear columnas para la configuración
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Configuración del Modelo")

        # Selector de proveedor
        provider = st.radio(
            "Proveedor de LLM:",
            ["GROQ", "Hugging Face"],
            help="Selecciona el proveedor de API para el análisis"
        )

        if provider == "GROQ":
            # Modelos disponibles en GROQ
            model = st.selectbox(
                "Modelo:",
                [
                    "llama3-8b-8192",
                    "llama3-70b-8192",
                    "mixtral-8x7b-32768",
                    "gemma-7b-it"
                ],
                help="Selecciona el modelo de lenguaje a usar"
            )

        # Tarea a realizar
        task = st.selectbox(
            "Tarea a realizar:",
            [
                "Resumir en 3 puntos clave",
                "Identificar las entidades principales",
                "Traducir al inglés",
                "Análisis de sentimiento",
                "Extraer información clave"
            ],
            help="Selecciona qué quieres hacer con el texto"
        )

        # Parámetros ajustables
        st.markdown("**Parámetros:**")
        temperature = st.slider(
            "Temperature (creatividad):",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Valores bajos: más determinista. Valores altos: más creativo"
        )

        max_tokens = st.slider(
            "Max Tokens (longitud de respuesta):",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
            help="Cantidad máxima de tokens en la respuesta"
        )

    with col2:
        st.subheader("Resultado del Análisis")

        # Botón para analizar
        if st.button("Analizar Texto", type="primary", use_container_width=True):

            # Construir el prompt según la tarea seleccionada
            task_prompts = {
                "Resumir en 3 puntos clave": "Resume el siguiente texto en 3 puntos clave concisos:",
                "Identificar las entidades principales": "Identifica y lista las entidades principales (personas, lugares, organizaciones, fechas) en el siguiente texto:",
                "Traducir al inglés": "Traduce el siguiente texto al inglés:",
                "Análisis de sentimiento": "Analiza el sentimiento del siguiente texto (positivo, negativo, neutral) y explica por qué:",
                "Extraer información clave": "Extrae la información más importante del siguiente texto:"
            }

            system_prompt = task_prompts[task]
            user_text = st.session_state['extracted_text']

            # OPCIÓN: GROQ
            if provider == "GROQ":
                try:
                    # Verificar API key
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    if not groq_api_key:
                        st.error("No se encontró GROQ_API_KEY en el archivo .env")
                    else:
                        with st.spinner("Analizando con GROQ..."):
                            # Instanciar cliente de GROQ
                            client = Groq(api_key=groq_api_key)

                            # Llamada a la API
                            chat_completion = client.chat.completions.create(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": system_prompt
                                    },
                                    {
                                        "role": "user",
                                        "content": user_text
                                    }
                                ],
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )

                            # Extraer respuesta
                            response = chat_completion.choices[0].message.content

                            # Mostrar resultado
                            st.markdown("**Respuesta del modelo:**")
                            st.markdown(response)

                            # Información adicional
                            st.info(f"Model: {model} | Temperature: {temperature} | Max Tokens: {max_tokens}")

                except Exception as e:
                    st.error(f"Error al conectar con GROQ: {str(e)}")

            # OPCIÓN: HUGGING FACE

            elif provider == "Hugging Face":
                try:
                    # Verificar API key
                    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
                    if not hf_api_key:
                        st.error("No se encontró HUGGINGFACE_API_KEY en el archivo .env")
                    else:
                        with st.spinner("Analizando con Hugging Face..."):
                            # Instanciar cliente de Hugging Face
                            client = InferenceClient(token=hf_api_key)

                            # Construir el prompt completo
                            full_prompt = f"{system_prompt}\n\n{user_text}"

                            # Llamada a la API usando chat completion
                            response_text = ""
                            for message in client.chat_completion(
                                model="meta-llama/Meta-Llama-3-8B-Instruct",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_text}
                                ],
                                max_tokens=max_tokens,
                                temperature=temperature,
                                stream=True,
                            ):
                                response_text += message.choices[0].delta.content

                            # Mostrar resultado
                            st.markdown("**Respuesta del modelo:**")
                            st.markdown(response_text)

                            # Información adicional
                            st.info(f"Modelo: Meta-Llama-3-8B | Temperature: {temperature} | Max Tokens: {max_tokens}")

                except Exception as e:
                    st.error(f"Error al conectar con Hugging Face: {str(e)}")
                    st.info("Tip: Asegúrate de que tu token de Hugging Face tenga permisos de 'Read' para Inference.")

else:
    st.info("👆 Primero extrae texto de una imagen en la sección superior")

# FOOTER

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Taller IA: Aplicación Multimodal con OCR y LLMs</strong></p>
    <p>Inteligencia Artificial | Universidad EAFIT | Prof. Jorge Padilla</p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR: Información y ayuda


with st.sidebar:
    st.header("ℹ️ Información")

    st.markdown("""
    ### Cómo usar esta aplicación:

    1. **Sube una imagen** con texto
    2. **Extrae el texto** con OCR
    3. **Selecciona un proveedor** (GROQ o Hugging Face)
    4. **Elige una tarea** a realizar
    5. **Ajusta los parámetros** si lo deseas
    6. **Analiza** el texto extraído

    ### Sobre los parámetros:

    **Temperature:** Controla la creatividad
    - Bajo (0.1-0.5): Respuestas más deterministas
    - Medio (0.6-1.0): Balance
    - Alto (1.1-2.0): Más creativo/aleatorio

    **Max Tokens:** Longitud máxima de la respuesta

    ### APIs requeridas:
    - GROQ API Key
    - Hugging Face Token
    """)

    st.markdown("---")

    # Verificar estado de las API keys
    st.markdown("### Estado de API Keys:")

    groq_key = os.getenv("GROQ_API_KEY")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")

    if groq_key:
        st.success(" GROQ configurado")
    else:
        st.error("GROQ no configurado")

    if hf_key:
        st.success(" Hugging Face configurado")
    else:
        st.error(" Hugging Face no configurado")

    if not groq_key or not hf_key:
        st.warning("Configura tus claves en el archivo .env")