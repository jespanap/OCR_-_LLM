# Aplicación Multimodal con OCR y LLMs

**Taller IA: OCR + LLM**
**Curso:** Inteligencia Artificial
**Universidad:** EAFIT
**Profesor:** Jorge Padilla

---

## Descripción del Proyecto

Aplicación web interactiva que integra **Visión Artificial** (OCR) y **Procesamiento de Lenguaje Natural** (NLP) para extraer texto de imágenes y procesarlo con Modelos de Lenguaje Grandes (LLMs).

### Características Principales

-  **Módulo 1: OCR con EasyOCR** - Extracción automática de texto desde imágenes
-  **Módulo 2: GROQ API** - Análisis ultrarrápido con LLMs
-  **Módulo 3: Hugging Face** - Alternativa con modelos open-source
-  **Interfaz Web Interactiva** - Construida con Streamlit
-  **Parámetros Ajustables** - Control de temperature y max_tokens
-  **Múltiples Tareas NLP** - Resumir, traducir, analizar sentimientos, extraer entidades

---

## Requisitos Previos

### 1. Obtener Claves de API

**GROQ API Key:**
1. Visita [https://console.groq.com](https://console.groq.com)
2. Crea una cuenta
3. Genera una API key en la sección de API Keys

**Hugging Face Token:**
1. Visita [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Crea una cuenta
3. Genera un "Access Token" con permisos de **"Read"** para **Inference**

### 2. Software Necesario

- Python 3.8 o superior
- pip

---

## Instalación

### Paso 1: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- `streamlit` - Framework web
- `easyocr` - Reconocimiento óptico de caracteres
- `groq` - Cliente de API GROQ
- `huggingface-hub` - Cliente de API Hugging Face
- `python-dotenv` - Gestión de variables de entorno
- `Pillow` - Procesamiento de imágenes
- `numpy` - Arrays numéricos

### Paso 2: Configurar Variables de Entorno

1. Crea un archivo `.env` en la raíz del proyecto:

```bash
cp .env.example .env
```

2. Edita el archivo `.env` y añade tus claves:

```env
GROQ_API_KEY="tu_clave_de_groq_aquí"
HUGGINGFACE_API_KEY="tu_clave_de_huggingface_aquí"
```

Reemplaza las claves de ejemplo con tus claves reales.

---

## Uso de la Aplicación

### Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente.

### Flujo de Trabajo

1. **Subir Imagen**
   - Haz clic en "Browse files"
   - Selecciona una imagen con texto (PNG, JPG, JPEG)
   - La imagen se mostrará en la interfaz

2. **Extraer Texto**
   - Haz clic en "Extraer Texto"
   - EasyOCR procesará la imagen y extraerá el texto
   - El texto aparecerá en un área editable

3. **Seleccionar Proveedor**
   - Elige entre **GROQ** o **Hugging Face**
   - Para GROQ: selecciona el modelo deseado

4. **Elegir Tarea**
   - Resumir en 3 puntos clave
   - Identificar las entidades principales
   - Traducir al inglés
   - Análisis de sentimiento
   - Extraer información clave

5. **Ajustar Parámetros**
   - **Temperature:** Controla la creatividad (0.0 = determinista, 2.0 = creativo)
   - **Max Tokens:** Longitud máxima de la respuesta

6. **Analizar**
   - Haz clic en "Analizar Texto"
   - Espera unos segundos para obtener el resultado
   - El análisis aparecerá en la columna derecha

---

## Estructura del Proyecto

```
OCR_LLM/
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias de Python
├── .env                      # Claves de API (NO subir a git)
├── .env.example             # Template para .env
├── .gitignore               # Archivos ignorados por git
├── README.md                # Este archivo
│   └── README.md
└── Taller_Final.pdf         # Documento del taller
```

---

## Módulos Implementados

### Módulo 1: El Lector de Imágenes (OCR)

- **Librería:** EasyOCR
- **Funcionalidad:**
  - Carga optimizada del modelo con `@st.cache_resource`
  - Extracción de texto con detección automática
  - Persistencia del texto con `st.session_state`

### Módulo 2: Conexión con GROQ API

- **Modelos disponibles:**
  - `llama3-8b-8192` - Rápido y eficiente
  - `llama3-70b-8192` - Más potente
  - `mixtral-8x7b-32768` - Contexto extenso
  - `gemma-7b-it` - Alternativa ligera

- **Características:**
  - Prompts estructurados con roles `system` y `user`
  - Parámetros ajustables (temperature, max_tokens)
  - Respuestas en formato Markdown

### Módulo 3: Integración con Hugging Face

- **Modelo:** Meta-Llama-3-8B-Instruct
- **Características:**
  - Streaming de respuestas
  - API de inferencia serverless
  - Alternativa gratuita a GROQ

---

## Parámetros del Modelo

### Temperature (Creatividad)

| Valor | Comportamiento |
|-------|----------------|
| 0.0 - 0.3 | Muy determinista, respuestas consistentes |
| 0.4 - 0.7 | Balance entre creatividad y precisión |
| 0.8 - 1.2 | Más creativo, mayor variabilidad |
| 1.3 - 2.0 | Muy creativo, menos predecible |

### Max Tokens

- **50-200:** Respuestas muy cortas
- **200-500:** Respuestas moderadas (recomendado)
- **500-1000:** Respuestas detalladas
- **1000-2000:** Respuestas muy extensas

---

## Solución de Problemas

### Error: "GROQ_API_KEY no encontrado"

**Solución:**
- Verifica que el archivo `.env` existe
- Asegúrate de que la clave esté correctamente escrita
- Reinicia la aplicación Streamlit

### Error: "Import easyocr could not be resolved"

**Solución:**
```bash
pip install --upgrade -r requirements.txt
```

### La primera carga es muy lenta

**Causa:** EasyOCR descarga modelos en la primera ejecución (~100MB)

**Solución:** Espera pacientemente en la primera ejecución. Las siguientes serán más rápidas gracias al caché.

### Error de Hugging Face: "Unauthorized"

**Solución:**
- Verifica que tu token tenga permisos de **Read** para **Inference**
- Genera un nuevo token si es necesario

---

## Tareas NLP Disponibles

1. **Resumir en 3 puntos clave**
   - Extrae las ideas principales del texto
   - Formato conciso y directo

2. **Identificar las entidades principales**
   - Detecta personas, lugares, organizaciones, fechas
   - Útil para análisis de documentos

3. **Traducir al inglés**
   - Traducción automática del texto extraído
   - Preserva el formato y significado

4. **Análisis de sentimiento**
   - Clasifica el sentimiento (positivo, negativo, neutral)
   - Proporciona explicación del análisis

5. **Extraer información clave**
   - Identifica los datos más relevantes
   - Filtra información secundaria

---

## Comparación: GROQ vs Hugging Face

| Característica | GROQ | Hugging Face |
|----------------|------|--------------|
| Velocidad |  Muy rápida |  Moderada |
| Modelos | Múltiples opciones | Llama-3-8B |
| Costo | Gratis (con límites) | Gratis (con límites) |
| Contexto | Hasta 32k tokens | 8k tokens |
| Mejor para | Respuestas rápidas | Open-source |

---

## Preguntas de Reflexión

1. **¿Qué diferencias de velocidad notaste entre GROQ y Hugging Face?**
   - GROQ suele ser significativamente más rápido
   - Hugging Face puede tener latencia variable

2. **¿Cómo afecta el temperature a las respuestas?**
   - Valores bajos: más consistente y preciso
   - Valores altos: más variado y creativo

3. **¿Qué tan importante fue la calidad del OCR?**
   - OCR de calidad = mejores resultados del LLM
   - Errores de OCR se propagan al análisis

4. **¿Qué otras aplicaciones se podrían desarrollar?**
   - Análisis de documentos legales
   - Extracción de datos de formularios
   - Digitalización de notas manuscritas
   - Traducción automática de señales

---


