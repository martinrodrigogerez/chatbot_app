"""Factory (fábrica) para crear instancias de chat model.

Por qué un factory acá: el modelo elegido en el sidebar (model_name) llega
como un string en tiempo de ejecución, y cada proveedor de LLM (Google,
OpenAI, Anthropic, ...) tiene su propia clase de LangChain con su propio
constructor. Si el código de la UI llamara a `ChatGoogleGenerativeAI(...)`
directamente, sumar un segundo proveedor significaría tocar el sidebar.

Centralizando la construcción acá:
- El sidebar/la UI solo conocen nombres de modelo (strings), nunca clases
  concretas de LangChain.
- Agregar un proveedor nuevo es agregar una entrada a los diccionarios de
  abajo, sin tocar el resto de la app (principio abierto/cerrado).
- Todo lo que devuelve crear_chat_model() implementa la misma interfaz de
  LangChain (BaseChatModel: .invoke(), .stream(), el operador "|", ...),
  así que el resto del código (chains.py, proyecto_chatbot.py) los usa de
  forma intercambiable sin saber qué proveedor hay detrás.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

# model_name -> proveedor. Para sumar OpenAI el día de mañana alcanza con:
#
#   from langchain_openai import ChatOpenAI
#   _FABRICANTES["openai"] = lambda model_name, temperature, api_key: ChatOpenAI(
#       model=model_name, temperature=temperature, api_key=api_key
#   )
#   MODELOS_DISPONIBLES["gpt-4o-mini"] = "openai"
#
# sin tocar el sidebar ni ningún otro módulo.
MODELOS_DISPONIBLES = {
    "gemini-3.5-flash-lite": "google",
    "gemini-3.1-flash-lite": "google",
    "gemini-2.0-flash": "google",
}

_FABRICANTES = {
    "google": lambda model_name, temperature, api_key: ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    ),
}


def crear_chat_model(model_name: str, temperature: float, api_key: str):
    """Instancia el chat model correspondiente a model_name.

    Se recrea en cada rerun de Streamlit porque las clases de chat model de
    LangChain fijan "model" y "temperature" en el constructor: no existen
    setters para cambiarlos en un objeto ya creado. Instanciar de nuevo es
    barato (no hace ninguna llamada de red), así que es la forma correcta
    de reflejar un cambio de slider o de selectbox en la próxima pregunta.
    """
    proveedor = MODELOS_DISPONIBLES.get(model_name)
    if proveedor is None:
        raise ValueError(f"Modelo desconocido: {model_name}")
    return _FABRICANTES[proveedor](model_name, temperature, api_key)
