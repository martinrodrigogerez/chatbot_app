"""Definición del prompt y de la cadena LCEL (prompt | modelo).

Separar esto del script principal deja la "personalidad" del chatbot y su
prompt en un solo lugar, versionable y testeable aparte de la lógica de UI.
"""

from langchain_core.prompts import PromptTemplate

PERSONALIDAD = (
    "Eres un asistente útil y amigable llamado ChatBot Pro. "
    "Respondes siempre en español."
)

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template=PERSONALIDAD
    + """

Historial de conversación:
{historial}

Responde de manera clara y concisa a la siguiente pregunta: {mensaje}""",
)


def crear_cadena(chat_model):
    """LCEL: encadena el prompt con el modelo usando el operador '|'.

    El diccionario de entrada ({"mensaje": ..., "historial": ...}) pasa
    primero por PROMPT_TEMPLATE, que lo convierte en el texto final del
    prompt, y ese texto se lo pasa directo a chat_model como su entrada.
    """
    return PROMPT_TEMPLATE | chat_model
