"""Carga de configuración desde el archivo .env del proyecto."""

from pathlib import Path

from dotenv import dotenv_values


def cargar_google_api_key(carpeta_proyecto: Path) -> str:
    """Lee GEMINI_API_KEY del .env ubicado en carpeta_proyecto.

    Centralizar esto acá (en vez de hacer dotenv_values() en el script
    principal) evita repetir la misma lectura en cada script del proyecto
    y deja un único lugar donde cambiar, por ejemplo, el nombre de la
    variable de entorno o la fuente de configuración.
    """
    config = dotenv_values(carpeta_proyecto / ".env")
    api_key = config.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")
    return api_key
