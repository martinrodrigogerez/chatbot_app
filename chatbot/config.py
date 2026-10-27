"""Carga de configuración: variables de entorno primero (así funciona igual
en Docker, donde docker-compose las inyecta), con .env local y st.secrets
como fallback para desarrollo."""

import os
from pathlib import Path

from dotenv import load_dotenv


def cargar_configuracion(carpeta_proyecto: Path) -> None:
    """Completa os.environ con lo que haya en el .env local, sin pisar
    variables que ya vinieran inyectadas (p. ej. por docker-compose). En el
    contenedor el .env no existe y esto es un no-op silencioso."""
    load_dotenv(carpeta_proyecto / ".env", override=False)


def cargar_google_api_key() -> str:
    """Lee GEMINI_API_KEY de las variables de entorno (ya sea inyectadas
    por Docker o cargadas desde el .env local) o, como último recurso, de
    st.secrets (Streamlit Community Cloud)."""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        import streamlit as st

        api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY: agregala al .env local, a las variables "
            "de entorno del contenedor o a los Secrets de Streamlit Cloud."
        )
    return api_key
