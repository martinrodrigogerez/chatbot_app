"""Carga de configuración desde .env (local) o st.secrets (Streamlit Cloud)."""

from pathlib import Path

from dotenv import dotenv_values


def cargar_google_api_key(carpeta_proyecto: Path) -> str:
    """Lee GEMINI_API_KEY del .env local o, si no existe, de st.secrets.

    En Streamlit Community Cloud no hay archivo .env (está en .gitignore),
    las claves se cargan ahí vía el panel de Secrets y se leen con
    st.secrets. Centralizar esto acá evita repetir la lectura en cada
    script y deja un único lugar donde cambiar la fuente de configuración.
    """
    config = dotenv_values(carpeta_proyecto / ".env")
    api_key = config.get("GEMINI_API_KEY")

    if not api_key:
        import streamlit as st

        api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY: agregala al .env local o a los Secrets de Streamlit Cloud."
        )
    return api_key
