"""Fábrica de la conexión a Postgres, análoga a llm_factory.py: centraliza
CÓMO se abre la conexión para que el resto del código (conversation_store.py)
solo hable en términos de "guardar/leer conversaciones", sin saber qué
motor de base de datos hay detrás.
"""

import os

import psycopg2
import streamlit as st


def _cadena_conexion() -> str:
    """Arma la cadena de conexión. Admite DATABASE_URL directa (útil para
    proveedores que la dan ya armada) o las variables sueltas POSTGRES_*
    (el esquema que usa docker-compose en el servidor)."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


@st.cache_resource
def obtener_conexion():
    """Abre (una sola vez por proceso) la conexión a Postgres a partir de
    variables de entorno. Cacheada con st.cache_resource porque abrir una
    conexión nueva en cada rerun de Streamlit sería carísimo e innecesario."""
    conexion = psycopg2.connect(_cadena_conexion())
    conexion.autocommit = True
    return conexion
