"""Fábrica del cliente de Supabase, análoga a llm_factory.py: centraliza
CÓMO se abre la conexión para que el resto del código (conversation_store.py)
solo hable en términos de "guardar/leer conversaciones", sin saber si detrás
hay Supabase, otra base o un mock de test.
"""

import streamlit as st
from supabase import Client, create_client


@st.cache_resource
def obtener_cliente_supabase() -> Client:
    """Crea (una sola vez por proceso) el cliente de Supabase a partir de
    st.secrets. Cacheado con st.cache_resource porque abrir una conexión
    nueva en cada rerun de Streamlit sería carísimo e innecesario."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)
