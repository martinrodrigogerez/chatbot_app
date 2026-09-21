from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage

from chatbot.chains import crear_cadena
from chatbot.config import cargar_google_api_key
from chatbot.conversation_store import ConversationStore
from chatbot.llm_factory import MODELOS_DISPONIBLES, crear_chat_model

APP_ICON = "🤖"
APP_TITULO = "ChatBot Pro"

## Configurar la pagina de la aplicación
st.set_page_config(page_title=APP_TITULO, page_icon=APP_ICON)

# --- Login obligatorio con Auth0 antes de mostrar nada del chat ---
if not st.user.is_logged_in:
    _, columna_central, _ = st.columns([1, 2, 1])
    with columna_central:
        st.markdown(
            f"<div style='text-align:center; font-size:3.5rem; margin-top:3rem;'>{APP_ICON}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h1 style='text-align:center; margin-top:0;'>{APP_TITULO}</h1>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                "Un asistente conversacional construido con **LangChain** y "
                "**Google Generative AI**. Iniciá sesión para empezar a chatear."
            )
            st.button(
                "Iniciar sesión con Auth0",
                on_click=st.login,
                args=("auth0",),
                type="primary",
                icon="🔐",
                use_container_width=True,
            )
    st.stop()

## Título de la aplicación
st.title(f"{APP_ICON} {APP_TITULO}")
st.caption("Chatbot de ejemplo construido con LangChain y Google Generative AI.")

google_api_key = cargar_google_api_key(Path(__file__).resolve().parent)
store = ConversationStore(user_id=st.user.sub)

# --- Sidebar de configuración ---
with st.sidebar:
    with st.container(border=True):
        avatar_usuario = st.user.get("picture") or "🙂"
        col_avatar, col_nombre = st.columns([1, 3], vertical_alignment="center")
        with col_avatar:
            if st.user.get("picture"):
                st.image(avatar_usuario, width=40)
            else:
                st.markdown("### 🙂")
        with col_nombre:
            st.markdown(f"**{st.user.get('name') or st.user.email}**")
            st.caption(st.user.email)
        if st.button("Cerrar sesión", icon="🚪", use_container_width=True):
            st.logout()

    st.divider()
    st.header("⚙️ Configuración")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", list(MODELOS_DISPONIBLES.keys()))

    # La UI no sabe qué clase de LangChain hay detrás de cada modelo: eso
    # lo resuelve el factory (chatbot/llm_factory.py).
    chat_model = crear_chat_model(model_name, temperature, google_api_key)

    st.divider()
    if st.button("Nueva conversación", icon="➕", use_container_width=True):
        store.nueva()
        st.rerun()

    st.caption("Conversaciones")
    for conv_id, conv in store.listar():
        es_actual = conv_id == store.id_actual()
        etiqueta = ("💬 " if es_actual else "") + conv["titulo"]
        if st.button(etiqueta, key=f"conv_{conv_id}", use_container_width=True, disabled=es_actual):
            store.seleccionar(conv_id)
            st.rerun()

# LCEL: prompt | modelo (ver chatbot/chains.py)
cadena = crear_cadena(chat_model)

AVATAR_ASISTENTE = "🤖"
avatar_usuario_chat = st.user.get("picture") or "🙂"

# Mostrar mensajes de la conversación activa
if not store.actual["historial"]:
    st.info("Escribí tu primer mensaje para empezar la conversación 👇")

for mensaje in store.actual["historial"]:
    es_ai = isinstance(mensaje, AIMessage)
    role = "assistant" if es_ai else "user"
    avatar = AVATAR_ASISTENTE if es_ai else avatar_usuario_chat
    with st.chat_message(role, avatar=avatar):
        st.markdown(mensaje.content)

# Entrada de chat para el usuario
pregunta = st.chat_input("Escribe tu mensaje...")

if pregunta:
    with st.chat_message("user", avatar=avatar_usuario_chat):
        st.markdown(pregunta)

    try:
        with st.chat_message("assistant", avatar=AVATAR_ASISTENTE):
            response_placeholder = st.empty()
            full_response = ""

            historial_texto = store.historial_como_texto()

            # Streaming de la respuesta, chunk por chunk
            for chunk in cadena.stream({"mensaje": pregunta, "historial": historial_texto}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")  # cursor parpadeante

            response_placeholder.markdown(full_response)

        titulo_cambio = store.agregar_turno(pregunta, full_response)
        if titulo_cambio:
            st.rerun()  # para que el nuevo título se vea ya en el listado del sidebar

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de Google esté configurada correctamente y que no hayas superado la cuota gratuita del modelo seleccionado.")
