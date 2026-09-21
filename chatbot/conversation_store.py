"""Repository de conversaciones: encapsula DÓNDE y CÓMO se guarda el
historial de chats.

Persiste en Supabase (tabla "conversaciones"), particionado por user_id (el
"sub" de Auth0), para que el historial sobreviva a reinicios del servidor y
cada usuario vea solo el suyo. El resto de la app (sidebar, chat) sigue
llamando a store.actual, store.nueva(), store.agregar_turno(...) sin
enterarse de dónde vive realmente el dato.
"""

from langchain_core.messages import AIMessage, HumanMessage

from chatbot.supabase_client import obtener_cliente_supabase

TITULO_POR_DEFECTO = "Nueva conversación"


def _mensajes_a_json(historial):
    return [
        {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
        for m in historial
    ]


def _json_a_mensajes(historial_json):
    mensajes = []
    for item in historial_json or []:
        clase = AIMessage if item["role"] == "assistant" else HumanMessage
        mensajes.append(clase(content=item["content"]))
    return mensajes


class ConversationStore:
    def __init__(self, user_id: str):
        self._user_id = user_id
        self._db = obtener_cliente_supabase()
        self._cargar()

    def _cargar(self) -> None:
        """Trae del más viejo al más nuevo, igual que crecía el dict en
        session_state, para que listar() (que hace reversed()) muestre el
        más reciente primero."""
        respuesta = (
            self._db.table("conversaciones")
            .select("id, titulo, historial")
            .eq("user_id", self._user_id)
            .order("created_at", desc=False)
            .execute()
        )
        filas = respuesta.data or [self._crear_fila()]
        self._conversaciones = {
            fila["id"]: {"titulo": fila["titulo"], "historial": _json_a_mensajes(fila["historial"])}
            for fila in filas
        }
        # Al entrar, la conversación "actual" es la más reciente.
        self._actual_id = filas[-1]["id"]

    def _crear_fila(self) -> dict:
        respuesta = (
            self._db.table("conversaciones")
            .insert({"user_id": self._user_id, "titulo": TITULO_POR_DEFECTO, "historial": []})
            .execute()
        )
        return respuesta.data[0]

    @property
    def actual(self) -> dict:
        """La conversación activa: {"titulo": str, "historial": [Human/AIMessage, ...]}"""
        return self._conversaciones[self._actual_id]

    def id_actual(self) -> str:
        return self._actual_id

    def listar(self):
        """(id, conversación) de la más reciente a la más antigua, para el sidebar."""
        return reversed(list(self._conversaciones.items()))

    def seleccionar(self, conv_id: str) -> None:
        self._actual_id = conv_id

    def nueva(self) -> None:
        # Si ya estamos parados en una conversación vacía, no creamos otra al lado.
        if self.actual["historial"]:
            fila = self._crear_fila()
            self._conversaciones[fila["id"]] = {"titulo": fila["titulo"], "historial": []}
            self._actual_id = fila["id"]

    def agregar_turno(self, pregunta: str, respuesta: str) -> bool:
        """Guarda el turno de pregunta/respuesta en la conversación activa.

        Si es el primer turno, usa la pregunta como título (como en
        ChatGPT/Claude). Devuelve True si el título cambió, para que la UI
        sepa que necesita un st.rerun() y refrescar el listado del sidebar.
        """
        conv = self.actual
        titulo_cambio = False
        if conv["titulo"] == TITULO_POR_DEFECTO:
            titulo = pregunta.strip().replace("\n", " ")
            conv["titulo"] = (titulo[:40].rstrip() + "…") if len(titulo) > 40 else titulo
            titulo_cambio = True

        conv["historial"].append(HumanMessage(content=pregunta))
        conv["historial"].append(AIMessage(content=respuesta))

        self._db.table("conversaciones").update(
            {"titulo": conv["titulo"], "historial": _mensajes_a_json(conv["historial"])}
        ).eq("id", self._actual_id).execute()

        return titulo_cambio

    def historial_como_texto(self) -> str:
        """Convierte el historial de la conversación activa a texto plano,
        porque PromptTemplate espera strings, no objetos de mensaje de
        LangChain."""
        lineas = []
        for m in self.actual["historial"]:
            if isinstance(m, HumanMessage):
                lineas.append(f"Usuario: {m.content}")
            elif isinstance(m, AIMessage):
                lineas.append(f"Asistente: {m.content}")
        return "\n".join(lineas)
