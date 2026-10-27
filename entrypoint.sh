#!/bin/sh
set -e

# Arma .streamlit/secrets.toml a partir de variables de entorno, inyectadas
# por docker-compose desde el .env del servidor. Así todo lo que hace falta
# configurar para desplegar vive en un solo .env, sin tocar archivos dentro
# del contenedor ni commitear secretos.
mkdir -p .streamlit

cat > .streamlit/secrets.toml <<EOF
GEMINI_API_KEY = "${GEMINI_API_KEY}"

[auth]
redirect_uri = "${AUTH_REDIRECT_URI}"
cookie_secret = "${AUTH_COOKIE_SECRET}"

[auth.auth0]
client_id = "${AUTH0_CLIENT_ID}"
client_secret = "${AUTH0_CLIENT_SECRET}"
server_metadata_url = "${AUTH0_SERVER_METADATA_URL}"
EOF

exec streamlit run proyecto_chatbot.py --server.port=8501 --server.address=0.0.0.0
