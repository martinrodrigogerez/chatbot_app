-- Ejecutar contra la base del chatbot (ya la crea init-multiple-dbs.sh /
-- el paso manual de despliegue). Equivalente al supabase_schema.sql
-- original, sin las partes específicas de Supabase (RLS vía auth.uid()):
-- el filtrado por user_id lo hace la propia app (conversation_store.py).

create extension if not exists pgcrypto;

create table if not exists conversaciones (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    titulo text not null default 'Nueva conversación',
    historial jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists conversaciones_user_id_idx on conversaciones (user_id);
