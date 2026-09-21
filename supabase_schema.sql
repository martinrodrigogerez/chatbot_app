-- Ejecutar en el SQL Editor de tu proyecto Supabase (supabase.com).
create table if not exists conversaciones (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    titulo text not null default 'Nueva conversación',
    historial jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists conversaciones_user_id_idx on conversaciones (user_id);

alter table conversaciones enable row level security;

-- El filtrado real lo hace el backend (Streamlit) con la service_role key,
-- que ignora RLS. Esta policy es un cinturón de seguridad extra si algún
-- día se accede con una clave anon/con JWT de usuario.
create policy "usuarios ven solo sus conversaciones"
    on conversaciones for all
    using (auth.uid()::text = user_id)
    with check (auth.uid()::text = user_id);
